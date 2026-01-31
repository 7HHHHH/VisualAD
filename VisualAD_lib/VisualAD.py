from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, need_weights: bool = False, text_layer = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        # print("text_layer", self.text_layer)
        
        # Simplified: always use basic ResidualAttentionBlock
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x, out_layers):
        idx = 0
        out_tokens = []

        for r in self.resblocks:
            idx += 1
            x = r(x)
            if idx in out_layers:
                out_tokens.append(x.clone())

        return x, out_tokens

    def forward_dispatch(self, x: torch.Tensor, out_layers=[6, 12, 18, 24]):
        if not self.text_layer:
            x, out_tokens = self.forward(x, out_layers)
            return x, out_tokens
        else:
            for r in self.resblocks:
                x = r(x)
            return x
            
    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # Add anomaly and normal aware tokens with different initialization
        self.anomaly_token = nn.Parameter(scale * torch.randn(width) * 0.1)  # Smaller init for stability
        self.normal_token = nn.Parameter(scale * torch.randn(width) * 0.1)   # Smaller init for stability

        # Position embeddings: separate trainable (anomaly/normal) from frozen (class/patches)
        num_patches = (input_resolution // patch_size) ** 2

        # Frozen positional embeddings for class token and patches (will be initialized from pretrained)
        self.positional_embedding_frozen = nn.Parameter(
            scale * torch.randn(num_patches + 1, width)  # [class, patches...]
        )
        self.positional_embedding_frozen.requires_grad = False
        self.anomaly_pos = nn.Parameter(scale * torch.randn(1, width))
        self.normal_pos = nn.Parameter(scale * torch.randn(1, width))
        self.anomaly_pos.requires_grad = True
        self.normal_pos.requires_grad = True
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, need_weights=True)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)

    def forward(self, x: torch.Tensor, features_list):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        batch_size = x.shape[0]

        def expand_token(token):
            return token.unsqueeze(0).expand(batch_size, -1, -1).to(x.dtype)

        anomaly_tokens = expand_token(self.anomaly_token)
        normal_tokens = expand_token(self.normal_token)
        class_tokens = expand_token(self.class_embedding)

        x = torch.cat([anomaly_tokens, normal_tokens, class_tokens, x], dim=1)

        side = int((self.positional_embedding_frozen.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 3) ** 0.5)

        if side != new_side:
            class_pos = self.positional_embedding_frozen[0:1, :]
            patch_pos = self.positional_embedding_frozen[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_patch_pos = torch.nn.functional.interpolate(patch_pos, (new_side, new_side), mode='bilinear')
            new_patch_pos = new_patch_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            pos = torch.cat([self.anomaly_pos, self.normal_pos, class_pos, new_patch_pos[0]], 0).to(x.dtype)
        else:
            pos = torch.cat([self.anomaly_pos, self.normal_pos, self.positional_embedding_frozen], 0).to(x.dtype)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        x, patch_tokens = self.transformer.forward_dispatch(x, features_list)
        x = x.permute(1, 0, 2)

        patch_tokens = [self.ln_post(pt.permute(1, 0, 2)) for pt in patch_tokens]

        anomaly_features = self.ln_post(x[:, 0, :])
        normal_features = self.ln_post(x[:, 1, :])
        class_features = self.ln_post(x[:, 2, :])

        return {
            'anomaly_features': anomaly_features,
            'normal_features': normal_features,
            'class_features': class_features,
            'patch_tokens': patch_tokens,
            'patch_start_idx': 3,
        }


class VisualAD(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text (kept for compatibility with pretrained weights loading)
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        # Only support ViT (no ResNet)
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        # Text components - kept for weight loading compatibility but not used in forward
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(), 
            text_layer=True
        )
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, feature_list=None):
        if feature_list is None:
            feature_list = []
        return self.visual(image.type(self.dtype), feature_list)

    # Note: encode_text and forward methods removed as they are unused in training/testing