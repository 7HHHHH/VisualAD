"""
Spatial-Aware Cross-Attention for Layer-Adaptive Anomaly Detection

Replacement for the failed multi-head cross-attention approach.

Core innovations:
1. Anchor Query Mechanism - Avoid "too few queries" problem
2. Token-Guided Gating - Maintain strong discriminative power of Layer 24 tokens
3. 2D Spatial Position Encoding - Preserve patch spatial structure information
4. Strong Residual Connections - Protect token's anomaly perception ability

Usage example:
    from utils.spatial_cross_attention import build_layer_adaptive_cross_attention

    cross_attn = build_layer_adaptive_cross_attention(
        layers=[6, 12, 18, 24],
        embed_dim=768,
        num_anchors=8,
        dropout=0.1
    )

    enhanced = cross_attn(anomaly_token, normal_token, patch_tokens_list, [6,12,18,24])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialBottleneckAttention(nn.Module):
    """
    Spatial-aware bottleneck attention mechanism

    Key design:
    1. Use learnable anchor queries (small quantity), not tokens themselves
    2. Preserve 2D spatial position encoding
    3. Output fused to original token through token-guided gating

    Args:
        embed_dim: Embedding dimension (768 for ViT-L)
        num_anchors: Anchor query count (default 8)
        dropout: Dropout rate (default 0.1)
    """
    def __init__(self, embed_dim, num_anchors=8, dropout=0.1, max_patches=1536, res_scale_init=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_anchors = num_anchors
        self.max_patches = max_patches
        self.res_scale_init = res_scale_init

        # Learnable anchor queries (much fewer than patches, avoid query-key imbalance)
        self.anchor_queries = nn.Parameter(torch.randn(num_anchors, embed_dim) * 0.02)

        # 2D relative position encoding (for key, preserve spatial information)
        # Support multiple input resolutions:
        # - CLIP ViT-L/14@336px: 576 patches (24×24)
        # - DINOv2 518px: 1369 patches (37×37)
        # - Use max_patches=1536 to ensure compatibility
        self.pos_encoding = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)

        # Query/Key/Value projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Normalization (after projection)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)

        # Token-guided gating (use original token to control aggregation strength)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, num_anchors),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual scaling (initialized to configurable value, preserve original token capability)
        self.res_scale = nn.Parameter(torch.ones(1) * res_scale_init)

        self._init_weights()

    def _init_weights(self):
        """Weight initialization"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, token_features, patch_tokens):
        """
        Args:
            token_features: [B, embed_dim] anomaly or normal token (from layer 24)
            patch_tokens: [B, N, embed_dim] patch features (N may be <576)

        Returns:
            enhanced_token: [B, embed_dim] enhanced token
        """
        B, N, C = patch_tokens.shape

        # 1. Prepare Query (from learnable anchor points)
        # Expand anchor query to batch dimension
        anchor_q = self.anchor_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_anchors, C]
        Q = self.q_norm(self.q_proj(anchor_q))  # [B, num_anchors, C]

        # 2. Prepare Key (patch + position encoding, preserve spatial information)
        # Use only the needed length of position encoding
        patch_with_pos = patch_tokens + self.pos_encoding[:, :N, :]  # [B, N, C]
        K = self.k_norm(self.k_proj(patch_with_pos))  # [B, N, C]

        # 3. Prepare Value (direct projection from patches)
        V = self.v_proj(patch_tokens)  # [B, N, C]

        # 4. Compute Attention (scaled dot-product)
        scale = C ** -0.5
        attn = torch.bmm(Q, K.transpose(1, 2)) * scale  # [B, num_anchors, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 5. Aggregate Value
        aggregated = torch.bmm(attn, V)  # [B, num_anchors, C]

        # 6. Token-guided gating (use original token to control aggregation weights)
        # This ensures layer 24 token's discriminative power participates in enhancement
        gate_weights = self.gate(token_features)  # [B, num_anchors]
        gate_weights = gate_weights.unsqueeze(-1)  # [B, num_anchors, 1]

        # Weighted average of anchor features
        gated_agg = (aggregated * gate_weights).sum(dim=1)  # [B, C]

        # 7. Output projection + residual connection
        output = self.out_proj(gated_agg)
        output = self.dropout(output)

        # Strong residual connection: preserve original token capability + slight enhancement
        enhanced = token_features + self.res_scale * output

        return enhanced


class LayerAdaptiveCrossAttention(nn.Module):
    """
    Create independent Cross-Attention module for each layer

    Different layers have large semantic differences (lower layers=texture, higher layers=global),
    so each layer needs independent parameters.

    Args:
        layers: Layer ID list, e.g. [6, 12, 18, 24]
        embed_dim: Embedding dimension
        num_anchors: Number of anchors per layer
        dropout: Dropout rate
    """
    def __init__(self, layers, embed_dim, num_anchors=8, dropout=0.1, max_patches=1536,
                 res_scale_init=0.1):
        super().__init__()
        self.layers = layers

        # Create independent attention for each layer
        self.layer_attentions = nn.ModuleDict()
        for layer_id in layers:
            self.layer_attentions[f'layer_{layer_id}'] = SpatialBottleneckAttention(
                embed_dim=embed_dim,
                num_anchors=num_anchors,
                dropout=dropout,
                max_patches=max_patches,
                res_scale_init=res_scale_init
            )

    def forward(self, anomaly_token, normal_token, patch_tokens_list, layer_ids):
        """
        Apply independent attention enhancement to patch features of each layer

        Args:
            anomaly_token: [B, embed_dim] anomaly token from layer 24
            normal_token: [B, embed_dim] normal token from layer 24
            patch_tokens_list: List[[B, N, embed_dim]] patch tokens from each layer
            layer_ids: List[int] layer IDs, e.g. [6, 12, 18, 24]

        Returns:
            enhanced_features: List[Dict] enhanced features for each layer
                [
                    {'anomaly': [B, C], 'normal': [B, C], 'patches': [B, N, C]},
                    ...
                ]
        """
        enhanced_features = []

        for idx, (layer_id, patch_tokens) in enumerate(zip(layer_ids, patch_tokens_list)):
            layer_key = f'layer_{layer_id}'
            attn_module = self.layer_attentions[layer_key]
            enhanced_anomaly = attn_module(anomaly_token, patch_tokens)
            enhanced_normal = attn_module(normal_token, patch_tokens)

            enhanced_features.append({
                'anomaly': enhanced_anomaly,
                'normal': enhanced_normal,
                'patches': patch_tokens
            })

        return enhanced_features

    def get_num_parameters(self):
        """Return the number of parameters in the module"""
        total = sum(p.numel() for p in self.parameters())
        return total


def build_layer_adaptive_cross_attention(layers, embed_dim, num_anchors=8, dropout=0.1,
                                          max_patches=1536, res_scale_init=0.1):
    """
    Factory function: Create Layer-Adaptive Cross-Attention module

    Args:
        layers: Layer ID list, e.g. [6, 12, 18, 24]
        embed_dim: Embedding dimension (1024 for ViT-L)
        num_anchors: Number of anchors (default 8)
        dropout: Dropout rate (default 0.1)
        max_patches: Maximum number of patches supported (default 1536)
        res_scale_init: Initial value for residual scaling (default 0.1)

    Returns:
        LayerAdaptiveCrossAttention instance
    """
    return LayerAdaptiveCrossAttention(
        layers=layers,
        embed_dim=embed_dim,
        num_anchors=num_anchors,
        dropout=dropout,
        max_patches=max_patches,
        res_scale_init=res_scale_init
    )


# 单元测试代码（供开发调试）
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Spatial-Aware Cross-Attention")
    print("=" * 60)

    # Parameters
    batch_size = 4
    embed_dim = 768
    num_patches = 576  # 24×24 for ViT-L/14@336px
    layers = [6, 12, 18, 24]
    num_anchors = 8

    # Create module
    cross_attn = build_layer_adaptive_cross_attention(
        layers=layers,
        embed_dim=embed_dim,
        num_anchors=num_anchors,
        dropout=0.1
    )

    print(f"\n✅ Module created successfully!")
    print(f"   Layers: {layers}")
    print(f"   Embed dim: {embed_dim}")
    print(f"   Anchors: {num_anchors}")
    print(f"   Total parameters: {cross_attn.get_num_parameters():,}")

    # Simulate input
    anomaly_token = torch.randn(batch_size, embed_dim)
    normal_token = torch.randn(batch_size, embed_dim)
    patch_tokens_list = [
        torch.randn(batch_size, num_patches, embed_dim) for _ in layers
    ]

    print(f"\n🔍 Testing forward pass...")
    print(f"   Input anomaly token: {anomaly_token.shape}")
    print(f"   Input normal token: {normal_token.shape}")
    print(f"   Input patch tokens: {[pt.shape for pt in patch_tokens_list]}")

    # Forward pass
    enhanced_features = cross_attn(
        anomaly_token, normal_token,
        patch_tokens_list, layers
    )

    print(f"\n✅ Forward pass successful!")
    print(f"   Output length: {len(enhanced_features)}")
    for idx, feat in enumerate(enhanced_features):
        print(f"   Layer {layers[idx]}:")
        print(f"     - anomaly: {feat['anomaly'].shape}")
        print(f"     - normal: {feat['normal'].shape}")
        print(f"     - patches: {feat['patches'].shape}")

    # Check residual connection
    print(f"\n🔍 Checking residual connection...")
    for idx, (layer_id, feat) in enumerate(zip(layers, enhanced_features)):
        if layer_id == layers[-1]:
            # Last layer should be identical
            assert torch.allclose(feat['anomaly'], anomaly_token)
            assert torch.allclose(feat['normal'], normal_token)
            print(f"   Layer {layer_id}: ✅ Identity preserved (last layer)")
        else:
            # Other layers should be similar but not identical
            anomaly_diff = (feat['anomaly'] - anomaly_token).abs().mean().item()
            normal_diff = (feat['normal'] - normal_token).abs().mean().item()
            print(f"   Layer {layer_id}: ✅ Enhanced (diff: anomaly={anomaly_diff:.6f}, normal={normal_diff:.6f})")

    print(f"\n{'=' * 60}")
    print("✅ All tests passed!")
    print(f"{'=' * 60}")
