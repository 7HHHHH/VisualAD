import torch
from torch import nn
from .VisualAD import VisualAD

def build_model(name: str, state_dict: dict, design_details = None):
    vit = "visual.proj" in state_dict
    
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    # Always use VisualAD for token-aware anomaly detection
    model = VisualAD(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # Remove our custom tokens from expected state_dict if they exist
    custom_keys = ["visual.anomaly_token", "visual.normal_token"]
    for key in custom_keys:
        if key in state_dict:
            del state_dict[key]
    
    # Handle positional embedding size mismatch due to extra tokens
    if "visual.positional_embedding" in state_dict:
        loaded_pos_embed = state_dict["visual.positional_embedding"]

        # The loaded checkpoint has [class, patches...] positions
        # Our model has separate: positional_embedding_frozen [class, patches...] and anomaly_pos/normal_pos

        # Initialize frozen positional embeddings (class + patches)
        state_dict["visual.positional_embedding_frozen"] = loaded_pos_embed

        # Initialize trainable anomaly/normal positions from class token position
        class_token_pos = loaded_pos_embed[0:1, :]  # First position for class token [1, width]
        state_dict["visual.anomaly_pos"] = class_token_pos  # Keep as [1, width]
        state_dict["visual.normal_pos"] = class_token_pos.clone()  # Keep as [1, width]

        # Remove old positional_embedding key
        del state_dict["visual.positional_embedding"]

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]


    model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow missing keys
    return model.eval()
