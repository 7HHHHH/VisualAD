"""
Training utility functions for VisualAD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_transform import create_feature_transform


def print_training_parameters(args, logger, token_insert_layer=0):
    """Print all training parameters before starting training"""
    logger.info("=" * 80)
    logger.info("TRAINING PARAMETERS")
    logger.info("=" * 80)

    # Dataset parameters
    logger.info("Dataset Parameters:")
    logger.info(f"  - Train data path: {args.train_data_path}")
    logger.info(f"  - Train dataset: {args.train_dataset}")
    logger.info(f"  - Image size: {args.image_size}")

    # Training parameters
    logger.info("\nTraining Parameters:")
    logger.info(f"  - Epochs: {args.epoch}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Device: {args.device}")
    logger.info(f"  - Random seed: {args.seed}")

    # Model parameters
    logger.info("\nModel Parameters:")
    backbone = getattr(args, 'backbone', None)
    if backbone is not None:
        logger.info(f"  - Backbone: {backbone}")
    feature_config = getattr(args, 'feature_config', None)
    if feature_config is not None:
        logger.info(f"  - Feature config: {feature_config}")
    logger.info(f"  - Features list: {args.features_list}")
    logger.info(f"  - Token insert layer: {token_insert_layer}")

    # Output parameters
    logger.info("\nOutput Parameters:")
    logger.info(f"  - Save path: {args.save_path}")
    logger.info(f"  - Print frequency: {args.print_freq}")
    logger.info(f"  - Save frequency: {args.save_freq}")

    logger.info("=" * 80)


def validate_training_setup(args, model, device, logger, token_insert_layer=0):
    """Validate training setup and requirements"""
    if isinstance(device, str):
        device = torch.device(device)

    # Check GPU memory
    if device.type == 'cuda':
        try:
            dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
            with torch.no_grad():
                _ = model.encode_image(dummy_input, args.features_list, token_insert_layer=token_insert_layer)
            del dummy_input
            torch.cuda.empty_cache()
        except RuntimeError as e:
            raise RuntimeError(f"GPU memory insufficient: {e}")

    import os
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data path does not exist: {args.train_data_path}")

    if token_insert_layer not in [0] + args.features_list:
        logger.warning(f"token_insert_layer={token_insert_layer} not in features_list={args.features_list}")

    logger.info("✅ Training setup validation passed")


def setup_model_training(model):
    """Configure model parameters for training

    Args:
        model: The model to configure

    Note:
        - Only anomaly/normal tokens and (optional) ln_post are trainable
        - Projection layer has been removed; features stay in DINOv2's native dimension
    """
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradients for trainable components
    if hasattr(model.visual, "anomaly_token"):
        model.visual.anomaly_token.requires_grad = True
    if hasattr(model.visual, "normal_token"):
        model.visual.normal_token.requires_grad = True

    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None:
        if hasattr(ln_post, "weight"):
            ln_post.weight.requires_grad = True
        if hasattr(ln_post, "bias"):
            ln_post.bias.requires_grad = True


def create_optimizer(model, layer_transforms, args, logger=None, cross_attn=None):
    """Create optimizer with different learning rates for different components

    Note:
        - Projection layer has been removed; optimizer only covers tokens/LN
        - Uses 1024-dim features for optimal performance
    """
    optimizer_params = []

    if hasattr(model.visual, "anomaly_token"):
        optimizer_params.append({'params': [model.visual.anomaly_token], 'lr': args.learning_rate, 'weight_decay': 0.01})
    if hasattr(model.visual, "normal_token"):
        optimizer_params.append({'params': [model.visual.normal_token], 'lr': args.learning_rate, 'weight_decay': 0.01})

    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None and hasattr(ln_post, "weight") and hasattr(ln_post, "bias"):
        optimizer_params.append({
            'params': [ln_post.weight, ln_post.bias],
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    if logger:
        logger.info("✅ Using 1024-dim features (no projection layer)")

    # Always add feature transforms to optimizer
    for transform in layer_transforms.values():
        optimizer_params.append({
            'params': transform.parameters(),
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    # Add cross-attention parameters if enabled
    if cross_attn is not None:
        optimizer_params.append({
            'params': cross_attn.parameters(),
            'lr': args.learning_rate * 0.1,  # Same learning rate as feature transform layer
            'weight_decay': 0.01
        })
        if logger is not None:
            logger.info(f"Added Cross-Attention params to optimizer (lr={args.learning_rate * 0.1:.6f})")

    optimizer = torch.optim.AdamW(optimizer_params, betas=(0.9, 0.999))
    return optimizer


def setup_feature_transforms(features_list, device, logger, feature_dim):
    """Setup feature transformation modules (always enabled)"""
    layer_transforms = nn.ModuleDict()
    for layer_idx in features_list:
        # Hardcoded: mlp with hidden_ratio=1.0, dropout=0.1
        hidden_dim = int(feature_dim * 1.0)

        layer_transforms[f'layer_{layer_idx}'] = create_feature_transform(
            transform_type="mlp", input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim, dropout=0.1
        ).to(device)

    logger.info(f"✅ Feature transform enabled for layers {features_list}")

    return layer_transforms


def check_for_nan(tensor, name, logger, epoch=None):
    """Check tensor for NaN values and log if found"""
    if torch.isnan(tensor).any():
        msg = f"NaN detected in {name}"
        if epoch is not None:
            msg += f" at epoch {epoch+1}"
        logger.error(msg)
        return True
    return False


def normalize_features(anomaly_features, normal_features, class_features):
    """Normalize feature tensors"""
    return (
        F.normalize(anomaly_features, dim=1, eps=1e-8),
        F.normalize(normal_features, dim=1, eps=1e-8), 
        F.normalize(class_features, dim=1, eps=1e-8)
    )


def compute_classification_loss(class_features_norm, anomaly_features_norm, normal_features_norm, labels, device, temperature=0.07):
    """Compute classification loss using token similarities"""
    class_anomaly_sim = F.cosine_similarity(class_features_norm, anomaly_features_norm, dim=1)
    class_normal_sim = F.cosine_similarity(class_features_norm, normal_features_norm, dim=1)
    
    classification_score = class_anomaly_sim - class_normal_sim
    classification_logits = torch.stack([-classification_score, classification_score], dim=1) / temperature
    classification_logits = torch.clamp(classification_logits, min=-20, max=20)
    
    return F.cross_entropy(classification_logits, labels.long().to(device))


def compute_segmentation_maps(
    patch_tokens,
    anomaly_features_norm,
    normal_features_norm,
    layer_transforms,
    args,
    patch_start_idx,
):
    """Compute segmentation maps for all layers"""
    from .anomaly_detection import generate_anomaly_map_from_tokens
    
    similarity_map_list = []
    
    for idx_layer, patch_feature in enumerate(patch_tokens):
        current_layer = args.features_list[idx_layer]
        transform_key = f'layer_{current_layer}'
        if transform_key in layer_transforms:
            batch_size, num_patches, feat_dim = patch_feature.shape
            patch_feature_flat = patch_feature.view(-1, feat_dim)
            transformed_feature = layer_transforms[transform_key](patch_feature_flat)
            patch_feature = transformed_feature.view(batch_size, num_patches, feat_dim)

        current_patch_start = patch_start_idx[idx_layer] if isinstance(patch_start_idx, (list, tuple)) else patch_start_idx
        
        anomaly_map = generate_anomaly_map_from_tokens(
            anomaly_features_norm, normal_features_norm,
            patch_feature[:, current_patch_start:, :], args.image_size
        )
        
        anomaly_map_sigmoid = torch.sigmoid(anomaly_map)
        similarity_map = torch.stack([1 - anomaly_map_sigmoid, anomaly_map_sigmoid], dim=1)
        similarity_map_list.append(similarity_map)
    
    return similarity_map_list


def compute_segmentation_loss(similarity_map_list, gt, loss_focal, loss_dice):
    """Compute segmentation loss from similarity maps"""
    seg_losses = []
    for similarity_map in similarity_map_list:
        # Hardcoded: beta_focal=1.0, beta_dice=1.0
        seg_losses.append(loss_focal(similarity_map, gt))
        # Only use anomaly channel to avoid double-counting (since channel 0 = 1 - channel 1)
        seg_losses.append(loss_dice(similarity_map[:, 1, :, :], gt))
    return sum(seg_losses) if seg_losses else torch.tensor(0.0, device=gt.device, requires_grad=False)


def validate_gradients(model, logger, epoch):
    """Validate and clip gradients"""
    if model.visual.anomaly_token.grad is not None:
        if check_for_nan(model.visual.anomaly_token.grad, "anomaly_token gradient", logger, epoch):
            return False
        torch.nn.utils.clip_grad_norm_([model.visual.anomaly_token], max_norm=1.0)
    
    if model.visual.normal_token.grad is not None:
        if check_for_nan(model.visual.normal_token.grad, "normal_token gradient", logger, epoch):
            return False
        torch.nn.utils.clip_grad_norm_([model.visual.normal_token], max_norm=1.0)
    
    return True


def save_checkpoint(model, layer_transforms, args, epoch, checkpoint_path, logger, token_insert_layer=0, cross_attn=None):
    """Save model checkpoint with all necessary components"""
    transform_state_dict = {}
    # Always save feature transforms
    for layer_name, transform in layer_transforms.items():
        transform_state_dict[layer_name] = transform.state_dict()

    ln_post = getattr(model.visual, "ln_post", None)
    ln_post_weight = ln_post.weight.data.clone() if ln_post is not None and hasattr(ln_post, "weight") else None
    ln_post_bias = ln_post.bias.data.clone() if ln_post is not None and hasattr(ln_post, "bias") else None

    checkpoint_data = {
        "anomaly_token": model.visual.anomaly_token.data.clone(),
        "normal_token": model.visual.normal_token.data.clone(),
        "token_insert_layer": token_insert_layer,
        # Keep ln_post parameters for compatibility (if exists)
        "ln_post_weight": ln_post_weight,
        "ln_post_bias": ln_post_bias,
        "features_list": args.features_list,
        "image_size": args.image_size,
        "epoch": epoch,
        "backbone": getattr(args, 'backbone', None),
        "use_1024_dim": True,  # New flag: using 1024-dim features
    }

    if hasattr(model.visual, "proj"):
        checkpoint_data["proj"] = model.visual.proj.data.clone()
    else:
        checkpoint_data["proj"] = None

    # Always save feature transform data (hardcoded: mlp, dropout=0.1, hidden_ratio=1.0)
    checkpoint_data.update({
        "layer_transforms": transform_state_dict,
        "transform_type": "mlp",
        "transform_config": {
            "dropout": 0.1,
            "mlp_hidden_ratio": 1.0
        }
    })

    # Save cross-attention if enabled
    if cross_attn is not None:
        checkpoint_data.update({
            "cross_attn": cross_attn.state_dict(),
            "cross_attn_config": {
                "num_anchors": getattr(args, 'num_anchors', 4),
                "dropout": getattr(args, 'cross_attn_dropout', 0.1),
                "res_scale_init": getattr(args, 'res_scale_init', 0.01),
                "apply_to_layer24": True  # Enabled by default
            }
        })
        logger.info("Cross-Attention state saved to checkpoint")

    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f'Model saved to {checkpoint_path}')
