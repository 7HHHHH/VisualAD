import VisualAD_lib
import torch
from torch.cuda.amp import GradScaler, autocast
import argparse
import torch.nn.functional as F
from utils.loss import FocalLoss, BinaryDiceLoss, ContrastiveLoss
from dataset import Dataset
from utils.logger import get_logger
from utils.training_utils import (
    print_training_parameters, validate_training_setup, setup_model_training,
    create_optimizer, setup_feature_transforms, check_for_nan,
    compute_segmentation_maps, compute_segmentation_loss,
    validate_gradients, save_checkpoint
)
from tqdm import tqdm
import numpy as np
import os
import random
from utils.transforms import get_transform
from utils.backbone_config import resolve_features_list, load_feature_layers_from_config
from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO
torch.use_deterministic_algorithms(True,warn_only=False)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Additional deterministic settings
    torch.use_deterministic_algorithms(True, warn_only=False)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def generate_anomaly_map_from_tokens(anomaly_features, normal_features, patch_tokens, image_size):
    """
    Generate pixel-level anomaly map using token features with numerical stability
    Args:
        anomaly_features: [B, dim] - anomaly token features
        normal_features: [B, dim] - normal token features  
        patch_tokens: [B, num_patches, dim] - patch token features
        image_size: target image size
    Returns:
        anomaly_map: [B, H, W] - pixel-level anomaly map
    """
    B = anomaly_features.shape[0]
    
    # Normalize all features to prevent numerical instability in cosine similarity
    anomaly_features_norm = F.normalize(anomaly_features, dim=1, eps=1e-8)
    normal_features_norm = F.normalize(normal_features, dim=1, eps=1e-8)
    patch_tokens_norm = F.normalize(patch_tokens, dim=2, eps=1e-8)
    
    # Compute similarity between each patch and anomaly/normal tokens
    anomaly_sim = torch.cosine_similarity(
        patch_tokens_norm, anomaly_features_norm.unsqueeze(1), dim=2
    )  # [B, num_patches]
    
    normal_sim = torch.cosine_similarity(
        patch_tokens_norm, normal_features_norm.unsqueeze(1), dim=2
    )  # [B, num_patches]
    
    # Anomaly score = anomaly_similarity - normal_similarity
    anomaly_score = anomaly_sim - normal_sim  # [B, num_patches]
    
    # Check for NaN in anomaly scores
    if torch.isnan(anomaly_score).any():
        print(f"Warning: NaN detected in anomaly_score")
        anomaly_score = torch.nan_to_num(anomaly_score, nan=0.0)
    
    # Reshape to spatial dimensions - dynamic patch calculation
    num_patches = anomaly_score.shape[1]
    # Calculate expected patches based on image size and patch size
    patch_size_from_model = int(np.sqrt(num_patches))
    
    # Handle case where num_patches is not a perfect square
    if patch_size_from_model * patch_size_from_model != num_patches:
        # Calculate expected patches based on image size (assuming square patches)
        expected_patch_size = int(np.sqrt(576))  # 24 for 336px input with 14px patches
        expected_patches = expected_patch_size * expected_patch_size
        
        if num_patches > expected_patches:
            # Truncate extra tokens
            anomaly_score = anomaly_score[:, :expected_patches]
            patch_size_from_model = expected_patch_size
        else:
            # Pad with zeros if fewer tokens
            padding = expected_patches - num_patches
            anomaly_score = F.pad(anomaly_score, (0, padding), mode='constant', value=0)
            patch_size_from_model = expected_patch_size
    
    anomaly_map = anomaly_score.reshape(B, patch_size_from_model, patch_size_from_model)
    
    # Resize to target image size
    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(1), 
        size=(image_size, image_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(1)
    
    return anomaly_map


def compute_classification_loss_V2(anomaly_maps_list, labels, device):
    """
    Compute classification loss aligned with test.py inference
    Uses segmentation scores directly without normalization (training doesn't need class-wise normalization)
    Args:
        anomaly_maps_list: List of anomaly maps from different layers [[B, H, W], [B, H, W], ...]
        labels: Ground truth labels [B]
        device: Device
    Returns:
        loss: Binary cross-entropy loss
    """
    if not anomaly_maps_list:
        return torch.tensor(0.0, device=device)
    
    # Sum anomaly maps from all layers (same as test.py)
    final_anomaly_maps = torch.stack(anomaly_maps_list).sum(dim=0)  # [B, H, W]
    
    # Reduce anomaly map with Top-K mean to stabilize classification score
    seg_scores = reduce_anomaly_map(
        final_anomaly_maps,
        mode="topk_mean",
        topk_ratio=DEFAULT_TOPK_RATIO
    )  # [B]
    
    # Use raw segmentation scores directly (same as test.py fused scores)
    # Apply sigmoid within loss function for numerical stability
    labels_float = labels.float().to(device)
    loss = F.binary_cross_entropy_with_logits(seg_scores, labels_float)
    
    return loss


def train(args):
    logger = get_logger(args.save_path)
    device = args.device

    # Load and setup model
    try:
        model, _ = VisualAD_lib.load(args.backbone, device=device)
    except RuntimeError as exc:
        logger.error(f"Failed to load backbone {args.backbone}: {exc}")
        raise

    model.train()
    model.to(device)

    if not args.features_list:
        config_layers = load_feature_layers_from_config(args.feature_config, args.backbone, logger)
        if config_layers:
            args.features_list = config_layers

    total_layers = getattr(model.visual.transformer, 'layers', None)
    if total_layers is None:
        total_layers = len(args.features_list) if args.features_list else 0

    resolved_layers = resolve_features_list(args.features_list, total_layers, logger)
    if not resolved_layers:
        raise ValueError(f"Unable to determine valid feature layers for backbone {args.backbone}.")
    args.features_list = resolved_layers

    # Validate token_insert_layer (hardcoded to 0)
    token_insert_layer = 0
    if token_insert_layer not in [0] + args.features_list:
        if token_insert_layer > total_layers:
            logger.warning(
                f"token_insert_layer={token_insert_layer} exceeds backbone depth {total_layers}; "
                f"using last feature layer {args.features_list[-1]} instead."
            )
            token_insert_layer = args.features_list[-1]
        else:
            logger.warning(
                f"token_insert_layer={token_insert_layer} not in features_list={args.features_list}; "
                f"defaulting to first feature layer {args.features_list[0]}."
            )
            token_insert_layer = args.features_list[0]

    preprocess, target_transform = get_transform(args)

    print_training_parameters(args, logger, token_insert_layer)

    validate_training_setup(args, model, device, logger, token_insert_layer)

    # Spatial-Aware Cross-Attention (enabled by default)
    cross_attn = None
    from utils.spatial_cross_attention import build_layer_adaptive_cross_attention

    embed_dim = model.visual.embed_dim  # 1024 for ViT-L

    cross_attn = build_layer_adaptive_cross_attention(
        layers=args.features_list,
        embed_dim=embed_dim,
        num_anchors=4,
        dropout=0.1,
        res_scale_init=0.01,
        apply_to_layer24=True
    ).to(device)

    cross_attn.train()

    logger.info(f"✅ Spatial-Aware Cross-Attention enabled:")
    logger.info(f"   - Layers: {args.features_list}")
    logger.info(f"   - Total parameters: {cross_attn.get_num_parameters():,}")

    # Load dataset
    try:
        train_data = Dataset(root=args.train_data_path, transform=preprocess,
                           target_transform=target_transform, dataset_name=args.train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        logger.info(f"Dataset loaded: {len(train_data)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Setup feature transforms and model training
    # 🔥 MODIFICATION: Use embed_dim (1024) instead of proj output (768)
    feature_dim = model.visual.embed_dim  # 1024 for ViT-L, preserves full ViT information
    layer_transforms = setup_feature_transforms(args.features_list, device, logger, feature_dim)

    # Setup model training (projection layer always frozen)
    setup_model_training(model)

    optimizer = create_optimizer(model, layer_transforms, args, logger,
                                cross_attn=cross_attn)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.learning_rate * 0.1)
    # scheduler = None  # Use fixed LR for reproducibility across different epoch counts

    device_obj = torch.device(args.device)
    amp_enabled = bool(getattr(args, 'use_amp', False)) and device_obj.type == 'cuda'
    scaler = GradScaler(enabled=amp_enabled)

    # Initialize losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_token_relation = ContrastiveLoss(temperature=0.1, margin=0.5)
    logger.info("Using ContrastiveLoss (target: cos(θ) < -0.5)")

    logger.info(f"Initial token norms - anomaly: {torch.norm(model.visual.anomaly_token).item():.6f}, "
               f"normal: {torch.norm(model.visual.normal_token).item():.6f}")
    
    for epoch in tqdm(range(args.epoch)):
        # Keep model in train mode for gradient computation
        model.train()

        loss_list = []
        image_loss_list = []
        token_relation_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            # Squeeze only the channel dimension (dim=1), preserve batch dimension
            gt = items['img_mask'].squeeze(1).to(device)  # [B, 1, H, W] -> [B, H, W]
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            def compute_losses():
                vision_output = model.encode_image(image, args.features_list, token_insert_layer=token_insert_layer)
                anomaly_features = vision_output['anomaly_features']
                normal_features = vision_output['normal_features']
                patch_tokens = vision_output['patch_tokens']
                patch_start_idx = vision_output['patch_start_idx']

                # Spatial-Aware Cross-Attention enhancement (new design)
                # Extract pure patch features (remove special tokens)
                patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]

                # Layer adaptation
                if cross_attn is not None:
                    adapted_list = cross_attn(
                        anomaly_features, normal_features,
                        patch_features_list, args.features_list
                    )
                    # Extract adapted features
                    anomaly_features_list = [adapted['anomaly'] for adapted in adapted_list]
                    normal_features_list = [adapted['normal'] for adapted in adapted_list]
                else:
                    # Fallback to original approach (all layers use same token)
                    anomaly_features_list = [anomaly_features] * len(patch_tokens)
                    normal_features_list = [normal_features] * len(patch_tokens)

                # Contrastive Loss: Use enhanced features from last layer
                # (Last layer has richest semantics, used for contrastive learning)
                final_anomaly_features = anomaly_features_list[-1]
                final_normal_features = normal_features_list[-1]

                final_anomaly_features_norm = F.normalize(final_anomaly_features, dim=1, eps=1e-8)
                final_normal_features_norm = F.normalize(final_normal_features, dim=1, eps=1e-8)

                if (check_for_nan(final_anomaly_features_norm, "normalized anomaly features", logger, epoch) or
                    check_for_nan(final_normal_features_norm, "normalized normal features", logger, epoch)):
                    return None

                token_relation_val = loss_token_relation(final_anomaly_features_norm, final_normal_features_norm)
                if check_for_nan(token_relation_val, "contrastive_loss", logger, epoch):
                    return None

                # Merged loop: Generate both Segmentation Maps and Classification Maps
                # Avoid duplicate computation, improve training speed
                similarity_map_list = []
                anomaly_maps_list = []

                for idx_layer, patch_feature in enumerate(patch_tokens):
                    # Each layer uses its own enhanced features
                    anomaly_feat_norm = F.normalize(anomaly_features_list[idx_layer], dim=1, eps=1e-8)
                    normal_feat_norm = F.normalize(normal_features_list[idx_layer], dim=1, eps=1e-8)

                    current_layer = args.features_list[idx_layer]
                    transform_key = f'layer_{current_layer}'

                    # Apply feature transform
                    if transform_key in layer_transforms:
                        batch_size, num_patches, feat_dim = patch_feature.shape
                        patch_feature_flat = patch_feature.view(-1, feat_dim)
                        transformed_feature = layer_transforms[transform_key](patch_feature_flat)
                        patch_feature = transformed_feature.view(batch_size, num_patches, feat_dim)

                    # Generate anomaly map (compute only once)
                    anomaly_map = generate_anomaly_map_from_tokens(
                        anomaly_feat_norm, normal_feat_norm,
                        patch_feature[:, patch_start_idx:, :], args.image_size
                    )

                    # For Segmentation Loss
                    anomaly_map_sigmoid = torch.sigmoid(anomaly_map)
                    similarity_map = torch.stack([1 - anomaly_map_sigmoid, anomaly_map_sigmoid], dim=1)
                    similarity_map_list.append(similarity_map)

                    # For Classification Loss (reuse same anomaly_map)
                    anomaly_maps_list.append(anomaly_map)

                image_val = compute_classification_loss_V2(anomaly_maps_list, label, device)
                if check_for_nan(image_val, "image_loss", logger, epoch):
                    return None

                seg_val = torch.tensor(0.0, device=device)
                if similarity_map_list and (anomaly_features.requires_grad or normal_features.requires_grad):
                    seg_val = compute_segmentation_loss(similarity_map_list, gt, loss_focal, loss_dice)

                loss_components = []
                if image_val.requires_grad:
                    loss_components.append(image_val)
                if token_relation_val.requires_grad:
                    loss_components.append(token_relation_val)
                if seg_val.requires_grad:
                    loss_components.append(seg_val)

                if not loss_components:
                    logger.error("No loss component requires gradients!")
                    return None

                total = sum(loss_components)
                return total, seg_val, image_val, token_relation_val

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                result = compute_losses()

            if result is None:
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss, seg_loss_val, image_loss_val, token_relation_loss_val = result

            seg_loss_value = float(seg_loss_val.detach().item())
            image_loss_value = float(image_loss_val.detach().item())
            token_relation_loss_value = float(token_relation_loss_val.detach().item())

            if amp_enabled:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            # Validate gradients after backward pass
            if not validate_gradients(model, logger, epoch):
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if check_for_nan(model.visual.anomaly_token, "anomaly_token after update", logger, epoch) or \
               check_for_nan(model.visual.normal_token, "normal_token after update", logger, epoch):
                break

            if seg_loss_value is not None:
                loss_list.append(seg_loss_value)
            if image_loss_value is not None:
                image_loss_list.append(image_loss_value)
            if token_relation_loss_value is not None:
                token_relation_loss_list.append(token_relation_loss_value)


        if scheduler is not None:
            scheduler.step()

        # Log training progress
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.learning_rate
        if (epoch + 1) % args.print_freq == 0:
            logger.info(f'Epoch [{epoch+1}/{args.epoch}] - seg_loss: {np.mean(loss_list):.4f}, '
                       f'image_loss: {np.mean(image_loss_list):.4f}, contrastive_loss: {np.mean(token_relation_loss_list):.4f}, '
                       f'lr: {current_lr:.6f}')

            # Log token norms and similarity
            anomaly_norm = torch.norm(model.visual.anomaly_token).item()
            normal_norm = torch.norm(model.visual.normal_token).item()

            # Compute cosine similarity between tokens
            with torch.no_grad():
                anomaly_token_norm = F.normalize(model.visual.anomaly_token.squeeze(0), dim=-1)
                normal_token_norm = F.normalize(model.visual.normal_token.squeeze(0), dim=-1)
                token_similarity = F.cosine_similarity(anomaly_token_norm, normal_token_norm, dim=0).item()

            logger.info(f"Token norms - anomaly: {anomaly_norm:.6f}, normal: {normal_norm:.6f}")
            logger.info(f"Token similarity: {token_similarity:.4f} "
                       f"({'✓ near -1 (opposite)' if token_similarity < -0.5 else '⚠ not converged'})")


        # Save model checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, layer_transforms, args, epoch + 1,
                          os.path.join(args.save_path, f'epoch_{epoch + 1}.pth'), logger,
                          token_insert_layer=token_insert_layer,
                          cross_attn=cross_attn)

    # Save final model
    final_ckp_path = os.path.join(args.save_path, 'final_model.pth')
    save_checkpoint(model, layer_transforms, args, args.epoch, final_ckp_path, logger,
                   token_insert_layer=token_insert_layer,
                   cross_attn=cross_attn)
    
    logger.info('🎉 Training completed!')
    logger.info(f'Final model saved to {final_ckp_path}')
    logger.info(f'To test: python test.py --checkpoint_path {final_ckp_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("VisualAD Training V2", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="/home/hyn/work/dataset/AD/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints', help='path to save results')
    parser.add_argument("--train_dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--backbone", type=str, default="ViT-L/14@336px",
                        choices=VisualAD_lib.available_models(), help="CLIP backbone to use")
    parser.add_argument("--feature_config", type=str, default=os.path.join('configs', 'backbone_layers.yaml'),
                        help="YAML file specifying default feature layers per backbone")
    parser.add_argument("--features_list", type=int, nargs="*", default=[6, 12, 18, 24],
                        help="Override feature layers (falls back to YAML config if omitted)")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:1", help="device to use")

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    train(args)
