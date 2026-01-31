import VisualAD_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset import Dataset
from utils.logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils.transforms import get_transform
from utils.metrics_h import compute_metrics
from scipy.ndimage import gaussian_filter
from utils.feature_transform import create_feature_transform
from utils.backbone_config import resolve_features_list, load_feature_layers_from_config

# Import refactored utility functions
from utils.analysis import (
    get_classification_from_segmentation,
    analyze_classification_distribution
)
from utils.visualization import visualize_anomaly_results
from utils.anomaly_detection import generate_anomaly_map_from_tokens

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=False)  # Commented out to prevent segfault
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def test(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    checkpoint = None
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading trained tokens from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        saved_backbone = checkpoint.get("backbone")
        if saved_backbone and not getattr(args, "_backbone_provided", False):
            print(f"Using saved backbone={saved_backbone} (from checkpoint)")
            args.backbone = saved_backbone
    else:
        if args.checkpoint_path:
            print(f"Checkpoint not found at {args.checkpoint_path}; using randomly initialized tokens")
        else:
            print("No checkpoint provided or found, using randomly initialized tokens")

    if checkpoint and "image_size" in checkpoint and not getattr(args, "_image_size_provided", False):
        args.image_size = checkpoint["image_size"]
        print(f"Using saved image_size: {args.image_size}")

    if checkpoint and "features_list" in checkpoint and not getattr(args, "_features_list_provided", False):
        args.features_list = checkpoint["features_list"]
        print(f"Using saved features_list: {args.features_list}")
    elif not args.features_list:
        config_layers = load_feature_layers_from_config(getattr(args, "feature_config", None), args.backbone, logger)
        if config_layers:
            args.features_list = config_layers

    preprocess, target_transform = get_transform(args)

    # Load model
    model, _ = VisualAD_lib.load(args.backbone, device=device)
    model.eval()

    total_layers = getattr(model.visual.transformer, 'layers', len(args.features_list) if args.features_list else 0)
    args.features_list = resolve_features_list(args.features_list, total_layers, logger)
    if not args.features_list:
        raise ValueError(f"Unable to determine valid feature layers for backbone {args.backbone}.")

    # 🔥 MODIFICATION: Use embed_dim (1024) instead of proj output (768)
    feature_dim = model.visual.embed_dim  # 1024 for ViT-L, preserves full ViT information

    # Load trained tokens if available
    layer_transforms = nn.ModuleDict()

    if checkpoint:
        model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
        model.visual.normal_token.data = checkpoint["normal_token"].to(device)

        if "layer_transforms" in checkpoint and checkpoint["layer_transforms"] is not None:
            print("Loading trained feature transform modules...")

            transform_type = checkpoint.get("transform_type", "mlp")

            for layer_name, transform_state_dict in checkpoint["layer_transforms"].items():
                if transform_type == "linear":
                    layer_transforms[layer_name] = create_feature_transform(
                        transform_type="linear",
                        input_dim=feature_dim,
                        output_dim=feature_dim,
                        dropout=0.0
                    ).to(device)
                else:
                    if 'mlp.0.weight' in transform_state_dict:
                        first_layer_weight = transform_state_dict['mlp.0.weight']
                    elif 'transform.0.weight' in transform_state_dict:
                        first_layer_weight = transform_state_dict['transform.0.weight']
                    elif 'down_proj.weight' in transform_state_dict:
                        first_layer_weight = transform_state_dict['down_proj.weight']
                    else:
                        first_layer_weight = None
                    hidden_dim = first_layer_weight.shape[0] if first_layer_weight is not None else int(feature_dim * 0.5)

                    layer_transforms[layer_name] = create_feature_transform(
                        transform_type=transform_type,
                        input_dim=feature_dim,
                        hidden_dim=hidden_dim,
                        output_dim=feature_dim,
                        dropout=0.0
                    ).to(device)

                layer_transforms[layer_name].load_state_dict(transform_state_dict)
                layer_transforms[layer_name].eval()

            print(f"✅ Loaded {len(layer_transforms)} {transform_type} transform modules: {list(layer_transforms.keys())}")

        elif "layer_mlps" in checkpoint and checkpoint["layer_mlps"] is not None:
            print("Loading trained MLP modules (legacy format)...")

            for layer_name, mlp_state_dict in checkpoint["layer_mlps"].items():
                first_layer_weight = mlp_state_dict['mlp.0.weight']
                hidden_dim = first_layer_weight.shape[0]

                layer_transforms[layer_name] = create_feature_transform(
                    transform_type="mlp",
                    input_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    output_dim=feature_dim,
                    dropout=0.0
                ).to(device)

                layer_transforms[layer_name].load_state_dict(mlp_state_dict)
                layer_transforms[layer_name].eval()

            print(f"✅ Loaded {len(layer_transforms)} MLP modules (legacy): {list(layer_transforms.keys())}")
        else:
            print("No feature transform parameters found in checkpoint")

        # Load cross-attention if available
        cross_attn = None
        if "cross_attn" in checkpoint and checkpoint["cross_attn"] is not None:
            from utils.spatial_cross_attention import build_layer_adaptive_cross_attention

            cross_attn_config = checkpoint.get("cross_attn_config", {})
            num_anchors = cross_attn_config.get("num_anchors", 8)
            dropout = cross_attn_config.get("dropout", 0.1)
            res_scale_init = cross_attn_config.get("res_scale_init", 0.1)
            apply_to_layer24 = cross_attn_config.get("apply_to_layer24", False)

            print(f"Loading trained Cross-Attention...")
            cross_attn = build_layer_adaptive_cross_attention(
                layers=args.features_list,
                embed_dim=feature_dim,
                num_anchors=num_anchors,
                dropout=dropout,
                res_scale_init=res_scale_init,
                apply_to_layer24=apply_to_layer24
            ).to(device)

            cross_attn.load_state_dict(checkpoint["cross_attn"])
            cross_attn.eval()

            print(f"✅ Loaded Spatial-Aware Cross-Attention:")
            print(f"   - Layers: {args.features_list}")
            print(f"   - Anchors: {num_anchors}")
            print(f"   - Residual Scale Init: {res_scale_init}")
            print(f"   - Apply to Layer 24: {apply_to_layer24}")
            print(f"   - Total parameters: {cross_attn.get_num_parameters():,}")
        else:
            cross_attn = None
            print("No Cross-Attention found in checkpoint")

        if "token_insert_layer" in checkpoint:
            args.token_insert_layer = checkpoint["token_insert_layer"]
            print(f"Using saved token_insert_layer={args.token_insert_layer} (from checkpoint)")
        else:
            args.token_insert_layer = 0
            print(f"Using default token_insert_layer=0")

        if "epoch" in checkpoint:
            print(f"Model was trained for {checkpoint['epoch']} epochs")

    else:
        args.token_insert_layer = 0
        print(f"Using default token_insert_layer=0")

    # Test dataset
    test_data = Dataset(root=args.test_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    model.to(device)
    
    results = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []

    # Data analysis lists
    all_original_images = []
    all_anomaly_maps = []
    all_gt_masks = []
    all_cls_names = []
    all_anomaly_labels = []
    all_img_paths = []

    total_samples = len(test_dataloader)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)
        print(f"Testing on {total_samples} samples (limited by --max_samples)...")
    else:
        print(f"Testing on {total_samples} samples...")
    
    sample_count = 0
    for items in tqdm(test_dataloader):
        if args.max_samples is not None and sample_count >= args.max_samples:
            break
        sample_count += 1
        image = items['img'].to(device)
        cls_name = items['cls_name']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            vision_output = model.encode_image(image, args.features_list, token_insert_layer=args.token_insert_layer)

            anomaly_features = vision_output['anomaly_features']
            normal_features = vision_output['normal_features']
            patch_tokens = vision_output['patch_tokens']
            patch_start_idx = vision_output['patch_start_idx']

            # Spatial-Aware Cross-Attention enhancement (consistent with training)
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
                # Fallback to original method (use same token for all layers)
                anomaly_features_list = [anomaly_features] * len(patch_tokens)
                normal_features_list = [normal_features] * len(patch_tokens)

            # Generate anomaly maps for each layer
            anomaly_map_list = []
            for idx_layer, patch_feature in enumerate(patch_tokens):
                anomaly_feat_norm = F.normalize(anomaly_features_list[idx_layer], dim=1, eps=1e-8)
                normal_feat_norm = F.normalize(normal_features_list[idx_layer], dim=1, eps=1e-8)

                current_layer = args.features_list[idx_layer]
                transform_key = f'layer_{current_layer}'
                if transform_key in layer_transforms:
                    batch_size, num_patches, feat_dim = patch_feature.shape
                    patch_feature_flat = patch_feature.view(-1, feat_dim)
                    transformed_feature = layer_transforms[transform_key](patch_feature_flat)
                    patch_feature = transformed_feature.view(batch_size, num_patches, feat_dim)

                anomaly_map = generate_anomaly_map_from_tokens(
                    anomaly_feat_norm, normal_feat_norm,
                    patch_feature[:, patch_start_idx:, :],
                    args.image_size
                )
                anomaly_map_list.append(anomaly_map)

            if anomaly_map_list:
                final_anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
                # Fix memory management for tensor conversion
                final_anomaly_map_cpu = final_anomaly_map.detach().cpu()
                filtered_maps = []
                for i in range(final_anomaly_map_cpu.shape[0]):
                    filtered_map = gaussian_filter(final_anomaly_map_cpu[i].numpy(), sigma=args.sigma)
                    filtered_maps.append(torch.from_numpy(filtered_map))
                final_anomaly_map = torch.stack(filtered_maps)
                
                # Clear intermediate tensors and GPU cache
                del anomaly_map_list, final_anomaly_map_cpu, filtered_maps
                torch.cuda.empty_cache()
                
                results[cls_name[0]]['anomaly_maps'].append(final_anomaly_map)
                
                # Collect data for analysis
                all_original_images.append(image.detach().cpu())
                all_anomaly_maps.append(final_anomaly_map)
                all_gt_masks.append(gt_mask)
                all_cls_names.append(cls_name[0])
                all_anomaly_labels.append(items['anomaly'].item())
                all_img_paths.append(items['img_path'][0])

    # Always compute metrics with score normalization and fusion
    if all_original_images:
        print(f"\n🔍 Starting metrics calculation...")

        # Get classification results from segmentation
        fused_scores, normalized_anomaly_maps = get_classification_from_segmentation(all_anomaly_maps, all_cls_names, results)

        # Compute and display metrics using fused scores
        print(f"\n📈 Computing fused classification metrics...")
        try:
            compute_metrics(results, obj_list, logger)
            print("✅ Fused metrics calculation complete!")
        except (ValueError, IndexError) as e:
            print(f"⚠️ Skipped fused metrics due to limited samples: {e}")

        # Data analysis and visualization (only when enabled)
        if args.enable_analysis:
            analysis_dir = os.path.join(args.save_path, 'analysis')
            print(f"\n🔍 Starting data analysis and visualization...")

            # Classification distribution analysis (using fused scores)
            print(f"\n📊 Analyzing fused classification distribution...")
            analyze_classification_distribution(
                fused_scores,  # Use fused scores for analysis
                all_cls_names,
                all_anomaly_labels,
                analysis_dir
            )

            # Visualize anomaly detection results (using fused scores)
            print(f"\n🎨 Generating visualization results...")
            visualize_anomaly_results(
                all_original_images,
                normalized_anomaly_maps,
                all_gt_masks,
                fused_scores,  # Use fused scores
                all_cls_names,
                all_img_paths,
                all_anomaly_labels,  # True anomaly labels
                args.test_dataset,  # Dataset name
                analysis_dir
            )

            print(f"\n✅ Analysis complete! Results saved to: {analysis_dir}")
        else:
            print("ℹ️ Analysis feature not enabled, use --enable_analysis parameter to enable data analysis and visualization")
    else:
        print("⚠️ No test data collected")

    print(f"\n🎉 Testing complete! Processed {sample_count} samples")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VisualAD Test", add_help=True)
    parser.add_argument("--test_data_path", type=str, required=True, help="test dataset path")
    parser.add_argument("--save_path", type=str, default='./test_results', help='path to save test results')
    parser.add_argument("--test_dataset", type=str, required=True, help="test dataset name")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to trained model checkpoint")
    parser.add_argument("--backbone", type=str, default="ViT-L/14@336px",
                        choices=VisualAD_lib.available_models(), help="CLIP backbone to use for testing")
    parser.add_argument("--feature_config", type=str, default=os.path.join('configs', 'backbone_layers.yaml'),
                        help="YAML file specifying default feature layers per backbone")
    parser.add_argument("--features_list", type=int, nargs="*", default=[6, 12, 18, 24], help="Override feature layers")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
    parser.add_argument("--device", type=str, default="cuda:1", help="device to use")
    parser.add_argument("--enable_analysis", action="store_true", help="enable data analysis and visualization")
    parser.add_argument("--max_samples", type=int, default=None, help="maximum number of samples to test (for debugging)")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    setup_seed(args.seed)
    
    # Mark which parameters were explicitly provided (not using defaults from checkpoint)
    import sys
    provided_args = set()
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--'):
            provided_args.add(arg[2:])
    
    # Set flags for explicitly provided parameters
    args._features_list_provided = 'features_list' in provided_args
    args._image_size_provided = 'image_size' in provided_args
    args._backbone_provided = 'backbone' in provided_args
    
    device = torch.device(args.device)
    test(args)
