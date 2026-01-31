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
from utils.metrics import compute_metrics
from scipy.ndimage import gaussian_filter
from utils.feature_transform import create_feature_transform
from utils.analysis import get_classification_from_segmentation, analyze_classification_distribution
from utils.visualization import visualize_anomaly_results
from utils.anomaly_detection import generate_anomaly_map_from_tokens


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def test(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    # Load checkpoint (required)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Use checkpoint values
    args.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    args.image_size = checkpoint.get("image_size", 518)
    args.features_list = checkpoint.get("features_list", [6, 12, 18, 24])

    preprocess, target_transform = get_transform(args)

    # Load model
    model, _ = VisualAD_lib.load(args.backbone, device=device)
    model.eval()
    model.to(device)

    feature_dim = model.visual.embed_dim

    # Load trained tokens
    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)

    # Load feature transforms
    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict['mlp.0.weight'].shape[0]
            layer_transforms[layer_name] = create_feature_transform(
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0
            ).to(device)
            layer_transforms[layer_name].load_state_dict(state_dict)
            layer_transforms[layer_name].eval()

    # Load cross-attention
    cross_attn = None
    if "cross_attn" in checkpoint:
        from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
        config = checkpoint.get("cross_attn_config", {})
        cross_attn = build_layer_adaptive_cross_attention(
            layers=args.features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            res_scale_init=config.get("res_scale_init", 0.01)
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    # Test dataset
    test_data = Dataset(root=args.test_data_path, transform=preprocess,
                       target_transform=target_transform, dataset_name=args.test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []} for obj in obj_list}

    # Data for analysis
    all_original_images = []
    all_anomaly_maps = []
    all_gt_masks = []
    all_cls_names = []
    all_anomaly_labels = []
    all_img_paths = []

    for items in tqdm(test_dataloader):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            vision_output = model.encode_image(image, args.features_list)
            anomaly_features = vision_output['anomaly_features']
            normal_features = vision_output['normal_features']
            patch_tokens = vision_output['patch_tokens']
            patch_start_idx = vision_output['patch_start_idx']

            # Cross-Attention enhancement
            patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
            if cross_attn is not None:
                adapted_list = cross_attn(anomaly_features, normal_features, patch_features_list, args.features_list)
                anomaly_features_list = [a['anomaly'] for a in adapted_list]
                normal_features_list = [a['normal'] for a in adapted_list]
            else:
                anomaly_features_list = [anomaly_features] * len(patch_tokens)
                normal_features_list = [normal_features] * len(patch_tokens)

            # Generate anomaly maps
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_tokens):
                anomaly_feat_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
                normal_feat_norm = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)

                transform_key = f'layer_{args.features_list[idx]}'
                if transform_key in layer_transforms:
                    B, N, D = patch_feature.shape
                    patch_feature = layer_transforms[transform_key](patch_feature.view(-1, D)).view(B, N, D)

                anomaly_map = generate_anomaly_map_from_tokens(
                    anomaly_feat_norm, normal_feat_norm,
                    patch_feature[:, patch_start_idx:, :],
                    args.image_size
                )
                anomaly_map_list.append(anomaly_map)

            # Fuse and filter
            final_anomaly_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()
            filtered_map = gaussian_filter(final_anomaly_map[0].numpy(), sigma=args.sigma)
            final_anomaly_map = torch.from_numpy(filtered_map).unsqueeze(0)

            results[cls_name]['anomaly_maps'].append(final_anomaly_map)

            all_original_images.append(image.detach().cpu())
            all_anomaly_maps.append(final_anomaly_map)
            all_gt_masks.append(gt_mask)
            all_cls_names.append(cls_name)
            all_anomaly_labels.append(items['anomaly'].item())
            all_img_paths.append(items['img_path'][0])

    # Compute metrics
    fused_scores, normalized_anomaly_maps = get_classification_from_segmentation(all_anomaly_maps, all_cls_names, results)
    compute_metrics(results, obj_list, logger)

    # Analysis (optional)
    if args.enable_analysis:
        analysis_dir = os.path.join(args.save_path, 'analysis')
        analyze_classification_distribution(fused_scores, all_cls_names, all_anomaly_labels, analysis_dir)
        visualize_anomaly_results(
            all_original_images, normalized_anomaly_maps, all_gt_masks, fused_scores,
            all_cls_names, all_img_paths, all_anomaly_labels, args.test_dataset, analysis_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VisualAD Test", add_help=True)
    parser.add_argument("--test_data_path", type=str, required=True, help="test dataset path")
    parser.add_argument("--save_path", type=str, default='./test_results', help='path to save test results')
    parser.add_argument("--test_dataset", type=str, required=True, help="test dataset name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
    parser.add_argument("--device", type=str, default="cuda:1", help="device to use")
    parser.add_argument("--enable_analysis", action="store_true", help="enable data analysis and visualization")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
