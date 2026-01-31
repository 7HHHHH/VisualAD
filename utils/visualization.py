"""
Visualization utilities for anomaly detection analysis and results
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
import cv2
import warnings

# Configure matplotlib to avoid font warnings
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def visualize_single_sample(original_image, anomaly_map, gt_mask, classification_score,
                            cls_name, img_path, anomaly_label, save_path):
    """
    Visualize anomaly detection results for a single sample (per-image normalization + heatmap overlay)

    Args:
        original_image: Original image tensor [C, H, W]
        anomaly_map: Anomaly map tensor [H, W]
        gt_mask: GT mask tensor [H, W]
        classification_score: Classification score float
        cls_name: Class name str
        img_path: Image path str
        anomaly_label: True anomaly label (0=normal, 1=anomaly)
        save_path: Save path str (for combined 3-in-1 image)
    """
    # Prepare save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get base filename (without extension)
    base_path = save_path.rsplit('.', 1)[0]

    # Original image
    img = original_image.squeeze().cpu().numpy()
    if img.shape[0] == 3:  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]

    # GT mask
    gt = gt_mask.squeeze().cpu().numpy()

    # Anomaly map (per-image normalization)
    amap = anomaly_map.squeeze().cpu().numpy()
    amap_min = amap.min()
    amap_max = amap.max()
    if amap_max > amap_min:
        amap = (amap - amap_min) / (amap_max - amap_min)
    else:
        amap = np.zeros_like(amap)

    # Convert to 0-255 heatmap
    amap_uint8 = (amap * 255).astype(np.uint8)

    # Apply jet colormap to generate colored heatmap
    heatmap = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    # Convert original image to uint8 format
    img_uint8 = (img * 255).astype(np.uint8)

    # Resize heatmap to match original image
    if heatmap.shape[:2] != img_uint8.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))

    # Overlay heatmap on original image (alpha blending: 0.5 * heatmap + 0.5 * original)
    overlay = cv2.addWeighted(heatmap, 0.5, img_uint8, 0.5, 0)

    # 1. Save individual original image (no title)
    fig_original = plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{base_path}_original.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig_original)

    # 2. Save individual GT mask (no title)
    fig_gt = plt.figure(figsize=(4, 4))
    plt.imshow(gt, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{base_path}_gt.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig_gt)

    # 3. Save individual heatmap overlay (no title)
    fig_overlay = plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{base_path}_overlay.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig_overlay)

    # 4. Save combined 3-in-1 image
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title(f'Original (Score: {classification_score:.3f})')
    axes[0].axis('off')

    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Heatmap Overlay (0.5:0.5)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_anomaly_results(original_images, anomaly_maps, gt_masks,
                              classification_scores, cls_names, img_paths,
                              anomaly_labels, dataset_name, save_dir):
    """
    Visualize anomaly detection results, with per-image independent normalization

    Args:
        original_images: List of original images
        anomaly_maps: List of anomaly maps (can be normalized or not, will be renormalized per-image)
        gt_masks: List of GT masks
        classification_scores: List of classification scores
        cls_names: List of class names
        img_paths: List of image paths
        anomaly_labels: List of true anomaly labels (0=normal, 1=anomaly)
        dataset_name: Dataset name
        save_dir: Save directory
    """
    # Organize data by class
    class_data = {}
    for i, cls_name in enumerate(cls_names):
        if cls_name not in class_data:
            class_data[cls_name] = {'images': [], 'maps': [], 'masks': [], 'scores': [],
                                   'labels': [], 'paths': [], 'indices': []}
        class_data[cls_name]['images'].append(original_images[i])
        class_data[cls_name]['maps'].append(anomaly_maps[i])
        class_data[cls_name]['masks'].append(gt_masks[i])
        class_data[cls_name]['scores'].append(classification_scores[i])
        class_data[cls_name]['labels'].append(anomaly_labels[i])
        class_data[cls_name]['paths'].append(img_paths[i])
        class_data[cls_name]['indices'].append(i)

    total_saved = 0
    # Generate visualization for each class
    for cls_name, data in class_data.items():
        # Create class directory: dataset_name/class_name
        class_save_dir = os.path.join(save_dir, dataset_name, cls_name)
        os.makedirs(class_save_dir, exist_ok=True)

        # Generate individual images for all samples
        for idx in range(len(data['images'])):
            score = data['scores'][idx]
            true_label = data['labels'][idx]
            img_path = data['paths'][idx]

            # Determine type based on true label
            true_type = "normal" if true_label == 0 else "anomaly"

            # Extract filename from path (without extension)
            source_filename = os.path.splitext(os.path.basename(img_path))[0]

            # New naming format: score_{classification_score}_{true_type}_{source_filename}.png
            filename = f'score_{score:.3f}_{true_type}_{source_filename}.png'
            save_path = os.path.join(class_save_dir, filename)

            # Visualize single sample using dedicated function
            visualize_single_sample(
                original_image=data['images'][idx],
                anomaly_map=data['maps'][idx],
                gt_mask=data['masks'][idx],
                classification_score=score,
                cls_name=cls_name,
                img_path=img_path,
                anomaly_label=true_label,
                save_path=save_path
            )
            total_saved += 1

        print(f"✅ Saved {len(data['images'])} individual images for class '{cls_name}' in {class_save_dir}")

    print(f"✅ Total visualization images saved: {total_saved}")


def generate_overall_analysis_chart(normal_scores, anomaly_scores, class_stats, save_dir):
    """Generate overall analysis charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1: 整体分布直方图
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        axes[0, 0].hist(normal_scores, bins=30, alpha=0.7, label=f'Normal ({len(normal_scores)})', color='blue')
        axes[0, 0].hist(anomaly_scores, bins=30, alpha=0.7, label=f'Anomaly ({len(anomaly_scores)})', color='red')
        axes[0, 0].set_xlabel('Classification Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 按类别的样本数量分布
    class_names = [stat['name'] for stat in class_stats]
    normal_counts = [stat['normal_count'] for stat in class_stats]
    anomaly_counts = [stat['anomaly_count'] for stat in class_stats]
    
    x = np.arange(len(class_names))
    width = 0.35
    axes[0, 1].bar(x - width/2, normal_counts, width, label='Normal', alpha=0.7, color='blue')
    axes[0, 1].bar(x + width/2, anomaly_counts, width, label='Anomaly', alpha=0.7, color='red')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Sample Count')
    axes[0, 1].set_title('Sample Count by Class')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 按类别的平均得分
    normal_means = [np.mean(stat['normal_scores']) if len(stat['normal_scores']) > 0 else 0 for stat in class_stats]
    anomaly_means = [np.mean(stat['anomaly_scores']) if len(stat['anomaly_scores']) > 0 else 0 for stat in class_stats]
    
    axes[1, 0].bar(x - width/2, normal_means, width, label='Normal', alpha=0.7, color='blue')
    axes[1, 0].bar(x + width/2, anomaly_means, width, label='Anomaly', alpha=0.7, color='red')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Mean Score')
    axes[1, 0].set_title('Mean Scores by Class')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 分类偏向分析
    anomaly_ratios = [stat['anomaly_ratio'] for stat in class_stats]
    colors = ['red' if r > 0.5 else 'blue' for r in anomaly_ratios]
    
    bars = axes[1, 1].barh(range(len(class_names)), anomaly_ratios, color=colors, alpha=0.7)
    axes[1, 1].set_yticks(range(len(class_names)))
    axes[1, 1].set_yticklabels(class_names)
    axes[1, 1].set_xlabel('Anomaly Classification Ratio')
    axes[1, 1].set_title('Classification Bias by Class')
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加数值标注
    for i, ratio in enumerate(anomaly_ratios):
        axes[1, 1].text(ratio + 0.01, i, f'{ratio:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'classification_overall_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_class_wise_analysis_charts(class_stats, save_dir):
    """为每个类别生成详细分析图表"""
    for stat in class_stats:
        cls_name = stat['name']
        normal_scores = stat['normal_scores']
        anomaly_scores = stat['anomaly_scores']
        all_scores = stat['all_scores']
        
        if len(all_scores) == 0:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Class: {cls_name} - Detailed Analysis', fontsize=16)
        
        # 子图1: 分布直方图
        if len(normal_scores) > 0 and len(anomaly_scores) > 0:
            axes[0, 0].hist(normal_scores, bins=20, alpha=0.7, label=f'Normal ({len(normal_scores)})', color='blue')
            axes[0, 0].hist(anomaly_scores, bins=20, alpha=0.7, label=f'Anomaly ({len(anomaly_scores)})', color='red')
        elif len(normal_scores) > 0:
            axes[0, 0].hist(normal_scores, bins=20, alpha=0.7, label=f'Normal ({len(normal_scores)})', color='blue')
        elif len(anomaly_scores) > 0:
            axes[0, 0].hist(anomaly_scores, bins=20, alpha=0.7, label=f'Anomaly ({len(anomaly_scores)})', color='red')
        
        axes[0, 0].set_xlabel('Classification Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 子图2: 箱线图
        data_to_plot = []
        labels_plot = []
        if len(normal_scores) > 0:
            data_to_plot.append(normal_scores)
            labels_plot.append('Normal')
        if len(anomaly_scores) > 0:
            data_to_plot.append(anomaly_scores)
            labels_plot.append('Anomaly')
        
        if data_to_plot:
            box_plot = axes[0, 1].boxplot(data_to_plot, tick_labels=labels_plot, patch_artist=True)
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                patch.set_facecolor(color)
            axes[0, 1].set_ylabel('Classification Score')
            axes[0, 1].set_title('Score Distribution (Box Plot)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 子图3: 得分随样本的变化
        axes[1, 0].plot(range(len(all_scores)), all_scores, 'o-', markersize=3, alpha=0.7)
        axes[1, 0].axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Threshold=0.0')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Classification Score')
        axes[1, 0].set_title('Score Sequence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 子图4: 统计信息
        axes[1, 1].axis('off')
        info_text = f"""
Statistics for {cls_name}:

Total Samples: {len(all_scores)}
Normal: {len(normal_scores)} ({len(normal_scores)/len(all_scores)*100:.1f}%)
Anomaly: {len(anomaly_scores)} ({len(anomaly_scores)/len(all_scores)*100:.1f}%)

Overall Score:
  Mean: {np.mean(all_scores):.3f}
  Std: {np.std(all_scores):.3f}
  Min: {np.min(all_scores):.3f}
  Max: {np.max(all_scores):.3f}

Classification Bias:
  Anomaly Ratio: {stat['anomaly_ratio']:.3f}
  Tendency: {stat['bias']}
        """
        axes[1, 1].text(0.1, 0.9, info_text.strip(), transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'classification_{cls_name}_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved detailed analysis for class '{cls_name}'")