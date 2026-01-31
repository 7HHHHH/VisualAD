"""
Analysis utilities for anomaly detection evaluation and statistics
"""
import os
import numpy as np
import torch
from .visualization import generate_overall_analysis_chart, generate_class_wise_analysis_charts
from .scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO


def get_classification_from_segmentation(all_anomaly_maps, all_cls_names, results=None,
                                        normalized_cls_scores=None, weight=0.0):
    """
    Complete process to get classification results from segmentation results
    Args:
        all_anomaly_maps: List of raw anomaly maps [tensor1, tensor2, ...]
        all_cls_names: List of class names
        results: Test result dictionary (optional, will update probability values if provided)
        normalized_cls_scores: List of normalized classification scores (optional, for fusion)
        weight: Weight, default 0.0 means using segmentation score only
    Returns:
        tuple: (classification_scores, normalized_anomaly_maps)
            - classification_scores: List of classification scores [float1, float2, ...]
            - normalized_anomaly_maps: List of normalized anomaly maps (for visualization)
    """
    from .normalization import normalize_anomaly_maps_per_image

    # Step 1: Directly compute classification scores using raw anomaly maps (no normalization)
    print(f"\n🔍 Computing classification scores using raw anomaly maps (no normalization)...")
    classification_scores = compute_and_fuse_scores(all_anomaly_maps, normalized_cls_scores, weight)

    # Step 2: Normalize anomaly maps (for visualization only)
    print(f"\n📊 Normalizing anomaly maps (for visualization)...")
    normalized_anomaly_maps = normalize_anomaly_maps_per_image(all_anomaly_maps)

    # Step 3: Update probability values in results (if results provided)
    if results is not None:
        print(f"\n🔄 Updating classification probabilities for metric calculation...")
        update_results_with_fused_scores(results, classification_scores, all_cls_names)

    return classification_scores, normalized_anomaly_maps


def compute_and_fuse_scores(anomaly_maps, normalized_cls_scores=None, weight=0.5):
    """
    Compute segmentation scores and fuse classification and segmentation scores
    Args:
        anomaly_maps: List of anomaly maps [tensor1, tensor2, ...] (can be normalized or unnormalized)
        normalized_cls_scores: List of normalized classification scores [float1, float2, ...] (optional)
        weight: Weight, default 0.5 means equal weight fusion, weight=0 uses only segmentation scores
    Returns:
        fused_scores: List of fused scores [float1, float2, ...]
    """
    # Compute segmentation scores
    seg_scores = []

    for anomaly_map in anomaly_maps:
        # Use Top-K average aggregation for anomaly maps to reduce noise impact
        reduced_score = reduce_anomaly_map(anomaly_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
        seg_scores.append(reduced_score.item())

    print(f"✅ Computed segmentation scores for {len(seg_scores)} samples (Top-K ratio={DEFAULT_TOPK_RATIO:.3f})")

    # If no classification scores provided or weight is 0, return segmentation scores directly
    if normalized_cls_scores is None or weight == 0:
        print(f"✅ Using segmentation scores only (weight={weight})")
        seg_mean = np.mean(seg_scores)
        seg_std = np.std(seg_scores)
        seg_min = np.min(seg_scores)
        seg_max = np.max(seg_scores)
        print(f"  Segmentation score statistics: mean={seg_mean:.3f}, std={seg_std:.3f}, min={seg_min:.3f}, max={seg_max:.3f}")
        return seg_scores

    # Fuse classification and segmentation scores
    if len(normalized_cls_scores) != len(seg_scores):
        raise ValueError(f"Classification score count ({len(normalized_cls_scores)}) does not match segmentation score count ({len(seg_scores)})")

    fused_scores = []

    for cls_score, seg_score in zip(normalized_cls_scores, seg_scores):
        # Fusion formula: weight * classification_score + (1-weight) * segmentation_score
        fused_score = weight * cls_score + (1 - weight) * seg_score
        fused_scores.append(fused_score)

    # Print fusion statistics
    cls_mean = np.mean(normalized_cls_scores)
    seg_mean = np.mean(seg_scores)
    fused_mean = np.mean(fused_scores)

    print(f"✅ Fusion completed for {len(fused_scores)} samples (weight={weight})")
    print(f"  Classification score mean: {cls_mean:.3f}")
    print(f"  Segmentation score mean: {seg_mean:.3f}")
    print(f"  Fused score mean: {fused_mean:.3f}")

    return fused_scores


def update_results_with_fused_scores(results, fused_scores, cls_names):
    """
    Update probability values in results with fused scores to affect metric calculation
    Args:
        results: Test result dictionary
        fused_scores: List of fused scores [float1, float2, ...]
        cls_names: List of class names
    """
    # Organize fused scores by class
    class_fused_scores = {}
    for i, cls_name in enumerate(cls_names):
        if cls_name not in class_fused_scores:
            class_fused_scores[cls_name] = []
        class_fused_scores[cls_name].append(fused_scores[i])

    # Update probability values for each class
    updated_count = 0
    for cls_name, scores in class_fused_scores.items():
        if cls_name in results:
            # Convert scores to tensor and replace original pr_sp
            fused_tensor = torch.tensor(scores, dtype=torch.float32)
            results[cls_name]['pr_sp'] = fused_tensor
            updated_count += len(scores)

    print(f"✅ Updated classification probabilities for {len(class_fused_scores)} classes totaling {updated_count} samples")


def analyze_classification_distribution(classification_scores, cls_names, anomaly_labels, save_dir):
    """
    Analyze classification distribution with detailed analysis by class
    Args:
        classification_scores: List of classification scores
        cls_names: List of class names
        anomaly_labels: List of anomaly labels (0=normal, 1=anomaly)
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy arrays
    scores = np.array(classification_scores)
    labels = np.array(anomaly_labels)

    # Organize data by class
    class_data = {}
    for i, cls_name in enumerate(cls_names):
        if cls_name not in class_data:
            class_data[cls_name] = {'normal_scores': [], 'anomaly_scores': [], 'all_scores': []}
        class_data[cls_name]['all_scores'].append(scores[i])
        if labels[i] == 0:
            class_data[cls_name]['normal_scores'].append(scores[i])
        else:
            class_data[cls_name]['anomaly_scores'].append(scores[i])

    # Print overall statistics
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    print(f"\n📊 Classification Score Distribution Analysis:")
    print(f"Total samples: {len(scores)}")
    print(f"Normal samples: {len(normal_scores)} ({len(normal_scores)/len(scores)*100:.1f}%)")
    print(f"Anomaly samples: {len(anomaly_scores)} ({len(anomaly_scores)/len(scores)*100:.1f}%)")

    if len(normal_scores) > 0:
        print(f"\nOverall normal sample score statistics:")
        print(f"  Mean: {normal_scores.mean():.3f}, Std: {normal_scores.std():.3f}")
        print(f"  Min: {normal_scores.min():.3f}, Max: {normal_scores.max():.3f}")
        print(f"  Median: {np.median(normal_scores):.3f}")

    if len(anomaly_scores) > 0:
        print(f"\nOverall anomaly sample score statistics:")
        print(f"  Mean: {anomaly_scores.mean():.3f}, Std: {anomaly_scores.std():.3f}")
        print(f"  Min: {anomaly_scores.min():.3f}, Max: {anomaly_scores.max():.3f}")
        print(f"  Median: {np.median(anomaly_scores):.3f}")

    # Detailed analysis by class
    print(f"\n📈 Detailed Analysis by Class:")
    threshold = 0.0
    class_stats = []

    for cls_name in sorted(class_data.keys()):
        data = class_data[cls_name]
        normal_scores_cls = np.array(data['normal_scores'])
        anomaly_scores_cls = np.array(data['anomaly_scores'])
        all_scores_cls = np.array(data['all_scores'])

        total_samples = len(all_scores_cls)
        normal_count = len(normal_scores_cls)
        anomaly_count = len(anomaly_scores_cls)

        # Compute classification bias
        predicted_anomaly = all_scores_cls > threshold
        anomaly_ratio = np.mean(predicted_anomaly)
        bias = "More anomaly" if anomaly_ratio > 0.5 else "More normal"

        print(f"\n🏷️  {cls_name}:")
        print(f"   Total samples: {total_samples}")
        print(f"   Normal: {normal_count} ({normal_count/total_samples*100:.1f}%), Anomaly: {anomaly_count} ({anomaly_count/total_samples*100:.1f}%)")

        if len(normal_scores_cls) > 0:
            print(f"   Normal sample scores: mean={normal_scores_cls.mean():.3f}, std={normal_scores_cls.std():.3f}")
        if len(anomaly_scores_cls) > 0:
            print(f"   Anomaly sample scores: mean={anomaly_scores_cls.mean():.3f}, std={anomaly_scores_cls.std():.3f}")

        print(f"   Classification bias: {anomaly_ratio:.3f} ({bias})")

        class_stats.append({
            'name': cls_name,
            'total': total_samples,
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'normal_scores': normal_scores_cls,
            'anomaly_scores': anomaly_scores_cls,
            'all_scores': all_scores_cls,
            'anomaly_ratio': anomaly_ratio,
            'bias': bias
        })

    # Generate overall analysis chart
    generate_overall_analysis_chart(normal_scores, anomaly_scores, class_stats, save_dir)

    # Generate detailed analysis charts for each class
    generate_class_wise_analysis_charts(class_stats, save_dir)

    # Print classification bias summary
    print(f"\n🎯 Classification Bias Summary (threshold={threshold}):")
    print(f"{'Class':<15} {'Samples':<8} {'Anomaly Ratio':<12} {'Bias'}")
    print("-" * 50)

    # Sort by anomaly ratio
    sorted_stats = sorted(class_stats, key=lambda x: x['anomaly_ratio'], reverse=True)
    for stat in sorted_stats:
        print(f"{stat['name']:<15} {stat['total']:<8} {stat['anomaly_ratio']:<12.3f} {stat['bias']}")

    overall_anomaly_ratio = np.mean(scores > threshold)
    overall_bias = "More anomaly" if overall_anomaly_ratio > 0.5 else "More normal"
    print(f"{'Overall':<15} {len(scores):<8} {overall_anomaly_ratio:<12.3f} {overall_bias}")

    print(f"\n✅ Analysis charts saved to: {save_dir}/")
