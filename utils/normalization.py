"""
Normalization utilities for anomaly detection data processing
"""
import torch
import numpy as np


def normalize_anomaly_maps_per_image(anomaly_maps):
    """
    Normalize each anomaly map independently (per-image normalization)
    Args:
        anomaly_maps: List of anomaly maps [tensor1, tensor2, ...]
    Returns:
        normalized_maps: List of normalized anomaly maps
    """
    normalized_maps = []

    for i, map_tensor in enumerate(anomaly_maps):
        # Get min/max values for current image
        map_min = map_tensor.min().item()
        map_max = map_tensor.max().item()

        # Normalize to [0, 1]
        if map_max > map_min:
            normalized_map = (map_tensor - map_min) / (map_max - map_min)
        else:
            # If all values are the same, set to 0
            normalized_map = torch.zeros_like(map_tensor)

        normalized_maps.append(normalized_map)

    print(f"Completed independent normalization for {len(anomaly_maps)} images")
    return normalized_maps


def normalize_anomaly_maps_by_class(anomaly_maps, cls_names):
    """
    Normalize anomaly maps by class (within-class normalization)
    Args:
        anomaly_maps: List of anomaly maps [tensor1, tensor2, ...]
        cls_names: List of class names
    Returns:
        normalized_maps: List of normalized anomaly maps
    """
    # Organize anomaly maps by class
    class_maps = {}
    for i, cls_name in enumerate(cls_names):
        if cls_name not in class_maps:
            class_maps[cls_name] = {'maps': [], 'indices': []}
        class_maps[cls_name]['maps'].append(anomaly_maps[i])
        class_maps[cls_name]['indices'].append(i)

    # Compute normalization parameters for each class and apply
    normalized_maps = [None] * len(anomaly_maps)

    for cls_name, data in class_maps.items():
        maps = data['maps']
        indices = data['indices']

        # Merge all anomaly maps of this class to compute global min/max
        all_values = []
        for map_tensor in maps:
            all_values.extend(map_tensor.flatten().numpy())

        global_min = np.min(all_values)
        global_max = np.max(all_values)

        print(f"  {cls_name}: Original range [{global_min:.3f}, {global_max:.3f}]")

        # Normalize each anomaly map of this class
        for map_tensor, idx in zip(maps, indices):
            if global_max > global_min:
                normalized_map = (map_tensor - global_min) / (global_max - global_min)
            else:
                # If all values are the same, set to 0
                normalized_map = torch.zeros_like(map_tensor)

            normalized_maps[idx] = normalized_map

    print(f"Completed anomaly map normalization for {len(class_maps)} classes")
    return normalized_maps


def normalize_classification_scores_by_class(classification_scores, cls_names):
    """
    Normalize classification scores by class
    Args:
        classification_scores: List of classification scores [float1, float2, ...]
        cls_names: List of class names
    Returns:
        normalized_scores: List of normalized classification scores
    """
    # Organize classification scores by class
    class_scores = {}
    for i, cls_name in enumerate(cls_names):
        if cls_name not in class_scores:
            class_scores[cls_name] = {'scores': [], 'indices': []}
        class_scores[cls_name]['scores'].append(classification_scores[i])
        class_scores[cls_name]['indices'].append(i)

    # Compute normalization parameters for each class and apply
    normalized_scores = [None] * len(classification_scores)

    for cls_name, data in class_scores.items():
        scores = np.array(data['scores'])
        indices = data['indices']

        score_min = np.min(scores)
        score_max = np.max(scores)

        print(f"  {cls_name}: Classification score original range [{score_min:.3f}, {score_max:.3f}]")

        # Normalize each classification score of this class
        for score, idx in zip(scores, indices):
            if score_max > score_min:
                normalized_score = (score - score_min) / (score_max - score_min)
            else:
                # If all values are the same, set to 0.5
                normalized_score = 0.5

            normalized_scores[idx] = normalized_score

    print(f"Completed classification score normalization for {len(class_scores)} classes")
    return normalized_scores
