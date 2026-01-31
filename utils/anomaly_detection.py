"""
Anomaly detection utilities for feature processing and map generation
"""
import torch
import torch.nn.functional as F
import numpy as np


def generate_anomaly_map_from_tokens(anomaly_features, normal_features, patch_tokens, image_size):
    """
    Generate pixel-level anomaly map using token features
    Args:
        anomaly_features: [B, dim] - anomaly token features
        normal_features: [B, dim] - normal token features  
        patch_tokens: [B, num_patches, dim] - patch token features
        image_size: target image size
    Returns:
        anomaly_map: [B, H, W] - pixel-level anomaly map
    """
    B = anomaly_features.shape[0]
    
    # Compute similarity between each patch and anomaly/normal tokens
    anomaly_sim = torch.cosine_similarity(
        patch_tokens, anomaly_features.unsqueeze(1), dim=2
    )  # [B, num_patches]
    
    normal_sim = torch.cosine_similarity(
        patch_tokens, normal_features.unsqueeze(1), dim=2
    )  # [B, num_patches]
    
    # Anomaly score = anomaly_similarity - normal_similarity
    anomaly_score = anomaly_sim - normal_sim  # [B, num_patches]
    
    # Reshape to spatial dimensions
    patch_size = int(np.sqrt(anomaly_score.shape[1]))
    anomaly_map = anomaly_score.reshape(B, patch_size, patch_size)
    
    # Resize to target image size
    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(1), 
        size=(image_size, image_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(1)
    
    return anomaly_map