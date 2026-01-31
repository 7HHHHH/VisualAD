"""
Scoring utilities for reducing anomaly maps to scalar classification scores.
"""
import math
from typing import Literal

import torch

# Default ratio for top-k aggregation, tuned to balance peak strength and area
DEFAULT_TOPK_RATIO = 0.01


def _validate_anomaly_map(anomaly_map: torch.Tensor) -> torch.Tensor:
    """
    Ensure anomaly_map is a 2D or 3D tensor and return it as a float tensor.
    """
    if anomaly_map.dim() not in (2, 3):
        raise ValueError(
            f"anomaly_map must be 2D or 3D tensor, got shape {tuple(anomaly_map.shape)}"
        )
    return anomaly_map if anomaly_map.is_floating_point() else anomaly_map.float()


def reduce_anomaly_map(
    anomaly_map: torch.Tensor,
    mode: Literal["topk_mean", "softmax", "mean"] = "topk_mean",
    topk_ratio: float = DEFAULT_TOPK_RATIO,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reduce an anomaly map to a scalar score in a differentiable manner.

    Args:
        anomaly_map: Tensor of shape [H, W] or [B, H, W].
        mode: Aggregation strategy ('topk_mean', 'softmax', or 'mean').
        topk_ratio: Ratio of pixels to use for top-k pooling (0, 1].
        temperature: Softmax temperature (only used when mode='softmax').
        eps: Numerical safety term.

    Returns:
        Tensor of shape [] or [B] containing reduced scores.
    """
    anomaly_map = _validate_anomaly_map(anomaly_map)
    if anomaly_map.dim() == 2:
        flat_map = anomaly_map.reshape(1, -1)
        squeeze_output = True
    else:
        flat_map = anomaly_map.reshape(anomaly_map.shape[0], -1)
        squeeze_output = False

    if mode == "topk_mean":
        ratio = float(topk_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
        num_elements = flat_map.shape[-1]
        k = max(1, int(math.ceil(num_elements * ratio)))
        topk_values = torch.topk(flat_map, k, dim=-1).values
        scores = topk_values.mean(dim=-1)
    elif mode == "softmax":
        temp = max(temperature, eps)
        weights = torch.softmax(flat_map / temp, dim=-1)
        scores = (weights * flat_map).sum(dim=-1)
    elif mode == "mean":
        scores = flat_map.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported reduction mode: {mode}")

    return scores.squeeze(0) if squeeze_output else scores


__all__ = ["reduce_anomaly_map", "DEFAULT_TOPK_RATIO"]
