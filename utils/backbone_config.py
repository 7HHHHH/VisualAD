"""Helper utilities for selecting backbone feature layers."""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
import yaml


def _log(logger, level: str, message: str) -> None:
    if logger is None:
        print(message)
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)
    else:
        print(message)


def resolve_features_list(
    requested_layers: Optional[Sequence[int]],
    total_layers: int,
    logger=None,
    default_count: int = 4,
) -> List[int]:
    """Return a valid, deduplicated list of feature layers for the backbone."""

    if requested_layers is None:
        requested_layers = []

    unique_requested = sorted({int(layer) for layer in requested_layers if layer and layer > 0})
    target_count = len(unique_requested) if unique_requested else default_count
    target_count = min(target_count, total_layers)

    if target_count == 0:
        return []

    valid_layers = [layer for layer in unique_requested if layer <= total_layers]
    missing_layers = sorted(set(unique_requested) - set(valid_layers))

    if missing_layers:
        _log(
            logger,
            "warning",
            f"Dropping unsupported feature layers {missing_layers} for backbone depth {total_layers}.",
        )

    def generate_fallback(count: int) -> List[int]:
        fallback: List[int] = []
        for value in np.linspace(1, total_layers, count):
            layer = int(round(value))
            layer = max(1, min(total_layers, layer))
            fallback.append(layer)
        fallback = sorted(set(fallback))
        if not fallback:
            fallback = [total_layers]
        return fallback

    if len(valid_layers) < target_count:
        fallback_layers = generate_fallback(target_count)
        for layer in fallback_layers:
            if layer not in valid_layers:
                valid_layers.append(layer)
            if len(valid_layers) >= target_count:
                break

    if not valid_layers:
        valid_layers = generate_fallback(target_count)
        _log(
            logger,
            "warning",
            "Requested layers are invalid for backbone depth "
            f"{total_layers}; using auto-selected layers {valid_layers}.",
        )
    else:
        valid_layers = sorted(set(valid_layers))
        if len(valid_layers) < target_count:
            _log(
                logger,
                "info",
                f"Using available feature layers {valid_layers} (target count {target_count}) "
                f"for depth {total_layers}.",
            )

    if not unique_requested:
        _log(
            logger,
            "info",
            f"No feature layers specified; using default layers {valid_layers} for depth {total_layers}.",
        )

    return valid_layers


def load_feature_layers_from_config(
    config_path: Optional[str],
    backbone: str,
    logger=None,
) -> Optional[List[int]]:
    """Load default feature layers for a backbone from a YAML config."""

    if not config_path:
        return None

    if not os.path.exists(config_path):
        _log(
            logger,
            "warning",
            f"Feature config file not found at {config_path}; falling back to dynamic selection.",
        )
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            config_data = yaml.safe_load(fp) or {}
    except yaml.YAMLError as exc:
        _log(logger, "error", f"Failed to parse feature config {config_path}: {exc}")
        raise

    entry = config_data.get(backbone)
    if entry is None:
        _log(
            logger,
            "info",
            f"No feature layers configured for backbone {backbone} in {config_path}; using dynamic selection.",
        )
        return None

    layers = entry.get("layers") if isinstance(entry, dict) else entry

    if not isinstance(layers, (list, tuple)):
        _log(
            logger,
            "warning",
            f"Invalid layer specification for {backbone} in {config_path}; using dynamic selection.",
        )
        return None

    try:
        layers = [int(layer) for layer in layers]
    except (TypeError, ValueError):
        _log(
            logger,
            "warning",
            f"Non-integer layer ids detected for {backbone} in {config_path}; using dynamic selection.",
        )
        return None

    _log(logger, "info", f"Loaded feature layers {layers} for backbone {backbone} from {config_path}.")
    return layers

