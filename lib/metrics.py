"""Evaluation metrics and dataset zero-leakage validation utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Set
import numpy as np

from .mask_utils import CLASS_NAMES


def compute_per_class_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Compute exact per-class IoU (%) and mean IoU (%) over present classes."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} != target {target.shape}")

    results: Dict[str, float] = {}
    valid_ious = []
    for cls_id in range(num_classes):
        pred_c = pred == cls_id
        target_c = target == cls_id
        inter = int(np.logical_and(pred_c, target_c).sum())
        union = int(np.logical_or(pred_c, target_c).sum())
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        if union == 0:
            results[name] = 100.0
        else:
            iou = 100.0 * inter / union
            results[name] = round(iou, 2)
            valid_ious.append(iou)

    results["mIoU"] = round(float(np.mean(valid_ious)) if valid_ious else 0.0, 2)
    return results


def validate_zero_leakage_sku_split(
    train_skus: Iterable[str],
    test_skus: Iterable[str],
) -> Set[str]:
    """Verify strict SKU-level disjointness between train and test splits.

    Returns the set of overlapping SKUs (empty set means 0% data leakage).
    """
    train_set = {s.strip() for s in train_skus if s and s.strip()}
    test_set = {s.strip() for s in test_skus if s and s.strip()}
    return train_set.intersection(test_set)
