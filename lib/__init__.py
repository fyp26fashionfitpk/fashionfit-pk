"""FashionFit-PK reusable core library for Pakistani 4-class parsing, masking, and metrics."""

from .mask_utils import (
    CLASS_NAMES,
    CLASS_TO_ID,
    build_tryon_mask,
    make_flared_lehenga_mask,
    preserve_dupatta_over_mask,
)
from .metrics import (
    compute_per_class_iou,
    validate_zero_leakage_sku_split,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_ID",
    "build_tryon_mask",
    "make_flared_lehenga_mask",
    "preserve_dupatta_over_mask",
    "compute_per_class_iou",
    "validate_zero_leakage_sku_split",
]
