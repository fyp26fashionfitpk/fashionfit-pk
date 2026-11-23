"""Core 4-class garment masking and flared lehenga A-line dilation utilities.

Classes used across FashionFit-PK (Adult Pret, Kids Ethnic, and Wedding Formals):
    0: other (background, face, hair, hands, skin, shoes)
    1: kurta_kameez (kurta, kameez, shirt, choli, or 1-piece pishwas/frock)
    2: bottom (shalwar, trousers, sharara, gharara, or lehenga skirt)
    3: dupatta (draped scarf/shawl/organza dupatta)
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

CLASS_NAMES: Tuple[str, ...] = ("other", "kurta_kameez", "bottom", "dupatta")
CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def preserve_dupatta_over_mask(
    target_mask: np.ndarray,
    seg_map: np.ndarray,
) -> np.ndarray:
    """Subtract dupatta pixels (class 3) from an editable garment mask.

    Prevents the VTON diffusion model from erasing or corrupting draped
    dupattas during upper-body or full-outfit swaps.
    """
    if target_mask.shape != seg_map.shape:
        raise ValueError(
            f"Shape mismatch: target_mask {target_mask.shape} != seg_map {seg_map.shape}"
        )
    dupatta_pixels = seg_map == CLASS_TO_ID["dupatta"]
    out = target_mask.astype(np.uint8).copy()
    out[dupatta_pixels] = 0
    return out


def make_flared_lehenga_mask(
    seg_map: np.ndarray,
    horizontal_expand_ratio: float = 0.18,
    preserve_dupatta: bool = True,
) -> np.ndarray:
    """Construct an A-line horizontally dilated full-outfit mask for flared lehengas.

    Standard tight pants/shalwar masks squeeze wide bridal lehengas into straight
    columns. This function progressively widens the lower-body region (below the
    upper-body waist boundary) into a smooth A-line silhouette while protecting
    face/hands (class 0 upper region) and optionally preserving the dupatta.
    """
    if seg_map.ndim != 2:
        raise ValueError(f"Expected 2D segmentation map, got shape {seg_map.shape}")

    h, w = seg_map.shape
    upper = (seg_map == CLASS_TO_ID["kurta_kameez"]).astype(np.uint8)
    lower = (seg_map == CLASS_TO_ID["bottom"]).astype(np.uint8)
    combined = np.clip(upper + lower, 0, 1).astype(np.uint8)

    lower_rows = np.where(lower.any(axis=1))[0]
    if len(lower_rows) > 0:
        y_start = int(lower_rows[0])
        y_end = int(lower_rows[-1])
        span = max(1, y_end - y_start)
        max_pad = max(1, int(w * horizontal_expand_ratio))

        flared = combined.copy()
        for y in range(y_start, y_end + 1):
            xs = np.where(combined[y] > 0)[0]
            if len(xs) == 0:
                continue
            progress = (y - y_start) / span  # 0 at waist -> 1 at hem
            pad = int(round(max_pad * progress))
            left = max(0, int(xs[0]) - pad)
            right = min(w - 1, int(xs[-1]) + pad)
            flared[y, left : right + 1] = 1
        combined = flared

    if preserve_dupatta:
        combined = preserve_dupatta_over_mask(combined, seg_map)

    return combined


def build_tryon_mask(
    seg_map: np.ndarray,
    mode: str = "upper_body",
    preserve_dupatta: bool = True,
) -> np.ndarray:
    """Build a binary uint8 try-on mask from a 4-class segmentation map.

    Supported modes:
        - "upper_body": kurta_kameez only (class 1)
        - "lower_body": bottom only (class 2)
        - "full_outfit": kurta_kameez + bottom (classes 1 and 2)
        - "flared_lehenga": A-line dilated full-outfit mask for wedding formals
    """
    if mode == "upper_body":
        mask = (seg_map == CLASS_TO_ID["kurta_kameez"]).astype(np.uint8)
        return preserve_dupatta_over_mask(mask, seg_map) if preserve_dupatta else mask
    if mode == "lower_body":
        mask = (seg_map == CLASS_TO_ID["bottom"]).astype(np.uint8)
        return preserve_dupatta_over_mask(mask, seg_map) if preserve_dupatta else mask
    if mode == "full_outfit":
        mask = np.isin(
            seg_map, [CLASS_TO_ID["kurta_kameez"], CLASS_TO_ID["bottom"]]
        ).astype(np.uint8)
        return preserve_dupatta_over_mask(mask, seg_map) if preserve_dupatta else mask
    if mode == "flared_lehenga":
        return make_flared_lehenga_mask(seg_map, preserve_dupatta=preserve_dupatta)

    raise ValueError(
        f"Unsupported mode '{mode}'. Choose from upper_body, lower_body, full_outfit, flared_lehenga."
    )
