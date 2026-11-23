"""Unit tests for 4-class Pakistani garment masking and flared lehenga dilation."""

import unittest
import numpy as np

from lib.mask_utils import (
    CLASS_TO_ID,
    build_tryon_mask,
    make_flared_lehenga_mask,
    preserve_dupatta_over_mask,
)


class TestMaskUtils(unittest.TestCase):
    def _sample_seg_map(self) -> np.ndarray:
        seg = np.zeros((100, 80), dtype=np.uint8)
        seg[20:55, 25:55] = CLASS_TO_ID["kurta_kameez"]
        seg[55:95, 28:52] = CLASS_TO_ID["bottom"]
        seg[22:70, 48:62] = CLASS_TO_ID["dupatta"]
        return seg

    def test_dupatta_preserved_during_upper_swap(self) -> None:
        seg = self._sample_seg_map()
        mask = build_tryon_mask(seg, mode="upper_body", preserve_dupatta=True)
        dupatta_overlap = np.logical_and(mask == 1, seg == CLASS_TO_ID["dupatta"]).sum()
        self.assertEqual(int(dupatta_overlap), 0)
        self.assertGreater(int(mask.sum()), 0)

    def test_flared_lehenga_expands_lower_hem(self) -> None:
        seg = self._sample_seg_map()
        tight = build_tryon_mask(seg, mode="full_outfit", preserve_dupatta=False)
        flared = make_flared_lehenga_mask(
            seg, horizontal_expand_ratio=0.20, preserve_dupatta=False
        )
        # Flared mask must be strictly wider at the hem row (row 90)
        tight_hem_width = int(tight[90].sum())
        flared_hem_width = int(flared[90].sum())
        self.assertGreater(flared_hem_width, tight_hem_width)

    def test_flared_lehenga_still_protects_dupatta(self) -> None:
        seg = self._sample_seg_map()
        flared = build_tryon_mask(seg, mode="flared_lehenga", preserve_dupatta=True)
        dupatta_overlap = np.logical_and(
            flared == 1, seg == CLASS_TO_ID["dupatta"]
        ).sum()
        self.assertEqual(int(dupatta_overlap), 0)

    def test_invalid_mode_raises(self) -> None:
        seg = self._sample_seg_map()
        with self.assertRaises(ValueError):
            build_tryon_mask(seg, mode="unknown_mode")

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            preserve_dupatta_over_mask(np.zeros((10, 10)), np.zeros((12, 10)))


if __name__ == "__main__":
    unittest.main()
