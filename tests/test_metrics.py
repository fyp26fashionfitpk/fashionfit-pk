"""Unit tests for segmentation IoU metrics and zero-leakage SKU split checks."""

import unittest
import numpy as np

from lib.metrics import compute_per_class_iou, validate_zero_leakage_sku_split


class TestMetrics(unittest.TestCase):
    def test_perfect_iou_is_100(self) -> None:
        gt = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        scores = compute_per_class_iou(gt, gt)
        self.assertEqual(scores["other"], 100.0)
        self.assertEqual(scores["kurta_kameez"], 100.0)
        self.assertEqual(scores["bottom"], 100.0)
        self.assertEqual(scores["dupatta"], 100.0)
        self.assertEqual(scores["mIoU"], 100.0)

    def test_zero_leakage_disjoint_skus(self) -> None:
        overlap = validate_zero_leakage_sku_split(
            ["GEN-001", "ETH-002", "LIM-003"],
            ["GEN-004", "PER-005"],
        )
        self.assertEqual(overlap, set())

    def test_leakage_detected_on_shared_sku(self) -> None:
        overlap = validate_zero_leakage_sku_split(
            ["GEN-001", "ETH-002"],
            ["ETH-002", "LIM-009"],
        )
        self.assertEqual(overlap, {"ETH-002"})


if __name__ == "__main__":
    unittest.main()
