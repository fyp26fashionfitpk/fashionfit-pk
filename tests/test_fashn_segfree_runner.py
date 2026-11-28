"""CPU software tests for 03D FASHN experiment runner & ZipSlip containment.

NOTE: These are software-integrity tests using disposable fixtures and test doubles —
NOT real neural model execution.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "research" / "v4_controlled_experiments"))
sys.path.insert(0, str(REPO_ROOT / "src" / "colab_server"))

import fashn_segfree_runner as runner
import v4_worker_core as worker_core


def _save_dummy(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path, format="PNG")


class TestFashnSegFreeRunnerSoftware(unittest.TestCase):
    def _make_fixtures(self, td: Path):
        p1 = td / "imgs" / "p1.png"
        d1 = td / "imgs" / "d1.png"
        _save_dummy(p1, (20, 40, 60))
        _save_dummy(d1, (180, 30, 50))
        inv = {
            "IMG_P1": {"image_id": "IMG_P1", "image_path": str(p1)},
            "IMG_D1": {"image_id": "IMG_D1", "image_path": str(d1)},
        }
        cases = [
            {
                "case_id": "C001",
                "scenario": "adult_to_adult",
                "source_image_id": "IMG_D1",
                "target_image_id": "IMG_P1",
                "mode": "upper_body",
            }
        ]
        return cases, inv

    def test_atomic_recording_partial_failure_resume_and_corrupt_flagging(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cases, inv = self._make_fixtures(td)
            out_dir = td / "run_03d"

            # Pass 1: seg_free=True succeeds, seg_free=False fails -> partial failure must preserve True metadata
            def _fail_second(person, donor, cat, seg_free, seed):
                if not seg_free:
                    raise RuntimeError("Simulated inference error on seg_free=False")
                return {"image": person.copy(), "seconds": 0.1, "peak_vram_mb": 100.0}

            s1 = runner.run_fashn_segfree_control(cases=cases, inventory=inv, out_dir=out_dir, infer_fn=_fail_second)
            self.assertEqual(s1["counts"]["succeeded"], 1)
            self.assertEqual(s1["counts"]["failed"], 1)
            self.assertTrue((out_dir / "outputs" / "C001__segfree_true.json").is_file())
            self.assertTrue((out_dir / "outputs" / "C001__segfree_false.json").is_file())

            # Pass 2: Resume -> seg_free=True must be resumed without calling infer_fn, seg_free=False runs & succeeds
            calls = []

            def _succeed_all(person, donor, cat, seg_free, seed):
                calls.append(seg_free)
                return {"image": donor.copy(), "seconds": 0.2, "peak_vram_mb": 110.0}

            s2 = runner.run_fashn_segfree_control(cases=cases, inventory=inv, out_dir=out_dir, infer_fn=_succeed_all)
            self.assertEqual(calls, [False])  # seg_free=True was resumed!
            self.assertEqual(s2["counts"]["resumed"], 1)
            self.assertEqual(s2["counts"]["succeeded"], 2)
            self.assertTrue((out_dir / "galleries" / "C001_paired.png").is_file())
            self.assertTrue((out_dir / "review_packet_03D.csv").is_file())

            # Pass 3: Tamper with output image -> must flag corrupt rather than silently overwriting!
            (out_dir / "outputs" / "C001__segfree_true.png").write_bytes(b"corrupted")
            s3 = runner.run_fashn_segfree_control(cases=cases, inventory=inv, out_dir=out_dir, infer_fn=_succeed_all)
            self.assertEqual(s3["counts"]["corrupt_flagged"], 1)
            self.assertTrue((out_dir / "outputs" / "C001__segfree_true.CORRUPT_FLAG.json").is_file())

    def test_protocol_drift_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cases, inv = self._make_fixtures(td)
            out_dir = td / "run_03d"
            runner.run_fashn_segfree_control(
                cases=cases, inventory=inv, out_dir=out_dir, seed=42,
                infer_fn=lambda p, d, c, sf, s: {"image": p.copy(), "seconds": 0.1, "peak_vram_mb": 50.0},
            )
            with self.assertRaises(ValueError):
                runner.run_fashn_segfree_control(
                    cases=cases, inventory=inv, out_dir=out_dir, seed=99,
                    infer_fn=lambda p, d, c, sf, s: {"image": p.copy(), "seconds": 0.1, "peak_vram_mb": 50.0},
                )


class TestZipSlipContainment(unittest.TestCase):
    def test_safe_extract_rejects_traversal_and_sibling_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            dest = td / "pkg"
            for bad_name in ("../escape.txt", "../pkg_evil/sibling.txt"):
                zpath = td / "bad.zip"
                with zipfile.ZipFile(zpath, "w") as zf:
                    zf.writestr(bad_name, "evil")
                with self.assertRaises(ValueError):
                    worker_core.safe_extract_zip(zpath, dest)


if __name__ == "__main__":
    unittest.main()
