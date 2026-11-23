"""Profiling utility (MIT Missing Semester: Debugging & Profiling).

Benchmarks latency and peak memory of 4-class dupatta-preserving masking and
A-line flared wedding lehenga dilation at standard 512x384 VTON resolution.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.mask_utils import CLASS_TO_ID, build_tryon_mask


def run_benchmark(iterations: int = 50) -> None:
    seg = np.zeros((512, 384), dtype=np.uint8)
    seg[80:280, 110:274] = CLASS_TO_ID["kurta_kameez"]
    seg[280:490, 120:264] = CLASS_TO_ID["bottom"]
    seg[90:360, 220:290] = CLASS_TO_ID["dupatta"]

    print(f"Profiling mask pipeline on 512x384 map ({iterations} iterations each):")
    for mode in ("upper_body", "full_outfit", "flared_lehenga"):
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = build_tryon_mask(seg, mode=mode, preserve_dupatta=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iterations
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            f"  - {mode:<15} | avg: {elapsed_ms:6.2f} ms/img | peak mem: {peak_bytes / 1024:7.1f} KB"
        )


if __name__ == "__main__":
    run_benchmark()
