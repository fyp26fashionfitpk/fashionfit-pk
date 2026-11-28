#!/usr/bin/env python3
"""Generate 03D_FASHN_SegmentationFree_Control_v1.ipynb embedding fashn_segfree_runner.py.

Distinct run ID (03D) to avoid any confusion with earlier 03C artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER_PY = HERE / "fashn_segfree_runner.py"
OUT_NB = HERE / "03D_FASHN_SegmentationFree_Control_v1.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + "\n" for l in text.strip().splitlines()]}


runner_src = RUNNER_PY.read_text(encoding="utf-8")

nb = {
    "cells": [
        md(
            "# 03D — Controlled Quality Experiment: FASHN VTON 1.5 `segmentation_free=True` vs `segmentation_free=False`\n\n"
            "- Distinct experiment ID (`03D_FASHN_SegmentationFree_Control_v1`) to avoid confusion with earlier `03C` runs.\n"
            "- Uses pinned official `fashn-AI/fashn-vton-1.5@7c0f10af3f91ad4048fe9729c470a13ef905d25a`.\n"
            "- Uses FASHN's internal parser when `segmentation_free=False` (NOT V4 SegFormer).\n"
            "- Freezes `protocol.json`, writes atomic per-case sidecars, resumes only hash-verified outputs, flags corrupt outputs, and generates paired comparison galleries + a separate review packet."
        ),
        code("%%writefile /content/fashn_segfree_runner.py\n" + runner_src),
        code(
            """
from google.colab import drive
import csv, os, shutil, subprocess, sys, time
from pathlib import Path
import torch

assert torch.cuda.is_available(), 'Colab CUDA GPU required.'
drive.mount('/content/drive')
V4_ROOT = Path(os.environ.get('FF_V4_ROOT', '/content/drive/MyDrive/FashionFit_PK_Advisor_V4_2026-11-26'))
SMOKE_ONLY = True  # Inspect 4-case smoke galleries first, then set False to resume full 20-case set
SMOKE_CASES = ['C001', 'C005', 'C009', 'C013']

subprocess.run(['pip', 'install', '-q', 'diffusers>=0.30,<0.33', 'transformers>=4.44,<4.49', 'accelerate>=0.33', 'huggingface_hub', 'onnxruntime-gpu', 'einops'], check=True)
subprocess.run(['pip', 'install', '-q', 'git+https://github.com/fashn-AI/fashn-human-parser.git'], check=True)
subprocess.run(['pip', 'install', '-q', '--no-deps', 'git+https://github.com/fashn-AI/fashn-vton-1.5.git@7c0f10af3f91ad4048fe9729c470a13ef905d25a'], check=True)

from fashn_vton import TryOnPipeline
from huggingface_hub import hf_hub_download, snapshot_download
sys.path.insert(0, '/content')
from fashn_segfree_runner import run_fashn_segfree_control

HF_REV = '7720683168567eb5a2a4c67f15116c6e29c83ded'
weights_dir = snapshot_download('fashn-ai/fashn-vton-1.5', revision=HF_REV)
dw = Path(weights_dir) / 'dwpose'
dw.mkdir(parents=True, exist_ok=True)
for fn in ('yolox_l.onnx', 'dw-ll_ucoco_384.onnx'):
    t = dw / fn
    if not t.exists():
        shutil.copy2(hf_hub_download('yzd-v/DWPose', fn, revision='1a7144101628d69ee7a3768d1ee3a094070dc388'), t)

pipe = TryOnPipeline(weights_dir=weights_dir)

with (V4_ROOT / '05_SPLITS' / 'cases.csv').open() as f:
    cases = list(csv.DictReader(f))
with (V4_ROOT / '03_METADATA' / 'inventory.csv').open() as f:
    inv = {r['image_id']: r for r in csv.DictReader(f)}

def _infer(person, donor, category, seg_free, seed):
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    res = pipe(
        person_image=person, garment_image=donor, category=category,
        garment_photo_type='model', segmentation_free=seg_free,
        num_timesteps=30, guidance_scale=1.5, seed=seed,
    )
    torch.cuda.synchronize()
    img = res.images[0] if hasattr(res, 'images') else res
    return {
        'image': img,
        'seconds': round(time.perf_counter() - t0, 2),
        'peak_vram_mb': round(torch.cuda.max_memory_allocated() / (1024**2), 1),
    }

out_dir = V4_ROOT / '10_RUNS' / ('03D_fashn_segfree_smoke_v1' if SMOKE_ONLY else '03D_fashn_segfree_full20_v1')
summary = run_fashn_segfree_control(
    cases=cases, inventory=inv, out_dir=out_dir, infer_fn=_infer,
    selected_case_ids=SMOKE_CASES if SMOKE_ONLY else None,
)
print('03D SUMMARY:', summary)
"""
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_NB.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Wrote", OUT_NB)
