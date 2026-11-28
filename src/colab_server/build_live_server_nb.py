#!/usr/bin/env python3
"""Generate FashionFit_PK_Live_GPU_Server.ipynb directly from colab_server/v4_worker_core.py.

Enforces Phase 3 reproducibility requirements:
1. Fails clearly if no Colab CUDA GPU is allocated.
2. Verifies FashionFit_Advisor_V4_Code.zip against its trusted SHA-256 before extraction
   and rejects unsafe ZipSlip archive paths.
3. Verifies 09_MODELS/universal_b2/best_checkpoint against the exact expected manifest
   from universal_test/protocol.json (distinguishing matched vs recorded).
4. Verifies local worker /health BEFORE launching and announcing the temporary tunnel.
5. Includes an executable smoke-test cell and machine-readable JSON evidence export.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE_PY = HERE / "v4_worker_core.py"
OUT_NB = HERE / "FashionFit_PK_Live_GPU_Server.ipynb"

# Trusted SHA-256 values verified from V4 Drive artifacts (never invented):
EXPECTED_CODE_ZIP_SHA256 = "84b862fca88cb0aed1656d2d674883db978742faaa5a818274e8d521ace01fb0"
EXPECTED_UNIVERSAL_MANIFEST = {
    "config.json": "4584bad889394a232e2603c1331dada5b50f60ef385e60f5d1f0d9d51033fcf8",
    "model.safetensors": "3d62ea9e1c2807706096e568f7b8e467a3059b8ccfbe1be6e18de14515075a69",
    "preprocessor_config.json": "5efb826f9a7f78c0316b914fe513a699a4410030110cede35f682055f22f0030",
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in text.strip().splitlines()],
    }


core_source = CORE_PY.read_text(encoding="utf-8")
core_sha256 = hashlib.sha256(core_source.encode("utf-8")).hexdigest()

nb = {
    "cells": [
        md(
            "# FashionFit-PK V4 — Reproducible Authenticated Colab GPU Worker\n\n"
            "**Interactive Research & Demo Worker (temporary tunnel — NOT 24/7 public hosting).**\n\n"
            f"- Embedded `v4_worker_core.py` SHA-256: `{core_sha256}`\n"
            f"- Expected `FashionFit_Advisor_V4_Code.zip` SHA-256: `{EXPECTED_CODE_ZIP_SHA256}`\n"
            "- Expected `universal_b2/best_checkpoint` verified against `universal_test/protocol.json` manifest.\n"
            "- Fails closed if CUDA GPU, code ZIP hash, or checkpoint manifest does not match."
        ),
        code(
            f"""
import hashlib, os, secrets, zipfile
from pathlib import Path
import torch
from google.colab import drive

if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU allocated. In Colab: Runtime -> Change runtime type -> T4/L4 GPU.')
print('Allocated GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)

drive.mount('/content/drive')
V4_ROOT = Path(os.environ.get('FF_V4_ROOT', '/content/drive/MyDrive/FashionFit_PK_Advisor_V4_2026-11-26'))
assert V4_ROOT.exists(), f'Missing V4 root directory: {{V4_ROOT}}'

CODE_ZIP = V4_ROOT / '07_CODE' / 'FashionFit_Advisor_V4_Code.zip'
EXPECTED_CODE_ZIP_SHA256 = '{EXPECTED_CODE_ZIP_SHA256}'
assert CODE_ZIP.is_file(), f'Missing {{CODE_ZIP}}'

h = hashlib.sha256(CODE_ZIP.read_bytes()).hexdigest()
if h != EXPECTED_CODE_ZIP_SHA256:
    raise ValueError(f'Code ZIP SHA-256 mismatch! Expected {{EXPECTED_CODE_ZIP_SHA256}}, got {{h}}')
print('Verified FashionFit_Advisor_V4_Code.zip SHA-256:', h)

CODE_DIR = Path('/content/fashionfit_v4_pkg').resolve()
CODE_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(CODE_ZIP, 'r') as zf:
    for member in zf.infolist():
        dest = (CODE_DIR / member.filename).resolve()
        if not str(dest).startswith(str(CODE_DIR)):
            raise ValueError(f'Unsafe archive path rejected (ZipSlip guard): {{member.filename}}')
    zf.extractall(CODE_DIR)
print('Safely extracted V4 code package to:', CODE_DIR)

os.environ['FF_V4_ROOT'] = str(V4_ROOT)
os.environ['FF_SERVER_TOKEN'] = os.environ.get('FF_SERVER_TOKEN') or secrets.token_urlsafe(24)
print('Session Bearer Token generated (keep private — never paste into public repos):', os.environ['FF_SERVER_TOKEN'])

!pip install -q "fastapi>=0.115,<0.120" "uvicorn>=0.30,<0.35" "python-multipart>=0.0.9" "diffusers>=0.30,<0.33" "transformers>=4.44,<4.49" "accelerate>=0.33" huggingface_hub onnxruntime-gpu einops
!pip install -q "git+https://github.com/fashn-AI/fashn-human-parser.git"
!pip install -q --no-deps "git+https://github.com/fashn-AI/fashn-vton-1.5.git@7c0f10af3f91ad4048fe9729c470a13ef905d25a"
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
"""
        ),
        code("%%writefile /content/v4_worker_core.py\n" + core_source),
        code(
            f"""
%%writefile /content/v4_gpu_api.py
import gc, json, os, sys
from pathlib import Path
import torch

sys.path.insert(0, '/content')
sys.path.insert(0, '/content/fashionfit_v4_pkg')

from v4_worker_core import create_worker_app, verify_checkpoint_dir
from fashionfit_v4.protocol import build_mask
from fashionfit_v4.segmentation import Parser, BASE, overlay
from fashionfit_v4.catvton import CatVtonAdapter
from fashionfit_v4.fashn import FashnVtonAdapter

EXPECTED_UNIVERSAL_MANIFEST = {json.dumps(EXPECTED_UNIVERSAL_MANIFEST, indent=4)}

V4_ROOT = Path(os.environ['FF_V4_ROOT'])
TOKEN = os.environ.get('FF_SERVER_TOKEN', '').strip()
DEFAULT_PARSER = os.environ.get('FF_DEFAULT_PARSER', 'universal')
UNIVERSAL_CKPT = V4_ROOT / '09_MODELS' / 'universal_b2' / 'best_checkpoint'
PATHS = {{'baseline': BASE, 'universal': UNIVERSAL_CKPT}}

provenance = (
    verify_checkpoint_dir(UNIVERSAL_CKPT, expected_manifest=EXPECTED_UNIVERSAL_MANIFEST)
    if DEFAULT_PARSER != 'baseline'
    else {{'files_verified': True, 'verification_status': 'pretrained_baseline', 'files': {{}}}}
)

def _cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = create_worker_app(
    token=TOKEN,
    default_parser=DEFAULT_PARSER,
    checkpoint_provenance=provenance,
    gpu_available_fn=torch.cuda.is_available,
    gpu_name_fn=lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable',
    parser_factory=lambda kind: Parser(PATHS[kind], kind),
    vton_factory=lambda m: CatVtonAdapter() if m == 'catvton' else FashnVtonAdapter(),
    mask_builder=build_mask,
    overlay_builder=overlay,
    cleanup_gpu_fn=_cleanup_gpu,
    max_queue=2,
)
"""
        ),
        code(
            """
import json, os, re, subprocess, time, urllib.request

uvicorn_proc = subprocess.Popen(
    ['uvicorn', 'v4_gpu_api:app', '--host', '127.0.0.1', '--port', '8000', '--app-dir', '/content'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)

# Verify local worker /health BEFORE launching public tunnel
ready_health = None
for _ in range(30):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2) as r:
            ready_health = json.loads(r.read().decode())
            break
    except Exception:
        time.sleep(1)

if not ready_health or not ready_health.get('files_verified'):
    raise RuntimeError(f'Worker failed startup health check before tunnel: {ready_health}')

print('Local Worker Health Verified:', json.dumps(ready_health, indent=2))

tunnel = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
tunnel_url = None
for line in tunnel.stdout:
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', line)
    if m:
        tunnel_url = m.group(0)
        break

if not tunnel_url:
    raise RuntimeError('Cloudflare tunnel failed to emit a public URL. Re-run this cell.')

print('=' * 72)
print('LIVE V4 COLAB GPU URL:', tunnel_url)
print('BEARER ACCESS TOKEN:  ', os.environ['FF_SERVER_TOKEN'])
print('=' * 72)
"""
        ),
        code(
            """
# Phase 4 Smoke Test & Machine-Readable Evidence Export (run after tunnel is live)
import json, os, urllib.request
from pathlib import Path

headers = {'Authorization': 'Bearer ' + os.environ['FF_SERVER_TOKEN']}
req_auth = urllib.request.Request('http://127.0.0.1:8000/auth/verify', headers=headers)
auth_resp = json.loads(urllib.request.urlopen(req_auth, timeout=5).read().decode())
health_resp = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read().decode())

evidence = {
    'worker_core_sha256': '"""
            + core_sha256
            + """',
    'code_zip_sha256': '"""
            + EXPECTED_CODE_ZIP_SHA256
            + """',
    'auth_verify': auth_resp,
    'health_before_inference': health_resp,
}
out_path = Path(os.environ['FF_V4_ROOT']) / '10_RUNS' / 'live_worker_smoke_evidence.json'
out_path.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print('Saved live worker startup evidence to:', out_path)
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
print(f"Wrote {OUT_NB} (worker_core_sha256={core_sha256})")
