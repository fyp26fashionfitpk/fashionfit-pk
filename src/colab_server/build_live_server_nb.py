#!/usr/bin/env python3
"""Generate FashionFit_PK_Live_GPU_Server.ipynb directly from colab_server/v4_worker_core.py.

Reproducibility & reliability guarantees:
1. Fails closed if no Colab CUDA GPU is allocated.
2. Verifies FashionFit_Advisor_V4_Code_v4_2.zip against its readback-verified SHA-256
   and extracts with Path.relative_to() containment (rejecting traversal & sibling-prefix paths).
3. Verifies 09_MODELS/universal_b2/best_checkpoint against the trusted universal_test/protocol.json manifest.
4. Redirects uvicorn and cloudflared subprocess logs to files (never undrained PIPEs) with explicit timeouts.
5. Verifies local worker /health BEFORE launching the temporary tunnel.
6. Provides a REAL executable inference smoke workflow (auth, /segment, catvton /vton, fashn /vton,
   concurrent /health during execution, output image decode check, and model-residency switch).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE_PY = HERE / "v4_worker_core.py"
OUT_NB = HERE / "FashionFit_PK_Live_GPU_Server.ipynb"

# Readback-verified SHA-256 of 07_CODE/FashionFit_Advisor_V4_Code_v4_2.zip (includes segmentation_free):
CODE_ZIP_NAME = "FashionFit_Advisor_V4_Code_v4_2.zip"
EXPECTED_CODE_ZIP_SHA256 = "106030a6c3c0663399b57ec436d7b80c5b19c8bfa30160b09958840dd087decd"
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
smoke_source=(HERE/'live_smoke_inputs.py').read_text(encoding='utf-8')

nb = {
    "cells": [
        md(
            "# FashionFit-PK V4.3 — Reproducible Authenticated Colab GPU Worker\n\n"
            "**Interactive Research & Demo Worker (temporary tunnel — NOT 24/7 public hosting).**\n\n"
            f"- Embedded `v4_worker_core.py` SHA-256: `{core_sha256}`\n"
            f"- Expected `{CODE_ZIP_NAME}` SHA-256: `{EXPECTED_CODE_ZIP_SHA256}`\n"
            "- Expected `universal_b2/best_checkpoint` verified against `universal_test/protocol.json` manifest.\n"
            "- Uses `Path.relative_to()` archive extraction guard and file-backed subprocess logs (no PIPE stalls)."
        ),
        code(
            f"""
import hashlib, os, secrets, subprocess, zipfile
from pathlib import Path
import torch
from google.colab import drive

if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU allocated. In Colab: Runtime -> Change runtime type -> T4/L4 GPU.')
print('Allocated GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)

drive.mount('/content/drive')
V4_ROOT = Path(os.environ.get('FF_V4_ROOT', '/content/drive/MyDrive/FashionFit_PK_Advisor_V4_2026-11-26'))
assert V4_ROOT.exists(), f'Missing V4 root directory: {{V4_ROOT}}'

CODE_ZIP = V4_ROOT / '07_CODE' / '{CODE_ZIP_NAME}'
EXPECTED_CODE_ZIP_SHA256 = '{EXPECTED_CODE_ZIP_SHA256}'
assert CODE_ZIP.is_file(), f'Missing {{CODE_ZIP}}'

h = hashlib.sha256(CODE_ZIP.read_bytes()).hexdigest()
if h != EXPECTED_CODE_ZIP_SHA256:
    raise ValueError(f'Code ZIP SHA-256 mismatch! Expected {{EXPECTED_CODE_ZIP_SHA256}}, got {{h}}')
print('Verified {CODE_ZIP_NAME} SHA-256:', h)

CODE_DIR = Path('/content/fashionfit_v4_pkg').resolve()
CODE_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(CODE_ZIP, 'r') as zf:
    for member in zf.infolist():
        dest = (CODE_DIR / member.filename).resolve()
        dest.relative_to(CODE_DIR)  # Raises ValueError on traversal OR sibling-prefix attack
    zf.extractall(CODE_DIR)
print('Safely extracted V4.2 code package to:', CODE_DIR)

os.environ['FF_V4_ROOT'] = str(V4_ROOT)
os.environ['FF_SERVER_TOKEN'] = os.environ.get('FF_SERVER_TOKEN') or secrets.token_urlsafe(24)
print('Session Bearer Token (keep private):', os.environ['FF_SERVER_TOKEN'])

subprocess.run([
    'pip', 'install', '-q',
    'fastapi>=0.115,<0.120', 'uvicorn>=0.30,<0.35', 'python-multipart>=0.0.9',
    'diffusers>=0.30,<0.33', 'transformers>=4.44,<4.49', 'accelerate>=0.33',
    'huggingface_hub', 'onnxruntime-gpu', 'einops', 'requests'
], check=True)
subprocess.run(['pip', 'install', '-q', 'git+https://github.com/fashn-AI/fashn-human-parser.git'], check=True)
subprocess.run(['pip', 'install', '-q', '--no-deps', 'git+https://github.com/fashn-AI/fashn-vton-1.5.git@7c0f10af3f91ad4048fe9729c470a13ef905d25a'], check=True)
subprocess.run(['wget', '-q', 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', '-O', '/usr/local/bin/cloudflared'], check=True)
subprocess.run(['chmod', '+x', '/usr/local/bin/cloudflared'], check=True)

# Verify adapter imports & signatures BEFORE starting server
import sys, inspect
sys.path.insert(0, str(CODE_DIR))
from fashionfit_v4.fashn import FashnVtonAdapter
from fashionfit_v4.catvton import CatVtonAdapter
from fashionfit_v4.segmentation import Parser
assert 'segmentation_free' in inspect.signature(FashnVtonAdapter.run_tryon).parameters, 'Missing segmentation_free in FashnVtonAdapter'
print('Verified imports & adapter signatures cleanly.')
"""
        ),
        code("%%writefile /content/v4_worker_core.py\n" + core_source),
        code("%%writefile /content/live_smoke_inputs.py\n" + smoke_source),
        code(
            f"""
%%writefile /content/v4_gpu_api.py
import gc, os, sys
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
    inference_timeout_s=1800.0,  # Includes cold model loading; slot stays owned after HTTP timeout.
)
"""
        ),
        code(
            """
import json, os, re, subprocess, time, urllib.request
from pathlib import Path

# Clean up any prior uvicorn/cloudflared instances before starting
subprocess.run(['pkill', '-f', 'uvicorn v4_gpu_api:app'], check=False)
subprocess.run(['pkill', '-f', 'cloudflared tunnel'], check=False)

uv_log = open('/content/uvicorn.log', 'w', encoding='utf-8')
uvicorn_proc = subprocess.Popen(
    ['uvicorn', 'v4_gpu_api:app', '--host', '127.0.0.1', '--port', '8000', '--app-dir', '/content'],
    stdout=uv_log, stderr=subprocess.STDOUT,
)

# Verify local worker /health BEFORE launching public tunnel (30s timeout)
ready_health = None
for _ in range(30):
    if uvicorn_proc.poll() is not None:
        uv_log.flush()
        raise RuntimeError('uvicorn exited early. Tail of /content/uvicorn.log:\\n' + Path('/content/uvicorn.log').read_text()[-1500:])
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2) as r:
            ready_health = json.loads(r.read().decode())
            break
    except Exception:
        time.sleep(1)

if not ready_health or not ready_health.get('files_verified'):
    raise RuntimeError(f'Worker failed startup health check before tunnel: {ready_health}')

print('Local Worker Health Verified:', json.dumps(ready_health, indent=2))

cf_log_path = Path('/content/cloudflared.log')
cf_log = open(cf_log_path, 'w', encoding='utf-8')
tunnel = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000'],
    stdout=cf_log, stderr=subprocess.STDOUT,
)

tunnel_url = None
deadline = time.time() + 30
while time.time() < deadline:
    cf_log.flush()
    txt = cf_log_path.read_text(encoding='utf-8', errors='ignore')
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', txt)
    if m:
        tunnel_url = m.group(0)
        break
    time.sleep(0.5)

if not tunnel_url:
    raise RuntimeError('Cloudflare tunnel timed out after 30s. Log:\\n' + cf_log_path.read_text()[-1500:])

print('=' * 72)
print('LIVE V4.3 COLAB GPU URL:', tunnel_url)
print('BEARER ACCESS TOKEN:    ', os.environ['FF_SERVER_TOKEN'])
print('=' * 72)
"""
        ),
        code(
            """
# Phase 4 REAL Executable GPU Inference Smoke Workflow (NOT health-only!)
import base64, csv, io, json, os, threading, time
from pathlib import Path
import requests
from PIL import Image
import sys
sys.path.insert(0, '/content')
from live_smoke_inputs import prepare_smoke_inputs

BASE_URL = 'http://127.0.0.1:8000'
TOKEN = os.environ['FF_SERVER_TOKEN']
AUTH = {'Authorization': f'Bearer {TOKEN}'}
V4_ROOT = Path(os.environ['FF_V4_ROOT'])
SMOKE_OUT = V4_ROOT / '10_RUNS' / 'live_worker_real_inference_smoke_v4_3'
SMOKE_OUT.mkdir(parents=True, exist_ok=True)

# 1. Verify invalid vs valid auth
r_bad = requests.get(f'{BASE_URL}/auth/verify', headers={'Authorization': 'Bearer wrong'}, timeout=5)
assert r_bad.status_code == 401, f'Expected 401 on wrong token, got {r_bad.status_code}'
r_ok = requests.get(f'{BASE_URL}/auth/verify', headers=AUTH, timeout=5)
assert r_ok.status_code == 200, f'Expected 200 on valid token, got {r_ok.status_code}'

# 2. Prepare a visible-fabric cutout for CatVTON and original model photo for FASHN.
inputs=prepare_smoke_inputs(V4_ROOT, SMOKE_OUT)
first_case=inputs['case']
person_path=Path(inputs['person_path'])
donor_path=Path(inputs['donor_model_photo'])
cat_reference_path=Path(inputs['catvton_reference'])

# 3. Real /segment call
with person_path.open('rb') as fp:
    r_seg = requests.post(f'{BASE_URL}/segment', headers=AUTH, data={'seg_variant': 'adapted'}, files={'image': ('person.png', fp, 'image/png')}, timeout=1860)
assert r_seg.status_code == 200, r_seg.text
seg_json = r_seg.json()

# 4. Real CatVTON /vton + concurrent /health responsiveness check while running
health_during_run = []
def _probe_health():
    time.sleep(2.0)
    t0 = time.perf_counter()
    hr = requests.get(f'{BASE_URL}/health', timeout=5)
    health_during_run.append({'status': hr.status_code, 'latency_ms': round((time.perf_counter()-t0)*1000, 1), 'body': hr.json()})

probe_t = threading.Thread(target=_probe_health)
probe_t.start()
with person_path.open('rb') as fp, cat_reference_path.open('rb') as fg:
    r_cat = requests.post(
        f'{BASE_URL}/vton', headers=AUTH,
        data={'model_id': 'catvton', 'category': 'adult_upper', 'tryon_mode': 'upper_body', 'garment_photo_type': 'flat-lay', 'person_age_group': 'adult', 'seed': '42'},
        files={'person_image': ('p.png', fp, 'image/png'), 'garment_image': ('g.png', fg, 'image/png')},
        timeout=1860,
    )
probe_t.join()
assert r_cat.status_code == 200, r_cat.text
cat_json = r_cat.json()

# Decode & save returned CatVTON image
cat_img = Image.open(io.BytesIO(base64.b64decode(cat_json['result_image'].split(',', 1)[1])))
cat_img.save(SMOKE_OUT / 'smoke_catvton.png')

# 5. Real FASHN /vton (verifies model-residency switch evicts catvton)
with person_path.open('rb') as fp, donor_path.open('rb') as fg:
    r_fashn = requests.post(
        f'{BASE_URL}/vton', headers=AUTH,
        data={'model_id': 'fashn_vton_1_5', 'category': 'adult_upper', 'tryon_mode': 'upper_body', 'garment_photo_type': 'model', 'person_age_group': 'adult', 'seed': '42'},
        files={'person_image': ('p.png', fp, 'image/png'), 'garment_image': ('g.png', fg, 'image/png')},
        timeout=1860,
    )
assert r_fashn.status_code == 200, r_fashn.text
fashn_json = r_fashn.json()
fashn_img = Image.open(io.BytesIO(base64.b64decode(fashn_json['result_image'].split(',', 1)[1])))
fashn_img.save(SMOKE_OUT / 'smoke_fashn.png')

health_after = requests.get(f'{BASE_URL}/health', timeout=5).json()
assert health_after['model_states']['catvton'] == 'not_loaded', 'Expected catvton evicted after switching to fashn'
assert health_after['model_states']['fashn_vton_1_5'] == 'ready'

evidence = {
    'worker_core_sha256': '"""
            + core_sha256
            + """',
    'code_zip_name': '"""
            + CODE_ZIP_NAME
            + """',
    'code_zip_sha256': '"""
            + EXPECTED_CODE_ZIP_SHA256
            + """',
    'case_id': first_case['case_id'],
    'input_provenance': inputs,
    'segment': {k: v for k, v in seg_json.items() if 'image' not in k},
    'catvton': {k: v for k, v in cat_json.items() if k != 'result_image'},
    'fashn': {k: v for k, v in fashn_json.items() if k != 'result_image'},
    'health_during_catvton': health_during_run,
    'health_after_model_switch': health_after,
}
assert health_during_run and all(p['status']==200 for p in health_during_run), 'Health responsiveness was not verified'
(SMOKE_OUT / 'real_inference_smoke_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print('REAL GPU INFERENCE SMOKE PASSED! Evidence saved to:', SMOKE_OUT)
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
print(f"Wrote {OUT_NB} (worker_core_sha256={core_sha256}, code_zip={CODE_ZIP_NAME})")
