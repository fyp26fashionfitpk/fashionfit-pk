"""FashionFit-PK Local-Only Development FastAPI Proxy (SSRF-Hardened).

DEPLOYMENT SCOPE: LOCAL DEVELOPMENT ONLY (bind 127.0.0.1). Do NOT expose publicly.
Security controls:
- Runtime upstream URL mutation (POST /api/config/colab) is DISABLED by default.
  To enable locally, operator must set FF_ALLOW_RUNTIME_UPSTREAM_CONFIG=true AND
  FF_ADMIN_TOKEN, and send X-Admin-Token header.
- Upstream URLs are strictly validated against FF_ALLOWED_UPSTREAM_HOSTS
  (default: *.trycloudflare.com), HTTPS-only, blocking private/internal/link-local
  IPs and refusing HTTP redirects so Bearer tokens are never leaked to arbitrary hosts.
- Uploaded images are validated (<= 10 MB, <= 12 MP, PIL decode/verify, EXIF transpose,
  PNG re-encode stripping metadata).
"""

from __future__ import annotations

import fnmatch
import io
import ipaddress
import json
import os
import secrets
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parent
DEMO_ASSETS_DIR = APP_ROOT / "demo_assets"
STATIC_DIR = APP_ROOT / "static"
DEMO_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("FF_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("FF_MAX_IMAGE_PIXELS", "12000000"))
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "FF_CORS_ORIGINS",
        "https://fashionfit-pk.vercel.app,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]

VALID_CATEGORIES = {"adult_upper", "girl_child", "cross_age", "wedding_lehenga"}
VALID_MODELS = {"catvton", "fashn_vton_1_5"}
VALID_TRYON_MODES = {"upper_body", "full_outfit", "flared_lehenga"}
VALID_PHOTO_TYPES = {"flat-lay", "model"}

app = FastAPI(
    title="FashionFit-PK Local Dev Proxy (SSRF-Hardened)",
    version="4.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
)

app.mount("/demo-assets", StaticFiles(directory=str(DEMO_ASSETS_DIR)), name="demo-assets")

STATE: dict[str, Any] = {
    "colab_gpu_url": os.environ.get("COLAB_GPU_URL", "").strip().rstrip("/"),
}


class ColabConfig(BaseModel):
    colab_gpu_url: str


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects so Bearer tokens are never forwarded to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, 502, "Upstream redirects are disallowed", headers, fp)


_SAFE_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def validate_upstream_url(raw_url: str) -> str:
    """Strictly validate upstream GPU worker URL against allowlist and block SSRF."""
    cleaned = (raw_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme != "https":
        raise HTTPException(status_code=400, detail="Upstream GPU worker URL must use https://")
    host = (parsed.hostname or "").lower()
    if not host:
        raise HTTPException(status_code=400, detail="Invalid upstream hostname.")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Embedded credentials in URL are forbidden.")

    # Reject raw IP addresses (especially private/loopback/link-local)
    try:
        ip_obj = ipaddress.ip_address(host)
        raise HTTPException(status_code=403, detail=f"Direct IP upstream '{ip_obj}' is disallowed.")
    except ValueError:
        pass

    allowed_patterns = [
        p.strip().lower()
        for p in os.environ.get("FF_ALLOWED_UPSTREAM_HOSTS", "*.trycloudflare.com").split(",")
        if p.strip()
    ]
    if not any(fnmatch.fnmatch(host, pat) for pat in allowed_patterns):
        raise HTTPException(
            status_code=403,
            detail=f"Upstream host '{host}' is not in FF_ALLOWED_UPSTREAM_HOSTS ({', '.join(allowed_patterns)}).",
        )

    # Guard against DNS resolving to private/internal/loopback addresses
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        for info in infos:
            ip_addr = ipaddress.ip_address(info[4][0])
            if ip_addr.is_private or ip_addr.is_loopback or ip_addr.is_link_local or ip_addr.is_reserved:
                raise HTTPException(status_code=403, detail="Upstream host resolves to a restricted private IP.")
    except HTTPException:
        raise
    except OSError:
        # Host may be unreachable/offline; allow validation error to be surfaced on actual connect or reject if desired
        pass

    return cleaned


async def _validate_and_strip_image(upload: UploadFile) -> tuple[bytes, str]:
    raw = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.")
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    try:
        im = Image.open(io.BytesIO(raw))
        im.verify()
        im = Image.open(io.BytesIO(raw))
        if im.width * im.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image exceeds 12 megapixel limit; resize before uploading.")
        im = ImageOps.exif_transpose(im).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload a valid JPG, PNG, or WebP image.") from exc


def _forward_multipart(
    base_url: str,
    path: str,
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
    authorization: str = "",
) -> bytes:
    validated_base = validate_upstream_url(base_url)
    url = f"{validated_base}{path}"
    boundary = "----FashionFitPKV4Boundary7MA4YWxkTrZu0gW"
    body = bytearray()

    for name, val in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{val}\r\n".encode())

    for field_name, filename, content, content_type in files:
        safe_name = Path(filename).name or "upload.png"
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_name}"\r\n'.encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode())

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if authorization:
        headers["Authorization"] = authorization

    req = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")
    try:
        with _SAFE_OPENER.open(req, timeout=600) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=exc.code, detail=f"GPU worker error: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach GPU worker: {exc}") from exc


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    colab_url = STATE["colab_gpu_url"]
    if not colab_url:
        return {
            "schema_version": "v4.1",
            "status": "ok",
            "mode": "local_proxy_awaiting_gpu_worker",
            "process_alive": True,
            "gpu_accessible": False,
            "files_verified": False,
            "model_loaded": False,
            "queue_accepting": False,
            "colab_gpu_url": None,
            "supported_categories": [],
        }

    try:
        validated = validate_upstream_url(colab_url)
        req = urllib.request.Request(f"{validated}/health", method="GET")
        with _SAFE_OPENER.open(req, timeout=5) as resp:
            remote = json.loads(resp.read().decode("utf-8"))
        return {
            "schema_version": "v4.1",
            "status": "ok",
            "mode": "colab_gpu_live",
            "process_alive": True,
            "gpu_accessible": bool(remote.get("gpu_accessible", False)),
            "files_verified": bool(remote.get("files_verified", False)),
            "model_loaded": bool(remote.get("model_loaded", False)),
            "queue_accepting": bool(remote.get("queue_accepting", False)),
            "colab_gpu_url": validated,
            "worker": remote,
        }
    except Exception as exc:
        return {
            "schema_version": "v4.1",
            "status": "degraded",
            "mode": "gpu_worker_unreachable",
            "process_alive": True,
            "gpu_accessible": False,
            "files_verified": False,
            "model_loaded": False,
            "queue_accepting": False,
            "colab_gpu_url": colab_url,
            "error": str(exc),
        }


@app.post("/api/config/colab")
def set_colab_url(
    cfg: ColabConfig,
    x_admin_token: str = Header(default="", alias="X-Admin-Token"),
) -> dict[str, Any]:
    allow_runtime = os.environ.get("FF_ALLOW_RUNTIME_UPSTREAM_CONFIG", "").strip().lower() == "true"
    if not allow_runtime:
        raise HTTPException(
            status_code=403,
            detail="Runtime upstream URL configuration is disabled. Set COLAB_GPU_URL in server environment.",
        )
    admin_token = os.environ.get("FF_ADMIN_TOKEN", "").strip()
    if not admin_token or not secrets.compare_digest(x_admin_token or "", admin_token):
        raise HTTPException(status_code=401, detail="Valid X-Admin-Token header required.")

    validated = validate_upstream_url(cfg.colab_gpu_url)
    STATE["colab_gpu_url"] = validated
    return {"status": "updated", "colab_gpu_url": validated or None}


@app.post("/api/vton")
async def run_vton(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    model_id: str = Form("catvton"),
    category: str = Form("adult_upper"),
    tryon_mode: str = Form("upper_body"),
    garment_photo_type: str = Form("flat-lay"),
    person_age_group: str = Form("adult"),
    seed: int = Form(42),
    authorization: str = Header(default=""),
) -> Any:
    colab_url = STATE["colab_gpu_url"]
    if not colab_url:
        raise HTTPException(
            status_code=503,
            detail="No authenticated V4 GPU worker configured. Set COLAB_GPU_URL in the local proxy environment.",
        )
    if model_id not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model_id '{model_id}'.")
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unsupported category '{category}'.")
    if tryon_mode not in VALID_TRYON_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported tryon_mode '{tryon_mode}'.")
    if garment_photo_type not in VALID_PHOTO_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported garment_photo_type '{garment_photo_type}'.")
    if category == "wedding_lehenga" and tryon_mode != "flared_lehenga":
        raise HTTPException(status_code=400, detail="wedding_lehenga requires tryon_mode='flared_lehenga'.")
    if category == "adult_upper" and (tryon_mode != "upper_body" or person_age_group != "adult"):
        raise HTTPException(status_code=400, detail="adult_upper requires tryon_mode='upper_body' and person_age_group='adult'.")
    if category == "girl_child" and (person_age_group != "child" or tryon_mode not in {"upper_body", "full_outfit"}):
        raise HTTPException(status_code=400, detail="girl_child requires person_age_group='child' and upper_body/full_outfit.")
    if category == "cross_age" and tryon_mode == "flared_lehenga":
        raise HTTPException(status_code=400, detail="cross_age with flared_lehenga is unsupported.")
    if model_id == "catvton" and garment_photo_type == "model":
        raise HTTPException(status_code=400, detail="CatVTON requires garment_photo_type='flat-lay'; use fashn_vton_1_5 for model-worn references.")

    p_bytes, p_mime = await _validate_and_strip_image(person_image)
    g_bytes, g_mime = await _validate_and_strip_image(garment_image)

    raw = _forward_multipart(
        colab_url,
        "/vton",
        fields={
            "model_id": model_id,
            "category": category,
            "tryon_mode": tryon_mode,
            "garment_photo_type": garment_photo_type,
            "person_age_group": person_age_group,
            "seed": str(seed),
        },
        files=[
            ("person_image", "person.png", p_bytes, p_mime),
            ("garment_image", "garment.png", g_bytes, g_mime),
        ],
        authorization=authorization,
    )
    return json.loads(raw.decode("utf-8"))


@app.post("/api/segment")
async def run_segmentation(
    image: UploadFile = File(...),
    seg_variant: str = Form("adapted"),
    authorization: str = Header(default=""),
) -> Any:
    colab_url = STATE["colab_gpu_url"]
    if not colab_url:
        raise HTTPException(
            status_code=503,
            detail="No authenticated V4 GPU worker configured.",
        )
    img_bytes, img_mime = await _validate_and_strip_image(image)
    raw = _forward_multipart(
        colab_url,
        "/segment",
        fields={"seg_variant": seg_variant},
        files=[("image", "outfit.png", img_bytes, img_mime)],
        authorization=authorization,
    )
    return json.loads(raw.decode("utf-8"))
