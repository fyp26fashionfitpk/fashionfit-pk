"""FashionFit-PK V4 GPU Worker Core (Explicit Job Ownership, Truthful OOM & Capability Contract).

Design guarantees:
- Explicit execution ownership: capacity slots are released ONLY when the underlying
  worker-thread task finishes, OR if future.cancel() succeeds BEFORE execution starts.
  An HTTP timeout/disconnect NEVER falsely frees a running GPU slot.
- Truthful per-model lifecycle & single-large-VTON-model residency:
  Evicting a model (on switch or OOM) happens strictly inside the single worker thread
  and resets model_states[k] = 'not_loaded' so /health never reports discarded models ready.
- Combination-level capability validation prevents silent downgrades (e.g. wedding_lehenga
  refuses upper_body; catvton refuses model-worn photo type).
"""

from __future__ import annotations

import asyncio
import base64
import gc
import hashlib
import io
import secrets
import threading
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> list[str]:
    """Safely extract a ZIP using Path.relative_to() containment (blocks traversal & sibling-prefix attacks)."""
    root = dest_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (root / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Unsafe archive path rejected: {member.filename}") from exc
        for member in zf.infolist():
            zf.extract(member, root)
            extracted.append(member.filename)
    return extracted

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_PIXELS = 12_000_000
VALID_MODELS = {"catvton", "fashn_vton_1_5"}
VALID_CATEGORIES = {"adult_upper", "girl_child", "cross_age", "wedding_lehenga"}
VALID_TRYON_MODES = {"upper_body", "full_outfit", "flared_lehenga"}
VALID_PHOTO_TYPES = {"flat-lay", "model"}
VALID_AGE_GROUPS = {"adult", "child"}


class _WorkerOOMError(RuntimeError):
    """Raised internally after worker thread safely evicts caches and resets model_states."""


def validate_capability_combination(
    *,
    model_id: str,
    category: str,
    tryon_mode: str,
    garment_photo_type: str,
    person_age_group: str,
) -> None:
    """Enforce strict combination-level capability contract (zero silent downgrades)."""
    if model_id not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model_id '{model_id}'.")
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unsupported category '{category}'.")
    if tryon_mode not in VALID_TRYON_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported tryon_mode '{tryon_mode}'.")
    if garment_photo_type not in VALID_PHOTO_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported garment_photo_type '{garment_photo_type}'.")
    if person_age_group not in VALID_AGE_GROUPS:
        raise HTTPException(status_code=400, detail=f"Unsupported person_age_group '{person_age_group}'.")

    if category == "wedding_lehenga" and tryon_mode != "flared_lehenga":
        raise HTTPException(
            status_code=400,
            detail="wedding_lehenga requires tryon_mode='flared_lehenga'; refusing silent downgrade to upper_body.",
        )
    if category == "adult_upper":
        if tryon_mode != "upper_body":
            raise HTTPException(status_code=400, detail="adult_upper requires tryon_mode='upper_body'.")
        if person_age_group != "adult":
            raise HTTPException(
                status_code=400,
                detail="adult_upper requires person_age_group='adult'; use category='cross_age' for child target.",
            )
    if category == "girl_child":
        if person_age_group != "child":
            raise HTTPException(
                status_code=400,
                detail="girl_child requires person_age_group='child'; use category='cross_age' for adult target.",
            )
        if tryon_mode not in {"upper_body", "full_outfit"}:
            raise HTTPException(status_code=400, detail="girl_child supports tryon_mode 'upper_body' or 'full_outfit'.")
    if category == "cross_age" and tryon_mode == "flared_lehenga":
        raise HTTPException(
            status_code=400,
            detail="cross_age with flared_lehenga is unsupported; use wedding_lehenga or cross_age upper_body/full_outfit.",
        )
    if model_id == "catvton" and garment_photo_type == "model":
        raise HTTPException(
            status_code=400,
            detail="CatVTON requires garment_photo_type='flat-lay' (isolated garment reference). Use fashn_vton_1_5 for model-worn references.",
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_dir(
    ckpt_dir: Path,
    expected_manifest: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not ckpt_dir.is_dir():
        return {
            "files_verified": False,
            "verification_status": "missing_directory",
            "files": {},
            "missing_required": ["config.json", "weights(.safetensors|.bin)", "preprocessor_config.json"],
        }

    missing = []
    if not (ckpt_dir / "config.json").is_file():
        missing.append("config.json")
    if not (ckpt_dir / "preprocessor_config.json").is_file():
        missing.append("preprocessor_config.json")
    has_weights = (ckpt_dir / "model.safetensors").is_file() or (ckpt_dir / "pytorch_model.bin").is_file()
    if not has_weights:
        missing.append("model.safetensors or pytorch_model.bin")

    files = {
        p.name: sha256_file(p)
        for p in sorted(ckpt_dir.glob("*"))
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}
    }

    if missing:
        return {
            "files_verified": False,
            "verification_status": "missing_required_files",
            "files": files,
            "missing_required": missing,
        }

    if expected_manifest:
        mismatches = [k for k, v in expected_manifest.items() if files.get(k) != v]
        if mismatches:
            return {
                "files_verified": False,
                "verification_status": "manifest_hash_mismatch",
                "files": files,
                "mismatches": mismatches,
            }
        return {
            "files_verified": True,
            "verification_status": "verified_against_manifest",
            "files": files,
        }

    return {
        "files_verified": True,
        "verification_status": "hashes_recorded_no_manifest",
        "files": files,
    }


def encode_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


async def decode_clean_upload(
    upload: UploadFile,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Image.Image:
    data = await upload.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds byte limit ({max_bytes} bytes).")
    try:
        im = Image.open(io.BytesIO(data))
        if im.format not in {"JPEG", "PNG", "WEBP"}:
            raise HTTPException(status_code=400, detail=f"Unsupported image format '{im.format}'. Use JPG, PNG, or WebP.")
        im.verify()
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > max_pixels:
            raise HTTPException(status_code=413, detail=f"Image exceeds pixel limit ({max_pixels} pixels).")
        return ImageOps.exif_transpose(im).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload a valid JPG, PNG, or WebP image.") from exc


def create_worker_app(
    *,
    token: str,
    default_parser: str,
    checkpoint_provenance: dict[str, Any],
    gpu_available_fn: Callable[[], bool],
    gpu_name_fn: Callable[[], str],
    parser_factory: Callable[[str], Any],
    vton_factory: Callable[[str], Any],
    mask_builder: Callable[[np.ndarray, str], np.ndarray],
    overlay_builder: Callable[[Image.Image, np.ndarray], Image.Image],
    cleanup_gpu_fn: Callable[[], None] | None = None,
    max_queue: int = 2,
    inference_timeout_s: float = 480.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> FastAPI:
    if not token or not token.strip():
        raise RuntimeError("FF_SERVER_TOKEN must be set; refusing to start an unauthenticated GPU worker.")

    files_verified = bool(checkpoint_provenance.get("files_verified")) if default_parser != "baseline" else True
    if default_parser != "baseline" and not files_verified:
        raise FileNotFoundError(
            f"Required V4 parser checkpoint incomplete ({checkpoint_provenance.get('verification_status')}): "
            f"missing {checkpoint_provenance.get('missing_required')}"
        )

    app = FastAPI(title="FashionFit-PK V4 GPU Worker", version="4.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v4-gpu-worker")
    state_lock = threading.Lock()
    queued_count = 0
    running_count = 0
    timed_out_waiters = 0
    total_completed = 0
    total_failed = 0
    worker_recovery_status = "healthy"

    model_states: dict[str, str] = {
        "parser": "not_loaded",
        "catvton": "not_loaded",
        "fashn_vton_1_5": "not_loaded",
    }
    loaded_cache: dict[str, Any] = {}
    last_error: dict[str, str] = {}

    def _authenticate(auth_header: str) -> None:
        if not secrets.compare_digest(auth_header or "", "Bearer " + token):
            raise HTTPException(status_code=401, detail="Invalid or missing Bearer access token.")

    def _set_model_state(name: str, status: str, err: str = "") -> None:
        with state_lock:
            model_states[name] = status
            if err:
                last_error[name] = err
            elif name in last_error:
                last_error.pop(name, None)

    def _evict_all_models_on_worker_thread(reason: str) -> None:
        """Called ONLY from inside the single worker thread (never races active inference)."""
        nonlocal worker_recovery_status
        loaded_cache.clear()
        gc.collect()
        if cleanup_gpu_fn:
            cleanup_gpu_fn()
        with state_lock:
            for k in list(model_states.keys()):
                model_states[k] = "not_loaded"
            worker_recovery_status = reason

    def _evict_other_vton_model(keep_model_id: str) -> None:
        """Single-large-VTON-model residency: evict the competing VTON model before loading another."""
        other = "fashn_vton_1_5" if keep_model_id == "catvton" else "catvton"
        cache_key = f"vton:{other}"
        if cache_key in loaded_cache:
            loaded_cache.pop(cache_key, None)
            gc.collect()
            if cleanup_gpu_fn:
                cleanup_gpu_fn()
            _set_model_state(other, "not_loaded")

    def _get_or_load(cache_key: str, state_key: str, factory: Callable[[], Any]) -> tuple[Any, float]:
        if cache_key in loaded_cache:
            return loaded_cache[cache_key], 0.0
        _set_model_state(state_key, "loading")
        t0 = time.perf_counter()
        try:
            obj = factory()
            obj.load()
            loaded_cache[cache_key] = obj
            _set_model_state(state_key, "ready")
            return obj, round(time.perf_counter() - t0, 3)
        except Exception as exc:
            loaded_cache.pop(cache_key, None)
            _set_model_state(state_key, "failed", str(exc))
            if cleanup_gpu_fn:
                cleanup_gpu_fn()
            raise

    async def _submit_bounded(fn: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
        """Submit task with strict slot ownership: capacity freed ONLY on completion or pre-start cancel."""
        nonlocal queued_count, running_count, timed_out_waiters, total_completed, total_failed

        job_state = {"started": False, "waiter_abandoned": False}
        job_lock = threading.Lock()

        with state_lock:
            if queued_count + running_count >= max_queue:
                raise HTTPException(
                    status_code=429,
                    detail=f"GPU worker queue is full ({running_count} running, {queued_count} queued, max {max_queue}).",
                )
            queued_count += 1

        def _wrapped() -> dict[str, Any]:
            nonlocal queued_count, running_count, timed_out_waiters, total_completed, total_failed
            with job_lock:
                job_state["started"] = True
            with state_lock:
                queued_count = max(0, queued_count - 1)
                running_count += 1
            try:
                out = fn(*args)
                with state_lock:
                    total_completed += 1
                return out
            except Exception:
                with state_lock:
                    total_failed += 1
                raise
            finally:
                with job_lock:
                    abandoned = job_state["waiter_abandoned"]
                with state_lock:
                    running_count = max(0, running_count - 1)
                    if abandoned:
                        timed_out_waiters = max(0, timed_out_waiters - 1)

        fut: Future[dict[str, Any]] = executor.submit(_wrapped)
        afut = asyncio.wrap_future(fut)
        try:
            return await asyncio.wait_for(asyncio.shield(afut), timeout=inference_timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            cancelled_before_start = fut.cancel()
            if cancelled_before_start:
                with state_lock:
                    queued_count = max(0, queued_count - 1)
                if isinstance(exc, asyncio.TimeoutError):
                    raise HTTPException(
                        status_code=504,
                        detail="Queued job timed out before starting and was cleanly cancelled.",
                    ) from exc
                raise
            # Already running on worker thread: do NOT release capacity prematurely!
            with job_lock:
                job_state["waiter_abandoned"] = True
            with state_lock:
                timed_out_waiters += 1
            if isinstance(exc, asyncio.TimeoutError):
                raise HTTPException(
                    status_code=504,
                    detail="HTTP wait timed out; underlying GPU task continues until completion and still holds its slot.",
                ) from exc
            raise

    @app.on_event("shutdown")
    def _shutdown() -> None:
        executor.shutdown(wait=False, cancel_futures=True)

    @app.get("/health")
    def health() -> dict[str, Any]:
        with state_lock:
            states_copy = dict(model_states)
            q = queued_count
            r = running_count
            t_out = timed_out_waiters
            errors_copy = dict(last_error)
            rec_status = worker_recovery_status
            resident_keys = set(loaded_cache.keys())
        any_ready = bool(resident_keys) and any(v == "ready" for v in states_copy.values())
        return {
            "schema_version": "v4.2",
            "status": "ok",
            "mode": "colab_gpu_live",
            "process_alive": True,
            "gpu_accessible": bool(gpu_available_fn()),
            "gpu": gpu_name_fn() if gpu_available_fn() else "unavailable",
            "files_verified": files_verified,
            "model_loaded": any_ready,
            "model_states": states_copy,
            "model_errors": errors_copy,
            "worker_recovery_status": rec_status,
            "auth_required": True,
            "parser": default_parser,
            "checkpoint_provenance": checkpoint_provenance,
            "running_count": r,
            "queued_count": q,
            "timed_out_waiters": t_out,
            "queue_depth": r + q,
            "max_queue": max_queue,
            "queue_accepting": (r + q) < max_queue,
            "models": sorted(VALID_MODELS),
            "supported_categories": sorted(VALID_CATEGORIES),
            "modes": sorted(VALID_TRYON_MODES),
            "honesty_note": "Visual appearance transfer research prototype; not physical tailoring/sizing.",
        }

    @app.get("/auth/verify")
    def verify_auth(authorization: str = Header(default="")) -> dict[str, Any]:
        _authenticate(authorization)
        return {"authenticated": True, "schema_version": "v4.2"}

    def _run_segment_sync(im: Image.Image, seg_variant: str) -> dict[str, Any]:
        try:
            kind = default_parser if seg_variant == "adapted" else "baseline"
            parser, load_s = _get_or_load(f"parser:{kind}", "parser", lambda: parser_factory(kind))
            t0 = time.perf_counter()
            labels = parser.predict(im)
            infer_s = round(time.perf_counter() - t0, 3)
            rgb = np.asarray(im)
            return {
                "model_id": kind,
                "checkpoint_provenance": (
                    checkpoint_provenance if kind == "universal" else {"verification_status": "pretrained_baseline"}
                ),
                "overlay_image": encode_png(overlay_builder(im, labels)),
                "kurta_cutout_image": encode_png(
                    Image.fromarray(np.where((labels == 1)[:, :, None], rgb, 255).astype(np.uint8))
                ),
                "class_coverage_pct": {
                    n: round(float(np.mean(labels == i)) * 100, 2)
                    for i, n in enumerate(["other", "kurta_kameez", "bottom", "dupatta"])
                },
                "load_seconds": load_s,
                "inference_seconds": infer_s,
                "seconds": round(load_s + infer_s, 3),
            }
        except Exception as exc:
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                _evict_all_models_on_worker_thread("recovered_after_oom")
                raise _WorkerOOMError("GPU out of memory during segmentation; caches evicted and states reset.") from exc
            raise

    def _run_vton_sync(
        person: Image.Image,
        garment: Image.Image,
        model_id: str,
        category: str,
        tryon_mode: str,
        garment_photo_type: str,
        person_age_group: str,
        seed: int,
    ) -> dict[str, Any]:
        try:
            total_load_s = 0.0
            parser_used = "internal_maskless"
            mask_img = None

            _evict_other_vton_model(model_id)

            if model_id == "catvton":
                parser, p_load = _get_or_load(
                    f"parser:{default_parser}", "parser", lambda: parser_factory(default_parser)
                )
                total_load_s += p_load
                labels = parser.predict(person)
                mask_arr = mask_builder(labels, tryon_mode)
                mask_img = Image.fromarray(mask_arr)
                parser_used = default_parser

            adapter, a_load = _get_or_load(f"vton:{model_id}", model_id, lambda: vton_factory(model_id))
            total_load_s += a_load

            t0 = time.perf_counter()
            if model_id == "catvton":
                res = adapter.run_tryon(person, garment, mask_img, seed=seed)
            else:
                fashn_cat = "tops" if tryon_mode == "upper_body" else "one-pieces"
                res = adapter.run_tryon(
                    person, garment, category=fashn_cat, garment_photo_type=garment_photo_type, seed=seed
                )
            infer_s = round(time.perf_counter() - t0, 3)

            return {
                "status": "ok",
                "engine": "colab_gpu_live",
                "model_id": model_id,
                "parser_id": parser_used,
                "checkpoint_provenance": (
                    checkpoint_provenance if parser_used == "universal" else {"verification_status": "pretrained_only"}
                ),
                "category": category,
                "tryon_mode": tryon_mode,
                "garment_photo_type": garment_photo_type,
                "person_age_group": person_age_group,
                "seed": seed,
                "load_seconds": round(total_load_s, 3),
                "inference_seconds": infer_s,
                "seconds": round(total_load_s + infer_s, 3),
                "peak_vram_mb": round(float(res.get("peak_vram_mb", 0.0)), 1),
                "filtered": bool(res.get("filtered", False)),
                "result_image": encode_png(res["image"]),
                "warnings": [
                    "Visual appearance preview only — not a physical fit or sizing guarantee.",
                    "Recipient dupatta exclusion is fixed in CatVTON masks; donor dupatta transfer is not modeled.",
                ],
            }
        except Exception as exc:
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                _evict_all_models_on_worker_thread("recovered_after_oom")
                raise _WorkerOOMError("GPU out of memory during try-on; caches evicted and states reset.") from exc
            raise

    @app.post("/segment")
    async def segment(
        image: UploadFile = File(...),
        seg_variant: str = Form("adapted"),
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        _authenticate(authorization)
        if seg_variant not in {"adapted", "baseline"}:
            raise HTTPException(status_code=400, detail=f"Unsupported seg_variant '{seg_variant}'.")
        im = await decode_clean_upload(image, max_bytes=max_bytes, max_pixels=max_pixels)
        try:
            return await _submit_bounded(_run_segment_sync, im, seg_variant)
        except HTTPException:
            raise
        except _WorkerOOMError as exc:
            raise HTTPException(status_code=503, detail={"code": "GPU_OOM_RECOVERED", "message": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc

    @app.post("/vton")
    async def vton(
        person_image: UploadFile = File(...),
        garment_image: UploadFile = File(...),
        model_id: str = Form("catvton"),
        category: str = Form("adult_upper"),
        tryon_mode: str = Form("upper_body"),
        garment_photo_type: str = Form("flat-lay"),
        person_age_group: str = Form("adult"),
        seed: int = Form(42),
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        _authenticate(authorization)
        validate_capability_combination(
            model_id=model_id,
            category=category,
            tryon_mode=tryon_mode,
            garment_photo_type=garment_photo_type,
            person_age_group=person_age_group,
        )
        person = await decode_clean_upload(person_image, max_bytes=max_bytes, max_pixels=max_pixels)
        garment = await decode_clean_upload(garment_image, max_bytes=max_bytes, max_pixels=max_pixels)
        try:
            return await _submit_bounded(
                _run_vton_sync,
                person,
                garment,
                model_id,
                category,
                tryon_mode,
                garment_photo_type,
                person_age_group,
                seed,
            )
        except HTTPException:
            raise
        except _WorkerOOMError as exc:
            raise HTTPException(status_code=503, detail={"code": "GPU_OOM_RECOVERED", "message": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"VTON inference failed: {exc}") from exc

    return app
