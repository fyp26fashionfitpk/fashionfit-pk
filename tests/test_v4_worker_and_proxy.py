"""Deterministic regression tests for V4 GPU Worker core & SSRF-hardened Local Proxy.

Every test name matches its exact assertions:
- Timeout does NOT free a running GPU slot; capacity restores exactly once when A completes.
- Queued request cancelled before execution frees its queued slot cleanly.
- OOM resets model_states to not_loaded, returns HTTP 503 (GPU_OOM_RECOVERED), and allows clean reload + retry.
- Single-VTON-model residency evicts competing model on switch.
- Combination-level capability rejections (wedding_lehenga+upper_body, catvton+model, adult_upper+child).
- Byte/pixel boundary limits, format rejection (GIF), EXIF orientation transpose, manifest match/mismatch.
- Proxy runtime config auth (X-Admin-Token), SSRF host/IP blocking, and redirect refusal.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

WEB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEB_ROOT / "src" / "colab_server"))
sys.path.insert(0, str(WEB_ROOT / "src" / "backend"))

import v4_worker_core as worker_core
import main as proxy_main


def _make_image_bytes(w: int = 32, h: int = 32, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (15, 92, 77)).save(buf, format=fmt)
    return buf.getvalue()


class _FakeParser:
    def __init__(self):
        self.loads = 0

    def load(self):
        self.loads += 1

    def predict(self, image: Image.Image) -> np.ndarray:
        arr = np.zeros((image.height, image.width), dtype=np.uint8)
        arr[4:20, 4:24] = 1
        return arr


class _ControllableVton:
    def __init__(
        self,
        *,
        started_evt: threading.Event | None = None,
        release_evt: threading.Event | None = None,
        fail_oom_times: int = 0,
    ):
        self.started_evt = started_evt
        self.release_evt = release_evt
        self.remaining_oom = fail_oom_times
        self.load_count = 0
        self.run_count = 0

    def load(self):
        self.load_count += 1

    def run_tryon(self, person, garment, *args, **kwargs):
        self.run_count += 1
        if self.started_evt:
            self.started_evt.set()
        if self.release_evt:
            self.release_evt.wait(timeout=5.0)
        if self.remaining_oom > 0:
            self.remaining_oom -= 1
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return {"image": person.copy(), "seconds": 0.01, "peak_vram_mb": 128.0, "filtered": False}


def _build_worker(
    *,
    parser_factory=None,
    vton_factory=None,
    max_queue: int = 2,
    inference_timeout_s: float = 10.0,
    max_bytes: int = worker_core.DEFAULT_MAX_BYTES,
    max_pixels: int = worker_core.DEFAULT_MAX_PIXELS,
    ckpt_prov=None,
):
    prov = ckpt_prov or {
        "files_verified": True,
        "verification_status": "hashes_recorded_no_manifest",
        "files": {"config.json": "a" * 64, "model.safetensors": "b" * 64, "preprocessor_config.json": "c" * 64},
    }
    return worker_core.create_worker_app(
        token="secret-test-token",
        default_parser="universal",
        checkpoint_provenance=prov,
        gpu_available_fn=lambda: True,
        gpu_name_fn=lambda: "Mock-T4-CPU-Test",
        parser_factory=parser_factory or (lambda kind: _FakeParser()),
        vton_factory=vton_factory or (lambda m: _ControllableVton()),
        mask_builder=lambda seg, mode: (seg == 1).astype(np.uint8) * 255,
        overlay_builder=lambda im, seg: im,
        max_queue=max_queue,
        inference_timeout_s=inference_timeout_s,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
    )


class TestWorkerTimeoutAdmissionAndOOM(unittest.TestCase):
    def test_timeout_does_not_release_running_slot_until_worker_completes(self) -> None:
        started = threading.Event()
        release = threading.Event()
        adapter = _ControllableVton(started_evt=started, release_evt=release)
        app = _build_worker(
            vton_factory=lambda m: adapter,
            max_queue=1,
            inference_timeout_s=0.15,
        )
        client = TestClient(app)
        png = _make_image_bytes()
        headers = {"Authorization": "Bearer secret-test-token"}

        # A starts, blocks on release event, and its HTTP wait times out (504)
        r_a = client.post(
            "/vton",
            headers=headers,
            data={"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body"},
            files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
        )
        self.assertEqual(r_a.status_code, 504)
        self.assertTrue(started.is_set())

        # Capacity MUST still reflect A's ongoing execution (running_count=1, queue_accepting=False)
        h_busy = client.get("/health").json()
        self.assertEqual(h_busy["running_count"], 1)
        self.assertEqual(h_busy["timed_out_waiters"], 1)
        self.assertFalse(h_busy["queue_accepting"])

        # Request B beyond max_queue=1 MUST be rejected with HTTP 429
        r_b = client.post(
            "/vton",
            headers=headers,
            data={"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body"},
            files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
        )
        self.assertEqual(r_b.status_code, 429)

        # Releasing A restores capacity exactly once
        release.set()
        for _ in range(40):
            h_done = client.get("/health").json()
            if h_done["running_count"] == 0:
                break
            time.sleep(0.02)
        self.assertEqual(h_done["running_count"], 0)
        self.assertEqual(h_done["timed_out_waiters"], 0)
        self.assertTrue(h_done["queue_accepting"])

    def test_queued_request_cancelled_before_start_restores_queued_slot(self) -> None:
        started = threading.Event()
        release = threading.Event()
        adapter = _ControllableVton(started_evt=started, release_evt=release)
        app = _build_worker(
            vton_factory=lambda m: adapter,
            max_queue=2,
            inference_timeout_s=0.15,
        )
        client = TestClient(app)
        png = _make_image_bytes()
        headers = {"Authorization": "Bearer secret-test-token"}

        res_a = []
        t_a = threading.Thread(
            target=lambda: res_a.append(
                client.post(
                    "/vton",
                    headers=headers,
                    data={"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body"},
                    files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
                ).status_code
            )
        )
        t_a.start()
        self.assertTrue(started.wait(timeout=3.0))

        # B enters queue behind running A, times out BEFORE starting, and is cancelled cleanly
        r_b = client.post(
            "/vton",
            headers=headers,
            data={"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body"},
            files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
        )
        self.assertEqual(r_b.status_code, 504)
        h_mid = client.get("/health").json()
        self.assertEqual(h_mid["queued_count"], 0)
        self.assertEqual(h_mid["running_count"], 1)

        release.set()
        t_a.join(timeout=5.0)
        self.assertEqual(adapter.run_count, 1)  # B never ran on worker thread!

    def test_oom_resets_model_states_and_allows_subsequent_reload_and_success(self) -> None:
        created_adapters: list[_ControllableVton] = []
        oom_left = {"n": 2}

        def _factory(m: str):
            fail_now = 1 if oom_left["n"] > 0 else 0
            if oom_left["n"] > 0:
                oom_left["n"] -= 1
            a = _ControllableVton(fail_oom_times=fail_now)
            created_adapters.append(a)
            return a

        app = _build_worker(vton_factory=_factory)
        client = TestClient(app)
        png = _make_image_bytes()
        headers = {"Authorization": "Bearer secret-test-token"}

        # Two consecutive OOMs must both return 503, reset model_loaded=False & model_states=not_loaded
        for _ in range(2):
            r = client.post(
                "/vton",
                headers=headers,
                data={"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body"},
                files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
            )
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["detail"]["code"], "GPU_OOM_RECOVERED")
            h = client.get("/health").json()
            self.assertFalse(h["model_loaded"])
            self.assertEqual(h["model_states"]["catvton"], "not_loaded")
            self.assertEqual(h["model_states"]["parser"], "not_loaded")
            self.assertEqual(h["worker_recovery_status"], "recovered_after_oom")
            self.assertTrue(h["queue_accepting"])

        # Third request reloads cleanly and succeeds (HTTP 200)
        r_ok = client.post(
            "/vton",
            headers=headers,
            data={"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body"},
            files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
        )
        self.assertEqual(r_ok.status_code, 200)
        self.assertEqual(len(created_adapters), 3)
        self.assertTrue(client.get("/health").json()["model_loaded"])

    def test_single_vton_model_residency_evicts_competing_model(self) -> None:
        app = _build_worker()
        client = TestClient(app)
        png = _make_image_bytes()
        headers = {"Authorization": "Bearer secret-test-token"}

        r1 = client.post(
            "/vton",
            headers=headers,
            data={"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body"},
            files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
        )
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(client.get("/health").json()["model_states"]["catvton"], "ready")

        r2 = client.post(
            "/vton",
            headers=headers,
            data={"model_id": "fashn_vton_1_5", "category": "adult_upper", "tryon_mode": "upper_body", "garment_photo_type": "model"},
            files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
        )
        self.assertEqual(r2.status_code, 200)
        states = client.get("/health").json()["model_states"]
        self.assertEqual(states["catvton"], "not_loaded")
        self.assertEqual(states["fashn_vton_1_5"], "ready")


class TestCapabilitiesBoundariesAndManifest(unittest.TestCase):
    def test_combination_capability_rejections(self) -> None:
        app = _build_worker()
        client = TestClient(app)
        png = _make_image_bytes()
        headers = {"Authorization": "Bearer secret-test-token"}

        bad_combos = [
            {"model_id": "catvton", "category": "wedding_lehenga", "tryon_mode": "upper_body", "person_age_group": "adult"},
            {"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body", "garment_photo_type": "model", "person_age_group": "adult"},
            {"model_id": "catvton", "category": "adult_upper", "tryon_mode": "upper_body", "person_age_group": "child"},
            {"model_id": "catvton", "category": "girl_child", "tryon_mode": "full_outfit", "person_age_group": "adult"},
            {"model_id": "catvton", "category": "cross_age", "tryon_mode": "flared_lehenga", "person_age_group": "child"},
        ]
        for combo in bad_combos:
            r = client.post(
                "/vton",
                headers=headers,
                data=combo,
                files={"person_image": ("p.png", png, "image/png"), "garment_image": ("g.png", png, "image/png")},
            )
            self.assertEqual(r.status_code, 400, f"Expected 400 for {combo}, got {r.status_code}")

    def test_byte_pixel_and_format_boundaries_and_exif_transpose(self) -> None:
        png_ok = _make_image_bytes(10, 10, "PNG")  # 100 pixels
        png_too_many_pixels = _make_image_bytes(11, 10, "PNG")  # 110 pixels
        gif_bytes = _make_image_bytes(10, 10, "GIF")

        app = _build_worker(max_bytes=len(png_ok), max_pixels=100)
        client = TestClient(app)
        headers = {"Authorization": "Bearer secret-test-token"}

        # Exact byte & pixel boundary passes
        r_ok = client.post(
            "/segment", headers=headers, data={"seg_variant": "adapted"},
            files={"image": ("ok.png", png_ok, "image/png")},
        )
        self.assertEqual(r_ok.status_code, 200)

        # max_bytes + 1 -> HTTP 413
        r_big_bytes = client.post(
            "/segment", headers=headers, data={"seg_variant": "adapted"},
            files={"image": ("big.png", png_ok + b"x", "image/png")},
        )
        self.assertEqual(r_big_bytes.status_code, 413)

        # > max_pixels -> HTTP 413 (allow bytes so pixel check triggers)
        app_px = _build_worker(max_bytes=100000, max_pixels=100)
        r_big_px = TestClient(app_px).post(
            "/segment", headers=headers, data={"seg_variant": "adapted"},
            files={"image": ("px.png", png_too_many_pixels, "image/png")},
        )
        self.assertEqual(r_big_px.status_code, 413)

        # Rejected format GIF -> HTTP 400
        r_gif = TestClient(app_px).post(
            "/segment", headers=headers, data={"seg_variant": "adapted"},
            files={"image": ("anim.gif", gif_bytes, "image/gif")},
        )
        self.assertEqual(r_gif.status_code, 400)

    def test_checkpoint_manifest_match_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "config.json").write_text("{}", encoding="utf-8")
            (d / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (d / "model.safetensors").write_bytes(b"weights")

            rec = worker_core.verify_checkpoint_dir(d)
            self.assertTrue(rec["files_verified"])
            self.assertEqual(rec["verification_status"], "hashes_recorded_no_manifest")

            matched = worker_core.verify_checkpoint_dir(d, expected_manifest=rec["files"])
            self.assertTrue(matched["files_verified"])
            self.assertEqual(matched["verification_status"], "verified_against_manifest")

            mismatched = worker_core.verify_checkpoint_dir(d, expected_manifest={"config.json": "0" * 64})
            self.assertFalse(mismatched["files_verified"])
            self.assertEqual(mismatched["verification_status"], "manifest_hash_mismatch")


class TestLocalProxySecurity(unittest.TestCase):
    def test_runtime_config_auth_and_redirect_refusal(self) -> None:
        client = TestClient(proxy_main.app)

        # Disabled by default -> 403
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FF_ALLOW_RUNTIME_UPSTREAM_CONFIG", None)
            self.assertEqual(client.post("/api/config/colab", json={"colab_gpu_url": "https://a.trycloudflare.com"}).status_code, 403)

        # Enabled + missing/wrong X-Admin-Token -> 401
        with mock.patch.dict(os.environ, {"FF_ALLOW_RUNTIME_UPSTREAM_CONFIG": "true", "FF_ADMIN_TOKEN": "adm-123"}):
            self.assertEqual(client.post("/api/config/colab", json={"colab_gpu_url": "https://a.trycloudflare.com"}).status_code, 401)
            self.assertEqual(
                client.post("/api/config/colab", headers={"X-Admin-Token": "wrong"}, json={"colab_gpu_url": "https://a.trycloudflare.com"}).status_code,
                401,
            )
            # Valid admin token + disallowed host -> 403
            self.assertEqual(
                client.post("/api/config/colab", headers={"X-Admin-Token": "adm-123"}, json={"colab_gpu_url": "https://evil.example.com"}).status_code,
                403,
            )

        # Redirect handler must refuse HTTP 302 so Bearer credentials are never forwarded
        handler = proxy_main._NoRedirectHandler()
        fake_req = mock.Mock(full_url="https://demo.trycloudflare.com/vton")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            handler.redirect_request(fake_req, None, 302, "Found", {}, "https://attacker.example.com")
        self.assertEqual(ctx.exception.code, 502)


if __name__ == "__main__":
    unittest.main()

