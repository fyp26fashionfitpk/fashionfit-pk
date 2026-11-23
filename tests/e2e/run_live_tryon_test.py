"""
FashionFit-PK End-to-End Live Try-On & 4-Class Segmentation Benchmark Runner
Usage:
    python run_live_tryon_test.py https://your-colab-url.trycloudflare.com
"""

import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(r"C:\Users\mutee\Downloads\AI Virtual Try-On FYP\test_images")
OUT_DIR = BASE_DIR / "live_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def post_multipart(url: str, fields: dict, files: list) -> dict:
    boundary = "----FashionFitLiveE2EBoundary987654"
    body = bytearray()
    for name, val in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{val}\r\n".encode())
    for field_name, filename, content, content_type in files:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "FashionFit-QA/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def save_data_url(data_url: str, out_path: Path):
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url
    out_path.write_bytes(base64.b64decode(b64))


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_live_tryon_test.py <COLAB_GPU_URL>")
        sys.exit(1)

    colab_url = sys.argv[1].strip().rstrip("/")
    print(f"=== 1. HEALTH CHECK: {colab_url}/health ===")
    with urllib.request.urlopen(f"{colab_url}/health", timeout=20) as r:
        health = json.loads(r.read().decode("utf-8"))
        print(json.dumps(health, indent=2))

    # 2. 4-Class Segmentation Test (Mountain Dupatta Stress Case)
    print("\n=== 2. 4-CLASS SEGMENTATION TEST (Adapted R2 SegFormer-B2) ===")
    seg_img = BASE_DIR / "1_person_tests" / "person_worst_challenging.jpg"
    t0 = time.time()
    seg_res = post_multipart(
        f"{colab_url}/segment",
        fields={"seg_variant": "adapted"},
        files=[("image", seg_img.name, seg_img.read_bytes(), "image/jpeg")],
    )
    dt = time.time() - t0
    print(f"  Completed in {dt:.2f}s | keys: {list(seg_res.keys())}")
    if seg_res.get("overlay_image"):
        save_data_url(seg_res["overlay_image"], OUT_DIR / "seg_worst_overlay.png")
    if seg_res.get("cutout_image"):
        save_data_url(seg_res["cutout_image"], OUT_DIR / "seg_worst_cutout.png")

    # 3. Live Virtual Try-On Benchmark Suite (Best, Middle, Worst)
    pairs = [
        ("1_best_case", BASE_DIR / "1_person_tests" / "person_best_studio.jpg", BASE_DIR / "2_kurta_tests" / "kurta_best_flatlay.jpg"),
        ("2_mid_case", BASE_DIR / "1_person_tests" / "person_mid_casual.jpg", BASE_DIR / "2_kurta_tests" / "kurta_mid_hanging.jpg"),
        ("3_worst_case", BASE_DIR / "1_person_tests" / "person_worst_challenging.jpg", BASE_DIR / "2_kurta_tests" / "kurta_worst_occluded.jpg"),
    ]

    for tag, person_path, kurta_path in pairs:
        print(f"\n=== VTON TEST [{tag}] ({person_path.name} + {kurta_path.name}) ===")
        t0 = time.time()
        res = post_multipart(
            f"{colab_url}/vton",
            fields={
                "model_id": "catvton",
                "category": "tops",
                "garment_photo_type": "flat-lay",
                "seed": "42",
            },
            files=[
                ("person_image", person_path.name, person_path.read_bytes(), "image/jpeg"),
                ("garment_image", kurta_path.name, kurta_path.read_bytes(), "image/jpeg"),
            ],
        )
        dt = time.time() - t0
        out_file = OUT_DIR / f"vton_{tag}_result.png"
        if res.get("result_image"):
            save_data_url(res["result_image"], out_file)
            print(f"  [PASS] Generated in {dt:.2f}s -> saved to {out_file}")
        else:
            print(f"  [WARN] No result_image in response: {list(res.keys())}")

    print(f"\nALL LIVE TESTS COMPLETE! Results saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
