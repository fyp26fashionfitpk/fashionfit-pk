# FashionFit-PK — Formal Test Cases (FYP I)

*Note: Test cases were structured with AI assistance and manually reviewed, verified, and finalized against the live Vercel deployment and local `unittest` suite.*

| Test ID | Category | Test Case Description | Input / Precondition | Expected Output | Status |
|---|---|---|---|---|---|
| **TC-01** | UI / Upload | Upload valid person image (JPG/PNG) | Portrait photo < 10 MB | Preview renders cleanly without layout overflow | PASS |
| **TC-02** | UI / Upload | Select catalog garment or upload custom kurta | Valid garment image | Selected garment highlights and enables Try-On CTA | PASS |
| **TC-03** | UI / 3-Pose | Switch between Front / Angled / Back tabs | Completed try-on session | Active pose switches without unmounting state | PASS |
| **TC-04** | UI / Responsive | Render hero & studio workspace on desktop & mobile | Viewport 390px and 1440px | Zero badge overlap, accessible controls | PASS |
| **TC-05** | API / Health | Call `GET /api/health` on Vercel deployment | Deployed app live | HTTP 200 JSON with service status | PASS |
| **TC-06** | API / Resilience | Trigger try-on while GPU tunnel is offline | Backend URL unreachable | Graceful user-facing recovery banner (no crash) | PASS |
| **TC-07** | ML / Parser | Run 4-class SegFormer-B2 on 3-piece Pakistani outfit | Held-out Gold test image | Distinct masks for `kurta_kameez`, `bottom`, `dupatta` | PASS |
| **TC-08** | ML / Masking | `preserve_dupatta_over_mask()` on overlapping scarf | Mask overlapping class 3 | Zero dupatta pixels remain inside editable mask | PASS |
| **TC-09** | ML / Flared Mask | `make_flared_lehenga_mask()` on bridal silhouette | Tight bottom segmentation | Lower mask expands smoothly in A-line toward hem | PASS |
| **TC-10** | ML / 2-Piece | Run parser on 2-piece outfit (no dupatta) | Dupatta-negative test image | 0% false positive dupatta pixels | PASS |
| **TC-11** | ML / VTON | Run CatVTON + Pakistani LoRA inference | Person + unseen SKU kurta | Photorealistic output; held-out FID 95.28, LPIPS 0.1531 | PASS |
| **TC-12** | ML / Validation | Verify invalid mode passed to `build_tryon_mask()` | `mode="invalid"` | Raises `ValueError` with supported mode list | PASS |
| **TC-13** | ML / Shape Guard | Mismatched `target_mask` and `seg_map` dimensions | `(64,64)` vs `(32,32)` | Raises `ValueError` describing shape mismatch | PASS |
| **TC-14** | Metrics / IoU | `compute_per_class_iou()` on identical pred/target | Exact match 4-class map | All class IoUs = `100.0%`, `mIoU = 100.0%` | PASS |
| **TC-15** | Metrics / Leakage | `validate_zero_leakage_sku_split()` on disjoint splits | Disjoint train/test SKUs | Returns empty overlap set (`set()`) | PASS |
| **TC-16** | Metrics / Leakage | `validate_zero_leakage_sku_split()` catches duplicate SKU | Overlapping SKU ID | Returns exact offending SKU IDs | PASS |
| **TC-17** | Security | Verify `.env.local` is excluded from Git | `.gitignore` check | Zero API keys or tokens tracked in repository | PASS |
| **TC-18** | Automation / CI | Run `Makefile` test target (`make test`) | Python 3.10+ environment | All unit tests pass with exit code 0 | PASS |
| **TC-19** | Profiling | Run `bin/profile_inference.py` benchmark | Synthetic 512x384 map | Outputs deterministic masking latency (ms) & memory | PASS |
| **TC-20** | API / Gemini | Call `/api/gemini-status` endpoint | Vercel serverless route | Returns valid JSON configuration state | PASS |
