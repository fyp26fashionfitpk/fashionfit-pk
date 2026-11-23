# Changelog & End-to-End Development Traceability Log

All notable engineering, dataset curation, research benchmarking, QA, and compliance milestones for **FashionFit-PK** are recorded chronologically below so reviewers and the ITU FoS FYP Committee can trace every stage of the project.

---

## [v1.1.0-fyp1] — Complete Evidence, Benchmarks & Traceability Bundle (Nov 24, 2026)
### Added
- **Live End-to-End Try-On Proof Gallery (`research/results/live_tryon_results/`):** Best-case, mid-case, and worst-case VTON inference outputs alongside baseline vs. adapted garment cutout overlays.
- **Dupatta 4-Class Segmentation Proofs (`research/results/dupatta_proofs/`):** 4 visual proof samples demonstrating clean dupatta separation alongside `segformer_gold_test_predictions_grid.jpg`.
- **Live Cloud QA Evidence (`docs/evidence/live_vercel_qa_screenshot.png`):** Verified live Vercel production screenshot following E2E QA fixes.
- **Reproducible E2E Test Harness & Fixtures (`tests/e2e/run_live_tryon_test.py`, `tests/fixtures/`):** Studio, casual, and challenging person/kurta test fixtures.
- **Machine-Readable & Publication Artifacts:** `research/results/measured_benchmarks.json` and `research/results/paper_tables.tex`.
- **Traceability & Governance Docs:** `CHANGELOG.md`, `docs/SUPERVISOR_QUESTIONS_AND_V3_EXTENSION.md`, and `docs/AI_USAGE_AND_ETHICS.md`.

---

## [v1.0.0-fyp1] — Unified ITU FoS Committee Repository & Missing Semester Integration (Nov 24, 2026)
### Added
- Created isolated official project repository under `fyp26fashionfitpk` (`fyp26_fashionfitpk@gmail.com`) following industry-standard folder structure (`src/`, `lib/`, `bin/`, `docs/`, `tests/`, `research/`).
- Authored complete Formal FYP I Report (`docs/FYP1_COMPLETE_DOCUMENTATION.md`) covering both Standard Software and Research-Based sections (5-project survey, architecture, mathematical formulations, experimental setup, measured results).
- Authored 20-case formal Test Case Matrix (`docs/TEST_CASES.md`) + automated `unittest` suite (12/12 passing in `0.037s`).
- Applied MIT *The Missing Semester*: executable shell scripts (`bin/`), root `Makefile`, latency/memory profiler (`bin/profile_inference.py`), GitHub Actions CI (`.github/workflows/ci.yml`), and isolated per-repo Git config.

---

## [Phase 6] — V3 Kids & Wedding Formal (Flared Lehenga) Extension (Nov 20–23, 2026)
### Completed
- Built clean V3 commercial catalog dataset via `research/v3_kids_formal_notebooks/01_V3_Kids_and_Wedding_Formal_Dataset_Builder.ipynb`:
  - **360 unique images across 181 unique SKUs** (Train: 291, Unseen-SKU Test: 62).
  - **Kids Ethnic (180 images):** Peraan Mini (96), Limelight Girls (84).
  - **Wedding Formals / Lehengas (180 images):** Peraan Formals (180).
- Exported 100 unique outfits (50 Kids + 50 Wedding Formals) to Roboflow (`FashionFit-PK-V3-Kids-Formal`) for Gold 4-class annotation.
- Authored and syntax-verified all 4 V3 notebooks (`01`..`04`) and implemented `make_flared_lehenga_mask()` in `lib/mask_utils.py`.

---

## [Phase 5] — Live Cloud Deployment, E2E QA & 5 Production Fixes (Nov 18–20, 2026)
### Completed
- Deployed production React + Vite + Vercel Serverless WebApp at [https://fashionfit-pk.vercel.app/](https://fashionfit-pk.vercel.app/).
- Executed full browser E2E QA pass and shipped 5 verified fixes (tracked in WebApp commit history `a4549cb` $\rightarrow$ `db199ef`):
  1. Added `/api/health` telemetry endpoint.
  2. Fixed 3-pose demo tab mount state persistence.
  3. Fixed hero badge responsive spacing overlap.
  4. Added graceful error recovery when Colab GPU backend is offline.
  5. Sanitized `/api/gemini-status` and locked original person pose/framing in person prep prompt.
- Generated 14-slide executive FYP I Defense Presentation (`docs/FashionFit_PK_FYP1_Defense_Presentation.pptx`).

---

## [Phase 4] — CatVTON Pakistani LoRA Fine-Tuning & Evaluation (Nov 14–18, 2026)
### Completed
- Fine-tuned CatVTON UNet with rank-16 LoRA for **1,000 steps** on **930 Pakistani kurta pairs** (converged training loss: **0.0343**).
- Evaluated on held-out unseen-SKU test set ($N = 50$):
  - **FID:** `120.16` $\rightarrow$ **`95.28`** (**-20.7%**)
  - **LPIPS:** `0.1869` $\rightarrow$ **`0.1531`** (**-18.1%**)
  - **CLIP-I:** **`0.9488`**

---

## [Phase 3] — Gold Human-Verified 4-Class SegFormer-B2 Parser (Nov 10–14, 2026)
### Completed
- Discovered zero-shot GroundingDINO-tiny produced ~85% false positives on dupattas (confusing embroidered kurtas/jackets with scarves).
- Pivoted to **95 human-verified Gold outfits** via Roboflow SAM (44 dupatta-positive, 51 dupatta-negative) and resolved Roboflow compressed COCO RLE decoding via `pycocotools.mask.decode`.
- Fine-tuned `SegFormer-B2` (25 epochs, T4 GPU) achieving **76.59% 4-class mIoU** (+24.06% over baseline **52.53%**), **55.82% dupatta IoU** (17.9x jump from **3.12%**), and **0% false positives** on 2-piece outfits.

---

## [Phase 2] — Pakistani Dataset Curation & Zero-Leakage Audit (Nov 5–10, 2026)
### Completed
- Curated multi-brand Pakistani women's ethnic wear studio dataset from **Generation** (237), **Ethnc** (180), **Limelight** (180), and **Reviewed Pilot** (27).
- Discovered and eliminated **597 `.source.*` duplicate twin images** via filename filtering, SHA-256, and perceptual `dHash`.
- Enforced strict **SKU-level train/test splitting** across **624 unique images (301 unique SKUs)** with **0% data leakage**.

---

## [Phase 1] — Initial Architecture, FastAPI Backend & Colab T4 GPU Server (Oct–Nov 2026)
### Completed
- Built initial React frontend, FastAPI inference proxy (`src/backend/main.py`), and Colab T4 GPU live inference server (`src/colab_server/FashionFit_PK_Live_GPU_Server.ipynb`).
- Wired Gemini garment canonicalization bridge and multi-key failover pool for reliable live demonstrations.
