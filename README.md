# FashionFit-PK — Culturally-Aware Pakistani Clothing Segmentation & Virtual Try-On (Research Prototype)

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](#)
[![Institution](https://img.shields.io/badge/ITU-FoS%20Fall%202026%20FYP-0f766e)](#)
[![Status](https://img.shields.io/badge/Release-Experimental%20Research%20Prototype-amber)](#)

**FashionFit-PK** is an open, reproducible research prototype for Pakistani ethnic-wear 4-class semantic segmentation and virtual try-on spanning adult upper-body kurtas, girl-child outfits, cross-age visual garment transfer, and wedding lehenga research.

> **Honest Scope & Limitations:** All generated outputs are **visual appearance previews**, not physical sizing/tailoring guarantees. Core try-on operates without paid Gemini billing (Gemini is disabled by default). Live GPU inference uses an authenticated temporary Colab T4/L4 worker (not 24/7 public hosting).

---

## Verified Empirical Results (Strictly Separated by Evidence Type)

### 1. V4 Universal 4-Class Semantic Segmentation (NB02, $N=30$ Held-Out Test Images)
*Segmentation overlap only — does NOT measure try-on visual quality.*

| Domain | $N$ | Baseline FG mIoU | Adult Gold FG mIoU | **V4 Universal FG mIoU** | Notes |
|---|---:|---:|---:|---:|---|
| `adult_pret` | 15 | 34.64% | 69.99% | **69.86%** (-0.13 pp) | Adult bottom regressed `69.10% → 59.74%`. |
| `kids_ethnic` | 8 | 43.31% | 54.00% | **77.63%** (+23.63 pp) | Strongest pilot gain; 6-image clean ethnic subset `78.05%`. |
| `wedding_formals` | 7 | 23.93% | 54.61% | **55.96%** (+1.36 pp) | Formal bottom regressed `30.80% → 21.98%`. |
| **Overall Pooled** | **30** | **36.51%** | **64.48%** | **72.64%** (+8.16 pp) | `09_MODELS/universal_b2/best_checkpoint` (epoch 11). |

### 2. Offline VTON Human Evaluation (`blind_full20_v2`, 80 Outputs, Single Reviewer: Mutee)
*Single-reviewer exploratory results only. Abdullah's independent review was intentionally skipped; the predeclared two-reviewer release gate remains **UNMET**.*

| Pipeline | $N$ | Strict Pass (`all≥4, !MA, !SV`) | Major Artifacts | Scope Violations | Mean Design | Mean ID | Mean Scope |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`fashn_vton_1_5__original_model_photo`** | 20 | **6 / 20 (30.0%)** | **7** | 10 | **4.35** | **5.00** | 2.75 |
| `fashn_vton_1_5__internal_maskless` (cutout) | 20 | 3 / 20 (15.0%) | 15 | 4 | 2.90 | 4.90 | 4.20 |
| `catvton__universal` (V4 masks) | 20 | 2 / 20 (10.0%) | 12 | **0** | 2.40 | 4.65 | **4.75** |
| `catvton__baseline` (Baseline masks) | 20 | 1 / 20 (5.0%) | 15 | 3 | 2.45 | 4.70 | 4.30 |

*Note on Legacy Experiment Quarantine:* Earlier historical SD1.5-Inpainting LoRA runs (`FID 120.16 → 95.28`, `LPIPS 0.1869 → 0.1531`) did not initialize from author CatVTON attention weights and used self-cutout pseudo-references on an older split; they are quarantined as **unverified legacy experiments** and excluded from current validated claims.

---

## Repository Structure & Clean-Clone Commands

```text
fyp26-fashionfit-pk/
├── .github/workflows/ci.yml             # Runs Python tests, JS/React component tests & Vite build
├── bin/                                 # Executable shell & profiling utilities
├── docs/                                # Completion tracker, capability matrix, FYP I docs & test cases
├── lib/                                 # Core 4-class masking & evaluation utilities
├── research/                            # Reproducible notebooks, 03D FASHN control runner & JSON/LaTeX results
├── src/                                 # Full-stack app (frontend/, backend/, api/, colab_server/)
└── tests/                               # Python worker/proxy/runner tests + JS orchestrator tests
```

```bash
# 1. Run all Python unit & regression tests (worker, proxy, 03D runner, lib)
python -m unittest discover -s tests -v

# 2. Run Node request-ownership & Gemini-disabled tests
node --test tests/test_studio_controller.mjs

# 3. Run rendered React <StudioPage /> component tests & production build
cd src/frontend && npm ci && npm test && npm run build
```

---

## Authoritative Documentation
* [Completion Tracker & ITU FoS Compliance Audit](docs/COMPLETION_TRACKER_AND_COMPLIANCE.md)
* [Capability Matrix, Separated Evidence & Provenance](docs/SYSTEM_CAPABILITY_MATRIX_AND_PROVENANCE.md)
* [Formal FYP I Documentation](docs/FYP1_COMPLETE_DOCUMENTATION.md) & [Architecture](docs/ARCHITECTURE.md)
* [Test Cases Checklist (Pending Manual Student Review)](docs/TEST_CASES.md)
* [MIT Missing Semester Applied Practices](docs/MISSING_SEMESTER_APPLIED.md)
* [AI Usage Transparency & Data Ethics](docs/AI_USAGE_AND_ETHICS.md)

## Current Colab smoke handover

Use the corrected live smoke notebook first, then `research/v4_controlled_experiments/03D_FASHN_SegmentationFree_Control_v2.ipynb` with SMOKE_ONLY=True. See `docs/COLAB_SMOKE_RUN_ORDER_2026-11-28.txt`. These notebooks are CPU-checked execution packages, not evidence of completed new GPU runs. Old 03C and 03D v1 outputs stay unchanged.
