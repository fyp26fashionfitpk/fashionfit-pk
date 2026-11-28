# FashionFit-PK V4 — Separated Evidence Tables, Capability Matrix & Provenance

**Updated:** 28 November 2026  
**WebApp Repo:** `muteekhan06/fashionfit-pk`  
**V4 Research Root (Google Drive):** `/content/drive/MyDrive/FashionFit_PK_Advisor_V4_2026-11-26` (`1aDpZcdx2XjBInfi0qN_Wvwneul8HApT6`)  
**Blind Review Folder (Drive):** `11_REPORTS/blind_full20_v2` (`1XQyMbwb5JhrZ0dHGqYskUUNrg6iRZRQ4`)  
**Completed Local Single-Reviewer Export:** `C:\Users\mutee\Downloads\reviews_mutee.csv` *(kept out of public git repo)*

---

## 1. Strictly Separated Evidence Tables

### Table A — Segmentation Evidence (NB02, $N=30$ Held-Out Test Images; Overlap Only — NOT Try-On Quality)
| Domain | $N$ | Baseline FG mIoU | Adult Gold FG mIoU | V4 Universal FG mIoU | Known Regressions / Limits |
|---|---:|---:|---:|---:|---|
| `adult_pret` | 15 | 34.64% | 69.99% | **69.86%** (-0.13 pp) | Adult bottom regressed `69.10% -> 59.74%`. |
| `kids_ethnic` | 8 | 43.31% | 54.00% | **77.63%** (+23.63 pp) | 6-image clean ethnic subset: 78.05%; small pilot ($N=8$). |
| `wedding_formals` | 7 | 23.93% | 54.61% | **55.96%** (+1.36 pp) | Formal bottom regressed `30.80% -> 21.98%`; lacks full adult lehenga-choli test set. |
| **Overall Pooled** | **30** | **36.51%** | **64.48%** | **72.64%** (+8.16 pp) | Segmentation overlap only; does not establish try-on fidelity. |

### Table B — Offline VTON Execution Evidence (NB03, 20 Cases $\times$ 3 Conditions; Execution Only — NOT Quality)
| Condition | Planned | Succeeded | Failed / Filtered | Mean Inference Time (T4) | Peak VRAM | Notes |
|---|---:|---:|---:|---:|---:|---|
| CatVTON + Baseline Masks (`lora=None`) | 20 | 20 | 0 / 0 | 66.30 s | 4.14 GiB | Execution count only; includes color/remnant failures. |
| CatVTON + V4 Universal Masks (`lora=None`) | 20 | 20 | 0 / 0 | 63.24 s | 4.14 GiB | Execution count only; mixed vs baseline across cases. |
| FASHN VTON v1.5 (`flat-lay`, mask-free) | 20 | 20 | 0 / 0 | 396.35 s | 10.90 GiB | Occluded cutout voids caused white waist gaps in several cases. |

### Table C — Human Visual Quality Evidence (`blind_full20_v2`, 80 Blinded Outputs)
* **Local Completed Export Audited (`C:\Users\mutee\Downloads\reviews_mutee.csv`):**
  * **Reviewer:** `mutee` (**1 reviewer only**; Abdullah's independent review was intentionally skipped and is NOT duplicated).
  * **Completeness:** **80 / 80** unique `output_id` rows populated across all 5 score dimensions (`design_score`, `silhouette_score`, `identity_score`, `artifact_score`, `scope_score`) + `reviewed_at`.
  * **Single-Reviewer Exploratory Strict Pass** (`all 5 scores >= 4`, `major_artifact == false`, `scope_violation == false`): **12 / 80 (15.0%)** (`major_artifact=true` on 49/80; `scope_violation=true` on 17/80).
  * **Flagged Score/Flag Contradictions (Preserved Unmodified for Reviewer Clarification):**
    * `BLIND_080`: `scope_score <= 2` with `scope_violation = false`
    * `BLIND_068`: `scope_score <= 2` with `scope_violation = false`
    * `BLIND_060`: `artifact_score >= 4` with `major_artifact = true`
    * `BLIND_015`: `artifact_score >= 4` with `major_artifact = true`
  * **Two-Reviewer Release Gate Status:** **UNMET** (single reviewer only + 15.0% strict pass rate; blind mapping remains private).

### Table D — Live Application Integration Evidence (CPU-Verified Contracts vs. GPU-Blocked)
* **Automated CPU Tests (PASS):** `tests/test_v4_worker_and_proxy.py` (worker cold/ready lifecycle, non-blocking `/health` during busy inference, HTTP 429 queue cap, OOM HTTP 503 recovery, Bearer auth rejection, missing checkpoint refusal, corrupt/empty upload rejection, SSRF upstream protection) and `tests/test_studio_controller.mjs` (request invalidation without stuck loading, late-response isolation, duplicate-click guard, provenance formatting, Gemini disabled by default).
* **Live GPU Worker Execution on Updated `FashionFit_PK_Live_GPU_Server.ipynb`:** **BLOCKED (No Local GPU)** — requires running the regenerated notebook on Colab GPU.

### Table E — Operational Reliability & Security Controls
* **Local Proxy (`backend/main.py`):** Explicitly **local-only**; `POST /api/config/colab` disabled by default (`FF_ALLOW_RUNTIME_UPSTREAM_CONFIG` + `FF_ADMIN_TOKEN` required), strict allowlist (`FF_ALLOWED_UPSTREAM_HOSTS`, default `*.trycloudflare.com`), private/loopback IP blocking, redirects refused (`_NoRedirectHandler`), 10 MB / 12 MP / PIL verify + EXIF transpose + metadata stripping.
* **V4 GPU Worker (`colab_server/v4_worker_core.py`):** Dedicated single-worker `ThreadPoolExecutor(max_workers=1)`, bounded queue (`MAX_QUEUE=2`), separate `load_seconds` / `inference_seconds`, per-model `model_states`, `/auth/verify` Bearer verification, and OOM cache cleanup.

---

## 2. Category-by-Category Status

| Category | Pipeline & Routing | UI & Provenance | Quality Gate Status | Verdict |
|---|---|---|---|---|
| **1. Adult Upper-Body** | Wired (`CatVTON` + V4 / `FASHN`) | Complete | Single-reviewer blind eval only (12/80 overall); 2-reviewer gate unmet | **Experimental** |
| **2. Girl-Child** | Wired (`CatVTON` + V4 / `FASHN`) | Complete | Strong parser pilot (`77.63%` FG mIoU, $N=8$); VTON 2-reviewer gate unmet | **Experimental** |
| **3. Cross-Age (`Adult ↔ Child`)** | Wired (explicit routing + warnings) | Complete | Visual transfer only (not physical sizing); 2-reviewer gate unmet | **Experimental** |
| **4. Wedding Lehengas** | Wired (`flared_lehenga` heuristic; choli+skirt only, no donor dupatta transfer) | Complete | Formal bottom parser regression (`21.98%`); independent adult `lehenga_choli` test set missing | **Experimental / Unvalidated** |
