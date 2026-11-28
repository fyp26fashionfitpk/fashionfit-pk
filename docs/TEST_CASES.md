# FashionFit-PK — Test Cases & Verification Matrix

> **Manual Student Review Status:** **PENDING MANUAL STUDENT REVIEW**  
> Per ITU FoS guidelines, AI-assisted test cases must be reviewed and finalized manually by the students before FYP I submission. Do not mark manual review completed until Mutee & Abdullah sign off below.

## 1. Automated CPU Software & Component Tests (Executed in CI)

| Test Suite | Command | What It Verifies | Status |
|---|---|---|---|
| **V4 Worker & Local Proxy (`tests/test_v4_worker_and_proxy.py`)** | `python -m unittest discover -s tests -v` | Timeout slot ownership, queued cancel, double-OOM state reset + reload, single-VTON residency, combination capability rejections, byte/pixel/GIF boundaries, manifest match/mismatch, proxy admin auth & redirect refusal | **Automated PASS (CPU test doubles — NOT real GPU)** |
| **03D FASHN Runner & ZipSlip (`tests/test_fashn_segfree_runner.py`)** | `python -m unittest discover -s tests -v` | Atomic per-case sidecars, partial-failure metadata retention, verified-only resume, corrupt-output flagging, protocol-drift refusal, `Path.relative_to()` traversal & sibling-prefix ZipSlip rejection | **Automated PASS (CPU software test)** |
| **Studio Request Orchestrator (`tests/test_studio_controller.mjs`)** | `node --test tests/test_studio_controller.mjs` | React Strict Mode cycle, mid-flight invalidation, late success/error isolation, synchronous double-click block, unmount abort, capability validation, provenance formatting, Gemini disabled | **Automated PASS** |
| **Rendered React Component (`src/frontend/src/pages/StudioPage.test.jsx`)** | `cd src/frontend && npm test` | Rendered `<React.StrictMode><StudioPage /></React.StrictMode>` with deferred `fetch`: Strict Mode submission + download, mid-flight model change invalidation, mid-flight segmentation parser switch invalidation | **Automated PASS (jsdom)** |
| **Core Masking & Metrics (`tests/test_mask_utils.py`, `test_metrics.py`, `test_api_contracts.py`)** | `python -m unittest discover -s tests -v` | 4-class dupatta exclusion, flared A-line expansion math, mIoU math, SKU split disjointness, `.env.local` absence | **Automated PASS** |

## 2. Pending Real GPU & Manual Student Verification Checklist

| Case ID | Category | Description | Current Status |
|---|---|---|---|
| **GPU-01** | Live Worker | Execute regenerated `FashionFit_PK_Live_GPU_Server.ipynb` on Colab T4/L4 GPU and verify `FashionFit_Advisor_V4_Code_v4_2.zip` + `universal_b2` manifest hashes | **Awaiting Colab GPU** |
| **GPU-02** | Live Inference | Run real `/segment`, `catvton` `/vton`, and `fashn_vton_1_5` `/vton` smoke workflow + concurrent `/health` probe | **Awaiting Colab GPU** |
| **GPU-03** | Quality Exp | Run `03D_FASHN_SegmentationFree_Control_v2.ipynb` (`SMOKE_ONLY=True` 4 cases) on Colab GPU | **Awaiting Colab GPU** |
| **MAN-01** | Human Review | Mutee clarifies `BLIND_068`, `BLIND_080`, `BLIND_060`, `BLIND_015` from private clarification sheet | **Pending Reviewer** |
| **MAN-02** | Bridal Scope | Collect & review independent adult separate `lehenga_choli` donor references for bridal evaluation | **Pending Data/Review** |
| **MAN-03** | FoS Compliance | Students manually review & sign off this test-case matrix | **Pending Student Sign-off** |
