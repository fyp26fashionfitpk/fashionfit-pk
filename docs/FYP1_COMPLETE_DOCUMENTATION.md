# FashionFit-PK: FYP I research and implementation record

Updated 28 November 2026. This document describes verified offline evidence and the implemented research prototype. Live v4.2 GPU integration, the new masking experiment and full adult bridal evaluation remain pending execution/evaluation.

## Background and problem
Pakistani clothing includes long kurtas, trousers/shalwar, overlapping dupattas and elaborate formal components. Catalog references are often model-worn photographs, not independent flat-lays. We study which segmentation and reference-processing choices help visual garment transfer. We do not assume all existing models fail or that fine-tuning is always necessary. No unsupported national return-rate estimate is used to justify the project.

## Questions and scope
1. Does an adapted four-class parser improve held-out segmentation across adult, child and formal clothing?
2. How do baseline/adapted target masks affect CatVTON outputs?
3. How do original model photographs versus visible-fabric cutouts affect FASHN outputs?
4. Does FASHN's native masked mode improve edit scope without harming fidelity or recipient identity?
5. What remains unsupported for adult lehenga/choli and dupatta transfer?

Advisor-required categories are adult tops, girl-child ethnic clothing, both cross-age directions and adult bridal components. Incomplete bridal evaluation is a pending required component, not silently deferred to FYP II. Physical sizing, hidden-fabric recovery and trustworthy unseen back views are outside current evidence.

## Related work and implementation references
The survey must distinguish each published paper from its code and any subsequent models. Relevant sources are VITON-HD (https://github.com/shadow2496/VITON-HD), HR-VITON (https://github.com/sangyun884/HR-VITON), IDM-VTON (https://github.com/yisol/IDM-VTON), CatVTON (https://github.com/Zheng-Chong/CatVTON) and SegFormer (https://github.com/NVlabs/SegFormer). FASHN v1.5 is an implemented comparison (https://github.com/fashn-AI/fashn-vton-1.5); do not label it a published conference paper without publication verification. These pointers require final publication/bibliography checks before submission. Broad architectural differences do not establish failure on our dataset without experiments.

## Requirements and design
The React/Vite studio accepts person and garment images, selects supported model/input combinations, displays processing state, renders returned results and downloads the completed run's image. A temporary authenticated Colab worker provides segmentation and try-on. The browser may connect directly to the worker; the local proxy is an alternative. Gemini remains optional and disabled by default.

Segmentation classes: 0 other, 1 kurta_kameez/choli, 2 bottom, 3 dupatta; ignore=255. Annotate visible fabric only. Sarees are not silently forced into these classes. Recipient dupatta preservation differs from donor dupatta transfer.

CatVTON uses an explicit target edit mask and a product-style reference or clearly identified visible-fabric pseudo-reference. No Pakistani CatVTON LoRA improvement is established by the current verified runs. FASHN uses its own parsing and either model-photo or flat-lay input handling. The v2 03D experiment changes only its segmentation_free option while holding the matched inputs/settings fixed. It does not use V4 masks as FASHN conditioning.

The GPU worker bounds admission, owns running slots until execution ends, verifies authentication, loads models lazily and evicts the competing VTON model. A timeout does not cancel GPU execution. The live notebook configures 1800 seconds for the worker's load/execution wait and 1860 seconds for smoke clients; these are test budgets, not latency promises. Colab/tunnel hosting is temporary.

## Data and experimental setup
The current reviewed segmentation set contains 95 adult and 97 accepted child/formal images after exclusion of one incorrectly mapped saree. Frozen assignment: 133 train, 29 validation, 30 test (adult 15, child 8, formal 7). Catalog identities/subjects can recur; SKU grouping is not proof of independent-person testing. Candidate counts and reviewed ground truth are distinct.

The adapted B2 parser starts from the existing four-class Gold model. Validation selected epoch 11. Geometry differs between Gold and adapted evaluation; the system comparison does not isolate fine-tuning alone. No new training is launched merely to improve previously inspected test scores.

Twenty original VTON case pairs were evaluated under four conditions (80 outputs), with recipient reuse. The new 03D v2 protocol binds complete case definitions, image hashes, settings and runtime identity before inference. It generates both conditions in a separate folder. Existing exposed cases provide exploratory evidence, not a fresh confirmatory holdout.

## Metrics
For a confusion matrix C with truth rows and prediction columns, IoU_c=C_cc/(sum_j C_cj+sum_i C_ic-C_cc). Foreground pooled mIoU is the average over the supported foreground classes under the recorded evaluation convention; absent support and false positives must be reported explicitly. It is not visual try-on accuracy.

The parser training loss and optimizer settings are defined in the actual training configuration/logs; no unverified weighted-loss or epoch claim is substituted here. A top edit mask includes predicted class 1 and excludes protected/ignore pixels. Flared masks are geometric heuristics, not learned bridal specialization or proof of correct fabric physics.

Visual acceptance in the existing study requires all five ordinal ratings >=4 and both major_artifact and scope_violation false. The completed current evaluation has one reviewer. The original two-reviewer gate remains unmet. Wrong-garment transfer can be a major failure even when pixel artifacts look clean.

## Verified offline results
| Test domain | N | Baseline FG mIoU | Gold | Adapted B2 |
|---|---:|---:|---:|---:|
| Adult | 15 | 34.64% | 69.99% | 69.86% |
| Child | 8 | 43.31% | 54.00% | 77.63% |
| Formal | 7 | 23.93% | 54.61% | 55.96% |
| Pooled | 30 | 36.51% | 64.48% | 72.64% |
Adult bottom and formal bottom regressions remain: 69.10 to 59.74%, and 30.80 to 21.98%. No blanket zero-false-positive claim is made.

| VTON condition | N | Single-reviewer strict passes | Major-artifact flags | Scope violations |
|---|---:|---:|---:|---:|
| CatVTON baseline mask | 20 | 1 | 15 | 3 |
| CatVTON adapted mask | 20 | 2 | 12 | 0 |
| FASHN cutout | 20 | 3 | 15 | 4 |
| FASHN original photo | 20 | 6 | 7 | 10 |
Pooled acceptance: 12/80. These are exploratory visual ratings, not general accuracy. Previous generation times were about 63-66 seconds for CatVTON and several minutes for FASHN on T4; cold loading adds time. No 12-25 second performance guarantee is supported.

## Verification and operational limits
See TEST_CASES.md and the current test receipt for automated checks. A CPU test double is not real GPU inference. A jsdom anchor-click test is not proof of a downloaded file in the deployed browser. Runtime smoke evidence must include real returned images and their provenance. The website-to-tunnel path, child/cross-age live runs and full bridal evaluation require separate verification.

## Pending required work and future work
Execute live smoke tests; inspect the native FASHN masking comparison; complete adult separate lehenga/choli evaluation with suitable references and reviewed components; reconcile human clarifications; complete student manual test review and university access checks. Production scalability, persistent job orchestration, broader held-out confirmation and potential 3D try-on are future work. 3D feasibility/novelty for Pakistani attire has not been established here.

Legacy SD1.5-inpainting LoRA numbers are quarantined and do not establish CatVTON adaptation. Archived documents in legacy_unverified preserve history, not current validated claims.
