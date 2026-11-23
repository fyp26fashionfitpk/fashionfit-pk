# FashionFit-PK: Culturally-Aware AI Virtual Try-On for Pakistani Ethnic Wear
## Final Year Project I (FYP I) — Complete Technical & Research Documentation

**Institution:** Information Technology University (ITU), Faculty of Sciences (FoS)  
**Term:** Fall 2026  
**Project Email:** `fyp26_fashionfitpk@gmail.com`  
**Live Cloud Deployment:** [https://fashionfit-pk.vercel.app/](https://fashionfit-pk.vercel.app/)  

---

## 1. Background & Problem Definition

Online fashion retail in Pakistan suffers from high return rates (25–35%) because shoppers cannot visualize how unstitched or prêt ethnic garments — featuring long hemlines, intricate embroidery, flowing dupattas, and flared bridal silhouettes — will drape on their own bodies. Existing state-of-the-art Virtual Try-On (VTON) systems (such as VITON-HD, HR-VITON, IDM-VTON, OOTDiffusion, and base CatVTON) are trained almost exclusively on Western e-commerce datasets (VITON-HD, DressCode) containing T-shirts, jeans, and short dresses.

When applied directly to Pakistani attire, Western VTON pipelines fail in three measurable ways:
1. **Dupatta Occlusion Failure:** Pretrained human parsers (e.g., `mattmdjaga/segformer_b2_clothes`) have no semantic class for `dupatta` (achieving only **3.12% IoU** on Pakistani outfits), causing the generator to erase or hallucinate over draped scarves.
2. **Garment Structure & Hemline Distortion:** Western upper-body models shorten long kurtas/kameez and blur dense South Asian embroidery (baseline CatVTON FID **120.16** on Pakistani held-out outfits).
3. **Domain Gaps in Kids & Flared Wedding Formals:** Tight straight-leg masks squeeze wide bridal lehengas, and adult-only parsers struggle with children's distinct head-to-body proportions (~1:4.5 vs. ~1:7.5 for adults).

---

## 2. Comprehensive Literature Review & Survey (5 Most Relevant Projects)

| # | Project / Paper | Core Architecture | Strengths | Key Limitations on Pakistani Ethnic Wear |
|---|---|---|---|---|
| **1** | **VITON-HD** (Choi et al., CVPR 2021) | TPS Warping + ALIAS Normalization GAN | High-resolution ($1024 \times 768$) upper-body alignment | Explicit warping fails on loose kurtas, heavy folds, and dupatta occlusions. |
| **2** | **HR-VITON** (Lee et al., ECCV 2022) | Unified Warping + Segmentation Generator | Eliminates misalignment artifacts on Western tops | Trained purely on VITON-HD; zero support for multi-piece ethnic layering. |
| **3** | **IDM-VTON** (Choi et al., ECCV 2024) | SDXL + Dual GarmentNet / IP-Adapter | Rich texture preservation and photorealism | Heavy VRAM footprint (>18 GB), slow inference, lacks Pakistani garment priors. |
| **4** | **OOTDiffusion** (Xu et al., AAAI 2025) | Outfitting UNet + Outfitting Fusion | Clean full-body / upper / lower control | Depends on Western 18-class parsing; erases dupattas during upper swaps. |
| **5** | **CatVTON** (Chong et al., ICLR 2025) | Lightweight Concatenation-Based Diffusion UNet | Parameter-efficient (~899M params), fits on 16 GB T4 GPU, no separate garment encoder | Zero-shot baseline suffers high FID (**120.16**) and embroidery degradation on Pakistani attire without domain LoRA adaptation. |

**Synthesis & Research Gap:** CatVTON offers the ideal memory-efficient foundation for real-world deployment on accessible GPUs (T4), but requires (a) a **custom 4-class Pakistani semantic parser** that explicitly isolates `dupatta` and `kurta_kameez`, (b) **parameter-efficient LoRA domain adaptation** on clean Pakistani studio datasets, and (c) **A-line dilated masking** for flared bridal lehengas.

---

## 3. Project Scope & Functional Specifications

### 3.1 In-Scope (FYP I Complete + Research Foundation)
* **Web Application:** Cloud-hosted React/Vite frontend + FastAPI/Vercel serverless backend (`https://fashionfit-pk.vercel.app/`).
* **4-Class Pakistani Garment Parser:** Fine-tuned `SegFormer-B2` separating `other`, `kurta_kameez`, `bottom`, and `dupatta`.
* **Pakistani VTON Generator:** `CatVTON` adapted with LoRA weights fine-tuned on verified Pakistani fashion pairs.
* **Multi-Pose Preview & Garment Prep:** Front, angled, and back view workflows plus automatic garment canonicalization.
* **V3 Extension Pipeline (In Progress):** Kids' ethnic wear (`Peraan Mini`, `Limelight Girls`), cross-age try-on analysis, and flared wedding lehenga masking (`Peraan` formals).

### 3.2 Functional Requirements
| ID | Requirement | Verification |
|---|---|---|
| **FR-01** | User uploads a target person photo and selects/uploads a Pakistani garment image. | Verified in WebApp UI (`TC-01`, `TC-02`). |
| **FR-02** | System parses person image into 4 semantic classes (`other`, `kurta_kameez`, `bottom`, `dupatta`). | Verified via Gold SegFormer-B2 (`TC-07`). |
| **FR-03** | System preserves draped dupattas during kurta/kameez swap (0% false positive erasure). | Verified via `preserve_dupatta_over_mask` (`TC-08`). |
| **FR-04** | System generates photorealistic virtual try-on output via CatVTON + Pakistani LoRA. | Verified via Colab T4 GPU backend (`TC-11`). |
| **FR-05** | System supports A-line flared masking for bridal/wedding lehengas. | Verified via `make_flared_lehenga_mask` (`TC-09`). |
| **FR-06** | Graceful error recovery and health telemetry when GPU backend is offline. | Verified via `/api/health` (`TC-05`, `TC-06`). |

### 3.3 Non-Functional Requirements
* **Performance:** Cloud UI loads in <2 s; GPU try-on inference executes within 12–25 s on a single NVIDIA T4 (16 GB VRAM).
* **Privacy & Ethics:** All training images are sourced strictly from public commercial brand catalogs (never personal photos of minors); user uploads are processed in-memory without persistent storage.
* **Reproducibility:** Strict SKU-level train/test splitting (0% data leakage) and deterministic evaluation scripts.

---

## 4. System Architecture, High-Level & Detailed Design

Detailed Mermaid diagrams and component breakdowns are documented in [ARCHITECTURE.md](./ARCHITECTURE.md).

### 4.1 High-Level Pipeline
1. **Client Layer (Vercel React + Vite):** Handles person/garment upload, outfit catalog selection, try-on mode selection (`upper_body`, `full_outfit`, `flared_lehenga`), and 3-pose preview display.
2. **API & Preprocessing Layer (Vercel Serverless + FastAPI):** Validates payloads, runs `/api/health` diagnostics, canonicalizes reference garments, and routes inference requests to the GPU worker.
3. **Segmentation & Masking Module (`lib/mask_utils.py` + SegFormer-B2):** Generates 4-class pixel maps, subtracts `dupatta` pixels to protect draped scarves, and applies progressive horizontal A-line dilation for flared lehengas.
4. **Generative Try-On Engine (CatVTON + Pakistani LoRA):** Concatenates target person latents, masked person latents, and reference garment latents along the spatial width axis inside the UNet, conditioned via rank-16 LoRA adapters trained on Pakistani ethnic wear.

### 4.2 UI Design
The production UI implements an editorial luxury aesthetic tailored for Pakistani retail:
* **Hero & Studio Workspace:** Two-column studio panel with drag-and-drop person/garment uploaders, preset Pakistani outfits, and real-time processing status.
* **3-Pose Interactive Viewer:** Tabbed Front / Angled / Back inspection panel with zoom and download controls.
* **Live Status Indicators:** Non-blocking health badge confirming API and GPU server availability.

---

## 5. Mathematical Formulations

### 5.1 4-Class Semantic Segmentation Objective (SegFormer-B2)
Given an input image $I \in \mathbb{R}^{H \times W \times 3}$ and ground-truth class map $Y \in \{0,1,2,3\}^{H \times W}$, SegFormer-B2 predicts pixel-wise class probabilities $\hat{P}_{i,c} = \text{softmax}(z_{i,c})$. We optimize class-weighted Cross-Entropy loss to handle the class imbalance of thin dupattas:
$$\mathcal{L}_{\text{seg}} = -\frac{1}{HW} \sum_{i=1}^{HW} \sum_{c=0}^{3} w_c \, \mathbb{1}[Y_i = c] \log \hat{P}_{i,c}$$

Per-class Intersection-over-Union ($\text{IoU}_c$) and mean IoU ($\text{mIoU}$) are computed as:
$$\text{IoU}_c = \frac{|\hat{Y}_c \cap Y_c|}{|\hat{Y}_c \cup Y_c|}, \qquad \text{mIoU} = \frac{1}{4}\sum_{c=0}^{3} \text{IoU}_c$$

### 5.2 Dupatta-Preserving & Flared A-Line Masking
Let $M_{\text{kurta}} = \mathbb{1}[\hat{Y}=1]$, $M_{\text{bottom}} = \mathbb{1}[\hat{Y}=2]$, and $M_{\text{dupatta}} = \mathbb{1}[\hat{Y}=3]$.
* **Dupatta-Preserving Upper Mask:** $M_{\text{upper}} = M_{\text{kurta}} \odot (1 - M_{\text{dupatta}})$
* **Flared Lehenga Mask:** For every row $y \in [y_{\text{waist}}, y_{\text{hem}}]$ with active horizontal span $[x_L(y), x_R(y)]$, we apply linear A-line expansion:
$$\Delta x(y) = \left\lfloor \alpha W \cdot \frac{y - y_{\text{waist}}}{y_{\text{hem}} - y_{\text{waist}}} \right\rfloor, \qquad M_{\text{flared}}(y, [x_L(y)-\Delta x(y),\, x_R(y)+\Delta x(y)]) = 1$$
followed by dupatta protection: $M_{\text{final}} = M_{\text{flared}} \odot (1 - M_{\text{dupatta}})$.

### 5.3 CatVTON Diffusion & LoRA Adaptation
CatVTON concatenates noisy person latents $z_t$ and reference garment latents $z_g$ spatially: $\tilde{z}_t = \text{Concat}_{\text{width}}(z_t, z_g)$. Rather than updating all UNet weights $W_0 \in \mathbb{R}^{d \times k}$, we inject trainable low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ ($r \ll \min(d, k)$) into self-attention projections:
$$W' = W_0 + \frac{\alpha_{\text{lora}}}{r} B A$$
and minimize the masked latent noise-prediction objective:
$$\mathcal{L}_{\text{VTON}} = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0,I), t} \left[ \left\| \epsilon - \epsilon_\theta(\tilde{z}_t, t, M) \right\|_2^2 \right]$$

---

## 6. Experimental Setup & Datasets

### 6.1 Clean Pakistani Adult Dataset (Zero-Leakage)
Raw brand crawls contained 597 `.source.*` duplicate twin files; all duplicates were removed via filename rules, SHA-256 hashing, and perceptual `dHash`, followed by strict SKU-level splitting (0% train/test overlap):
* **Generation (`generation.com.pk`):** 237 unique studio images
* **Ethnc (`ethnc.com`):** 180 unique studio images
* **Limelight (`limelight.pk`):** 180 unique studio images
* **Curated Pilot Validation Set:** 27 images
* **Total Clean Adult Corpus:** **624 unique images across 301 unique SKUs**
* **Gold Human-Verified 4-Class Segmentation Set:** **95 unique outfits** manually verified via Roboflow SAM (44 dupatta-positive, 51 dupatta-negative).

### 6.2 V3 Kids & Wedding Formal Extension Dataset
* **Total Clean V3 Corpus:** **360 unique catalog images across 181 unique SKUs** (Train: 291, Unseen-SKU Test: 62)
  * **Kids Ethnic (180 images):** Peraan Mini (96), Limelight Girls (84)
  * **Wedding Formals / Lehengas (180 images):** Peraan Formals (180)

### 6.3 Hardware & Training Configuration
* **Hardware:** NVIDIA Tesla T4 (16 GB VRAM)
* **SegFormer-B2:** Initialized from `mattmdjaga/segformer_b2_clothes` with reinitialized 4-class decode head; AdamW, 25 epochs.
* **CatVTON LoRA:** 1,000 optimization steps on 930 Pakistani kurta pairs; converged training loss **0.0343**.

---

## 7. Measured Empirical Results

### 7.1 4-Class Pakistani Semantic Segmentation (Held-Out Gold Test Set)

| Semantic Class | Baseline (`segformer_b2_clothes`) | **FashionFit-PK Gold SegFormer-B2** | Absolute Gain |
|---|---:|---:|---:|
| `other` (bg, face, hands, skin) | 95.72% | **96.81%** | +1.09% |
| `kurta_kameez` | 74.31% | **84.29%** | +9.98% |
| `bottom` (shalwar / trousers) | 36.98% | **69.43%** | **+32.45%** |
| `dupatta` | 3.12% | **55.82%** | **+52.70% (17.9x)** |
| **Overall 4-Class mIoU** | **52.53%** | **76.59%** | **+24.06%** |

*Qualitative Verification:* **0% false positives** on 2-piece outfits without dupattas (verified in `research/results/segformer_gold_test_predictions_grid.jpg`).

### 7.2 Virtual Try-On Generation (Held-Out Unseen-SKU Test Set, $N = 50$)

| Metric | Baseline CatVTON (Zero-Shot) | **CatVTON + Pakistani LoRA (Ours)** | Improvement |
|---|---:|---:|---:|
| **FID** ($\downarrow$ lower is better) | 120.16 | **95.28** | **-20.7% (-24.88)** |
| **LPIPS** ($\downarrow$ lower is better) | 0.1869 | **0.1531** | **-18.1% (-0.0338)** |
| **CLIP-I** ($\uparrow$ higher is better) | — | **0.9488** | High semantic fidelity |

---

## 8. Limitations & FYP II Roadmap
1. **Cross-Age Try-On (Adult $\leftrightarrow$ Child):** Diffusion VTON performs visual texture/appearance transfer rather than physical tailoring; extreme age-gap swaps can stretch or compress embroidery scale without age-proportioned fine-tuning.
2. **FYP II Targets:** Complete Universal 3-Domain SegFormer & V3 Multi-Mode LoRA training (Kids + Flared Wedding Lehengas), full ablation tables, scalable dedicated GPU hosting, AI-use transparency report, and research paper submission.

---

## 9. Verification & Test Cases
See [TEST_CASES.md](./TEST_CASES.md) for the complete 20-case test matrix and `tests/` for the automated unit/integration test suite.
