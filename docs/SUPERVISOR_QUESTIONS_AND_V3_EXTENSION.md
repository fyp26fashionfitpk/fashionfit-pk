# Supervisor Research Questions & V3 Extension Plan

This document records the four targeted technical questions raised by our FYP advisor and documents our exact empirical answers, architectural solutions, and ongoing V3 implementation status.

---

## 1. Which Websites Was the Dataset Gathered From?
* **Core Clean Adult Pakistani Dataset (624 unique studio images, 301 unique SKUs, 0% leakage):**
  * **Generation** (`generation.com.pk`) — 237 images
  * **Ethnc** (`ethnc.com`) — 180 images
  * **Limelight** (`limelight.pk`) — 180 images
  * **Curated Pilot Validation Set** — 27 images
* **V3 Kids & Wedding Formal Extension Dataset (360 unique catalog images, 181 unique SKUs):**
  * **Peraan Mini** (Kids Ethnic) — 96 images
  * **Limelight Girls** (Kids Ethnic) — 84 images
  * **Peraan Formals** (Wedding Formals / Lehengas) — 180 images

---

## 2. Can We Extend It for Kids Too? (Feasibility & Minor Privacy)
* **Verdict:** **Yes — 100% feasible with the exact same 4-class SegFormer-B2 + CatVTON LoRA architecture** (no architectural redesign needed).
* **Why:** Both models are resolution-normalized vision backbones that learn garment semantics (`kurta_kameez`, `bottom`, `dupatta`) regardless of subject age. Adding **180 kids' ethnic wear catalog images** adapts the parser and LoRA to children's distinct body proportions (head-to-body ratio ~1:4.5 vs. adults ~1:7.5, shorter hemlines/sleeves).
* **Academic & Minor Privacy Safeguard:** All kids' training images come **strictly from public commercial retail catalogs** (never scraped personal photos of minors). Live user uploads are processed strictly in-memory with zero persistent storage.

---

## 3. Cross-Age Try-On (`Adult Outfit ↔ Child`): Can Our Current Model Do This?
* **Honest Technical Verdict:** **Partially — it performs visual appearance transfer, NOT true physical tailoring/sizing.**
* **What works right now:** Because CatVTON warps the reference garment into the target person's segmented garment mask, it transfers fabric color, print, and motif style between an adult and a child.
* **Known Physical Limitations:** Putting an adult's long embroidered kameez onto a small child compresses fine embroidery; putting a small child's frock onto an adult stretches proportions. Our V3 Evaluation Suite (`research/v3_kids_formal_notebooks/04_V3_Supervisor_Proof_Suite_CrossAge_and_Lehenga.ipynb`) explicitly benchmarks this cross-age matrix (`Adult → Child` and `Child → Adult`) to document both strengths and honest failure boundaries.

---

## 4. Can It Swap Wide, Flared Wedding Lehengas?
* **Current Capability:** Swapping the upper shirt/choli works out-of-the-box with our 4-class parser.
* **Full Flared Lehenga Solution (Implemented in `lib/mask_utils.py`):** Standard tight pants/shalwar masks squeeze wide bridal lehengas into straight columns. We implemented **`make_flared_lehenga_mask()`**, which applies progressive horizontal A-line dilation from waist to hem while subtracting `dupatta` pixels so draped scarves remain intact, paired with **180 Peraan wedding formal outfits** for LoRA bridal embroidery adaptation.
