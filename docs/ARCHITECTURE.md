# System Architecture & Detailed Design — FashionFit-PK

## 1. High-Level System Architecture

```mermaid
flowchart LR
    U["User Browser"] --> FE["React + Vite Frontend\n(Vercel Cloud)"]
    FE --> API["Serverless API Gateway\n(/api/health, /api/gemini-*)"]
    API --> BE["FastAPI Inference Service\n(Colab T4 GPU / Backend)"]
    BE --> SEG["Gold SegFormer-B2\n(4-Class Pakistani Parser)"]
    SEG --> MASK["Mask Engine (lib/mask_utils.py)\nDupatta Preservation + Flared A-Line"]
    MASK --> VTON["CatVTON + Pakistani LoRA\n(Diffusion Try-On Generator)"]
    VTON --> BE
    BE --> FE
```

## 2. Detailed ML Inference Pipeline

```mermaid
flowchart TD
    P["Person Photo (RGB)"] --> S["SegFormer-B2 4-Class Parser"]
    S --> C0["Class 0: other (face, hands, bg)"]
    S --> C1["Class 1: kurta_kameez"]
    S --> C2["Class 2: bottom (shalwar / lehenga)"]
    S --> C3["Class 3: dupatta"]

    C1 --> M["Mask Builder (mode: upper / full / flared_lehenga)"]
    C2 --> M
    C3 --> SUB["Subtract Dupatta Layer\n(Preserve Draped Scarf)"]
    M --> SUB

    G["Target Pakistani Garment"] --> CAN["Garment Canonicalizer"]
    SUB --> CAT["CatVTON Spatial Concatenation\n+ Rank-16 Pakistani LoRA"]
    P --> CAT
    CAN --> CAT
    CAT --> OUT["Final Try-On Output (RGB)"]
```

## 3. Repository Directory Architecture (Industry Standard)

| Directory | Responsibility |
|---|---|
| `src/` | Production full-stack source (`frontend/`, `backend/`, `api/`, `colab_server/`) |
| `lib/` | Reusable Python ML core (`mask_utils.py`, `metrics.py`) |
| `bin/` | Executable shell & profiling scripts (MIT Missing Semester) |
| `docs/` | Formal FYP I documentation, architecture, test cases, defense PPTX |
| `tests/` | Automated unit and contract tests (`unittest`) |
| `research/` | Reproducible FYP2 & V3 training/evaluation notebooks and empirical artifacts |
| `.github/workflows/` | Automated Continuous Integration (`ci.yml`) |
