# AI Usage Transparency, Data Ethics & Scientific Rigor

In alignment with ITU Faculty of Sciences (FoS) FYP guidelines, this document transparently records how AI tools were used across development, testing, and documentation, along with our data privacy and scientific integrity controls.

---

## 1. Transparent Record of AI Assistance
| Project Area | How AI Was Used | Human Verification & Final Ownership |
|---|---|---|
| **Test Case Design** | Drafted initial edge-case matrix (`TC-01`..`TC-20`) and `unittest` scaffolding. | All 20 test cases manually reviewed, executed against the live Vercel app and local codebase (`12/12` automated unit/contract tests passing). |
| **Dataset Annotation** | Evaluated GroundingDINO-tiny (rejected due to 85% dupatta false positives) and used Roboflow SAM / Auto Label for initial polygon proposals. | Human-verified 95 Gold Adult outfits (44 dupatta-positive, 51 dupatta-negative) and visually audited V3 preview masks before dataset export. |
| **Code & DevOps Scaffolding** | Assisted with React UI polish, Vercel serverless failover routes, `Makefile`, shell scripts, and GitHub Actions CI. | All code reviewed, executed, debugged on live T4 GPU / Vercel, and verified via CI and browser E2E QA. |
| **Documentation & Diagrams** | Assisted with structuring Markdown/LaTeX tables and Mermaid architecture diagrams. | Every metric in all tables is 100% empirically measured from actual Colab T4 evaluation runs (zero synthetic or estimated figures). |

---

## 2. Data Ethics & Minor Privacy Policy
1. **Commercial Catalogs Only:** All adult and kids' images originate exclusively from public Pakistani fashion brand catalogs (Generation, Ethnc, Limelight, Peraan). Zero private social media or personal photos of minors are collected or stored.
2. **In-Memory Inference:** User-uploaded images in the live web application are processed transiently in memory for try-on generation and are never persisted to a database or reused for training.
3. **Zero-Leakage Scientific Integrity:** Raw crawls were audited for duplicate `.source.*` twin files (597 removed via SHA-256 and perceptual `dHash`), and all train/test splits are strictly disjoint at the outfit SKU level (`validate_zero_leakage_sku_split()`).
