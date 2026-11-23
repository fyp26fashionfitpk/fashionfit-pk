# MIT "The Missing Semester" — Applied & Live Demo Guide

Per ITU FoS FYP Committee guidelines, core engineering practices from [missing.csail.mit.edu](https://missing.csail.mit.edu/) are directly applied across this repository and ready for live demonstration during the FYP I defense.

| Missing Semester Topic | How It Is Applied in `fyp26-fashionfit-pk` | Live 30-Second Demo Command |
|---|---|---|
| **1. Shell Tools & Scripting** | Automated executable scripts in `bin/` (`setup.sh`, `run_local.sh`, `run_tests.sh`, `check_health.sh`) using `set -euo pipefail`, exit codes, and CLI flags. | `bash bin/check_health.sh` |
| **2. Version Control (Git)** | Isolated repository ownership (`fyp26fashionfitpk`), strict `.gitignore`, semantic commit history, annotated release tag `v1.0.0-fyp1`, zero secrets in history. | `git log --oneline --decorate` |
| **3. Metaprogramming & Build Systems** | Root `Makefile` (`make test`, `make lint`, `make profile`, `make verify`) + **GitHub Actions CI** (`.github/workflows/ci.yml`) auto-running checks on every push. | `make verify` |
| **4. Debugging & Profiling** | `bin/profile_inference.py` profiles 4-class mask construction & flared A-line dilation latency and peak memory using `time.perf_counter` and `tracemalloc`. | `python bin/profile_inference.py` |
| **5. Security & Cryptography** | Isolated per-repo Git identity & SSH key / token workflow (zero conflict with global credentials), `.env.example` template with `.env.local` gitignored. | `git config --local --list` |
| **6. Data Wrangling** | Zero-leakage dataset deduplication via SHA-256 + perceptual `dHash` and SKU-level disjoint validation (`lib/metrics.py`). | `python -m unittest discover -s tests -v` |
