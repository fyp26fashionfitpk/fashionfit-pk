#!/usr/bin/env bash
set -euo pipefail

echo "[FashionFit-PK] Running Python syntax compilation check..."
python -m py_compile lib/*.py tests/*.py bin/profile_inference.py

echo "[FashionFit-PK] Running automated unit & contract test suite..."
python -m unittest discover -s tests -v

echo "[OK] All tests passed."
