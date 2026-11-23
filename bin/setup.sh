#!/usr/bin/env bash
set -euo pipefail

echo "[FashionFit-PK] Setting up local development environment..."
python3 --version || python --version

if [ ! -f ".env.local" ] && [ -f ".env.example" ]; then
  cp .env.example .env.local
  echo "[OK] Created .env.local from .env.example (fill in your local keys)."
fi

if [ -d "src/frontend" ]; then
  echo "[FashionFit-PK] Installing frontend dependencies..."
  (cd src/frontend && npm install)
fi

if [ -f "src/backend/requirements.txt" ]; then
  echo "[FashionFit-PK] Backend requirements located at src/backend/requirements.txt"
fi

echo "[DONE] Setup complete."
