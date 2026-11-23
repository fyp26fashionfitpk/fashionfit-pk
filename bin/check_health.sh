#!/usr/bin/env bash
set -euo pipefail

TARGET_URL="${1:-https://fashionfit-pk.vercel.app/api/health}"
echo "[FashionFit-PK] Checking cloud deployment health: ${TARGET_URL}"

HTTP_CODE=$(curl -sS -o /tmp/fashionfit_health.json -w "%{http_code}" "${TARGET_URL}")
echo "HTTP Status: ${HTTP_CODE}"
cat /tmp/fashionfit_health.json
echo ""

if [ "${HTTP_CODE}" -ne 200 ]; then
  echo "[ERROR] Health check failed with status ${HTTP_CODE}"
  exit 1
fi
echo "[OK] Cloud deployment healthy."
