#!/usr/bin/env bash
# Single-container entrypoint: seed → FastAPI (internal :8000) → Streamlit (public $PORT).
# If either process dies the container exits, so the host restarts it cleanly.
set -euo pipefail
cd "$(dirname "$0")/.."

python deploy/bootstrap.py

# A free Space exposes one port. AEGIS_PUBLIC_UI picks what lives on it:
#   dashboard (default) -> Streamlit operator dashboard; the API stays internal
#   console             -> FastAPI itself: HTML console at "/", REST API, webhooks
if [ "${AEGIS_PUBLIC_UI:-dashboard}" = "console" ]; then
  exec uvicorn backend.api.main:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1 --proxy-headers
fi

# One worker on purpose: SQLite + the in-process per-customer spend lock want a single process.
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers &

for _ in $(seq 1 90); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" 2>/dev/null; then
    break
  fi
  sleep 1
done

# XSRF off: the hosted app is often embedded in an iframe (HF Spaces), where Streamlit's
# XSRF cookie breaks file uploads. Every state-changing API call still needs a JWT.
streamlit run frontend/app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableXsrfProtection false \
  --server.maxUploadSize "${MAX_UPLOAD_MB:-50}" \
  --browser.gatherUsageStats false &

wait -n
exit 1
