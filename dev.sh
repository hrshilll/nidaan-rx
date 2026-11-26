#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "[ERROR] Virtualenv Python not found at $ROOT_DIR/.venv/bin/python"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[ERROR] npm not found in PATH."
  exit 1
fi

for PORT in 8000 3000; do
  PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$PIDS" ]]; then
    if [[ "${1:-}" == "--clean" ]]; then
      echo "[INFO] Stopping existing process(es) on port $PORT: $PIDS"
      kill $PIDS 2>/dev/null || true
      sleep 1
    else
      echo "[ERROR] Port $PORT is already in use (PID: $PIDS)."
      echo "        Stop the old terminal or run: ./dev.sh --clean"
      exit 1
    fi
  fi
done

BACK_PID=""
FRONT_PID=""

cleanup() {
  trap - INT TERM EXIT
  echo ""
  echo "Shutting down Nidaan-Rx servers..."
  [[ -n "$FRONT_PID" ]] && kill "$FRONT_PID" 2>/dev/null || true
  [[ -n "$BACK_PID" ]] && kill "$BACK_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "Stopped cleanly."
}
trap cleanup INT TERM EXIT

echo "=================================================="
echo "  Starting Nidaan-Rx v2.0 Full-Stack Workspace"
echo "  Backend API : http://localhost:8000"
echo "  OpenAPI Docs: http://localhost:8000/docs"
echo "  Frontend UI : http://localhost:3000"
echo "  Press Ctrl+C to stop both servers."
echo "=================================================="

PYTHONPATH="$ROOT_DIR/backend" "$ROOT_DIR/.venv/bin/python" -m uvicorn app.main:app --reload --port 8000 &
BACK_PID=$!

npm --prefix "$ROOT_DIR/frontend" run dev &
FRONT_PID=$!

wait "$BACK_PID" "$FRONT_PID"
