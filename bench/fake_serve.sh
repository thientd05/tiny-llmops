#!/usr/bin/env bash
# Bật / tắt server giả (bench/fake_server.py) theo cùng giao diện với scripts/serve_default.sh — chỉ để thử harness ở local.
#   bash bench/fake_serve.sh LOG_FILE [cờ cho fake_server.py...]   → chờ /health OK; đang chạy đúng cờ thì dùng lại
#   bash bench/fake_serve.sh --stop
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
PY=${FAKE_PY:-$DIR/../.venv/bin/python}
PORT=${FAKE_PORT:-8123}
STATE=${TMPDIR:-/tmp}/fake_vllm_server.state

if [ "${1:-}" = "--stop" ]; then
  [ -f "$STATE" ] && kill "$(sed -n 1p "$STATE")" 2>/dev/null || true
  rm -f "$STATE"; exit 0
fi
LOG=${1:?cần LOG_FILE}; shift || true
if [ -f "$STATE" ] && curl -sf "localhost:$PORT/health" >/dev/null; then
  if [ "$(sed -n 2p "$STATE")" = "$*" ]; then echo "server đang chạy với đúng cờ '$*' -> dùng lại"; exit 0; fi
  kill "$(sed -n 1p "$STATE")" 2>/dev/null || true; sleep 1
fi
nohup "$PY" "$DIR/fake_server.py" --port "$PORT" "$@" > "$LOG" 2>&1 &
PID=$!
printf '%s\n%s\n' "$PID" "$*" > "$STATE"
t0=$(date +%s)
until curl -sf "localhost:$PORT/health" >/dev/null; do
  kill -0 "$PID" 2>/dev/null || { echo "SERVER CHẾT"; tail -20 "$LOG"; exit 1; }
  sleep 0.2
done
echo "ready_after $(( $(date +%s) - t0 ))s"
