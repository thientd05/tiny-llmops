#!/usr/bin/env bash
# Tắt server vLLM do scripts/serve_default.sh bật (đọc pid từ file trạng thái), chờ tới khi GPU được giải phóng.
set -uo pipefail
STATE=/root/vllm_server.state
[ -f "$STATE" ] || { echo "không có $STATE"; exit 0; }
PID=$(sed -n 1p "$STATE")
kill "$PID" 2>/dev/null || true
for _ in $(seq 1 120); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -9 "$PID" 2>/dev/null || true
rm -f "$STATE"
# chờ tiến trình con (EngineCore) trả bộ nhớ GPU
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$used" ] || [ "$used" -lt 1000 ] && break
  sleep 1
done
echo "server đã tắt (pid $PID)"
