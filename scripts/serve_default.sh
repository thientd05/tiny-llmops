#!/usr/bin/env bash
# Chạy vLLM cấu hình MẶC ĐỊNH (định nghĩa: docs/gd1-vllm-baseline/README.md).
# Cờ thêm (chỉ khi bắt buộc, có ghi lý do) truyền qua tham số:
#   bash serve_default.sh LOG_FILE [cờ thêm...]
# Chờ /health OK rồi in thời gian khởi động; server tiếp tục chạy nền.
# Chạy lại được nhiều lần: server đang chạy với đúng cờ thì dùng lại; khác cờ thì tắt server cũ, khởi động server mới.
set -euo pipefail

VENV=/root/venv
MODEL_DIR=/root/models/Qwen3-30B-A3B-GPTQ-Int4
LOG=${1:?cần LOG_FILE}; shift || true
PORT=8000
# FlashInfer biên dịch kernel lúc chạy (JIT), cần `ninja` trong venv có trên PATH
export PATH="$VENV/bin:$PATH"

STATE=/root/vllm_server.state     # dòng 1: pid, dòng 2: cờ thêm
if [ -f "$STATE" ] && curl -sf "localhost:$PORT/health" >/dev/null; then
  old_pid=$(sed -n 1p "$STATE"); old_args=$(sed -n 2p "$STATE")
  if [ "$old_args" = "$*" ]; then
    echo "server đang chạy (pid=$old_pid) với đúng cờ '$*' -> dùng lại"; exit 0
  fi
  echo "server đang chạy với cờ khác ('$old_args') -> tắt"
  kill "$old_pid" 2>/dev/null || true
  while kill -0 "$old_pid" 2>/dev/null; do sleep 1; done
fi

# VLLM_SERVER_DEV_MODE=1 chỉ mở thêm /reset_prefix_cache, không đổi tốc độ
t0=$(date +%s)
VLLM_SERVER_DEV_MODE=1 nohup "$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name Qwen/Qwen3-30B-A3B-GPTQ-Int4 --port "$PORT" "$@" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "${LOG%.log}.pid"
printf '%s\n%s\n' "$PID" "$*" > "$STATE"
echo "vllm serve pid=$PID, log=$LOG, extra args: $*"

until curl -sf "localhost:$PORT/health" >/dev/null; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "SERVER CHẾT khi khởi động — 40 dòng cuối log:"; tail -40 "$LOG"; exit 1
  fi
  sleep 2
done
echo "ready_after $(( $(date +%s) - t0 ))s"
nvidia-smi --query-gpu=memory.used,memory.total,power.draw,clocks.sm --format=csv
