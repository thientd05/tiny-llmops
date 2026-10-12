#!/usr/bin/env bash
# Cài môi trường trên máy L4 thuê (image nvidia/cuda:13.0.2-devel-ubuntu22.04).
# Cài vLLM và tải model chạy SONG SONG để giảm thời gian thuê máy.
#   bash setup_node.sh [OUT_DIR]      -> ghi env.txt, setup_times.txt vào OUT_DIR (mặc định /root/results/setup)
set -euo pipefail

VLLM_VERSION=0.30.0
TORCH_BACKEND=cu130
UV_VERSION=0.12.19
HF_HUB_VERSION=2.0.0
MODEL=Qwen/Qwen3-30B-A3B-GPTQ-Int4
MODEL_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441
MODEL_DIR=/root/models/Qwen3-30B-A3B-GPTQ-Int4
VENV=/root/venv
OUT=${1:-/root/results/setup}

mkdir -p "$OUT"
TIMES="$OUT/setup_times.txt"
: > "$TIMES"
t0=$(date +%s)
mark() { echo "$1 $(( $(date +%s) - t0 ))s" | tee -a "$TIMES"; }

command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl; }

# uv (ghim phiên bản)
if ! command -v uv >/dev/null; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_NO_MODIFY_PATH=1 sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"
mark "uv_ready"

# Tải model (nền) — tách venv riêng qua uvx để không phụ thuộc cài vLLM
(
  uvx --from "huggingface_hub==${HF_HUB_VERSION}" hf download "$MODEL" --revision "$MODEL_REV" \
      --local-dir "$MODEL_DIR" > "$OUT/model_download.log" 2>&1
  echo "model_downloaded $(( $(date +%s) - t0 ))s" >> "$TIMES"
) &
DL_PID=$!

# Cài vLLM
uv venv -q --python 3.12 "$VENV"
uv pip install -q --python "$VENV/bin/python" "vllm==${VLLM_VERSION}" --torch-backend="$TORCH_BACKEND"
mark "vllm_installed"

wait "$DL_PID"
mark "all_done"
du -sh "$MODEL_DIR" | tee -a "$TIMES"

# Thông tin môi trường để tái lập
{
  echo "### date";            date -Is
  echo "### nvidia-smi";      nvidia-smi
  echo "### nvidia-smi -q";   nvidia-smi -q
  echo "### nvcc";            nvcc --version
  echo "### cpu/mem/disk";    nproc; free -g; df -h /
  echo "### uv pip freeze";   uv pip freeze --python "$VENV/bin/python"
  echo "### vllm collect-env"; "$VENV/bin/vllm" collect-env 2>&1 || true
} > "$OUT/env.txt" 2>&1
echo "env -> $OUT/env.txt"
