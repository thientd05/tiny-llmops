#!/usr/bin/env bash
# Chạy trên MÁY THUÊ, trong tmux: setup (bỏ qua bước đã xong) → venv lm-eval (ghim) → quality/run.py (chạy / chạy tiếp).
#   bash quality/remote_quality.sh CONFIG OUT_DIR        (trong thư mục repo, ví dụ /root/td)
# Chạy lại cùng lệnh sau khi bị dừng → chỉ làm phần còn dở (docs/thi-nghiem-chay-tiep.md).
set -euo pipefail
CFG=${1:?cần CONFIG}; OUT=${2:?cần OUT_DIR}
LMEVAL_VERSION=0.4.13
TRANSFORMERS_VERSION=5.19.0       # cùng bản với venv vLLM (env.txt P05)
LMEVAL_VENV=/root/lmeval-venv
mkdir -p "$OUT/setup"
bash scripts/setup_node.sh "$OUT/setup" 2>&1 | tee -a "$OUT/setup/setup.out"
export PATH="$HOME/.local/bin:/root/venv/bin:$PATH"
have=$("$LMEVAL_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("lm_eval"))' 2>/dev/null || true)
if [ "$have" != "$LMEVAL_VERSION" ]; then
  [ -x "$LMEVAL_VENV/bin/python" ] || uv venv -q --python 3.12 "$LMEVAL_VENV"
  uv pip install -q --python "$LMEVAL_VENV/bin/python" "lm-eval[api,ifeval]==${LMEVAL_VERSION}" \
      "transformers==${TRANSFORMERS_VERSION}" 2>&1 | tee -a "$OUT/setup/setup.out"
fi
uv pip freeze --python "$LMEVAL_VENV/bin/python" > "$OUT/setup/lmeval-freeze.txt"
/root/venv/bin/python quality/run.py --config "$CFG" --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
echo "REMOTE_QUALITY_DONE $(date -Is)" | tee -a "$OUT/run.log"
