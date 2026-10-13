#!/usr/bin/env bash
# Chạy trên MÁY THUÊ, trong tmux: nhiều cấu hình nối tiếp trong một phiên (P08, phần tune GĐ1).
#   bash scripts/remote_multi.sh LIST_FILE        (trong thư mục repo, ví dụ /root/td)
# LIST_FILE: mỗi dòng "<loại> <cấu hình> <thư mục kết quả>", loại = bench | quality; dòng trống / bắt đầu bằng # bỏ qua.
#   bench   → bench/sweep.py + bench/summarize.py   (như scripts/remote_bench.sh)
#   quality → quality/run.py                       (như quality/remote_quality.sh, venv lm-eval cài khi cần)
# Mỗi cấu hình tự chạy tiếp được (done.json) → dừng ở đâu, chạy lại cùng lệnh là đi tiếp (docs/thi-nghiem-chay-tiep.md).
# Đọc LIST_FILE lại sau mỗi dòng → có thể nối thêm dòng vào cuối file trong lúc đang chạy.
set -euo pipefail
LIST=${1:?cần LIST_FILE}
LMEVAL_VERSION=0.4.13
TRANSFORMERS_VERSION=5.19.0
LMEVAL_VENV=${MULTI_LMEVAL_VENV:-/root/lmeval-venv}
PY=${MULTI_PY:-/root/venv/bin/python}           # thử local với server giả: MULTI_PY=.venv/bin/python MULTI_SKIP_SETUP=1
SETUP_OUT=${MULTI_SETUP_OUT:-/root/results/setup-$(date +%Y%m%d)}
mkdir -p "$SETUP_OUT"
[ -n "${MULTI_SKIP_SETUP:-}" ] || bash scripts/setup_node.sh "$SETUP_OUT" 2>&1 | tee -a "$SETUP_OUT/setup.out"
export PATH="$HOME/.local/bin:/root/venv/bin:$PATH"

ensure_lmeval() {
  local have
  have=$("$LMEVAL_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("lm_eval"))' 2>/dev/null || true)
  [ -n "${MULTI_SKIP_SETUP:-}" ] && return 0
  if [ "$have" != "$LMEVAL_VERSION" ]; then
    [ -x "$LMEVAL_VENV/bin/python" ] || uv venv -q --python 3.12 "$LMEVAL_VENV"
    uv pip install -q --python "$LMEVAL_VENV/bin/python" "lm-eval[api,ifeval]==${LMEVAL_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" 2>&1 | tee -a "$SETUP_OUT/setup.out"
  fi
}

i=0
while true; do
  i=$((i + 1))
  line=$(grep -v '^\s*\(#\|$\)' "$LIST" | sed -n "${i}p" || true)
  [ -z "$line" ] && break
  read -r kind cfg out <<<"$line"
  mkdir -p "$out"
  echo "MULTI_STEP $i $kind $cfg $out $(date -Is)" | tee -a "$out/multi.log"
  case "$kind" in
    bench)
      "$PY" bench/sweep.py --config "$cfg" --out "$out" 2>&1 | tee -a "$out/sweep.log"
      "$PY" bench/summarize.py --config "$cfg" --out "$out" --quiet || true
      ;;
    quality)
      ensure_lmeval
      uv pip freeze --python "$LMEVAL_VENV/bin/python" > "$out/lmeval-freeze.txt" 2>/dev/null || true
      "$PY" quality/run.py --config "$cfg" --out "$out" 2>&1 | tee -a "$out/run.log"
      ;;
    *) echo "loại không biết: $kind"; exit 2 ;;
  esac
  echo "MULTI_STEP_DONE $i $kind $cfg $(date -Is)" | tee -a "$out/multi.log"
done
echo "REMOTE_MULTI_DONE $(date -Is)"
