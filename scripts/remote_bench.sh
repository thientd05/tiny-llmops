#!/usr/bin/env bash
# Chạy trên MÁY THUÊ, trong tmux: setup (bỏ qua bước đã xong) → sweep (chạy / chạy tiếp) → summarize.
# Server do sweep.py tự bật/tắt theo cấu hình (scripts/serve_default.sh, scripts/stop_server.sh).
#   bash scripts/remote_bench.sh CONFIG OUT_DIR        (chạy trong thư mục repo, ví dụ /root/td)
# Chạy lại cùng lệnh sau khi bị dừng → chỉ làm phần còn dở (docs/thi-nghiem-chay-tiep.md).
set -euo pipefail
CFG=${1:?cần CONFIG}; OUT=${2:?cần OUT_DIR}
mkdir -p "$OUT/setup"
bash scripts/setup_node.sh "$OUT/setup" 2>&1 | tee -a "$OUT/setup/setup.out"
export PATH=/root/venv/bin:$PATH
/root/venv/bin/python bench/sweep.py --config "$CFG" --out "$OUT" 2>&1 | tee -a "$OUT/sweep.log"
/root/venv/bin/python bench/summarize.py --config "$CFG" --out "$OUT" --compare-clients --quiet
echo "REMOTE_BENCH_DONE $(date -Is)" | tee -a "$OUT/sweep.log"
