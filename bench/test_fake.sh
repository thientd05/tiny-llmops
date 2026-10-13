#!/usr/bin/env bash
# Thử harness ở local với server giả (không GPU). Dùng `vllm bench serve` 0.30.0 thật (cài bản CPU ở
# ~/.local/share/vllm-bench-venv, chỉ để gửi tải) → kiểm luôn cờ, định dạng dataset, file kết quả.
#   bash bench/test_fake.sh OUT_DIR [case...]     case: k100 k5 k40 resume config closed scenario w4 multi (mặc định: tất cả)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?cần OUT_DIR}; shift
CASES=${*:-k100 k5 k40 resume config closed scenario w4 multi}
PY=.venv/bin/python
export TMPDIR=${TMPDIR:-$OUT}
mkdir -p "$OUT"

goodput() {  # in dòng goodput của --status
  FAKE_K=$1 $PY bench/sweep.py --config "$2" --out "$3" --status | grep -E "^w[0-9]:"
}

for c in $CASES; do
  echo "=================== $c"
  case $c in
    k100|k5|k40)
      K=${c#k}; D=$OUT/$c; rm -rf "$D"
      FAKE_K=$K $PY bench/sweep.py --config bench/configs/test-fake.yaml --out "$D" > "$D.log" 2>&1
      bash bench/fake_serve.sh --stop
      goodput "$K" bench/configs/test-fake.yaml "$D"
      ;;
    resume)
      # 3 lần lặp, khởi động lại server giữa các lần lặp ở lưới gốc; ngắt 2 lần (SIGINT rồi SIGKILL) giữa chừng
      D=$OUT/resume; rm -rf "$D"; CFG=$OUT/test-fake-r3.yaml
      sed 's/^repeats: 1/repeats: 3/' bench/configs/test-fake.yaml > "$CFG"
      export FAKE_K=40
      timeout -s INT 75 $PY bench/sweep.py --config "$CFG" --out "$D" > "$D.1.log" 2>&1 || echo "lần 1 bị ngắt (SIGINT), mã $?"
      ls "$D/units" | sort > "$D.after1"
      echo "sau lần 1: $(ls "$D"/units/*/done.json | wc -l) đơn vị xong, $(ls "$D/units" | wc -l) thư mục"
      timeout -s KILL 150 $PY bench/sweep.py --config "$CFG" --out "$D" > "$D.2.log" 2>&1 || echo "lần 2 bị ngắt (SIGKILL), mã $?"
      echo "sau lần 2: $(ls "$D"/units/*/done.json | wc -l) đơn vị xong, $(ls "$D/units" | wc -l) thư mục"
      grep -h "dở dang" "$D.2.log" || true
      stat -c '%Y %n' "$D"/units/*/done.json | sort -k2 > "$D.mtime2"
      $PY bench/sweep.py --config "$CFG" --out "$D" > "$D.3.log" 2>&1
      grep -h "dở dang\|đợt mới\|xong toàn bộ" "$D.3.log"
      stat -c '%Y %n' "$D"/units/*/done.json | sort -k2 > "$D.mtime3"
      # đơn vị đã xong trước lần 3 không bị đo lại
      join -1 2 -2 2 "$D.mtime2" "$D.mtime3" | awk '$2 != $3 {bad=1; print "ĐO LẠI NHẦM:", $1} END {exit bad}'
      echo "lần 3: các đơn vị đã xong không bị đo lại ✓; số lần khởi động server: $(wc -l < "$D/server_starts.jsonl")"
      goodput 40 "$CFG" "$D"
      # so với chạy một mạch (không ngắt) ở K=40, 1 lần lặp: cùng các mức
      bash bench/fake_serve.sh --stop
      unset FAKE_K
      ;;
    config)
      # sửa cấu hình (số request tối thiểu) → hash khác → mọi đơn vị cũ thành "chưa xong"
      D=$OUT/k40; CFG=$OUT/test-fake-min120.yaml
      sed 's/num_requests: {min: 100/num_requests: {min: 120/' bench/configs/test-fake.yaml > "$CFG"
      FAKE_K=40 $PY bench/sweep.py --config "$CFG" --out "$D" --dry-run | grep "đợt mới"
      FAKE_K=40 $PY bench/sweep.py --config bench/configs/test-fake.yaml --out "$D" --dry-run | grep "đợt mới" \
        || echo "cấu hình gốc: không còn gì phải đo ✓"
      ;;
    closed)
      # client tự viết (closed-loop) vs vllm bench serve, cùng mẫu, server giả kiểu hàng đợi
      D=$OUT/closed; rm -rf "$D"
      $PY bench/sweep.py --config bench/configs/test-fake-closed.yaml --out "$D" > "$D.log" 2>&1
      bash bench/fake_serve.sh --stop
      $PY bench/summarize.py --out "$D" --config bench/configs/test-fake-closed.yaml --compare-clients
      ;;
    scenario)
      # phát lại kịch bản W4 peak k=10 với thời gian co 100 lần (40 phút → 24 s) trên server giả kiểu hàng đợi
      D=$OUT/scenario; mkdir -p "$D"
      bash bench/fake_serve.sh "$D/server.log" --mode queue --capacity 64 --prefill-s 0.00001 --token-s 0.0002 --chunk-tokens 50
      $PY bench/session_client.py scenario --scenario data/workloads/w4/peak-k10-s1.jsonl \
        --tokenizer data/raw/qwen3-tokenizer --base-url http://localhost:8123 --model Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
        --out "$D/w4.json.gz" --ignore-eos --temperature 0 --time-scale 0.01
      bash bench/fake_serve.sh --stop
      $PY bench/summarize.py --scenario-check "$D/w4.json.gz" data/workloads/w4/peak-k10-s1.jsonl
      ;;
    w4)
      # chế độ w4 (P05): điểm mốc + tìm k + kiểm tra, thời gian co 100 lần; ngắt giữa một lần chạy (SIGKILL) rồi chạy lại:
      # đơn vị dở bị xoá và chạy lại, đơn vị đã xong không bị chạy lại
      D=$OUT/w4; rm -rf "$D"; CFG=bench/configs/test-fake-w4.yaml
      export FAKE_CAP=${FAKE_CAP:-24}
      timeout -s KILL 100 $PY bench/sweep.py --config $CFG --out "$D" > "$D.1.log" 2>&1 || echo "lần 1 bị ngắt (SIGKILL), mã $?"
      echo "sau lần 1: $(ls "$D"/units/*/done.json 2>/dev/null | wc -l) đơn vị xong, $(ls "$D/units" | wc -l) thư mục"
      $PY bench/sweep.py --config $CFG --out "$D" --status | sed -n '1,40p'
      stat -c '%Y %n' "$D"/units/*/done.json | sort -k2 > "$D.mtime1"
      $PY bench/sweep.py --config $CFG --out "$D" > "$D.2.log" 2>&1
      grep -h "dở dang\|đợt mới\|điểm mốc\|xong toàn bộ" "$D.2.log"
      stat -c '%Y %n' "$D"/units/*/done.json | sort -k2 > "$D.mtime2"
      join -1 2 -2 2 "$D.mtime1" "$D.mtime2" | awk '$2 != $3 {bad=1; print "ĐO LẠI NHẦM:", $1} END {exit bad}'
      echo "lần 2: các đơn vị đã xong không bị chạy lại ✓"
      bash bench/fake_serve.sh --stop
      $PY bench/sweep.py --config $CFG --out "$D" --status
      $PY bench/summarize.py --config $CFG --out "$D" --quiet && sed -n '1,200p' "$D/summary.md"
      for u in "$D"/units/w4-*; do
        $PY bench/summarize.py --scenario-check "$u/bench.json.gz" \
          "$(python3 -c "import json,sys; print(json.load(open('$u/meta.json'))['dataset'])")" | tail -4
      done
      unset FAKE_CAP
      ;;
    multi)
      # P08: scripts/remote_multi.sh — 2 cấu hình bench (A grid, B search khác cờ server) + 1 cổng chất lượng, nối tiếp.
      # Lần 1 bị SIGKILL cả nhóm tiến trình giữa chừng (như máy chết); lần 2 cùng lệnh → không đo lại đơn vị đã xong.
      D=$OUT/multi; rm -rf "$D"; mkdir -p "$D"
      export FAKE_K=40 MULTI_PY=$PY MULTI_SKIP_SETUP=1 MULTI_SETUP_OUT=$D/setup \
             MULTI_LMEVAL_VENV=$HOME/.local/share/lmeval-venv
      printf '%s\n' "# loại cấu hình thư-mục" "bench bench/configs/test-fake-multi-a.yaml $D/a" \
        "bench bench/configs/test-fake-multi-b.yaml $D/b" "quality quality/configs/test-fake-tune.yaml $D/q" > "$D/list.txt"
      setsid bash scripts/remote_multi.sh "$D/list.txt" > "$D.1.log" 2>&1 & PG=$!
      # ngắt khi B đã xong ít nhất 1 đơn vị và đang chạy đơn vị tiếp theo
      for _ in $(seq 600); do
        [ "$(ls "$D"/b/units/*/done.json 2>/dev/null | wc -l)" -ge 1 ] && break; sleep 0.5
      done
      sleep 3; kill -KILL -- -"$PG" 2>/dev/null || true; bash bench/fake_serve.sh --stop
      echo "sau lần 1: A $(ls "$D"/a/units/*/done.json 2>/dev/null | wc -l) xong; B $(ls "$D"/b/units/*/done.json 2>/dev/null | wc -l) xong," \
           "$(ls "$D/b/units" | wc -l) thư mục; Q $(ls "$D"/q/units 2>/dev/null | wc -l) thư mục"
      stat -c '%Y %n' "$D"/{a,b}/units/*/done.json | sort -k2 > "$D.mtime1"
      bash scripts/remote_multi.sh "$D/list.txt" > "$D.2.log" 2>&1
      grep -h "MULTI_STEP \|dở dang\|REMOTE_MULTI_DONE" "$D.2.log"
      stat -c '%Y %n' "$D"/{a,b}/units/*/done.json | sort -k2 > "$D.mtime2"
      join -1 2 -2 2 "$D.mtime1" "$D.mtime2" | awk '$2 != $3 {bad=1; print "ĐO LẠI NHẦM:", $1} END {exit bad}'
      echo "lần 2: các đơn vị đã xong không bị đo lại ✓"
      bash bench/fake_serve.sh --stop
      goodput 40 bench/configs/test-fake-multi-b.yaml "$D/b"
      echo "(B: server giả sức chứa 48 → goodput mong đợi 48)"
      $PY bench/sweep.py --config bench/configs/test-fake-multi-b.yaml --out "$D/b" --status | tail -2
      $PY - "$D/b" <<'PYEOF'
import json, sys
for x in open(sys.argv[1] + "/server_starts.jsonl"):
    j = json.loads(x); print("server_starts:", {k: j.get(k) for k in ("reason", "kv_cache_tokens", "attention_backend", "kv_cache_dtype")})
PYEOF
      $PY quality/run.py --config quality/configs/test-fake-tune.yaml --out "$D/q" --status
      $PY quality/analyze.py --gate results/gd1/06-chat-luong-baseline/20261013-p06 \
        results/gd1/06-chat-luong-baseline/20261013-p06r2 "$D/q" | tail -12
      unset FAKE_K MULTI_PY MULTI_SKIP_SETUP MULTI_SETUP_OUT MULTI_LMEVAL_VENV
      ;;
  esac
done
