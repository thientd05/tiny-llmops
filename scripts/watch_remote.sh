#!/usr/bin/env bash
# Chạy ở LOCAL, nền: kéo kết quả từ máy thuê về (rsync) mỗi INTERVAL giây và đọc các dòng log MỚI (vị trí đã đọc lưu trong
# STATE_FILE, không mất sự kiện giữa hai lần chạy watcher). Thoát (để Claude/người dùng được báo) khi:
#   - có đơn vị đo xong ("xong sau N s —", "xong N mục"; không tính từng khúc 100 prompt), một bước của remote_multi xong, cả lượt xong → mã 0, in dòng sự kiện;
#   - có lỗi (Traceback, SERVER CHẾT, lỗi, Killed, mất SSH 3 lần liên tiếp) → mã 1;
#   - hết MAX_S giây không có gì → mã 0, in "HEARTBEAT".
#   bash scripts/watch_remote.sh SSH_PORT SSH_HOST REMOTE_DIR LOCAL_DIR STATE_FILE [MAX_S=1200] [INTERVAL=60]
# Ví dụ: bash scripts/watch_remote.sh 41234 1.2.3.4 /root/results/p09/ results/gd1/09-tang-cho-kv/20261014-p09/ /tmp/w.state
set -uo pipefail
PORT=$1; HOST=$2; RDIR=$3; LDIR=$4; STATE=$5; MAX_S=${6:-1200}; INTERVAL=${7:-60}
SSH="ssh -i $HOME/.ssh/vast -p $PORT -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
mkdir -p "$LDIR"; touch "$STATE"
t0=$(date +%s); fails=0
EVENT_RE='xong sau [0-9]+ s —|xong [0-9]+ mục|MULTI_STEP_DONE|REMOTE_MULTI_DONE|xong toàn bộ'
ERR_RE='Traceback|SERVER CHẾT|[Ee]rror|lỗi \(mã|Killed|DỪNG|CUDA out of memory'
while true; do
  if rsync -az --partial -e "$SSH" "root@$HOST:$RDIR" "$LDIR" 2>/dev/null; then fails=0; else fails=$((fails + 1)); fi
  if [ "$fails" -ge 3 ]; then echo "LỖI: rsync/SSH thất bại 3 lần liên tiếp"; exit 1; fi
  found_evt=""; found_err=""
  while IFS= read -r f; do
    off=$(awk -v f="$f" '$1 == f {print $2}' "$STATE"); off=${off:-0}
    size=$(stat -c %s "$f")
    [ "$size" -le "$off" ] && continue
    new=$(tail -c +"$((off + 1))" "$f")
    grep -v '^$' <<<"$new" | sed "s|^|$(basename "$(dirname "$f")")/$(basename "$f"): |" >> "$STATE.new_lines"
    e=$(grep -E "$ERR_RE" <<<"$new" | grep -v "lỗi 0" | head -5); [ -n "$e" ] && found_err+="$f: $e"$'\n'
    v=$(grep -E "$EVENT_RE" <<<"$new" | tail -5); [ -n "$v" ] && found_evt+="$v"$'\n'
    awk -v f="$f" '$1 != f' "$STATE" > "$STATE.tmp"; echo "$f $size" >> "$STATE.tmp"; mv "$STATE.tmp" "$STATE"
  done < <(find "$LDIR" -maxdepth 2 -name '*.log' 2>/dev/null)   # sweep.log, run.log, multi.log (không đọc log server)
  if [ -n "$found_err" ]; then echo "LỖI:"; echo "$found_err"; exit 1; fi
  if [ -n "$found_evt" ]; then echo "SỰ KIỆN:"; echo "$found_evt"; exit 0; fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAX_S" ]; then echo "HEARTBEAT $(date +%H:%M) — không có sự kiện mới"; exit 0; fi
  sleep "$INTERVAL"
done
