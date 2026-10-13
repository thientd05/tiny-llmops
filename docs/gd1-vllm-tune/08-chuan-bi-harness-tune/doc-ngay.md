# P08 — Đọc ngay: chuẩn bị công cụ đo cho phần tune

**Một câu:** sửa bộ công cụ đo (harness) ở laptop để một phiên thuê máy chạy được nhiều cấu hình vLLM nối tiếp, tự chạy tiếp khi bị ngắt,
và tự ghi lại vLLM thật sự chạy với kernel nào; thử hết bằng server giả, **0 đồng**. Ba việc chỉ cần cho P10 được để lại làm trước P10.

---

## Bước 1 — Một file chung, mỗi cấu hình chỉ ghi phần khác

Phần tune sẽ thử khoảng 10 cấu hình. Chép tay cả file cấu hình cho từng cái thì dễ lệch một tham số mà không biết.
Giờ mỗi cấu hình viết như sau (`bench/configs/tune/p09-kv8.yaml`):

```yaml
include: base.yaml                     # ← lấy toàn bộ file chung (seed, số request, SLO, model, lệnh tắt server…)
server:
  start: "bash scripts/serve_default.sh {log} --max-model-len 16384 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95"
workloads:
  w1: {path: data/workloads/w1.jsonl, warm_start: [72, 88]}   # ← bắt đầu tìm ở 72 và 88 user, không phải 8/16/32/64
```

Ghi `w1: null` nghĩa là bỏ W1 khỏi file chung (dùng cho điểm mốc chỉ đo W2).

## Bước 2 — Chạy nhiều cấu hình nối tiếp trên máy thuê

`scripts/remote_multi.sh DANH_SÁCH` đọc từng dòng "loại, cấu hình, thư mục kết quả" rồi chạy lần lượt.
Danh sách cho P09 (`bench/configs/tune/p09-list.txt`):

```
bench   bench/configs/tune/p09-anchor.yaml  /root/results/p09/anchor        ← vLLM mặc định, W2 16 user × 3 lần: so máy hôm nay với máy P04
quality quality/configs/tune-kv8.yaml       /root/results/p09/quality-kv8   ← đo chất lượng của KV FP8 trước
bench   bench/configs/tune/p09-kv8.yaml     /root/results/p09/kv8           ← rồi mới tìm số user tối đa
```

Đổi cấu hình thì script bật lại vLLM với cờ mới. Có thể nối thêm dòng vào cuối file trong lúc đang chạy (dùng cho P12).

**Thử:** chạy với server giả, giết toàn bộ tiến trình (`kill -KILL`, giống máy chết) khi đang đo giữa chừng, rồi chạy lại đúng lệnh cũ:

```
sau lần 1: A 2 xong; B 1 xong, 2 thư mục                              ← B có 1 đơn vị xong, 1 đang đo dở
[02:09:06] w1-c52-r1: thư mục dở dang (không có done.json hợp lệ) → xoá, đo lại
lần 2: các đơn vị đã xong không bị đo lại ✓
w1: done, goodput 48 (52 trượt)                                       ← server giả chịu được đúng 48 user → đúng
```

## Bước 3 — Ghi lại vLLM thật sự chạy cái gì

Khi bật KV FP8, vLLM có thể tự đổi kernel attention (trên GPU Ada, FlashAttention 2 không nhận KV FP8). Nếu không ghi lại, sẽ không biết
kết quả là do KV FP8 hay do đổi kernel. Mỗi lần server khởi động, harness đọc log và ghi vào `server_starts.jsonl`. Thử trên log thật của P04:

| Trường | Giá trị P04 | Nghĩa |
|---|---|---|
| `attention_backend` | `FLASH_ATTN` | kernel tính attention |
| `kv_cache_dtype` | `auto` | KV lưu cùng kiểu với model (FP16) |
| `moe_backend` | `MARLIN` | kernel nhân ma trận cho 128 expert INT4 |
| `max_num_batched_tokens` | `2048` | số token tối đa xử lý trong một bước |
| `kv_cache_tokens` | `37,264` | số token KV chứa được |

## Bước 4 — Cổng chất lượng cho cấu hình mới

KV FP8 lưu "bộ nhớ ngắn hạn" của model bằng số 8 bit thay vì 16 bit, nên có thể làm câu trả lời kém đi. Trước khi đo tốc độ, chạy 4 bài
(`quality/configs/tune-kv8.yaml`): chấm độ gần từng token trên đúng văn bản baseline đã sinh, sinh lại 1,000 câu, FinQA (1,147 câu hỏi tài chính),
MMLU-Pro (1,000 câu). Rồi:

```bash
.venv/bin/python quality/analyze.py --gate <baseline lần 1> <baseline lần 2> <kv8>
```

so kv8 với **cả hai** lần chạy baseline P06 (vì chính baseline chạy hai lần đã lệch 1.6 điểm MMLU-Pro). Quy tắc "kém đi" đặt trước khi đo:
thấp hơn lần baseline thấp hơn quá mức dao động, hoặc số câu đổi đúng/sai nhiều hơn 1.5 lần mức tự nhiên, hoặc xác suất token lệch hẳn về phía giảm.

## Bước 5 — Mấy công cụ nhỏ

- `bench/compare.py A B`: đặt hai lần đo cạnh nhau theo từng đơn vị cùng seed (cùng bộ câu hỏi), in hiệu %.
- `--status` ước giờ GPU còn lại bằng tốc độ đo thật ở P04 (trước đây đoán 1 giây / câu hỏi).
- `scripts/watch_remote.sh`: ở laptop, cứ 60 giây kéo kết quả về và đọc log mới; báo ngay khi một đơn vị đo xong, khi có lỗi, hoặc sau 20 phút.

## Để lại làm trước P10 (không cần cho P09, P12)

W4 đo ở một mức cố định, metric đếm token lấy từ RAM CPU, client gán độ ưu tiên theo độ dài câu hỏi, phần W4 của `compare.py`.
Sẽ viết và thử bằng server giả trước phiên P10.

## Rút ra

- Lỗi tìm được trước khi thuê máy: script chạy nhiều cấu hình ghi cứng đường dẫn máy thuê (không thử được ở laptop);
  thông báo lỗi trong đo chất lượng sẽ tự gây lỗi khi dùng văn bản baseline. Cả hai đã sửa.
- Bước tiếp theo: P09 (KV FP8 + thêm bộ nhớ cho KV) trên L4.
