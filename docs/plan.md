# Kế hoạch & ngân sách

> Ước tính lập ngày 2026-10-12, **chưa có số đo thật**. Sau GĐ1 (có throughput thật trên L4) → cập nhật lại file này.

## Giả định

| Tham số | Giá trị ước tính | Nguồn |
|---|---|---|
| Decode Qwen3-30B-A3B INT4 trên L4 (tổng tok/s) @ 8 / 16 / 32 / 64 user | 260 / 380 / 520 / 600 | Suy từ Jetson Thor (273 GB/s): C1 ~81, C8 ~262 tok/s (W4A16, vLLM) |
| Prefill | ~5.000 tok/s | 2 × 3.3B FLOP/token, ~50% của 121 TFLOPS FP16 |

> **Số đo thật (P04, 2026-10-13, vLLM mặc định):** decode W1 243 / 323 / 458 / 610 tok/s @ 8 / 16 / 32 / 64 user; prefill lúc bão hoà
> ~2,780 tok/s (W2) / ~1,940 (W3); đơn giá trọn gói ≈ 7,570 VND/h ($0.289/h). Chi tiết: [measurements.md](measurements.md) mục P04.
| Đơn giá L4 trọn gói | ~10.200 VND/h | Vast ~$0.33/h × 26.200 + ~15% ổ đĩa/băng thông |
| Thời gian viết code | Không tính | Dùng Claude Code |

## Đơn vị chi phí

| Đơn vị | GPU-giờ |
|---|---|
| Sweep rút gọn (tìm goodput khởi đầu nóng từ cấu hình trước, ~4 mức, 1 lần) | ~0.7 |
| Sweep đầy đủ (W1+W2+W3 closed-loop: lưới gốc 4 mức + mở rộng/tìm nhị phân ~3–5 mức, ×3 lần lặp ~7h; + W4 tải giống thật khởi đầu nóng ~4 lần chạy + 2 lần kiểm ổn định, mỗi lần ~40 phút ~4.5h) | ~11.5 |
| Bộ đo chất lượng (MMLU-Pro, GSM8K, RAG accuracy, token-match/KL) | ~1 |
| 1 vòng lặp kernel (compile + unit test + microbench + profile) | ~0.25 |
| Setup 1 phiên máy mới | ~0.25 |

Cách chọn mức tải: [workloads.md — Cách tìm goodput](workloads.md#cách-tìm-goodput-lưới-thích-nghi) (W1–W3) và
[workloads.md — W4](workloads.md#w4--tải-giống-thật-bài-test-cuối). Số mức không cố định, vì tuỳ điểm gãy SLO của từng workload.
Hai số sweep ở trên là ước tính, sẽ chỉnh sau P03/P04/P05.

## Giai đoạn & giờ GPU

Kịch bản **A**: mọi thư viện thua vLLM. Kịch bản **B**: 1 thư viện ngang vLLM → giữ cả hai.

| GĐ | Nội dung | A | B |
|---|---|---|---|
| 0 | Chuẩn bị máy, harness, bộ sinh workload (RAGBench có sẵn context → không cần embed) | ~3 | ~3 |
| 1 | Tune vLLM tối đa: tune config MoE (2h), ~20 sweep rút gọn dò tham số, sweep đầy đủ cho bản mặc định + bản tune, chất lượng | ~41 | ~41 |
| 2 | Sàng lọc: SGLang ~3h, TRT-LLM ~4h (dừng sớm nếu không chạy), LMDeploy ~3h, llama.cpp ~1.5h, LMCache ~2h; B: + full sweep thư viện sống sót + thử kết hợp | ~14 | ~31 |
| 3 | Profile cấu hình tốt nhất (nsys/ncu), danh sách nghẽn | ~8 | ~8 |
| 4 | Custom CUDA: kernel khởi động ~6h, kernel MoE ~19h, kernel thứ hai ~13h; B: + gắn vào thư viện 2 | ~38 | ~44 |
| 5 | Đánh giá cuối: ~5 biến thể × sweep đầy đủ + chất lượng; B: + 2 biến thể | ~62 | ~86 |
| — | Chạy lại + setup các phiên | ~13 | ~14 |
| | **Tổng tính toán thuần** | **~179h** | **~227h** |

> 2026-10-12: tăng từ 129h / 158h do chuyển sang lưới thích nghi + tìm nhị phân goodput. Sweep rút gọn 0.5 → 0.7h,
> sweep đầy đủ 5 → 8h. Phần tăng: GĐ1 +10h, GĐ2 (B) +3h, GĐ5 +15h (A) / +21h (B).
> 2026-10-12 (sau): thêm W4 tải giống thật thay cho 1h replay BurstGPT → sweep đầy đủ 8 → 11.5h; tổng 154h → 179h (A), 192h → 227h (B).

## Chi phí

| | A | B |
|---|---|---|
| Thuần | 179h ≈ 1,82 triệu VND (~26 ngày × 7h) | 227h ≈ 2,31 triệu (~32 ngày) |
| Thực tế (×1,3 thời gian GPU chờ) | 233h ≈ 2,37 triệu (~33 ngày) | 295h ≈ 3,00 triệu (~42 ngày) |
| + 25% chạy lại | 291h ≈ 2,96 triệu (~42 ngày) | 369h ≈ 3,75 triệu (~53 ngày) |

**So với ngân sách** (theo thời gian làm: 1 tháng ≤ ~2,1 triệu, 2 tháng ≤ ~3 triệu, 3 tháng ≤ ~4 triệu, thêm ~1 triệu mỗi tháng):

| Kịch bản | Cần (có chạy lại) | Vừa ngân sách khi trải trong | Ghi chú |
|---|---|---|---|
| A | ~2,96 triệu | **2 tháng** (≤ 3 triệu) | Sát trần, dư ~40k. Nên giữ các mẹo giảm giờ GPU bên dưới |
| B | ~3,75 triệu | **3 tháng** (≤ 4 triệu) | Dư ~250k |

Không vừa nếu ép trong 1 tháng (cả A và B đều > 2,1 triệu).

Phụ phí dự phòng: ncu trên AWS g6 (L4 VM) 10–20h ≈ 100–400k VND — xem [challenges.md](challenges.md) #1.

## Mẹo giảm giờ GPU

- Trong lúc phát triển: sweep rút gọn; sweep đầy đủ ×3 chỉ ở GĐ5.
- Sweep rút gọn **khởi đầu nóng** từ `lo`/`hi` goodput của cấu hình trước. Một thay đổi tham số hiếm khi dời điểm gãy xa,
  nên thường chỉ cần ~4 điểm thay vì đo lại cả lưới gốc.
- Chỉ tìm nhị phân tới độ phân giải `max(2, 10% × lo)`. Mịn hơn thì chìm trong nhiễu đo, tốn giờ mà không thêm thông tin.
- Kernel dạng torch extension + vLLM `VLLM_USE_PRECOMPILED=1` → không build lại vLLM (~30 phút/lần).
- ncu chỉ profile kernel mục tiêu (`--kernel-name`, `--launch-count`).
- Chọn host có phí tải xuống thấp (`inet_down_cost`); tải model 17GB mỗi phiên.
