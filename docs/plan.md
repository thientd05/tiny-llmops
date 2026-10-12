# Kế hoạch & ngân sách

> Ước tính lập ngày 2026-10-12, **chưa có số đo thật**. Sau GĐ1 (có throughput thật trên L4) → cập nhật lại file này.

## Giả định

| Tham số | Giá trị ước tính | Nguồn |
|---|---|---|
| Decode Qwen3-30B-A3B INT4 trên L4 (tổng tok/s) @ 8 / 16 / 32 / 64 user | 260 / 380 / 520 / 600 | Suy từ Jetson Thor (273 GB/s): C1 ~81, C8 ~262 tok/s (W4A16, vLLM) |
| Prefill | ~5.000 tok/s | 2 × 3.3B FLOP/token, ~50% của 121 TFLOPS FP16 |
| Đơn giá L4 trọn gói | ~10.200 VND/h | Vast ~$0.33/h × 26.200 + ~15% ổ đĩa/băng thông |
| Thời gian viết code | Không tính | Dùng Claude Code |

## Đơn vị chi phí

| Đơn vị | GPU-giờ |
|---|---|
| Sweep rút gọn (3 mức concurrency, 1 lần) | ~0.5 |
| Sweep đầy đủ (W1+W2+W3, 4 mức, ×3 lần lặp + 1h replay BurstGPT) | ~5 |
| Bộ đo chất lượng (MMLU-Pro, GSM8K, RAG accuracy, token-match/KL) | ~1 |
| 1 vòng lặp kernel (compile + unit test + microbench + profile) | ~0.25 |
| Setup 1 phiên máy mới | ~0.25 |

## Giai đoạn & giờ GPU

Kịch bản **A**: mọi thư viện thua vLLM. Kịch bản **B**: 1 thư viện ngang vLLM → giữ cả hai.

| GĐ | Nội dung | A | B |
|---|---|---|---|
| 0 | Chuẩn bị máy, harness, bộ sinh workload (RAGBench có sẵn context → không cần embed) | ~3 | ~3 |
| 1 | Tune vLLM tối đa: tune config MoE (2h), ~20 sweep rút gọn dò tham số, sweep đầy đủ cho bản mặc định + bản tune, chất lượng | ~24 | ~24 |
| 2 | Sàng lọc: SGLang ~3h, TRT-LLM ~4h (dừng sớm nếu không chạy), LMDeploy ~3h, llama.cpp ~1.5h, LMCache ~2h; B: + full sweep thư viện sống sót + thử kết hợp | ~14 | ~24 |
| 3 | Profile cấu hình tốt nhất (nsys/ncu), danh sách nghẽn | ~8 | ~8 |
| 4 | Custom CUDA: kernel khởi động ~6h, kernel MoE ~19h, kernel thứ hai ~13h; B: + gắn vào thư viện 2 | ~38 | ~44 |
| 5 | Đánh giá cuối: ~5 biến thể × sweep đầy đủ + chất lượng; B: + 2 biến thể | ~29 | ~40 |
| — | Chạy lại + setup các phiên | ~13 | ~14 |
| | **Tổng tính toán thuần** | **~129h** | **~158h** |

## Chi phí

| | A | B |
|---|---|---|
| Thuần | 129h ≈ 1,31 triệu VND (~18 ngày × 7h) | 158h ≈ 1,61 triệu (~23 ngày) |
| Thực tế (×1,3 thời gian GPU chờ) | 167h ≈ 1,70 triệu (~24 ngày) | 205h ≈ 2,09 triệu (~29 ngày) |
| + 25% chạy lại | 209h ≈ 2,13 triệu (~30 ngày) | 256h ≈ 2,61 triệu (~37 ngày) |

Phụ phí dự phòng: ncu trên AWS g6 (L4 VM) 10–20h ≈ 100–400k VND — xem [challenges.md](challenges.md) #1.

## Mẹo giảm giờ GPU

- Trong lúc phát triển: sweep rút gọn; sweep đầy đủ ×3 chỉ ở GĐ5.
- Kernel dạng torch extension + vLLM `VLLM_USE_PRECOMPILED=1` → không build lại vLLM (~30 phút/lần).
- ncu chỉ profile kernel mục tiêu (`--kernel-name`, `--launch-count`).
- Chọn host có phí tải xuống thấp (`inet_down_cost`); tải model 17GB mỗi phiên.
