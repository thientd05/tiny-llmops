# Kế hoạch & ngân sách

> Ước tính lập ngày 2026-10-12, **chưa có số đo thật**. Sau GĐ1 (có throughput thật trên L4) → cập nhật lại file này.
>
> **Đã cập nhật ngày 2026-10-14 theo số đo GĐ1** (báo cáo [P07](gd1-vllm-baseline/07-tong-hop-baseline/bao-cao.md)): giữ nguyên các dòng ước tính cũ,
> thêm cột / mục "đo được" và "ước lại 2026-10-14". Tỉ giá vẫn 26,200 VND/USD.

## Giả định

| Tham số | Giá trị ước tính (12/10) | Nguồn | Đo được GĐ1 (cập nhật 14/10) |
|---|---|---|---|
| Decode Qwen3-30B-A3B INT4 trên L4 (tổng tok/s) @ 8 / 16 / 32 / 64 user | 260 / 380 / 520 / 600 | Suy từ Jetson Thor (273 GB/s): C1 ~81, C8 ~262 tok/s (W4A16, vLLM) | W1: 243 / 323 / 458 / 610 (−7 / −15 / −12 / +2%); bão hoà ~615 từ 64 user (P04) |
| Prefill | ~5.000 tok/s | 2 × 3.3B FLOP/token, ~50% của 121 TFLOPS FP16 | ~2,780 tok/s vào lúc bão hoà W2 (kèm decode), ~1,940 W3 (−44 / −61%); GPU luôn ở trần 72 W, xung ~1.1 GHz (P04) |
| Đơn giá L4 trọn gói | ~10.200 VND/h | Vast ~$0.33/h × 26.200 + ~15% ổ đĩa/băng thông | **8,770 VND/h** = $4.432 / 13.24 h của 5 phiên GĐ1 (gồm ổ đĩa, tải về); theo máy 7,560–9,900 (P07 mục 3.4). Dùng làm **giá tham chiếu** cho mọi phép tính VND / 1M token |
| Thời gian viết code | Không tính | Dùng Claude Code | — |
| Hệ số GPU chờ (thời gian máy / thời gian đo) | ×1.3 | ước | ~×1.1 ở các phiên benchmark (P04, P05); giữ ×1.3 cho phần ước vì GĐ4 viết kernel sẽ có nhiều thời gian máy chờ hơn |

> **Số đo thật (P04, 2026-10-13, vLLM mặc định):** decode W1 243 / 323 / 458 / 610 tok/s @ 8 / 16 / 32 / 64 user; prefill lúc bão hoà
> ~2,780 tok/s (W2) / ~1,940 (W3); đơn giá trọn gói ≈ 7,570 VND/h ($0.289/h). Chi tiết: [measurements.md](measurements.md) mục P04.
> (Ghi chú cũ, giữ nguyên. Đơn giá 7,570 là của riêng phiên P03 + P04; giá tham chiếu chung của GĐ1 là 8,770, xem bảng trên.)

## Đơn vị chi phí

| Đơn vị | GPU-giờ (ước 12/10) | Thực tế / ước lại 14/10 |
|---|---|---|
| Sweep rút gọn (tìm goodput khởi đầu nóng từ cấu hình trước, ~4 mức, 1 lần) | ~0.7 | **~0.8** (4 mức × W1 2.8 + W2 2.1 + W3 5.4 phút/đơn vị + ~7 phút compile và bật lại server cho cấu hình mới). ×3 lần lặp: ~2.2 |
| Sweep đầy đủ (W1+W2+W3 closed-loop: lưới gốc 4 mức + mở rộng/tìm nhị phân ~3–5 mức, ×3 lần lặp ~7h; + W4 tải giống thật khởi đầu nóng ~4 lần chạy + 2 lần kiểm ổn định, mỗi lần ~40 phút ~4.5h) | ~11.5 | **9.4** (P04 4.44 + P05 4.97) |
| W4 một seed, khởi đầu nóng (lo, hi) | (trong sweep đầy đủ) | **~1.5–2.2** (lần đạt ~41 phút, lần trượt dừng sớm 19–28 phút) |
| Bộ đo chất lượng (MMLU-Pro, GSM8K, RAG accuracy, token-match/KL) | ~1 | Đủ bộ: 1.78 (P06). Cổng chất lượng cho một cấu hình: **~0.8 một lần, ~1.6 hai lần** (teacher forcing + sinh tham chiếu + FinQA + MMLU-Pro) |
| 1 vòng lặp kernel (compile + unit test + microbench + profile) | ~0.25 | chưa đo |
| Setup 1 phiên máy mới | ~0.25 | ~0.2 (cài 3–4 phút song song tải model, đẩy code, 2 lần khởi động server) |

Cách chọn mức tải: [workloads.md — Cách tìm goodput](workloads.md#cách-tìm-goodput-lưới-thích-nghi) (W1–W3) và
[workloads.md — W4](workloads.md#w4--tải-giống-thật-bài-test-cuối). Số mức không cố định, vì tuỳ điểm gãy SLO của từng workload.
Hai số sweep ở trên là ước tính, sẽ chỉnh sau P03/P04/P05.

## Giai đoạn & giờ GPU

Kịch bản **A**: mọi thư viện thua vLLM. Kịch bản **B**: 1 thư viện ngang vLLM → giữ cả hai.

| GĐ | Nội dung | A | B | Thực tế / ước lại 14/10 |
|---|---|---|---|---|
| 0 | Chuẩn bị máy, harness, bộ sinh workload (RAGBench có sẵn context → không cần embed) | ~3 | ~3 | **Xong: 1.2 h** (P01 0.35 + P03 0.84, + thử máy 0.07) |
| 1 | Tune vLLM tối đa: tune config MoE (2h), ~20 sweep rút gọn dò tham số, sweep đầy đủ cho bản mặc định + bản tune, chất lượng | ~41 | ~41 | Baseline **xong: 12.0 h** (P04 4.44 + P05 4.97 + P06 2.67). Phần tune: **~31 h, chốt đo kiểu rẻ** (14/10; đo bổ sung khi cần, tối đa ~52 h nếu phải làm đủ kiểu chặt; P07 mục 6.3). → **Lập plan P08–P14 (14/10): ~14.8 h (12.0–23.1)**, vì sàng lọc ở một mức tải thay vì tìm goodput cho từng cấu hình ([gd1-vllm-tune/README.md](gd1-vllm-tune/README.md)) → **GĐ1 ≈ 27 h**. (Ước lúc P07: bỏ "tune config MoE" vì không có dấu hiệu, P07 mục 4.6 → GĐ1 ≈ 43–64 h.) |
| 2 | Sàng lọc: SGLang ~3h, TRT-LLM ~4h (dừng sớm nếu không chạy), LMDeploy ~3h, llama.cpp ~1.5h, LMCache ~2h; B: + full sweep thư viện sống sót + thử kết hợp | ~14 | ~31 | giữ nguyên. LMCache đã có sẵn làm backend offloading trong vLLM 0.30.0 → có thể thử ngay ở GĐ1 tune |
| 3 | Profile cấu hình tốt nhất (nsys/ncu), danh sách nghẽn | ~8 | ~8 | giữ nguyên |
| 4 | Custom CUDA: kernel khởi động ~6h, kernel MoE ~19h, kernel thứ hai ~13h; B: + gắn vào thư viện 2 | ~38 | ~44 | giữ nguyên. Ứng viên mới (bộ xếp lịch, giữ KV khi cắt lịch sử) chưa ước giờ |
| 5 | Đánh giá cuối: ~5 biến thể × sweep đầy đủ + chất lượng; B: + 2 biến thể | ~62 | ~86 | sweep đầy đủ 11.5 → 9.4 h: ~5 × (9.4 + 1.8) ≈ **~56** (A) / ~78 (B); giữ số cũ làm trần |
| — | Chạy lại + setup các phiên | ~13 | ~14 | giữ nguyên |
| | **Tổng tính toán thuần** | **~179h** | **~227h** | **A: ~179–200 h; B: ~227–248 h** (GĐ0 thật + GĐ1 ước lại, các GĐ khác giữ số cũ) |

> 2026-10-12: tăng từ 129h / 158h do chuyển sang lưới thích nghi + tìm nhị phân goodput. Sweep rút gọn 0.5 → 0.7h,
> sweep đầy đủ 5 → 8h. Phần tăng: GĐ1 +10h, GĐ2 (B) +3h, GĐ5 +15h (A) / +21h (B).
> 2026-10-12 (sau): thêm W4 tải giống thật thay cho 1h replay BurstGPT → sweep đầy đủ 8 → 11.5h; tổng 154h → 179h (A), 192h → 227h (B).
> 2026-10-14 (P07): GĐ0 thật 1.2 h; GĐ1 baseline thật 12.0 h, phần tune ước 31–52 h tuỳ cách đo (chưa chốt). GĐ5 có thể giảm (sweep đầy đủ rẻ hơn),
> chưa trừ vào tổng để giữ biên.

## Chi phí

| | A | B |
|---|---|---|
| Thuần | 179h ≈ 1,82 triệu VND (~26 ngày × 7h) | 227h ≈ 2,31 triệu (~32 ngày) |
| Thực tế (×1,3 thời gian GPU chờ) | 233h ≈ 2,37 triệu (~33 ngày) | 295h ≈ 3,00 triệu (~42 ngày) |
| + 25% chạy lại | 291h ≈ 2,96 triệu (~42 ngày) | 369h ≈ 3,75 triệu (~53 ngày) |

**Ước lại 2026-10-14** (giá tham chiếu 8,770 VND/h; giờ theo cột "Thực tế / ước lại" ở trên):

| | A (tune rẻ … chặt) | B (tune rẻ … chặt) |
|---|---|---|
| Thuần | 179–200 h ≈ 1.57–1.75 triệu | 227–248 h ≈ 1.99–2.17 triệu |
| ×1.3 chờ | 233–260 h ≈ 2.04–2.28 triệu | 295–322 h ≈ 2.59–2.83 triệu |
| + 25% chạy lại | 291–325 h ≈ **2.55–2.85 triệu** | 369–403 h ≈ **3.24–3.53 triệu** |
| Nếu giá như máy đắt nhất GĐ1 (9,900 VND/h) | 2.88–3.22 triệu | 3.65–3.99 triệu |
| Đã tiêu (tới hết GĐ1 baseline, 14/10) | **~116.6k** (tín dụng giảm $4.451) | |
| Sau khi lập plan tune (GĐ1 ≈ 27 h thay vì 43–64 h), + 25% chạy lại | A: ~163 h thuần → ×1.3 × 1.25 ≈ **2.32 triệu** | B: ~211 h → ≈ **3.01 triệu** |

**So với ngân sách** (theo thời gian làm: 1 tháng ≤ ~2,1 triệu, 2 tháng ≤ ~3 triệu, 3 tháng ≤ ~4 triệu, thêm ~1 triệu mỗi tháng):

| Kịch bản | Cần (có chạy lại) | Vừa ngân sách khi trải trong | Ghi chú |
|---|---|---|---|
| A | ~2,96 triệu | **2 tháng** (≤ 3 triệu) | Sát trần, dư ~40k. Nên giữ các mẹo giảm giờ GPU bên dưới |
| B | ~3,75 triệu | **3 tháng** (≤ 4 triệu) | Dư ~250k |
| A (ước lại 14/10) | ~2.55–2.85 triệu | **2 tháng** | Dư ~150–450k. Nếu giá lên 9,900 VND/h và đo tune kiểu chặt: ~3.2 triệu, vượt mốc 2 tháng ~200k → cảnh báo lúc đó |
| B (ước lại 14/10) | ~3.24–3.53 triệu | **3 tháng** | Dư ~470–760k |

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
- (14/10) W4: dừng sớm khi chắc chắn trượt đã tiết kiệm ~1 h ở P05; giữ cho mọi lần đo W4.
- (14/10) Gộp các cấu hình cùng cần một máy vào một phiên dài: phí tải model ~$0.027 mỗi máy mới làm phiên ngắn đắt hơn ~10–25% mỗi giờ.
