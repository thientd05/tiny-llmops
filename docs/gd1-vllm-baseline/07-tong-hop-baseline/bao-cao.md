# Báo cáo P07 — Tổng hợp baseline, cập nhật kế hoạch

> Ngày làm: 2026-10-14 (~00:40 → 01:20) · Không thuê GPU (0 VND) · Nguồn: báo cáo P01–P06, [measurements.md](../../measurements.md),
> kết quả thô `results/gd1/*`, mã nguồn vLLM 0.30.0 (tag `v0.30.0` trên GitHub) ·
> Script phân tích: `bench/p07_analyze.py` (lệnh tái lập ở measurements.md mục "P07").

## 1. Tóm tắt

- **Mốc baseline** (vLLM 0.30.0 mặc định, 1 L4, SLO chính TTFT p95 ≤ 10 s / TPOT p95 ≤ 150 ms):

  | Workload | Goodput | Dao động đã đo | VND / 1 triệu token ra (giá tham chiếu 8,770 VND/h) |
  |---|---|---|---|
  | W1 chat | **68** user (72 trượt) | từng lần lặp: 68 / ≥ 72 / 68 | **3,960** (3,910–4,030 theo 3 lần lặp) |
  | W2 RAG | **18** user (20 trượt) | từng lần lặp: 18 / 20 / 18 | **14,460** (14,250–14,580) |
  | W3 tóm tắt | **6** user (8 trượt) | từng lần lặp: 6 / 6 / 6 | **24,030** (23,950–24,130) |
  | W4 tải giống thật, cả ngày | **18** ngày tải chồng (seed 1) | seed 2: 16; chạy lại cùng seed: **chưa đo** | **20,240** (seed 1) – **22,770** (seed 2) |

  Giá thuê thật dao động theo máy 7,560–9,900 VND/h, nên mọi con số chi phí có thể lệch thêm −14% … +13% (mục 3.5).
- **Chất lượng:** trung bình hai lần chạy MMLU-Pro 68.2 (69.0 / 67.4), IFEval 84.1, GSM8K 91.4, FinQA-RAG 59.7. Flips tự nhiên 1.6–6.6%.
- **Kế hoạch so với thực tế:** phần baseline tốn **13.2 h GPU ≈ 116k VND** (ước 16.25 h ≈ 166k). Giá thật thấp hơn giả định 14%.
  Throughput decode đúng tới −7…−15%; prefill thấp hơn giả định 44–61%. Ngân sách 2 tháng (≤ 3 triệu) vẫn đủ cho kịch bản A
  với dư ~150–450k tuỳ cách đo phần tune (mục 3.7).
- **Năm hướng tune có dấu hiệu thật** (mục 4), xếp theo lợi ích dự kiến: (1) tăng chỗ KV; (2) giữ lịch sử hội thoại bằng RAM CPU;
  (3) request dài chặn request ngắn ở hàng đợi; (4) preemption W1; (5) giữ khối `<think>` rỗng trong lịch sử. Loại khỏi danh sách: config
  MoE cho L4, speculative decoding, cờ fusion (không có dấu hiệu đo được).
- **Tiêu chí xong của plan:** bảng mốc + chi phí ✓; plan.md đã cập nhật ✓; danh sách nghi vấn có dẫn chứng ✓. Plan phần tune **chưa lập**,
  theo yêu cầu dừng để thảo luận (mục 6.3 là các câu hỏi cần quyết).

## 2. Đã làm gì và vì sao

1. **Gom số từ báo cáo P01–P06 và measurements.md.** Số nào có thể tính lại từ kết quả thô thì tính lại (`units.csv`, `bench.json.gz`,
   `metrics_*.txt`) để có thêm khoảng min–max giữa các lần lặp. Không đo gì mới trên GPU.
2. **Chọn một giá thuê tham chiếu** cho mọi phép tính VND / 1M token (mục 3.5). Lý do: giá các phiên chênh nhau tới 31% chỉ vì máy và phí tải
   model một lần. Nếu mỗi cấu hình dùng giá của phiên đo nó, chênh chi phí giữa hai cấu hình sẽ lẫn chênh giá máy (Hoefler & Belli 2015,
   quy tắc 9: ghi mọi yếu tố thay đổi; references G1).
3. **Bốn phân tích thêm trên dữ liệu đã có (0 GPU)** để kiểm tra ngược các nghi vấn:
   - W4: bao nhiêu token prompt là lịch sử hội thoại **dùng lại được** (mục 4.2).
   - W4: request ngắn bị chậm khi có request dài, **tách theo độ dài hàng đợi** để loại nguyên nhân "lúc đó đông" (mục 4.3).
   - W1: preemption tốn bao nhiêu (mục 4.4).
   - Nhiễu đo: so TTFT trung bình / p50 / p95, so cặp cùng seed giữa P04 và P05, p95 gộp 3 lần lặp kèm bootstrap (mục 6.1).

   Tất cả gộp trong `bench/p07_analyze.py` (đọc `units.csv`, `bench.json.gz`, `metrics_*.txt`, `metrics_timeseries.jsonl`, `cost.json`),
   chạy `.venv/bin/python bench/p07_analyze.py` ~1 phút ở laptop.
4. **Đọc mã nguồn vLLM 0.30.0** (đúng phiên bản đã ghim) để kiểm cơ chế và tìm cờ có sẵn: mặc định cho L4, cách bộ xếp lịch nhận request
   chờ, các cờ `watermark`, `long_prefill_token_threshold`, `policy`, `kv_offloading_size`, `scheduler_cls` (references F1–F3).
5. **Tra thêm tài liệu** cho các nghi vấn và cho phương pháp so sánh: Sarathi-Serve, CachedAttention, model card Qwen3, Hoefler & Belli,
   Cawley & Talbot. Lưu bản gốc vào `docs/references/files/`, chỉ mục ở [references nhóm F, G](../../references/README.md).
6. **Khác plan:**
   - Plan dự kiến thử "config MoE cho L4" và "speculative decoding". Cả hai bị loại vì không có dấu hiệu đo được (mục 4.6).
   - Plan nói "sẵn sàng lập plan tune". Theo yêu cầu của user, P07 dừng ở danh sách nghi vấn + câu hỏi, chưa lập plan.

## 3. Kết quả

### 3.1 Bảng mốc baseline

Mọi số dưới đây là vLLM 0.30.0 + `--max-model-len 16384`, model `Qwen/Qwen3-30B-A3B-GPTQ-Int4@9b534e43`, L4 host 506934.

| | W1 chat | W2 RAG | W3 tóm tắt | W4 tải giống thật (peak 16h) |
|---|---|---|---|---|
| Prompt / output TB (token) | 407 / 333 | 2,858 / 174 (622 dùng chung) | 5,617 / 299 | ~767 / ~272, đổi theo giờ |
| **Goodput SLO chính** | **68 (72 trượt)** | **18 (20 trượt)** | **6 (8 trượt)** | **k = 18 (20 trượt)** seed 1; **16 (18 trượt)** seed 2 |
| Goodput từng lần lặp / bộ ngày | 68 / ≥ 72 / 68 | 18 / 20 / 18 | 6 / 6 / 6 | 18 / 16 (mỗi seed chạy 1 lần) |
| Goodput SLO phụ (2 s / 100 ms) | 32 (64) | < 8 | < 4 | < 16 |
| TTFT p95 tại goodput (trung vị [min–max]) | 8.30 s [7.28–8.36] | 7.42 s [6.44–9.68] | 5.14 s [3.45–5.91] | 9.11 s (1 lần) |
| TPOT p95 tại goodput | 104 ms [102–106] | 90 ms [90–93] | 57 ms [57–60] | 82 ms |
| Token ra / giây tại goodput | 615 [605–623] | 168 [167–171] | 101 [101–102] | 259 (TB 30 phút) |
| Request / giờ tại goodput | 6,570 | 3,540 | 1,220 | 3,476 (363 phiên đang mở) |
| Chỉ số hỏng trước | TTFT | TTFT | TTFT | TTFT, theo đợt |
| Nguồn | measurements "P04"; `results/gd1/04-.../units.csv` | như W1 | như W1 | measurements "P05"; `results/gd1/05-.../summary.md` |

Chất lượng (P06, hai lần chạy cùng cấu hình, cùng máy 116596; measurements "P06" và "P06 lần 2"):

| Bài | Lần 1 / lần 2 | Trung bình | CI 95% một lần chạy (±) | Flips tự nhiên | Qwen công bố (bản gốc) |
|---|---|---|---|---|---|
| MMLU-Pro (1,000 câu) | 69.0 / 67.4 | 68.2 | ±2.9 | 6.6% | 69.1 |
| IFEval prompt strict (541) | 84.7 / 83.5 | 84.1 | ±3.0 | 5.2% | 83.7 |
| GSM8K flexible (1,319) | 91.4 / 91.4 | 91.4 | ±1.5 | 1.6% | — |
| FinQA-RAG dung sai 1% (1,147) | 59.2 / 60.1 | 59.7 | ±2.8 | 4.5% | — |

Mức tự lệch (1,000 prompt, greedy 256 token): giống hệt 23.1–24.5%, vị trí lệch trung vị token 46; teacher forcing same top-1 99.10% ± 0.09.

### 3.2 W1–W3 theo từng mức user

Trung vị 3 lần lặp [min–max]. Nguồn: `results/gd1/04-do-baseline-day-du/20261013-p04/units.csv`, cửa sổ "all".

| w | N | TTFT p95 (s) | TPOT p95 (ms) | Token ra / s | Token vào / s |
|---|---|---|---|---|---|
| W1 | 8 | 0.43 [0.42–0.43] | 33 [32–34] | 243 [239–248] | 291 [259–301] |
| W1 | 16 | 0.78 [0.77–1.11] | 48 [47–48] | 323 [323–340] | 398 [350–401] |
| W1 | 32 | 1.77 [1.76–2.04] | 66 [65–68] | 458 [451–474] | 541 [510–550] |
| W1 | 64 | 5.89 [4.79–5.90] | 98 [97–102] | 610 [609–621] | 747 [717–751] |
| **W1** | **68** | 8.30 [7.28–8.36] | 104 [102–106] | 615 [605–623] | 746 [728–758] |
| W1 | 72 | 11.14 [9.20–12.29] | 104 [104–105] | 612 [611–632] | 757 [727–771] |
| W1 | 128 | 41.90 [39.71–42.33] | 113 [111–115] | 620 [618–621] | 760 [742–768] |
| W2 | 8 | 2.03 [1.94–2.28] | 57 [56–57] | 146 [145–146] | 2,376 [2,372–2,451] |
| W2 | 16 | 6.03 [4.87–6.30] | 90 [90–93] | 169 [167–171] | 2,777 [2,714–2,854] |
| **W2** | **18** | 7.42 [6.44–9.68] | 90 [90–93] | 168 [167–171] | 2,775 [2,720–2,850] |
| W2 | 20 | 10.06 [8.76–11.07] | 93 [90–94] | 168 [166–170] | 2,759 [2,710–2,847] |
| W2 | 64 | 53.76 [53.31–55.62] | 91 [91–93] | 172 [170–172] | 2,788 [2,757–2,795] |
| W3 | 4 | 2.15 [1.96–2.47] | 43 [43–44] | 93 [93–93] | 1,766 [1,764–1,768] |
| **W3** | **6** | 5.14 [3.45–5.91] | 57 [57–60] | 101 [101–102] | 1,935 [1,923–1,941] |
| W3 | 8 | 13.06 [11.59–13.84] | 59 [58–60] | 102 [102–102] | 1,945 [1,940–1,950] |
| W3 | 64 | 176.57 [173.08–177.14] | 60 [60–61] | 103 [102–104] | 1,944 [1,940–1,953] |

Các mức khác (W1 80/96, W2 24/32, W3 16/32): [báo cáo P04](../04-do-baseline-day-du/bao-cao.md) bảng 3.1.

### 3.3 W4 tóm tắt

| Lần chạy | Kết quả | TTFT p95 | TPOT p95 | Token ra / s |
|---|---|---|---|---|
| peak seed 1, k = 16 / 18 / 20 / 24 | đạt / **đạt** / trượt / trượt | 5.49 / **9.11** / ≥ 30.9 / ∞ s | 76 / 82 / ≥ 191 / ∞ ms | 243 / 259 / — / — |
| peak seed 2, k = 16 / 18 | **đạt** / trượt | 0.87 / ∞ s | 73 / ∞ ms | 259 / — |
| peak2 (10h), normal (13h), k = 18 | đạt / đạt | 0.62 / 0.56 s | 54 / 36 ms | 194 / 96 |

Nguồn: measurements "P05". Mỗi (seed, k) chỉ chạy **một lần**. Ở seed 1, k = 18 đạt với TTFT p95 9.11 s, chỉ cách ngưỡng 9%,
trong khi TTFT p95 của một lần đo W1–W3 dao động 16–23% (mục 6.1). Vì vậy goodput 18 của seed 1 **chưa chắc lặp lại được**.

### 3.4 Thời gian GPU và tiền đã tiêu

| Phiên | Máy (machine) | Giờ tính tiền | Tín dụng bị trừ | $/h thực | VND/h thực | Nguồn |
|---|---|---|---|---|---|---|
| P01 | 116596 | 0.346 | $0.126 | 0.364 | 9,540 | measurements "P01" (hoá đơn: GPU + ổ đĩa + tải về 21 GB) |
| P03 + P04 | 116596 | 5.251 | $1.516 | 0.289 | 7,560 | measurements "P04" (hoá đơn) |
| P05 | 150535 | 4.97 | $1.878 | 0.378 | 9,900 | measurements "P05" (tín dụng trước/sau) |
| P06 | 116596 | 1.78 | $0.592 | 0.333 | 8,710 | measurements "P06" |
| P06 lần 2 | 116596 | 0.89 | $0.320 | 0.359 | 9,400 | measurements "P06 lần 2" |
| **Tổng** | | **13.24** | **$4.432 ≈ 116,100 VND** | **0.335** | **8,770** | |

- Tín dụng tài khoản giảm từ $10.000 xuống $5.549 = **$4.451 ≈ 116,600 VND**. Phần chênh $0.019 so với tổng các phiên gồm phiên thử máy
  ngày 12/10 ($0.0038) và $0.014 bị trừ giữa P04 và P05 (8.353 → 8.339) chưa rõ khoản gì (có thể phí ổ đĩa ghi hoá đơn trễ).
- Hai phiên ngắn (P01, P06 lần 2) có $/h cao hơn vì phí tải model ~$0.027 mỗi máy mới chia cho ít giờ.

### 3.5 Chi phí VND / 1 triệu token ra

**Giá tham chiếu: 8,770 VND/h** = tổng tín dụng của 5 phiên ÷ tổng giờ ($4.432 / 13.24 h = $0.335/h × 26,200). Đã gồm ổ đĩa và phí tải về.
Dùng một giá cố định cho mọi cấu hình của đồ án, để chênh chi phí chỉ phản ánh chênh throughput. Độ nhạy theo giá: thấp nhất 7,560 (máy 116596,
phiên dài), cao nhất 9,900 (máy 150535).

Công thức W1–W3: `giá (VND/h) ÷ (token ra / s × 3,600 / 10⁶)` tại mức goodput. Khoảng trong ngoặc là min–max của 3 lần lặp.
W4: `giá × 24 h ÷ token ra cả ngày`, với token ra cả ngày = k × 577,632 (token ra trung bình mỗi ngày của trace, theo đủ 24 giờ;
`results/gd1/05-.../cost.json`).

| Workload | Tại | **VND / 1M token ra** (giá 8,770) | Theo 3 lần lặp | Giá 7,560 … 9,900 | VND / 1M token vào + ra | VND / 1,000 request |
|---|---|---|---|---|---|---|
| W1 | 68 user | **3,960** | 3,910–4,030 | 3,420 … 4,470 | 1,790 | 1,340 |
| W2 | 18 user | **14,460** | 14,250–14,580 | 12,470 … 16,320 | 830 | 2,480 |
| W3 | 6 user | **24,030** | 23,950–24,130 | 20,720 … 27,130 | 1,200 | 7,190 |
| W4 cả ngày | k = 18 (seed 1) | **20,240** | seed 2 (k = 16): 22,770 | 17,450 … 25,710 | — | 5,550 (seed 2: 6,240) |

Ghi chú khi đọc bảng:
- **W1–W3 là mức trần**, vì closed-loop giữ GPU bận 100% thời gian. Doanh nghiệp thật không có tải đều như vậy cả ngày.
  W4 cả ngày là con số thực tế hơn, vì tính cả giờ đêm gần như không có người dùng.
- **W4 so với báo cáo P05 (22,510):** P05 dùng giá niêm yết của máy 150535 ($0.372/h = 9,752 VND/h). Tính lại bằng giá tham chiếu ra 20,240.
  Hai số khác nhau chỉ do giá, cùng một throughput.
- **Chi phí ở mức goodput gần như không phụ thuộc goodput là 18 hay 20.** Ở W1–W3, throughput đã đi ngang từ trước điểm gãy (W2: 168–169 token/s
  từ 16 đến 64 user). Nhiễu goodput không làm nhiễu chi phí.
- Token ra W4 tính theo trace, khớp với đo được: client dùng `ignore_eos`, nên mọi request sinh đúng `max_tokens` của trace
  (kiểm 1,738 request peak k = 18: tổng token ra đo được = tổng `max_tokens`).

### 3.6 Kế hoạch so với thực tế

**Giả định của [plan.md](../../plan.md):**

| Tham số | Giả định (12/10) | Đo được | Chênh | Nguồn |
|---|---|---|---|---|
| Decode W1 @ 8 / 16 / 32 / 64 user (token ra / s) | 260 / 380 / 520 / 600 | 243 / 323 / 458 / 610 | −7% / −15% / −12% / +2% | P04 |
| Prefill (token vào / s lúc bão hoà) | ~5,000 | W2 2,777 [2,714–2,854]; W3 1,935–1,945 | −44% / −61% | P04 |
| Đơn giá trọn gói | ~10,200 VND/h | 8,770 (7,560–9,900) | −14% | mục 3.4 |
| Hệ số GPU chờ (thời gian máy / thời gian đo) | ×1.3 | ~1.1 (P05: 4.97 h máy / 4.49 h chạy; P04: 4.44 / 4.35) | thấp hơn | P04, P05 |

Lý do prefill thấp (P04 mục 4.3): GPU luôn ở trần công suất 72 W, xung SM ~1,100 MHz (một nửa tối đa); bước prefill còn gánh decode của request
khác; attention của prompt 4–8k token. Chưa tách được phần nào lớn hơn (cần profile, GĐ3).

**Giờ GPU từng plan:**

| Plan | Ước ban đầu | Thực tế | Ghi chú |
|---|---|---|---|
| P01 cài đặt | 0.75 h | 0.35 h | |
| P03 harness + smoke | 1 h | 0.84 h | |
| P04 W1–W3 đầy đủ ×3 | 6.5 h | 4.44 h | phần mở rộng ít hơn dự kiến |
| P05 W4 | 6.5 h | 4.97 h | dừng sớm khi chắc trượt tiết kiệm ~1 h |
| P06 chất lượng | 1.5 h | 1.78 + 0.89 h | thêm lần 2 để đo flips tự nhiên |
| **Tổng** | **16.25 h ≈ 166k VND** | **13.24 h ≈ 116k VND** | −19% giờ, −30% tiền |

**Đơn vị chi phí trong plan.md so với thực tế:**

| Đơn vị | plan.md | Thực tế / ước lại | Cách tính |
|---|---|---|---|
| Sweep đầy đủ (W1–W3 ×3 + W4) | 11.5 h | **9.4 h** | P04 4.44 + P05 4.97 |
| Sweep rút gọn (W1–W3 ×1, khởi đầu nóng ~4 mức) | 0.7 h | **~0.8 h** | 4 mức × (W1 2.8 + W2 2.1 + W3 5.4 phút mỗi đơn vị tại goodput) + ~7 phút khởi động lại (cấu hình mới phải compile lại, rồi bật lại lần 2) |
| W4 một seed, khởi đầu nóng (18, 20) | — | **~1.5–2.2 h** | lần đạt ~41 phút, lần trượt dừng sớm 19–28 phút |
| Bộ đo chất lượng | 1 h | **~0.8 h một lần; ~1.6 h nếu chạy hai lần** | teacher forcing 2 + sinh tham chiếu c8 19 + FinQA 10 + MMLU-Pro 14 phút; lý do chạy hai lần: mục 6.1 |
| Setup máy mới | 0.25 h | ~0.2 h | cài 3–4 phút + đẩy code + 2 lần khởi động |

### 3.7 Ngân sách

Đã tiêu tới hết GĐ1 baseline: **~116.6k VND** (tín dụng giảm $4.451). Ước lại cả dự án với giá 8,770 VND/h (chi tiết ở [plan.md](../../plan.md)
mục "Ước lại 2026-10-14"):

| Kịch bản | Giờ thuần | Có ×1.3 chờ + 25% chạy lại | So với ngân sách |
|---|---|---|---|
| A, phần tune đo kiểu rẻ (~31 h) | ~179 h | ~2.55 triệu | vừa **2 tháng** (≤ 3 triệu), dư ~450k |
| A, phần tune đo kiểu chặt (~52 h) | ~200 h | ~2.85 triệu | vừa 2 tháng, dư ~150k |
| B, tune chặt | ~248 h | ~3.53 triệu | vừa **3 tháng** (≤ 4 triệu), dư ~470k |

- Ngân sách là ngưỡng cảnh báo, không phải trần cứng. Hiện **không có cảnh báo**: giá thật thấp hơn giả định bù được phần tune có thể tốn hơn.
- **Rủi ro giá:** nếu chỉ thuê được máy giá như P05 (9,900 VND/h), kịch bản A tune chặt lên ~3.2 triệu, vượt mốc 2 tháng ~200k.
  Nguồn cung L4 trên Vast mỏng (~5 máy, phần lớn cùng host 506934).
- Phụ phí ncu trên AWS g6 (100–400k) vẫn chưa tính (challenges #1).

## 4. Danh sách nghi vấn tune

Mỗi nghi vấn: dấu hiệu đo được → cơ chế (đo được / đã kiểm trong mã / phỏng đoán) → kiểm tra ngược (dấu hiệu có thể do nguyên nhân khác không)
→ còn thiếu đo gì → tham số thử. Xếp theo lợi ích dự kiến có dẫn chứng. Bảng tổng hợp ở mục 4.7.

### 4.1 N1 — Tăng chỗ KV cache (W1, W2, W3, W4)

**Dấu hiệu đo được.**
- Số request chạy cùng lúc dừng ở ~60 / ~15 / ~6 (W1 / W2 / W3), bằng KV 37,264 token chia cho kích thước một request (P04 mục 4.1).
- TTFT tại goodput chủ yếu là chờ trong hàng đợi: 83% (W1/68), 73% (W2/18) (server tự chia, P04).
- KV dùng tối đa 100% ở mọi mức từ goodput trở lên.
- P03: cùng máy, cùng cờ, lần khởi động có KV ít hơn 23% (28,704 token) cho TTFT p95 W2/16 = 9.91 s, so với 4.58–5.55 s khi KV đủ;
  TPOT p95 75.7 ms so với 89–93 ms; token ra / s 161 so với 168–170. **Ít KV hơn → TTFT tăng gần gấp đôi, TPOT giảm.**

**Cơ chế (suy ra từ số đo).** Closed-loop với N user: N = (request / s) × (thời gian một request). Khi số request chạy được bị KV chặn,
phần dư nằm trong hàng đợi, nên thời gian chờ dồn hết vào TTFT. Nhiều KV hơn thì nhiều request chạy song song hơn: thời gian chờ chuyển từ
"trước token đầu" sang "giữa các token". SLO TPOT còn dư (90–104 ms so với 150), nên chuyển như vậy có lợi cho goodput.

**Ước lợi ích tối đa (suy ra, chưa đo).** Giả sử throughput giữ ở mức bão hoà hiện tại, goodput bị chặn bởi lúc TPOT chạm ngưỡng:
N_max ≈ request/s × (TTFT TB ~3.5 s + TPOT TB cho phép ~134 ms × số token ra). Lấy TPOT TB cho phép = 150 / 1.12 vì ở W1/68 TPOT p95 / p50 ≈ 1.12.

| | request/s bão hoà | N_max ước | Hiện tại | KV cần để chạy N_max song song | KV FP8 (~74.5k token) đủ? |
|---|---|---|---|---|---|
| W1 | 1.83 | ~88 | 68 | ~55k | đủ |
| W2 | 0.98 | ~26 | 18 | ~65k | đủ |
| W3 | 0.34 | ~15 | 6 | ~88k | chưa đủ (~12 request) |

Số này **chưa chắc**: throughput có thể tăng khi batch lớn hơn (decode MoE chia chi phí đọc weight cho nhiều token hơn), hoặc giảm
(giải lượng tử FP8, đổi backend attention). Chỉ dùng để xếp hạng.

**Kiểm tra ngược.**
- Trần ~60 request ở W1 có thể do `max_num_seqs`? Không: mặc định trên L4 (24 GB < 70 GB) là `max_num_seqs` 256, `max_num_batched_tokens` 2,048
  (đã kiểm `vllm/engine/arg_utils.py` v0.30.0, `get_batch_defaults`; references F3).
- Bằng chứng P03 chỉ là **một lần đo** (r1) so với hai lần (r2, r3), lại khác lần khởi động. Hướng ảnh hưởng rõ (gấp đôi, vượt nhiễu ~20%),
  nhưng độ lớn chưa chắc.
- KV FP8 trên Ada: FlashAttention 2 không hỗ trợ KV FP8, vLLM sẽ chuyển sang FlashInfer hoặc Triton (library-gaps #3; issue #60262 có báo crash).
  Đổi backend tự nó có thể đổi tốc độ. **Phải tách hai hiệu ứng**: đo thêm cấu hình "cùng backend mới, KV FP16".
- KV FP8 đổi số học của attention → bắt buộc qua cổng chất lượng (P06 mục 6). Tài liệu vLLM: không hiệu chuẩn thì mọi scale = 1.0;
  khuyến nghị hiệu chuẩn bằng llm-compressor (references F2).

**Tham số thử:** `--kv-cache-dtype fp8` (e4m3); đối chứng `--attention-backend FLASHINFER` / `TRITON_ATTN` với KV FP16;
`--gpu-memory-utilization 0.94–0.96` (mỗi 0.01 ≈ 0.22 GiB ≈ 2,400 token FP16); `--kv-cache-memory-bytes` (tên cờ trong v0.30.0; tài liệu gọi là `--kv-cache-memory`; cố định dung lượng KV, tránh lệch KV
giữa các lần khởi động; tài liệu vLLM, references F1).

### 4.2 N2 — Giữ lịch sử hội thoại trong RAM CPU (W4)

**Dấu hiệu đo được (P05 + phân tích mới hôm nay).**
- Trúng prefix cache ở peak k = 18 chỉ **3.0%** token prompt (server). Lượt sau với think time ≥ 120 s: 0% lịch sử trúng.
- **Mới (0 GPU):** tính trên từng request của các lần chạy W4, phần prompt là lịch sử của lượt trước **đã chạy trong cùng lần đo**, không bị client
  cắt bớt, tức về nguyên tắc dùng lại được:

  | Lần chạy | Trúng cache thật | Dùng lại được (lý thuyết) | Trong đó: câu trả lời lượt trước |
  |---|---|---|---|
  | peak seed 1, k = 16 | 4.4% | **45.8%** | 13.5% |
  | peak seed 1, k = 18 | 3.0% | **44.7%** | 13.4% |
  | peak seed 2, k = 16 | 3.7% | **44.2%** | 14.6% |
  | peak2 k = 18 | 9.9% | 46.4% | 14.0% |
  | normal k = 18 | 14.5% | 42.1% | 12.6% |

  Cách tính: lượt sau của cùng phiên có `input_len ≥ input_len trước + output_len trước` thì phần "lịch sử dùng lại được" = prompt trước + câu trả lời
  trước − 4 token (khối `<think>` rỗng); lượt bị client cắt lịch sử tính 0. Tính trên mọi request đã gửi (gồm 10 phút khởi động).

**Cơ chế.** Ở k = 18 có 363 phiên mở, lịch sử ~770 token mỗi phiên, tổng ~280k token, gấp ~7 lần KV trên GPU (P05 mục 4.4).
Bộ đệm prefix của vLLM đẩy khối ít dùng gần nhất ra khi cần chỗ (LRU, tài liệu thiết kế prefix caching của vLLM, references F1).

**Ước khả thi (suy ra).** ~280k token × 96 KiB/token (KV FP16, P01) ≈ **26 GiB**, vừa RAM máy thuê (94–125 GB). Chép lại 1,000 token KV
(96 MB) qua PCIe Gen3 đo được 8.4 GB/s mất ~11 ms, so với prefill 1,000 token ~370 ms (ước từ P04). CachedAttention (ATC'24, references F5)
báo giảm TTFT tới 87% cho hội thoại nhiều lượt bằng cách giữ KV ở bộ nhớ chậm hơn GPU.

**Kiểm tra ngược.**
- **Đợt dồn chủ yếu là nội dung mới.** Phút 16 (phút hỏng nặng nhất ở k = 18) có 85,318 token prompt nhưng chỉ 38% là lịch sử dùng lại được,
  so với trung vị ~51% ở các phút khác. Dù dùng lại hết lịch sử, phút đó vẫn còn ~53k token prefill mới, gấp ~2.5 lần phút bình thường.
  → Giữ lịch sử giảm **tải trung bình** gần một nửa phần prefill, nhưng **không xoá đợt dồn**. Goodput W4 tăng bao nhiêu thì phải đo.
- Nội dung các lượt trong W4 ghép từ nhiều nguồn (workloads.md "Hạn chế"). Với prefix cache chỉ độ dài và cấu trúc prefix quan trọng, phần này là thật.
- 13.4% là câu trả lời lượt trước, **không bao giờ trúng** vì template (N5). Nếu không sửa template, phần dùng lại được chỉ còn ~31%.

**Tham số thử (có sẵn trong vLLM 0.30.0, đã kiểm `vllm/config/cache.py`):** `--kv-offloading-size <GiB>` với `--kv-offloading-backend native`
(hoặc `lmcache`, để GĐ2 so sánh). Chỉ đo được bằng W4 (W1–W3 không có hội thoại nhiều lượt).

### 4.3 N3 — Request dài chặn request ngắn ở hàng đợi (W4; cả W2/W3)

**Dấu hiệu đo được (P05 mục 3.6).** Request < 1k token có TTFT p95 26.6–32.2 s nếu được gửi lúc có request ≥ 4k đang chờ token đầu; 1.4–4.2 s nếu không.

**Cơ chế đã kiểm trong mã vLLM 0.30.0 (`vllm/v1/core/sched/scheduler.py`, references F3):**
1. Bộ xếp lịch xét hàng đợi theo thứ tự đến (`policy = "fcfs"` mặc định).
2. `scheduler_reserve_full_isl = True` (mặc định): request mới chỉ được nhận khi **cả prompt** vừa KV còn trống, không chỉ khúc đầu.
3. Request đứng đầu hàng không cấp được KV thì vòng lặp dừng (`if new_blocks is None: ... break`): **mọi request phía sau phải chờ**,
   kể cả request ngắn vừa chỗ trống.
→ Một prompt 5k token đứng đầu hàng lúc KV gần đầy sẽ chặn cả hàng tới khi ~5k token KV được giải phóng.

**Kiểm tra ngược: có phải chỉ vì lúc đó đông?** Request dài đang chờ thường rơi vào lúc hàng đợi dài, nên tương quan trên có thể chỉ do tải chung.
Phân tích mới (0 GPU): tách request ngắn theo số request đang chờ trên server lúc gửi (`metrics_timeseries.jsonl`, căn thời gian gần đúng ±1–2 s):

| peak seed 1 k = 18 | Không có request ≥ 4k đang chờ | Có |
|---|---|---|
| hàng đợi 0 | p50 0.21 s / p95 0.49 s (n = 1,081) | p50 1.10 / p95 3.06 (n = 34) |
| hàng đợi 4–9 | p50 3.70 / p95 15.3 (n = 69) | p50 7.08 / p95 15.2 (n = 16) |
| hàng đợi ≥ 10 | p50 9.84 / p95 17.5 (n = 42) | p50 22.9 / p95 38.7 (n = 31) |

(k = 16 seed 1 cho cùng xu hướng: hàng đợi ≥ 10 → p50 10.0 so với 24.5 s.)
- Cùng độ dài hàng đợi, request ngắn vẫn chậm hơn ~2 lần (p50) khi có request dài đang chờ. Hiệu ứng **không chỉ do đông**.
- Nhưng mẫu nhỏ (n = 16–34 ở nhóm "có"), và đây vẫn là tương quan. Ở hàng đợi 0 còn có nguồn thứ hai: chunked prefill của prompt dài chiếm ngân sách
  2,048 token mỗi bước.
- **Còn thiếu:** thí nghiệm có kiểm soát (ví dụ chạy W1 ở mức dưới goodput, chèn request 6k token theo lịch cố định, đo TTFT request ngắn
  quanh lúc chèn). Hoặc bật/tắt `scheduler_reserve_full_isl` trên W4 cùng seed.

**Tham số thử (đều có trong vLLM 0.30.0):**
- `scheduler_reserve_full_isl` = false (cờ `--scheduler-reserve-full-isl`, kiểm cú pháp tắt khi chạy): nhận request khi khúc đầu vừa chỗ. Rủi ro: nhận quá nhiều rồi preemption (chú thích trong mã nói cờ này
  "prevents over-admission and KV cache thrashing").
- `--max-num-batched-tokens` 1,024 / 4,096 / 8,192 và `--long-prefill-token-threshold`: hai tác động ngược nhau. Tài liệu vLLM: nhỏ thì ITL tốt,
  lớn thì TTFT tốt, > 8,192 cho throughput (references F1). Sarathi-Serve: chọn ngân sách lớn nhất mà không vượt SLO TBT (references F4).
  TPOT còn dư (82–104 ms so với 150) nên có chỗ tăng.
- `--scheduling-policy priority` + client gán độ ưu tiên theo độ dài prompt. Đây là thay đổi phía ứng dụng (ai gán độ ưu tiên), phải ghi rõ khi so.
- Nếu các cờ trên không đủ: **ứng viên GĐ4** (mục 6.2).

### 4.4 N4 — Preemption ở W1

**Dấu hiệu đo được.** 83–151 lần preemption mỗi lần đo ở W1/64–72 (P04). W2, W3 chỉ 0–19 lần.

**Kiểm tra ngược: tốn bao nhiêu?**
- Thử đếm token phải tính lại từ bộ đếm của server: `iteration_tokens_total` = token prompt − token trúng cache + token sinh ra, khớp tới từng token
  ở mọi đơn vị W1/68 (sai lệch = đúng số token trúng cache). Nghĩa là **bộ đếm của vLLM không tính phần tính lại sau preemption**.
  Chi phí thật của preemption **chưa đo được**.
- Ước cận trên (suy ra): ~130 lần × ~565 token (prompt + nửa output) ≈ 73k token mỗi lần đo ≈ 27 s GPU / 147 s (~18%) nếu phải tính lại hết.
  Cận dưới ~0 nếu khối KV vừa giải phóng vẫn còn trong prefix cache khi request chạy lại. Khoảng quá rộng để xếp hạng cao.
- Thời gian prefill TB phía server tăng từ 155 ms (W1/8) lên 450 ms (W1/68), một phần có thể do tính lại. Chưa tách được.

**Tham số thử:** `--watermark` 0.01–0.05 (để dành một phần KV khi nhận request mới, chú thích trong mã nói để tránh preemption lặp; mặc định 0;
references F3), `--max-num-seqs` 56–64. Rủi ro: nhận ít request hơn → hàng đợi dài hơn → TTFT có thể tệ hơn. Đo bằng W1.

**Còn thiếu:** đếm token tính lại (log mức debug của scheduler, hoặc so throughput W1/68 khi `max_num_seqs` = 56 chặn hẳn preemption).

### 4.5 N5 — Giữ khối `<think></think>` rỗng trong lịch sử (W4)

**Dấu hiệu:** 0% lượt trúng phần câu trả lời trước (P05); phần này là 13.4% token prompt ở peak k = 18 (mục 4.2). Cơ chế đã kiểm ở
`workloads/test_w4_session.py`: template Qwen3 bỏ khối `<think></think>` của lượt trước, mọi token từ đó lệch vị trí.

**Kiểm tra ngược.** Qwen khuyến nghị lịch sử **chỉ chứa phần trả lời cuối**, không chứa nội dung thinking ("No Thinking Content in History",
model card Qwen3-30B-A3B, references F6). Khối rỗng không có nội dung, nhưng giữ nó là đổi prompt so với cách dùng khuyến nghị.
Bộ đo chất lượng P06 **không có bài nhiều lượt** nên chưa có cổng chất lượng phù hợp. Lợi ích chỉ có khi đi cùng N2 (không có chỗ chứa thì
lịch sử vẫn bị đẩy ra).

**Tham số:** sửa chat template phía client (`chat_template` tuỳ biến). Thay đổi phía ứng dụng → báo riêng, không gộp vào "vLLM đã tune".

### 4.6 Các nghi vấn bị loại (không có dấu hiệu đo được)

| Nghi vấn trong plan | Vì sao loại | Khi nào xét lại |
|---|---|---|
| Config MoE cho L4 (library-gaps #4) | vLLM 0.30.0 chọn backend MoE **Marlin** (log P01). File config JSON chỉ dùng cho đường Triton. Không có số đo nào cho thấy MoE là nghẽn | GĐ3 profile: nếu kernel MoE chiếm nhiều thời gian và đường Triton có triển vọng |
| Speculative decoding (TPOT cao ở batch lớn) | TPOT p95 lớn nhất 117 ms < 150 ở mọi mức đã đo; mọi workload hỏng vì TTFT. Spec decode thêm tính toán mỗi bước và (với model nháp) tốn VRAM vốn đã thiếu cho KV | Nếu sau khi tăng KV, TPOT trở thành chỉ số hỏng trước |
| Cờ fusion (`enable_qk_norm_rope_fusion`, library-gaps #5) | Không có dấu hiệu; lợi ích dự kiến vài %, dưới độ phân giải goodput | Có thể thử nhanh bằng throughput bão hoà (nhiễu 1.6%) nếu rẻ |
| Đổi backend attention cho prefill dài | Chưa có dấu hiệu riêng; sẽ được đo như đối chứng của N1 | — |

### 4.7 Bảng xếp hạng

| # | Nghi vấn | Workload bị ảnh hưởng | Dấu hiệu (nguồn) | Lợi ích dự kiến | Mức chắc chắn | Còn thiếu đo |
|---|---|---|---|---|---|---|
| N1 | Tăng chỗ KV (FP8, `gpu_memory_utilization`) | W1, W2, W3, W4 | trần chạy = KV/kích thước; TTFT 73–83% là chờ; P03 KV −23% → TTFT ×2 | goodput cận trên +30% (W1) / +45% (W2) / ×2–2.5 (W3) | cao về hướng, thấp về độ lớn | đối chứng backend; cổng chất lượng |
| N2 | Giữ lịch sử ở RAM CPU | W4 | 45–50% token prompt dùng lại được, trúng 3–5% | giảm tới ~45% token prefill ở giờ cao điểm; goodput chưa rõ vì đợt dồn | TB | chỉ đo được bằng W4 |
| N3 | Request dài chặn hàng đợi + ngân sách token | W4 (W2, W3) | TTFT request ngắn ×2 (p50) cùng độ dài hàng đợi; cơ chế đã thấy trong mã | chưa ước được | TB (cơ chế chắc, độ lớn chưa) | thí nghiệm có kiểm soát |
| N4 | Preemption W1 | W1 | 83–151 lần / lần đo | 0–18% thời gian GPU ở W1/68 | thấp | đếm token tính lại |
| N5 | Giữ `<think>` rỗng | W4 (cùng N2) | 13.4% token prompt không bao giờ trúng | ≤ 13% token prefill, chỉ khi có N2 | TB | cổng chất lượng nhiều lượt |

## 5. Vấn đề gặp phải

| Vấn đề | Xử lý |
|---|---|
| Bộ đếm `/metrics` của vLLM không tính token phải tính lại sau preemption | Ghi rõ chi phí preemption chưa đo được (mục 4.4); đề xuất cách đo |
| P05 tính chi phí W4 bằng giá niêm yết ($0.372/h), không phải tín dụng thực bị trừ | Tính lại bằng giá tham chiếu chung; số cũ giữ nguyên trong measurements |
| plan.md có một khối ghi chú chen giữa bảng giả định, làm vỡ bảng | Chuyển ghi chú xuống dưới bảng, nội dung giữ nguyên |
| Trang tài liệu docs.vllm.ai trả lỗi 429 (giới hạn truy cập) | Lấy đúng các file đó từ GitHub ở tag `v0.30.0` (khớp phiên bản đã ghim, tốt hơn bản "latest") |
| Căn thời gian giữa chuỗi số đo server (mỗi giây) và thời điểm gửi của client chỉ gần đúng (±1–2 s) | Ghi rõ ở mục 4.3; đủ cho phân nhóm theo độ dài hàng đợi |

## 6. Ảnh hưởng tới bước sau

### 6.1 Những điểm của cách đo hiện tại có thể làm so sánh khi tune bị sai

| # | Rủi ro | Bằng chứng | Cách chặt hơn | Chi phí thêm |
|---|---|---|---|---|
| R1 | **Goodput W4 baseline chỉ có một lần chạy mỗi seed.** Seed 1, k = 18 đạt với TTFT p95 9.11 s, cách ngưỡng 9% | Nhiễu TTFT p95 một lần đo W1–W3: trung vị 16%, gần ngưỡng 23% (P04) | Chạy lại peak seed 1 k = 18 và seed 2 k = 16 một lần nữa, **trước** khi so cấu hình tune trên W4 | ~1.4 h ≈ 12k VND |
| R2 | **Một lần đo W1–W3 không phân biệt được chênh 1 bước tìm** (2 user W2/W3, 4 user W1) | TTFT p95 gần ngưỡng lệch 16–48% giữa 3 lần lặp; W2/20 lần lượt 11.07 / 8.76 / 10.06 s | Đo lặp ×3 ở lo / hi; chỉ kết luận khi chênh ≥ 2 bước hoặc khoảng không chồng nhau (đã là quy tắc workloads.md) | sweep rút gọn ×3 ~2.2 h thay vì ~0.8 h mỗi cấu hình |
| R3 | **Chọn người thắng trong ~10 cấu hình đo một lần → kết quả của người thắng bị đẩy lên do nhiễu** | Cawley & Talbot 2010 (references G2): tối ưu theo một tiêu chí có phương sai thì người được chọn bị lệch lạc quan, có thể ngang chênh lệch thật giữa các phương án | Đo lại cấu hình thắng trên **seed mới** (1004–1006) và W4 seed 2, không dùng lại số của vòng sàng lọc | ~2–4 h mỗi cấu hình vào chung kết |
| R4 | **So cặp cùng seed chặt hơn so khác seed** | Cùng seed W2/16, P05 so P04 (khác máy): TTFT p95 −9.3% / +2.0% / −5.0%; throughput +2.3 / +2.4 / +1.8% (đều đặn). Khác seed trong P04: TTFT p95 rải 4.87–6.30 s (24% so với trung vị). Seed r3 của W2 có prompt TB dài hơn r1 3.6% → throughput luôn thấp nhất | Mọi cấu hình dùng **đúng** seed 1001–1003 như baseline, so hiệu theo từng cặp seed. Chỉ có 3 cặp nên chưa đủ để định lượng; cần thêm số | 0 (đã là thiết kế harness); ghi rõ thành quy tắc |
| R5 | **Khác máy / khác lần khởi động đổi KV** | KV 37,264 (machine 116596) vs 38,032 (150535), +2%; lần compile đầu −23% (challenges #4) | Điểm mốc W2/16 ×3 đầu mỗi phiên (đã có); nếu khác máy P04 thì chạy lại baseline ở lo / hi cùng phiên; ghi KV mỗi lần khởi động (đã có). Cân nhắc `--kv-cache-memory-bytes` để cố định KV | mốc: 6 phút; baseline lại: ~0.8–2.2 h |
| R6 | **Đổi hai thứ một lúc** (KV FP8 kéo theo đổi backend attention) | library-gaps #3 | Thêm cấu hình đối chứng chỉ đổi backend | ~0.8 h |
| R7 | **Cổng chất lượng một lần chạy cho kết quả "khác biệt" giả** | Baseline với chính nó: MMLU-Pro −1.6 điểm, CI 95% paired −3.19 … −0.01 (P06 lần 2) | Hiệu của cấu hình mới phải nằm ngoài khoảng hiệu giữa hai lần chạy baseline; kết luận "kém đi" chỉ khi lặp lại được (đã đề xuất ở P06) | ~0.8 h mỗi lần chạy thêm |
| R8 | **W1–W3 closed-loop không đại diện tải thật**: cờ có lợi cho W1–W3 có thể hại W4 (ví dụ ngân sách token lớn giúp prefill nhưng tăng dao động TPOT) | P05: goodput W4 do đợt dồn quyết định, không do tải trung bình; ước theo P04 lệch 20% | Kết luận cuối về một cờ phải có W4 | xem câu hỏi Q2 |
| R9 | **Bootstrap / CI theo request lạc quan** | p95 gộp 300 request W2/20 = 10.87 s, CI bootstrap 8.97–15.14 s; nhưng request trong một lần đo không độc lập (chậm theo cụm, challenges #5); Hoefler & Belli quy tắc 6 (references G1): không giả định phân phối khi chưa kiểm | Dùng số lần lặp (không phải số request) làm đơn vị thống kê; báo khoảng min–max giữa các lần lặp | 0 |

Phụ: gộp request của 3 lần lặp rồi tính p95 cho **cùng goodput** với cách "trung vị 3 lần" ở cả ba workload (W1 68, W2 18, W3 6; W2/20 gộp 10.87 s).
Nếu muốn đổi quy tắc đạt/trượt sang p95 gộp, phải quyết **trước** khi đo cấu hình tune và tính lại baseline bằng cùng quy tắc.

### 6.2 Ứng viên GĐ4 (custom code) rút ra từ P07 — chưa làm

| Ứng viên | Vấn đề giải quyết | Vì sao có thể cần code riêng | Cắm vào vLLM thế nào | Tài liệu minh chứng cần tìm trước |
|---|---|---|---|---|
| **Bộ xếp lịch biết độ dài, không chặn đầu hàng** | N3: request dài chặn request ngắn | `policy=priority` chỉ sắp theo số do client gán; không có chính sách "cho request ngắn vượt khi request đầu hàng chưa đủ KV", không chống đói cho request dài | `scheduler_cls` (có trong `SchedulerConfig` v0.30.0) → lớp con của `Scheduler`, không sửa mã vLLM | Lập lịch theo độ dài / dự đoán độ dài (FastServe; "Efficient LLM Scheduling by Learning to Rank", NeurIPS'24); công bằng giữa người dùng (VTC, OSDI'24); TTFT theo trải nghiệm (Andes); chống đói (aging) trong lập lịch ưu tiên; số đo HOL trên vLLM |
| **Giữ KV lịch sử khi client cắt lịch sử** | 337 / 1,266 lượt sau ở peak k = 18 bị client cắt lịch sử → prefix đổi → không dùng lại được | Prefix cache theo mã băm chỉ dùng lại phần đầu giống hệt | Connector KV / offloading backend | CachedAttention (ATC'24, references F5) mục tách mã hoá vị trí; các nghiên cứu về chất lượng khi ghép KV lệch vị trí |
| **Chính sách nhận request theo đợt dồn token prefill** | P05: SLO hỏng theo phút có token prompt gấp ~2 lần | `max_num_queued_tokens` của vLLM từ chối request (HTTP 503), không phù hợp goodput | `scheduler_cls` | Admission control cho LLM serving theo SLO TTFT |

Theo quy tắc 11 của dự án, mọi thiết kế GĐ4 phải có tài liệu minh chứng trong docs/references trước khi làm. Bảng trên chỉ nêu cần tìm gì.
Nếu N3 giải quyết được bằng cờ có sẵn (mục 4.3) thì ứng viên thứ nhất mất lý do.

### 6.3 Câu hỏi cần quyết trước khi lập plan tune

Bản dễ đọc: [doc-ngay.md](doc-ngay.md) mục 7.

**Ước giờ một cấu hình** (giá 8,770 VND/h; đơn vị lấy từ mục 3.6): W2 rút gọn ×1 ~0.25 h · W1–W3 rút gọn ×1 ~0.8 h · ×3 ~2.2 h ·
W4 một seed ~1.5–2.2 h · W4 hai seed ~3–4.5 h · cổng chất lượng ×1 / ×2 ~0.8 / ~1.6 h.

**Hai phương án cho phần tune** (~13 cấu hình: N1 ×3 gồm đối chứng backend, N3 ×5, N4 ×2, N2 ×1, N5 ×1, gộp cuối ×1):

| Hạng mục | Kiểu rẻ | Kiểu chặt |
|---|---|---|
| Đo lại mốc W4 (R1) | — | 1.4 h |
| ~7 cấu hình đo bằng W1–W3 | rút gọn ×1: 5.6 h | rút gọn ×3: 15.4 h |
| ~6 cấu hình đo bằng W4 | một seed: 11.4 h | 3 chung kết × hai seed + 3 × một seed: 17.1 h |
| Cổng chất lượng (KV FP8, template) | ×1: 1.6 h | ×2: 3.2 h |
| Đo lại người thắng trên seed mới (R3) | — | 2.2 h |
| Cấu hình cuối: sweep đầy đủ + chất lượng ×2 | 11 h | 11 h |
| Setup phiên | 1.2 h | 1.6 h |
| **Tổng** | **~31 h ≈ 270k VND** | **~52 h ≈ 456k VND** |

**Câu hỏi:**
1. Thứ tự tune (đề xuất N1 → N3 → N2 → N4 → N5), hay ưu tiên W4 ngay từ đầu?
2. Workload cho mỗi cấu hình: cờ KV / preemption bằng W1–W3, cờ hàng đợi / lịch sử bằng W4, cấu hình cuối đủ cả hai — hay mọi cấu hình đều có W4?
3. Số lần lặp: ×1 (chỉ tin chênh ≥ 2 bước), ×3 + đo lại người thắng, hay trộn (sàng lọc ×1, chung kết ×3)?
4. Cổng chất lượng: chỉ cho cấu hình đổi số học, ×1 hay ×2? N5 cần bài chất lượng nhiều lượt chưa có: bỏ N5 hay thêm bài đo?
5. Đo lại mốc W4 trước (R1, ~12k VND)? Đề xuất: có.
6. Quy tắc đạt/trượt cho phần tune: giữ trung vị 3 lần hay p95 gộp? Phải chốt trước khi đo.
7. Thay đổi phía ứng dụng (độ ưu tiên do client gán, sửa template): báo thành nhóm riêng "vLLM đã tune + chỉnh ứng dụng"?
8. Dùng giá tham chiếu 8,770 VND/h cho mọi phép tính chi phí từ nay?
9. Ngân sách phần tune: ~270k hay ~456k?

**Quyết định 2026-10-14 (sau thảo luận):** user giao Claude tự cân bằng, nguyên tắc "đo rẻ trước, đo bổ sung khi thật sự cần". Đã chọn:

| Câu | Chọn | Lý do một câu |
|---|---|---|
| 1 Thứ tự | N1 KV → N3 hàng đợi (cờ có sẵn) → N2 RAM CPU → N4 preemption; **N5 hoãn** | N1 bằng chứng mạnh nhất, ảnh hưởng mọi workload; N5 cần bài đo chất lượng nhiều lượt chưa có |
| 2 Workload | N1, N4: W1–W3 rút gọn ×1; N2, N3: W4 seed 1 một lần; cấu hình cuối: sweep đầy đủ ×3 + W4 | Mỗi cờ đo bằng tải nơi nó có tác dụng; cấu hình cuối đo đủ nên tự xác nhận người thắng trước khi sang GĐ2 |
| 3 Lặp | ×1; chỉ coi là cải thiện khi chênh ≥ 2 bước tìm; chênh 1 bước ghi "chưa rõ", chỉ đo thêm nếu quyết định phụ thuộc vào nó | Chênh 1 bước nằm trong nhiễu; đo lại được bất cứ lúc nào |
| 4 Chất lượng | Chỉ cấu hình đổi số học (KV FP8), ×1 lúc định chọn nó; lần 2 chỉ khi lần 1 trông kém hơn mức tự lệch | Lần 2 chỉ cần khi có nghi ngờ |
| 5 Mốc W4 | Không chạy riêng. Chạy lại baseline W4 **cùng phiên** với cấu hình tune W4 đầu tiên chỉ khi chênh ≤ 1 bước | Chênh lớn thì không cần; chạy cùng phiên còn loại được lệch do khác máy |
| 6 Quy tắc đạt/trượt | Giữ trung vị 3 lần; dữ liệu thô từng request đã lưu nên tính thêm p95 gộp bất cứ lúc nào | Không tốn gì, không cần chọn trước |
| 7 Chỉnh phía ứng dụng | Báo nhóm riêng "vLLM đã tune + chỉnh ứng dụng" | So với thư viện khác vẫn công bằng |
| 8 Giá | 8,770 VND/h cố định | Chênh chi phí chỉ do chênh tốc độ |
| 9 Ngân sách tune | ~31 h ≈ 270k VND | Phương án rẻ; khoản đo bổ sung tính khi phát sinh |

Không có phép đo nào bắt buộc làm ngay: vLLM, model, workload, seed đều đã ghim và dữ liệu thô từng request đã lưu, nên mọi lần đo bổ sung
(mốc W4, lặp ×3, seed mới, chất lượng lần 2) làm lại được ở bất kỳ L4 nào vào lúc cần.


### 6.4 Việc khác

- Thuật ngữ mới đã thêm vào [thuat-ngu.md](../../thuat-ngu.md).
- `docs/library-gaps.md` có thay đổi chưa commit của user; P07 chỉ đọc, không sửa.

## 7. Thuật ngữ dùng trong báo cáo này

| Từ | Nghĩa |
|---|---|
| Giá tham chiếu | Một đơn giá thuê cố định (8,770 VND/h) dùng cho mọi phép tính chi phí, để chênh chi phí chỉ do chênh throughput |
| Chặn đầu hàng (head-of-line blocking) | Request đứng đầu hàng đợi chưa được nhận (thiếu KV) làm mọi request phía sau cũng phải chờ |
| `scheduler_reserve_full_isl` | Cờ vLLM: chỉ nhận request mới khi cả prompt vừa KV trống (mặc định bật) |
| Đẩy KV sang RAM CPU (KV offloading) | Chép KV ra RAM của máy chủ khi bị đẩy khỏi GPU; lần sau chép lại thay vì prefill lại |
| Watermark | Phần KV để dành, không dùng để nhận request mới, để giảm preemption |
| Lệch do chọn người thắng | Chọn cấu hình tốt nhất trong nhiều cấu hình đo có nhiễu thì số của nó bị đẩy lên; phải đo lại trên dữ liệu mới |
| So cặp cùng seed | So hai cấu hình trên cùng tập request (cùng seed), lấy hiệu từng cặp, thay vì so hai giá trị lấy từ tập request khác nhau |
| Bootstrap | Lấy mẫu lại có hoàn lại từ dữ liệu đã đo để ước khoảng tin cậy; sai nếu các mẫu không độc lập |
| Cổng chất lượng | Bộ đo chất lượng bắt buộc cho cấu hình đổi số học (P06 mục 6) |
| Goodput, SLO, TTFT, TPOT, preemption, prefix cache, chunked prefill, flips | Xem [thuat-ngu.md](../../thuat-ngu.md) |
