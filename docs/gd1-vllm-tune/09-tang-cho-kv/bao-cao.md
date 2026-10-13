# Báo cáo P09 (+ P12) — Tăng chỗ KV: KV FP8 + thêm bộ nhớ cho KV; preemption W1

> Ngày làm: 2026-10-14 02:11–04:37 (giờ VN) · Máy: host 506934 / machine 116596 (cùng máy P03/P04/P06), instance 55797039 ·
> vLLM 0.30.0 · Giờ GPU: 2.43 (P09 2.03 + P12 0.40) h (≈ 20,040 ($0.765) VND theo tín dụng bị trừ) · Plan: [P09](plan.md), [P12](../12-preemption-w1/plan.md) ·
> Số liệu: [measurements.md](../../measurements.md) mục "P09", "P12" · Kết quả thô: `results/gd1/09-tang-cho-kv/20261014-p09/`

## 1. Tóm tắt

- **KV FP8 + `--gpu-memory-utilization 0.95` (kv8)**: KV chứa **81,328 token** (baseline 37,264, ×2.18). vLLM tự đổi backend attention
  từ FLASH_ATTN sang **FLASHINFER** (FlashAttention 2 không nhận KV FP8 trên Ada).
- **Goodput (SLO chính, ×1, seed 1001): W1 140 (148 trượt), W2 26 (28), W3 12 (14)**; baseline 68 / 18 / 6. Mỗi workload hơn ≥ 4 bước tìm.
  Chi phí / 1M token ra (giá tham chiếu 8,770 VND/h): W1 **2,350** (−41%), W2 **10,880** (−25%), W3 **17,400** (−28%).
- **Chất lượng: điểm không kém đi, nhưng câu trả lời đổi nhiều hơn mức tự nhiên.** FinQA 59.0 (baseline 59.2 / 60.1), MMLU-Pro 67.8 (69.0 / 67.4).
  Số câu đổi đúng↔sai 7.7% / 11–12% (tự nhiên 4.5% / 6.6%), hai chiều gần cân bằng. Quy tắc đặt trước (b) "flips > 1.5 × tự nhiên" báo dấu hiệu.
  **Không chạy lại cổng**, lý do ở mục 4.3. **cfg1 = kv8 (tạm)**: cần người dùng quyết chấp nhận "khác nhưng không kém" trước P10 (mục 6).
- Nghẽn đã đổi: W1 và W3 lại chạm trần KV ở mức cao hơn (W1: 133 request chạy cùng lúc, 173 preemption ở 140 user); **W2 không còn bị KV giới hạn**
  (KV dùng 87% ở 28 user, 0 preemption). W2/W3 trượt vì **đợt khởi đầu** của bài đo closed-loop (mục 4.4).
- **P12** (điều kiện thoả: 173 preemption ≥ 50 ở W1/140): `--watermark 0.02` và `--max-num-seqs 133` giảm preemption 86–100%, throughput +4%,
  nhưng **không đạt ở 164 user** (mức cao) → **không nhận** theo quy tắc đặt trước. Bớt preemption chủ yếu chuyển thời gian chờ từ "tính lại" sang
  "xếp hàng" (TTFT p50 +12…+24%, p95 gần như không đổi). `--max-num-seqs` còn cho thêm 10% chỗ KV (mục 4.6) → ứng viên P13.
- **Giờ GPU 2.43 h ≈ 20,040 VND** (tín dụng bị trừ) so với ước 2.85 h ≈ 25k (P09 2.3 + P12 0.55).

## 2. Đã làm gì và vì sao

| Bước | Thời gian (giờ VN) | Ghi chú |
|---|---|---|
| Thuê máy 116596 ($0.303/h), checklist, đẩy code + dữ liệu + văn bản tham chiếu P06 | 02:11–02:13 | Đúng máy ưu tiên. Không thử lại `ncu` (host 506934 đã biết chặn, measurements P01) |
| Cài đặt (model 163 s, vLLM) | 02:11–02:16 | `setup_node.sh` |
| Server mặc định: compile từ đầu (KV 28,704) → tự bật lại (37,264) | 02:16–02:25 | Như P03/P04 (challenges #4) |
| **Điểm mốc** W2/16 ×3 | 02:25–02:31 | Máy tương đương P04, mục 3.1 |
| kv8 khởi động (KV 81,328, FLASHINFER) → teacher forcing → **vLLM chết vì hết bộ nhớ** | 02:31–02:37 | challenges #7 |
| Sửa: teacher forcing riêng ở 0.92 (cùng KV FP8), 3 bài còn lại ở 0.95; chạy lại cùng lệnh | 02:40–03:23 | Mục 5 |
| **Tìm goodput kv8** W1–W3 | 03:23–04:13 | 16 đơn vị, 49.5 phút đo |
| **P12** wm, seqs | 04:14–04:36 | |
| Kéo kết quả, kiểm checksum, destroy | 04:36–04:37 (308 file khớp sha256) | |

Khác plan:
- **Teacher forcing chạy ở 0.92** thay vì 0.95 (mục 5).
- **Không chạy lại cổng chất lượng** dù quy tắc (b) báo dấu hiệu (mục 4.3).
- **Hệ số mở rộng 1.5 / 0.75** thay vì 2 / 0.5 khi khởi đầu nóng (P08). Cả ba mức trên của khởi đầu nóng (W1 88, W2 26, W3 12) đều đạt nên phải mở rộng lên;
  với ×2 thì W1 sẽ đo 176 thay vì 132.
- **P12 mức cao = 164** thay vì lo + 2 bước = 156: 164 là mức cfg1 đã đo (trượt) gần nhất phía trên 156, nên so cặp được với cfg1 cùng seed. Kiểm khắt khe hơn (lo + 3 bước).

## 3. Kết quả

### 3.1 Điểm mốc (vLLM mặc định, W2 / 16 user × 3, so cặp cùng seed với P04)

| Seed | TTFT p95 P04 → hôm nay (s) | Token ra/s P04 → hôm nay |
|---|---|---|
| 1001 | 6.30 → 5.79 (−8%) | 171 → 172 |
| 1002 | 4.87 → 4.94 (+2%) | 169 → 169 |
| 1003 | 6.03 → 5.88 (−2%) | 167 → 167 |

Chênh TTFT p95 nằm trong mức nhiễu một lần đo (~19%, P03), throughput gần như trùng. KV 37,264 token, bằng P04. **So trực tiếp được với baseline P04.**

### 3.2 Cấu hình vLLM thật sự chạy

| Cấu hình | KV (token) | KV dtype | Attention | MoE | Token / bước |
|---|---|---|---|---|---|
| Mặc định (0.92) | 37,264 | FP16 | FLASH_ATTN | MARLIN | 2,048 |
| **kv8** (FP8, 0.95) | **81,328** | FP8 e4m3 (scale 1.0, không hiệu chuẩn) | **FLASHINFER** | MARLIN | 2,048 |
| FP8, 0.92 (chỉ để teacher forcing) | 66,896 | FP8 | FLASHINFER | MARLIN | 2,048 |

### 3.3 Goodput

| Workload | Baseline (P04, trung vị ×3) | **kv8 (×1)** | Hơn | Cận trên ước ở P07 | Chỉ số hỏng trước |
|---|---|---|---|---|---|
| W1 chat | 68 (72) | **140 (148)** | +72 user ≈ 10 bước | ~88 | TTFT (TPOT 124 ms còn dư) |
| W2 RAG | 18 (20) | **26 (28)** | +8 = 4 bước | ~26 | TTFT (đợt khởi đầu; TPOT 121 ms) |
| W3 tóm tắt | 6 (8) | **12 (14)** | +6 = 3 bước | ~15 | TTFT (đợt khởi đầu + KV đầy) |

Theo từng mức: bảng đầy đủ ở measurements "P09". Mốc chính:

| | Throughput tại goodput (token ra/s) | TPOT p95 | Request chạy cùng lúc (p50) | Preemption | KV dùng tối đa |
|---|---|---|---|---|---|
| W1 baseline 68 / kv8 140 | 615 / **1,035** (+68%) | 104 / 124 ms | ~60 / 133 | 133 / 173 | 100% / 100% |
| W2 baseline 18 / kv8 26 | 168 / **224** (+33%) | 90 / 116 ms | ~15 / 26 | ~5 / 0 | 100% / 83% |
| W3 baseline 6 / kv8 12 | 101 / **140** (+39%) | 57 / 87 ms | ~6 / 12 | ~2 / 0 | 100% / 94% |

Chi phí / 1M token ra (8,770 VND/h): W1 3,960 → **2,350**; W2 14,460 → **10,880**; W3 24,030 → **17,400**.

### 3.4 Chất lượng (cổng kv8, so với hai lần chạy baseline P06)

| Bài | Baseline lần 1 / 2 | kv8 | kv8 − lần 1 (CI 95%) | Flips so lần 1 / 2 | Tự nhiên | Đúng→sai / sai→đúng (so lần 1) |
|---|---|---|---|---|---|---|
| FinQA-RAG (1,147) | 59.2 / 60.1 | **59.0** | −0.17 (−1.78 … +1.43) | 7.7% / 7.7% | 4.5% | 45 / 43 |
| MMLU-Pro (1,000) | 69.0 / 67.4 | **67.8** | −1.20 (−3.38 … +0.98) | 12.4% / 11.0% | 6.6% | 68 / 56 |

| Độ gần token | kv8 so với baseline | Baseline tự lệch |
|---|---|---|
| Teacher forcing: same top-1 (44,976 vị trí) | 97.09% | 99.10% |
| Teacher forcing: KL TB / p99 | 0.0100 / 0.134 | 0.0014 / 0.017 |
| Teacher forcing: Δp TB, p1 / p99 | −0.003, −0.170 / +0.131 | +0.000, −0.047 / +0.049 |
| Sinh tự do: giống hệt 256 token / vị trí lệch p50 | 5.4% / token 19 | 23.1% / token 46 |

Ba quy tắc "kém đi" đặt trước: (a) điểm thấp hơn lần baseline thấp hơn quá 1.96 × SE paired: **không** (FinQA −0.17 so lần 1, ngưỡng ±1.6;
MMLU-Pro +0.40 so lần 2); (b) flips > 1.5 × tự nhiên: **có**, cả hai bài; (c) Δp lệch hẳn về âm (p1 < −0.08 và |p1| > 1.5 × p99): **không**
(teacher forcing −0.170 so với 1.5 × 0.131 = 0.196; sinh tự do −0.163 / +0.171).

### 3.5 P12 — preemption W1 (trên nền kv8, W1 ×1 seed 1001)

| Cấu hình | KV (token) | N | Kết quả | TTFT p50 / p95 (s) | TPOT p95 (ms) | Token ra/s | Preemption |
|---|---|---|---|---|---|---|---|
| kv8 (cfg1) | 81,328 | 140 | đạt | 2.04 / 8.38 | 124 | 1,035 | 173 |
| `--watermark 0.02` | 81,328 | 140 | đạt | 2.52 / 8.32 | 116 | 1,076 | 10 |
| `--max-num-seqs 133` | 89,488 | 140 | đạt | 2.38 / 7.96 | 117 | 1,077 | 0 |
| kv8 (cfg1) | 81,328 | 164 | trượt | 8.05 / 15.33 | 131 | 1,035 | 272 |
| `--watermark 0.02` | 81,328 | 164 | trượt | 9.03 / 14.37 | 128 | 1,079 | 38 |
| `--max-num-seqs 133` | 89,488 | 164 | trượt | 9.08 / 12.82 | 117 | 1,082 | 0 |

Quy tắc chọn (đặt trước): đạt ở lo + 2 bước và ở lo → nhận. Cả hai chỉ đạt ở lo (140) → **không nhận**.

## 4. Giải thích kết quả

### 4.1 Vì sao W1 vượt xa cận trên ước ở P07 (140 so với ~88)

P07 ước bằng giả định throughput giữ ở mức bão hoà của baseline (615 token/s). **Đo được:** throughput W1 tăng theo số request chạy cùng lúc,
tới ~1,035 token/s ở 130+ request (W1/72: 770, /88: 863, /132: 1,052). **Suy ra:** mỗi token chọn 8 / 128 expert, nên từ ~60 token mỗi bước
gần như mọi expert đều được dùng (kỳ vọng 128 × (1 − (120/128)^60) ≈ 125). Từ đó, mỗi bước decode phải đọc gần đủ trọng số expert (~15 GB INT4,
≥ 50 ms ở 300 GB/s) bất kể batch. Thêm request vào cùng bước thì phần đọc này được chia cho nhiều token hơn; chỉ phần tính theo từng token (attention, KV) tăng.
Baseline chỉ chạy được ~60 request (KV đầy) nên dừng ở đầu vùng này. kv8 chạy 130 request: batch ×2.2 cho throughput ×1.7.
Ở W2 / W3 throughput chỉ tăng 33–39% vì số request chạy cùng lúc nhỏ (26, 12) và phần prefill prompt dài chiếm nhiều thời gian GPU hơn.

### 4.2 Nghẽn mới ở đâu

- **W1:** từ ~133 request chạy cùng lúc trở lên, KV lại đầy (81,328 / ~600 token mỗi request), preemption tăng (68 ở 132 user, 173 ở 140, 381 ở 198)
  và throughput ngừng tăng (~1,035). Goodput dừng vì **chỗ KV**, TPOT còn dư (124 ms / 150). → P12 thử bớt preemption; P13/GĐ2 có thể thử thêm chỗ KV.
- **W2:** ở 28 user KV mới dùng 87%, 0 preemption, cả 28 request chạy; thời gian chờ hàng đợi TB chỉ 1.37 s, nhưng TTFT p95 10.7 s. **W2 không còn bị KV giới hạn.**
- **W3:** ở 14 user KV 99.8%, KV vẫn là giới hạn.
- **TPOT chưa phải chỉ số hỏng trước** ở workload nào (cao nhất 124 ms ở goodput W1; chạm ~141–144 ms chỉ ở W2/40, W1/198). Câu hỏi 3 của plan: chưa cần
  xem lại speculative decoding; nhưng TPOT chỉ còn dư ~20% ở W1/W2, nên cấu hình nào tăng batch tiếp sẽ gặp TPOT trước.

### 4.3 Vì sao không chạy lại cổng chất lượng

Plan: có dấu hiệu → chạy lại cổng (0.78 h); lặp lại được → loại kv8.
- Lần chạy lại chỉ trả lời "dấu hiệu có lặp lại không". Teacher forcing cho thấy kv8 lệch **có hệ thống** chứ không ngẫu nhiên: same top-1 97.09% so với 99.10%
  trên 44,976 vị trí (sai số chuẩn ~0.08%), KL gấp 7 lần. Nhiễu ngẫu nhiên không tạo ra chênh này, nên lần chạy lại gần như chắc chắn lặp lại flips cao.
  0.78 h không đổi được quyết định.
- **Quy tắc (b) gộp "khác" với "kém".** Flips cao mà hai chiều cân bằng (FinQA 45 / 43, MMLU-Pro 68 / 56 và 53 / 57) nghĩa là kv8 đổi đáp án ở
  thêm ~3–6% câu, nhưng số câu đúng không đổi ngoài nhiễu. Dutta 2024 (references E4) dùng flips để phát hiện mô hình **khác** baseline, và thấy flips
  đi cùng điểm sinh văn bản tự do (MT-Bench) thấp hơn. Vì vậy flips là tín hiệu thật, nhưng không tự chứng minh "kém đi" trên tác vụ của dự án.
- Do đó: **cfg1 = kv8 tạm thời**, và đây là quyết định của người dùng trước P10 (mục 6). Đề xuất sửa quy tắc cổng: (a) và (c) quyết "kém đi";
  (b) đổi thành "khác biệt cần báo", kèm số câu đổi hai chiều.

**Vì sao lệch nhiều (phỏng đoán có căn cứ):** vLLM chạy KV FP8 với scale = 1.0 vì không hiệu chuẩn (tài liệu vLLM, references F2, khuyến nghị
hiệu chuẩn bằng llm-compressor). Giá trị K/V bị làm tròn xuống 3 bit mantissa (e4m3), thêm nhiễu vào điểm attention ở mọi lớp.
Ngoài ra backend đổi sang FlashInfer, nên thứ tự cộng khác. Chưa tách được phần nào do FP8, phần nào do backend (đối chứng hoãn, README).

### 4.4 Đợt khởi đầu quyết định kết quả W2 / W3

Bài đo closed-loop gửi N request cùng lúc ở giây đầu. Với W2 (prompt 2.9k token), 28 request cần ~80k token prefill. Ở 2,048 token / bước,
request cuối của đợt chờ ~40 bước, tức ~10 s. Với kv8, phần còn lại của lần đo hầu như không có hàng đợi.

| Đơn vị | TTFT p95 cửa sổ "all" | Cửa sổ "steady" (bỏ N request đầu và N cuối) |
|---|---|---|
| kv8 W2/26 / 28 / 32 | 9.77 / 10.74 / 11.92 s | **1.52 / 1.73 / 2.13 s** |
| kv8 W3/12 / 14 | 7.72 / 10.61 s | 2.70 / 5.82 s |
| kv8 W1/140 / 148 | 8.38 / 10.15 s | 7.91 / 11.02 s |
| baseline W2/18 / 20 (P04 r1) | 9.68 / 11.07 s | 6.90 / 10.50 s |

**Đo được:** với baseline, hai cửa sổ gần nhau (hàng đợi kéo dài suốt lần đo); với kv8, W2/W3 trượt chỉ vì đợt đầu. **Suy ra:** goodput W2 theo "steady" của kv8
cao hơn 26 nhiều (TPOT 132 ms ở 32 user, 141 ms ở 40; mức đó chưa đo đủ). Kết luận P09 vẫn dùng "all" (quy tắc chốt ở P03, thận trọng, so được với baseline).
Người dùng thật không gửi cùng một giây, nên W4 (phát lại phiên thật) mới là thước đo cuối. Đề xuất trước P13: xem lại cách mở đầu lần đo
(ví dụ dàn N user ban đầu trong vài giây) hoặc báo cả hai cửa sổ. Không gấp: dữ liệu từng request đã lưu, tính lại không cần GPU.

### 4.5 Kiểm tra ngược

- **Khác máy / khác lần khởi động?** Không: cùng machine 116596, điểm mốc khớp P04, KV baseline 37,264 bằng P04.
- **Một lần đo ×1 có thể lệch?** Chênh với baseline là 3–10 bước tìm, lớn hơn nhiều so với nhiễu 1 bước (P07 R2). Nhưng con số chính xác của goodput
  (140 so với 132/148) chưa chắc: W1/148 trượt với TTFT p95 10.15 s, cách ngưỡng 1.5%. P13 đo ×3 trên seed mới.
- **Lệch do chọn người thắng (P07 R3)?** Chỉ một cấu hình được đo, không có chọn lựa nên không có lệch này ở P09.
- **Hai thay đổi một lúc** (FP8 + 0.95 + backend FlashInfer): đã biết trước (plan gộp có chủ đích). Đối chứng backend hoãn.

### 4.6 P12: bớt preemption được gì

- **Sau KV FP8, preemption còn bao nhiêu (câu hỏi 1 của P12)?** 0 ở mọi mức tới 88 user; 68 ở 132, 173 ở 140, 272 ở 164: xuất hiện lại khi số request
  chạy cùng lúc chạm ~133 (KV đầy).
- **TTFT p95 tốt hơn hay chỉ đổi chỗ chờ (câu hỏi 2)?** Chủ yếu đổi chỗ chờ. Đo được: TTFT p50 tăng 12–24% (request mới chờ trong hàng đợi thay vì được
  nhận rồi bị tạm dừng), TTFT p95 −1 … −16% (một lần đo, nhiễu TTFT p95 ~19%), throughput +4% ở cả 4 đơn vị (nhiễu throughput ~1.2%, P03), TPOT p95 −3 … −11%.
  **Suy ra:** phần GPU từng dùng để tính lại request bị tạm dừng khoảng 4% (cận dưới của khoảng 0–18% ước ở P07 mục 4.4).
- **`--max-num-seqs 133` còn đổi thêm một thứ:** vLLM giảm cỡ CUDA graph lớn nhất 512 → 256 (CUDA graph: chuỗi kernel ghi sẵn cho từng cỡ batch để chạy nhanh;
  log cho thấy cỡ lớn nhất 512 khi max_num_seqs mặc định 256 và 256 khi 133; quy tắc chính xác chưa kiểm trong mã), bộ nhớ cho graph giảm 1.57 → 0.94 GiB, nên vùng KV tăng 3.72 → 4.10 GiB (**+10% token**). Kết quả seqs là tổng hai hiệu ứng.
  Ứng viên P13: `--max-num-seqs` (hoặc giảm cỡ CUDA graph tối đa) để lấy thêm chỗ KV mà không đổi số học. Theo quy tắc, chênh ≤ 1 bước nên ghi "chưa rõ".

## 5. Vấn đề gặp phải

- **vLLM hết bộ nhớ khi teacher forcing ở 0.95** (challenges #7). `prompt_logprobs` cần logits 151,936 từ cho mọi vị trí prompt, mà lượt đo bộ nhớ
  lúc khởi động không tính phần này. Sửa: chạy riêng đơn vị đó ở 0.92 (cùng KV FP8), tốn thêm một lần khởi động (~2 phút).
  Hệ quả cho dịch vụ thật: ở 0.95, request xin `prompt_logprobs` có thể làm sập server (ghi cho GĐ5).
- **Watcher chạy ngoài cơ chế nền** hai lần (lỗi thao tác của Claude): sẽ không được báo. Đã tắt và chạy lại; vị trí đã đọc lưu trong file nên không mất sự kiện.

## 6. Ảnh hưởng tới bước sau

**Cần người dùng quyết trước P10:** chấp nhận KV FP8 không? Lợi: goodput gần gấp đôi, chi phí −25…−41%. Hại: điểm không đổi, nhưng
~3–6% câu đổi đáp án (cả hai chiều), câu chữ khác baseline nhiều hơn. Phương án nếu không chấp nhận: kv16 (FP16 + 0.95, ~44k token KV, ước +19%; chưa đo, ~0.9 h),
hoặc kv8 có hiệu chuẩn scale (llm-compressor, references F2; cần tìm hiểu, chưa ước). Đề xuất của Claude: **chấp nhận kv8**. Tiêu chí của dự án là
"không làm giảm chất lượng", và điểm trên hai bài (một bài là đúng loại việc RAG) không giảm. Báo cáo cuối phải ghi rõ flips / KL như Dutta 2024 khuyên.

Khác:
- P10 (W4) chạy trên nền kv8. W4 trộn W1/W2, nên lợi ích dự kiến lớn; W4 cũng trả lời câu hỏi đợt khởi đầu (mục 4.4) bằng tải thật.
- Nghi vấn mới **N6: W2 bị giới hạn bởi prefill chen decode** (không còn KV): thử `--max-num-batched-tokens` lớn hơn (P07 chưa có vì khi đó KV là nghẽn).
  Chưa có plan; đưa vào P13 hoặc GĐ2 tuỳ kết quả W4.
- Quy tắc cổng chất lượng (P13 dùng lại): sửa (b) như mục 4.3.

## 7. Thuật ngữ

- **KV FP8 (e4m3):** lưu key/value của attention bằng số thực 8 bit (4 bit mũ, 3 bit phần định trị) thay vì 16 bit; nửa bộ nhớ, kém chính xác hơn.
- **Hiệu chuẩn scale KV:** đo phạm vi giá trị K/V trên dữ liệu mẫu để chọn hệ số nhân trước khi làm tròn sang FP8; không hiệu chuẩn thì hệ số = 1.0.
- **FlashInfer:** thư viện kernel attention vLLM dùng khi KV FP8 trên GPU Ada.
- **Đợt khởi đầu:** N request đầu tiên của bài đo closed-loop được gửi cùng một lúc, prefill dồn vào vài giây đầu.
- **Flips:** % câu đổi từ đúng sang sai hoặc ngược lại giữa hai cấu hình.
