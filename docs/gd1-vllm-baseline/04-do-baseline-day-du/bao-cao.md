# Báo cáo P04 — Đo baseline đầy đủ W1/W2/W3 (vLLM mặc định)

> Ngày làm: 2026-10-13 (00:19 → 04:45) · Máy: host 506934 / machine 116596, instance 55583011 (cùng máy P01, P03) ·
> vLLM 0.30.0 + `--max-model-len 16384` · Giờ GPU: 4.44 h (≈ $1.26 ≈ 32,700 VND; hoá đơn cả phiên P03 + P04: $1.516 ≈ 39,700 VND)
> Số liệu: [measurements.md](../../measurements.md) mục "P04". Kết quả thô: `results/gd1/04-do-baseline-day-du/20261013-p04/`
> (`summary.md` là bảng đầy đủ, `units.csv` từng đơn vị × cửa sổ). Hình: [hinh/](hinh/).

## 1. Tóm tắt

Goodput của vLLM mặc định trên một L4, theo SLO chính (TTFT p95 ≤ 10 s, TPOT p95 ≤ 150 ms), quyết định theo trung vị 3 lần lặp:

| Workload | Goodput SLO chính | Goodput SLO phụ (2 s / 100 ms) | Số mức đã đo |
|---|---|---|---|
| **W1** chat (prompt ~410, ra ~330 token) | **68** (72 trượt) | 32 (64 trượt) | 9 |
| **W2** RAG (prompt ~2,860, ra ~170 token) | **18** (20 trượt) | < 8 | 7 |
| **W3** tóm tắt hợp đồng (prompt ~5,600, ra ~300 token) | **6** (8 trượt) | < 4 | 6 |

- Cả ba workload đều **trượt vì TTFT trước**. TPOT p95 cao nhất chỉ 117 ms, vẫn dưới ngưỡng 150 ms.
- TTFT vượt ngưỡng vì request phải **xếp hàng chờ chỗ trong KV cache** (37,264 token). Số request chạy cùng lúc bị chặn ở khoảng ~60 (W1), ~15 (W2), ~6 (W3).
- Đạt tiêu chí xong của plan:
  - đủ 36 lần đo lưới gốc và 30 lần đo mở rộng / tìm nhị phân;
  - mọi khoảng đều hẹp (`hi − lo ≤ max(2, 10% × lo)`);
  - không có vi phạm đơn điệu, không có request lỗi.
- Lưu ý: W2 / 20 user có TTFT p95 trung vị **10.06 s**, chỉ cách ngưỡng 0.6%. Tính theo cửa sổ "steady" thì goodput W2 là 20 (24 trượt), xem mục 4.5.

## 2. Đã làm gì và vì sao

- Chạy `bench/configs/full.yaml` (chế độ search) bằng harness của P03, ngay sau smoke và trên cùng instance để khỏi cài lại (theo README GĐ1, phiên 2).
- **Lưới gốc:** W1/W2/W3 × {8, 16, 32, 64} × 3 lần lặp = 36 đơn vị, 00:19 → 02:59 (2.7 h).
  - Thứ tự chạy xen kẽ: toàn bộ r1, rồi r2, rồi r3.
  - Khởi động lại server trước r2 và r3. Lần lặp r1 dùng lại server của P03 (lần khởi động 3, KV 37,264). r2 và r3 đều có KV 37,264, compile 0.4–0.6 s (đọc cache).
  - Nhờ vậy cả 3 lần lặp chạy trên 3 tiến trình server khác nhau nhưng cùng dung lượng KV. Không lần nào chạy trên server vừa compile từ đầu (quy tắc P03, [challenges](../../challenges.md) #4).
- **Các đợt sau do harness tự sinh** từ các `done.json`:

| Đợt | W1 | W2 | W3 |
|---|---|---|---|
| 2 | 128 (64 đạt → nhân đôi) | 24 (giữa 16 đạt / 32 trượt) | 4 (8 trượt → chia đôi) |
| 3 | 96 | 20 | 6 |
| 4 | 80 | 18 | — (xong: 6 đạt, 8 trượt, chênh 2) |
| 5 | 72 | — (xong: 18 / 20) | |
| 6 | 68 → xong (68 / 72, chênh 4 ≤ 6.8) | | |

- Mọi mức đều đo đủ 3 lần, mỗi lần dùng seed khác (1001–1003), tức tập request khác nhau.
- Số request mỗi lần đo: max(100, 4N). Không workload nào chạm trần số mẫu.
- **Không khác plan về cách làm.** Tổng giờ GPU (4.44 h) thấp hơn ước lượng P03 (5.1 h) và plan ban đầu (6.5 h), chủ yếu vì phần mở rộng ít hơn dự kiến: W2 và W3 mỗi workload chỉ cần 2–3 mức thêm, W1 cần 5.

## 3. Kết quả

### 3.1 Bảng chính

Trung vị 3 lần lặp, cửa sổ "all". "Chạy" là số request đang chạy (trung vị trong lúc đo / tối đa). Độ lệch là (max − min) / trung vị giữa 3 lần.

| w | N | TTFT p50 / p95 / p99 (s) | TPOT p50 / p95 / p99 (ms) | tok/s ra | tok/s vào | SLO chính | Chạy | Chờ max | Preempt / lần | Lệch tok/s | Lệch TTFT p95 | Lệch TPOT p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W1 | 8 | 0.18 / 0.43 / 0.6 | 32 / 33 / 34 | 243 | 291 | ✓ | 8 / 8 | 2 | 0 | 3.9% | 0.6% | 3.4% |
| W1 | 16 | 0.22 / 0.78 / 1.1 | 47 / 48 / 48 | 323 | 398 | ✓ | 16 / 16 | 0 | 0 | 5.1% | 43.6% | 3.6% |
| W1 | 32 | 0.31 / 1.77 / 2.1 | 64 / 66 / 66 | 458 | 541 | ✓ | 32 / 32 | 20 | 0 | 5.0% | 15.9% | 4.0% |
| W1 | 64 | 1.44 / 5.89 / 6.9 | 91 / 98 / 109 | 610 | 747 | ✓ | 60 / 64 | 45 | 97 | 1.9% | 18.9% | 5.0% |
| **W1** | **68** | 3.00 / **8.30** / 9.3 | 93 / 104 / 114 | 615 | 746 | ✓ | 60 / 68 | 52 | 133 | 3.0% | 13.1% | 3.1% |
| W1 | 72 | 4.05 / 11.14 / 12.7 | 93 / 104 / 119 | 612 | 757 | ✗ | 60 / 72 | 53 | 143 | 3.5% | 27.8% | 1.1% |
| W1 | 80 | 7.59 / 15.60 / 16.7 | 94 / 114 / 136 | 617 | 758 | ✗ | 59 / 80 | 64 | 164 | 0.9% | 14.6% | 5.9% |
| W1 | 96 | 17.8 / 23.0 / 26.2 | 95 / 117 / 149 | 619 | 753 | ✗ | 60 / 88 | 80 | 230 | 0.9% | 15.5% | 5.0% |
| W1 | 128 | 34.6 / 41.9 / 47.7 | 95 / 113 / 146 | 620 | 760 | ✗ | 60 / 86 | 113 | 330 | 0.4% | 6.3% | 3.6% |
| W2 | 8 | 0.71 / 2.03 / 3.5 | 51 / 57 / 58 | 146 | 2,376 | ✓ | 8 / 8 | 6 | 0 | 1.0% | 16.8% | 1.1% |
| W2 | 16 | 1.57 / 6.03 / 6.8 | 82 / 90 / 94 | 169 | 2,777 | ✓ | 15 / 16 | 13 | 6 | 2.6% | 23.8% | 3.4% |
| **W2** | **18** | 3.35 / **7.42** / 15.1 | 82 / 90 / 94 | 168 | 2,775 | ✓ | 15 / 17 | 15 | 7 | 2.3% | 43.7% | 3.8% |
| W2 | 20 | 5.28 / 10.06 / 15.7 | 83 / 93 / 103 | 168 | 2,759 | ✗ | 15 / 17 | 16 | 8 | 2.1% | 23.0% | 3.7% |
| W2 | 24 | 9.16 / 16.03 / 18.0 | 82 / 91 / 96 | 169 | 2,777 | ✗ | 15 / 17 | 21 | 7 | 2.7% | 6.7% | 3.5% |
| W2 | 32 | 17.1 / 21.3 / 28.4 | 82 / 92 / 100 | 169 | 2,743 | ✗ | 15 / 17 | 28 | 7 | 2.5% | 7.1% | 3.2% |
| W2 | 64 | 49.0 / 53.8 / 56.3 | 84 / 91 / 100 | 172 | 2,788 | ✗ | 15 / 18 | 61 | 17 | 1.2% | 4.3% | 1.3% |
| W3 | 4 | 1.33 / 2.15 / 3.4 | 39 / 43 / 45 | 93 | 1,766 | ✓ | 4 / 4 | 2 | 0 | 0.2% | 23.7% | 2.4% |
| **W3** | **6** | 1.72 / **5.14** / 7.7 | 52 / 57 / 60 | 101 | 1,935 | ✓ | 6 / 6 | 5 | 1 | 0.7% | 47.8% | 5.6% |
| W3 | 8 | 6.97 / 13.06 / 17.1 | 54 / 59 / 62 | 102 | 1,945 | ✗ | 6 / 7 | 6 | 2 | 0.5% | 17.2% | 4.4% |
| W3 | 16 | 31.1 / 36.5 / 38.0 | 54 / 59 / 62 | 102 | 1,942 | ✗ | 6 / 7 | 14 | 3 | 0.6% | 5.5% | 4.1% |
| W3 | 32 | 75.6 / 83.5 / 86.2 | 54 / 61 / 62 | 103 | 1,943 | ✗ | 6 / 7 | 30 | 7 | 1.1% | 3.8% | 4.1% |
| W3 | 64 | 165.5 / 176.6 / 178.7 | 54 / 60 / 65 | 103 | 1,944 | ✗ | 6 / 7 | 61 | 14 | 1.4% | 2.3% | 2.5% |

TTFT p95 từng lần lặp ở các mức sát ngưỡng (r1 / r2 / r3, giây):

| W1 64 | W1 68 | W1 72 | W2 16 | W2 18 | W2 20 | W2 24 | W3 4 | W3 6 | W3 8 |
|---|---|---|---|---|---|---|---|---|---|
| 5.90 / 4.79 / 5.89 | 8.36 / 7.28 / 8.30 | 11.14 / 9.20 / 12.29 | 6.30 / 4.87 / 6.03 | 9.68 / 6.44 / 7.42 | **11.07 / 8.76 / 10.06** | 16.45 / 15.38 / 16.02 | 2.15 / 2.47 / 1.96 | 5.14 / 5.91 / 3.45 | 11.59 / 13.84 / 13.06 |

### 3.2 Goodput theo các ngưỡng (tính lại từ cùng dữ liệu, không đo thêm)

| | SLO chính 10 s / 150 ms | SLO phụ 2 s / 100 ms | TTFT ≤ 2 s | ≤ 5 s | ≤ 10 s | ≤ 20 s | SLO chính, cửa sổ steady |
|---|---|---|---|---|---|---|---|
| W1 | 68 (72) | 32 (64) | 32 (64) | 32 (64) | 68 (72) | 80 (96) | 68 (72) |
| W2 | 18 (20) | < 8 | < 8 | 8 (16) | 18 (20) | 24 (32) | **20 (24)** |
| W3 | 6 (8) | < 4 | < 4 | 4 (6) | 6 (8) | 8 (16) | 6 (8) |

Các cột không phải SLO chính chỉ có độ phân giải của các mức đã đo. Ví dụ "W1 ≤ 5 s → 32 (64)" nghĩa là chỉ biết điểm gãy nằm giữa 32 và 64, vì không có mức đo nào giữa hai số đó.

### 3.3 Hình

| | |
|---|---|
| ![TTFT p95](hinh/h1-ttft-p95.png) | ![TPOT p95](hinh/h2-tpot-p95.png) |
| ![Độ nhạy](hinh/h3-do-nhay-ttft.png) | ![Throughput](hinh/h4-throughput.png) |

### 3.4 Bảng phụ: bộ nhớ, GPU

- VRAM dùng tối đa: 21,145–21,237 MiB ở mọi mức. Cố định, vì vLLM giữ trước 92% bộ nhớ.
- Công suất tối đa: 73.1–74.5 W ở **mọi** mức, tức luôn chạm trần 72 W.
- Xung SM thấp nhất: 990–1,125 MHz (tối đa 2,040). Nhiệt độ tối đa 70–74 °C.
- KV dùng tối đa: 18% (W1/8), 32% (W1/16), 60% (W1/32), 64% (W2/8), 78% (W3/4). Từ W1/64, W2/16, W3/6 trở lên luôn là 100%.
- Trúng prefix cache: W1 0–0.4%, W2 21.6% (phần system prompt chung), W3 1.6%.
- TTFT trung bình đo ở client trừ TTFT trung bình server tự đo (66 đơn vị): 13–90 ms (trung vị 31 ms).

### 3.5 Thời gian tìm goodput

| | Mức lưới gốc | Mức thêm | Đơn vị | Giờ (tổng thời gian đơn vị) |
|---|---|---|---|---|
| W1 | 4 | 5 (128, 96, 80, 72, 68) | 27 | 1.37 h |
| W2 | 4 | 3 (24, 20, 18) | 21 | 0.91 h |
| W3 | 4 | 2 (4, 6) | 18 | 2.07 h |
| Khởi động lại server (2 lần, ~2.5 phút mỗi lần) + phần ngoài đơn vị | | | | ≈ 0.09 h |
| **Tổng** | | | **66** | **4.35 h đơn vị; 4.44 h máy** |

## 4. Giải thích kết quả

### 4.1 Vì sao TTFT hỏng trước, không phải TPOT (đo được + suy ra)

**Đo được: số request chạy cùng lúc có trần, và trần đó do KV cache đặt ra.**
- Khi tải vượt goodput, số request **đang chạy** gần như cố định: W1 ~60, W2 ~15, W3 ~6. Phần còn lại nằm trong hàng đợi (cột "Chờ max").
- Chia KV cache 37,264 token cho số request đang chạy:
  - W1: 37,264 / 60 ≈ 620 token mỗi request. Khớp với một request W1 đang sinh dở (prompt ~410 + một phần output ~330).
  - W2: 37,264 / 15 ≈ 2,480 token, gần bằng prompt W2 (2,860, trừ 622 token system prompt dùng chung).
  - W3: 37,264 / 6 ≈ 6,200 token, khoảng một hợp đồng (5,600 + 300).
  - → **Số request chạy được = KV cache / kích thước một request.** Đây là giới hạn của bộ nhớ, không phải của tốc độ tính.

**Đo được: TTFT chủ yếu là thời gian chờ.** Server tự chia TTFT trung bình (histogram `request_queue_time` và `request_prefill_time`, lần r1):

| Mức | TTFT TB | Chờ trong hàng | Prefill |
|---|---|---|---|
| W1 / 68 | 3.35 s | 2.79 s (83%) | 0.45 s |
| W2 / 18 | 3.83 s | 2.78 s (73%) | 0.88 s |
| W3 / 6 | 2.23 s | 0.72 s (32%) | 1.42 s |
| W3 / 8 | 7.23 s | 5.68 s (79%) | 1.43 s |

**Suy ra: vì sao chờ thì TTFT tăng, còn TPOT không tăng.**
- Closed-loop với N user luôn có N request trong hệ thống. Khi N vượt trần KV, có N − trần request phải chờ.
- Một request mới chỉ vào chạy được khi một request khác xong, tức là sau khoảng E2EL / trần.
- Vì vậy TTFT tăng gần tuyến tính theo N. Ví dụ W2: 6.0 s ở 16 user, 16.0 s ở 24, 21.3 s ở 32, 53.8 s ở 64.
- Trong khi đó số request cùng chạy không tăng nữa, nên mỗi bước tính không nặng thêm. TPOT dừng ở mức bão hoà: W2 ~90 ms, W3 ~60 ms, W1 ~95–117 ms.
- Đây cũng là lý do throughput đi ngang sau điểm gãy: W1 ~615, W2 ~169, W3 ~102 token/s ra (hình 4).

**Suy ra: vì sao TPOT W1 cao hơn W2, W3.** Mỗi bước decode của vLLM tính cho mọi request đang chạy, nên bước càng dài khi càng nhiều request. W1 chạy ~60 request cùng lúc, W2 ~15, W3 ~6. W2 và W3 có TPOT không quá thấp so với số request vì mỗi bước còn phải gánh prefill (tối đa 2,048 token / bước) của các prompt dài.

### 4.2 Preemption ở W1 (đo được, giải thích là phỏng đoán có cơ sở)

- Ở W1, từ 64 user trở lên có 97–330 lần preemption mỗi lần đo (256–512 request). W2 và W3 chỉ có 1–17 lần.
- Lý do (phỏng đoán theo cơ chế vLLM V1): vLLM nhận request mới khi KV còn đủ chỗ cho **prompt**, chưa tính phần output sẽ sinh. Output W1 dài (200–600 token, ~45% kích thước request), nên khi các request cùng sinh thêm token thì KV đầy và vLLM phải tạm dừng một số request, rồi tính lại prompt của chúng sau.
- Phần tính lại này tốn prefill thừa. Một phần nguyên nhân khiến TPOT W1 p99 tăng lên 136–149 ms ở 80–96 user.
- → Ứng viên tune: giới hạn số request được nhận (`max_num_seqs`), và tăng KV.

### 4.3 So với giả định trong [plan.md](../../plan.md)

| | Giả định | Đo được | Chênh |
|---|---|---|---|
| Decode W1 (token/s ra) @ 8 / 16 / 32 / 64 user | 260 / 380 / 520 / 600 | 243 / 323 / 458 / 610 | −7% / −15% / −12% / +2% |
| Prefill (token/s vào, lúc bão hoà) | ~5,000 | W2 ~2,780 (kèm decode), W3 ~1,940 | −44% / −61% |
| Đơn giá máy | ~10,200 VND/h | $1.516 / 5.25 h = $0.289/h ≈ 7,570 VND/h (gồm ổ đĩa, tải về) | −26% |

**Vì sao prefill thấp hơn nhiều so với giả định.** Giả định dựa trên 50% của 121 TFLOPS FP16 đỉnh. Thực tế:
1. GPU chạy ở trần công suất 72 W, xung chỉ ~1,100 MHz, khoảng một nửa xung tối đa (đo được).
2. Ở trạng thái bão hoà, mỗi bước còn gánh decode của các request khác.
3. Prompt W3 dài 4–8k token, nên phần attention (tăng theo bình phương độ dài) chiếm đáng kể.

Ước 1 và 3 đóng góp bao nhiêu thì cần profile (GĐ3).

### 4.4 Nhiễu đo

Trên 22 mức × 3 lần (độ lệch (max − min) / trung vị):

| Chỉ số | Trung vị | p75 | Max |
|---|---|---|---|
| Throughput ra | 1.6% | 2.7% | 5.1% |
| TPOT p95 | 3.6% | 4.1% | 5.9% |
| E2EL p50 | 5.7% | 10.1% | 14.8% |
| **TTFT p95** | **15.7%** | 23.5% | 47.8% |

Xác nhận phát hiện của P03 ([challenges](../../challenges.md) #5):
- Throughput và TPOT ổn định trong ~5%.
- TTFT p95 của một lần đo dao động 15–25%, đôi khi gần 50%. Lớn nhất ở các mức mà KV vừa đầy (W1/16, W2/18, W3/6). Ở đó một vài cụm request phải chờ hay không phải chờ là đổi hẳn p95.
- Ở các mức đã quá tải sâu (W2/64, W3/32–64), TTFT p95 lệch chỉ 2–4%, vì mọi request đều chờ đều.
- **Hệ quả:** chênh goodput 1 bước tìm (2 user ở W2) **chưa đủ** để nói hai cấu hình khác nhau. Cần chênh ≥ 2 bước, hoặc khoảng không chồng nhau rõ ràng.

### 4.5 W2: 18 hay 20?

- Theo quy tắc đã chốt (cửa sổ all, trung vị 3 lần), goodput W2 = 18, vì 20 có trung vị 10.06 s.
- Nhưng ở 20 user có 1/3 lần đạt (8.76 s), và tính theo cửa sổ steady thì 20 đạt (8.86 s), goodput W2 = 20 (24 trượt).
- **Kết luận trung thực: điểm gãy của W2 nằm quanh 18–20 user.**
- Báo cáo giữ con số chính thức là **18** để đúng quy tắc chốt trước khi đo (không đổi quy tắc sau khi thấy số). Các so sánh về sau (tune, thư viện khác) dùng cùng quy tắc nên vẫn công bằng.
- W1 và W3 không phụ thuộc cửa sổ.

## 5. Vấn đề gặp phải

Không có lỗi trong P04: 66/66 đơn vị xong, 0 request lỗi, không phải chạy lại đơn vị nào.

| Điều cần ghi | Chi tiết |
|---|---|
| Log P04 ghi "KV cache = None" ở lần khởi động 1 | r1 dùng lại server đang chạy từ P03, nên thư mục P04 chưa có lần khởi động nào để lấy số. KV thật là 37,264 (lần khởi động 3 của P03, `results/gd1/03-.../server_starts.jsonl`) |
| `--status` ước giờ còn lại quá cao (4.2 h lúc còn ~1.8 h) | Cách đoán số mức còn phải tìm còn thô. Không ảnh hưởng kết quả; cải thiện sau nếu cần |
| W2 / 20 sát ngưỡng | Mục 4.5 |

## 6. Ảnh hưởng tới bước sau

**P05 (W4):**
- *Ước k khởi đầu (phép tính, chưa đo):*
  - Từ throughput bão hoà W1 và W2, giải ra chi phí mỗi request ≈ 0.37 ms × token prompt (chưa trúng cache) + 1.17 ms × token output. Kiểm với W3 thì mô hình cho năng lực cao hơn đo được 23%.
  - Kịch bản peak có prompt TB ~950 token và output TB ~270 token, nên ~0.67 s/request, tức trần ~5,400 request/giờ.
  - Giới hạn KV (~34 request đồng thời ở ~1,085 token mỗi request) cho ~4,500 request/giờ.
  - Peak có ~203 request/giờ cho mỗi k, nên **k ≈ 22** là trần. Trừ biên cho đột biến tải và tài liệu dài (TTFT p95 nhạy với các request 4–8k token), tôi đề xuất **khởi đầu nóng ở k = 16 và k = 24**, rồi nhân đôi / chia đôi / tìm nhị phân.
- *Giờ GPU P05:*
  - ~4–5 lần chạy để tìm k, cộng 3 lần chạy kiểm ổn định (seed 2, peak2, normal) = 7–8 lần × 40 phút ≈ 4.7–5.3 h, thêm cài đặt và khởi động ~0.3 h.
  - **≈ 5–5.6 h ≈ 38–43k VND**, ít hơn ước tính 6.5 h của plan.
- *Còn phải làm ở local trước P05 (0 giờ GPU):* harness chưa có chế độ chạy W4 thành đơn vị chạy tiếp được. `session_client.py scenario` chạy được một lần đo, nhưng chưa có `done.json`, khởi đầu nóng trên k, hay `--status`. Cần thêm chế độ `w4` vào `sweep.py`, dùng chung `search_step`.
- Ghi lại độ trễ gửi so với lịch (client ghép prompt bằng tokenizer).

**P06:** không đổi.

**P07 / tune GĐ1:** các nghi vấn đầu tiên (chưa profile, chỉ suy từ số):
1. **Dung lượng KV là nghẽn số 1 của cả ba workload.** KV FP8 (gấp đôi số token) và `gpu_memory_utilization` > 0.92 là hai cờ phải thử đầu tiên. P03 đã thấy +30% KV giảm TTFT p95 W2/16 khoảng một nửa.
2. **Preemption W1** (100–330 lần mỗi lần đo) → thử `max_num_seqs` nhỏ hơn và chính sách nhận request.
3. **Prefill chậm hơn giả định 44–61%** → `max_num_batched_tokens` (mặc định 2,048), backend attention (FlashInfer so với FlashAttention 2) cho prompt 4–8k.
4. GPU luôn ở trần 72 W → mọi giảm số phép tính / lượng đọc bộ nhớ đều thành tốc độ; cần đo % roofline ở GĐ3 với xung thật ~1.1 GHz, không phải xung tối đa.
5. **Sweep rút gọn khi tune:** dùng khởi đầu nóng ở (lo, hi) = (68, 72) / (18, 20) / (6, 8).

## 7. Thuật ngữ dùng trong báo cáo này

| Từ | Nghĩa |
|---|---|
| Goodput | Số user đồng thời lớn nhất vẫn đạt SLO; báo dạng "lo (hi trượt)" |
| SLO chính / phụ | TTFT p95 ≤ 10 s và TPOT p95 ≤ 150 ms / 2 s và 100 ms |
| Lưới gốc | Các mức 8 / 16 / 32 / 64 luôn đo đủ |
| Tìm nhị phân | Đo mức giữa khoảng đạt–trượt, thu hẹp dần |
| Cửa sổ all / steady | Tính p95 trên mọi request / bỏ N request gửi đầu và N cuối |
| Bão hoà | Trạng thái số request chạy cùng lúc đã đạt trần (do KV), thêm user chỉ làm hàng đợi dài ra |
| Hàng đợi (queue time) | Thời gian request nằm chờ trước khi được chạy |
| Preemption | vLLM tạm dừng request đang chạy vì hết KV, rồi tính lại sau |
| Trần công suất | L4 giới hạn 72 W; chạm trần thì GPU tự giảm xung |
| Khởi đầu nóng | Bắt đầu tìm từ (lo, hi) của cấu hình trước thay vì lưới gốc |
