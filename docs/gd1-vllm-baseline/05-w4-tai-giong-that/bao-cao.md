# Báo cáo P05 — W4: tải giống thật (phiên người dùng BurstGPT)

> Ngày làm: 2026-10-13 (chuẩn bị local buổi sáng, đo trên L4 14:28 → 19:26) · Máy: host 506934 / machine 150535, instance 55701260 ·
> vLLM 0.30.0 + `--max-model-len 16384` (+ `--enable-prompt-tokens-details`, chỉ để ghi số token trúng cache) ·
> Giờ GPU: 4.97 h (≈ $1.878 ≈ 49,200 VND theo tín dụng bị trừ)

## 1. Tóm tắt

- vLLM mặc định trên một L4 phục vụ được **18 ngày tải giờ cao điểm (16h) chồng lên nhau** mà vẫn đạt SLO chính
  (TTFT p95 9.1 s ≤ 10 s, TPOT p95 82 ms ≤ 150 ms). Con số này tương ứng **~3,480 request/giờ** và **~363 phiên chat đang mở cùng lúc**.
  Với k = 20 thì trượt.
- Với một bộ 18 ngày khác (seed 2), k = 18 trượt và k = 16 đạt. **Goodput W4 vì vậy là 16–18 tuỳ bộ ngày**, lệch 1 bước tìm.
- SLO hỏng **theo đợt**. TTFT p95 tính theo từng phút chỉ vượt 10 s ở 4 trong 30 phút. Đợt hỏng trùng với **phút có lượng token prompt
  gấp ~2 lần bình thường** (người dùng dán tài liệu dài), không trùng với phút có nhiều request.
- Request ngắn bị request dài làm chậm rõ rệt. Lúc có request ≥ 4k token đang chờ token đầu, TTFT p95 của request < 1k token
  là 26–32 s; lúc không có là 1.4–4.2 s.
- Prefix cache gần như không giúp được hội thoại nhiều lượt. Người dùng nghỉ ≥ 2 phút thì lịch sử đã bị đẩy khỏi cache hoàn toàn
  (98–100% lượt trúng 0 token). Phần câu trả lời của lượt trước **không bao giờ** trúng cache, vì template chat của Qwen3.
- Thuê cả ngày cho mức k = 18: **~22,500 VND / 1 triệu token ra**. Nếu chỉ tính giờ cao điểm thì là ~10,300, tức gấp ~2.2 lần
  so với chỉ nhìn giờ cao điểm.
- Đạt tiêu chí xong của plan: goodput dạng khoảng hẹp (18 / 20), đã chạy seed 2, peak2, normal; số tóm tắt ở measurements.md.

## 2. Đã làm gì và vì sao

### 2.1 Chuẩn bị ở local (0 giờ GPU)

Chi tiết và lý do từng quyết định: [plan.md](plan.md) mục "Chuẩn bị local". Tóm tắt:
- Thêm chế độ `w4` vào harness. Mỗi lần chạy (khung giờ × k × lần lặp) là một đơn vị chạy tiếp được.
  Tìm k bằng đúng logic tìm nhị phân của P04, xong thì tự chạy các kịch bản kiểm tra ở k = lo.
- Kịch bản k ngày = kịch bản gốc 110 ngày lọc theo k ngày đầu của hoán vị.
  Bước kiểm tra "lọc ra đúng bằng sinh riêng" phát hiện lỗi phiên nhiều ngày trong `build_w4.py` (challenges.md #6). Đã sửa và sinh lại.
- **Dừng sớm**: khi số request chắc chắn trượt SLO vượt 5% số request tối đa có thể gửi, p95 của cả lần chạy chắc chắn vượt SLO,
  nên dừng. Kết quả đạt/trượt giống hệt chạy đủ.
- **Chờ request còn dở tối đa ~654 s sau cửa sổ.** Request nào đạt SLO đều xong trước mốc này.
- Thêm cờ `--enable-prompt-tokens-details`. Cờ này chỉ ghi số token trúng cache vào câu trả lời; theo code vLLM 0.30.0, số này được tính
  dù có cờ hay không, nên không đổi tốc độ.

### 2.2 Trên L4

1. Thuê machine 150535 (cùng host 506934 với P03/P04). Checklist đạt; `ncu` vẫn bị chặn.
2. `remote_bench.sh bench/configs/w4.yaml` trong tmux. Lần khởi động đầu biên dịch lại từ đầu và chỉ có KV 29,472 token (hiện tượng của
   challenges.md #4); harness tự bật lại, được KV 38,032 token.
3. Điểm mốc W2 / 16 user × 3 để so với máy P04.
4. Tìm k: 16 → 24 → 20 → 18 (bảng 3.2).
5. Kiểm tra ở k = 18: peak seed 2, peak2 (10h), normal (13h).
6. **Thêm một lần chạy so với plan: peak seed 2 ở k = 16.**
   - Lý do: seed 2 trượt ở k = 18. Plan chỉ yêu cầu ghi "khác > 1 bước tìm" vào báo cáo, nhưng chỉ với một điểm thì không biết
     bộ ngày thứ hai lệch đúng 1 bước hay nhiều hơn.
   - Giới hạn đặt trước khi chạy: chỉ 1 lần chạy. Nếu trượt thì ghi "< 16" và dừng.
   - Goodput chính vẫn tính trên seed 1 như plan.
7. Kéo kết quả về (kiểm checksum khớp), destroy.

Laptop tắt khoảng 17:05–17:35. Máy thuê chạy tiếp trong tmux, không ảnh hưởng kết quả.

## 3. Kết quả

### 3.1 Điểm mốc: máy này so với máy P04

| W2 / 16 user | TTFT p95 (ms) | TPOT p95 (ms) | tok/s ra |
|---|---|---|---|
| P05 (machine 150535), 3 lần | 5,714 / 4,961 / 5,722 | 87.9 / 92.1 / 88.7 | 175 / 173 / 170 |
| P04 (machine 116596), 3 lần | 6,303 / 4,866 / 6,025 | 89.6 / 92.7 / 89.9 | 171 / 169 / 167 |

Throughput cao hơn ~2%, KV nhiều hơn 2% (38,032 vs 37,264 token). Hai máy so sánh được với nhau, trong phạm vi nhiễu của P03.

### 3.2 Tìm goodput (peak 16h, seed 1)

| k | Phiên | Request đo | TTFT p50 / p95 / p99 (s) | TPOT p50 / p95 (ms) | Kết quả | Thời gian |
|---|---|---|---|---|---|---|
| 16 | 916 | 1,628 | 0.24 / 5.49 / 25.5 | 47 / 76 | đạt | 40.6 phút |
| 24 | 1,257 | 617 | 0.46 / ∞ / ∞ | 74 / ∞ | trượt, dừng sớm sau 8.7 phút đo | 18.9 phút |
| 20 | 1,084 | 1,099 | 0.32 / ≥ 30.9 / ∞ | 63 / ≥ 191 | trượt, dừng sớm sau 17.4 phút đo | 27.7 phút |
| **18** | 974 | **1,738** | 0.27 / **9.11** / 25.3 | 56 / **82** | **đạt** | 40.7 phút |

**Goodput W4 = 18 (20 trượt).** Ở k = 18:
- 3,476 request/giờ;
- 363 phiên đang mở (p50; phiên mở = từ lượt đầu tới 10 phút sau lượt cuối);
- 14 request đang chạy trên server (p50), tối đa 32.

"∞" nghĩa là hơn 5% request bị huỷ lúc dừng sớm hoặc lỗi. Số của lần dừng sớm là của phần đã chạy.

### 3.3 Kiểm tra độ ổn định và khung giờ khác

| Lần chạy | Request đo | TTFT p95 | TPOT p95 | Kết quả |
|---|---|---|---|---|
| peak seed 2, k = 18 | 685 | ∞ | ∞ | trượt, dừng sớm sau 10 phút đo |
| peak seed 2, k = 16 (thêm) | 1,770 | 0.87 s | 73 ms | đạt |
| peak2 10h, k = 18 | 1,555 | 0.62 s | 54 ms | đạt |
| normal 13h, k = 18 | 628 | 0.56 s | 36 ms | đạt |

- Seed 2 cho goodput **16 (18 trượt)**, lệch 1 bước tìm so với seed 1.
- Ở cùng k = 16, seed 1 có TTFT p95 5.5 s, còn seed 2 chỉ 0.87 s. TTFT p95 thay đổi rất mạnh theo bộ ngày (mục 4.2).
- Giờ 10h và 13h đạt SLO với dư rất nhiều ở k = 18. Nếu server đủ cho giờ 16h thì đủ cho các giờ khác.

### 3.4 Goodput theo các ngưỡng (tính lại từ cùng dữ liệu, không đo thêm)

| SLO | Goodput k (seed 1) |
|---|---|
| Chính: TTFT 10 s / TPOT 150 ms | 18 (20 trượt) |
| Phụ: 2 s / 100 ms | < 16 |
| TTFT 2 s / 5 s (TPOT 150 ms) | < 16 |
| TTFT 20 s | ≥ 18 (k = 20 dừng sớm nên không biết với ngưỡng 20 s) |

### 3.5 Tách theo loại request (peak seed 1, k = 18)

| Nhóm | Số request | TTFT p50 / p95 (s) | TPOT p95 (ms) |
|---|---|---|---|
| prompt < 1k token | 1,322 | 0.23 / 8.6 | 86 |
| 1–2k | 255 | 0.40 / 9.4 | 75 |
| ≥ 2k | 161 | 0.77 / 10.0 | 72 |
| lượt đầu phiên mới | 420 | 0.20 / 9.5 | 80 |
| quay lại phiên cũ | 239 | 0.37 / 7.4 | 77 |
| lượt sau (≥ 2) | 1,079 | 0.29 / 9.5 | 84 |

Mọi nhóm có TTFT p95 gần bằng nhau (7–10 s). Request ngắn chiếm 76% số request nhưng cũng chịu đuôi TTFT như request dài.

### 3.6 Request ngắn khi có / không có request dài đang chờ

| Lần chạy | TTFT p95 request < 1k: có request ≥ 4k đang chờ token đầu | không có | TPOT p95 < 1k: có / không |
|---|---|---|---|
| peak k = 16 | **26.6 s** (92 request) | 1.4 s (1,145) | 90 / 76 ms |
| peak k = 18 | **32.2 s** (115) | 4.2 s (1,207) | 95 / 81 ms |
| peak2 k = 18 | 2.9 s (28) | 0.30 s (1,245) | 68 / 53 ms |
| normal k = 18 | 0.38 s (2) | 0.21 s (437) | 48 / 34 ms |

Thêm: trong số request < 1k có TTFT > 10 s, 21/30 (k = 16) và 28/53 (k = 18) được gửi đúng lúc có request ≥ 4k đang chờ token đầu.

### 3.7 Prefix cache theo think time

Lượt sau không phải bỏ bớt lịch sử. Mỗi ô: phần lịch sử trúng cache (trung bình) · số lượt.

| Lần chạy | think < 30 s | 30–120 s | 120–300 s | 300–900 s | ≥ 900 s |
|---|---|---|---|---|---|
| peak k = 16 | 32% · 128 | 2% · 184 | 0% · 118 | 0% · 109 | 0% · 27 |
| peak k = 18 | 25% · 134 | 0% · 195 | 0% · 124 | 0% · 115 | 0% · 27 |
| peak2 k = 18 | 46% · 153 | 14% · 175 | 0% · 117 | 0% · 101 | 0% · 26 |
| normal k = 18 | 51% · 54 | 40% · 81 | 4% · 42 | 0% · 30 | 0% · 8 |

- Tỉ lệ trúng prefix của cả server ở peak k = 18 chỉ **3.0%** số token prompt.
- **Phần câu trả lời lượt trước không trúng ở bất kỳ lượt nào** (0% lượt có số token trúng > prompt lượt trước + 16). Phần này chiếm p50 37%
  lịch sử.
- Ở 13h (vắng), lượt sau trong 30 s trúng ~93% (p50) prompt lượt trước. Ở 16h k = 18, p50 là 0%: đã bị đẩy ra ngay trong 30 s.

### 3.8 Hình

- [Hình 1](hinh/h1-ttft-tpot-theo-k.png): TTFT p95 và TPOT p95 theo k.
- [Hình 2](hinh/h2-theo-phut-k-lo.png): ở k = 18, theo từng phút của cửa sổ đo: số request (trên), TTFT p95 (giữa), TPOT p95 (dưới).

### 3.9 Chi phí theo cả ngày

| Cách tính | VND / 1 triệu token ra |
|---|---|
| **Thuê cả ngày, phục vụ k = 18 ngày tải theo đủ 24 giờ của trace** (10.40 triệu token ra / ngày) | **22,510** |
| Nếu cả ngày đều đông như 16h (đo được 944,804 token ra / giờ) | 10,322 |
| Nếu cả ngày như 10h | 13,833 |
| Nếu cả ngày như 13h | 28,340 |

Giá dùng: 9,752 VND/h ($0.372 × 26,200). Token ra đo được so với trace × k: 16h −5%, 10h −12%, 13h −26%.
Cửa sổ 13:00–13:30 vắng hơn trung bình cả giờ 13 (nghỉ trưa), xem measurements mục "P05 chuẩn bị local".

### 3.10 Thời gian

| Phần | Thời gian |
|---|---|
| Thuê → máy chạy | 3.5 phút |
| Đẩy code + kịch bản (92 MB) lên | ~6 phút (mạng tải lên của laptop chậm) |
| Cài đặt (tải model 179 s, vLLM 187 s) + 2 lần khởi động server | ~10.5 phút |
| Điểm mốc W2/16 × 3 | 6 phút |
| 8 lần chạy W4 | 4.49 h (3 lần dừng sớm: 18.9 + 27.7 + 20.3 phút) |
| Tổng thuê | 4.97 h |

Dừng sớm tiết kiệm ít nhất ~55 phút so với chạy đủ ~41 phút cho cả 3 lần trượt, tức khoảng 9,000 VND. Thực tế còn nhiều hơn, vì lần
chạy quá tải không dừng sớm còn phải chờ request dở tới ~11 phút sau cửa sổ.

Kết quả thô: `results/gd1/05-w4-tai-giong-that/20261013-p05/` (`summary.md`, `summary.json`, `cost.json`, `units/<id>/`).
Số tóm tắt: [measurements.md](../../measurements.md) mục "2026-10-13 — P05".

## 4. Giải thích kết quả

### 4.1 SLO hỏng vì đợt dồn token prefill, không phải vì tải trung bình (đo được + suy ra)

**Đo được:**
- Ở k = 18, server chỉ ra 259 tok/s, bằng 42% mức W1 đạt khi bão hoà ở P04 (605–623 tok/s ở 68 user). Độ dài câu hỏi của hai bài gần nhau:
  W4 TB 767 / 272 token vào / ra, W1 ~405 / 333.
- Số request đang chạy p50 là 14, nhưng tối đa 32. Hàng đợi chỉ xuất hiện trong 11% số giây.
- TTFT p95 theo phút (Hình 2) chỉ vượt 10 s ở phút 7–8 và 16–17. Phút 16 có 85,318 token prompt, gấp 1.9 lần trung vị 43,872 token/phút,
  với 4 request ≥ 4k. Số request phút đó là 65, nằm trong khoảng bình thường 41–73.

**Suy ra:** goodput W4 do khả năng chịu đợt dồn token prefill quyết định. Phần lớn thời gian server rảnh hơn W1 của P04. Theo ước tính
của P04 (0.37 ms / token prefill), 85k token prompt cần ~31 s GPU chỉ để prefill, trong một phút mà server còn phải sinh token
cho các request khác. Hàng đợi lúc đó dâng lên và TTFT tăng tới 39 s (p95 của phút 16).

**Suy ra:** vì vậy ước tính k ≈ 22 của P04 (dựa trên throughput trung bình) cao hơn thực tế 18 khoảng 20%. Ước tính bằng tải trung bình
không dùng được cho tải giống thật; cần tính cả đuôi phân bố của token prompt mỗi phút.

### 4.2 Vì sao goodput nhạy với bộ ngày

Đo được: cùng k = 16, seed 1 có TTFT p95 5.5 s, seed 2 chỉ 0.87 s. Ở k = 18, seed 1 đạt với 9.1 s, seed 2 trượt.

Suy ra: hai bộ ngày có số request gần nhau (1,628 vs 1,770 ở k = 16). Khác biệt nằm ở **thời điểm các request dài rơi vào nhau**. Nếu
vài tài liệu dài đến cùng một phút, phút đó hỏng và kéo p95 cả 30 phút lên. Đây là bản chất của p95 trên tải có đột biến: kết luận
"18 người" hay "16 người" phụ thuộc vào việc có đột biến lớn trong cửa sổ đo hay không.

Hệ quả cho báo cáo cuối: goodput W4 nên báo dạng **khoảng 16–18** và ghi rõ là do bộ ngày. So sánh cấu hình (tune, thư viện khác, custom CUDA)
phải dùng **cùng bộ ngày** (cùng seed), nếu không thì chênh lệch 1 bước có thể chỉ là do chọn ngày.

### 4.3 Request dài làm chậm request ngắn (đo được tương quan; cơ chế là phỏng đoán)

Đo được (mục 3.6):
- Request ngắn chịu TTFT p95 26–32 s nếu đúng lúc đó có request ≥ 4k đang chờ token đầu, so với 1.4–4.2 s nếu không.
- TPOT của request ngắn cũng tăng (90 vs 76 ms), nhưng vẫn dưới SLO.

Phỏng đoán cơ chế, có hai nguồn, cần profile để xác nhận:
1. **Ngân sách token mỗi bước.** vLLM mặc định xử lý tối đa 2,048 token mỗi bước (`max_num_batched_tokens`), chia prompt dài thành
   nhiều khúc (chunked prefill). Một prompt 5k token chiếm ngân sách của ~3 bước. Trong lúc đó request mới phải xếp hàng.
2. **KV đầy.** Trong các giây có hàng đợi, 58% có KV ≥ 97% (k = 18). Có 80 lần preemption: vLLM tạm dừng request đang chạy vì hết KV,
   rồi tính lại sau. Request dài cần nhiều khối KV nên dễ gây đầy.

Đây là vấn đề "request dài chặn request ngắn" mà W1–W3 thuần không đo được: trong W1–W3 mọi request cùng loại.

### 4.4 Prefix cache không giữ được lịch sử hội thoại (đo được + cơ chế đã kiểm)

**Phần câu trả lời không bao giờ trúng.**
- Đo được: 0% lượt.
- Cơ chế (đã kiểm ở `workloads/test_w4_session.py`):
  - Khi sinh câu trả lời, prompt kết thúc bằng khối `<think></think>` rỗng (4 token, do `enable_thinking=False`).
  - Khi câu trả lời đó nằm trong lịch sử của lượt sau, template Qwen3 bỏ khối này.
  - Mọi token từ chỗ đó trở đi lệch vị trí, nên các khối KV của câu trả lời có mã băm khác và không dùng lại được.
- Hệ quả: ở mọi lượt, server phải prefill lại toàn bộ câu trả lời trước (p50 37% lịch sử). Ứng dụng chat thật dùng template này cũng chịu
  y như vậy.

**Phần prompt cũ bị đẩy khỏi cache rất nhanh.** KV có 38,032 token. Ở k = 18 có 363 phiên mở, lịch sử trung bình ~770 token, tức tổng
~280,000 token lịch sử. Con số này gấp ~7 lần KV. Với think time trung vị 124 s, lịch sử bị đẩy ra trước khi người dùng quay lại. Ở 13h
(117 phiên mở) KV vẫn giữ được trong ~2 phút.

**Suy ra:** muốn dùng lại lịch sử cần chỗ chứa lớn hơn KV trên GPU nhiều lần. Ví dụ đẩy KV sang RAM CPU (LMCache, máy thuê có 125 GB RAM)
hoặc KV FP8 (gấp đôi chỗ). Đây là ứng viên "bổ trợ" rõ ràng cho GĐ2.

### 4.5 Một L4 phục vụ bao nhiêu "người dùng"? (so với P04)

| | P04 W1 (chat, closed-loop) | P05 W4 (người thật, peak 16h) |
|---|---|---|
| Goodput | 68 user luôn có request trên server | 18 ngày tải ≈ **363 phiên đang mở**, 974 phiên trong 40 phút |
| request / giờ | ~6,600 | 3,476 |
| Request đang chạy trên server | ~60 | 14 (p50), 32 (max) |
| tok/s ra | 605–623 | 259 |

- **Số người dùng gấp ~5 lần** (363 vs 68), vì người thật đọc và nghỉ (think time trung vị 124 s). Phần lớn thời gian họ không có request
  trên server.
- **Số request/giờ chỉ bằng ~53%** W1. Lý do: server phải để dư chỗ cho đợt dồn (mục 4.1), và prompt W4 dài hơn W1 do mang lịch sử
  hội thoại, lại không trúng cache (mục 4.4).

### 4.6 Chi phí cả ngày gấp ~2.2 lần chỉ tính giờ cao điểm

GPU thuê theo giờ phải chạy cả 24 giờ, nhưng tải giờ hành chính cao còn đêm gần như không có ai (trace: 3–7h chỉ 4–10 request/giờ cho mỗi k).
Nếu đặt đủ GPU cho giờ cao điểm thì các giờ khác dùng ít. 22,510 VND / 1 triệu token ra là con số thực tế cho doanh nghiệp tự host một
chatbot theo nhịp giờ hành chính. Dùng con số giờ cao điểm (10,322) thì đánh giá thấp chi phí khoảng một nửa.

## 5. Vấn đề gặp phải

| Vấn đề | Xử lý |
|---|---|
| Lỗi phiên nhiều ngày trong `build_w4.py` (phát hiện ở local, trước khi thuê) | Sửa, sinh lại mọi kịch bản; [challenges.md](../../challenges.md) #6 |
| Lần khởi động đầu biên dịch lại từ đầu, KV ít hơn 23% | Harness tự bật lại (cơ chế từ P03, challenges.md #4). Tốn ~2 phút |
| Bộ canh ở laptop bỏ lỡ một lần chạy xong (18:03), báo trễ 8 phút | Do bộ canh mới đọc log từ vị trí lúc nó khởi động. Sửa: lưu vị trí đã đọc vào file. Không ảnh hưởng số đo |
| Laptop tắt khoảng 17:05–17:35 | Không ảnh hưởng: chạy trong tmux trên máy thuê; kết quả kéo về đủ, kiểm checksum khớp |
| Mạng tải lên của laptop chậm (~6 phút cho 92 MB) | Lần sau chỉ đẩy các kịch bản cần dùng, hoặc nén trước |

## 6. Ảnh hưởng tới bước sau

**P06 (chất lượng):** không đổi.

**P07 / tune GĐ1.** Thứ tự ưu tiên, kết hợp với P04:
1. **Tăng chỗ KV** (KV FP8, `gpu_memory_utilization`). Cả P04 lẫn P05 đều chỉ ra KV đầy. Với W4, KV còn quyết định cache lịch sử hội thoại.
2. **`max_num_batched_tokens` / chính sách prefill** để request dài không chặn request ngắn (mục 4.3). Đo bằng bảng 3.6 và TTFT p95 theo phút.
3. **Dùng lại KV của câu trả lời trước.** Có thể giữ khối `<think></think>` rỗng trong lịch sử khi render, để lượt sau trùng prefix
   với lượt trước kể cả phần câu trả lời. Cần kiểm chất lượng output vì đổi prompt. Đây là thay đổi phía ứng dụng, phải ghi rõ khi so sánh.
4. **GĐ2 bổ trợ:** LMCache (đẩy KV sang RAM CPU) cho hội thoại có think time ~2 phút.

**Phương pháp đo W4 cho các bước sau:**
- So sánh hai cấu hình phải dùng **cùng seed** (cùng bộ ngày). Goodput khác nhau 1 bước giữa hai seed là nhiễu tự nhiên của W4.
- Khởi đầu nóng ở (lo, hi) = (18, 20). Mỗi lần tìm ~3–4 lần chạy, ~2–2.5 h. Lần trượt dừng sớm nên ngắn.
- Có thể báo thêm "% phút có TTFT p95 > SLO". Chỉ số này ít nhạy hơn p95 cả cửa sổ với một đợt dồn đơn lẻ. Chỉ thêm như chỉ số phụ,
  không đổi quy tắc đạt/trượt đã chốt.

## 7. Thuật ngữ dùng trong báo cáo này

| Từ | Nghĩa |
|---|---|
| k (W4) | Số ngày của trace phát chồng lên nhau ở cùng khung giờ; k ngày ≈ k lần số người dùng |
| Seed (W4) | Số chọn hoán vị ngày; seed khác → bộ k ngày khác |
| Kịch bản gốc | Kịch bản W4 sinh với cả 110 ngày; kịch bản k ngày = lọc k ngày đầu của hoán vị |
| Phiên đang mở | Phiên đã gửi lượt đầu và chưa quá 10 phút sau lượt cuối |
| Dừng sớm | Dừng lần chạy khi số request chắc chắn trượt SLO > 5% số request tối đa có thể gửi |
| Think time | Thời gian người dùng đọc và nghỉ giữa hai lượt hỏi |
| Prefix cache | vLLM giữ KV của các khối prompt đã tính; prompt sau có cùng phần đầu thì dùng lại, không prefill lại |
| Chunked prefill / ngân sách token mỗi bước | vLLM chia prompt dài thành khúc; mỗi bước xử lý tối đa `max_num_batched_tokens` token (mặc định 2,048) |
| Preemption | vLLM tạm dừng request đang chạy vì hết KV, rồi tính lại sau |
| Goodput | Mức tải lớn nhất vẫn đạt SLO; báo dạng "lo (hi trượt)" |
