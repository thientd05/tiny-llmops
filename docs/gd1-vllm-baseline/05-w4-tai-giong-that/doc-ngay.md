# P05 — Đọc ngay: một L4 phục vụ được bao nhiêu người dùng thật

> Bản dễ đọc, kể lại từng bước. Bản đầy đủ số liệu: [bao-cao.md](bao-cao.md).

**Một câu:** chạy 5 giờ trên L4, tốn khoảng **49,200 VND**. Kết quả: vLLM mặc định chịu được giờ cao điểm 16h của **18 ngày chồng lên nhau**,
tức khoảng **3,480 câu hỏi/giờ** từ **~363 người đang chat cùng lúc**, mà vẫn đạt mức phục vụ đã chọn. Với một bộ ngày khác thì
được 16. Server không hỏng vì tải trung bình. Nó hỏng trong vài phút có nhiều người dán tài liệu dài cùng lúc.

---

## 0. Nhắc lại: W4 khác P04 ở đâu

- **P04** đo N người gửi câu hỏi liên tục, không nghỉ, mọi câu cùng một loại.
- **W4 (P05)** phát lại hành vi người dùng thật từ trace BurstGPT (bản ghi dịch vụ chat thật). Trace chỉ có thời điểm và số token,
  không có chữ, nên phần chữ lấy từ ShareGPT và RAGBench.
  - Mỗi người mở một **phiên** (cuộc hội thoại). Lượt sau mang theo lịch sử, nên hội thoại dài dần.
  - Gửi câu sau khi đã **nhận xong câu trả lời** và đọc/nghỉ một lúc (**think time**, trung vị 124 s).
  - Đôi khi dán một tài liệu dài.
- **Tăng tải** bằng cách phát chồng khung giờ 15:50–16:30 của **k ngày khác nhau**. k = 18 nghĩa là 18 lần số người dùng của một ngày
  bình thường lúc 16h.
- Mỗi lần chạy: **10 phút khởi động** (không tính) + **30 phút đo**.
- **Đạt** khi trên mọi câu hỏi gửi trong 30 phút đo:
  - TTFT p95 ≤ 10 s: 95% câu hỏi thấy chữ đầu tiên trong 10 giây;
  - TPOT p95 ≤ 150 ms: chữ chạy nhanh hơn tốc độ đọc.

---

## 1. Chuẩn bị ở laptop (buổi sáng, 0 đồng)

1. Thêm chế độ **`w4`** vào công cụ đo (`bench/sweep.py`). Nó tự chọn k tiếp theo bằng tìm nhị phân, giống P04. Mỗi lần chạy có file
   `done.json` đánh dấu xong, nên máy chết giữa chừng thì chạy lại cùng lệnh là đi tiếp.
2. **Dừng sớm.** Khi đã có quá nhiều câu hỏi chắc chắn trượt (hơn 5% số câu có thể gửi), kết quả chắc chắn là trượt, nên dừng để đỡ
   tiền GPU.
3. Thử toàn bộ với **server giả** (chương trình trả lời giống vLLM nhưng không chạy model), thời gian chạy nhanh gấp 100 lần. Có thử
   ngắt giữa chừng bằng `kill -9`: chạy lại thì phần đã xong không bị đo lại.
4. **Bắt được một lỗi trước khi thuê máy.** Trace có người quay lại hội thoại cũ vào ngày khác. Code cũ chỉ lấy lần đầu nên mất các lượt
   sau. Đã sửa và sinh lại mọi kịch bản ([challenges.md](../../challenges.md) #6).

---

## 2. Thuê máy và chạy (14:28)

```bash
vastai create instance 54285092 --image nvidia/cuda:13.0.2-devel-ubuntu22.04 --disk 140 --ssh --direct
```

| Phần | Nghĩa |
|---|---|
| `54285092` | Mã máy cho thuê: L4, host 506934 (cùng host với P03/P04), machine 150535 |
| `--image ...` | Môi trường có sẵn CUDA 13.0, khớp vLLM 0.30.0 |
| `--disk 140` | Ổ 140 GB cho vLLM + model 16 GB |
| `--ssh --direct` | Kết nối SSH thẳng vào máy |

Giá $0.372/giờ (~9,750 VND), đã gồm ổ đĩa.

Đẩy code và 4 kịch bản lên máy (92 MB, mất ~6 phút vì mạng tải lên của laptop chậm), rồi chạy:

```bash
tmux new -d -s exp "bash scripts/remote_bench.sh bench/configs/w4.yaml /root/results/p05"
```

| Phần | Nghĩa |
|---|---|
| `tmux new -d -s exp` | Chạy trong phiên tmux tên `exp` **trên máy thuê**. Laptop tắt thì nó vẫn chạy |
| `remote_bench.sh` | Cài đặt (bỏ qua phần đã có) → đo → tổng hợp |
| `w4.yaml` | Chế độ W4: tìm k bắt đầu từ 16 và 24, xong thì kiểm tra ở k tìm được |

Trên laptop có hai việc chạy nền: kéo kết quả về mỗi 2 phút, và một bộ canh đánh thức tôi mỗi 20 phút, khi có lỗi, hoặc khi một lần
chạy xong.

**Lần khởi động server đầu** biên dịch lại từ đầu (61 s) và chỉ có KV cache 29,472 token. Đây là hiện tượng đã biết từ P03. Công cụ tự tắt và
bật lại, được **38,032 token** (máy P04: 37,264).

---

## 3. Điểm mốc: máy này có giống máy P04 không? (14:49–14:55)

Đo lại W2 với 16 user, 3 lần, giống hệt P04:

| | TTFT p95 | TPOT p95 | Chữ ra / giây |
|---|---|---|---|
| Máy này | 5.0–5.7 s | 88–92 ms | 170–175 |
| Máy P04 | 4.9–6.3 s | 90–93 ms | 167–171 |

Máy này nhanh hơn ~2%, so sánh được với P04.

---

## 4. Tìm k (14:55–17:03)

```
k=16  ✓ đạt     TTFT p95 5.5 s,  TPOT p95 76 ms        (40 phút)
k=24  ✗ trượt   dừng sớm sau 8.7 phút đo                (19 phút)
k=20  ✗ trượt   TTFT p95 ≥ 31 s, TPOT p95 ≥ 191 ms, dừng sớm  (28 phút)
k=18  ✓ đạt     TTFT p95 9.1 s,  TPOT p95 82 ms        (41 phút)
→ goodput W4 = 18 (20 trượt)
```

Ở k = 18:
- **3,476 câu hỏi/giờ**;
- **~363 người có hội thoại đang mở**, trong đó phần lớn đang đọc hoặc nghỉ;
- trên server trung bình chỉ có **14 câu đang được trả lời cùng lúc** (cao nhất 32).

Dừng sớm tiết kiệm khoảng 55 phút GPU cho 3 lần trượt.

---

## 5. Kiểm tra (17:03–19:25)

| Lần chạy (k = 18) | Kết quả | TTFT p95 |
|---|---|---|
| Giờ 16h, **bộ ngày khác** (seed 2) | ✗ trượt, dừng sớm | — |
| Giờ 10h sáng | ✓ đạt | 0.6 s |
| Giờ 13h trưa | ✓ đạt | 0.6 s |

**Bộ ngày khác trượt ở 18**, nên tôi quyết định chạy thêm đúng một lần: seed 2 ở k = 16.
- Mục đích: biết bộ ngày thứ hai lệch 1 bước hay nhiều hơn.
- Kết quả: **đạt**, TTFT p95 0.87 s.
- Vậy goodput thật là **16–18 tuỳ bộ ngày**.

Giờ 10h và 13h ở k = 18 đạt rất dư. Server đủ cho 16h thì đủ cho các giờ khác.

---

## 6. Phát hiện chính

### 6.1 Hỏng theo đợt, do tài liệu dài dồn lại

Ở k = 18, chia 30 phút đo thành từng phút ([Hình 2](hinh/h2-theo-phut-k-lo.png)):
- TTFT p95 chỉ vượt 10 s ở **4 trong 30 phút**; các phút còn lại dưới 2 s.
- **Phút tệ nhất** (phút 16, TTFT p95 39 s) có **85,318 token prompt**, gấp 1.9 lần một phút bình thường, gồm 4 tài liệu dài ≥ 4k token.
- Số câu hỏi phút đó lại bình thường (65 câu, các phút khác 41–73).

Nghĩa là: **nhiều người dán tài liệu dài cùng một lúc** làm GPU phải đọc quá nhiều chữ một lần, hàng đợi dâng lên. Đây cũng là lý do bộ ngày
khác cho kết quả khác: chỉ cần vài tài liệu dài rơi vào cùng một phút là p95 của cả 30 phút bị kéo lên.

### 6.2 Câu ngắn bị câu dài làm chậm

| Câu hỏi ngắn (< 1k token), k = 18 | TTFT p95 |
|---|---|
| Đúng lúc có câu dài (≥ 4k) đang chờ chữ đầu | **32 s** |
| Lúc không có | 4.2 s |

Người hỏi một câu ngắn phải chờ lâu vì người khác vừa dán tài liệu. Bài đo W1/W2/W3 của P04 không thấy được điều này, vì mỗi bài chỉ
có một loại câu hỏi.

### 6.3 Cache lịch sử hội thoại gần như không giúp

vLLM có **prefix cache**: lưu kết quả tính của phần đầu prompt, lượt sau có cùng phần đầu thì khỏi tính lại. Với chat nhiều lượt, đây đáng lẽ
là chỗ tiết kiệm lớn. Đo được:
- Người dùng nghỉ **≥ 2 phút** (đa số) → **0%** lịch sử còn trong cache. Bộ nhớ cache (38k token) quá nhỏ so với ~280k token lịch sử của
  363 phiên đang mở.
- **Câu trả lời của lượt trước không bao giờ trúng cache.** Lý do: template chat của Qwen3 bỏ 4 token `<think></think>` rỗng khi đưa
  câu trả lời vào lịch sử. Mọi token sau đó lệch vị trí nên không khớp với cache.

### 6.4 Một L4 phục vụ bao nhiêu người

| | P04 W1 (gửi liên tục) | P05 W4 (người thật) |
|---|---|---|
| "Người dùng" | 68 | **~363 phiên đang mở** |
| Câu hỏi / giờ | ~6,600 | 3,476 |
| Chữ ra / giây | ~610 | 259 |

Người thật nghỉ nhiều, nên một GPU phục vụ được **gấp ~5 lần số người**. Nhưng số câu hỏi/giờ chỉ bằng một nửa: GPU phải để dư chỗ cho
những phút dồn tài liệu dài.

### 6.5 Chi phí

| | VND / 1 triệu token ra |
|---|---|
| Thuê GPU cả ngày, phục vụ k = 18 theo nhịp 24 giờ thật | **22,510** |
| Nếu cả ngày đông như 16h | 10,322 |

Đêm gần như không có ai mà GPU vẫn tính tiền, nên chi phí thật **gấp ~2.2 lần** con số tính theo giờ cao điểm.

---

## 7. Dòng thời gian và chi phí

| Giờ | Việc |
|---|---|
| 14:28 | Thuê máy; 14:32 máy chạy; 14:38 đẩy xong dữ liệu, bắt đầu |
| 14:49 | Server sẵn sàng (sau một lần bật lại tự động) |
| 14:49–14:55 | Điểm mốc W2/16 × 3 |
| 14:55–17:03 | Tìm k: 16 ✓, 24 ✗, 20 ✗, 18 ✓ |
| 17:03–18:44 | Kiểm tra: seed 2 ✗, 10h ✓, 13h ✓ (laptop tắt khoảng 17:05–17:35, máy thuê vẫn chạy) |
| 18:44–19:25 | Chạy thêm: seed 2 ở k = 16 ✓ |
| 19:26 | Kéo kết quả (kiểm checksum khớp), **destroy** |

**Chi phí:** tín dụng Vast giảm $8.339 → $6.461 = **$1.88 ≈ 49,200 VND** cho 4.97 giờ (ước tính trước: 33–41k VND; vượt vì giá máy
$0.372/h cao hơn P04 $0.283/h, và chạy thêm 1 lần).

Một lỗi của tôi: bộ canh trên laptop bỏ lỡ lúc giờ 10h chạy xong (18:03), nên tôi báo trễ 8 phút. Nguyên nhân là bộ canh mới đọc log
từ vị trí lúc nó khởi động. Đã sửa: lưu vị trí đã đọc vào file. Không ảnh hưởng số đo.

---

## 8. Rút ra điều gì

1. **Goodput W4 ≈ 16–18 ngày tải giờ cao điểm** (~3,200–3,500 câu/giờ, ~360 người đang chat) trên một L4 với vLLM mặc định.
2. Thứ làm hỏng là **đợt dồn tài liệu dài**, không phải tải trung bình. Ước tính của P04 dựa trên tải trung bình (k ≈ 22) cao hơn thực tế ~20%.
3. **Ba hướng tune rõ ràng:**
   - tăng chỗ KV (KV FP8): giữ lịch sử lâu hơn, ít hết bộ nhớ;
   - chỉnh cách vLLM chia prefill: câu dài không chặn câu ngắn;
   - giữ khối `<think>` trong lịch sử để dùng lại cache câu trả lời.
   Ngoài ra, đẩy KV sang RAM CPU (LMCache) là ứng viên "bổ trợ" cho GĐ2.
4. So sánh các cấu hình sau này bằng W4 phải dùng **cùng bộ ngày**, vì đổi bộ ngày đã đủ lệch 1 bước.

## Bước tiếp theo

- **P06**: đo chất lượng baseline + output tham chiếu (~1.5 h GPU).
- **P07**: tổng hợp P01–P06, lập plan tune GĐ1 theo các hướng ở mục 8.
