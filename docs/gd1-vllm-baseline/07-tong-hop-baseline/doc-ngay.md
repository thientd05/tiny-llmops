# P07 — Đọc ngay: vLLM mặc định phục vụ được bao nhiêu, tốn bao nhiêu, và nên tune gì trước

> Bản dễ đọc, kể lại từng bước. Bản đầy đủ số liệu: [bao-cao.md](bao-cao.md).

**Một câu:** P07 không thuê GPU (0 đồng). Gom kết quả P01–P06 thành một bảng mốc, tính tiền theo giá thuê thật, so với kế hoạch, rồi kiểm tra lại
từng nghi vấn tune bằng dữ liệu đã có và mã nguồn vLLM. Kết quả chính:
- Một L4 với vLLM mặc định phục vụ **68 / 18 / 6 user** đồng thời cho chat / RAG / tóm tắt hợp đồng, và **16–18 "ngày tải" chồng lên nhau** ở giờ cao điểm
  của tải giống thật (W4).
- Chạy cả ngày cho W4 tốn khoảng **20,000–23,000 VND cho mỗi triệu token trả lời**.
- Cả giai đoạn baseline tốn **13.2 giờ GPU ≈ 116,600 VND**, ít hơn kế hoạch 30%. Ngân sách 2 tháng vẫn đủ.
- Nghi vấn số 1 để tune vẫn là **thiếu chỗ KV cache**. Phát hiện mới: ở giờ cao điểm, **gần một nửa token prompt là lịch sử hội thoại dùng lại được**
  nhưng chỉ 3% được dùng lại thật.

---

## 0. P07 để làm gì

Sáu plan trước đo từng mảnh: cài đặt (P01), bộ câu hỏi (P02), công cụ đo (P03), tốc độ với 3 loại tải thuần (P04), tải giống người dùng thật (P05),
chất lượng câu trả lời (P06). P07 làm ba việc:
1. Gom lại thành **một bảng mốc** để mọi bước sau (tune vLLM, thư viện khác, kernel tự viết) so vào.
2. **Tính tiền thật** và so với kế hoạch, để biết ngân sách còn đủ không.
3. **Lọc danh sách nghi vấn tune**: chỉ giữ cái có dấu hiệu đo được, và với mỗi cái tự hỏi "dấu hiệu này có thể do nguyên nhân khác không".

Theo yêu cầu, P07 **dừng ở đây**: chưa lập plan tune, vì có mấy quyết định về cách đo cần bạn chốt trước (mục 7).

---

## 1. Bảng mốc baseline

"Goodput" = số user đồng thời lớn nhất mà 95% câu hỏi vẫn có chữ đầu tiên trong ≤ 10 giây (TTFT p95) và chữ chạy không chậm hơn 150 ms/token (TPOT p95).
Ghi dạng "18 (20 trượt)": 18 đạt, 20 không đạt.

| Loại tải | Goodput | Mỗi lần đo lặp cho ra | Chỉ số hỏng trước |
|---|---|---|---|
| W1 chat (câu hỏi ~400 token, trả lời ~330) | **68** (72 trượt) | 68 / ≥ 72 / 68 | chờ chữ đầu |
| W2 RAG (tài liệu ~2,900 token, trả lời ~170) | **18** (20 trượt) | 18 / 20 / 18 | chờ chữ đầu |
| W3 tóm tắt hợp đồng (~5,600 token, trả lời ~300) | **6** (8 trượt) | 6 / 6 / 6 | chờ chữ đầu |
| W4 người dùng thật, giờ cao điểm 16h | **18 ngày tải** (seed 1), **16** (seed 2) | mỗi bộ ngày mới chạy **một lần** | chờ chữ đầu, theo từng đợt |

- "Mỗi lần đo lặp cho ra": P04 đo mỗi mức 3 lần với 3 bộ câu hỏi khác nhau. Nếu chỉ nhìn từng lần thì W2 có lần ra 20, W1 có lần ra ≥ 72.
  Con số chính thức lấy trung vị 3 lần, như quy tắc đã chốt trước khi đo.
- W4: "18 ngày tải" ≈ 3,480 câu hỏi/giờ, ≈ 363 người đang mở cửa sổ chat cùng lúc. Seed là số chọn bộ ngày; đổi bộ ngày thì ra 16.
- **Chỗ yếu của mốc W4:** ở seed 1, mức 18 đạt với TTFT p95 = 9.11 s, chỉ cách ngưỡng 10 s có 9%. Một lần đo W1–W3 thường lệch 16–23% ở các mức
  sát ngưỡng. Nên "18" có thể ra "16" nếu chạy lại. Chưa đo lặp lại lần nào (mục 5, rủi ro R1).

Chất lượng (P06, đo hai lần cùng cấu hình):

| Bài kiểm tra | Lần 1 / lần 2 | Trung bình | Số câu đổi đúng ↔ sai giữa hai lần |
|---|---|---|---|
| MMLU-Pro (kiến thức 14 môn) | 69.0 / 67.4 | 68.2 | 6.6% |
| IFEval (làm đúng định dạng yêu cầu) | 84.7 / 83.5 | 84.1 | 5.2% |
| GSM8K (toán lời văn) | 91.4 / 91.4 | 91.4 | 1.6% |
| FinQA-RAG (số liệu tài chính trên tài liệu) | 59.2 / 60.1 | 59.7 | 4.5% |

---

## 2. Tính tiền: dùng một giá chung

Bước đầu tôi gom tín dụng Vast bị trừ ở từng phiên (đã ghi trong measurements.md):

| Phiên | Máy | Giờ | Bị trừ | Ra VND/giờ |
|---|---|---|---|---|
| P01 | 116596 | 0.35 | $0.126 | 9,540 |
| P03 + P04 | 116596 | 5.25 | $1.516 | 7,560 |
| P05 | 150535 | 4.97 | $1.878 | 9,900 |
| P06 + lần 2 | 116596 | 2.67 | $0.912 | 8,710 / 9,400 |
| **Tổng** | | **13.24** | **$4.432** | **8,770** |

Giá mỗi phiên khác nhau tới 31%. Lý do: máy 150535 đắt hơn máy 116596, và mỗi máy mới tốn ~$0.027 tải model nên phiên ngắn đắt hơn mỗi giờ.

**Quyết định: dùng một giá chung 8,770 VND/giờ** (tổng tiền ÷ tổng giờ) cho mọi phép tính chi phí của đồ án.
Nếu mỗi cấu hình dùng giá của phiên đo nó, thì "cấu hình B rẻ hơn A 20%" có thể chỉ vì B được đo trên máy rẻ hơn.
Căn cứ: Hoefler & Belli 2015 (bài hướng dẫn báo cáo hiệu năng ở hội nghị SC), quy tắc 9 "ghi lại mọi yếu tố thay đổi"; đã lưu ở references G1.

Công thức: `VND / 1 triệu token = giá mỗi giờ ÷ số triệu token trả lời sinh ra mỗi giờ` tại mức goodput.

| Loại tải | VND / 1 triệu token trả lời | Khoảng theo 3 lần lặp | Nếu giá máy 7,560 … 9,900 VND/h |
|---|---|---|---|
| W1 chat, 68 user | **3,960** | 3,910–4,030 | 3,420 … 4,470 |
| W2 RAG, 18 user | **14,460** | 14,250–14,580 | 12,470 … 16,320 |
| W3 tóm tắt, 6 user | **24,030** | 23,950–24,130 | 20,720 … 27,130 |
| **W4 thuê cả ngày** | **20,240** (seed 1) – **22,770** (seed 2) | | 17,450 … 25,710 |

Đọc bảng thế nào:
- W1–W3 là **mức thấp nhất có thể**, vì công cụ đo giữ GPU bận suốt. Không doanh nghiệp nào có tải đều như vậy 24 giờ.
- **W4 cả ngày là con số thực tế**: GPU thuê theo giờ phải chạy cả đêm, mà đêm gần như không ai hỏi.
- P05 từng ghi 22,510 cho W4. Khác 20,240 chỉ vì P05 dùng giá niêm yết của máy 150535 ($0.372/h); tốc độ vẫn vậy.
- W2 đắt mỗi token trả lời vì phần lớn công sức là đọc tài liệu (~2,900 token vào cho ~170 token ra). Tính cả token vào thì W2 chỉ 830 VND / 1 triệu token.
- Chi phí gần như không đổi dù goodput là 18 hay 20: từ 16 user trở lên W2 đã ra đều ~168 token/giây, thêm user chỉ làm hàng đợi dài ra.

---

## 3. Kế hoạch ban đầu đoán đúng, sai ở đâu

| | Kế hoạch đoán (12/10) | Thực tế | |
|---|---|---|---|
| Tốc độ sinh chữ W1 @ 8 / 16 / 32 / 64 user (token/s) | 260 / 380 / 520 / 600 | 243 / 323 / 458 / 610 | khá sát (−15% … +2%) |
| Tốc độ đọc prompt | ~5,000 token/s | ~2,780 (W2), ~1,940 (W3) | **sai, thấp hơn 44–61%** |
| Giá thuê | 10,200 VND/h | 8,770 | rẻ hơn 14% |
| Giờ GPU cho P01–P06 | 16.25 h ≈ 166k VND | 13.24 h ≈ 116k VND | ít hơn 19% giờ, 30% tiền |
| Một lần "sweep đầy đủ" (W1–W3 ×3 + W4) | 11.5 h | 9.4 h | ít hơn |
| Một lần đo chất lượng | 1 h | 0.8 h một lần, 1.6 h nếu chạy hai lần cho chặt | nhiều hơn nếu chạy hai lần |

Vì sao đọc prompt chậm hơn đoán nhiều: L4 luôn chạm trần điện 72 W nên tự hạ xung nhịp xuống ~1,100 MHz (một nửa mức tối đa); mỗi bước đọc prompt
còn phải sinh chữ cho người khác cùng lúc; prompt 4–8k token làm phần attention tốn hơn. Muốn biết phần nào lớn nhất thì phải profile (GĐ3).

Đã sửa [plan.md](../../plan.md): giữ nguyên các số đoán cũ, thêm cột "đo được / ước lại 14/10".

**Ngân sách:** đã tiêu ~116,600 VND. Ước lại cả dự án với giá thật:
- Kịch bản A (mọi thư viện khác thua vLLM): **~2.55–2.85 triệu**, vừa mốc 2 tháng (≤ 3 triệu), dư 150–450k tuỳ cách đo phần tune.
- Kịch bản B (có một thư viện ngang vLLM): ~3.24–3.53 triệu, vừa mốc 3 tháng.
- **Hiện chưa có gì phải cảnh báo.** Rủi ro: nếu chỉ thuê được máy giá như P05 (9,900 VND/h) mà phần tune đo kiểu chặt, A lên ~3.2 triệu, vượt mốc 2 tháng ~200k.
  Vast chỉ có ~5 máy L4, phần lớn cùng một chủ.

---

## 4. Kiểm tra lại từng nghi vấn tune

Mọi phân tích ở bước này chạy trên laptop, đọc kết quả thô đã kéo về:

```bash
.venv/bin/python bench/p07_analyze.py --price 8770 --price-lo 7560 --price-hi 9900
```

| Phần | Nghĩa |
|---|---|
| `bench/p07_analyze.py` | Script mới của P07, in mọi bảng của mục này (chạy ~1 phút) |
| `--price 8770` | Giá chung VND/giờ để tính chi phí |
| `--price-lo / --price-hi` | Giá máy rẻ nhất / đắt nhất đã thuê, để tính khoảng |

Ngoài ra tôi đọc **mã nguồn vLLM đúng bản 0.30.0** đang dùng (tải từ GitHub, lưu ở `docs/references/files/`) để kiểm cơ chế và tìm cờ có sẵn.

### 4.1 Nghi vấn 1 — thiếu chỗ KV cache (ảnh hưởng mọi loại tải)

KV cache là vùng bộ nhớ GPU giữ kết quả trung gian của các token đã đọc. L4 chỉ còn 3.4 GB cho nó (37,264 token) sau khi chứa model.

**Dấu hiệu:**
- Số câu hỏi chạy cùng lúc dừng ở ~60 (chat), ~15 (RAG), ~6 (hợp đồng). Đúng bằng 37,264 chia cho kích thước một câu hỏi.
- Ở mức goodput, 73–83% thời gian chờ chữ đầu là **nằm trong hàng đợi**, không phải đang tính.
- Bằng chứng mạnh nhất từ P03: cùng máy, một lần server khởi động lỗi chỉ có KV ít hơn 23%. Lần đó RAG 16 user có TTFT p95 9.9 s, so với 4.6–5.6 s
  khi KV đủ. Tức **ít KV hơn 23% → chờ chữ đầu gần gấp đôi**.

**Ước lợi ích (chưa đo, chỉ để xếp hạng):** thêm KV thì nhiều câu hỏi chạy song song hơn, mỗi câu sinh chữ chậm đi một chút (vẫn còn dư so với 150 ms),
nhưng chờ chữ đầu giảm mạnh. Nếu tốc độ tổng giữ nguyên, goodput có thể lên tới ~88 (chat), ~26 (RAG), ~15 (hợp đồng). KV FP8 (lưu KV bằng số 8 bit
thay vì 16 bit, gấp đôi chỗ) đủ cho chat và RAG.

**Tự hỏi ngược:**
- Có phải giới hạn ~60 câu hỏi do cờ `max_num_seqs`? Không. Đọc mã vLLM 0.30.0: trên GPU < 70 GB, mặc định cho phép 256 câu hỏi cùng lúc.
- Bật KV FP8 thì vLLM phải đổi bộ kernel attention (FlashAttention 2 không hỗ trợ FP8 trên L4). **Đổi hai thứ một lúc** → phải đo thêm cấu hình
  "đổi kernel nhưng giữ KV 16 bit" để tách hai hiệu ứng.
- KV FP8 làm tròn số → phải qua cổng chất lượng của P06.

**Cờ thử:** `--kv-cache-dtype fp8`, `--attention-backend FLASHINFER` (đối chứng), `--gpu-memory-utilization 0.94–0.96` (mỗi 0.01 thêm ~2,400 token).

### 4.2 Nghi vấn 2 — lịch sử hội thoại bị tính lại mỗi lượt (chỉ W4)

Khi người dùng hỏi lượt 2, prompt gồm cả lượt 1 + câu trả lời lượt 1. Nếu KV của phần đó còn trong bộ nhớ, vLLM dùng lại, không phải đọc lại.

**Phân tích mới hôm nay:** với từng câu hỏi của W4, tôi tính phần prompt nào là lịch sử của lượt trước (lượt trước đã chạy trong cùng lần đo,
không bị cắt bớt), tức về nguyên tắc dùng lại được:

| Lần chạy W4 | Dùng lại thật | Dùng lại được |
|---|---|---|
| giờ cao điểm, 18 ngày | 3.0% | **44.7%** |
| giờ cao điểm, 16 ngày (2 bộ ngày) | 3.7–4.4% | 44–46% |
| giờ vắng 13h | 14.5% | 42.1% |

Tức là ở giờ cao điểm, **gần một nửa công đọc prompt là đọc lại thứ đã đọc**. Lý do: ~363 hội thoại đang mở có tổng lịch sử ~280k token, gấp 7 lần KV.
Người dùng nghỉ trung vị 2 phút giữa hai lượt, đủ lâu để lịch sử bị đẩy ra.

**Có làm được không:** 280k token KV ≈ 26 GB, vừa RAM của máy thuê (94–125 GB). Chép 1,000 token KV qua PCIe (đo được 8.4 GB/s) mất ~11 ms,
so với ~370 ms để đọc lại. vLLM 0.30.0 **có sẵn** cờ làm việc này: `--kv-offloading-size <GB>` (chép KV ra RAM CPU khi bị đẩy khỏi GPU).
Paper CachedAttention (ATC'24) dùng cách tương tự cho hội thoại nhiều lượt, báo giảm thời gian chờ chữ đầu tới 87%.

**Tự hỏi ngược:** các phút làm hỏng SLO chủ yếu do **tài liệu mới** dán vào, không phải lịch sử. Phút tệ nhất (phút 16) có 85k token prompt nhưng chỉ
38% là lịch sử. Dù dùng lại hết, phút đó vẫn còn ~53k token mới, gấp 2.5 lần phút bình thường. Nên cách này **giảm tải trung bình gần một nửa**,
nhưng **không xoá được đợt dồn**. Goodput W4 tăng bao nhiêu thì phải đo.

### 4.3 Nghi vấn 3 — câu hỏi dài chặn câu hỏi ngắn ở hàng đợi (W4)

P05 thấy: câu hỏi ngắn gửi đúng lúc có câu hỏi ≥ 4k token đang chờ thì TTFT p95 26–32 s; không có thì 1.4–4.2 s.

**Tìm thấy cơ chế trong mã vLLM 0.30.0:**
1. Hàng đợi xét theo thứ tự đến.
2. Câu hỏi mới chỉ được nhận khi **cả prompt** vừa chỗ KV trống (cờ `scheduler_reserve_full_isl`, mặc định bật).
3. Câu đứng đầu hàng không vừa thì vòng xét **dừng luôn**. Các câu ngắn phía sau, dù vừa chỗ, cũng phải chờ.

**Tự hỏi ngược: có phải chỉ vì lúc đó đông?** Câu dài hay rơi vào lúc hàng đợi đã dài, nên tương quan trên có thể chỉ do đông. Tôi chia câu ngắn theo số câu
đang chờ trên server lúc gửi:

| Giờ cao điểm, 18 ngày | Không có câu dài đang chờ | Có |
|---|---|---|
| Hàng đợi trống | trung vị 0.21 s (1,081 câu) | 1.10 s (34 câu) |
| Hàng đợi ≥ 10 câu | trung vị 9.8 s (42 câu) | 22.9 s (31 câu) |

Cùng độ đông, câu ngắn vẫn chậm gấp ~2 khi có câu dài. Vậy **không chỉ do đông**. Nhưng số câu nhóm "có" ít (16–34), nên vẫn cần một thí nghiệm có kiểm soát.

**Cờ thử (đều có sẵn):** tắt `scheduler_reserve_full_isl`; đổi `--max-num-batched-tokens` (số token tối đa mỗi bước, mặc định 2,048);
`--long-prefill-token-threshold`; `--scheduling-policy priority` (client gán độ ưu tiên theo độ dài). Nếu không cờ nào đủ → ứng viên code riêng GĐ4 (mục 6).

### 4.4 Nghi vấn 4 — vLLM tạm dừng câu hỏi khi hết KV (W1)

Ở chat 64–72 user, mỗi lần đo có 83–151 lần vLLM tạm dừng câu đang chạy để lấy chỗ, rồi tính lại sau (preemption).

**Tự hỏi ngược: tốn bao nhiêu?** Tôi thử đếm token phải tính lại từ bộ đếm của server, thì thấy bộ đếm khớp đúng từng token với "prompt − phần trúng cache + token sinh ra".
Nghĩa là **vLLM không đếm phần tính lại**. Chi phí thật chưa đo được: có thể gần 0 (nếu KV vừa bỏ vẫn còn trong cache), có thể tới ~18% thời gian GPU.
Khoảng quá rộng → xếp thấp.

**Cờ thử:** `--watermark 0.01–0.05` (để dành một phần KV khi nhận câu mới), `--max-num-seqs` 56–64.

### 4.5 Nghi vấn 5 — câu trả lời lượt trước không bao giờ được dùng lại (W4)

Template chat của Qwen3 bỏ khối `<think></think>` rỗng của lượt trước khi ghép lịch sử, làm mọi token sau đó lệch vị trí. Phần này là 13.4% token prompt ở giờ cao điểm.
Sửa template để giữ khối rỗng thì dùng lại được, **nhưng** Qwen khuyến nghị lịch sử chỉ chứa câu trả lời cuối (model card), và bộ đo chất lượng P06
không có bài hội thoại nhiều lượt. Chỉ có lợi khi đi cùng nghi vấn 2. Xếp cuối.

### 4.6 Loại khỏi danh sách

| Nghi vấn trong plan cũ | Vì sao loại |
|---|---|
| Tune config MoE cho L4 | vLLM chọn kernel MoE Marlin, kernel này không đọc file config đó. Không có số đo nào cho thấy MoE là chỗ nghẽn → để GĐ3 profile |
| Speculative decoding | Tốc độ sinh chữ chưa bao giờ là chỉ số hỏng (TPOT p95 cao nhất 117 ms < 150). Mọi lần hỏng đều do chờ chữ đầu |
| Cờ fusion | Không có dấu hiệu; lợi ích dự kiến vài %, nhỏ hơn độ phân giải của goodput |

---

## 5. Chỗ nào trong cách đo có thể làm so sánh khi tune bị sai

| # | Rủi ro | Cách chặt hơn | Tốn thêm |
|---|---|---|---|
| R1 | Mốc W4 mỗi bộ ngày mới chạy một lần, lại sát ngưỡng (9.11 s) | Chạy lại seed 1 k = 18 và seed 2 k = 16 **trước khi** so cấu hình tune trên W4 | ~1.4 h ≈ 12k VND |
| R2 | Một lần đo W1–W3 lệch 16–23% ở mức sát ngưỡng → không phân biệt được chênh 1 bước (2 user) | Lặp ×3 ở lo / hi; chỉ kết luận khi chênh ≥ 2 bước | ~2.2 h thay vì ~0.8 h mỗi cấu hình |
| R3 | Thử ~10 cấu hình, mỗi cái một lần, rồi chọn cái tốt nhất → cái được chọn thường "may" hơn thực | Đo lại cấu hình thắng trên bộ câu hỏi mới (seed 1004–1006) và W4 seed 2 (Cawley & Talbot 2010) | ~2–4 h mỗi cấu hình vào chung kết |
| R4 | So hai cấu hình trên hai bộ câu hỏi khác nhau | Luôn dùng đúng seed 1001–1003 như baseline, so từng cặp. Cùng seed, P05 so P04 (khác máy) chỉ lệch −9…+2% TTFT, trong khi khác seed lệch 24% | 0 |
| R5 | Khác máy → KV khác 2%; lần khởi động đầu KV ít hơn 23% | Giữ điểm mốc đầu phiên (đã có); khác máy P04 thì đo lại baseline cùng phiên | 6 phút … 2.2 h |
| R6 | KV FP8 kéo theo đổi kernel attention | Thêm cấu hình đối chứng chỉ đổi kernel | ~0.8 h |
| R7 | Cổng chất lượng chạy một lần cho "khác biệt" giả (baseline với chính nó đã ra MMLU-Pro −1.6, có ý nghĩa thống kê) | So với khoảng giữa hai lần chạy baseline; "kém đi" chỉ khi lặp lại được | ~0.8 h mỗi lần thêm |
| R8 | Cờ có lợi cho W1–W3 có thể hại W4 | Kết luận cuối về một cờ phải có W4 | xem câu hỏi 2 |
| R9 | Khoảng tin cậy tính theo từng câu hỏi quá hẹp, vì câu hỏi chậm theo cụm | Dùng số lần lặp làm đơn vị, báo min–max | 0 |

Một kiểm tra phụ (0 GPU): nếu gộp câu hỏi của 3 lần lặp rồi mới tính p95, baseline ra **cùng goodput** 68 / 18 / 6. Muốn đổi sang quy tắc này cho phần tune
thì phải chốt **trước** khi đo cấu hình nào, và áp cho cả baseline.

---

## 6. Ứng viên code riêng cho GĐ4 (chỉ ghi lại, chưa làm)

| Ứng viên | Giải quyết | Cắm vào vLLM | Cần tìm tài liệu gì trước |
|---|---|---|---|
| Bộ xếp lịch biết độ dài, không để câu dài chặn đầu hàng, có chống "đói" cho câu dài | Nghi vấn 3 | vLLM 0.30.0 có `--scheduler-cls`: thay lớp xếp lịch mà không sửa mã vLLM | FastServe; "Efficient LLM Scheduling by Learning to Rank" (NeurIPS'24); VTC (công bằng giữa người dùng, OSDI'24); Andes; lập lịch ưu tiên có aging |
| Giữ KV lịch sử khi client cắt bớt lịch sử | 337 / 1,266 lượt sau ở giờ cao điểm bị cắt → prefix đổi → không dùng lại được | connector KV | CachedAttention (đã lưu, references F5) phần tách mã hoá vị trí; nghiên cứu chất lượng khi ghép KV lệch vị trí |
| Nhận câu hỏi theo đợt dồn token | SLO hỏng theo phút có token prompt gấp đôi | `--scheduler-cls` | Admission control theo SLO cho phục vụ LLM |

Nếu cờ có sẵn (mục 4.3) giải quyết được nghi vấn 3 thì ứng viên đầu mất lý do. Theo quy tắc dự án, phải có tài liệu minh chứng trong docs/references trước khi làm.

---

## 7. Cần bạn quyết trước khi lập plan tune

**Ước giờ GPU một cấu hình theo cách đo** (giá 8,770 VND/h):

| Cách đo | Giờ | VND | Phát hiện được |
|---|---|---|---|
| Chỉ W2 rút gọn ×1 | ~0.25 | ~2k | chênh ≥ 2 bước ở RAG |
| W1–W3 rút gọn ×1 | ~0.8 | ~7k | chênh ≥ 2 bước |
| W1–W3 rút gọn ×3 | ~2.2 | ~19k | chênh 1–2 bước, so cặp cùng seed |
| W4 một bộ ngày | ~1.5–2.2 | ~13–19k | tải thật, một seed |
| W4 hai bộ ngày | ~3–4.5 | ~26–39k | tải thật, bỏ được may rủi của bộ ngày |
| Cổng chất lượng ×1 / ×2 | ~0.8 / ~1.6 | ~7k / ~14k | chỉ cần cho cấu hình đổi số học |

**Hai phương án cho cả phần tune** (~13 cấu hình theo 5 nghi vấn + cấu hình gộp cuối):

| | Kiểu rẻ (~31 h ≈ 270k VND) | Kiểu chặt (~52 h ≈ 456k VND) |
|---|---|---|
| Đo lại mốc W4 trước (R1) | không | có (1.4 h) |
| Cấu hình W1–W3 (KV, ngân sách token, preemption: ~7) | rút gọn ×1 (5.6 h) | rút gọn ×3 (15.4 h) |
| Cấu hình W4 (~6) | một bộ ngày (11.4 h) | 3 cái chung kết hai bộ ngày, còn lại một (17.1 h) |
| Cổng chất lượng (KV FP8, template) | ×1 (1.6 h) | ×2 (3.2 h) |
| Đo lại người thắng trên seed mới (R3) | không | có (2.2 h) |
| Cấu hình cuối: sweep đầy đủ + chất lượng ×2 | 11 h | 11 h |
| Setup các phiên | 1.2 h | 1.6 h |
| Ngân sách cả dự án (kịch bản A) | ~2.55 triệu | ~2.85 triệu |

**Các câu hỏi:**
1. **Thứ tự tune.** Đề xuất: (1) KV → (3) câu dài chặn hàng đợi bằng cờ có sẵn → (2) giữ lịch sử ở RAM CPU → (4) preemption → (5) template.
   Lý do: (1) có bằng chứng mạnh nhất và ảnh hưởng mọi loại tải; (3) và (2) chỉ thấy trên W4 nên làm sau khi mốc W4 được đo lại.
   Hay bạn muốn ưu tiên W4 (bài test cuối) ngay từ đầu?
2. **Đo mỗi cấu hình bằng gì.** Đề xuất: cờ về KV và preemption đo bằng W1–W3 rút gọn; cờ về hàng đợi và lịch sử đo bằng W4; cấu hình cuối đo đủ cả hai.
   Hay mọi cấu hình đều phải có W4?
3. **Lặp bao nhiêu lần.** Kiểu rẻ (×1, chỉ tin chênh ≥ 2 bước) hay kiểu chặt (×3 + đo lại người thắng trên seed mới)?
   Có thể trộn: sàng lọc ×1, chung kết ×3.
4. **Cổng chất lượng.** Chỉ cho cấu hình đổi số học (KV FP8, template), chạy ×2? Nghi vấn 5 cần bài chất lượng hội thoại nhiều lượt mà P06 chưa có:
   bỏ nghi vấn 5, hay thêm bài đo (tốn thêm công và giờ GPU)?
5. **Có đo lại mốc W4 trước không** (R1, ~12k VND)? Tôi đề xuất có: không có số này thì mọi so sánh trên W4 đều không biết chênh bao nhiêu mới là thật.
6. **Quy tắc đạt/trượt:** giữ "trung vị 3 lần" hay đổi sang "p95 gộp"? Phải chốt trước khi đo.
7. **Thay đổi phía ứng dụng** (client gán độ ưu tiên theo độ dài, sửa template) có tính vào "vLLM đã tune" không? Đề xuất: báo thành nhóm riêng
   "vLLM đã tune + chỉnh ứng dụng", để so với thư viện khác vẫn công bằng.
8. **Giá chung 8,770 VND/h** cho mọi phép tính chi phí từ nay: đồng ý không?
9. **Ngân sách cho phần tune:** ~270k (rẻ) hay ~456k (chặt)? Cả hai đều vừa mốc 2 tháng.

---

## 8. Đã chốt sau khi thảo luận (14/10)

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

---

## Dòng thời gian

| Lúc (14/10, +07) | Việc |
|---|---|
| ~00:40 | Đọc báo cáo P01–P06, measurements, kết quả thô |
| ~00:50 | Tính lại khoảng min–max, chi phí, phân tích lịch sử W4, hàng đợi W4, preemption W1, nhiễu |
| ~01:00 | Đọc mã vLLM 0.30.0, tra và lưu 8 mục tài liệu mới (12 file; references nhóm F, G) |
| ~01:05 → 01:20 | Viết báo cáo, sửa plan.md, measurements.md, README, thuật ngữ, CLAUDE.md |

Không thuê máy. `vastai show instances` không cần chạy (không tạo máy nào trong P07).

## Chi phí

**0 VND.** Tổng cả GĐ1 baseline: 13.24 giờ GPU, tín dụng bị trừ $4.451 ≈ 116,600 VND.

## Rút ra điều gì

- vLLM mặc định trên L4 bị giới hạn bởi **chỗ chứa KV**, không phải tốc độ tính, ở cả bốn loại tải.
- Tải giống thật (W4) hỏng theo **đợt dồn tài liệu dài**, và có hai chỗ lãng phí lớn: **đọc lại lịch sử hội thoại** (~45% token prompt) và
  **câu dài chặn câu ngắn ở hàng đợi**. Cả hai đều có cờ sẵn trong vLLM để thử trước khi nghĩ tới code riêng.
- Chi phí thật cho doanh nghiệp tự chạy chatbot theo giờ hành chính: **~20,000–23,000 VND / 1 triệu token trả lời** với vLLM mặc định.
- Muốn so sánh khi tune cho chặt, chỗ yếu nhất hiện tại là **mốc W4 mới đo một lần mỗi bộ ngày**.

## Bước tiếp theo

1. ~~Bạn trả lời các câu hỏi ở mục 7~~ → đã chốt ở mục 8.
2. Lập plan tune GĐ1 (thư mục mới), kèm tài liệu minh chứng cho mọi quyết định phương pháp.
3. Commit P07 khi bạn duyệt (chưa commit; `docs/library-gaps.md` của bạn không bị đụng tới).
