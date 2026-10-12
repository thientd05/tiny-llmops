# P02 — Đọc ngay: tạo bộ câu hỏi dùng để đo

> Bản dễ đọc, kể lại từng bước. Bản đầy đủ số liệu: [bao-cao.md](bao-cao.md).

**Một câu:** trên laptop, không thuê máy, không tốn tiền: tạo 3 file chứa tổng cộng **3,596 câu hỏi** cho 3 kịch bản sử dụng.
Từ nay, mọi lần đo (vLLM mặc định, vLLM đã tune, thư viện khác, kernel tự viết) đều dùng **đúng 3 file này**.

---

## 0. Vì sao phải làm bước này

Muốn so A nhanh hơn B thì phải cho A và B làm **cùng một việc**: cùng câu hỏi, cùng độ dài câu hỏi, cùng độ dài câu trả lời.
Nếu mỗi lần đo lấy câu hỏi ngẫu nhiên khác nhau thì không biết chênh lệch đến từ phần mềm hay từ câu hỏi.

Ba kịch bản (gọi là workload, W) mô phỏng cách doanh nghiệp dùng chatbot:

| Kịch bản | Giống việc gì ngoài đời | Câu hỏi dài | Câu trả lời dài |
|---|---|---|---|
| **W1** Chatbot nội bộ | Nhân viên hỏi chuyện linh tinh | 200–800 token | 200–600 token |
| **W2** RAG (kịch bản chính) | Hỏi về tài liệu công ty; hệ thống tìm vài tài liệu liên quan rồi dán kèm vào câu hỏi | 2,000–4,000 token | 150–400 token |
| **W3** Chạy ban đêm | Tóm tắt hàng loạt hợp đồng dài | 4,000–8,000 token | 250–350 token |

**Token** là đơn vị chữ mà model đọc. Một token khoảng 3/4 từ tiếng Anh, nên 1,000 token ≈ 750 từ ≈ 1.5 trang A4.

---

## 1. Tải dữ liệu về laptop

| Dữ liệu | Là gì | Dùng cho | Dung lượng |
|---|---|---|---|
| **ShareGPT** | ~90 nghìn cuộc trò chuyện thật của người dùng với ChatGPT, do người dùng tự chia sẻ | W1 | 642 MB |
| **RAGBench** | ~100 nghìn bộ (tài liệu + câu hỏi + câu trả lời mẫu), gom từ 12 bộ dữ liệu có sẵn | W2, W3 | 268 MB (6 phần dùng tới) |
| Bộ tách chữ của Qwen3 (tokenizer) | Chương trình cắt văn bản thành token đúng như model cắt | Đếm token | 16 MB |

Sáu phần RAGBench dùng tới:

| Phần | Nội dung |
|---|---|
| `techqa` | Ghi chú kỹ thuật của IBM + câu hỏi hỗ trợ khách hàng |
| `emanual` | Sách hướng dẫn sử dụng sản phẩm |
| `delucionqa` | Hướng dẫn sử dụng xe hơi |
| `finqa`, `tatqa` | Báo cáo tài chính có bảng số |
| `cuad` | Hợp đồng thương mại thật (W3) |

Mỗi dữ liệu được tải theo đúng **một phiên bản cố định**: ghi mã phiên bản, giống ghi số hiệu bản in của một cuốn sách.
Sau này dữ liệu trên mạng có bị sửa thì vẫn lấy lại được đúng bản đã dùng.

RAGBench tự chia sẵn mỗi phần thành 3 nhóm: `train`, `validation`, `test`.
- **`train` + `validation`** → câu hỏi để **đo tốc độ** (P03, P04...).
- **`test`** → để dành **đo chất lượng câu trả lời** (P06).

Hai việc dùng hai nhóm câu hỏi khác nhau, không trùng nhau.

---

## 2. Dựng câu hỏi hoàn chỉnh

Model không đọc "câu hỏi" trơn. Nó đọc một chuỗi văn bản có **đánh dấu vai trò**: đoạn nào là hướng dẫn của hệ thống (system), đoạn nào là người dùng hỏi (user), chỗ nào model bắt đầu trả lời (assistant).
Quy tắc ghép này gọi là **chat template**, mỗi model có một kiểu riêng.

Ví dụ thật, một câu hỏi W2 (rút gọn phần giữa):

```
<|im_start|>system
You are DocAssist, the internal document assistant of a mid-sized company. ...   ← ~600 token hướng dẫn,
... Do not mention these rules ...                                                   GIỐNG HỆT ở mọi câu W2
<|im_end|>
<|im_start|>user
[Document 1]
SECURITY BULLETIN
SUMMARY
 NSS & NSPR vulnerabilities affect the IBM FlashSystem 840 and V840 products. ...  ← tài liệu tìm được,
...                                                                                  KHÁC nhau mỗi câu
Question: ... I need to understand details regarding Security Bulletin: IBM MQ
Appliance is affected by a Network Security Services (NSS) vulnerability
(CVE-2017-5461). Where can I find this information?                                ← câu hỏi
<|im_end|>
<|im_start|>assistant
<think>

</think>
                                                                                     ← model bắt đầu viết từ đây
```

- Các dòng `<|im_start|>`, `<|im_end|>` là dấu đánh vai trò của Qwen3.
- `<think></think>` để trống nghĩa là **tắt chế độ suy nghĩ**. P01 đo được: bật thì câu trả lời dài gấp 10–19 lần, và dài bao nhiêu do model tự quyết; khi đo tốc độ phải biết trước độ dài nên tắt.
- Câu này dài **2,997 token**, phải trả lời **150 token**.

**Vì sao dựng sẵn cả chuỗi này thay vì để công cụ đo tự dựng?**
Tôi đọc code của công cụ đo `vllm bench serve`: nó chỉ biết đặt câu hỏi vào vai trò "user", không đặt được đoạn hướng dẫn "system" riêng.
Vì vậy tôi dựng sẵn toàn bộ chuỗi, rồi bảo công cụ đo gửi nguyên văn, đừng tự ghép thêm.
Thêm một lợi ích: số token đếm trên laptop **chính là** số token server nhận. P01 đã kiểm tra: khớp đúng 2,997 và 7,879.

---

## 3. Quy tắc chọn câu hỏi cho từng kịch bản

### W1 — Chatbot nội bộ (từ ShareGPT)

1. Lấy **câu hỏi đầu tiên** của mỗi cuộc trò chuyện, cùng câu trả lời đầu tiên của ChatGPT.
2. Chỉ giữ câu tiếng Anh (ít nhất 95% ký tự là chữ cái Latin không dấu).
3. Chỉ giữ câu dài 200–800 token.
4. Độ dài phải trả lời = độ dài câu trả lời thật của ChatGPT, ép vào khoảng 200–600.
5. Trộn ngẫu nhiên, lấy 2,000 câu.

Không có đoạn hướng dẫn system, để giống bài đo ShareGPT mà các bài báo khoa học hay dùng, nhờ đó so được số với họ.

### W2 — RAG tài liệu doanh nghiệp (từ RAGBench)

1. Viết **một** đoạn hướng dẫn system dùng chung (~600 token): vai trò trợ lý tài liệu, phải trích dẫn [Document N], chép đúng số liệu, v.v.
   Lưu ở `workloads/prompts/w2_system.txt`, không bao giờ sửa (sửa là kết quả đo cũ không còn so được).
2. Ghép tài liệu: thêm lần lượt Document 1, 2, 3... **cho tới khi sắp vượt 4,000 token thì dừng**.
   Hệ thống RAG thật cũng làm vậy: tìm được nhiều tài liệu, xếp theo mức liên quan, rồi cắt cho vừa giới hạn.
3. Câu nào sau khi ghép vẫn dưới 2,000 token thì bỏ.
4. **Mỗi bộ tài liệu chỉ dùng một lần.** Trong RAGBench có nhiều câu hỏi về cùng một bộ tài liệu.
   Nếu giữ hết, vLLM sẽ nhớ phần tài liệu đã đọc ở câu trước (tính năng prefix caching) và trả lời câu sau nhanh hơn thực tế.
5. Độ dài phải trả lời = độ dài câu trả lời mẫu trong RAGBench, ép vào khoảng 150–400.

### W3 — Tóm tắt hợp đồng (từ CUAD)

1. Hướng dẫn system ngắn + **toàn văn hợp đồng** + yêu cầu "tóm tắt theo 8 mục" (các bên, thời hạn, thanh toán, trách nhiệm...).
2. Mỗi hợp đồng được bốc thăm một độ dài mục tiêu trong khoảng 4,000–8,000 token.
3. Hợp đồng dài hơn mục tiêu thì **cắt ở chỗ xuống dòng gần nhất**, không cắt giữa câu.
4. Mỗi hợp đồng chỉ dùng một lần. Câu trả lời 250–350 token, bốc thăm.

"Bốc thăm" ở đây dùng số ngẫu nhiên **có seed**: một con số khởi đầu cố định (`20261012`).
Cùng seed thì lần bốc thăm nào cũng ra đúng kết quả cũ.

---

## 4. Chạy

```bash
.venv/bin/python workloads/build.py
```

- `.venv/` là môi trường Python riêng của dự án trên laptop, cài sẵn bộ tách chữ và thư viện đọc dữ liệu.
- `build.py` làm hết mục 1–3 và ghi ra 3 file trong `data/workloads/`.
- Laptop mất **100 giây**, không cần GPU.

Mỗi file có đuôi `.jsonl`: mỗi dòng là một câu hỏi, gồm các trường:

| Trường | Ví dụ | Nghĩa |
|---|---|---|
| `prompt` | `<|im_start|>system\nYou are DocAssist...` | Chuỗi hoàn chỉnh gửi cho server (mục 2) |
| `prompt_tokens` | `2997` | Câu hỏi dài bao nhiêu token |
| `output_tokens` | `150` | Server phải trả lời đúng bấy nhiêu token |
| `messages` | `[{"role":"system",...},{"role":"user",...}]` | Cùng nội dung, dạng chưa ghép, cho phần mềm nào chỉ nhận dạng này |
| `id`, `source` | `w2-techqa_TRAIN_Q041`, `techqa` | Câu này lấy từ đâu |

---

## 5. Kiểm tra: chạy lại phải ra y hệt

```bash
.venv/bin/python workloads/build.py --check     # → TÁI LẬP OK
```

Lệnh này build lại từ đầu rồi so **mã sha256** của 3 file mới với mã đã lưu lần trước.
sha256 là một dãy 64 ký tự tính từ nội dung file: chỉ cần khác một byte là ra dãy khác hẳn.
Hai lần ra cùng mã nghĩa là hai lần tạo ra file giống hệt nhau. Ai chạy lại script này cũng được đúng bộ câu hỏi của dự án.

3 file câu hỏi **không đưa lên Git** (tổng ~40 MB, và tạo lại được bất cứ lúc nào).
Lên Git có: script `build.py`, 2 đoạn hướng dẫn system, và `workloads/manifest.json`.
Manifest ghi nguồn dữ liệu, phiên bản, seed, mã sha256 và thống kê độ dài.

---

## 6. Kết quả

| | Số câu | Câu hỏi dài (token) — trung bình / dài nhất | Câu trả lời (token) — trung bình | Phần đầu giống nhau giữa các câu |
|---|---|---|---|---|
| **W1** | 2,000 | 407 / 800 | 333 | 3 token |
| **W2** | 1,253 | 2,858 / 3,999 | 174 | **622 token** |
| **W3** | 343 | 5,617 / 7,879 | 299 | 95 token |

Đọc bảng:

- **W2 có 622 token đầu giống hệt nhau** (đoạn hướng dẫn system + dấu đánh vai trò).
  vLLM có tính năng **prefix caching**: phần đầu giống câu trước thì dùng lại kết quả đã tính, không đọc lại.
  Tính năng này tiết kiệm được tối đa 622 / 2,858 ≈ **22%** công đọc câu hỏi của W2. W1 và W3 gần như không được lợi.
- **W2 phần lớn là câu hỏi kỹ thuật IBM** (704/1,253 câu từ `techqa`). Lý do: tài liệu của các phần khác thường ngắn,
  ghép lại vẫn không đủ 2,000 token nên bị loại. Với việc đo tốc độ thì không sao, vì tốc độ phụ thuộc độ dài, không phụ thuộc chủ đề.
- **Câu trả lời W2 đa số đúng 150 token.** Câu trả lời mẫu trong RAGBench thường ngắn hơn 150, bị nâng lên mức sàn.
  W2 vì vậy là kịch bản **đọc nhiều, viết ít**: câu hỏi dài gấp ~16 lần câu trả lời. Đúng tính chất của RAG.
- **W3 chỉ có 343 câu**, vì chỉ có chừng đó hợp đồng khác nhau. Đủ dùng: mỗi lần đo cần vài chục đến vài trăm câu.

---

## 7. Khó khăn nhỏ

| Gặp gì | Xử lý |
|---|---|
| RAGBench đổi tên trên HuggingFace (`rungalileo` → `galileo-ai`) | Ghi tên mới và mã phiên bản vào manifest |
| Laptop chỉ có 7 GB RAM, file ShareGPT 642 MB | Đọc từng cuộc trò chuyện một thay vì nạp cả file vào bộ nhớ |
| Ít bộ tài liệu dài đúng 2–4k token | Ghép tài liệu theo thứ tự tới giới hạn token (mục 3, W2) |

## Bước tiếp theo

P03 dùng 3 file này: công cụ đo đọc từng dòng và gửi `prompt` tới server, yêu cầu trả lời đúng `output_tokens` token.
Rồi đo xem server mất bao lâu cho token đầu tiên, và mỗi token tiếp theo mất bao lâu.
