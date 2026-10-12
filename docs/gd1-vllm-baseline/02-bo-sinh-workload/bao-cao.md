# Báo cáo P02 — Bộ sinh workload W1/W2/W3

> Ngày làm: 2026-10-12 · Máy: laptop local (không GPU) · Giờ GPU: 0
> Code: `workloads/build.py`, `workloads/prompts/`. Manifest (nguồn, seed, sha256, thống kê): `workloads/manifest.json`.
> Số liệu: [measurements.md](../../measurements.md) mục "P02".

## 1. Tóm tắt

- Đã tạo 3 file request cố định: W1 có 2000 mẫu, W2 có 1253 mẫu, W3 có 343 mẫu. Độ dài đều nằm trong khoảng thiết kế.
- Build lại lần thứ hai cho **cùng sha256**, tức là ai chạy lại script cũng được đúng bộ dữ liệu này.
- Mỗi prompt đã được **dựng sẵn theo chat template của Qwen3** (có system prompt, thinking tắt). Số token server nhận khớp đúng số đếm ở đây (đã kiểm ở P01: 2997 và 7879 token).
- Đạt tiêu chí xong. Riêng W3 ít hơn 600 mẫu do giới hạn số hợp đồng CUAD (plan đã lường trước).

## 2. Đã làm gì và vì sao

**Định dạng file.** Tôi đọc mã nguồn `vllm bench serve` bản 0.30.0. Định dạng `custom` của nó đọc file JSONL có trường `prompt` và `output_tokens`.
Tuy nhiên nó tự áp chat template **chỉ với vai trò user**, nên không đặt được system prompt riêng.
Vì vậy script dựng sẵn toàn bộ prompt (system + user + phần mở đầu trả lời của model) thành văn bản.
Khi đo sẽ gửi vào endpoint `/v1/completions` và bảo công cụ đo không áp template nữa (`--skip-chat-template`).
Mỗi dòng còn lưu bản dạng hội thoại (`messages`), dành cho thư viện nào chỉ có endpoint chat.

**Chia dữ liệu.** RAGBench: phần train + validation dùng cho benchmark. Phần **test để dành cho P06** (đo chất lượng), không trùng với benchmark.
Riêng CUAD lấy cả 3 phần, vì P06 không dùng CUAD và cần thêm hợp đồng.

**Mỗi mẫu dùng một bộ tài liệu riêng.** Trong RAGBench, nhiều câu hỏi dùng chung một bộ tài liệu (ví dụ FinQA có 14k câu hỏi nhưng chỉ 2.4k bộ tài liệu khác nhau).
Nếu giữ nguyên, prefix caching sẽ dùng lại KV cache của cả bộ tài liệu giữa các request, và tốc độ đo được sẽ cao hơn thực tế.
Script chỉ giữ một câu hỏi cho mỗi bộ tài liệu.

**W1 — Chatbot nội bộ.** Lấy lượt hỏi đầu tiên của các hội thoại ShareGPT.
- Lọc: phần lớn là tiếng Anh (≥ 95% ký tự ASCII), prompt 200–800 token.
- Độ dài output = độ dài câu trả lời thật trong hội thoại, giới hạn trong 200–600.
- Không có system prompt, giữ giống bài benchmark ShareGPT chuẩn để so được với các bài báo.

**W2 — RAG tài liệu doanh nghiệp.**
- Tự viết một system prompt chung cho "trợ lý tài liệu nội bộ": quy tắc trích dẫn [Document N], giữ đúng số liệu, xử lý tài liệu mâu thuẫn, v.v.
- Prompt = system prompt + các tài liệu đánh số + câu hỏi.
- Tài liệu được thêm **theo thứ tự** đến khi chạm 4000 token thì dừng. Hệ RAG thật cũng làm vậy: lấy các đoạn tìm được theo thứ tự liên quan, cắt theo ngân sách token.
- Mẫu nào dưới 2000 token thì bỏ.
- Độ dài output = độ dài câu trả lời mẫu trong RAGBench, giới hạn trong 150–400.

**W3 — Batch ban đêm (tóm tắt hợp đồng).**
- System prompt ngắn + toàn văn hợp đồng + yêu cầu tóm tắt theo 8 mục.
- Mỗi hợp đồng nhận một độ dài mục tiêu ngẫu nhiên trong 4000–8000 token (seed cố định).
- Hợp đồng dài hơn mục tiêu bị cắt tại **ranh giới đoạn văn** gần nhất, để không cắt giữa câu.
- Output 250–350 token, chọn ngẫu nhiên có seed.

**Thinking tắt** cho cả 3 workload (P01 đo được: bật thì output dài hơn 10–19 lần, độ dài do model tự quyết).

## 3. Kết quả

| Workload | Số mẫu | Prompt (token) TB / p50 / p95 / max | Output (token) TB / p50 / p95 | Phần đầu dùng chung |
|---|---|---|---|---|
| W1 | 2000 | 407 / 368 / 726 / 800 | 333 / 272 / 600 | 3 token |
| W2 | 1253 | 2858 / 2785 / 3887 / 3999 | 174 / 150 / 287 | **622 token** |
| W3 | 343 | 5617 / 5577 / 7483 / 7879 | 299 / 301 / 344 | 95 token |

Nguồn W2: techqa 704, finqa 326, delucionqa 127, tatqa 80, emanual 16.
W3: 343 hợp đồng khác nhau, 286 hợp đồng bị cắt.

License: RAGBench CC-BY-4.0 (các dataset thành phần có license riêng). ShareGPT ghi apache-2.0 nhưng nội dung do người dùng ChatGPT tạo ra.
Cả hai **chỉ dùng cho nghiên cứu**.

## 4. Giải thích kết quả

**W2 nghiêng về techqa (56%).** Bộ tài liệu của emanual, delucionqa, finqa và tatqa phần lớn ngắn hơn 1.4k token.
Sau khi cộng system prompt vẫn không đủ 2000 token, nên phần lớn bị loại (ví dụ tatqa: chỉ giữ 80, loại 2399).
techqa có tài liệu dài (trung vị 4.9k token) nên đóng góp nhiều nhất.
Hệ quả: W2 chủ yếu là câu hỏi hỗ trợ kỹ thuật (IBM technote). Với mục tiêu đo tốc độ thì không sao, vì tốc độ phụ thuộc vào độ dài, không phụ thuộc chủ đề.

**Output W2 đa số đúng 150 token.** Câu trả lời mẫu của RAGBench thường ngắn (trung vị ~60–136 token), nên bị nâng lên mức sàn 150 theo thiết kế.
Trung bình output W2 vì vậy chỉ 174 token. W2 là workload **nặng phần prefill** (prompt dài ~16 lần output), đúng tính chất của RAG.

**Phần dùng chung của W2 là 622 token**, không phải ~500 như thiết kế, vì system prompt tôi viết dài hơn dự kiến một chút.
Prefix caching (dùng lại KV cache của phần đầu giống nhau) bỏ qua được tối đa 622 / 2858 ≈ **22%** khối lượng prefill của W2.
Với W1 (3 token) và W3 (95 token) thì gần như không có tác dụng.

**Đếm token tính cả chat template.** Số token ở đây là đúng số token server nhận, gồm cả các token đánh dấu vai trò của Qwen3 và khối thinking rỗng.
P01 đã xác nhận server báo đúng con số này.

## 5. Vấn đề gặp phải

| Vấn đề | Cách xử lý |
|---|---|
| RAGBench đã đổi tên trên HuggingFace (`rungalileo` → `galileo-ai`) | Ghi tên mới + commit hash trong manifest |
| Laptop chỉ có 7 GB RAM, file ShareGPT 642 MB | Đọc từng hội thoại một (thư viện `ijson`) thay vì nạp cả file |
| Ít bộ tài liệu RAG có độ dài 2–4k | Thêm tài liệu theo thứ tự tới ngân sách token (mục 2) |

## 6. Ảnh hưởng tới bước sau

- **P03:**
  - Dùng `--dataset-name custom --dataset-path data/workloads/wX.jsonl --backend openai --endpoint /v1/completions --skip-chat-template --ignore-eos`.
  - Thêm `--disable-shuffle`, hoặc ghi lại seed, vì công cụ tự xáo trộn thứ tự mẫu.
- **Máy thuê không có sẵn file workload.** Cần copy `data/workloads/*.jsonl` (~40 MB) lên, hoặc build lại trên máy thuê (~2 phút CPU, không cần GPU).
- File JSONL không commit vào git. Muốn có lại: chạy `workloads/build.py`, rồi kiểm bằng `--check`.
- `docs/workloads.md` đã cập nhật: quyết định tắt thinking, độ dài phần dùng chung thực tế.

## 7. Thuật ngữ dùng trong báo cáo này

| Từ | Nghĩa |
|---|---|
| Chat template | Quy tắc model dùng để ghép các vai trò (system, user, assistant) thành một chuỗi văn bản, có token đánh dấu riêng |
| JSONL | File văn bản, mỗi dòng là một bản ghi JSON |
| sha256 | Mã băm của file: hai file cho cùng sha256 thì giống hệt nhau từng byte |
| Manifest | File ghi nguồn dữ liệu, phiên bản, seed và sha256 để tái lập |
| Seed | Số khởi tạo cho bộ sinh ngẫu nhiên; cùng seed thì ra cùng kết quả |
| Prefix caching | Dùng lại KV cache của phần đầu prompt giống nhau giữa các request |
| Prefill | Bước model đọc toàn bộ prompt trước khi sinh token đầu tiên |
