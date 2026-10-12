# P02 — Bộ sinh workload W1/W2/W3

> Cần GPU: không (chỉ cần tokenizer, chạy trên máy local) · Phụ thuộc: làm song song được với P01
> Thiết kế workload gốc: [workloads.md](../../workloads.md)

## Mục tiêu

Tạo bộ request **cố định** cho W1/W2/W3, lưu thành file. Mọi lần đo về sau (vLLM mặc định, vLLM đã tune, thư viện khác,
custom CUDA) dùng **đúng bộ file này** → so sánh công bằng, chạy lại ra cùng đầu vào.

## Quyết định cần chốt (đề xuất sẵn)

| Câu hỏi | Đề xuất | Lý do |
|---|---|---|
| Thinking mode của Qwen3 | **Tắt** cho cả 3 workload | Chatbot/RAG doanh nghiệp ưu tiên độ trễ; nếu bật, độ dài output phụ thuộc model → khó kiểm soát tải. Xác nhận lại theo kết quả P01 |
| Độ dài output | Cố định theo từng mẫu (lấy từ độ dài câu trả lời thật trong dataset, kẹp vào khoảng của workload), server được yêu cầu sinh đúng số token đó (`ignore_eos`) | Mọi hệ thống phải sinh cùng số token → so tốc độ công bằng |
| Lưu file workload trong git? | Không commit file JSONL (có thể vài chục MB); commit **script + manifest** (seed, phiên bản dataset, sha256 của từng file) | Build lại phải ra đúng sha256 |

## Các bước

1. **Tải dataset** (ghi phiên bản/commit HF và license vào manifest):
   - ShareGPT (bản vLLM benchmark dùng: `anon8231489123/ShareGPT_Vicuna_unfiltered`) cho W1;
   - RAGBench (`rungalileo/ragbench`), các phần gần doanh nghiệp: `techqa`, `emanual`, `delucionqa`, `finqa`, `tatqa` cho W2; `cuad` cho W3.
2. **Đếm token bằng tokenizer Qwen3** sau khi đã áp chat template (đếm đúng cái server nhận).
3. **W1 — Chatbot nội bộ:** lượt hỏi đầu của hội thoại ShareGPT. Lọc prompt 200–800 token; output = độ dài câu trả lời thật, kẹp 200–600.
4. **W2 — RAG (chính):**
   - viết **một** system prompt chung ~500 token (vai trò trợ lý tài liệu nội bộ, quy tắc trích dẫn, định dạng trả lời), lưu file riêng, không đổi về sau;
   - prompt = system prompt chung + context của RAGBench + câu hỏi; lọc tổng 2–4k token; output kẹp 150–400.
5. **W3 — Batch ban đêm:** hợp đồng CUAD + yêu cầu tóm tắt cố định; lọc 4–8k token; output ~300.
   Mỗi hợp đồng chỉ dùng **một lần** (CUAD trong RAGBench có nhiều câu hỏi trên cùng hợp đồng → nếu không lọc, prefix cache sẽ được lợi không thực tế).
6. **Số mẫu:** mỗi workload ≥ 600 mẫu khác nhau (đủ cho mức 64 user × vài vòng). W3 bị giới hạn bởi số hợp đồng CUAD
   (~500) → lấy tối đa có thể, ghi số thật.
7. **Định dạng file:** theo định dạng dataset `custom` của `vllm bench serve` ở phiên bản đã ghim (kiểm tra tên trường,
   có hỗ trợ độ dài output riêng từng mẫu không). Nếu không hỗ trợ → ghi thêm trường riêng, P03 tự xử lý.
8. **Thống kê và kiểm tra:** phân bố độ dài prompt/output mỗi workload (min / trung bình / p50 / p95 / max), số mẫu,
   tỉ lệ token dùng chung giữa các prompt W2 (phải ≈ độ dài system prompt). Lưu `workloads/manifest.json`.
9. **Chạy build 2 lần** → sha256 phải trùng (chứng minh tái lập được).

## Sản phẩm

- `workloads/build.py` (+ file system prompt W2), `workloads/manifest.json`
- `data/workloads/w1.jsonl`, `w2.jsonl`, `w3.jsonl` (không commit, thêm vào `.gitignore`)
- Bảng thống kê độ dài trong báo cáo

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Dataset bị gỡ / đổi trên HF | Ghi commit hash; lưu bản sao local (không commit) |
| RAGBench thiếu mẫu 2–4k token ở một số phần | Lấy từ các phần còn lại; ghi tỉ lệ từng phần |
| License hạn chế | Chỉ dùng cho nghiên cứu, ghi rõ trong manifest và báo cáo |

## Tiêu chí xong

- [ ] 3 file workload + manifest; build lại ra cùng sha256.
- [ ] Phân bố độ dài nằm trong khoảng thiết kế của [workloads.md](../../workloads.md).
- [ ] Quyết định thinking mode đã chốt và ghi vào workloads.md.

## Báo cáo P02 cần trả lời

1. Mỗi workload có bao nhiêu mẫu, độ dài thực tế phân bố thế nào, khác thiết kế ở đâu.
2. Vì sao chọn các nguồn/phần dữ liệu này — chúng đại diện cho tác vụ doanh nghiệp nào.
3. Phần dùng chung giữa các prompt W2 dài bao nhiêu → prefix caching có thể tiết kiệm tối đa bao nhiêu % tính toán prefill.
