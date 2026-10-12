# P06 — Chất lượng baseline + output tham chiếu

> Cần GPU: có · Ước tính ~1.5 giờ GPU (~15k VND) · Phụ thuộc: P01, P02 (không cần P04, làm cùng phiên được)

## Mục tiêu

1. Đo chất lượng của vLLM mặc định → mốc mà mọi tối ưu sau (KV FP8, thư viện khác, kernel tự viết) **không được làm giảm**.
2. Lưu **output tham chiếu** để sau này so từng token: kernel mới có cho ra cùng kết quả như vLLM gốc không.
3. Đo **mức sai khác tự nhiên** của chính vLLM: cùng prompt, cùng greedy, nhưng chạy trong batch khác nhau có thể ra token khác
   (thứ tự cộng số thực khác nhau). Không biết mức này thì không thể nói kernel mới "làm sai".

## Quyết định cần chốt (đề xuất sẵn)

| Câu hỏi | Đề xuất |
|---|---|
| Thinking mode | Giống P02 (đề xuất tắt) để chất lượng đo đúng chế độ dùng khi benchmark |
| Công cụ đo tác vụ | `lm-evaluation-harness` (ghim phiên bản), gọi vào server vLLM qua API |
| MMLU-Pro | Tập con cố định ~1000 câu, chia đều theo môn, seed ghi lại (bản đủ 12k câu quá tốn giờ GPU) |
| Độ chính xác RAG | Chọn phần RAGBench có đáp án ngắn chấm tự động được (ví dụ số liệu trong FinQA/TAT-QA); không dùng model khác làm giám khảo ở giai đoạn này. Chốt cách chấm cụ thể khi bắt đầu plan |

## Các bước

1. **Output tham chiếu:** ~1000 prompt cố định (trộn W1/W2/W3, tách biệt khỏi phần dùng cho benchmark), greedy
   (`temperature=0`), tối đa 256 token output. Lưu: token id từng vị trí + top-20 logprob từng vị trí.
   (Không lưu toàn bộ logits ~152k giá trị/token — quá lớn; top-20 đủ để tính KL xấp xỉ.)
2. **Mức sai khác tự nhiên:** chạy lại bộ tham chiếu ở 2 mức số user đồng thời khác nhau (ví dụ 8 và 64) → tỉ lệ token trùng
   và KL xấp xỉ giữa các lần chạy của chính vLLM mặc định.
3. **Tác vụ:** GSM8K (đủ 1319 câu), MMLU-Pro tập con, RAG accuracy. Ghi điểm + khoảng sai số (độ lệch chuẩn của điểm theo số câu).
4. Rsync output (token id + logprob, có thể vài trăm MB → nén), destroy máy. Ghi điểm vào measurements.md.

## Sản phẩm

- `quality/reference.py` (sinh + so sánh output tham chiếu), `quality/configs/` (lệnh lm-eval)
- `results/gd1/06-chat-luong-baseline/` — output tham chiếu (không commit nếu lớn; ghi sha256 trong manifest)

## Tiêu chí xong

- [ ] Output tham chiếu đã lưu, có script so sánh (tỉ lệ token trùng, KL xấp xỉ) chạy được.
- [ ] Mức sai khác tự nhiên của vLLM đã đo.
- [ ] Điểm GSM8K, MMLU-Pro (tập con), RAG accuracy + sai số đã ghi measurements.md.

## Báo cáo P06 cần trả lời

1. Điểm chất lượng của baseline, so với điểm Qwen công bố cho bản gốc BF16 (bản INT4 kém đi bao nhiêu, nếu so được).
2. Chính vLLM chạy 2 lần ở batch khác nhau thì output khác nhau bao nhiêu → ngưỡng "sai khác chấp nhận được" cho kernel sau này.
3. Với số câu đã chọn, điểm chênh bao nhiêu mới là khác biệt thật (không phải do ngẫu nhiên chọn câu).
