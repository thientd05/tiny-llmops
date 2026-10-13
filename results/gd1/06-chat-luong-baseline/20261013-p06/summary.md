# Tổng hợp chất lượng `results/gd1/06-chat-luong-baseline/20261013-p06`

## Mức tự lệch của vLLM mặc định (sinh tự do, greedy, 256 token)

| So sánh | prompt | giống hệt | lệch: vị trí p10 / p50 / p90 | còn giống tới token 64 / 256 | Δp đoạn chung: mean / p1 / p99 | KL≈ mean / p99 | độ phủ top-20 | chênh top1−top2 ở điểm lệch p50 |
|---|---|---|---|---|---|---|---|---|
| c8 r1 vs c8 r2 (cùng mức tải) | 1000 | 23.1% | 5 / 46 / 143 | 53.2% / 23.1% | 0.00006 / -0.0561 / 0.0574 | 0.00150 / 0.0256 | 0.9990 | 0.0366 |
| c8 r1 vs c64 r1 (khác mức tải) | 1000 | 24.5% | 5 / 46 / 147 | 53.6% / 24.5% | -0.00012 / -0.0607 / 0.0578 | 0.00157 / 0.0295 | 0.9990 | 0.0364 |

Giống hệt = cả 256 token trùng. Δp = p(token đã chọn) của lần sau − lần đầu, chỉ trên đoạn trước điểm lệch (cùng ngữ cảnh). KL≈ từ top-20 + phần đuôi (xấp xỉ, xem docs/references E6).

## Teacher forcing (200 prompt ngắn + 256 token output tham chiếu)

| So sánh | vị trí | same top-1 | Δp mean / RMS / p1 / p99 | KL≈ mean / p99 | độ phủ top-20 |
|---|---|---|---|---|---|
| c1 vs c8 | 44976 | 99.10% ± 0.09 | 0.00020 / 0.0171 / -0.0474 / 0.0488 | 0.00136 / 0.0166 | 0.9977 |

## Điểm tác vụ (CI 95% = p ± 1.96·SE, Miller 2024)

| Tác vụ | n | Điểm | CI 95% | Ghi chú |
|---|---|---|---|---|
| FinQA (RAG), dung sai 0.1% | 1147 | 53.6 | 50.7–56.5 | ; không có dòng Answer: 9, bị cắt độ dài: 0 |
| FinQA (RAG), dung sai 1% | 1147 | 59.2 | 56.4–62.0 | **chính**; không có dòng Answer: 9, bị cắt độ dài: 0 |
| FinQA (RAG), dung sai 5% | 1147 | 61.1 | 58.3–63.9 | ; không có dòng Answer: 9, bị cắt độ dài: 0 |
| gsm8k exact_match@flexible-extract | 1319 | 91.4 | 89.8–92.9 | lm-eval 0.4.13 |
| gsm8k exact_match@strict-match | 1319 | 90.4 | 88.8–92.0 | lm-eval 0.4.13 |
| ifeval prompt_level_strict_acc@none | 541 | 84.7 | 81.6–87.7 | lm-eval 0.4.13 |
| ifeval prompt_level_loose_acc@none | 541 | 86.9 | 84.0–89.7 | lm-eval 0.4.13 |
| mmlu_pro exact_match@custom-extract | 1000 | 69.0 | 66.1–71.9 | lm-eval 0.4.13 |
