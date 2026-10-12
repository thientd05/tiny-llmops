# P01 — Cài vLLM và chạy thử model trên L4

> Cần GPU: có · Ước tính ~0.75 giờ GPU (~8k VND) · Phụ thuộc: không

## Mục tiêu

1. Có script cài đặt **chạy lại được y hệt** trên máy thuê mới (phiên bản ghim, không `latest`).
2. vLLM ở cấu hình mặc định nạp được `Qwen/Qwen3-30B-A3B-GPTQ-Int4` trên L4 và trả lời đúng.
3. Ghi lại **những gì vLLM tự chọn** khi để mặc định — đây là thông tin nền cho việc tune sau này.

## Các bước

### A. Ở máy local (không tốn GPU)

1. **Chọn phiên bản vLLM**: bản ổn định mới nhất đã phát hành ít nhất ~2 tuần, có bản cài sẵn (wheel) cho CUDA 12.8,
   hỗ trợ Qwen3 MoE dạng GPTQ. Đọc release notes tìm lỗi liên quan sm89 / GPTQ MoE / Qwen3
   (ví dụ lỗi FP8 KV trên Ada trong [library-gaps.md](../../library-gaps.md) #3). Ghi phiên bản + lý do vào báo cáo.
2. Viết `scripts/setup_node.sh`:
   - cài `uv`, tạo venv, `uv pip install vllm==<phiên bản>`;
   - tải model bằng `hf download` vào `/root/models/`;
   - ghi thông tin môi trường vào `env.txt`: `vllm collect-env`, `pip freeze`, `nvidia-smi -q`, `nvcc --version`;
   - in thời gian từng bước (cài, tải model).
3. Viết `scripts/serve_default.sh`: chạy `vllm serve` với cấu hình mặc định, log ra file, chờ `/health` trả OK rồi in
   thời gian khởi động.

### B. Trên L4

4. Thuê máy theo [vast-guide.md](../../vast-guide.md) (không interruptible), chạy checklist mục 4, ghi host_id/machine_id.
5. Chạy `setup_node.sh`. Ghi: thời gian cài vLLM, thời gian tải model, dung lượng model trên đĩa.
6. Chạy `serve_default.sh`. Từ log khởi động, ghi lại:
   - kernel lượng tử được chọn (dự kiến `gptq_marlin`) và đường MoE (dự kiến Marlin MoE);
   - backend attention (FlashAttention / FlashInfer / Triton);
   - có dòng kiểu "Using default MoE config" không (nghĩa là không có config tune sẵn cho L4);
   - các tính năng bật mặc định: prefix caching, chunked prefill, CUDA graph, torch.compile;
   - `max_model_len`, kích thước KV cache (số token), "maximum concurrency" vLLM tự tính;
   - VRAM: phần cho weight, phần cho KV cache, phần còn lại;
   - thời gian: nạp weight, compile, capture CUDA graph, tổng tới khi sẵn sàng.
7. Gửi thử qua `/v1/chat/completions` (dùng `curl` qua tunnel SSH hoặc ngay trên máy):
   - 1 câu hỏi ngắn; 1 prompt dạng RAG ~3k token (lấy tay từ RAGBench);
   - thử chế độ suy nghĩ (thinking) của Qwen3 bật và tắt (`/no_think` hoặc `chat_template_kwargs.enable_thinking=false`)
     → xem độ dài output khác nhau thế nào. Kiểm tra câu trả lời có nghĩa, không ra ký tự rác.
8. Rsync log + `env.txt` + output thử về `results/gd1/01-cai-dat-chay-thu/<run_id>/`. **Destroy máy.**
9. Ghi số liệu vào [measurements.md](../../measurements.md).

## Rủi ro đã thấy trước

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Không đủ bộ nhớ KV cho độ dài ngữ cảnh mặc định (40960 token). Ước tính: weight ~16–17 GB, vLLM dùng 90% của 23 GB ≈ 20.7 GB → còn ~3–4 GB cho KV; mỗi token KV (BF16) = 2 × 48 lớp × 4 KV head × 128 × 2 byte = 96 KiB → chỉ chứa ~35–40k token | Lỗi khi khởi động dạng "max_model_len lớn hơn số token KV cache chứa được" | Thêm `--max-model-len` nhỏ nhất đủ cho W3 (8k prompt + output → chọn 16384). Ghi vào báo cáo là cờ bắt buộc |
| Bản wheel vLLM không khớp CUDA / driver | Lỗi import torch / CUDA | Đổi sang bản vLLM khác hỗ trợ CUDA 12.8; ghi lý do |
| Tải torch + vLLM (~3–5 GB) hoặc model chậm | Cài > 10 phút | Ghi thời gian; nếu host mạng yếu → đổi host |
| Host 506934 hết máy | Không thuê được | Chọn host L4 khác, ghi host mới |

## Tiêu chí xong

- [ ] Server chạy với cấu hình mặc định (hoặc mặc định + ít cờ nhất, có lý do) và trả lời đúng.
- [ ] `scripts/setup_node.sh`, `scripts/serve_default.sh` đã commit; chạy lại trên máy mới không cần sửa tay.
- [ ] Bảng "vLLM mặc định chọn gì" + thời gian + VRAM đã ghi vào measurements.md.
- [ ] Máy đã destroy.

## Báo cáo P01 cần trả lời

1. Khi để mặc định, vLLM đã tự bật/chọn những gì (kernel, backend, tính năng)? Cái nào có thể không tối ưu cho L4?
2. 23 GB VRAM được chia thế nào (weight / KV cache / phần khác)? KV cache chứa được bao nhiêu token
   → suy ra tối đa bao nhiêu request RAG ~3.5k token có thể chạy cùng lúc mà không phải chờ.
3. Từ lúc thuê máy đến lúc server sẵn sàng mất bao lâu, tốn bao nhiêu tiền → chi phí cố định mỗi phiên thuê.
4. Chế độ thinking ảnh hưởng độ dài output thế nào → đề xuất bật hay tắt cho các workload (chốt ở P02).
