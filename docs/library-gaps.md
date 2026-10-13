# Điểm nghi thư viện còn yếu trên L4 (sm_89) + Qwen3-30B-A3B INT4

> Nghiên cứu ngày 2026-10-12 (issue/PR/docs của vLLM, Humming, SGLang, TRT-LLM, FlashInfer, llama.cpp).
> Đây là **giả thuyết có cơ sở**, phần lớn số liệu chưa đo trên L4 → GĐ3 phải xác minh.

## Nhận định nền

Qwen3-30B-A3B: 128 expert, top-8. Số expert khác nhau bị chạm mỗi layer tăng nhanh theo batch:
M=16 → ~82 expert (64%), M=32 → ~112 (87%), nhưng **mỗi expert chỉ nhận ~1–3 token**
→ GEMM MoE luôn ở chế độ M rất nhỏ (gần GEMV), bị chặn băng thông; L4 chỉ 300 GB/s.
Thêm: ở batch 16, context 4k, KV cache đọc ~6.4 GB/bước — cùng cỡ weight expert → attention cũng quan trọng.
Cần kiểm: `lm_head` (151936×2048, thường không lượng tử) có thể chiếm ~30% byte/token ở batch nhỏ.

## Bảng ứng viên

| # | Nghi vấn | Bằng chứng | Gain tiềm năng | Độ khó |
|---|---|---|---|---|
| 1 | GEMM MoE W4A16 ở M nhỏ chỉ đạt ~30–40% băng thông | Mạnh (sm80/sm86) | 1.1–1.7× kernel MoE | TB |
| 2 | Không có kernel FP8 blockwise / W4A8 dùng FP8 tensor core của Ada | Mạnh (code + issue open) | 1.4–2.8× GEMM prefill | Cao |
| 3 | Attention decode + FP8 KV trên Ada ít được chăm | TB | Chưa rõ | TB–cao |
| 4 | Không có config fused MoE cho L4 | Mạnh (đã kiểm repo) | Nhỏ — **việc làm baseline**, không phải đóng góp | Thấp |
| 5 | Fusion tắt/không hỗ trợ trên sm89 | Mạnh (docs chính thức) | Vài % | Thấp–TB |
| 6 | Overhead routing (topk/align) batch nhỏ | Yếu, phần lớn đã sửa | Nhỏ | Thấp |
| 7 | Kernel Gated DeltaNet (nếu đổi sang Qwen3.5/3.6) | TB, chưa có số sm89 | Chưa rõ | Cao |

## Chi tiết & nguồn

**#1 MoE W4A16 small-M**
- [Humming #111](https://github.com/vllm-project/humming/issues/111): GEMM MoE đạt ~30–39% HBM ở M=1–4 (sm80); đường Triton GEMV nhanh hơn 1.36–1.51× (sm80), **1.13–1.74× trên RTX 3090**. Maintainer từ chối nhánh riêng → gap có thể còn tồn tại upstream.
- [vLLM #48066](https://github.com/vllm-project/vllm/issues/48066): heuristic `block_size_m` Marlin MoE mất 9–30% TPOT (chủ yếu khi EP; 1 GPU E=128 đã chọn tile 16). Stale, chưa fix.
- [vLLM #35922](https://github.com/vllm-project/vllm/issues/35922): Marlin MoE W4A16 chạy được trên sm89.
- Hướng: kernel MoE W4A16 chuyên shape (E=128, top-8, hidden 2048, inter 768), GEMV + split-K, gộp gate/up + SiLU, mục tiêu ≥70% của 300 GB/s.

**#2 FP8 / W4A8 trên Ada**
- [vLLM #58241](https://github.com/vllm-project/vllm/issues/58241) (open, 9/2026): block-FP8 CUTLASS loại trừ sm89 (kernel blockwise cần TMA) → fallback Marlin weight-only. **Số đo trên L4** (shape Qwen3-8B): FP8 tensor core nhanh hơn Marlin **1.4–2.8× khi M ≥ 256**; M nhỏ Marlin thắng (M=1 down_proj CUTLASS chậm ~2.3×). Không có config Triton block-FP8 cho L4/L40S.
- W4A8 CUTLASS chỉ sm90 ([PR #23198](https://github.com/vllm-project/vllm/pull/23198)); theo [model card GLM-5.3-W4A8](https://huggingface.co/gpustack/GLM-5.3-W4A8) (bên thứ ba) MoE W4A8 của vLLM và SGLang đều không hỗ trợ sm89.
- Hướng: kernel lai theo M — W4A16 GEMV khi decode, W4A8 FP8 (`mma.sync`, không TMA) khi prefill → giảm TTFT cho RAG.

**#3 Attention FP8 KV trên Ada**
- FA2 không hỗ trợ FP8 KV; FA3 chỉ Hopper → Ada dùng FlashInfer hoặc Triton.
- [Blog FP8 KV của vLLM (4/2026)](https://vllm.ai/blog/2026-04-22-fp8-kvcache) chỉ nói Hopper/Blackwell.
- [vLLM #60262](https://github.com/vllm-project/vllm/issues/60262): FP8 KV tự chọn FlashInfer rồi crash thay vì fallback Triton → đường này ít được kiểm thử.
- **Đo thật (P09, 2026-10-14, vLLM 0.30.0, L4):** `--kv-cache-dtype fp8` tự chọn **FLASHINFER**, không crash khi phục vụ; KV 81,328 token ở 0.95.
  Chỉ chết khi xin `prompt_logprobs` ở 0.95 (hết bộ nhớ, challenges #7). FlashInfer tốn thêm ~0.35 GiB ngoài KV so với FLASH_ATTN.
  KV FP8 không hiệu chuẩn đổi đáp án nhiều gấp ~2 mức tự nhiên (điểm không đổi). Tốc độ riêng của kernel FlashInfer FP8 chưa tách được (đối chứng hoãn).
- Hướng: paged decode attention GQA 32q/4kv, head_dim 128, KV FP8, split-K theo sequence.

**#4 Config MoE cho L4**
- [Thư mục configs](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe/configs): `E=128,N=768` có H20/H200/B200/GB200/MI300X/Radeon 8060S — **không có L4**.
- Phải tự chạy script tune trước để có baseline công bằng. Config JSON chỉ áp dụng đường Triton; Marlin MoE dùng heuristic riêng.

**#5 Fusion trên sm89** ([docs fusion vLLM](https://docs.vllm.ai/en/stable/design/fusions/))
- `enable_qk_norm_rope_fusion`: có sm89 nhưng mặc định tắt (chú thích † chưa rõ).
- `fuse_rope_kvcache`: chỉ ROCm. Không có fusion INT4, không có fusion MoE.
- Hướng: kernel gộp QK-RMSNorm + RoPE + ghi KV (FP8) — chương "khởi động" custom CUDA.

**#6 Routing** — [PR #55225](https://github.com/vllm-project/vllm/pull/55225): `moe_align_block_size` ~20–27 µs/lần; fast path đã có trên main nhưng loại trừ W4A16. Đo nsys trước khi đầu tư.

**#7 Gated DeltaNet** — Qwen3.6-35B-A3B: 30 lớp GDN + 10 lớp attention, 256 expert, INT4 ~20GB (sát 24GB). Kernel GDN trên NVIDIA chủ yếu Triton; vendor khác đang tự viết fused decode ([XPU #608](https://github.com/vllm-project/vllm-xpu-kernels/pull/608), [Gaudi #1834](https://github.com/vllm-project/vllm-gaudi/issues/1834)). Để ở "hướng phát triển".

## Tình trạng các thư viện trên sm89

| Thư viện | Ghi chú |
|---|---|
| vLLM | Hỗ trợ sm89 đầy đủ nhất; kernel mới (Machete, W4A8, block-FP8, FA3) ưu tiên sm90+ |
| SGLang | [Bỏ GPTQ thường trên NVIDIA](https://docs.sglang.ai/advanced_features/quantization.html), dùng awq/gptq_marlin; W4A8 MoE chỉ sm90 |
| TensorRT-LLM | [Support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) ghi sm89 có INT4 AWQ; Qwen3 MoE AWQ tài liệu mâu thuẫn — chưa xác nhận |
| LMDeploy | TurboMind mạnh AWQ INT4; hỗ trợ Qwen3 MoE chưa xác nhận ([supported models](https://lmdeploy.readthedocs.io/en/stable/supported_models/supported_models.html)) |
| llama.cpp | Chạy được Qwen3 MoE GGUF trên CUDA, nhưng **thua vLLM rõ ràng khi nhiều user** — xem mục kiểm chứng dưới |
| FlashInfer | Hỗ trợ SM75+, chưa thấy số đo trên L4 |

## Kiểm chứng: llama.cpp có thua toàn tập khi serving nhiều user? (2026-10-12)

Kết luận: **gần như chắc chắn thua trên mọi chỉ số hiệu năng mục tiêu**; chưa có số trên L4.

| Chỉ số | Bằng chứng công khai | Đánh giá |
|---|---|---|
| Goodput / throughput | [RTX 3090 24GB, **Qwen3-Coder-30B-A3B**](https://dev.to/sikamikanikobg/vllm-vs-llamacpp-vs-ollama-what-happens-when-your-model-doesnt-fit-in-24gb-vram-56eb) (cùng kiến trúc với model chính, vLLM AWQ vs GGUF, `-np 8`): c=8 → vLLM 678 tok/s vs llama.cpp 184 (**3.7×**); c1→c8 llama.cpp chỉ ×1.2. [Red Hat, H200, Llama-3.1-8B](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case): throughput llama.cpp gần như phẳng theo tải, vLLM gấp ~44× ở đỉnh | Thua |
| TTFT p95 | Red Hat: TTFT llama.cpp tăng vọt khi tải tăng (request phải chờ slot trống) | Thua |
| TPOT/ITL p95 | Red Hat: từ ~5 user, ITL p99 llama.cpp **thấp hơn**. Nhưng đây là do request xếp hàng ngoài slot, ít request chạy đồng thời hơn; vLLM đạt cùng điểm bằng `--max-num-seqs` nhỏ. Cái giá là TTFT, nên goodput (TTFT + TPOT) vẫn thua | Không phải ưu thế thật |
| W2 prefix dùng chung | Prefix cache theo từng slot, không chia sẻ giữa slot ([#8860](https://github.com/ggml-org/llama.cpp/discussions/8860)); paged KV + prefix sharing mới là đề xuất, chưa merge ([#21961](https://github.com/ggml-org/llama.cpp/discussions/21961)) | Thua (system prompt bị prefill lặp) |
| Chất lượng | Do định dạng lượng tử (GGUF Q4_K vs GPTQ/AWQ), không do thư viện. vLLM nạp GGUF qwen3moe chưa chắc ([#18382](https://github.com/vllm-project/vllm/issues/18382)) | Chưa rõ. Nếu GGUF tốt hơn → tìm định dạng tương đương cho vLLM, không đổi sang llama.cpp |

Nguyên nhân kiến trúc: KV cấp cố định theo slot (`-np` quyết định lúc khởi động), không có paged attention.
Throughput server ở batch 32 chỉ ~56% so với `llama-batched-bench` ([#18308](https://github.com/ggml-org/llama.cpp/discussions/18308)).
MoE `mul_mat_id` trên CUDA có đồng bộ host, cản CUDA Graphs ([#12859](https://github.com/ggml-org/llama.cpp/issues/12859), issue cũ, chưa thấy xác nhận đã sửa).

## Kế hoạch xác minh (GĐ3)

1. Log vLLM: backend MoE/attention được chọn, có dòng "Using default MoE config" không.
2. `nsys`: tỉ lệ thời gian theo nhóm kernel (MoE, attention, lm_head, routing, norm) ở M = 8/32/64, context 4k.
3. % của 300 GB/s cho từng kernel (bytes/time; ncu nếu có). Kernel < ~60% và chiếm > 20% thời gian → mục tiêu.
