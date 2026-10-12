# Workload, dataset & chỉ số đánh giá

> Thiết kế ngày 2026-10-12. Không đo kịch bản 1 user (không phải mục tiêu dự án).

## Ba kịch bản

| Kịch bản | Prompt / output (token) | Tải | Gap chính kiểm tra (xem [library-gaps.md](library-gaps.md)) |
|---|---|---|---|
| **W1 Chatbot nội bộ** | 200–800 / 200–600 | Trace hội thoại Azure | MoE decode (#1) |
| **W2 RAG tài liệu doanh nghiệp** (chính) | 2–4k / 150–400, system prompt ~500 token dùng chung | BurstGPT (bursty) | MoE decode (#1), attention FP8 KV (#3), TTFT |
| **W3 Batch ban đêm** (tóm tắt hợp đồng/tài liệu dài hàng loạt) | 4–8k / ~300 | Concurrency cao, throughput | Prefill FP8/W4A8 (#2) |

Concurrency: **8 / 16 / 32 / 64** + replay trace thực (scale tốc độ cho vừa năng lực L4).

## Dataset

### A. Thời điểm đến & độ dài (trace production thật, không có nội dung)

| Dataset | Nội dung | Ghi chú |
|---|---|---|
| [Azure LLM Inference Trace 2023/2024](https://github.com/Azure/AzurePublicDataset) | timestamp, ContextTokens, GeneratedTokens (Splitwise, ISCA'24) | Trace hội thoại 2024 có thể đã bị gỡ → kiểm tra repo |
| [BurstGPT](https://github.com/HPMLL/BurstGPT) | ~5.29M request, 121 ngày, Azure OpenAI, API vs conversation | Tải đột biến — kiểm SLO lúc cao điểm |
| Mooncake trace (Kimi, FAST'25) | Hash prefix, context dài | Mô phỏng prefix cache |

### B. Nội dung prompt

| Dataset | Dùng cho |
|---|---|
| **[RAGBench](https://huggingface.co/datasets/rungalileo/ragbench)** ([arXiv 2407.11005](https://arxiv.org/pdf/2407.11005)): ~100k mẫu RAG chuẩn hoá từ 12 dataset, **context có sẵn** (100–11k token). Domain doanh nghiệp: hợp đồng (CUAD), hỗ trợ khách hàng / tài liệu kỹ thuật (TechQA, EManual, DelucionQA), tài chính (FinQA, TAT-QA) | W2 — RAG; W3 — CUAD (hợp đồng dài) |
| ShareGPT, LMSYS-Chat-1M, WildChat | W1 — ShareGPT là chuẩn benchmark vLLM, so được với paper |

**Ghép:** timestamp + độ dài từ (A), nội dung từ (B) khớp độ dài.
Prompt RAG = system prompt chung + context của RAGBench + câu hỏi → không cần tự dựng hệ retrieval/embedding.

### C. Chất lượng (tối ưu không làm hỏng model)

| Cấp | Cách đo |
|---|---|
| Kernel | Unit test vs PyTorch reference (max abs/rel error) |
| Model | ~1k prompt greedy: tỉ lệ token trùng, KL divergence logits vs vLLM gốc |
| Tác vụ | MMLU-Pro, độ chính xác RAG (RAGBench test split), GSM8K |

⚠️ Kiểm tra license từng dataset (WildChat, LMSYS, RAGBench và các dataset thành phần) — chỉ dùng cho nghiên cứu.

## Chỉ số

- **SLO đề xuất:** TTFT p95 < 2s, TPOT p95 < 100ms
- **Goodput:** số user đồng thời tối đa vẫn đạt SLO / 1 L4
- **Chi phí:** VND / 1 triệu token (đầu ra)
- Phụ: throughput (tok/s), TTFT/TPOT p50/p95/p99, VRAM, % roofline từng kernel
