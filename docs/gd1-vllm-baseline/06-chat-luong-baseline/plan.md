# P06 — Chất lượng baseline + output tham chiếu

> Cần GPU: có · Ước tính ~1.5 giờ GPU (~15k VND) · Phụ thuộc: P01, P02 (không cần P04, làm cùng phiên được)
>
> **Xong 2026-10-13.** Đo trên L4 21:05–22:52 (1.78 h, ≈ 15,500 VND). Báo cáo: [đọc ngay](doc-ngay.md) · [đầy đủ](bao-cao.md).
>
> **Bước 1 (local, 0 giờ GPU) xong 2026-10-13:** quyết định đã chốt kèm tài liệu minh chứng (mục "Quyết định đã chốt" bên dưới,
> bằng chứng ở [references nhóm E](../../references/README.md)); code `quality/`; thử đầu-cuối với server giả.
> Ước lại giờ GPU: **~2.2–2.6 h (~22–26k VND)** — mục "Ước giờ GPU".

## Mục tiêu

1. Đo chất lượng của vLLM mặc định → mốc mà mọi tối ưu sau (KV FP8, thư viện khác, kernel tự viết) **không được làm giảm**.
2. Lưu **output tham chiếu** để sau này so từng token: kernel mới có cho ra cùng kết quả như vLLM gốc không.
3. Đo **mức sai khác tự nhiên** của chính vLLM: cùng prompt, cùng greedy, nhưng chạy trong batch khác nhau có thể ra token khác
   (thứ tự cộng số thực khác nhau). Không biết mức này thì không thể nói kernel mới "làm sai".

## Quyết định cần chốt (đề xuất ban đầu, 2026-10-12 — đã thay bằng mục "Quyết định đã chốt" cuối file)

| Câu hỏi | Đề xuất |
|---|---|
| Thinking mode | Giống P02 (đề xuất tắt) để chất lượng đo đúng chế độ dùng khi benchmark |
| Công cụ đo tác vụ | `lm-evaluation-harness` (ghim phiên bản), gọi vào server vLLM qua API |
| MMLU-Pro | Tập con cố định ~1000 câu, chia đều theo môn, seed ghi lại (bản đủ 12k câu quá tốn giờ GPU) |
| Độ chính xác RAG | Chọn phần RAGBench có đáp án ngắn chấm tự động được (ví dụ số liệu trong FinQA/TAT-QA); không dùng model khác làm giám khảo ở giai đoạn này. Chốt cách chấm cụ thể khi bắt đầu plan |

## Chạy tiếp được

- Output tham chiếu: chia thành nhóm 100 prompt, mỗi nhóm một file + `done.json` → dừng thì mất tối đa 1 nhóm (~1 phút).
- lm-evaluation-harness: mỗi tác vụ (GSM8K, MMLU-Pro, RAG) là một đơn vị; bật `--use_cache <file>` để lưu từng câu đã trả lời,
  chạy lại chỉ hỏi những câu còn thiếu. File cache nằm trong thư mục kết quả nên được rsync về local cùng kết quả.

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

- [x] Output tham chiếu đã lưu, có script so sánh (tỉ lệ token trùng, KL xấp xỉ) chạy được. (`quality/analyze.py --compare`)
- [x] Mức sai khác tự nhiên của vLLM đã đo. (giống hệt 23–25%, same top-1 99.10%)
- [x] Điểm GSM8K, MMLU-Pro (tập con), RAG accuracy + sai số đã ghi measurements.md. (+ IFEval)

## Báo cáo P06 cần trả lời

1. Điểm chất lượng của baseline, so với điểm Qwen công bố cho bản gốc BF16 (bản INT4 kém đi bao nhiêu, nếu so được).
2. Chính vLLM chạy 2 lần ở batch khác nhau thì output khác nhau bao nhiêu → ngưỡng "sai khác chấp nhận được" cho kernel sau này.
3. Với số câu đã chọn, điểm chênh bao nhiêu mới là khác biệt thật (không phải do ngẫu nhiên chọn câu).

## Quyết định đã chốt (2026-10-13, bước 1) — mỗi quyết định có tài liệu minh chứng

Mã tài liệu (E1…E12) là mục trong [docs/references/README.md](../../references/README.md) nhóm E; bản gốc lưu ở `docs/references/files/`.

| # | Quyết định | Căn cứ |
|---|---|---|
| D1 | **Tắt thinking** (`enable_thinking=False`, giống mọi lần đo tốc độ). **Greedy** (temperature 0), không lấy mẫu như Qwen | Qwen công bố điểm non-thinking của đúng model (E1 Bảng 16, E2). Qwen đánh giá non-thinking bằng lấy mẫu (E1 mục 4.6) → điểm của mình không trùng tuyệt đối với số công bố; greedy loại phương sai do lấy mẫu, cần cho so sánh theo từng câu (E3 khuyến nghị 3–4) |
| D2 | **Tác vụ:** MMLU-Pro (tập con 1,000 câu), IFEval (541), GSM8K (1,319), FinQA-RAG (1,147) | MMLU-Pro 69.1 và IFEval 83.7 là số công bố non-thinking (E2, E1) → trả lời được "INT4 kém bản gốc bao nhiêu". GSM8K: Qwen chỉ công bố cho bản Base (91.81, E1 Bảng 5) → chỉ dùng làm cổng kiểm cho các bước tối ưu (bài toán nhiều bước, nhạy với sai số tính; Dutta dùng GSM8K để đo flips, E4). FinQA: bài toán chính của đồ án (RAG trên báo cáo doanh nghiệp) |
| D3 | **MMLU-Pro lấy ngẫu nhiên 1,000 / 12,032 câu**, chia theo tỉ lệ 14 môn, seed cố định (`data/quality/mmlu_pro_samples.json`, truyền `lm-eval --samples`) | E3: n ≈ 969 đủ phát hiện chênh 3 điểm (α 0.05, power 80%), "at least 1,000 questions"; so hai cấu hình dùng paired SE nên còn nhạy hơn. Chạy đủ 12k tốn ~12× giờ GPU |
| D4 | **Mọi điểm báo kèm CI 95%** (p ± 1.96·√(p(1−p)/n)); **so hai cấu hình bằng paired SE + flips** | E3 (khuyến nghị 1, 4), E4 (flips tới 13.6% dù độ chính xác chênh 0–2%) |
| D5 | **FinQA-RAG:** câu hỏi RAGBench `finqa` test (khớp 1,138/1,138 câu với FinQA test gốc), prompt dựng như W2 + dòng cuối "Answer: <value>"; **đáp án gốc `exe_ans`**; **đúng nếu lệch tương đối ≤ 1%**, chấp nhận dạng phần trăm (0.1446 ↔ 14.46), yes/no so khớp; báo thêm 0.1% và 5% | E10: cột `response` của RAGBench là câu do LLM sinh, không phải đáp án. E9: chỉ số gốc là execution accuracy. E12: FinLFQA dùng dung sai tương đối 0.01 + chuẩn hoá đơn vị; paper GPT-3 tài chính báo accuracy theo nhiều mức dung sai từ 1% |
| D6 | **Độ gần token: so khi sinh tự do là chính**, teacher forcing chỉ trên 200 prompt ngắn (≤ 300 token) | Thay cho "lưu top-20 rồi tính KL" của plan cũ. llama.cpp so trên cùng văn bản (teacher forcing) với KLD, Δp, same top p (E5) nhưng cần lưu đủ logits (37 GiB). vLLM 0.30 tính `prompt_logprobs` cho **mọi vị trí prompt**: logits float32 cả khúc 2,048 token rồi logprob (vllm/v1/worker/gpu_model_runner.py ~dòng 5711–5724) → ~2.5 GB bộ nhớ tạm ngoài phần đã dành → rủi ro OOM trên L4 với prompt RAG dài. Sinh tự do + `logprobs=20` chỉ tính cho token sinh ra |
| D7 | **Chỉ số độ gần:** % output giống hệt; vị trí bắt đầu lệch; trên đoạn chung trước điểm lệch: Δp (mean, RMS, p1/p99 — đối xứng hay lệch âm), KL xấp xỉ top-20 + đuôi, độ phủ top-20; teacher forcing: same top-1 | E5 (Δp đối xứng = nhiễu, lệch âm = kém đi), E7 (đo vị trí lệch), E6 (top-k KL bị lệch → ghi là xấp xỉ, báo độ phủ) |
| D8 | **Mức tự lệch:** sinh tham chiếu 3 lần: c8 r1, c8 r2 (cùng mức tải), c64 r1 (khác mức tải); teacher forcing c1 vs c8 | E7: lệch do forward pass không bất biến theo kích thước batch (Qwen3, 80 kết quả khác nhau / 1,000 lần ở T = 0). E8: vLLM không đảm bảo tái lập mặc định, chỉ tái lập trên cùng phần cứng + phiên bản → mọi so sánh chất lượng phải cùng máy L4, cùng vLLM |
| D9 | **lm-eval 0.4.13 ghim**, model `local-completions-nothink` (lớp con 15 dòng, `quality/lmeval_run.py`), mức đồng thời 32, `--use_cache`, `max_gen_toks=1024` (mặc định 256 của model API cắt lời giải GSM8K; MMLU-Pro 2,048 và IFEval 1,280 theo yaml của task). Chỉ số chính chốt trước khi đo: GSM8K `flexible-extract` (strict-match báo kèm), IFEval `prompt_level_strict_acc` (khớp số Qwen công bố "IFEval strict prompt"), MMLU-Pro `custom-extract` | E11: model API của lm-eval không truyền `enable_thinking` → phải tự render template. Mức đồng thời cố định và ghi lại vì kết quả phụ thuộc batch (E7) |
| D10 | **Bộ tham chiếu 1,000 prompt:** 500 W1-like (ShareGPT ngoài w1.jsonl), 400 W2-like (RAGBench **test**, prompt 1–4k), 100 W3-like (lấy từ `w3.jsonl`) | Lệch 2 điểm so với plan cũ, có lý do: (a) split test có tài liệu ngắn hơn: ≥ 2k token chỉ có 180 câu, ≥ 1k có 472 → nới ngưỡng dưới xuống 1k; (b) W3 đã dùng toàn bộ CUAD (train+val+test) nên không còn tài liệu dài riêng — trùng với bộ đo tốc độ không ảnh hưởng vì bộ tham chiếu đo tính đúng, không đo tốc độ |

## Code (bước 1)

| File | Việc |
|---|---|
| `quality/build_sets.py` | Dựng `data/quality/{reference,finqa_rag}.jsonl`, `mmlu_pro_samples.json`; sha256 + thống kê ở `quality/manifest.json` |
| `quality/run.py` | Chạy trên máy thuê, chạy tiếp được: đơn vị gen / tf (khúc 100 prompt) / lmeval (`--use_cache`), `--status` |
| `quality/lmeval_run.py` | Model `local-completions-nothink` cho lm-eval |
| `quality/analyze.py` | Chấm FinQA, đọc lm-eval, độ gần token, CI; `--compare BASE VARIANT` (paired + flips) cho các bước sau |
| `quality/remote_quality.sh` | Setup (dùng lại `scripts/setup_node.sh`) + venv lm-eval ghim + chạy |
| `quality/configs/p06.yaml`, `test-fake-p06.yaml` | Cấu hình thật / thử với server giả |
| `bench/fake_server.py` | Thêm `logprobs`, `prompt_logprobs`, `token_ids`, prompt dạng id, `--noise` (giả lập lệch do batch) |

## Ước giờ GPU (cập nhật 2026-10-13)

Ước từ mô hình chi phí P04 (0.37 ms / token prompt chưa cache, 1.17 ms / token ra khi bão hoà):

| Phần | Token prompt / ra | Ước |
|---|---|---|
| Sinh tham chiếu × 3 (1,000 prompt, TB ~1.3k token, 256 ra) | 3 × (1.3M / 0.26M) | ~0.75 h |
| Teacher forcing × 2 (200 × ~600 token) | nhỏ | ~0.05 h |
| FinQA-RAG (1,147 × ~1.7k, ra ~300) | 2.0M / 0.35M | ~0.35 h |
| GSM8K 5-shot (1,319, ra ~200) | ~1.2M (cache few-shot) / 0.26M | ~0.2 h |
| IFEval (541, ra ~400) | nhỏ / 0.22M | ~0.1 h |
| MMLU-Pro 5-shot CoT (1,000, ra ~500) | ~2.5M / 0.5M | ~0.4 h |
| Cài máy + tải model + dataset + khởi động | | ~0.3 h |
| **Tổng** | | **~2.2–2.6 h ≈ 22–26k VND** (giá P05 ~9,900 VND/h) |

Cao hơn ước cũ 1.5 h vì thêm FinQA, IFEval và 3 lần sinh tham chiếu (cần cho mức tự lệch).

## Chạy trên L4 (bước 2, khi được duyệt)

```bash
# local: đẩy code + 3 bộ dữ liệu (~15 MB) + w2.jsonl (warmup)
rsync -az --relative bench/*.py bench/configs scripts/*.sh quality/*.py quality/*.sh quality/configs quality/manifest.json \
      workloads/build.py workloads/manifest.json workloads/prompts data/workloads/w2.jsonl data/quality root@<IP>:/root/td/
ssh ... "cd /root/td && tmux new -d -s q 'BENCH_MACHINE=... bash quality/remote_quality.sh quality/configs/p06.yaml /root/results/p06'"
# local: kéo kết quả mỗi 2 phút; xem tiến độ
.venv/bin/python quality/run.py --config quality/configs/p06.yaml --out results/gd1/06-chat-luong-baseline/<run> --status
# sau khi destroy
.venv/bin/python quality/analyze.py --out results/gd1/06-chat-luong-baseline/<run>
```

Kết quả thô (token id + top-20 logprob của 3 lần sinh, ~100–150 MB nén) **không commit** nếu lớn; ghi sha256 vào measurements.

## Lần 2 (2026-10-14): flips tự nhiên

Chạy lại 4 bài chấm điểm (`quality/configs/p06-r2.yaml`) trên cùng máy 116596 (chờ máy này trống thay vì dùng host khác).
Flips tự nhiên 1.6–6.6%; chi tiết [bao-cao.md](bao-cao.md) mục 4.3b.
