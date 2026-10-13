# Nhật ký đo đạc

> Ghi **mọi** số liệu đo được, kèm ngày, máy (host/machine ID), phần mềm và lệnh tái lập.
> Mục mới thêm lên **đầu** file. Không sửa số cũ — nếu đo lại thì thêm mục mới.

---

## 2026-10-14 — P07: tổng hợp baseline + phân tích thêm trên dữ liệu đã có (local, không GPU)

Báo cáo: [gd1-vllm-baseline/07-tong-hop-baseline/bao-cao.md](gd1-vllm-baseline/07-tong-hop-baseline/bao-cao.md). Máy: laptop, Python 3 hệ thống + numpy.
Nguồn: `results/gd1/04-do-baseline-day-du/20261013-p04/` (`units.csv`, `units/*/bench.json.gz`, `metrics_*.txt`),
`results/gd1/05-w4-tai-giong-that/20261013-p05/` (`units/w4-*/bench.json.gz`, `metrics_timeseries.jsonl`, `cost.json`), các mục P01–P06 bên dưới.

**Giá thuê thực theo tín dụng bị trừ** (tỉ giá 26,200):

| Phiên | Machine | Giờ tính tiền | $ | $/h | VND/h |
|---|---|---|---|---|---|
| P01 | 116596 | 0.346 | 0.126 | 0.364 | 9,540 |
| P03 + P04 | 116596 | 5.251 | 1.516 | 0.289 | 7,560 |
| P05 | 150535 | 4.97 | 1.878 | 0.378 | 9,900 |
| P06 / P06 lần 2 | 116596 | 1.78 / 0.89 | 0.592 / 0.320 | 0.333 / 0.359 | 8,710 / 9,400 |
| **Tổng** | | **13.24** | **4.432** | **0.335** | **8,770 = giá tham chiếu** |

Tín dụng $10.000 → $5.549 = $4.451 ≈ 116,600 VND (gồm thử máy $0.0038 và $0.014 bị trừ giữa P04 và P05 chưa rõ khoản).

**Mốc baseline + chi phí tại giá tham chiếu** (W1–W3: trung vị 3 lần lặp [min–max]; giá 7,560 … 9,900 cho khoảng theo máy):

| | Goodput | Goodput từng lần lặp / seed | Token ra/s tại goodput | VND / 1M token ra | Theo giá 7,560 … 9,900 | VND / 1M token vào+ra | VND / 1,000 request |
|---|---|---|---|---|---|---|---|
| W1 | 68 (72) | 68 / ≥ 72 / 68 | 615 [605–623] | **3,960** [3,910–4,030] | 3,420 … 4,470 | 1,790 | 1,340 |
| W2 | 18 (20) | 18 / 20 / 18 | 168 [167–171] | **14,460** [14,250–14,580] | 12,470 … 16,320 | 830 | 2,480 |
| W3 | 6 (8) | 6 / 6 / 6 | 101 [101–102] | **24,030** [23,950–24,130] | 20,720 … 27,130 | 1,200 | 7,190 |
| W4 cả ngày | k = 18 (20) seed 1 | seed 2: 16 (18); mỗi seed 1 lần | 259 (peak) | **20,240** (seed 2: 22,770) | 17,450 … 25,710 | — | 5,550 (seed 2: 6,240) |

W4: token ra cả ngày = k × 577,632 (`cost.json`); kiểm 1,738 request peak k = 18: tổng token ra = tổng `max_tokens` (client `ignore_eos`).
P05 ghi 22,510 với giá niêm yết 9,752 VND/h; cùng throughput.

**W4 — phần prompt là lịch sử dùng lại được** (lượt sau cùng phiên, lượt trước đã chạy trong lần đo, client không cắt lịch sử;
= prompt trước + output trước − 4; mọi request đã gửi):

| Lần chạy | Request | Trúng cache (server) | Dùng lại được | Trong đó câu trả lời trước | Lượt sau / bị client cắt |
|---|---|---|---|---|---|
| peak s1 k = 16 | 2,105 | 4.4% | 45.8% | 13.5% | 1,189 / 312 |
| peak s1 k = 18 | 2,240 | 3.0% | 44.7% | 13.4% | 1,266 / 337 |
| peak s2 k = 16 | 2,395 | 3.7% | 44.2% | 14.6% | 1,423 / 391 |
| peak2 s1 k = 18 | 2,019 | 9.9% | 46.4% | 14.0% | 1,202 / 327 |
| normal s1 k = 18 | 837 | 14.5% | 42.1% | 12.6% | 487 / 127 |

Theo phút (peak s1 k = 18, cửa sổ đo): phần dùng lại được 33–72% mỗi phút, trung vị ~51%; phút 16 (85,318 token prompt, TTFT p95 38.0 s) chỉ 38%;
phút 7 (55,651, 11.1 s) 39%.

**W4 — request < 1k token: TTFT theo số request đang chờ trên server lúc gửi × có request ≥ 4k đang chờ token đầu** (`metrics_timeseries.jsonl`,
căn thời gian theo giây đầu tiên có request chạy, sai số ~1–2 s):

| Lần chạy | Hàng đợi | Không có request dài: n, p50 / p95 (s) | Có: n, p50 / p95 (s) |
|---|---|---|---|
| peak s1 k = 18 | 0 | 1,081, 0.21 / 0.49 | 34, 1.10 / 3.06 |
| | 1–3 | 42, 0.33 / 4.85 | 7, 2.36 / 6.12 |
| | 4–9 | 69, 3.70 / 15.29 | 16, 7.08 / 15.19 |
| | ≥ 10 | 42, 9.84 / 17.47 | 31, 22.94 / 38.66 |
| peak s1 k = 16 | 0 | 1,078, 0.19 / 0.52 | 32, 1.08 / 3.74 |
| | ≥ 10 | 16, 10.01 / 15.40 | 21, 24.53 / 28.55 |
| peak s2 k = 16 | 0 | 1,308, 0.19 / 0.41 | 13, 0.80 / 1.19 |

**W1 — preemption:** delta `vllm:iteration_tokens_total_sum` = delta `prompt_tokens_total` − delta `prefix_cache_hits_total` + delta `generation_tokens_total`
ở mọi đơn vị P04 (W1/68 r1: 201,063 = 111,208 − 240 + 90,095) → bộ đếm **không tính token phải tính lại sau preemption**; chi phí preemption chưa đo được.
Thời gian prefill TB phía server: 155 ms (W1/8) → ~450 ms (W1/68).

**Nhiễu giữa 3 lần lặp P04** ((max − min) / trung vị, 22 mức; trong ngoặc: 10 mức sát ngưỡng W1 64–72, W2 16–24, W3 4–8):
TTFT TB 7.6% (17.0%), TTFT p50 11.2% (22.7%), TTFT p95 15.7% (23.4%), thời gian chờ TB phía server 17.3%, E2EL TB 5.2%, token ra/s 1.6% (2.2%), TPOT p95 3.6%.

**p95 gộp 3 lần lặp (300–864 request)**, CI bootstrap 2,000 lần (coi request độc lập → lạc quan):
W1/64 5.47 s [5.04–5.91], W1/68 8.20 [7.84–8.36], W1/72 11.20 [10.73–11.75]; W2/16 5.97 [5.13–6.37], W2/18 7.23 [6.86–9.68], W2/20 **10.87 [8.97–15.14]**,
W2/24 16.02 [14.44–17.24]; W3/4 2.29 [1.87–2.55], W3/6 5.16 [4.51–6.39], W3/8 13.06 [12.20–13.84] → cùng goodput 68 / 18 / 6 như trung vị 3 lần.

**Cặp cùng seed W2/16, P05 (machine 150535) so P04 (116596):** TTFT p95 −9.3% / +2.0% / −5.0%; token ra/s +2.3% / +2.4% / +1.8% (seed 1001 / 1002 / 1003).
Seed 1003 có prompt TB 2,923 token, dài hơn seed 1001 (2,821) 3.6%.

**Mặc định vLLM 0.30.0 trên L4** (đọc `vllm/engine/arg_utils.py` `get_batch_defaults`, tag v0.30.0; GPU < 70 GB): `max_num_batched_tokens` 2,048,
`max_num_seqs` 256 (server OpenAI). `SchedulerConfig`: `policy` fcfs, `scheduler_reserve_full_isl` True, `watermark` 0.0, `long_prefill_token_threshold` 0.
`CacheConfig`: `kv_offloading_size` None (backend `native` | `lmcache`).

Lệnh tái lập: `.venv/bin/python bench/p07_analyze.py --price 8770 --price-lo 7560 --price-hi 9900` (in đủ mọi bảng trên; ~1 phút).
---

## 2026-10-14 — P06 lần 2: chạy lại các bài chấm điểm của baseline → flips tự nhiên

Kết quả thô: `results/gd1/06-chat-luong-baseline/20261013-p06r2/` (`self_flips.md`, `self_flips.json`, `units/<id>-r2/`; 30 MB).
Cấu hình `quality/configs/p06-r2.yaml` = 4 đơn vị chấm điểm của `p06.yaml`, y nguyên tham số, đơn vị mới `-r2` (không dùng cache lm-eval lần 1).

| Trường | Giá trị |
|---|---|
| Thời gian | thuê 23:42 (13/10) → destroy 00:35:31 (14/10) = 53 phút; chạy đo 23:55 → 00:34 |
| Máy | instance 55776612, **host 506934 / machine 116596** (cùng máy P06 lần 1; driver 595.84), $0.305/h. Chờ máy này thay vì dùng host khác (570878) để không lẫn lệch do đổi máy |
| Chi phí | tín dụng $5.869 → $5.549 = **$0.320 ≈ 8,400 VND** |
| Lệnh | `bash quality/remote_quality.sh quality/configs/p06-r2.yaml /root/results/p06r2`; local `quality/analyze.py --self-flips <p06> <p06r2>` |

Thời gian đơn vị: FinQA 9.6 phút, GSM8K 10.9, IFEval 5.2, MMLU-Pro 13.6 (bằng lần 1).

**Điểm lần 2:** FinQA 60.1 (1%) / 54.0 (0.1%) / 62.0 (5%); GSM8K 91.4 flexible / 90.7 strict; IFEval strict 83.5 / loose 86.9; MMLU-Pro **67.4**.

**Flips tự nhiên lần 1 vs lần 2** (paired, Miller E3; flips theo Dutta E4):

| Bài (chỉ số) | n | Hiệu r2 − r1 (điểm) | paired SE | CI 95% của hiệu | Flips đúng→sai / sai→đúng | Flips |
|---|---|---|---|---|---|---|
| FinQA dung sai 1% | 1,147 | +0.87 | 0.63 | −0.36 … +2.10 | 21 / 31 | **4.5%** |
| FinQA dung sai 0.1% / 5% | 1,147 | +0.35 / +0.87 | 0.63 / 0.60 | | 24 / 28 · 19 / 29 | 4.5% / 4.2% |
| GSM8K flexible / strict | 1,319 | +0.08 / +0.30 | 0.35 / 0.37 | −0.61 … +0.76 / −0.42 … +1.03 | 10 / 11 · 10 / 14 | **1.6%** / 1.8% |
| IFEval prompt strict / loose | 541 | −1.11 / 0.00 | 0.98 / 0.98 | −3.03 … +0.81 / −1.92 … +1.92 | 17 / 11 · 14 / 14 | **5.2%** / 5.2% |
| MMLU-Pro custom-extract | 1,000 | **−1.60** | 0.81 | **−3.19 … −0.01** | 41 / 25 | **6.6%** |

- FinQA: đáp án trích ra khác nhau giữa 2 lần ở 15.1% câu (nhưng phần lớn vẫn cùng đúng / cùng sai).
- MMLU-Pro: cùng cấu hình mà hiệu −1.6 điểm có CI 95% paired vừa không chứa 0 → kiểm định paired một lần chạy có thể báo "khác biệt"
  giả. Với 8 cặp chỉ số so cùng lúc, kỳ vọng ~0.4 kết quả giả ở mức 5%.
---

## 2026-10-13 — P06: chất lượng baseline + output tham chiếu + mức tự lệch (vLLM 0.30.0 mặc định, L4)

Báo cáo: [gd1-vllm-baseline/06-chat-luong-baseline/bao-cao.md](gd1-vllm-baseline/06-chat-luong-baseline/bao-cao.md).
Kết quả thô: `results/gd1/06-chat-luong-baseline/20261013-p06/` (`summary.md`, `summary.json`, `finqa_per_question.json`,
`<task>_per_doc.json`, `units/<id>/`; 129 MB).

| Trường | Giá trị |
|---|---|
| Thời gian | thuê 21:05:19 → destroy 22:51:59 (13/10, +07) = 1.78 h; chạy đo 21:18:51 → 22:51:08 |
| Máy | instance 55753483, **host 506934 / machine 116596** (máy của P03/P04), $0.306/h trọn gói |
| Chi phí | tín dụng $6.461 → $5.869 = **$0.592 ≈ 15,500 VND** |
| Checklist | L4 23,034 MiB, driver 595.84, 72 W, 140 GB đĩa, 94 GB RAM, 30 vCPU; ncu ERR_NVGPUCTRPERM |
| Cài đặt | model 158 s, vLLM 192 s; lm-eval 0.4.13 + transformers 5.19.0 + datasets 5.1.0 (`setup/lmeval-freeze.txt`) |
| Server | `serve_default.sh --max-model-len 16384` (baseline đúng nghĩa, không cờ thêm). Lần 1 compile từ đầu 66.9 s → KV 28,704 → tự bật lại; lần 2 compile 1.3 s → **KV 37,264** (= P04) |
| Cấu hình | `quality/configs/p06.yaml`; greedy, `enable_thinking=False`; dữ liệu sha256 như mục "P06 bước 1" |
| Lệnh | `bash quality/remote_quality.sh quality/configs/p06.yaml /root/results/p06`; local `quality/analyze.py --out <dir>` |

**Thời gian từng đơn vị:** ref-gen c8 r1 / r2 / c64 r1 = 19.0 / 19.0 / 13.0 phút; teacher forcing c1 / c8 = 1.2 / 0.8 phút;
FinQA 9.6; GSM8K 10.8; IFEval 5.3; MMLU-Pro 13.6 phút.

**Điểm tác vụ** (CI 95% = p ± 1.96·√(p(1−p)/n)):

| Tác vụ | n | Điểm | CI 95% | Số công bố (Qwen, bản gốc, non-thinking, lấy mẫu T = 0.7) |
|---|---|---|---|---|
| MMLU-Pro (custom-extract, 5-shot CoT, tập con ngẫu nhiên) | 1,000 | **69.0** | 66.1–71.9 | 69.1 (model card 2507, cột "Qwen3-30B-A3B Non-Thinking") |
| IFEval prompt-level strict | 541 | **84.7** | 81.6–87.7 | 83.7 (tech report Bảng 16, "IFEval strict prompt") |
| IFEval prompt-level loose / inst-level strict / inst-level loose | 541 | 86.9 / 89.6 / 91.1 | | |
| GSM8K flexible-extract (chính) / strict-match, 5-shot | 1,319 | **91.4** / 90.4 | 89.8–92.9 / 88.8–92.0 | (chỉ có bản Base: 91.81) |
| **FinQA-RAG** dung sai 1% (chính) / 0.1% / 5% | 1,147 | **59.2** / 53.6 / 61.1 | 56.4–62.0 / 50.7–56.5 / 58.3–63.9 | — |

- MMLU-Pro theo môn: math 90.2, biology 90.0, economics 84.3, health 76.5, chemistry 72.3, business 71.2, physics 69.4, CS 67.6,
  psychology 66.7, history 65.6, other 64.9, philosophy 58.5, engineering 50.6, **law 34.1**. 25 câu (2.5%) không trích được "answer is (X)".
- GSM8K: strict-match không trích được 23 câu; trả lời dài nhất 1,179 ký tự (không chạm max_gen_toks 1,024).
- FinQA: 60 câu không có số trong dòng Answer — đều là từ chối ("not enough information", "cannot be determined..."); 9 câu không có
  dòng Answer; 0 câu bị cắt độ dài; prompt dài nhất 3,183 token (không tài liệu nào bị cắt bởi giới hạn 4k).

**Mức tự lệch** (1,000 prompt tham chiếu, greedy 256 token; 435 dừng tự nhiên, 565 chạm 256):

| So sánh | Giống hệt | Vị trí lệch p10 / p50 / p90 | Còn giống tới 16 / 64 / 128 | Δp đoạn chung mean / RMS / p1 / p99 | KL≈ mean / p99 / p99.9 | Độ phủ top-20 | Chênh top1−top2 ở điểm lệch p50 / p90 |
|---|---|---|---|---|---|---|---|
| c8 r1 vs c8 r2 | **23.1%** | 5 / **46** / 143 | 84.3% / 53.2% / 33.7% | 0.00006 / 0.0186 / −0.0561 / +0.0574 | 0.0015 / 0.026 / 0.139 | 0.999 | 0.037 / 0.240 |
| c8 r1 vs c64 r1 | **24.5%** | 5 / 46 / 147 | — / 53.6% / — | −0.00012 / 0.0193 / −0.0607 / +0.0578 | 0.0016 / 0.029 / 0.149 | 0.999 | 0.036 / 0.245 |

Theo nguồn prompt (c8 r1 vs c8 r2, giống hệt % / vị trí lệch p50): ShareGPT 20.6% / 49; CUAD 13.0% / 77; techqa 10.9% / 45;
tatqa 43.8% / 32.5; delucionqa 32.9% / 48; emanual 18.4% / 35.

**Teacher forcing** (200 prompt ≤ 300 token + 256 token output của c8 r1, 44,976 vị trí): c1 vs c8 **same top-1 99.10% ± 0.09**;
Δp mean 0.0002, RMS 0.0171, p1 −0.0474, p99 +0.0488, max |Δp| 0.93; KL≈ mean 0.0014, p99 0.017, p99.9 0.079; độ phủ top-20 0.998.
---

## 2026-10-13 — P06 bước 1: bộ dữ liệu đo chất lượng + thử harness với server giả (local, không GPU)

Plan + quyết định: [gd1-vllm-baseline/06-chat-luong-baseline/plan.md](gd1-vllm-baseline/06-chat-luong-baseline/plan.md) mục "Quyết định đã chốt".
Lệnh: `.venv/bin/python quality/build_sets.py` (seed 20261013, 22 s). Nguồn ghim: MMLU-Pro `TIGER-Lab/MMLU-Pro@b189ec76`, FinQA `czyssrs/FinQA@0f16e286`
(`dataset/test.json`), RAGBench / ShareGPT / tokenizer như P02.

| Bộ | Nội dung | sha256 |
|---|---|---|
| `data/quality/reference.jsonl` | 1000 prompt: ShareGPT 500 (TB 369 token), RAGBench test techqa 129 / tatqa 160 / delucionqa 73 / emanual 38 (TB 3284 / 1331 / 1654 / 1311), CUAD 100 (TB 5582); 200 prompt ngắn (≤ 300 token) cho teacher forcing | `54f2f731d02f…` |
| `data/quality/finqa_rag.jsonl` | 1147 câu (RAGBench finqa test, mỗi id 1 câu), 20 câu yes/no; prompt TB 1700, max 3183 token; 0 câu bỏ (1,138/1,138 câu hỏi khớp FinQA test) | `bd637fb7da46…` |
| `data/quality/mmlu_pro_samples.json` | 1,000 / 12,032 câu MMLU-Pro, theo tỉ lệ 14 môn (math 112, physics 108, chemistry 94, law 91 … history 32) | `705c194f7e1f…` |

- Ứng viên W2-like ở RAGBench test: ≥ 2k token chỉ 180, ≥ 1k có 472 → bộ tham chiếu dùng prompt 1–4k token.
- Tập test gốc (HF datasets-server, 2026-10-13): GSM8K test 1,319; MMLU-Pro test 12,032 / validation 70; IFEval 541.

**Thử với server giả** (`quality/configs/test-fake-p06.yaml`, `bench/fake_server.py --noise 0.02`; số đo vô nghĩa, chỉ kiểm đường chạy):

| Kiểm tra | Kết quả |
|---|---|
| Chạy đủ 9 đơn vị (3 gen tham chiếu, 2 teacher forcing, FinQA, GSM8K / IFEval / MMLU-Pro qua lm-eval 0.4.13) | 3.7 phút, mã thoát 0 |
| Prompt lm-eval gửi đi (model `local-completions-nothink`) | kết thúc bằng `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` — giống lúc đo tốc độ |
| `--samples` MMLU-Pro (2 câu × 14 môn) | 28 câu, chỉ số khớp đúng; câu lm-eval đọc = câu trong parquet ở chỉ số đó |
| `kill -9` giữa đơn vị gen rồi chạy lại | 3 khúc đã xong giữ nguyên (mtime không đổi), chạy tiếp tới hết |
| `quality/test_analyze.py` (bộ đọc đáp án FinQA 14 trường hợp + KL) | OK sau khi sửa lỗi lấy nhầm số trong trích dẫn "[Document 2]" |
---

## 2026-10-13 — P05: W4 tải giống thật trên L4 (vLLM 0.30.0 mặc định)

Báo cáo: [gd1-vllm-baseline/05-w4-tai-giong-that/bao-cao.md](gd1-vllm-baseline/05-w4-tai-giong-that/bao-cao.md).
Kết quả thô: `results/gd1/05-w4-tai-giong-that/20261013-p05/` (`summary.md`, `summary.json`, `cost.json`, `units/<id>/`).
Hình: `docs/gd1-vllm-baseline/05-w4-tai-giong-that/hinh/`.

| Trường | Giá trị |
|---|---|
| Thời gian | thuê 14:28:16 → destroy 19:26:08 (13/10, +07) = 4.97 h |
| Máy | instance 55701260, **host 506934 / machine 150535** (cùng host với P03/P04, khác machine 116596), $0.372/h trọn gói |
| Chi phí | tín dụng $8.339 → $6.461 = **$1.878 ≈ 49,200 VND** (≈ $0.378/h ≈ 9,900 VND/h) |
| Checklist máy | L4 23,034 MiB, driver 615.71.09, 72 W, 140 GB đĩa, 125 GB RAM, 40 vCPU, nvcc 13.0.88; **ncu: ERR_NVGPUCTRPERM** (như mọi lần trên host 506934) |
| Cài đặt | tải model 179 s, cài vLLM 187 s |
| Phần mềm | vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.19.0, model `Qwen/Qwen3-30B-A3B-GPTQ-Int4@9b534e43`, image `nvidia/cuda:13.0.2-devel-ubuntu22.04` |
| Server | `serve_default.sh --max-model-len 16384 --enable-prompt-tokens-details` (cờ thứ hai chỉ thêm số token trúng cache vào câu trả lời) |
| Lần khởi động | #1 compile từ đầu 60.8 s → KV **29,472** token → harness tự bật lại; #2 compile 0.37 s → KV **38,032** token (P04: 37,264); #3 dùng lại #2 |
| Cấu hình | `bench/configs/w4.yaml` (chế độ w4): khởi đầu nóng k = 16 / 24, 1 lần chạy mỗi mức, dừng sớm khi chắc chắn trượt; kiểm tra ở k = lo; thêm `extra: [[peak-s2, 16]]` sau khi peak-s2 trượt ở 18 |
| Lệnh | `bash scripts/remote_bench.sh bench/configs/w4.yaml /root/results/p05` (tmux); local: `summarize.py --config bench/configs/w4.yaml --out <dir>`, `w4_report.py ... --price-vnd-h 9752` |

**Điểm mốc W2 / 16 user** (vllm bench serve, 100 request, seed 1001–1003):

| | r1 | r2 | r3 | P04 (machine 116596) |
|---|---|---|---|---|
| TTFT p95 (ms) | 5,714 | 4,961 | 5,722 | 6,303 / 4,866 / 6,025 |
| TPOT p95 (ms) | 87.9 | 92.1 | 88.7 | 89.6 / 92.7 / 89.9 |
| tok/s ra | 175 | 173 | 170 | 171 / 169 / 167 |

→ máy này nhanh hơn máy P04 ~2% (throughput), KV nhiều hơn 2%. So sánh được.

**W4** (SLO chính TTFT p95 ≤ 10 s, TPOT p95 ≤ 150 ms; chỉ request gửi trong 30 phút đo; "dừng sớm" = số là của phần đã chạy):

| Lần chạy | Bắt đầu (+07) | Phút | Phiên | Request đo | req/giờ | Phiên mở p50 | TTFT p50 / p95 / p99 (ms) | TPOT p50 / p95 (ms) | tok/s ra | Kết quả |
|---|---|---|---|---|---|---|---|---|---|---|
| peak-s1 k=16 | 14:54:56 | 40.6 | 916 | 1,628 | 3,256 | 342 | 238 / 5,492 / 25,516 | 47.4 / 76.1 | 243 | **đạt** |
| peak-s1 k=24 | 15:35:31 | 18.9 | 1,257 | 617 | — | — | 459 / ∞ / ∞ | 74.0 / ∞ | 287 | trượt, dừng sớm ở t = 1121 s (153 chắc trượt TTFT) |
| peak-s1 k=20 | 15:54:25 | 27.7 | 1,084 | 1,099 | — | — | 322 / 30,939 / ∞ | 63.0 / 191.2 | 270 | trượt, dừng sớm ở t = 1646 s (131 TTFT, 12 TPOT) |
| **peak-s1 k=18** | 16:22:05 | 40.7 | 974 | 1,738 | **3,476** | **363** | 272 / **9,109** / 25,258 | 55.7 / **81.9** | 259 | **đạt → goodput 18 (20 trượt)** |
| peak-s2 k=18 | 17:02:45 | 20.3 | 1,085 | 685 | — | — | 508 / ∞ / ∞ | 70.0 / ∞ | 289 | trượt, dừng sớm ở t = 1206 s |
| peak2-s1 k=18 (10h) | 17:23:04 | 40.5 | 817 | 1,555 | 3,110 | 288 | 165 / 617 / 1,485 | 31.9 / 53.7 | 194 | đạt |
| normal-s1 k=18 (13h) | 18:03:31 | 40.2 | 350 | 628 | 1,256 | 117 | 127 / 564 / 996 | 18.5 / 36.4 | 96 | đạt |
| peak-s2 k=16 (thêm) | 18:44:21 | 40.7 | 972 | 1,770 | 3,540 | 377 | 224 / 872 / 4,313 | 47.3 / 73.1 | 259 | đạt → seed 2: 16 (18 trượt) |

Phía server (trung vị theo giây trong lần chạy; preempt và trúng prefix tính hiệu /metrics):

| Lần chạy | đang chạy p50 / max | chờ max | KV max % | giây KV ≥ 97% | giây có hàng đợi | preempt | trúng prefix (server) |
|---|---|---|---|---|---|---|---|
| peak-s1 k=16 | 11 / 33 | 25 | 100 | 4% | 7% | 42 | 4.4% |
| peak-s1 k=18 | 14 / 32 | 29 | 100 | 8% | 11% | 80 | 3.0% |
| peak-s2 k=16 | 13 / 33 | 16 | 100 | 3% | 4% | 37 | 3.7% |
| peak2 k=18 | 5 / 26 | 6 | 83 | 0% | 0% | 0 | 9.9% |
| normal k=18 | 2 / 11 | 1 | 85 | — | — | 0 | 14.5% |

- **Goodput W4 (peak 16h, seed 1) = 18 (20 trượt)** ≈ 3,476 req/giờ, 363 phiên đang mở, 14 request đang chạy trên server (p50).
  Bộ ngày khác (seed 2): **16 (18 trượt)** → lệch 1 bước tìm. SLO phụ (2 s / 100 ms) và TTFT ≤ 5 s: < 16. TTFT ≤ 20 s: ≥ 18.
- **TTFT vượt SLO theo đợt:** peak-s1 k=18, TTFT p95 theo phút > 10 s chỉ ở phút 7–8 và 16–17 của cửa sổ đo (Hình 2). Phút 16:
  85,318 token prompt (trung vị 43,872/phút), 4 request ≥ 4k; số request 65 (bình thường 41–73).
- **Request ngắn (< 1k) khi có request ≥ 4k đang chờ token đầu:** TTFT p95 26.6 s vs 1.4 s (k=16), 32.2 s vs 4.2 s (k=18).
  40–70% request ngắn có TTFT > 10 s được gửi đúng lúc có request ≥ 4k đang chờ (21/30 ở k=16, 28/53 ở k=18).
- **Prefix cache theo think time** (lượt sau không bỏ lịch sử, phần lịch sử trúng cache trung bình): think < 30 s: 25% (peak k=18),
  51% (normal); 30–120 s: 0% / 40%; ≥ 120 s: 0% (98–100% lượt không trúng token nào). Phần câu trả lời của lượt trước (p50 37% lịch sử)
  **không bao giờ trúng** (0% lượt có cached > prompt lượt trước + 16).
- **Gửi trễ của client** (giây thật): p50 0.01, p99 ≤ 0.33, max ≤ 0.93 (peak2). 0 request bị cắt sau cửa sổ.
- **Chi phí** (`cost.json`, giá 9,752 VND/h = $0.372 × 26,200): k = 18 phục vụ 10.40 triệu token ra / ngày (theo 24 giờ của trace) →
  **22,510 VND / 1 triệu token ra** nếu thuê cả ngày. Nếu cả ngày như giờ cao điểm: 10,322. Token ra đo được / trace × k:
  16h 944,804 / 998,074 (−5%), 10h 704,978 / 800,261 (−12%), 13h 344,102 / 462,638 (−26%, cửa sổ 13:00–13:30 vắng hơn cả giờ).
---

## 2026-10-13 — P05 chuẩn bị local: kịch bản W4 sửa phiên nhiều ngày, harness W4 thử với server giả (không GPU)

Plan: [gd1-vllm-baseline/05-w4-tai-giong-that/plan.md](gd1-vllm-baseline/05-w4-tai-giong-that/plan.md) mục "Chuẩn bị local".
Máy: laptop (local), `.venv`, tokenizer Qwen3 (`data/raw/qwen3-tokenizer`). Server giả: `bench/fake_server.py`.

**Sửa `build_w4.py` (phiên nhiều ngày).** Một Session ID của BurstGPT có thể xuất hiện ở cửa sổ của nhiều ngày (người dùng quay lại
hội thoại cũ vài ngày sau). Bản cũ chỉ lấy lần xuất hiện đầu → mất lượt ở ngày sau, và lọc kịch bản gốc theo ngày lệch với sinh riêng
(peak k=10: 540 vs 554 phiên). Bản mới: mỗi (phiên, ngày) là một phiên của kịch bản (`sid` = `<Session ID>@d<ngày>`).
Mọi kịch bản sinh lại (thay số ở mục "Phân tích trace BurstGPT_3 … Kịch bản" bên dưới; số cũ giữ nguyên làm lịch sử).
Lệnh: `.venv/bin/python workloads/build_w4.py scenario --segment <khung> --k <k> --seed <s>`; sha256 ở `workloads/manifest_w4.json` mục `scenarios`.

| Kịch bản | Phiên (quay lại) | Request tổng / đo | req/giờ đo / kỳ vọng | Phiên mở p50 / max | prompt p50 / p90 / p99 | <1k / 1–2k / 2–4k / 4–8k / ≥8k (%) | max_tokens p50 / p90 | sha256 |
|---|---|---|---|---|---|---|---|---|
| **peak k=110 s1** (gốc) | 6,170 (2,674) | 15,137 / 11,400 | 22,800 / 22,917 | 2,352 / 2,469 | 528 / 2362 / 5324 | 67.9 / 18.6 / 11.4 / 1.8 / 0.4 | 183 / 675 | `120bf130…` |
| **peak k=110 s2** (gốc) | 6,170 (2,674) | 15,137 / 11,400 | 22,800 / 22,917 | 2,352 / 2,469 | 528 / 2367 / 5304 | 67.9 / 18.5 / 11.4 / 1.8 / 0.4 | 183 / 675 | `6ef578b9…` |
| **peak2 k=110 s1** (gốc) | 4,925 (1,348) | 12,262 / 9,435 | 18,870 / 19,807 | 1,869 / 1,928 | 390 / 1923 / 5109 | 74.1 / 16.4 / 7.8 / 1.4 / 0.3 | 161 / 604 | `d93b12b5…` |
| **normal k=110 s1** (gốc) | 2,146 (837) | 5,082 / 3,869 | 7,738 / 10,156 | 751 / 799 | 504 / 2457 / 5946 | 66.5 / 18.6 / 11.7 / 2.5 / 0.6 | 212 / 686 | `94367d09…` |
| peak k=1 | 17 (10) | 43 / 32 | 64 / 208 | 8 / 10 | 831 / 2766 / 3754 | 62.5 / 18.8 / 18.8 / 0 / 0 | 170 / 658 | `143149ef…` |
| peak k=10 | 559 (223) | 1,231 / 965 | 1,930 / 2,083 | 201 / 226 | 422 / 2157 / 5039 | 72.0 / 16.3 / 9.3 / 2.2 / 0.2 | 173 / 698 | `a4b5d301…` |
| peak k=40 | 2,248 (985) | 5,394 / 4,139 | 8,278 / 8,334 | 848 / 904 | 516 / 2414 / 6143 | 68.3 / 17.5 / 11.5 / 2.0 / 0.7 | 173 / 676 | `dc30696e…` |
| peak2 k=10 | 462 (145) | 1,210 / 927 | 1,854 / 1,801 | 159 / 176 | 389 / 1720 / 6606 | 78.1 / 14.6 / 5.3 / 1.4 / 0.6 | 131 / 509 | `396be71c…` |
| normal k=10 | 196 (82) | 499 / 401 | 802 / 923 | 69 / 74 | 501 / 2439 / 4948 | 64.1 / 17.7 / 15.5 / 2.5 / 0.2 | 190 / 662 | `146cc291…` |

- peak k=110: thiếu so với kỳ vọng còn 0.5% (bản cũ 3.4%).
- normal (13h) đo thấp hơn kỳ vọng 24%. Kỳ vọng tính trên cả giờ 13:00–14:00, còn cửa sổ đo là 13:00–13:30, mà nửa đầu giờ 13 vắng hơn
  (nghỉ trưa). Đây là đặc điểm của trace, không phải lỗi.
- Hai kịch bản gốc peak s1 / s2 chứa cùng 110 ngày, chỉ khác nội dung tin nhắn và thứ tự ngày (hoán vị theo seed). Ở k = lo, s2 lấy bộ ngày khác s1.
- **Lọc kịch bản gốc theo k ngày đầu = sinh riêng k** (peak k=1/10/40, peak2 k=10, normal k=10): cùng số phiên, cùng phiên,
  cùng thời điểm, think time, độ dài trace, max_tokens. Chỉ 1–2% lượt khác ngân sách prompt (8–103 lượt), là các lượt có độ dài tin nhắn
  lấy ngẫu nhiên (client gốc đã cắt lịch sử).

**Thử harness W4 với server giả** (`bash bench/test_fake.sh <dir> w4`, cấu hình `bench/configs/test-fake-w4.yaml`: thời gian co 100 lần,
server giả hàng đợi 24 chỗ, SLO co theo 100 ms / 3 ms):

| Kiểm tra | Kết quả |
|---|---|
| Tìm k (khởi đầu nóng 16 / 24) | 16 đạt → 24 trượt (dừng sớm sau 204 s trace của cửa sổ đo) → 20 đạt → 22 trượt (dừng sớm) → **20 (22 trượt)**, rồi tự chạy peak-s2 và normal ở k = 20 |
| Ngắt bằng SIGKILL giữa một lần chạy, chạy lại cùng lệnh | thư mục dở bị xoá và chạy lại; 4–5 đơn vị đã xong không bị chạy lại (so mtime `done.json`) |
| `--scenario-check` 6 lần chạy | 0 lượt gửi sớm hơn "nhận xong + think time" |
| Điểm mốc W2/16 (vllm bench serve bản CPU) | chạy trước W4, đúng 1 đơn vị |
| Chạy lại tổng hợp P04 bằng `bench/common.py` mới | `summary.md` giống hệt (trừ dòng đường dẫn) |
| `bench/test_search.py`, `workloads/test_w4_session.py`, `test_fake.sh k40 config scenario` | đều OK |

**Tải CPU của client** (`session_client.py scenario`, peak k=24, thời gian co **10 lần** → 12.0 request/giây, gấp ~10 lần tải thật ở k=24;
server giả 256 chỗ, 5 ms/token):

| Lần | Thời gian | CPU client | Gửi trễ so với lịch (giây thật) p50 / p99 / max | Ghi chú |
|---|---|---|---|---|
| 1 | 9:55 | 9% (47.8 s) | 0.005 / 1.02 / 3.5 | phần phát lại xong ở giây 245; đợt gửi trễ dồn ở t ≈ 107 s và thời gian thừa trùng lúc phiên làm việc bị ngắt → coi là tiến trình bị dừng tạm, không dùng |
| 2 | **4:07** | 19% (41.9 s) | **0.004 / 0.079 / 0.31** | xong ở giây 243.7 (cửa sổ 240 s); 0 lượt gửi sớm, 0 lỗi |

→ Client không phải điểm nghẽn ở tải P05 (dự kiến k ≈ 16–24 → 1–1.5 request/giây).
---

## 2026-10-13 — P04: baseline đầy đủ W1/W2/W3 (vLLM 0.30.0 mặc định, L4)

Báo cáo: [gd1-vllm-baseline/04-do-baseline-day-du/bao-cao.md](gd1-vllm-baseline/04-do-baseline-day-du/bao-cao.md).
Kết quả thô: `results/gd1/04-do-baseline-day-du/20261013-p04/` (`summary.md`, `units.csv`, `units/<id>/`). Hình: `docs/gd1-vllm-baseline/04-do-baseline-day-du/hinh/`.

| Trường | Giá trị |
|---|---|
| Thời gian | 00:19:15 → 04:45:06 (13/10, +07), ngay sau P03 trên cùng instance |
| Máy | instance 55583011, **host 506934 / machine 116596**, $0.2833/h; destroy 04:45:50 |
| Phần mềm / server / client | như mục P03 (vLLM 0.30.0, `--max-model-len 16384`, `vllm bench serve` completions, ignore_eos, temperature 0) |
| Cấu hình | `bench/configs/full.yaml`: search, lưới gốc 8/16/32/64, ×3 lần lặp (seed 1001–1003), max(100, 4N) request, đạt/trượt theo trung vị, cửa sổ "all"; khởi động lại server giữa các lần lặp lưới gốc |
| Lần khởi động | r1: dùng lại server lần 3 của P03 (KV 37,264); r2, r3: khởi động lại, **KV 37,264** cả hai, compile 0.4–0.6 s |
| Lệnh | `bash scripts/remote_bench.sh bench/configs/full.yaml /root/results/p04` · local: `python bench/summarize.py --config bench/configs/full.yaml --out <dir>`, `python bench/plot.py ... --fig-dir <hinh>` |
| Số đơn vị | 66 (22 mức × 3), 0 request lỗi; tổng thời gian đơn vị 4.35 h |

**Goodput** (trung vị 3 lần; "lo (hi trượt)"):

| | SLO chính 10 s / 150 ms | SLO phụ 2 s / 100 ms | TTFT ≤ 2 s | ≤ 5 s | ≤ 10 s | ≤ 20 s | Cửa sổ steady, SLO chính | Mức đã đo |
|---|---|---|---|---|---|---|---|---|
| W1 | **68 (72)** | 32 (64) | 32 (64) | 32 (64) | 68 (72) | 80 (96) | 68 (72) | 8, 16, 32, 64, 128, 96, 80, 72, 68 |
| W2 | **18 (20)** | < 8 | < 8 | 8 (16) | 18 (20) | 24 (32) | 20 (24) | 8, 16, 32, 64, 24, 20, 18 |
| W3 | **6 (8)** | < 4 | < 4 | 4 (6) | 6 (8) | 8 (16) | 6 (8) | 8, 16, 32, 64, 4, 6 |

Chỉ số hỏng trước ở cả ba: **TTFT** (TPOT p95 lớn nhất 117 ms < 150). W2 / 20: TTFT p95 trung vị **10.06 s** (r1/r2/r3 = 11.07 / 8.76 / 10.06 s) — sát ngưỡng.

**Từng mức** (trung vị 3 lần, cửa sổ all; "chạy" = số request đang chạy p50 giữa lần đo / max):

| w | N | TTFT p50 / p95 / p99 (s) | TPOT p50 / p95 / p99 (ms) | tok/s ra | tok/s vào | KV max % | chạy | chờ max | preempt | VRAM max (MiB) | W max | xung SM min | °C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W1 | 8 | 0.18 / 0.43 / 0.6 | 32.1 / 33.2 / 34 | 243 | 291 | 18 | 8 / 8 | 2 | 0 | 21,145 | 74.4 | 990 | 70 |
| W1 | 16 | 0.22 / 0.78 / 1.1 | 46.8 / 47.9 / 48 | 323 | 398 | 32 | 16 / 16 | 0 | 0 | 21,145 | 73.5 | 1,005 | 71 |
| W1 | 32 | 0.31 / 1.77 / 2.1 | 64.3 / 66.1 / 66 | 458 | 541 | 60 | 32 / 32 | 20 | 0 | 21,145 | 73.1 | 1,035 | 71 |
| W1 | 64 | 1.44 / 5.89 / 6.9 | 90.5 / 97.9 / 109 | 610 | 747 | 100 | 60 / 64 | 45 | 97 | 21,145 | 74.0 | 1,125 | 72 |
| W1 | 68 | 3.00 / 8.30 / 9.3 | 92.7 / 103.6 / 114 | 615 | 746 | 100 | 60 / 68 | 52 | 133 | 21,237 | 73.6 | 1,125 | 72 |
| W1 | 72 | 4.05 / 11.14 / 12.7 | 93.1 / 104.4 / 119 | 612 | 757 | 100 | 60 / 72 | 53 | 143 | 21,237 | 73.2 | 1,110 | 73 |
| W1 | 80 | 7.59 / 15.60 / 16.7 | 94.1 / 113.5 / 136 | 617 | 758 | 100 | 59 / 80 | 64 | 164 | 21,237 | 73.4 | 1,125 | 72 |
| W1 | 96 | 17.76 / 22.97 / 26.2 | 95.0 / 117.1 / 149 | 619 | 753 | 100 | 60 / 88 | 80 | 230 | 21,237 | 73.4 | 1,110 | 73 |
| W1 | 128 | 34.65 / 41.90 / 47.7 | 95.1 / 112.7 / 146 | 620 | 760 | 100 | 60 / 86 | 113 | 330 | 21,209 | 73.8 | 1,125 | 73 |
| W2 | 8 | 0.71 / 2.03 / 3.5 | 50.8 / 56.9 / 58 | 146 | 2,376 | 64 | 8 / 8 | 6 | 0 | 21,145 | 74.0 | 1,095 | 73 |
| W2 | 16 | 1.57 / 6.03 / 6.8 | 82.4 / 89.9 / 94 | 169 | 2,777 | 100 | 15 / 16 | 13 | 6 | 21,145 | 73.3 | 1,110 | 73 |
| W2 | 18 | 3.35 / 7.42 / 15.1 | 81.6 / 90.2 / 94 | 168 | 2,775 | 100 | 15 / 17 | 15 | 7 | 21,237 | 73.8 | 1,125 | 73 |
| W2 | 20 | 5.28 / 10.06 / 15.7 | 82.8 / 93.2 / 103 | 168 | 2,759 | 100 | 15 / 17 | 16 | 8 | 21,237 | 73.2 | 1,110 | 73 |
| W2 | 24 | 9.16 / 16.03 / 18.0 | 81.7 / 91.1 / 96 | 169 | 2,777 | 100 | 15 / 17 | 21 | 7 | 21,209 | 73.8 | 1,125 | 73 |
| W2 | 32 | 17.08 / 21.26 / 28.4 | 82.1 / 92.2 / 100 | 169 | 2,743 | 100 | 15 / 17 | 28 | 7 | 21,145 | 73.5 | 1,095 | 73 |
| W2 | 64 | 49.01 / 53.76 / 56.3 | 84.1 / 91.4 / 100 | 172 | 2,788 | 100 | 15 / 18 | 61 | 17 | 21,145 | 73.9 | 1,080 | 73 |
| W3 | 4 | 1.33 / 2.15 / 3.4 | 38.9 / 43.2 / 45 | 93 | 1,766 | 78 | 4 / 4 | 2 | 0 | 21,209 | 74.5 | 1,065 | 74 |
| W3 | 6 | 1.72 / 5.14 / 7.7 | 52.3 / 57.3 / 60 | 101 | 1,935 | 100 | 6 / 6 | 5 | 1 | 21,237 | 73.8 | 1,065 | 74 |
| W3 | 8 | 6.97 / 13.06 / 17.1 | 53.9 / 58.6 / 62 | 102 | 1,945 | 100 | 6 / 7 | 6 | 2 | 21,145 | 74.0 | 1,080 | 73 |
| W3 | 16 | 31.12 / 36.53 / 38.0 | 54.1 / 59.1 / 62 | 102 | 1,942 | 100 | 6 / 7 | 14 | 3 | 21,145 | 74.1 | 1,095 | 73 |
| W3 | 32 | 75.58 / 83.51 / 86.2 | 53.9 / 60.6 / 62 | 103 | 1,943 | 100 | 6 / 7 | 30 | 7 | 21,145 | 74.0 | 1,080 | 73 |
| W3 | 64 | 165.53 / 176.57 / 178.7 | 53.9 / 60.2 / 65 | 103 | 1,944 | 100 | 6 / 7 | 61 | 14 | 21,145 | 74.2 | 1,065 | 73 |

Trúng prefix cache: W1 0–0.4%, W2 21.6%, W3 1.6%. TTFT TB client − server: 13–90 ms (trung vị 31 ms).
Phân tách TTFT TB phía server (r1): W1/68 3.35 s = chờ 2.79 + prefill 0.45; W2/18 3.83 = 2.78 + 0.88; W3/6 2.23 = 0.72 + 1.42; W3/8 7.23 = 5.68 + 1.43.

**Nhiễu giữa 3 lần lặp** ((max − min) / trung vị, 22 mức): tok/s ra trung vị 1.6% (max 5.1%); TPOT p95 3.6% (max 5.9%);
E2EL p50 5.7% (max 14.8%); **TTFT p95 15.7% (p75 23.5%, max 47.8%)**; TTFT p95 steady 14.4% (max 97%).

**Chi phí cả phiên P03 + P04** (hoá đơn instance 55583011): GPU 5.251 h = $1.400 · ổ đĩa $0.088 · tải về 21.2 GB $0.028 →
**$1.516 ≈ 39,700 VND** (credit 9.870 → 8.353). Chia theo thời gian: P03 0.84 h ≈ 7,000 VND; **P04 4.44 h ≈ 32,700 VND**.

---

## 2026-10-12 — P03: harness benchmark + smoke trên L4 (vLLM 0.30.0 mặc định)

Báo cáo: [gd1-vllm-baseline/03-cong-cu-benchmark/bao-cao.md](gd1-vllm-baseline/03-cong-cu-benchmark/bao-cao.md).
Kết quả thô: `results/gd1/03-cong-cu-benchmark/20261012-p03/` (`summary.md`, `units/<id>/`, `server_starts.jsonl`).

| Trường | Giá trị |
|---|---|
| Thời gian | instance chạy 23:28:55 (12/10) → smoke xong 00:17:30 (13/10, +07); P04 chạy tiếp trên cùng máy |
| Instance / offer | 55583011 / 49498406 |
| Host / machine | **host 506934 / machine 116596** (cùng máy P01), $0.2833/h tổng (disk 60 GB) |
| Phần mềm | vLLM 0.30.0 (`vllm[bench]`, thêm pandas cho `vllm bench serve`), torch 2.13.0+cu130, driver 595.84, image `nvidia/cuda:13.0.2-devel-ubuntu22.04`; chi tiết `setup/env.txt` |
| Server | `scripts/serve_default.sh <log> --max-model-len 16384` (VLLM_SERVER_DEV_MODE=1) |
| Client | `vllm bench serve --backend openai --endpoint /v1/completions --dataset-name custom --custom-output-len -1 --skip-chat-template --ignore-eos --temperature 0 --max-concurrency N --num-prompts max(100,4N) --seed 1000+r --no-oversample --save-detailed`; và `bench/session_client.py closed` (cùng mẫu) |
| Trước mỗi đơn vị | warmup 4 request (16 token ra) → `POST /reset_prefix_cache` (success) |
| Lệnh | `bash scripts/remote_bench.sh bench/configs/smoke.yaml /root/results/p03` · local: `python bench/summarize.py --config bench/configs/smoke.yaml --out <dir> --compare-clients` |

**Checklist máy:** băng thông VRAM (D2D 1 GiB × 10) **231.2 GB/s**; `ncu` → `ERR_NVGPUCTRPERM`; PCIe Gen3; power limit 72 W, xung SM max 2040 MHz.
Setup máy mới (uv + vLLM + model 16 GB song song): **186 s**.

**Dung lượng KV cache theo lần khởi động** (giả thuyết P01 — đã xác nhận):

| Lần | Loại | torch.compile (s) | init engine (s) | sẵn sàng (s) | KV cache |
|---|---|---|---|---|---|
| 1 | máy mới, compile từ đầu | 71.05 | 244.2 | 342.6 | **2.63 GiB = 28,704 token** |
| 2 | khởi động lại, dùng cache compile | 0.56 | 52.3 | 142.3 | **3.41 GiB = 37,264 token** |
| 3 | khởi động lại, dùng cache compile | 1.21 | 39.1 | 123.8 | 3.41 GiB = 37,264 token |

→ Lần compile từ đầu cho KV ít hơn 23% (cùng máy, cùng cờ). Harness tự bật lại server nếu compile > 10 s (`recompile_restart_s`).

**Smoke** (1 lần/điểm trừ W2/16; cửa sổ "all" = mọi request; "steady" = bỏ N request gửi đầu và N cuối). Lần khởi động: r1 = KV 28,704; r2, r3 = KV 37,264.

| Điểm | KV (token) | TTFT p50 / p95 (s) | TTFT p95 steady (s) | TPOT p50 / p95 (ms) | tok/s ra | tok/s vào | chạy max | chờ max | KV dùng max | preempt | SLO chính |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W1 / 16 r1 | 28,704 | 0.22 / 1.09 | 0.31 | 46.8 / 48.7 | 322 | 399 | 16 | 0 | 41% | 0 | đạt |
| W2 / 8 r1 | 28,704 | 0.71 / 2.35 | 1.50 | 48.9 / 57.0 | 146 | 2,378 | 8 | 6 | 86% | 0 | đạt |
| W2 / 16 r1 | 28,704 | 4.50 / 9.91 | 8.93 | 64.6 / 75.7 | 161 | 2,611 | 13 | 13 | 100% | 9 | đạt (sát) |
| W2 / 16 r2 | 37,264 | 1.41 / 4.58 | 3.62 | 82.1 / 92.6 | 170 | 2,879 | 16 | 12 | 100% | 7 | đạt |
| W2 / 16 r3 | 37,264 | 2.28 / 5.55 | 4.97 | 81.5 / 89.2 | 168 | 2,743 | 16 | 13 | 100% | 6 | đạt |
| W2 / 32 r1 | 28,704 | 22.09 / 25.05 | 23.85 | 65.9 / 75.0 | 160 | 2,573 | 13 | 29 | 100% | 9 | trượt |
| W3 / 8 r1 | 28,704 | 12.88 / 15.73 | 15.53 | 43.0 / 48.5 | 96 | 1,840 | 6 | 6 | 100% | 5 | trượt |

**Kiểm tra công cụ:**
- (a) token vào khớp **từng request** với `prompt_tokens` của P02 (mọi đơn vị), token ra = tổng `output_tokens` (ignore_eos).
- (b) TTFT trung bình client − server (`vllm:time_to_first_token_seconds`): +17 … +33 ms (0.2–6.5%).
- (c) trúng prefix cache W2 **21.6%** = phần system prompt chung 622 / 2,858 token (21.8%); W3 1.6% → reset có tác dụng.
- (d) nhiễu r2 vs r3 (W2/16, cùng KV 37,264, có khởi động lại giữa hai lần), (max − min) / trung vị: tok/s ra 1.2%, TPOT p95 3.7%,
  E2EL p50 ~9%, **TTFT p95 19% (all) / 31% (steady)**; client tự viết: TTFT p95 0.9% / 36%.
- Client tự viết vs `vllm bench serve`, gộp r2+r3 (200 request mỗi bên): TTFT p50 +5.3%, p95 **+3.7%**, TPOT p95 +0.4%, E2EL p50 −0.5%, tok/s ra −1%.
- GPU chạm trần công suất: 73–74.5 W, xung SM thấp nhất 975–1,140 MHz (max 2,040), nhiệt độ tối đa 73 °C.

**Thời gian mỗi đơn vị** (cả warmup + reset + nạp tokenizer ~2–3 s + client): W1/16 124 s (1.24 s/request), W2/8–32 101–164 s (1.0–1.4 s/request),
W3/8 334 s (3.3 s/request). Khởi động lại server (có cache): 124–142 s.

---

## 2026-10-12 — Bộ sinh kịch bản W4 (local, không GPU)

| Trường | Giá trị |
|---|---|
| Code | `workloads/build_w4.py`, `workloads/w4_session.py`, test `workloads/test_w4_session.py` |
| Lệnh | `.venv/bin/python workloads/build_w4.py pool` (27 s) · `... profile` · `... scenario --segment <peak\|peak2\|normal> --k <k> [--seed 1]` |
| Kho nội dung | chat (tin nhắn người dùng ShareGPT, tiếng Anh, ≤ 1600 token) **239,548**; tài liệu RAGBench train+val (≥ 50 token) **18,921**. sha256 trong `workloads/manifest_w4.json` |
| Tái lập | chạy `scenario --segment peak --k 10` hai lần → cùng sha256 `a200a785…` |

**Khung giờ** (`profile`, req TB/giờ của phần hội thoại theo giờ trong ngày): peak **16h** (208.3), peak2 **10h** (180.1),
normal **13h** (92.3), quiet 5h (4.0).

**Kiểm tra ghép prompt** (`test_w4_session.py`): lượt 2–5 dùng lại 133/137, 453/457, 773/777, 1093/1097 token prompt lượt trước
(chỉ khác 4 token khối `<think>` rỗng của template Qwen3); ngân sách 708 token → bỏ 3 cặp cũ, còn 656 token.

**Kịch bản** (cửa sổ 10 phút khởi động + 30 phút đo, seed 1; "kỳ vọng" = k × req TB/giờ của trace ở giờ đó):

| Kịch bản | Phiên (đang dở) | Request đo được | req/giờ đo / kỳ vọng | Phiên mở p50 | prompt p50 / p90 / p99 | <1k / 1–2k / 2–4k / 4–8k / ≥8k (%) | max_tokens p50 / p90 |
|---|---|---|---|---|---|---|---|
| peak k=1 | 17 (10) | 32 | 64 / 208 | 8 | 831 / 2766 / 3754 | 62.5 / 18.8 / 18.8 / 0 / 0 | 170 / 658 |
| peak k=10 | 554 (218) | 957 | 1,914 / 2,083 | 199 | 404 / 2217 / 5306 | 71.8 / 16.2 / 9.5 / 2.2 / 0.3 | 174 / 700 |
| peak k=40 | 2,203 (940) | 4,065 | 8,130 / 8,334 | 828 | 506 / 2449 / 6272 | 68.2 / 17.4 / 11.6 / 2.0 / 0.7 | 176 / 679 |
| peak k=110 | 5,997 (2,501) | 11,065 | 22,130 / 22,917 | 2,277 | 518 / 2365 / 5368 | 68.1 / 18.3 / 11.4 / 1.8 / 0.4 | 185 / 678 |
| peak2 k=10 | 458 (141) | 918 | 1,836 / 1,801 | 158 | 384 / 1701 / 6565 | 78.1 / 14.7 / 5.3 / 1.2 / 0.7 | 133 / 509 |
| normal k=10 | 196 (82) | 401 | 802 / 923 | 69 | 501 / 2439 / 4948 | 64.1 / 17.7 / 15.5 / 2.5 / 0.2 | 190 / 662 |

- k=1 thấp hơn kỳ vọng vì một ngày dao động lớn (cuối tuần, ngày vắng); từ k=10 sai lệch ≤ 13%.
- Lần sinh đầu chỉ lấy phiên **bắt đầu** trong cửa sổ → peak k=10 chỉ 1,086 req/giờ (52% kỳ vọng): thiếu người đang chat dở
  (think time dài). Đã sửa: đưa cả phiên đang dở vào (cột "đang dở").
- Bị chặn bởi `max_model_len` 16384: 8 request ở k=110. Độ dài tin nhắn mới phải lấy ngẫu nhiên (client gốc đã cắt lịch sử): 3,864 / 11,065 ở k=110.

## 2026-10-12 — Phân tích trace BurstGPT_3 (thiết kế W4, local, không GPU)

| Trường | Giá trị |
|---|---|
| Dữ liệu | `BurstGPT_3.csv`, release v2.0, CC-BY-4.0, sha256 `2299986a…a43b8f` (đầy đủ trong script); 5,344,021 dòng, 110 ngày, Azure OpenAI (ChatGPT/GPT-4) |
| Phần mềm | pandas 3.0.6, numpy 2.5.4 (`.venv`) |
| Lệnh | `.venv/bin/python workloads/analyze_burstgpt.py` |

Chỉ BurstGPT_3 có `Session ID` + `Elapsed time`; BurstGPT_1/2 (bản v1.1) không có.

**Tổng quan:** API log (chương trình gọi, không có phiên) **95.6%**; Conversation log (người chat) 4.4% = 233,617 request.
Thất bại (resp = 0) 7.3%. Mọi request: req p50/p90/p99 = 340 / 884 / 3482, resp = 14 / 144 / 912.

**Phiên (Conversation log):**

| Chỉ số | Giá trị |
|---|---|
| Số phiên | 55,920; lượt/phiên p50/p75/p90/p99 = 2 / 4 / 9 / 31, TB 4.18; phiên nhiều lượt 64.5% |
| Độ dài req (token) | <1k 67.4% · 1–2k 18.9% · 2–4k 11.2% · 4–8k 2.1% · ≥8k 0.4% |
| Độ dài resp (token) | p25/p50/p75/p90/p99 = 57 / 190 / 414 / 671 / 1068 |
| req theo lượt, p50 (p90) | 1: 44 (311) · 2: 314 (1013) · 3: 600 (1774) · 4: 894 (2522) · 5: 1188 (3214) · 6–20: ~1200–1260 (~3100–3200) |
| Lịch sử hội thoại | lượt sau dài hơn lượt trước 77.9%; corr(req, req_trước + resp_trước) = **0.905** → **`Request tokens` gồm lịch sử**; dừng tăng từ lượt ~5 (p50 ~1.2k) → nghi phía client cắt bớt lịch sử |
| Think time (s) = ts − (ts_trước + elapsed_trước) | p10/p25/p50/p75/p90/p95 = 17 / 38 / **124** / 526 / 2330 / 7232 |

**Tải hội thoại theo thời gian:**

| Chỉ số | Giá trị |
|---|---|
| req/giờ | p10/p50/p90/p95/p99 = 3 / 59 / 216 / **293** / 413; max 509; p95/p50 = 5.0 |
| req/phút | p50/p90/p99 = 1 / 4 / 9; max 28 |
| Theo giờ trong ngày (req TB/giờ) | 3–7h: 4–10 · **9–11h: 116–182** · **12h: 69** · **14–17h: 168–210** · 18–23h: 76–117 → nhịp giờ hành chính, có nghỉ trưa |
| % request ≥ 2k token mỗi giờ (1432 giờ có ≥50 req) | p10/p50/p90 = 5 / 13 / 25 % → tỉ lệ dài/ngắn đổi mạnh theo giờ |
| Giờ đại diện | p10 ≈ 3 req/giờ (#27); p50 ≈ 59 (#45); p95 ≈ 293 (#1529, ngày 63, 17h) |

**API log:** req/giờ p10/p50/p95/p99 = 0 / 159 / 9306 / 22052, max 45,454 → đột biến cực mạnh (max/p50 ~290×), không có phiên.

## 2026-10-12 — P01: cài vLLM 0.30.0 và chạy thử Qwen3-30B-A3B-GPTQ-Int4 trên L4

Báo cáo: [gd1-vllm-baseline/01-cai-dat-chay-thu/bao-cao.md](gd1-vllm-baseline/01-cai-dat-chay-thu/bao-cao.md).
Kết quả thô: `results/gd1/01-cai-dat-chay-thu/20261012-p01/`.

### Máy & phần mềm

| Trường | Giá trị |
|---|---|
| Thời gian chạy (tính tiền) | 14:46:51 → 15:03:58 (+07), ~17 phút |
| Instance / offer | 55518236 / 49498406 (đã destroy) |
| Host / machine | **host 506934 / machine 116596** — Czechia (khác machine 77226 lần test trước) |
| Giá | $0.2833/h tổng (disk 60 GB) |
| Image | `nvidia/cuda:13.0.2-devel-ubuntu22.04` (nvcc 13.0.88, ncu 2025.3.1) |
| Driver | 595.84 |
| CPU / RAM | Xeon Silver 4114, 30 luồng thấy được / 94 GB |
| PCIe | Gen3 x16 |
| Phần mềm | vLLM **0.30.0**, torch 2.13.0 (cu130), flashinfer-python 0.6.18.post1, Python 3.12.14 (uv 0.12.19) |
| Model | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441` (desc_act=false, group 128, 15.77 GiB) |

### Checklist máy

| Phép đo | Kết quả |
|---|---|
| Băng thông VRAM (cudaMemcpy D2D 1 GiB × 10) | **230.8 GB/s** (lần trước machine 77226: 231.2) |
| `ncu` | `ERR_NVGPUCTRPERM` — host 506934 vẫn chặn |
| `nsys` | không có trong image |

### Thời gian (giây)

| Bước | Thời gian |
|---|---|
| Cài uv | 1 |
| Tải model 16 GB từ HF (song song với cài vLLM) | 168 (~95 MB/s) |
| Cài vLLM + torch (uv) | 193 tính từ đầu |
| **Setup tổng** | **193** |
| Khởi động lần 1 (mặc định, compile từ đầu) — lỗi | nạp weight 47, torch.compile 62, capture CUDA graph 12 → lỗi KV |
| Khởi động thành công (`--max-model-len 16384`, đã có cache compile) | **235** tới `/health` OK: nạp weight 53, torch.compile 1.2 (cache), capture CUDA graph 11 + 16, khoảng trống ~110 (nghi FlashInfer JIT biên dịch kernel sampling) |

### Cấu hình mặc định vLLM 0.30.0 tự chọn

| Mục | Giá trị |
|---|---|
| Kernel linear GPTQ | `MarlinLinearKernel` |
| MoE | backend `MARLIN` WNA16, `MarlinExperts` (không có dòng "default MoE config" — đường Marlin không dùng config JSON Triton) |
| Attention | `FLASH_ATTN` (FlashAttention 2), chọn từ FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION |
| Sampling top-k/top-p | FlashInfer |
| dtype / KV dtype | float16 / auto (=float16) |
| Prefix caching / chunked prefill | bật / bật |
| torch.compile / CUDA graph | VLLM_COMPILE / FULL_AND_PIECEWISE, max capture size 512 |
| Model runner | V2 |
| `gpu_memory_utilization` | 0.92 |
| `max_model_len` | 40960 (theo config model) → **không khởi động được**, phải đặt 16384 |
| max_num_batched_tokens | 2048 (suy từ "compile range (1, 2048)", chưa thấy in trực tiếp) |
| Sampling mặc định | lấy từ `generation_config.json` của model: temperature 0.6, top_k 20, top_p 0.95 |

### Bộ nhớ (GiB, lần khởi động thành công)

| Phần | GiB |
|---|---|
| VRAM khả dụng / vLLM được dùng (0.92) | 22.04 / 20.27 |
| Weight + non-torch | 16.1 (weight 15.68) |
| Activation đỉnh | 0.76 |
| CUDA graph | 0.33 thực tế (ước tính 0.64) |
| **KV cache** | **3.41 → 37,264 token** (≈ 96 KiB/token, khớp 2×48×4×128×2 byte) |
| nvidia-smi khi chạy | 21,057 / 23,034 MiB |

Lần khởi động đầu (compile từ đầu, max_model_len 40960): KV chỉ **2.63 GiB → 28,688 token**. Chênh 0.78 GiB chưa rõ
nguyên nhân (nghi bộ nhớ do torch.compile chiếm khi compile lần đầu) — kiểm ở P03.

### Smoke test (1 request mỗi lần, chỉ để kiểm tra đúng/sai — không phải số benchmark)

| Request | Prompt tok | Output tok | Thời gian | Ghi chú |
|---|---|---|---|---|
| 3 câu hỏi ngắn, thinking **bật** | 22–36 | 422 / 2345 / 1018 | 4.6 / 26.8 / 11.1 s | ~88–92 tok/s |
| cùng câu, thinking **tắt** | 26–40 | 39 / 202 / 54 | 0.5 / 2.2 / 0.6 s | output ngắn hơn 10–19× |
| W2 `techqa_TRAIN_Q041` (completions, greedy) | **2997** (= số đếm P02) | 150 | 2.3 s | trả lời đúng định dạng system prompt |
| W3 hợp đồng dài nhất (completions, greedy) | **7879** (= số đếm P02) | 265 | 5.3 s | tóm tắt đúng 8 mục |

### Chi phí

Theo hoá đơn Vast (`vastai show invoices`): GPU 0.346 h = $0.092 · ổ đĩa 0.394 h = $0.007 · **tải xuống 21 GB = $0.027**
→ **tổng $0.126 ≈ 3,300 VND**. Vast tính GPU 0.346 h (~21 phút), nhiều hơn ~17 phút đo từ lúc trạng thái `running`
→ một phần thời gian `loading` cũng bị tính. Phí tải xuống (model 16 GB + gói Python ~5 GB) ≈ $0.027 **mỗi phiên máy mới**.

### Lệnh tái lập

```bash
vastai create instance <OFFER> --image nvidia/cuda:13.0.2-devel-ubuntu22.04 --disk 60 --ssh --direct
scp scripts/{setup_node.sh,serve_default.sh,smoke_test.py} <smoke_samples.jsonl> root@<IP>:/root/work/
bash /root/work/setup_node.sh /root/results/p01
bash /root/work/serve_default.sh /root/results/p01/server_default.log                       # mặc định → lỗi KV
bash /root/work/serve_default.sh /root/results/p01/server.log --max-model-len 16384          # chạy được
/root/venv/bin/python /root/work/smoke_test.py smoke_samples.jsonl /root/results/p01/smoke.json
```

---

## 2026-10-12 — P02: bộ workload W1/W2/W3 (local, không GPU)

Báo cáo: [gd1-vllm-baseline/02-bo-sinh-workload/bao-cao.md](gd1-vllm-baseline/02-bo-sinh-workload/bao-cao.md).
Chi tiết + sha256: `workloads/manifest.json`. Đếm bằng tokenizer Qwen3, **tính cả chat template**.

| Workload | Số mẫu | Prompt token min / TB / p50 / p95 / max | Output token min / TB / p50 / p95 / max | Phần đầu chung | Nguồn |
|---|---|---|---|---|---|
| W1 | 2000 | 200 / 407 / 368 / 726 / 800 | 200 / 333 / 272 / 600 / 600 | 3 token | ShareGPT (lượt hỏi đầu, ≥95% ký tự ASCII) |
| W2 | 1253 | 2000 / 2858 / 2785 / 3887 / 3999 | 150 / 174 / 150 / 287 / 400 | **622 token** | RAGBench train+val: techqa 704, finqa 326, delucionqa 127, tatqa 80, emanual 16 |
| W3 | 343 | 4008 / 5617 / 5577 / 7483 / 7879 | 250 / 299 / 301 / 344 / 350 | 95 token | CUAD (mọi split), 343 hợp đồng khác nhau, 286 bị cắt tại ranh giới đoạn |

Build lại lần 2 → sha256 trùng (tái lập OK). Thời gian build ~100 s trên laptop.

```bash
.venv/bin/python workloads/build.py           # build
.venv/bin/python workloads/build.py --check   # kiểm tra tái lập
```

---

## 2026-10-12 — Test máy L4 đầu tiên (Vast.ai)

### Thông tin máy

| Trường | Giá trị |
|---|---|
| Thời gian | 13:31 → 13:34 (+07), ~4 phút chạy |
| Instance / offer | 55510860 / offer 35835963 (đã destroy) |
| **Host / machine** | **host 506934 / machine 77226** — Czechia, CZ, verified, reliability 99.75% |
| Giá | GPU $0.3333/h, tổng $0.3407/h (gồm disk 40GB) |
| Phí ổ đĩa | $0.1333 / GB / tháng |
| Phí băng thông | tải lên $0.0013/GB, tải xuống $0.0013/GB (= $1.3/TB) |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu22.04`, launch mode SSH direct |
| GPU | NVIDIA L4, compute cap 8.9, **23034 MiB** khả dụng, power limit 72 W |
| Driver | 615.71.09 (CUDA tối đa 13.4) |
| CPU | Intel Xeon Silver 4114, 40 luồng thấy được, **20 vCPU được cấp** |
| RAM | 128 GB trên host (container thấy 125 GB) |
| Mainboard | 0W23H8 (Dell) |
| **PCIe** | **Gen3 x16**, đo được host↔GPU **8.4 GB/s** (L4 hỗ trợ Gen4 — host này giới hạn) |
| Ổ đĩa | INTEL SSDSC2BA40 (SATA SSD), 2277 MB/s theo Vast |
| Mạng (Vast đo) | down 943.7 Mbps / up 97.1 Mbps |
| OS host | Ubuntu 24.04 |
| Vast DLPerf | 25.3 · total FLOPS 24.2 TFLOPS (FP32) |

### Kết quả đo

| Phép đo | Kết quả | Ghi chú |
|---|---|---|
| Băng thông VRAM (cudaMemcpy D2D 1 GiB × 10, đếm đọc+ghi) | **231.2 GB/s** | ≈ 77% của 300 GB/s lý thuyết. Vast tự đo: 220.7 GB/s |
| Tải từ HuggingFace (500 MB đầu `model.safetensors`) | **80.9 MB/s** (6.48 s) | → model ~17 GB tải ≈ 3.5 phút |
| `nvcc` | 12.8 (`cuda_12.8.r12.8/compiler.35583870_0`) | có sẵn trong image devel |
| `ncu` | Nsight Compute 2025.1.1 có sẵn, **chạy lỗi `ERR_NVGPUCTRPERM`** | host chặn performance counter → xem `challenges.md` |
| `nsys` | **Không có** trong `/usr/local/cuda/bin` của image này | cần cài thêm nếu dùng |
| Model `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | Tồn tại trên HF, 11 file, 1 file `model.safetensors` | |

### Chi phí

- Số dư trước: $10.0000 → sau: $9.99615 → **tốn $0.0038 (~100 VND)**

### Lệnh tái lập

```bash
# Thông tin máy
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader
df -h / ; free -g ; nproc ; nvcc --version
vastai show instance <ID> --raw          # metadata host/machine/pcie/giá

# Tốc độ tải HF
curl -sL -r 0-524287999 -o /dev/null -w 'speed=%{speed_download}B/s\n' \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/resolve/main/model.safetensors

# Băng thông VRAM
cat > /tmp/bw.cu <<'CU'
#include <cstdio>
int main(){ size_t n=1ull<<30; void *a,*b; cudaMalloc(&a,n); cudaMalloc(&b,n);
 cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
 cudaMemcpy(b,a,n,cudaMemcpyDeviceToDevice); cudaEventRecord(s);
 for(int i=0;i<10;i++) cudaMemcpy(b,a,n,cudaMemcpyDeviceToDevice);
 cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e);
 printf("D2D copy: %.1f GB/s (read+write)\n", 2.0*10*n/ms/1e6); }
CU
nvcc -O2 -arch=sm_89 /tmp/bw.cu -o /tmp/bw && /tmp/bw

# Thử ncu
cat > /tmp/t.cu <<'CU'
__global__ void k(float *x){ x[threadIdx.x] += 1.f; }
int main(){ float *x; cudaMalloc(&x, 1024); k<<<1,256>>>(x); cudaDeviceSynchronize(); return 0; }
CU
nvcc -arch=sm_89 /tmp/t.cu -o /tmp/t && ncu --metrics gpu__time_duration.sum,dram__bytes.sum /tmp/t
```

---

## 2026-10-12 — Ảnh chụp thị trường Vast.ai (~13:00 +07)

### L4 dạng container (1 GPU)

| Offer ID | Nơi | $/h tổng | Reliability | Disk max | Tải xuống $/TB | Disk $/GB/tháng | Host |
|---|---|---|---|---|---|---|---|
| 49498406 | Czechia | 0.268 | 99.8% | 147 GB | 1.3 | 0.20 | — (đã hết khi thuê) |
| 49883396 | Utah, US | 0.323 | 99.0–99.2% | 182 GB | 3.9 | 0.47 | 323419 |
| 35835963 | Czechia | 0.334 | 99.8% | 354 GB | 1.3 | 0.13 | 506934 ← đã test |
| 36579964 | Czechia | 0.334 | 99.9% | 522 GB | 1.3 | 0.13 | 506934 |
| 54285094 | Czechia | 0.335 | 99.2% | — | 2.6 | 0.20 | 506934 |
| 55313287 | Utah, US | 0.337 | 92.8% | 50 GB | 3.9 | 0.60 | — |

→ Nguồn cung L4 rất mỏng (~5 máy), phần lớn từ **cùng host 506934** — host này **chặn ncu**.

### Máy dạng VM (`vms_enabled=true`, 1 GPU)

- **L4 dạng VM: 0 máy.**
- Ada (sm_89) dạng VM: 13 máy

| GPU | Số máy | Giá thấp nhất | Nơi | Offer |
|---|---|---|---|---|
| RTX 4000 Ada (AD104, cùng die với L4) | 1 | $0.201/h | Hungary | 36173713 |
| RTX 4060 Ti | 2 | $0.225/h | Quebec, CA | 49744628 |
| RTX 4080 | 1 | $0.243/h | Florida, US | 54308059 |
| RTX 4090 | 9 | $0.415/h | Denmark | 51927459 |

- Tổng VM mọi loại GPU: 32 máy (nhiều nhất RTX 4090: 9).

### Lệnh tái lập

```bash
vastai search offers 'gpu_name=L4 num_gpus=1 verified=true reliability>0.98 disk_space>=40' -o 'dph'
vastai search offers 'gpu_name=L4 vms_enabled=true' -o 'dph'
vastai search offers 'vms_enabled=true compute_cap=890 num_gpus=1' -o 'dph'
```

---

## Tham khảo giá ngoài Vast (tra cứu 2026-10-12, nguồn bên thứ ba)

| Nhà cung cấp | Máy | Giá |
|---|---|---|
| AWS | g6.xlarge (1×L4, VM) | spot ~$0.36–0.55/h (interruption 10–15%), on-demand $0.80/h (us-east-1) |
| GCP | g2-standard-4 (1×L4, VM) | on-demand ~$0.71/h, spot ~$0.62/h (us-central1); free trial **không** dùng được GPU |
| RunPod | L4 | Community ~$0.44/h, Secure ~$0.49/h |
| Tỷ giá dùng để tính | | ~26,200 VND/USD (tháng 8/2026) |
