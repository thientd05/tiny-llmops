# Báo cáo P03 — Công cụ chạy benchmark (harness) + kiểm tra rút gọn trên L4

> Ngày làm: 2026-10-12 → 13 · Máy: host 506934 / machine 116596 (cùng máy P01), instance 55583011 · vLLM 0.30.0 ·
> Giờ GPU phần P03: ~0.84 h (23:28:55 → 00:19:15, ≈ $0.24 + phí tải về ≈ 7,000 VND; hoá đơn tách theo thời gian, xem mục 3.7)
> Số liệu: [measurements.md](../../measurements.md) mục "P03". Kết quả thô: `results/gd1/03-cong-cu-benchmark/20261012-p03/`.

## 1. Tóm tắt

- Đã có harness chạy tiếp được.
  - `bench/sweep.py` có hai chế độ: grid (danh sách điểm cho trước) và search (tìm goodput).
  - `bench/summarize.py` tổng hợp kết quả.
  - `bench/session_client.py` là client phiên cho W4, có chế độ closed-loop để kiểm chứng.
  - `bench/fake_server.py` là server giả để test ở laptop.
- Ở laptop:
  - Logic tìm goodput đúng với mọi điểm gãy K = 1…1099.
  - Trên server giả, cả ba trường hợp đều tìm đúng khoảng: K = 100 cho 96 (104 trượt), K = 5 cho 4 (6 trượt), K = 40 cho 40 (44 trượt).
  - Ngắt hai lần (Ctrl+C rồi kill -9), chạy lại cùng lệnh: harness chỉ đo phần còn thiếu.
- Trên L4: bốn kiểm tra (a)–(d) xong.
  - (a) Token vào khớp **từng request** với P02.
  - (b) TTFT phía client khớp phía server, lệch ~25 ms.
  - (c) Reset prefix cache có tác dụng: tỉ lệ trúng 21.6%, đúng bằng phần system prompt chung.
  - (d) Nhiễu throughput 1.2% và TPOT 3.7%, đạt mục tiêu < 5%. **TTFT p95 nhiễu ~19%, không đạt 5%.** Đã chọn cách xử lý ([challenges](../../challenges.md) #5).
- **Giả thuyết KV của P01 đúng.** Lần khởi động compile từ đầu cho KV **28,704 token**. Các lần sau dùng cache compile cho **37,264 token**. Chênh lệch này đủ đổi kết luận đạt/trượt. Harness đã tự xử lý.
- Client phiên tự viết lệch ≤ 3.7% so với `vllm bench serve` ở mọi chỉ số (gộp 200 request mỗi bên). Đạt yêu cầu dùng cho P05.
- Ước lượng P04: **~5.1 giờ GPU (≈ 38k VND)**, trường hợp xấu ~6.1 giờ, thấp hơn mức 6.5 giờ của plan.

## 2. Đã làm gì và vì sao

### 2.1 Kiểm tra `vllm bench serve` 0.30.0 trước khi viết code

Plan ghi một số cờ tạm (`--save-detailed`, `--goodput`, ...). Tôi kiểm theo hai nguồn:
- `--help` đã lưu từ máy L4 ở P01 (`results/gd1/01-.../vllm_bench_serve_help_all.txt`). Bộ cờ trùng 100% với bản cài ở laptop.
- Mã nguồn bản 0.30.0 (`vllm/benchmarks/serve.py`, `datasets/datasets.py`, `lib/endpoint_request_func.py`).

Kết quả và những điểm khác với plan:

| Điều cần kiểm | Thực tế ở 0.30.0 | Ảnh hưởng |
|---|---|---|
| Lưu số liệu từng request | `--save-result --save-detailed`: ghi `ttfts`, `itls`, `latencies`, `start_times`, `input_lens`, `output_lens`, `errors`, `generated_texts` | Dùng đúng tên này |
| `--goodput`, `--max-concurrency` | Có, như plan | — |
| Dataset custom | JSONL có trường `prompt`, `output_tokens`. **`--custom-output-len` mặc định 256 và ghi đè** số token ra của dataset | Phải đặt `--custom-output-len -1`, nếu không mọi request ra 256 token |
| Lặp mẫu | Thiếu mẫu thì tự lặp (oversample) | Đặt `--no-oversample`; harness tự báo lỗi trước khi chạy nếu `max(100, 4N)` > số mẫu |
| Thứ tự mẫu | `random.seed(seed)` rồi xáo trộn cả file, lấy `num_prompts` mẫu đầu | Lần lặp r dùng seed `1000 + r` → mỗi lần lặp một tập mẫu khác |
| `--num-warmups` | Gửi lại **chính request đầu tiên** của tập đo | Không dùng: request đó sẽ trúng prefix cache trong lần đo. Harness tự warmup bằng 4 mẫu, rồi gọi `/reset_prefix_cache` |
| `temperature` | Không còn mặc định = 0 | Đặt `--temperature 0` |
| `/reset_prefix_cache` | `POST`, trả `{"success": bool}`; `false` nếu còn request đang giữ block | Harness thử lại tới khi `success` |
| Tên metric `/metrics` | `vllm:num_requests_running`, `..._waiting`, `vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`, `vllm:prefix_cache_{queries,hits}_total`, histogram `vllm:time_to_first_token_seconds`, `vllm:request_queue_time_seconds` | Dùng các tên này |

**Khác plan:** plan ghi `--backend openai-chat`, nhưng tôi dùng `--backend openai --endpoint /v1/completions --skip-chat-template`. Lý do: P02 đã dựng sẵn toàn bộ prompt theo chat template Qwen3, gồm system prompt và khối thinking rỗng. Còn `openai-chat` với dataset custom chỉ gửi một tin nhắn vai trò user, nên không đặt được system prompt của W2. Hai cách cho cùng token vào server (đã kiểm ở (a)).

### 2.2 Harness

**Một đơn vị đo** = workload × N × lần lặp (ví dụ `w2-c16-r2`), nằm ở `<out>/units/<id>/`. Trình tự:

1. Server chưa chạy hoặc đã chết thì bật lên. Sang lần lặp mới thì khởi động lại, tuỳ cấu hình.
2. Warmup: 4 request lấy từ đầu file workload, mỗi request ra 16 token.
3. `POST /reset_prefix_cache`, rồi chụp `/metrics` trước khi đo.
4. Bật hai luồng ghi mỗi 1 s: `/metrics` (request đang chạy, đang chờ, % KV, preemption, prefix cache) và `nvidia-smi` (bộ nhớ, công suất, xung, nhiệt độ).
5. Chạy client: `vllm bench serve` hoặc `session_client.py closed`.
6. Chụp `/metrics` sau khi đo. Nén kết quả từng request thành `bench.json.gz` (giữ 200 ký tự đầu của câu trả lời). Tính chỉ số theo hai cửa sổ ("all" và "steady"). Ghi `meta.json` gồm máy, phiên bản, lệnh, git commit và KV của lần khởi động.
7. Ghi `done.json` **cuối cùng**. File này chứa sha256 của cấu hình đơn vị: workload và sha256 của nó, N, lần lặp, số request, seed, client, tham số lấy mẫu, warmup, lệnh server, phiên bản phần mềm.

**Chạy tiếp được** ([thi-nghiem-chay-tiep.md](../../thi-nghiem-chay-tiep.md)):
- Đơn vị có `done.json` với hash đúng thì bỏ qua.
- Thư mục có mà không có `done.json` hợp lệ thì xoá và đo lại.
- Mọi file đều ghi ra `.tmp` trước rồi mới đổi tên.

**Chế độ search:**
- Mỗi đợt tính lại hoàn toàn từ các `done.json` đã có. Trong `bench/common.py`, hàm `search_step` là hàm thuần, không có trạng thái ẩn, nên chạy lại cùng lệnh sẽ ra đúng điểm tiếp theo.
- Các bước theo docs/workloads.md: lưới gốc, nhân đôi lên (trần 1024 và trần số mẫu), chia đôi xuống (sàn 2), tìm nhị phân (mức chẵn) tới `hi − lo ≤ max(2, 10% × lo)`.
- Kiểm tra đơn điệu: nếu có mức đạt nằm trên mức trượt thì đo thêm cho đủ 6 lần lặp ở hai mức đó. Vẫn ngược thì dừng với trạng thái `non_monotonic`.
- Trong một đợt, thứ tự chạy xen kẽ theo lần lặp (r1 mọi điểm, rồi r2, rồi r3).

**Trần số mẫu:** N tối đa = ⌊số mẫu / 4⌋ làm tròn xuống số chẵn: W1 500, W2 312, W3 84. Khi mở rộng lên, harness thử đúng mức trần rồi dừng với trạng thái `sample_cap`. Không bao giờ lặp mẫu.

**`--status`:** in số đơn vị đã xong, các mức của từng workload, đạt/trượt, và ước tính giờ GPU còn lại (từ số giây/request đã đo).

**`summarize.py`** tính lại mọi thứ từ số liệu từng request:
- goodput dạng khoảng theo SLO chính, SLO phụ, và độ nhạy với ngưỡng TTFT 2 / 5 / 10 / 20 s, cho cả hai cửa sổ;
- nhiễu giữa các lần lặp;
- TTFT phía client so với phía server;
- so sánh hai cửa sổ;
- so sánh hai client;
- thời gian đo, công suất và xung GPU.

p95 tính bằng `numpy.percentile`, cùng cách nội suy với vLLM. So với số vLLM tự in, lệch < 0.002%.

### 2.3 Client phiên (`bench/session_client.py`)

- Gửi `/v1/completions` có stream. Payload và cách đo giống hệt `vllm bench serve` 0.30.0: TTFT là thời điểm nhận chunk đầu có `choices`; số token ra lấy từ `usage`; TPOT = (latency − TTFT) / (out − 1).
- **Chế độ `closed`** (để kiểm chứng): chọn mẫu đúng như vLLM (`random.seed` + xáo trộn), N luồng gửi liên tục.
- **Chế độ `scenario`** (cho P05): đọc file kịch bản `build_w4.py`.
  - Lượt đầu của mỗi phiên gửi đúng `start_s`.
  - Lượt sau gửi khi lượt trước **nhận xong** + `think_s`.
  - Prompt = lịch sử thật (gồm câu trả lời server vừa sinh) + tin nhắn mới, ghép bằng `workloads/w4_session.assemble`.
  - Chỉ request gửi trong cửa sổ đo được đánh dấu `in_window`. Hết cửa sổ thì không gửi lượt mới.

### 2.4 Test ở laptop (0 giờ GPU)

`vllm bench serve` 0.30.0 được cài ở laptop (bản CPU, chỉ để gửi tải), nên test đi qua đúng cờ, đúng định dạng file và đúng đường xử lý kết quả như trên L4.

| Test | Kết quả |
|---|---|
| `bench/test_search.py`: mức đạt ⇔ N ≤ K, K = 1…1099 | Luôn ra đúng khoảng; nhiều nhất 12 mức đo |
| Có nhiễu (một lần lặp trượt nhầm) / ngược thật (16 trượt, 32 đạt) | Vẫn ra 40 (44 trượt) / dừng `non_monotonic` |
| Trần mẫu (1253 mẫu, K = 500) | Dừng ở `sample_cap`, goodput ≥ 312 |
| Khởi đầu nóng [40, 44], K = 70 | 70 (76 trượt) sau 6 mức |
| `test_fake.sh k100 / k5 / k40` (server giả, điểm gãy biết trước) | 96 (104 trượt) · 4 (6 trượt) · 40 (44 trượt) |
| `resume`: 3 lần lặp, Ctrl+C sau 75 s, kill -9 sau 150 s, chạy lại | Đơn vị dở bị xoá và đo lại; đơn vị đã xong không bị đo lại (so mtime `done.json`); kết quả 40 (44 trượt) |
| `config`: đổi số request tối thiểu 100 → 120 | Mọi đơn vị thành "chưa xong" (hash khác) |
| `closed`: hai client trên server giả kiểu hàng đợi | Lệch 0.3–4.5%, trừ TTFT p95 một lần (8%), nằm trong mức dao động giữa hai lần của chính vLLM (8.5%) |
| `scenario`: phát lại W4 peak k=10, co thời gian 100 lần | 1221 request, 957 trong cửa sổ (đúng bằng số lượt của trace), 0 lỗi, 0 lượt gửi sớm hơn "nhận xong + think" |

### 2.5 Trên L4: smoke

Cấu hình `bench/configs/smoke.yaml`, chế độ grid. Có hai điểm khác plan:
- W2 / 16 đo **3 lần** bằng `vllm bench serve` và **3 lần** bằng client tự viết, xen kẽ nhau. Plan chỉ ghi 2 lần. Mục đích là kiểm chứng client trong cùng phiên, như yêu cầu.
- Khởi động lại server giữa các lần lặp (`restart: per_repeat`), để kiểm giả thuyết KV và để nhiễu do khởi động lại cũng nằm trong số đo.

Các điểm còn lại theo plan: W2 / 8, W2 / 32, W1 / 16, W3 / 8, mỗi điểm 1 lần.

## 3. Kết quả

### 3.1 Máy

Băng thông VRAM 231.2 GB/s (giống P01). `ncu` bị chặn. PCIe Gen3. Giới hạn công suất 72 W. Setup máy mới 186 s.

### 3.2 KV cache theo lần khởi động

| Lần | Loại | torch.compile | init engine | Sẵn sàng sau | KV cache |
|---|---|---|---|---|---|
| 1 | Máy mới, compile từ đầu | 71.1 s | 244 s | 343 s | **2.63 GiB = 28,704 token** |
| 2 | Khởi động lại (có cache compile) | 0.6 s | 52 s | 142 s | **3.41 GiB = 37,264 token** |
| 3 | Khởi động lại (có cache compile) | 1.2 s | 39 s | 124 s | 3.41 GiB = 37,264 token |

### 3.3 Smoke

Cửa sổ "all". r1 chạy trên lần khởi động 1 (KV 28,704); r2 và r3 chạy trên lần 2, 3 (KV 37,264).

| Điểm | KV | TTFT p50 / p95 | TPOT p50 / p95 | token/s ra | token/s vào | Chạy tối đa | Chờ tối đa | Preemption | SLO chính |
|---|---|---|---|---|---|---|---|---|---|
| W1 / 16 r1 | 28.7k | 0.22 / 1.09 s | 46.8 / 48.7 ms | 322 | 399 | 16 | 0 | 0 | đạt |
| W2 / 8 r1 | 28.7k | 0.71 / 2.35 s | 48.9 / 57.0 ms | 146 | 2,378 | 8 | 6 | 0 | đạt |
| W2 / 16 r1 | 28.7k | 4.50 / 9.91 s | 64.6 / 75.7 ms | 161 | 2,611 | 13 | 13 | 9 | đạt (sát) |
| W2 / 16 r2 | 37.3k | 1.41 / 4.58 s | 82.1 / 92.6 ms | 170 | 2,879 | 16 | 12 | 7 | đạt |
| W2 / 16 r3 | 37.3k | 2.28 / 5.55 s | 81.5 / 89.2 ms | 168 | 2,743 | 16 | 13 | 6 | đạt |
| W2 / 32 r1 | 28.7k | 22.1 / 25.1 s | 65.9 / 75.0 ms | 160 | 2,573 | 13 | 29 | 9 | trượt (TTFT) |
| W3 / 8 r1 | 28.7k | 12.9 / 15.7 s | 43.0 / 48.5 ms | 96 | 1,840 | 6 | 6 | 5 | trượt (TTFT) |

Trong mọi lần đo, KV dùng tối đa 100% (trừ W1 41%, W2/8 86%). GPU chạm trần 73–74.5 W, xung SM thấp nhất 975–1,140 MHz (tối đa 2,040).

### 3.4 Kiểm tra (a)–(d)

| Kiểm tra | Cách làm | Kết quả |
|---|---|---|
| (a) token vào/ra | Lặp lại cách chọn mẫu (seed, xáo trộn) trên file P02, so `input_lens` (server báo) với `prompt_tokens` P02 | **Khớp từng request** ở cả 6 đơn vị đã kiểm; token ra khớp tổng (ví dụ W3/8: 29,671 / 29,671) |
| (b) TTFT client vs server | TTFT trung bình client so với histogram `vllm:time_to_first_token_seconds` (hiệu sau − trước) | Client cao hơn **17–33 ms** (0.2–6.5%). Phần chênh là đường HTTP + tách luồng SSE, cố định, không đổi theo tải |
| (c) reset prefix cache | `prefix_cache_hits / queries` trong lúc đo | W2 **21.6%** ≈ 622 / 2,858 = 21.8% (chỉ system prompt chung); W3 1.6% ≈ 95 / 5,617; W1 0% |
| (d) nhiễu | W2 / 16, r2 vs r3 (cùng KV, có khởi động lại giữa hai lần) | tok/s ra **1.2%**, TPOT p95 **3.7%**, E2EL p50 9%, **TTFT p95 19%** (vllm) / 0.9% (client tự viết) |

### 3.5 Client tự viết vs `vllm bench serve`

W2 / 16, cùng mẫu (seed), chạy xen kẽ trên cùng server.

| | Lần r2 | Lần r3 | **Gộp r2 + r3 (200 request mỗi bên)** |
|---|---|---|---|
| tok/s ra | −0.4% | −1.1% | −1% |
| TTFT p50 | +14.3% | +6.4% | +5.3% |
| TTFT p95 | +23.0% | +0.5% | **+3.7%** |
| TPOT p95 | +1.1% | +0.4% | +0.4% |
| E2EL p50 | −0.2% | +1.6% | −0.5% |

### 3.6 Cửa sổ "all" và "steady"

"steady" bỏ N request gửi đầu và N request gửi cuối.

| Đơn vị | Số mẫu all → steady | TTFT p95 all → steady | TPOT p95 all → steady |
|---|---|---|---|
| W1 / 16 | 100 → 68 | 1.09 → 0.31 s (−71%) | +0.2% |
| W2 / 8 | 100 → 84 | 2.35 → 1.50 s (−36%) | −0.1% |
| W2 / 16 r1, r2, r3 | 100 → 68 | −10%, −21%, −11% | +3%, −2%, +1% |
| W2 / 32 | 128 → 64 | 25.1 → 23.9 s (−5%) | −1% |
| W3 / 8 | 100 → 84 | −1% | −1% |

Nhiễu r2 vs r3 của TTFT p95 ở W2/16: "all" 19% (vllm) / 1% (client tự viết); "steady" 31% / 36%.

### 3.7 Thời gian và chi phí

| Việc | Thời gian |
|---|---|
| Một đơn vị W1 / 16 | 124 s (1.24 s / request) |
| Một đơn vị W2 / 8, 16, 32 | 101–164 s (1.0–1.4 s / request) |
| Một đơn vị W3 / 8 | 334 s (3.3 s / request) |
| Phần cố định mỗi đơn vị | warmup + reset ~2–3 s; `vllm bench serve` tự tốn thêm **~20 s** (nạp tokenizer, đếm token từng prompt) trước khi gửi request, client tự viết ~1 s |
| Khởi động lại server (có cache) | 124–142 s |
| Smoke (10 đơn vị + 2 lần khởi động lại) | 29 phút |
| Phần P03 của phiên (thuê máy → bắt đầu P04) | 50 phút ≈ 0.84 h × $0.2833 ≈ $0.24, cộng phí tải về ~$0.03 → **≈ 7,000 VND** |

Hoá đơn Vast tính cho cả instance (P03 + P04). Tổng thật ghi trong báo cáo P04. Phần P03 chia theo thời gian.

## 4. Giải thích kết quả

**W2 hết KV cache ngay từ 16 user (đo được).**
- Một request W2 khi xong chiếm ~2,858 + 174 ≈ 3,030 token KV. KV 37,264 token chứa trọn được khoảng 12 request (suy ra).
- vLLM vẫn nhận tới 16 request đang chạy vì phần lớn request chưa sinh hết output. Khi thiếu chỗ, nó tạm dừng request (preemption, 6–7 lần mỗi lần đo) và đẩy request về hàng đợi (tới 12–13 request chờ).
- Ở lần khởi động 1 (KV 28.7k), trong TTFT trung bình 5.3 s thì thời gian chờ (`request_queue_time`) là 4.3 s, còn prefill chỉ 0.8 s. **TTFT ở W2 chủ yếu là thời gian xếp hàng chờ KV cache**, không phải thời gian tính toán.
- Dự đoán P01 "W2 chạy cùng lúc ~15–16" đúng về hướng. Số thật là 16 request đang chạy, nhưng có preemption: số request vừa với KV khi đã xong là ~12, phần còn lại sống nhờ việc request sinh output dần.

**Vì sao KV nhỏ hơn 23% thì TTFT p95 gấp đôi (đo được + suy ra).**
- Khi KV đầy, số request được chạy giới hạn bởi KV: 13 request ở KV 28.7k, 16 request ở KV 37.3k.
- Phần dư phải chờ. Thời gian chờ bằng thời gian vài request phía trước chạy xong (~15 s E2EL mỗi request).
- Vì vậy chênh 3–4 chỗ trong KV làm TTFT p95 đổi từ 4.6–5.6 s lên 9.9 s.
- Đây là bằng chứng sớm rằng **goodput W2/W3 trên L4 do dung lượng KV quyết định**.

**W3 / 8 trượt (KV 28.7k).** Một request W3 cần ~5,900 token KV, nên chỉ 6 request chạy cùng lúc (đo được). Với KV 37.3k, phép tính cho ~6.3 request. Vì vậy W3 có thể vẫn trượt ở 8 và P04 sẽ phải chia xuống 4 (suy ra).

**TPOT tăng theo N, nhưng còn xa ngưỡng 150 ms.**
- W2: 57 ms ở 8 user, 89–93 ms ở 16 user (KV 37.3k).
- Nguyên nhân: mỗi bước của vLLM gộp phần sinh token của các request đang chạy với một khúc prefill (tối đa 2,048 token) của request mới. Càng nhiều request mới vào thì càng nhiều bước có khúc prefill dài, nên TPOT tăng.
- Ở r1 (13 request chạy) TPOT p95 thấp hơn r2 (16 request chạy): 75.7 so với 92.6 ms, khớp với cơ chế này.

**Throughput thấp hơn giả định trong plan.md.** W1 / 16 đạt 322 token/s ra (giả định 380). W2 đạt ~170 token/s ra, nhưng kèm ~2,800 token/s prefill.

**GPU luôn chạy ở trần công suất 72 W (đo được).** Xung SM giảm còn ~1,000–1,100 MHz, khoảng một nửa mức tối đa 2,040 MHz. Đây là giới hạn của chính L4. Mọi kết quả về sau đều đo trong điều kiện này. Ghi để nhớ khi tính % roofline ở GĐ3.

**Vì sao TTFT p95 nhiễu ~20% còn throughput chỉ ~1%.**
- Throughput là tổng của 100 request, nên dao động trung bình đi.
- p95 của 100 mẫu là khoảng request chậm thứ 5. Mà TTFT chậm xuất hiện theo cụm: nhìn chuỗi TTFT theo thứ tự gửi, thấy từng nhóm 4–6 request liền nhau cùng 9–13 s, trùng với các lần preemption.
  Nghi do preemption (chưa đối chiếu từng thời điểm).
- Một cụm nhiều hay ít hơn là p95 đổi cả giây.

**Vì sao giữ cửa sổ "all" dù "steady" khác 5–21% (quyết định, có lý do).**
- Trước khi đo, tôi đặt quy tắc "lệch quá ~5% thì dùng steady". Số đo cho thấy lệch nhiều hơn. Nhưng số đo cũng cho thấy steady chỉ còn ~2N mẫu và nhiễu gần gấp đôi (31–36% so với 19%), lớn hơn cả độ lệch nó sửa được.
- Phần lệch lớn nhất (−36% đến −71%) nằm ở các mức tải nhẹ, nơi TTFT chỉ 0.3–2 s, còn xa ngưỡng 10 s, nên không đổi kết luận đạt/trượt. Gần ngưỡng (W2/16 r1, W2/32) lệch chỉ 5–10%.
- "all" lệch về phía thận trọng (TTFT xấu hơn thật) và giống cách `vllm bench serve` và các bài báo báo số. Mọi cấu hình về sau đo cùng cách nên so với nhau vẫn công bằng.
- Harness lưu từng request, nên báo cáo P04 có thêm goodput theo steady, không cần đo thêm. Nếu hai cách cho kết luận khác nhau thì ghi rõ.

**Vì sao không tăng số request để giảm nhiễu.** Nhiễu p95 giảm theo căn bậc hai số mẫu. Muốn giảm từ ~20% xuống 5% cần ~16 lần số request, không khả thi. Gấp đôi số request chỉ giảm nhiễu xuống ~14% (suy ra) mà tốn thêm ~5 giờ GPU ở P04. Thay vào đó: quyết định theo trung vị 3 lần, và so cấu hình bằng khoảng goodput ([challenges](../../challenges.md) #5).

## 5. Vấn đề gặp phải

| Vấn đề | Nguyên nhân | Xử lý | Chi phí |
|---|---|---|---|
| Lần đầu sweep không chạy sau setup | Lỗi của tôi trong `scripts/remote_bench.sh`: `tee` ghi vào `setup/setup.out` khi thư mục chưa có → lỗi; `set -e` + `pipefail` dừng script | Tạo thư mục trước | ~7 phút máy chạy không (~900 VND) |
| `ImportError: Please install vllm[bench]` | `vllm bench serve` cần `pandas`, `setup_node.sh` chỉ cài `vllm` (ở laptop tôi đã cài pandas tay nên không thấy) | `setup_node.sh` cài `vllm[bench]==0.30.0` | ~1 phút |
| KV nhỏ ở lần khởi động đầu | Xem mục 3.2, 4 | Harness tự khởi động lại khi compile > 10 s; r1 smoke chỉ dùng minh hoạ | [challenges](../../challenges.md) #4 |
| TTFT p95 nhiễu ~20% | Xem mục 4 | Trung vị 3 lần + so bằng khoảng goodput | [challenges](../../challenges.md) #5 |
| Server dùng lại sau khi sweep chết không ghi KV | Server còn chạy, log nằm ở lần khởi động trước | Lấy KV từ lần khởi động thật gần nhất trong `server_starts.jsonl` | — |
| Lệnh canh tiến độ báo lỗi giả | `sweep.log` ghi nối tiếp, còn Traceback của lần chạy trước | Lệnh canh chỉ đọc phần log ghi thêm | — |

## 6. Ảnh hưởng tới bước sau

**P04:**
- Chạy `bench/configs/full.yaml` (search, 3 lần lặp, cửa sổ "all", `recompile_restart_s: 10`, khởi động lại giữa các lần lặp của lưới gốc).
- Chạy ngay sau smoke trên cùng máy. Server lần khởi động 3 của P03 (KV 37,264) được dùng lại cho r1 của P04.
- Dự đoán:
  - W2: goodput khoảng 16–24 (16 đạt với TTFT p95 4.6–5.6 s; 32 trượt rất xa với 25 s).
  - W3: có thể < 8, cần chia xuống 4.
  - W1: chưa rõ (16 user mới dùng 41% KV, TTFT 1.1 s), có thể phải lên 128.
- Ước lượng (từ s/request đo được):
  - Lưới gốc ×3: 2.9 h.
  - Tìm nhị phân: W1 ~0.9 h, W2 ~0.5 h, W3 ~0.8 h.
  - **Tổng ~5.1 h ≈ 38k VND**, trường hợp xấu ~6.1 h (cập nhật README).

**P05:**
- Client phiên đã kiểm chứng (lệch ≤ 3.7% khi gộp).
- Chế độ `scenario` dùng tokenizer của model trên máy thuê để ghép prompt. Phần ghép prompt chạy trong thread pool, có thể trễ khi tải cao. Trên server giả (co thời gian 100 lần), lượt gửi trễ tối đa ~0.18 s so với lịch. P05 cần ghi lại độ trễ này.

**Hướng tune (nghi vấn, chưa đo):**
1. Tăng chỗ cho KV: KV FP8 (gấp đôi số token), `gpu_memory_utilization` cao hơn 0.92. P03 cho thấy chỉ +30% KV đã giảm TTFT p95 W2/16 một nửa.
2. Giảm preemption bằng `max_num_seqs` / giới hạn admission hợp lý.
3. GPU luôn chạm trần 72 W → kernel tiết kiệm năng lượng cũng là tăng tốc.

## 7. Thuật ngữ dùng trong báo cáo này

| Từ | Nghĩa |
|---|---|
| Harness | Bộ công cụ tự động chạy loạt đo, lưu kết quả và tổng hợp |
| Đơn vị đo | Một lần đo: workload × số user × lần lặp, có ID như `w2-c16-r2` |
| `done.json` | File đánh dấu đơn vị đã đo xong, ghi cuối cùng, chứa hash cấu hình |
| Closed-loop | N client, mỗi client gửi request, chờ xong rồi gửi tiếp ngay |
| Cửa sổ all / steady | Tính p95 trên mọi request / bỏ N request gửi đầu và N cuối |
| Preemption | vLLM tạm dừng một request đang chạy vì hết KV cache, trả chỗ, rồi tính lại sau |
| Prefix cache | Dùng lại KV của phần đầu prompt giống nhau giữa các request |
| Warmup | Vài request chạy trước khi đo, không tính vào kết quả |
| Oversample | Lặp lại mẫu khi thiếu mẫu (không dùng) |
| torch.compile | Bước biên dịch đồ thị tính toán của model; lần đầu mất ~70 s, lần sau đọc lại từ đĩa |
| Trần công suất | Mức điện tối đa GPU được dùng (L4: 72 W); chạm trần thì GPU tự giảm xung |
| Server giả | Chương trình trả lời giống vLLM nhưng không chạy model, để test harness ở laptop |
