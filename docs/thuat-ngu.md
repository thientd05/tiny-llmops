# Thuật ngữ

> Giải thích ngắn các từ dùng trong plan và báo cáo. Gặp từ mới → thêm vào đây (giữ thứ tự theo nhóm).

## Model và lượng tử

| Từ | Nghĩa |
|---|---|
| Token | Đơn vị văn bản model xử lý (một từ hoặc một phần của từ). Tiếng Anh trung bình ~0.75 từ/token |
| Prompt | Toàn bộ văn bản gửi vào model (system prompt + context + câu hỏi) |
| System prompt | Phần hướng dẫn cố định đặt đầu prompt, giống nhau cho mọi request của cùng ứng dụng |
| MoE (Mixture of Experts) | Model có nhiều khối "expert"; mỗi token chỉ đi qua vài expert (Qwen3-30B-A3B: 8 trong 128) → tổng 30B tham số nhưng mỗi token chỉ tính ~3B |
| Lượng tử hoá (quantization) | Lưu weight bằng ít bit hơn (INT4 = 4 bit thay vì 16 bit) → model nhỏ hơn ~4 lần, đọc từ bộ nhớ nhanh hơn, chất lượng giảm nhẹ |
| GPTQ, AWQ | Hai phương pháp lượng tử hoá weight xuống 4 bit phổ biến |
| W4A16 | Weight 4 bit, activation (dữ liệu trung gian khi tính) 16 bit |
| Marlin | Tên kernel (chương trình chạy trên GPU) của vLLM để nhân ma trận với weight 4 bit |
| Act-order (`desc_act`, `g_idx`) | Kiểu lượng tử GPTQ xử lý các cột weight theo thứ tự đã đảo; cần kernel riêng. vLLM 0.30.0 bỏ hỗ trợ — model của dự án không dùng (`desc_act=false`) |
| Chat template | Quy tắc ghép các vai trò (system, user, assistant) thành một chuỗi văn bản có token đánh dấu riêng của model |
| Thinking mode | Chế độ của Qwen3: model viết phần suy luận dài trước câu trả lời. Tắt bằng `/no_think` hoặc `enable_thinking=false` |
| Greedy | Cách sinh token luôn chọn token có xác suất cao nhất (`temperature=0`) → kết quả gần như cố định |
| Logprob | Log của xác suất model gán cho một token |
| KL divergence | Số đo mức khác nhau giữa hai phân phối xác suất; 0 = giống hệt |

## Quá trình phục vụ (serving)

| Từ | Nghĩa |
|---|---|
| Prefill | Bước đầu: model đọc toàn bộ prompt một lần. Tốn tính toán, tỉ lệ với độ dài prompt |
| Decode | Các bước sau: sinh từng token output một. Mỗi bước phải đọc lại toàn bộ weight cần dùng từ VRAM → bị giới hạn bởi băng thông bộ nhớ |
| KV cache | Vùng VRAM lưu kết quả trung gian (key, value) của các token đã xử lý để không tính lại. Dung lượng KV cache quyết định bao nhiêu request chạy cùng lúc được |
| PagedAttention | Cách vLLM chia KV cache thành khối nhỏ (16 token) và cấp cho request khi cần, không giữ chỗ trước cho độ dài tối đa |
| Prefix caching | Giữ lại KV cache của phần đầu prompt giống nhau (ví dụ system prompt) để request sau dùng lại, bỏ qua prefill phần đó |
| Chunked prefill | Chia prefill của prompt dài thành nhiều phần, xen với các bước decode của request khác → request khác không bị dừng lâu |
| Batch | Nhóm request được GPU xử lý cùng một bước |
| Preemption | Khi hết KV cache, vLLM tạm dừng một số request (bỏ KV của chúng, tính lại sau) → các request đó bị chậm |
| CUDA graph | Ghi lại chuỗi lệnh GPU một lần, chạy lại nhiều lần → giảm thời gian CPU gửi lệnh |
| `max_model_len` | Độ dài tối đa (prompt + output) một request được phép |
| `gpu_memory_utilization` | Tỉ lệ VRAM vLLM được dùng (mặc định 0.9 = 90%) |
| Backend attention | Bộ kernel tính phần attention (FlashAttention, FlashInfer, Triton) |

## Đo đạc

| Từ | Nghĩa |
|---|---|
| Concurrency (số user đồng thời) | Số request đang được phục vụ cùng lúc; xong một cái thì gửi cái mới |
| Closed-loop (tải vòng kín) | Kiểu đo đặt cố định N client; mỗi client gửi request, chờ xong rồi mới gửi tiếp. Số request trong hệ thống luôn là N; req/s và latency là kết quả đo được. P03/P04 dùng kiểu này |
| Open-loop (tải vòng hở) | Kiểu đo đặt cố định tốc độ request đến (req/s), không chờ request trước xong. Quá tải thì hàng đợi tăng mãi |
| Partly-open | Người dùng đến theo nhịp riêng (open), nhưng trong một phiên thì chờ câu trả lời xong, nghỉ, rồi mới hỏi tiếp (closed). Gần thực tế nhất; W4 dùng kiểu này |
| Phiên (session) | Một cuộc hội thoại gồm nhiều lượt hỏi–đáp; lượt sau mang theo lịch sử các lượt trước |
| Think time | Thời gian **người dùng** đọc và nghỉ, từ lúc nhận xong câu trả lời tới lúc gửi câu tiếp theo (không phải chế độ thinking của model) |
| Chồng phiên | Cách tăng tải W4: phát chồng các phiên của cùng khung giờ từ k ngày khác nhau của trace → k lần số người dùng, giữ nguyên think time |
| Điểm gãy (knee) | Mức N mà TTFT hoặc TPOT p95 bắt đầu vượt SLO |
| Lưới thích nghi + tìm nhị phân | Cách tìm goodput: đo lưới gốc 8/16/32/64, nhân đôi lên hoặc chia đôi xuống nếu chưa thấy điểm gãy, rồi chia đôi khoảng giữa mức đạt và mức trượt. Chi tiết: [workloads.md](workloads.md#cách-tìm-goodput-lưới-thích-nghi) |
| Hiệu ứng trần | Kết quả bị chặn ở biên vùng đo (ví dụ "≥ 64") → hai cấu hình khác nhau trông như bằng nhau |
| TTFT (Time To First Token) | Thời gian từ lúc gửi request tới lúc nhận token đầu tiên. Chủ yếu do chờ hàng đợi + prefill |
| TPOT (Time Per Output Token) | Thời gian trung bình giữa các token output sau token đầu, của một request |
| ITL (Inter-Token Latency) | Khoảng thời gian giữa hai token liên tiếp (từng khoảng, không lấy trung bình theo request) |
| E2EL | Tổng thời gian một request, từ gửi tới nhận token cuối |
| p50 / p95 / p99 | Giá trị mà 50% / 95% / 99% số request nhỏ hơn hoặc bằng. p95 TTFT ≤ 10 s = 95% request có TTFT không quá 10 s |
| SLO | Mức độ trễ cam kết với người dùng. Dự án dùng SLO chính TTFT p95 ≤ 10 s, TPOT p95 ≤ 150 ms (căn cứ: nghiên cứu người dùng + tốc độ đọc, xem [workloads.md](workloads.md#slo)); SLO phụ 2 s / 100 ms để so sánh |
| Phân tích độ nhạy | Tính lại kết quả với nhiều giá trị của một giả định (ví dụ ngưỡng TTFT 2/5/10/20 s) để xem kết luận có đổi không |
| Goodput | Số user đồng thời tối đa trên 1 GPU mà vẫn đạt SLO. Báo cáo dạng khoảng: `lo` (mức đạt lớn nhất), kèm `hi` (mức trượt nhỏ nhất) |
| Bão hoà | Số request chạy cùng lúc đã chạm trần (trên L4 do KV cache: P04 W1 ~60, W2 ~15, W3 ~6); thêm user chỉ làm hàng đợi dài ra, TTFT tăng, throughput đi ngang |
| Thời gian chờ (queue time) | Thời gian request nằm trong hàng đợi của server trước khi được chạy; vLLM báo qua `vllm:request_queue_time_seconds` |
| Khởi đầu nóng (warm start) | Sweep rút gọn bắt đầu từ (lo, hi) của cấu hình trước thay vì lưới gốc 8/16/32/64 |
| Throughput | Số token xử lý được mỗi giây (tính riêng token vào và token ra) |
| Sweep | Một loạt lần đo thay đổi một tham số (ví dụ số user). Với số user, các mức được chọn theo lưới thích nghi, không cố định |
| Harness | Bộ script tự động chạy đo, lưu kết quả và thông tin môi trường |
| Warmup | Vài request gửi trước khi đo, không tính, để GPU và server vào trạng thái ổn định |
| Đơn vị đo | Một lần đo: workload × số user × lần lặp, có ID như `w2-c16-r2`; xong thì ghi `done.json` (chứa hash cấu hình) cuối cùng |
| Cửa sổ all / steady | Tính p95 trên mọi request của lần đo / bỏ N request gửi đầu (đợt cùng bắt đầu) và N request gửi cuối (số request đang chạy giảm dần). Dự án quyết định đạt/trượt theo "all" (P03) |
| Nhiễu đo | Hai lần đo cùng một điểm lệch nhau bao nhiêu, tính (max − min) / trung vị. P03: throughput ~1%, TPOT ~4%, TTFT p95 ~20% |
| Oversample | Công cụ đo tự lặp lại mẫu khi thiếu mẫu. Dự án tắt (`--no-oversample`) vì mẫu lặp trúng prefix cache |
| Server giả | Chương trình trả lời giống vLLM nhưng không chạy model (`bench/fake_server.py`), để thử harness ở laptop |
| Trace | Bản ghi thời điểm đến (và độ dài) của request từ hệ thống thật, dùng để phát lại |
| Baseline | Kết quả mốc để so sánh; ở dự án này là vLLM cấu hình mặc định |
| Roofline / % băng thông | So tốc độ thực của kernel với giới hạn phần cứng (L4: 300 GB/s băng thông bộ nhớ) |

## Công cụ & phần mềm

| Từ | Nghĩa |
|---|---|
| torch.compile | PyTorch biên dịch phần tính toán của model thành code GPU tối ưu hơn; lần đầu ~60 s, sau đó dùng lại cache |
| Trần công suất (power limit) | Mức điện tối đa GPU được dùng; L4 là 72 W. Chạm trần thì GPU tự hạ xung nhịp (P03: ~1,100 MHz thay vì 2,040) |
| JIT (biên dịch lúc chạy) | Kernel được biên dịch khi chương trình cần lần đầu, thay vì biên dịch sẵn lúc cài |
| `ninja` | Chương trình điều phối biên dịch, FlashInfer dùng khi biên dịch JIT |
| FlashAttention / FlashInfer | Thư viện kernel tính attention (FlashInfer còn có kernel lấy mẫu token) |
| venv | Môi trường Python riêng chứa đúng các gói và phiên bản dự án cần |
| uv | Công cụ cài gói Python nhanh, dùng để cài vLLM trên máy thuê |
| Wheel | Gói Python đã biên dịch sẵn, cài không cần build |
| JSONL | File văn bản, mỗi dòng là một bản ghi JSON |
| sha256 | Mã băm của file: hai file cùng sha256 thì giống hệt nhau từng byte |
| Manifest | File ghi nguồn dữ liệu, phiên bản, seed và sha256 để tái lập |
| Seed | Số khởi tạo cho bộ sinh ngẫu nhiên; cùng seed thì ra cùng kết quả |
