# P08 — Chuẩn bị harness cho tune

> Cần GPU: không · Ước giờ GPU: 0 · Phụ thuộc: P07 · Làm ở laptop, thử bằng server giả

## Mục tiêu

Harness hiện đo được một cấu hình server mỗi lần chạy, và W1–W3 luôn tìm goodput. Phần tune cần: nhiều cấu hình trong một phiên thuê,
đo ở một mức tải cố định, so cặp với cấu hình trước, và ghi đủ thông tin để biết vLLM thật sự chạy với backend / KV nào.
P08 làm các việc đó ở laptop, để giờ GPU của P09–P13 chỉ dùng cho đo.

## Các việc

| # | Việc | Vì sao | Cách làm dự kiến |
|---|---|---|---|
| 1 | **Cấu hình biến thể**: một file chung + mỗi biến thể chỉ ghi cờ server khác và mức đo | ~10 biến thể; chép tay dễ lệch tham số | Khoá `include:` (gộp sâu với file chung) trong `bench/common.py` khi đọc cấu hình; thư mục `bench/configs/tune/` |
| 2 | **Đo ở mức cố định cho W4** (`w4.fixed: [[peak-s1, k], ...]`, không tìm k) | Sàng lọc N2, N3 ở một mức (README mục "Nguyên tắc đo") | Thêm nhánh trong `W4Runner.plan_batch`. W1–W3 đã có sẵn chế độ `grid` |
| 3 | **Chạy nhiều cấu hình liên tiếp trong một phiên**, chạy tiếp được | Đổi cấu hình không phải can thiệp tay; máy chết thì chạy lại cùng lệnh | `scripts/remote_multi.sh LIST` (mỗi dòng: cấu hình, thư mục kết quả); setup một lần; mỗi cấu hình là một `sweep.py` vốn đã chạy tiếp được |
| 4 | **So cặp** hai lần đo cùng đơn vị (cùng seed) | Quyết định sàng lọc so với cấu hình trước ở cùng mức, cùng seed | `bench/compare.py A B`: theo từng đơn vị: đạt/trượt, TTFT p50/p95, TPOT p95, throughput, preemption, trúng cache; W4 thêm % phút có TTFT p95 > 10 s và TTFT request ngắn khi có request dài (lấy từ `p07_analyze.py`) |
| 5 | **Ghi cấu hình thật vLLM chọn** mỗi lần khởi động: backend attention, KV dtype, số token KV, `max_num_batched_tokens`, offloading | KV FP8 có thể làm vLLM đổi backend (library-gaps #3); phải biết đã đo cái gì | Thêm regex vào `Server.start` (đọc log), ghi vào `server_starts.jsonl` |
| 6 | **Theo dõi thêm metric**: `vllm:prompt_tokens_by_source_total` (gồm phần lấy từ RAM CPU) | Đo được P11 có dùng lại lịch sử không | Thêm vào `POLL_METRICS` và phần tổng hợp |
| 7 | **Client gán độ ưu tiên theo độ dài prompt** (`--priority-by-length`) | Cấu hình `--scheduling-policy priority` của P10 | `session_client.py`: < 1k token → 0, 1–4k → 1, ≥ 4k → 2. Trường `priority` có trong request completions của vLLM 0.30.0 (`vllm/entrypoints/openai/completion/protocol.py`, đã lưu ở references F3); vLLM báo lỗi nếu gửi priority ≠ 0 mà server không bật policy priority → chỉ bật cờ này cùng cấu hình đó |
| 8 | **Cấu hình cổng chất lượng cho biến thể** | P09, P13 | `quality/configs/tune-*.yaml` (4 đơn vị: teacher forcing c8, sinh tham chiếu c8, FinQA, MMLU-Pro) với `server.start` của biến thể; `quality/analyze.py --compare` so với **cả hai** lần chạy P06 và in khoảng chênh giữa hai lần baseline |
| 8b | **Hệ số mở rộng và độ mịn riêng cho W4** (`search.grow: 1.25`, `search.abs_tol` theo cấu hình) | Khi cả hai mức khởi đầu nóng đều đạt, nhân đôi k (22 → 44) tốn thêm 1–2 lần chạy 20–40 phút; P10 chỉ cần độ mịn 4 | Thêm tham số vào `common.search_step` (mặc định giữ ×2 như cũ) |
| 9 | **Ước giờ trong `--status`** dùng đơn giá đo thật (README "Đơn giá thời gian") | Theo dõi chi phí trong phiên | Sửa bảng thời gian đơn vị trong `sweep.py` |
| 10 | **Thử với server giả** | Không tốn GPU khi lỗi harness | `bench/test_fake.sh`: 2 biến thể nối tiếp, W4 mức cố định, ngắt giữa chừng rồi chạy lại; `quality` với server giả |

Không đổi: cách đo của một đơn vị (warmup → reset prefix cache → client → metric 1 s), seed, cửa sổ "all", SLO.

## Tiêu chí xong

- [ ] Tất cả việc 1–9 có code, test cũ (`bench/test_search.py`, `workloads/test_w4_session.py`, `quality/test_analyze.py`) vẫn qua.
- [ ] Thử với server giả: chạy danh sách 2 cấu hình, ngắt bằng SIGKILL giữa đơn vị, chạy lại cùng lệnh → không đo lại đơn vị đã xong.
- [ ] `compare.py` chạy được trên kết quả P04 (so r1 với r2) và P05.
- [ ] Cấu hình P09 (`bench/configs/tune/p09-*.yaml`, `quality/configs/tune-p09.yaml`) viết sẵn, `--dry` in đúng danh sách đơn vị.

## Báo cáo P08 cần trả lời

1. Harness đổi gì, đã thử thế nào.
2. Lệnh chạy của P09 và danh sách đơn vị sẽ đo.
