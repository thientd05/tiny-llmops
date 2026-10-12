# P03 — Công cụ chạy benchmark (harness) + kiểm tra rút gọn trên L4

> Cần GPU: có · Ước tính ~1 giờ GPU (~10k VND) · Phụ thuộc: P01, P02

## Mục tiêu

1. Một lệnh chạy được cả loạt đo (workload × số user đồng thời × lần lặp) trên một server đang chạy,
   tự lưu kết quả kèm đầy đủ thông tin để tái lập.
2. Chứng minh công cụ **đo đúng** và biết **mức nhiễu** giữa các lần đo.
3. Đo thời gian thực mỗi lần chạy → chốt số request mỗi lần đo và ước lượng lại giờ GPU cho P04.

## Thiết kế

- **Bộ gửi tải:** dùng `vllm bench serve` của đúng phiên bản vLLM đã ghim. Lý do: công cụ chuẩn, cách tính TTFT/TPOT
  giống các bài báo và issue của vLLM → số so được với bên ngoài; dùng được cho cả thư viện khác (cùng API kiểu OpenAI).
- **Client chạy trên chính máy thuê**, gọi `localhost`. Gọi từ Việt Nam sang máy ở Czechia sẽ cộng thêm ~200 ms
  thời gian mạng vào TTFT → sai.
- **Chế độ tải:** số user đồng thời cố định (`--max-concurrency C`, gửi request mới ngay khi có request xong).
  Chế độ theo thời gian thực của trace để ở P05.
- **Trước mỗi lần đo:**
  1. gọi `/reset_prefix_cache` (xoá bộ nhớ đệm prefix của lần trước, tránh lần sau được lợi);
  2. gửi vài request khởi động (warmup), không tính vào kết quả.
- **Tham số chính:** `--backend openai-chat`, dataset từ P02, `--ignore-eos`, `temperature=0`,
  `--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99`, `--goodput ttft:2000 tpot:100`, `--save-result`.
- **Ghi song song trong lúc đo:**
  - `nvidia-smi` mỗi 1 giây: VRAM, công suất, xung nhịp, nhiệt độ (L4 giới hạn 72 W → kiểm tra có bị hạ xung không);
  - `/metrics` của vLLM trước và sau: % KV cache đang dùng, số request bị tạm dừng do hết KV (preemption),
    tỉ lệ trúng prefix cache, số request đang chờ.
- **Chạy tiếp được** (theo [thi-nghiem-chay-tiep.md](../../thi-nghiem-chay-tiep.md)):
  - mỗi điểm đo (workload × số user × lần lặp) là một đơn vị, có ID tính từ cấu hình (`w2-c32-r2`), lưu ở `results/.../<ID>/`;
  - `done.json` (chứa hash cấu hình) ghi cuối cùng; chạy lại cùng lệnh thì bỏ qua điểm đã xong, điểm dở dang bị xoá và chạy lại;
  - `--status` in số điểm đã xong / còn lại và ước tính giờ GPU còn cần;
  - trong phiên, local chạy vòng rsync kéo kết quả về mỗi ~2 phút.
- **Metadata mỗi lần đo:** `run_id`, giờ bắt đầu/kết thúc, host_id/machine_id, phiên bản phần mềm, lệnh server,
  lệnh client, sha256 workload, git commit của repo.

## Các bước

### A. Ở máy local

1. `bench/sweep.py`: đọc file cấu hình (YAML) → chạy lần lượt các điểm đo; mỗi điểm lưu vào `results/.../<run_id>/`.
   Thứ tự chạy **xen kẽ** giữa các lần lặp (không chạy 3 lần liên tiếp cùng một điểm) để giảm ảnh hưởng của nhiệt độ GPU theo thời gian.
2. `bench/summarize.py`: gom JSON → bảng CSV + Markdown: throughput (token/s vào, ra), TTFT/TPOT/ITL p50/p95/p99,
   đạt SLO không, trung bình và độ lệch giữa các lần lặp.
3. Cấu hình `bench/configs/smoke.yaml` (rút gọn) và `bench/configs/full.yaml` (cho P04).
4. **Thử chạy tiếp ở local** bằng một server giả (trả lời ngay, không cần GPU):
   - ngắt giữa chừng (Ctrl+C), chạy lại → chỉ chạy các điểm còn thiếu;
   - sửa cấu hình → các điểm cũ bị chạy lại.
   Làm phần này ở local để không tốn giờ GPU.

### B. Trên L4

5. Thuê máy, chạy `setup_node.sh`, `serve_default.sh` (từ P01) trong `tmux`.
6. Chạy `smoke.yaml`: W2 ở 8 và 32 user, W1 ở 16 user, W3 ở 8 user — mỗi điểm 1 lần; thêm 1 lần lặp cho W2 / 16 user × 2 lần.
7. Kiểm tra công cụ:
   - (a) số token vào/ra thực tế khớp workload;
   - (b) TTFT client đo khớp số trên `/metrics` của server (lệch nhỏ, giải thích được);
   - (c) reset prefix cache có tác dụng: tỉ lệ trúng cache ≈ chỉ phần system prompt chung của W2;
   - (c2) **kiểm tra dự đoán KV cache của P01:** ghi theo thời gian (mỗi 1 s) `num_requests_running`, `num_requests_waiting`,
     % KV cache đã dùng và tổng số preemption. Dự đoán: W2 có số request đang chạy tối đa ~15–16, W3 ~6–7. Ghi dung lượng KV cache
     (token) của **mỗi lần khởi động** để kiểm tra giả thuyết "lần đầu compile cho KV nhỏ hơn";
   - (d) độ lệch giữa 2 lần đo cùng điểm. **Cần < 5%** thì quy tắc loại thư viện "kém hơn > 5%" mới có nghĩa;
     nếu lớn hơn → tăng số request mỗi lần đo hoặc số lần lặp.
8. Từ thời gian thực từng điểm → chốt số request mỗi lần đo (đề xuất ban đầu: `max(100, 4 × C)`) và ước lượng giờ GPU P04.
9. Rsync kết quả về, destroy máy (hoặc chuyển thẳng sang P04 nếu đã sẵn sàng — xem README).

## Tiêu chí xong

- [ ] `bench/sweep.py`, `bench/summarize.py`, 2 file cấu hình đã commit.
- [ ] 4 kiểm tra (a)–(d) đạt, hoặc đã có cách xử lý.
- [ ] Đã thử: ngắt giữa chừng rồi chạy lại cùng lệnh → chỉ chạy phần còn thiếu.
- [ ] Ước lượng giờ GPU cho P04 đã cập nhật trong README GĐ1.

## Báo cáo P03 cần trả lời

1. Công cụ đo những chỉ số nào, mỗi chỉ số được tính từ đâu (client hay server), vì sao tin được.
2. Nhiễu giữa 2 lần đo là bao nhiêu % → chênh lệch nhỏ nhất có thể khẳng định là khác biệt thật.
3. Kết quả sơ bộ đầu tiên (chưa phải baseline chính thức): TTFT/TPOT ở vài điểm, có thấy dấu hiệu hết KV cache không.
4. Mỗi lần đo mất bao lâu → P04 cần bao nhiêu giờ GPU.
