# P04 — Đo baseline đầy đủ W1/W2/W3 (vLLM mặc định)

> Cần GPU: có · Ước tính ~6.5 giờ GPU (~66k VND, chỉnh lại theo P03) · Phụ thuộc: P03

## Mục tiêu

Bộ số liệu chính thức của vLLM mặc định trên L4: với mỗi workload, biết TTFT, TPOT, throughput ở từng mức user đồng thời
và **goodput** (số user tối đa vẫn đạt SLO chính: TTFT p95 ≤ 10 s, TPOT p95 ≤ 150 ms; kèm goodput theo SLO phụ 2 s / 100 ms).
SLO và căn cứ: [workloads.md — SLO](../../workloads.md#slo). Plan này đo các ca thuần W1/W2/W3 (closed-loop); W4 (tải giống thật) đo ở [P05](../05-w4-tai-giong-that/plan.md).
Các mức user **không cố định**: lưới gốc 8 / 16 / 32 / 64, tự mở rộng lên/xuống và tìm nhị phân tới điểm gãy, theo
[workloads.md — Cách tìm goodput](../../workloads.md#cách-tìm-goodput-lưới-thích-nghi).

Dự đoán từ P01 (chưa đo): W1 có thể vẫn đạt ở 64 → cần mở rộng lên 128+. W2 (~15–16 request chạy cùng lúc) và
W3 (~6–7) dễ trượt sớm → có thể cần xuống 4/2. Lưới cố định sẽ không bắt được các trường hợp này.

## Các bước

1. Thuê L4 (không interruptible; ưu tiên cùng host với P03), chạy setup + server mặc định.
2. Chạy `bench/configs/full.yaml` ở chế độ `search`, ×3 lần lặp mỗi điểm, quyết định đạt/trượt theo trung vị 3 lần:
   - **Đợt 1 — lưới gốc:** W1, W2, W3 × {8, 16, 32, 64} × 3 = 36 lần đo, thứ tự xen kẽ. Khởi động lại server giữa
     các lần lặp (3 lần khởi động) → nhiễu do trạng thái server cũng nằm trong độ lệch đo được;
   - **Các đợt sau:** harness tự sinh điểm tiếp theo cho từng workload. Nếu 64 đạt thì nhân đôi 128, 256…; nếu 8 trượt thì đo 4, 2;
     sau đó tìm nhị phân tới `hi − lo ≤ max(2, 10% × lo)`. Dự kiến thêm ~3–5 điểm × 3 lần mỗi workload.
   - Reset prefix cache + warmup trước mỗi lần đo (đã có trong harness).
   Nếu bị dừng giữa chừng: thuê lại máy, rsync `results/gd1/04-do-baseline-day-du/` từ local lên, chạy lại
   **cùng lệnh** → harness tính lại điểm tiếp theo từ các `done.json` và chỉ đo phần còn thiếu.
   Nếu sang host khác → đo lại điểm mốc W2/16 user để so hai host.
3. Nếu harness báo **thiếu mẫu** (N lớn cần nhiều request hơn số mẫu của workload) → dừng workload đó, ghi trần vào báo cáo,
   cân nhắc sinh thêm mẫu (P02) rồi đo tiếp.
4. Nếu thấy **vi phạm đơn điệu** (mức đạt nằm trên mức trượt) → đo lặp lại hai mức đó, ghi vào báo cáo và [challenges.md](../../challenges.md).
5. Rsync kết quả, destroy máy. Chạy `summarize.py` ở local.
6. Ghi bảng tóm tắt vào [measurements.md](../../measurements.md).

## Kết quả cần có

| Bảng / hình | Nội dung |
|---|---|
| Bảng chính | Workload × concurrency: throughput ra (token/s), TTFT p50/p95/p99, TPOT p50/p95/p99, đạt SLO (có/không), độ lệch giữa 3 lần |
| Goodput | W1 / W2 / W3: `lo` (mức đạt lớn nhất) và `hi` (mức trượt nhỏ nhất), kèm số điểm đã đo để tìm ra; một cột theo SLO chính, một cột theo SLO phụ |
| Hình 1 | TTFT p95 theo số user (trục log₂ vì N có thể từ 2 tới 256+), mỗi workload một đường, kẻ đường SLO 10 s (chính) và 2 s (phụ), đánh dấu goodput |
| Hình 2 | TPOT p95 theo số user, kẻ đường SLO 150 ms (chính) và 100 ms (phụ) |
| Hình 3 | Độ nhạy: goodput theo ngưỡng TTFT 2 / 5 / 10 / 20 s, mỗi workload một đường (tính lại từ số liệu từng request, không đo thêm) |
| Bảng phụ | % KV cache đã dùng, số lần preemption, VRAM, công suất GPU, xung nhịp ở mỗi điểm |

## Tiêu chí xong

- [ ] Đủ 36 lần đo lưới gốc + các điểm mở rộng/tìm nhị phân, độ lệch giữa các lần lặp được báo cáo.
- [ ] Goodput W1/W2/W3 xác định được **dạng khoảng hẹp** (`hi − lo ≤ max(2, 10% × lo)`), không còn dạng "≥ 64" hay "< 8".
- [ ] Bảng tóm tắt đã ghi measurements.md; kết quả thô ở `results/gd1/04-do-baseline-day-du/`.

## Báo cáo P04 cần trả lời

1. vLLM mặc định phục vụ được bao nhiêu user mỗi workload mà vẫn đạt SLO (dạng khoảng `lo`–`hi`)? Mỗi workload cần mở rộng
   lên/xuống không, và mất bao nhiêu điểm đo để tìm ra?
2. Khi vượt goodput, chỉ số nào hỏng trước (TTFT hay TPOT) và vì sao — ví dụ: hết KV cache nên request phải chờ
   (TTFT tăng), hoặc batch lớn làm mỗi bước decode lâu hơn (TPOT tăng). Dẫn chứng bằng số từ `/metrics`.
3. So với giả định trong [plan.md](../../plan.md) (260 / 380 / 520 / 600 token/s ở 8 / 16 / 32 / 64 user), số thật cao hơn hay thấp hơn bao nhiêu.
4. Nghi vấn đầu tiên về chỗ chậm (chưa profile, chỉ suy từ số) → gợi ý tham số cần tune.
