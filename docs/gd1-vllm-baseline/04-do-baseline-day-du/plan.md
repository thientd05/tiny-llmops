# P04 — Đo baseline đầy đủ W1/W2/W3 (vLLM mặc định)

> Cần GPU: có · Ước tính ~4 giờ GPU (~41k VND, chỉnh lại theo P03) · Phụ thuộc: P03

## Mục tiêu

Bộ số liệu chính thức của vLLM mặc định trên L4: với mỗi workload và mỗi mức 8 / 16 / 32 / 64 user đồng thời,
biết TTFT, TPOT, throughput và **goodput** (số user tối đa vẫn đạt SLO: TTFT p95 < 2 s, TPOT p95 < 100 ms).

## Các bước

1. Thuê L4 (không interruptible; ưu tiên cùng host với P03), chạy setup + server mặc định.
2. Chạy `bench/configs/full.yaml`: W1, W2, W3 × {8, 16, 32, 64} × 3 lần lặp = 36 lần đo,
   thứ tự xen kẽ, reset prefix cache + warmup trước mỗi lần (đã có trong harness).
   Nếu bị dừng giữa chừng: thuê lại hoặc `start` máy, rsync `results/gd1/04-do-baseline-day-du/` từ local lên, chạy lại
   **cùng lệnh** → chỉ đo các điểm chưa có `done.json`. Nếu sang host khác → đo lại điểm mốc W2/16 user để so hai host.
3. Khởi động lại server giữa các lần lặp (3 lần khởi động) → nhiễu do trạng thái server cũng nằm trong độ lệch đo được.
4. **Điểm bổ sung (nếu cần):** nếu ranh giới đạt/không đạt SLO nằm giữa hai mức (ví dụ đạt ở 16, trượt ở 32) →
   đo thêm điểm giữa (24) × 3 lần, để goodput không bị làm tròn quá thô. Tương tự giữa 32 và 64 (48).
5. Rsync kết quả, destroy máy. Chạy `summarize.py` ở local.
6. Ghi bảng tóm tắt vào [measurements.md](../../measurements.md).

## Kết quả cần có

| Bảng / hình | Nội dung |
|---|---|
| Bảng chính | Workload × concurrency: throughput ra (token/s), TTFT p50/p95/p99, TPOT p50/p95/p99, đạt SLO (có/không), độ lệch giữa 3 lần |
| Goodput | W1 / W2 / W3: số user tối đa đạt SLO |
| Hình 1 | TTFT p95 theo số user, mỗi workload một đường, kẻ đường SLO 2 s |
| Hình 2 | TPOT p95 theo số user, kẻ đường SLO 100 ms |
| Bảng phụ | % KV cache đã dùng, số lần preemption, VRAM, công suất GPU, xung nhịp ở mỗi điểm |

## Tiêu chí xong

- [ ] Đủ 36 lần đo (+ điểm bổ sung nếu có), độ lệch giữa các lần lặp được báo cáo.
- [ ] Goodput W1/W2/W3 xác định được.
- [ ] Bảng tóm tắt đã ghi measurements.md; kết quả thô ở `results/gd1/04-do-baseline-day-du/`.

## Báo cáo P04 cần trả lời

1. vLLM mặc định phục vụ được bao nhiêu user mỗi workload mà vẫn đạt SLO?
2. Khi vượt goodput, chỉ số nào hỏng trước (TTFT hay TPOT) và vì sao — ví dụ: hết KV cache nên request phải chờ
   (TTFT tăng), hoặc batch lớn làm mỗi bước decode lâu hơn (TPOT tăng). Dẫn chứng bằng số từ `/metrics`.
3. So với giả định trong [plan.md](../../plan.md) (260 / 380 / 520 / 600 token/s ở 8 / 16 / 32 / 64 user), số thật cao hơn hay thấp hơn bao nhiêu.
4. Nghi vấn đầu tiên về chỗ chậm (chưa profile, chỉ suy từ số) → gợi ý tham số cần tune.
