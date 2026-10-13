# GĐ1 (phần đầu) — vLLM mặc định làm mốc so sánh (baseline)

> Lập ngày 2026-10-12. P01, P02 xong ngày 2026-10-12. P03, P04, P05, P06 xong 2026-10-13 (P06 lần 2: 2026-10-14). P07 xong 2026-10-14.
> Mục đích: đo vLLM ở cấu hình mặc định trên L4 với đúng 3 workload của dự án, để mọi bước tune / đổi thư viện /
> custom CUDA về sau có một mốc cố định để so. Phần "tune tối đa" của GĐ1 sẽ lập plan riêng sau khi có báo cáo P07. → Đã lập: [../gd1-vllm-tune/README.md](../gd1-vllm-tune/README.md) (P08–P14).

## Thế nào là "vLLM mặc định"

- Lệnh: `vllm serve Qwen/Qwen3-30B-A3B-GPTQ-Int4` + **chỉ những cờ bắt buộc để chạy được** (cổng, đường dẫn model).
- **Đã chốt (P01):** vLLM **0.30.0** + đúng 1 cờ bắt buộc `--max-model-len 16384` (mặc định 40960 không đủ KV cache, xem
  [challenges.md](../challenges.md) #2). Script: `scripts/serve_default.sh <log> --max-model-len 16384`.
- Nếu cấu hình mặc định không khởi động được trên L4 (ví dụ thiếu bộ nhớ cho độ dài ngữ cảnh mặc định) → thêm **ít cờ nhất có thể**,
  ghi rõ cờ nào và vì sao vào báo cáo P01. Bộ cờ đó dùng nguyên cho P03–P06.
- Biến môi trường `VLLM_SERVER_DEV_MODE=1` được phép: chỉ mở thêm endpoint `/reset_prefix_cache` để xoá bộ nhớ đệm giữa các lần đo,
  không đổi tốc độ chạy.
- Phiên bản vLLM chọn ở P01 được **ghim** cho toàn bộ GĐ1.

## Danh sách plan

| # | Plan | Cần GPU? | Giờ GPU ước tính | Trạng thái | Báo cáo |
|---|---|---|---|---|---|
| P01 | [Cài vLLM và chạy thử model trên L4](01-cai-dat-chay-thu/plan.md) | Có | ~0.75 h (thực tế 0.35 h, $0.126) | **Xong** 2026-10-12 | [đọc ngay](01-cai-dat-chay-thu/doc-ngay.md) · [đầy đủ](01-cai-dat-chay-thu/bao-cao.md) |
| P02 | [Bộ sinh workload W1/W2/W3](02-bo-sinh-workload/plan.md) | Không | 0 | **Xong** 2026-10-12 | [đọc ngay](02-bo-sinh-workload/doc-ngay.md) · [đầy đủ](02-bo-sinh-workload/bao-cao.md) |
| P03 | [Công cụ chạy benchmark + kiểm tra rút gọn](03-cong-cu-benchmark/plan.md) | Có | ~1 h (thực tế 0.84 h ≈ 7k VND) | **Xong** 2026-10-13 | [đọc ngay](03-cong-cu-benchmark/doc-ngay.md) · [đầy đủ](03-cong-cu-benchmark/bao-cao.md) |
| P04 | [Đo baseline đầy đủ W1/W2/W3](04-do-baseline-day-du/plan.md) (lưới thích nghi + tìm nhị phân goodput) | Có | ~6.5 h → ước lại từ P03 ~5.1 h (thực tế 4.44 h, ≈ 32.7k VND) | **Xong** 2026-10-13 | [đọc ngay](04-do-baseline-day-du/doc-ngay.md) · [đầy đủ](04-do-baseline-day-du/bao-cao.md) |
| P05 | [W4: tải giống thật (phiên người dùng BurstGPT)](05-w4-tai-giong-that/plan.md) | Có | ~6.5 h → ước lại ~4.4–5.4 h (thực tế 4.97 h, ≈ 49.2k VND) | **Xong** 2026-10-13 | [đọc ngay](05-w4-tai-giong-that/doc-ngay.md) · [đầy đủ](05-w4-tai-giong-that/bao-cao.md) |
| P06 | [Chất lượng baseline + output tham chiếu](06-chat-luong-baseline/plan.md) | Có | ~1.5 h → ước lại ~2.2–2.6 h (thực tế 1.78 h ≈ 15.5k VND + lần 2 0.88 h ≈ 8.4k VND) | **Xong** 2026-10-14 | [đọc ngay](06-chat-luong-baseline/doc-ngay.md) · [đầy đủ](06-chat-luong-baseline/bao-cao.md) |
| P07 | [Tổng hợp baseline, cập nhật kế hoạch](07-tong-hop-baseline/plan.md) | Không | 0 | **Xong** 2026-10-14 (đã chốt cách đo phần tune, mục 6.3) | [đọc ngay](07-tong-hop-baseline/doc-ngay.md) · [đầy đủ](07-tong-hop-baseline/bao-cao.md) |
| | **Tổng** | | **~16.25 h ≈ 166k VND** (thực tế 13.24 h ≈ 116k VND, P07 mục 3.4) | | |

Thứ tự phụ thuộc: P01 → P02 (P02 làm song song được với P01) → P03 → P04 → P05, P06 → P07.

## Chạy tiếp được khi bị dừng

Mọi plan có dùng GPU tuân theo [thi-nghiem-chay-tiep.md](../thi-nghiem-chay-tiep.md): dừng giữa chừng → chạy lại cùng lệnh,
chỉ làm phần còn dở. Vì vậy có thể `destroy` máy để làm việc khác bất kỳ lúc nào mà không mất phần đã đo (kết quả đã kéo về laptop).

## Gộp phiên thuê máy

- **Phiên 1:** P01 riêng — ngắn, để phát hiện sớm vấn đề cài đặt trước khi viết nhiều code.
- **Phiên 2:** P03 → P04 (cùng máy, khỏi tải lại model).
- **Phiên 3:** P05 (W4, ~6.5 h, mỗi lần chạy ~40 phút) → P06.
- Báo cáo viết **sau khi** đã rsync kết quả về máy local và đã tắt máy — không giữ máy chạy trong lúc viết.
- Cố gắng đo P03–P06 trên **cùng một host**; nếu buộc phải đổi host, chạy lại 1 điểm đo chuẩn (W2, 16 user) để so hai host.

## Quy ước thư mục

```
docs/gd1-vllm-baseline/
  README.md                 ← file này: tổng quan + trạng thái
  mau-bao-cao.md            ← quy tắc viết + khung báo cáo
  0X-<ten-plan>/plan.md     ← plan
  0X-<ten-plan>/doc-ngay.md  ← báo cáo dễ đọc, kể từng bước — ĐỌC CÁI NÀY TRƯỚC
  0X-<ten-plan>/bao-cao.md  ← báo cáo đầy đủ số liệu, đọc sau khi cần đi sâu
scripts/                    ← cài đặt máy, khởi động server
workloads/                  ← code sinh workload
bench/                      ← code chạy benchmark + tổng hợp
results/gd1/<plan>/<run_id>/← kết quả thô (JSON, log), rsync từ máy thuê về
```

- Số liệu tóm tắt luôn ghi vào [../measurements.md](../measurements.md) (theo quy tắc chung của dự án).
- Khó khăn đáng kể ghi vào [../challenges.md](../challenges.md).
- Thuật ngữ dùng trong plan/báo cáo: [../thuat-ngu.md](../thuat-ngu.md).

## Khi xong mỗi plan

1. Cập nhật cột **Trạng thái** và **Báo cáo** ở bảng trên.
2. Viết **hai** báo cáo theo [mau-bao-cao.md](mau-bao-cao.md): `doc-ngay.md` (dễ đọc, kể từng bước) và `bao-cao.md` (đầy đủ).
3. Thuật ngữ mới xuất hiện → thêm vào [../thuat-ngu.md](../thuat-ngu.md).
4. `vastai show instances` → chắc chắn không còn máy chạy.
