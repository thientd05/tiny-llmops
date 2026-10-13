# Tổng hợp `results/gd1/09-tang-cho-kv/20261014-p09/p12-seqs`

Cấu hình `bench/configs/tune/p12-seqs.yaml` · 2 đơn vị đã xong · lặp 1 lần/mức · cửa sổ quyết định đạt/trượt: **all** (all = mọi request; steady = bỏ N request gửi đầu và N request gửi cuối)

## Các lần khởi động server

| # | thời điểm | lý do | sẵn sàng (s) | dùng lại | KV cache (token) | KV (GiB) | torch.compile (s) |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-13T21:26:46+00:00 | chưa chạy | 182.5 | False | 72,368 | 3.31 | 64.21 |
| 2 | 2026-10-13T21:28:42+00:00 | bỏ lần khởi động compile từ đầu | 113.7 | False | 89,488 | 4.1 | 1.27 |

## Goodput (số user đồng thời lớn nhất đạt SLO, trung vị các lần lặp)

| workload | cửa sổ | SLO chính (10 s / 150 ms) | SLO phụ (2 s / 100 ms) | SLO TTFT ≤ 2 s (2 s / 150 ms) | SLO TTFT ≤ 5 s (5 s / 150 ms) | SLO TTFT ≤ 10 s (10 s / 150 ms) | SLO TTFT ≤ 20 s (20 s / 150 ms) |
|---|---|---|---|---|---|---|---|
| w1 | all | 140 (164 trượt) | < 140 | < 140 | < 140 | 140 (164 trượt) | ≥ 164 |
| w1 | steady | 140 (164 trượt) | < 140 | < 140 | 140 (164 trượt) | 140 (164 trượt) | ≥ 164 |

SLO phụ / độ nhạy: chỉ có độ phân giải của các mức đã đo (tìm nhị phân chạy theo SLO chính).

## Từng mức N (trung vị các lần lặp, cửa sổ all; [steady] trong ngoặc)

| w | N | client | lần | TTFT p50 / p95 (ms) [steady p95] | TPOT p50 / p95 (ms) [steady p95] | ITL p95 | tok/s ra | tok/s vào | SLO chính | đang chạy max | đang chạy p50 | chờ max | KV dùng max % | preempt | trúng prefix % | lỗi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1 | 140 | vllm | 1 | 2,376 / 7,958 [3,544] | 111.5 / 116.6 [116.5] | 169.9 | 1,077 | 1,330 | đạt | 133 | 132 | 124 | 98 | 0 | 1.5 | 0 |
| w1 | 164 | vllm | 1 | 9,081 / 12,822 [11,695] | 112.1 / 117.0 [117.0] | 173.4 | 1,082 | 1,327 | trượt | 133 | 133 | 151 | 99 | 0 | 1.4 | 0 |

## Nhiễu giữa các lần lặp ((max − min) / trung vị, %)

(chưa có mức nào lặp ≥ 2 lần)

## TTFT client vs server (/metrics, trung bình)

| đơn vị | client (ms) | server (ms) | chênh (ms) | chênh % |
|---|---|---|---|---|
| w1-c140-r1 | 3,167.5 | 3,073.3 | 94.1 | 3.1 |
| w1-c164-r1 | 9,073.1 | 8,960.2 | 112.9 | 1.3 |

## Cửa sổ all vs steady (p95, chênh % so với all)

| đơn vị | req all | req steady | TTFT p95 all | steady | Δ% | TPOT p95 all | steady | Δ% |
|---|---|---|---|---|---|---|---|---|
| w1-c140-r1 | 560 | 280 | 7,958 | 3,544 | -55.5 | 116.6 | 116.5 | -0.1 |
| w1-c164-r1 | 656 | 328 | 12,822 | 11,695 | -8.8 | 117.0 | 117.0 | 0.1 |

## Thời gian đo

| đơn vị | request | client (s) | cả đơn vị (s) | s/request | token vào TB | token ra TB | W max | xung SM min | °C max |
|---|---|---|---|---|---|---|---|---|---|
| w1-c140-r1 | 560 | 194.21 | 198.26 | 0.35 | 411 | 333 | 73.6 | 1,005 | 69 |
| w1-c164-r1 | 656 | 225.22 | 229.55 | 0.35 | 409 | 334 | 74.8 | 915 | 72 |

Tổng thời gian các đơn vị: 0.12 h.

