# Tổng hợp `/root/results/p09/anchor`

Cấu hình `bench/configs/tune/p09-anchor.yaml` · 3 đơn vị đã xong · lặp 1 lần/mức · cửa sổ quyết định đạt/trượt: **all** (all = mọi request; steady = bỏ N request gửi đầu và N request gửi cuối)

## Các lần khởi động server

| # | thời điểm | lý do | sẵn sàng (s) | dùng lại | KV cache (token) | KV (GiB) | torch.compile (s) |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-13T19:22:09+00:00 | chưa chạy | 371.2 | False | 28,704 | 2.63 | 66.3 |
| 2 | 2026-10-13T19:24:41+00:00 | bỏ lần khởi động compile từ đầu | 150.5 | False | 37,264 | 3.41 | 1.37 |

## Goodput (số user đồng thời lớn nhất đạt SLO, trung vị các lần lặp)

| workload | cửa sổ | SLO chính (10 s / 150 ms) | SLO phụ (2 s / 100 ms) | SLO TTFT ≤ 2 s (2 s / 150 ms) | SLO TTFT ≤ 5 s (5 s / 150 ms) | SLO TTFT ≤ 10 s (10 s / 150 ms) | SLO TTFT ≤ 20 s (20 s / 150 ms) |
|---|---|---|---|---|---|---|---|
| w2 | all | ≥ 16 | < 16 | < 16 | < 16 | ≥ 16 | ≥ 16 |
| w2 | steady | ≥ 16 | < 16 | < 16 | ≥ 16 | ≥ 16 | ≥ 16 |

SLO phụ / độ nhạy: chỉ có độ phân giải của các mức đã đo (tìm nhị phân chạy theo SLO chính).

## Từng mức N (trung vị các lần lặp, cửa sổ all; [steady] trong ngoặc)

| w | N | client | lần | TTFT p50 / p95 (ms) [steady p95] | TPOT p50 / p95 (ms) [steady p95] | ITL p95 | tok/s ra | tok/s vào | SLO chính | đang chạy max | đang chạy p50 | chờ max | KV dùng max % | preempt | trúng prefix % | lỗi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w2 | 16 | vllm | 3 | 1,479 / 5,789 [3,681] | 82.2 / 91.6 [90.9] | 363.1 | 169 | 2,797 | đạt | 16 | 15 | 13 | 100 | 6 | 21.6 | 0 |

## Nhiễu giữa các lần lặp ((max − min) / trung vị, %)

| w | N | client | lần | tok/s ra | TTFT p50 | TTFT p95 | TTFT p95 steady | TPOT p50 | TPOT p95 | TPOT p95 steady | E2EL p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w2 | 16 | vllm | 3 | 3.1 | 66.7 | 16.3 | 64.0 | 3.8 | 2.9 | 1.2 | 13.9 |

## TTFT client vs server (/metrics, trung bình)

| đơn vị | client (ms) | server (ms) | chênh (ms) | chênh % |
|---|---|---|---|---|
| w2-c16-r1 | 1,981.0 | 1,958.7 | 22.2 | 1.1 |
| w2-c16-r2 | 2,015.6 | 1,992.6 | 23.1 | 1.2 |
| w2-c16-r3 | 2,721.5 | 2,698.5 | 23.0 | 0.9 |

## Cửa sổ all vs steady (p95, chênh % so với all)

| đơn vị | req all | req steady | TTFT p95 all | steady | Δ% | TPOT p95 all | steady | Δ% |
|---|---|---|---|---|---|---|---|---|
| w2-c16-r1 | 100 | 68 | 5,789 | 2,658 | -54.1 | 90.9 | 90.5 | -0.4 |
| w2-c16-r2 | 100 | 68 | 4,940 | 3,681 | -25.5 | 93.5 | 91.6 | -2.0 |
| w2-c16-r3 | 100 | 68 | 5,883 | 5,014 | -14.8 | 91.6 | 90.9 | -0.7 |

## Thời gian đo

| đơn vị | request | client (s) | cả đơn vị (s) | s/request | token vào TB | token ra TB | W max | xung SM min | °C max |
|---|---|---|---|---|---|---|---|---|---|
| w2-c16-r1 | 100 | 123.14 | 125.6 | 1.26 | 2821 | 174 | 73.6 | 1,140 | 69 |
| w2-c16-r2 | 100 | 119.18 | 121.52 | 1.22 | 2818 | 167 | 74.4 | 1,095 | 73 |
| w2-c16-r3 | 100 | 128.2 | 130.52 | 1.31 | 2923 | 180 | 73.3 | 1,125 | 74 |

Tổng thời gian các đơn vị: 0.10 h.

