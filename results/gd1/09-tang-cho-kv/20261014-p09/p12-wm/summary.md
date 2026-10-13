# Tổng hợp `results/gd1/09-tang-cho-kv/20261014-p09/p12-wm`

Cấu hình `bench/configs/tune/p12-wm.yaml` · 2 đơn vị đã xong · lặp 1 lần/mức · cửa sổ quyết định đạt/trượt: **all** (all = mọi request; steady = bỏ N request gửi đầu và N request gửi cuối)

## Các lần khởi động server

| # | thời điểm | lý do | sẵn sàng (s) | dùng lại | KV cache (token) | KV (GiB) | torch.compile (s) |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-13T21:16:33+00:00 | chưa chạy | 138.1 | False | 81,328 | 3.72 | 0.44 |

## Goodput (số user đồng thời lớn nhất đạt SLO, trung vị các lần lặp)

| workload | cửa sổ | SLO chính (10 s / 150 ms) | SLO phụ (2 s / 100 ms) | SLO TTFT ≤ 2 s (2 s / 150 ms) | SLO TTFT ≤ 5 s (5 s / 150 ms) | SLO TTFT ≤ 10 s (10 s / 150 ms) | SLO TTFT ≤ 20 s (20 s / 150 ms) |
|---|---|---|---|---|---|---|---|
| w1 | all | 140 (164 trượt) | < 140 | < 140 | < 140 | 140 (164 trượt) | ≥ 164 |
| w1 | steady | 140 (164 trượt) | < 140 | < 140 | < 140 | 140 (164 trượt) | ≥ 164 |

SLO phụ / độ nhạy: chỉ có độ phân giải của các mức đã đo (tìm nhị phân chạy theo SLO chính).

## Từng mức N (trung vị các lần lặp, cửa sổ all; [steady] trong ngoặc)

| w | N | client | lần | TTFT p50 / p95 (ms) [steady p95] | TPOT p50 / p95 (ms) [steady p95] | ITL p95 | tok/s ra | tok/s vào | SLO chính | đang chạy max | đang chạy p50 | chờ max | KV dùng max % | preempt | trúng prefix % | lỗi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1 | 140 | vllm | 1 | 2,523 / 8,319 [8,158] | 110.1 / 115.9 [114.4] | 193.2 | 1,076 | 1,329 | đạt | 140 | 131 | 125 | 100 | 10 | 1.3 | 0 |
| w1 | 164 | vllm | 1 | 9,029 / 14,375 [15,198] | 111.1 / 127.9 [115.6] | 193.4 | 1,079 | 1,324 | trượt | 164 | 130 | 146 | 100 | 38 | 1.4 | 0 |

## Nhiễu giữa các lần lặp ((max − min) / trung vị, %)

(chưa có mức nào lặp ≥ 2 lần)

## TTFT client vs server (/metrics, trung bình)

| đơn vị | client (ms) | server (ms) | chênh (ms) | chênh % |
|---|---|---|---|---|
| w1-c140-r1 | 3,402.4 | 3,310.4 | 91.9 | 2.8 |
| w1-c164-r1 | 8,865.8 | 8,755.3 | 110.5 | 1.3 |

## Cửa sổ all vs steady (p95, chênh % so với all)

| đơn vị | req all | req steady | TTFT p95 all | steady | Δ% | TPOT p95 all | steady | Δ% |
|---|---|---|---|---|---|---|---|---|
| w1-c140-r1 | 560 | 280 | 8,319 | 8,158 | -1.9 | 115.9 | 114.4 | -1.3 |
| w1-c164-r1 | 656 | 328 | 14,375 | 15,198 | 5.7 | 127.9 | 115.6 | -9.6 |

## Thời gian đo

| đơn vị | request | client (s) | cả đơn vị (s) | s/request | token vào TB | token ra TB | W max | xung SM min | °C max |
|---|---|---|---|---|---|---|---|---|---|
| w1-c140-r1 | 560 | 195.2 | 199.28 | 0.36 | 411 | 333 | 75.5 | 1,050 | 71 |
| w1-c164-r1 | 656 | 225.24 | 229.68 | 0.35 | 409 | 334 | 73.1 | 945 | 72 |

Tổng thời gian các đơn vị: 0.12 h.

