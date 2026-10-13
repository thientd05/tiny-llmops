# P12 — Preemption ở W1 (có điều kiện)

> Cần GPU: có · **Ước 0.55 h ≈ 5k VND, hoặc 0 nếu không thoả điều kiện** · Phụ thuộc: P09 · Chạy nối ngay sau P09, cùng phiên 1

## Vì sao có điều kiện

Baseline W1 có 83–151 lần preemption mỗi lần đo ở 64–72 user (vLLM tạm dừng request đang chạy vì hết KV, tính lại sau).
Chi phí thật chưa đo được: bộ đếm của vLLM không tính phần tính lại, ước 0–18% thời gian GPU (P07 mục 4.4).
KV FP8 ở P09 gấp đôi chỗ KV nên preemption có thể tự hết. Đo tiếp chỉ khi vẫn còn nhiều.

**Điều kiện chạy (đặt trước):** ở mức goodput W1 của cfg1 (P09), số preemption ≥ 50 trong lần đo đó (baseline: 133 ở 68 user).
Không thoả → bỏ P12, ghi số preemption vào báo cáo P09.

## Cấu hình (trên nền cfg1)

| Tên | Cờ thêm | Lý do |
|---|---|---|
| wm | `--watermark 0.02` | Để dành 2% KV khi nhận request mới; mã vLLM 0.30.0 ghi rõ mục đích là tránh preemption lặp (references F3). 2% ≈ 1,800 token (KV FP8 + 0.95 ước ~89k token) ≈ 3 request W1 |
| seqs | `--max-num-seqs <số request chạy p50 của cfg1 ở W1 lo>` | Không nhận quá số request KV chứa được; tài liệu vLLM gợi ý giảm `max_num_seqs` khi preemption nhiều (references F1) |

## Các bước

Mỗi cấu hình đo W1 ×1, seed 1001, ở hai mức: `lo` của cfg1 (kiểm tra không kém đi) và `lo + 2 bước` (kiểm tra cải thiện rõ).

## Quy tắc chọn

- Đạt ở `lo + 2 bước` và đạt ở `lo` → nhận vào cfg (lấy cấu hình có TTFT p95 thấp hơn ở `lo + 2 bước` nếu cả hai đạt).
- Chỉ đạt ở `lo`, số preemption giảm > 50% nhưng TTFT không tốt hơn → không nhận (không có lợi cho chỉ số mục tiêu), ghi lại.
- Không ảnh hưởng W2, W3 (P07: preemption ở đó chỉ 0–19 lần); P13 đo lại cả ba nên sẽ thấy nếu có tác dụng phụ.

## Ước giờ GPU

| Bước | Giờ | Cách tính |
|---|---|---|
| 2 lần đổi cấu hình | 0.26 | 2 × 0.13 |
| 4 đơn vị W1 ở N ~80–100 | 0.24 | 4 × 3.6 phút |
| **Cộng × 1.1** | **0.55 h ≈ 5k VND** | |

## Tiêu chí xong

- [ ] Có số preemption, TTFT p95, throughput của wm và seqs ở hai mức, so cặp với cfg1.
- [ ] Quyết định nhận / không nhận ghi trong báo cáo (báo cáo chung với P09 nếu chạy cùng phiên).

## Báo cáo P12 cần trả lời

1. Sau KV FP8, preemption còn bao nhiêu?
2. Bớt preemption có làm TTFT p95 ở mức tải cao tốt hơn không, hay chỉ đổi chỗ chờ (từ tính lại sang hàng đợi)?
