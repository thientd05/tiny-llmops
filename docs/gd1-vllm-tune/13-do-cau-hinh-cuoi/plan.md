# P13 — Đo đầy đủ cấu hình tune cuối

> Cần GPU: có · **Ước 5.0 h ≈ 44k VND** (4.6–8.4 h) · Phụ thuộc: P09–P12 · Phiên 3

## Mục tiêu

Cấu hình tune cuối (cfg_final = cfg3, gồm mọi cờ đã nhận ở P09–P12) là **mốc công bằng** cho GĐ2 và GĐ4. Các số sàng lọc ở P09–P11 đo một lần,
có thể "may" (chọn người thắng từ nhiều cấu hình đo có nhiễu thì số của nó bị đẩy lên; Cawley & Talbot 2010, references G2).
P13 đo lại cfg_final với số lần lặp như baseline P04/P05, để so với baseline cho chặt và làm mốc dài hạn.

Nếu P10 có cấu hình nhóm "chỉnh ứng dụng" (prio) tốt hơn, đo thêm nó ở W4 (mục "Có điều kiện").

## Các bước

1. Thuê (ưu tiên machine 116596), setup, điểm mốc W2/16 ×3 với baseline.
2. **W1–W3 ×3 lần lặp** (seed 1001–1003), khởi đầu nóng ở (lo, hi) của cfg_final từ P09/P12, tìm tới `hi − lo ≤ max(2, 10% × lo)`,
   đạt/trượt theo trung vị 3 lần (như P04). Khởi động lại server giữa các lần lặp.
3. **W4 peak seed 1:** khởi đầu nóng ở (lo, hi) từ P10/P11, độ mịn như P05 (`max(2, 10% × lo)`).
4. **W4 peak seed 2** ở k = lo (bộ ngày khác, như P05). Trượt → thêm một lần ở lo − 2.
5. **Cổng chất lượng ×1** cho cfg_final, đồng thời lưu output tham chiếu mới (dùng để so kernel GĐ4 và thư viện GĐ2 với vLLM đã tune).
6. Kéo kết quả, destroy.

## Có điều kiện (quyết định trong phiên theo quy tắc đặt trước)

| Điều kiện | Việc thêm | Giờ |
|---|---|---|
| Goodput W4 cfg_final ≤ baseline + 1 bước (seed 1) | Chạy lại baseline W4 seed 1 ở k = 18 và 20 trong cùng phiên, để biết chênh là thật hay do lần chạy (P07 rủi ro R1) | 0.13 + 0.68–1.1 |
| P10 có cấu hình "chỉnh ứng dụng" đạt ở `k_s` | Đo nó ở W4 seed 1 tại lo và lo + 2 bước của cfg_final | 0.13 + 0.55–1.36 |
| Cổng chất lượng có dấu hiệu kém đi (định nghĩa ở P09) | Chạy lại cổng một lần | 0.78 |

## Ước giờ GPU

| Bước | Giờ | Cách tính |
|---|---|---|
| Thuê + setup + điểm mốc | 0.35 | |
| Khởi động cfg_final | 0.13 | |
| W1–W3 ×3, trung bình 2.5 mức mỗi workload | 1.47 | 2.5 mức × 3 lần × (W1 3.5 + W2 2.2 + W3 5.5 phút) = 84 phút + 2 lần khởi động lại giữa các lần lặp (4 phút) |
| W4 seed 1: lo đạt + hi trượt | 1.10 | 0.68 + 0.37 (+ đôi khi một lần nữa) |
| W4 seed 2 ở lo | 0.68 | |
| Cổng chất lượng ×1 | 0.78 | |
| Kéo kết quả, destroy | 0.05 | |
| **Cộng × 1.1** | **5.0 h ≈ 44k VND** | |

- Tốt nhất 4.6 h: mọi mức khởi đầu nóng xác nhận ngay (2 mức mỗi workload, 1.18 h).
- Xấu nhất 8.4 h: W1–W3 phải tìm thêm (2.0 h), W4 seed 1 ba lần (1.7 h), seed 2 trượt (+0.68), phải chạy lại baseline W4 (+1.23).
  Chưa tính cổng chất lượng lần 2 và cấu hình "chỉnh ứng dụng" (tối đa +2.3 h nếu cả hai xảy ra).

## Tiêu chí xong

- [ ] Goodput W1 / W2 / W3 (×3, trung vị) và W4 (seed 1, seed 2) của cfg_final, dạng "lo (hi trượt)", kèm khoảng theo từng lần lặp.
- [ ] Chi phí VND / 1M token ra mỗi workload và W4 cả ngày, giá chung 8,770 VND/h.
- [ ] Chất lượng cfg_final so với hai lần baseline P06; output tham chiếu mới đã lưu.
- [ ] Số ghi vào measurements.md.

## Báo cáo P13 cần trả lời

1. vLLM đã tune phục vụ được bao nhiêu user mỗi workload, so với mặc định (bao nhiêu bước, có vượt dao động giữa các lần lặp không)?
2. Chi phí mỗi triệu token giảm bao nhiêu?
3. Các số sàng lọc P09–P11 có lặp lại được không (kiểm "lệch do chọn người thắng")?
4. Chất lượng có giữ nguyên trong mức dao động của baseline không?
