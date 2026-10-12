# P07 — Tổng hợp baseline, cập nhật kế hoạch

> Cần GPU: không · Phụ thuộc: P04, P05, P06

## Mục tiêu

Gom kết quả P01–P06 thành **một trang mốc baseline** dùng cho cả đồ án, và dùng số thật để sửa lại kế hoạch/ngân sách.

## Các bước

1. **Bảng mốc baseline** (đặt trong báo cáo P07, tóm tắt vào measurements.md):
   goodput W1/W2/W3, throughput ở từng mức user, TTFT/TPOT p95, goodput W4 tải giống thật (P05) + chi phí cả ngày, điểm chất lượng (P06).
2. **Chi phí VND / 1 triệu token output** cho từng workload, tại mức goodput:
   `giá thuê thực tế (VND/giờ, gồm ổ đĩa + băng thông) ÷ số triệu token output sinh được mỗi giờ`.
   Ghi rõ giá thuê lấy từ đâu (hoá đơn Vast của các phiên đã chạy).
3. **So với giả định** trong [plan.md](../../plan.md): throughput, prefill, giờ GPU thực tế đã tiêu cho P01–P06 so với ước tính.
   Cập nhật plan.md (bảng giả định + giờ GPU các giai đoạn) — giữ dòng cũ, ghi "đã cập nhật ngày ... theo số đo GĐ1".
4. **Danh sách nghi vấn để tune** (đầu vào cho plan "tune vLLM tối đa"): mỗi nghi vấn ghi dấu hiệu quan sát được + tham số
   sẽ thử. Ví dụ dự kiến (chỉ giữ cái có dấu hiệu thật):
   - KV cache không đủ → preemption ở W2/W3 → thử KV FP8, chỉnh `gpu_memory_utilization`, `max_num_seqs`;
   - không có config MoE cho L4 → chạy script tune MoE ([library-gaps.md](../../library-gaps.md) #4);
   - TTFT cao ở W2 → xem `max_num_batched_tokens` (kích thước mỗi phần prefill), prefix caching;
   - TPOT cao ở batch lớn → thử speculative decoding.
5. Cập nhật CLAUDE.md: dòng "Trạng thái hiện tại".

## Tiêu chí xong

- [ ] Bảng mốc baseline + chi phí VND/1M token hoàn chỉnh.
- [ ] plan.md đã cập nhật theo số thật.
- [ ] Danh sách nghi vấn tune có dẫn chứng → sẵn sàng lập plan phần tune của GĐ1.

## Báo cáo P07 cần trả lời

1. Với vLLM mặc định trên 1 L4: phục vụ được bao nhiêu user mỗi workload, mỗi triệu token tốn bao nhiêu VND.
2. Kế hoạch ban đầu đoán đúng/sai ở đâu, ngân sách còn lại có đủ cho các giai đoạn sau không.
3. 3–5 hướng tune đáng thử nhất, xếp theo lợi ích dự kiến và có dẫn chứng từ số đo.
