# P14 — Tổng hợp GĐ1 tune, chuẩn bị GĐ2

> Cần GPU: không · Ước giờ GPU: 0 · Phụ thuộc: P13

## Mục tiêu

Khép GĐ1: một trang so vLLM mặc định với vLLM đã tune, đóng góp của từng cờ, và danh sách việc còn lại cho GĐ2 (thư viện khác), GĐ3 (profile),
GĐ4 (code riêng).

## Các bước

1. **Bảng mốc "vLLM đã tune"** (từ P13), đặt cạnh bảng mốc baseline (P07): goodput W1–W4, TTFT/TPOT p95, VND / 1M token, chất lượng.
2. **Đóng góp từng cờ** (ablation theo thứ tự cộng dồn P09 → P12 → P10 → P11), ghi rõ cái nào đo một lần (sàng lọc) và cái nào đo lặp (P13).
3. **Cập nhật nghi vấn còn lại:** sau tune, chỉ số nào hỏng trước, nghẽn còn ở đâu (KV, prefill, đợt dồn, hàng đợi) → đầu vào cho GĐ3 profile
   và danh sách ứng viên GĐ4 (P07 mục 6.2 cập nhật theo số mới), [library-gaps.md](../../library-gaps.md).
4. **Cập nhật [plan.md](../../plan.md):** giờ GPU thật của GĐ1, ước lại GĐ2–GĐ5, ngân sách.
5. **Lập plan GĐ2** (sàng lọc thư viện) dùng cfg_final làm mốc, rồi dừng báo cáo.
6. Cập nhật CLAUDE.md dòng "Trạng thái hiện tại".

## Tiêu chí xong

- [ ] Bảng mặc định so với đã tune, có nguồn từng số.
- [ ] plan.md cập nhật theo giờ thật của GĐ1.
- [ ] Danh sách việc cho GĐ2–GĐ4 có dẫn chứng từ số đo.

## Báo cáo P14 cần trả lời

1. Tune vLLM trên L4 được bao nhiêu (goodput, chi phí), cờ nào đóng góp nhiều nhất?
2. Sau tune, L4 còn nghẽn ở đâu — đó là chỗ thư viện khác hoặc code riêng phải vượt qua.
