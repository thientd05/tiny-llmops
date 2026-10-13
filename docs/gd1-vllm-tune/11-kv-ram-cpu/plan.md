# P11 — Giữ lịch sử hội thoại ở RAM CPU (W4)

> Cần GPU: có · **Ước 0.9 h ≈ 8k VND** (0.7–1.7 h; +0.35 h nếu phải thuê phiên riêng) · Phụ thuộc: P10 (cfg2) · Phiên 2, nối ngay sau P10

## Vì sao

P07 mục 4.2:
- ở giờ cao điểm W4, **~45% token prompt là lịch sử hội thoại dùng lại được**, nhưng chỉ 3–5% trúng cache thật;
- lịch sử bị đẩy khỏi KV trên GPU vì ~363 phiên đang mở cần ~280k token, gấp ~7 lần KV; người dùng nghỉ trung vị 2 phút giữa hai lượt;
- 280k token KV ≈ 26 GB ở FP16 (≈ 13 GB nếu cfg1 dùng KV FP8), vừa RAM máy thuê (94–125 GB). Chép lại 1,000 token qua PCIe Gen3 (8.4 GB/s)
  ~11 ms (FP16), so với ~370 ms prefill lại;
- vLLM 0.30.0 có sẵn: `--kv-offloading-size <GiB> --kv-offloading-backend native` (references F3). CachedAttention (references F5) dùng cùng ý tưởng
  cho hội thoại nhiều lượt.

Kiểm tra ngược đã biết trước: phút hỏng nặng nhất chủ yếu là tài liệu mới (chỉ 38% là lịch sử), nên cách này giảm tải trung bình chứ không xoá
đợt dồn. Mức lợi cho goodput chỉ biết khi đo.

## Cấu hình

| Tên | Cờ thêm vào cfg2 | Lý do |
|---|---|---|
| off32 | `--kv-offloading-size 32 --kv-offloading-backend native` | 32 GiB > ~26 GB lịch sử ở FP16, dư cho phiên mới. Kiểm RAM container trước khi chạy (P05 thấy 125 GB, P06 94 GB); RAM < 64 GB thì giảm còn 16 |

LMCache (backend `lmcache`) để GĐ2: phải cài thêm thư viện, là bài so sánh thư viện bổ trợ.

## Các bước

1. Khởi động off32. Ghi log offloading (dung lượng thật, lỗi nếu có).
2. **Kiểm đúng sai nhanh:** teacher forcing 200 prompt ngắn (~1 phút). Chép KV ra RAM rồi chép lại không được làm đổi số;
   same top-1 phải ở mức baseline (~99.1%).
3. Một lần chạy W4 peak seed 1 ở mức sàng lọc:
   - nếu cfg2 đã có lần trượt ở P10 (`k_s` hoặc cao hơn) → đo off32 ở mức trượt thấp nhất đó;
   - nếu cfg2 đạt ở mọi mức đã đo (một cấu hình P10 đạt ở `k_s`) → đo **cả** cfg2 và off32 ở `k_s + 2 bước` (thêm một lần chạy, có điều kiện).
4. Kéo kết quả, destroy.

## Số đo chính (so cặp với cfg2, cùng mức, cùng seed)

Đạt / trượt; phần token prompt lấy từ cache GPU / từ RAM CPU (`prompt_tokens_by_source`, P08 việc 6); trúng cache theo think time
(bảng P05 mục 3.7); TTFT p95, % phút hỏng; TPOT p95 (chép KV có thể chen vào bước decode).

## Quy tắc chọn

- off32 đạt ở mức cfg2 trượt → cfg3 = cfg2 + off32.
- Trượt nhưng token prefill thực sự phải tính giảm ≥ 30% và TTFT p95 thấp hơn cfg2 → vẫn nhận (chi phí phần cứng bằng 0, giảm tải trung bình);
  ghi "chưa rõ lợi ích goodput".
- TPOT p95 tăng > 20% hoặc teacher forcing lệch khỏi mức baseline → không nhận, ghi lý do.

## Ước giờ GPU

| Bước | Giờ | Cách tính |
|---|---|---|
| Đổi cấu hình | 0.13 | |
| Teacher forcing 200 prompt | 0.05 | P06: 1.2 phút + khởi chạy |
| Một lần chạy W4 off32 | 0.55 | trung bình đạt 0.68 / trượt 0.40 |
| Kéo kết quả, destroy | 0.05 | |
| **Cộng × 1.1** | **0.9 h ≈ 8k VND** | 0.78 × 1.1 |

Tốt nhất 0.7 h: off32 trượt, dừng sớm (0.40). Xấu nhất 1.7 h: phải đo thêm cfg2 ở `k_s + 2 bước` (+0.68) và off32 đạt (0.68).

## Tiêu chí xong

- [ ] Dung lượng offload thật, teacher forcing đạt.
- [ ] So cặp off32 với cfg2 ở cùng mức W4: đạt / trượt, token tính lại, TTFT, TPOT.
- [ ] cfg3 được chọn theo quy tắc.

## Báo cáo P11 cần trả lời

1. Bao nhiêu % lịch sử được lấy lại từ RAM CPU thay vì tính lại (so với cận trên ~45%)?
2. Goodput W4 có tăng không? Nếu không, đợt dồn tài liệu mới có phải lý do (so phân tích theo phút)?
3. Phần câu trả lời lượt trước (13% token, không bao giờ trúng vì template) còn đáng làm N5 không?
