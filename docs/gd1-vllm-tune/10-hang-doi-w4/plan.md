# P10 — Câu dài chặn câu ngắn ở hàng đợi (W4)

> Cần GPU: có · **Ước 6.0 h ≈ 53k VND** (4.6–8.3 h) · Phụ thuộc: P09 (cfg1) · Phiên 2, P11 nối ngay sau

## Vì sao

W4 (tải giống thật, bài test cuối) hỏng theo đợt, và câu ngắn bị câu dài làm chậm (P07 mục 4.3):
- câu < 1k token gửi lúc có câu ≥ 4k đang chờ: TTFT p95 26–32 s, so với 1.4–4.2 s;
- cùng độ dài hàng đợi, câu ngắn vẫn chậm gấp ~2 (trung vị) khi có câu dài → không chỉ do đông;
- cơ chế thấy trong mã vLLM 0.30.0: hàng đợi xét theo thứ tự đến, câu mới chỉ được nhận khi **cả prompt** vừa KV trống
  (`scheduler_reserve_full_isl`, mặc định bật), câu đầu hàng không vừa thì cả hàng phía sau phải chờ (`break`).
- Mỗi bước vLLM xử lý tối đa 2,048 token (mặc định trên L4); prompt dài chiếm ngân sách nhiều bước liền. TPOT còn dư (82–104 ms so với 150),
  Sarathi-Serve khuyên đặt ngân sách lớn nhất mà không vượt SLO độ trễ giữa các token (references F4).

## Cấu hình

**Bước A — mốc W4 mới:** cfg1 (cấu hình thắng P09) chưa có số W4. Tìm goodput W4 của cfg1 (peak seed 1, ×1, khởi đầu nóng k = [18, 22], mở rộng lên ×1.25 thay vì ×2, dừng khi `hi − lo ≤ 4`).
Độ mịn 4 (2 bước) là đủ ở đây: bước B chỉ cần một mức cfg1 trượt; goodput mịn của cấu hình cuối đo ở P13.
Kết quả này vừa là số W4 của "tăng KV", vừa là mốc để sàng lọc bước B.

**Bước B — sàng lọc,** mỗi cấu hình = cfg1 + một cờ, đo **một lần chạy W4** ở mức `k_s` = mức thấp nhất mà cfg1 trượt và ≥ `lo1 + 2 bước`:

| Tên | Cờ thêm | Cơ chế mong đợi |
|---|---|---|
| nofull | `--scheduler-reserve-full-isl` tắt (cú pháp tắt kiểm ở P08) | Nhận câu mới khi khúc đầu vừa chỗ → câu ngắn không phải chờ câu dài đủ KV cho cả prompt. Rủi ro: nhận quá nhiều → preemption (chú thích trong mã) |
| mnbt4k | `--max-num-batched-tokens 4096` | Prefill câu dài xong trong ít bước hơn; TPOT tăng (còn dư) |
| lpt1k | `--long-prefill-token-threshold 1024` | Mỗi bước, một câu dài lấy tối đa 1,024 token → còn ≥ 1,024 token cho câu khác |
| prio | `--scheduling-policy priority` + client `--priority-by-length` (< 1k: 0, 1–4k: 1, ≥ 4k: 2) | Câu ngắn được xét trước câu dài. **Nhóm "chỉnh ứng dụng"**: client phải gán độ ưu tiên |

**Bước C — gộp (có điều kiện):** nếu ≥ 2 cấu hình đạt ở `k_s`, đo cấu hình gộp các cờ đạt ở `k_s + 2 bước` (một lần chạy).
`prio` chỉ gộp vào nhóm "chỉnh ứng dụng", không vào cfg2 thuần vLLM.

## Số đo ngoài đạt / trượt (so cặp với cfg1 cùng `k_s`, cùng seed)

TTFT p95 cả cửa sổ; % phút có TTFT p95 > 10 s; TTFT p50 / p95 của câu < 1k khi có / không có câu ≥ 4k đang chờ (cách phân nhóm của P07);
TPOT p95; preemption; trúng prefix cache. Lấy bằng `bench/compare.py` (P08).

## Quy tắc chọn (đặt trước)

- Đạt ở `k_s` → cải thiện rõ (≥ 2 bước so với lo1). Nhiều cấu hình đạt → bước C; cfg2 = cấu hình gộp nếu đạt, không thì cấu hình đạt có TTFT p95 thấp nhất.
- Trượt ở `k_s` nhưng TTFT p95 của câu ngắn có câu dài đang chờ giảm ≥ 50% so với cfg1 → ghi "có tác dụng lên cơ chế, chưa đủ cho goodput", không nhận.
- Không cấu hình nào đạt → cfg2 = cfg1. Khi đó ứng viên GĐ4 "bộ xếp lịch không để câu dài chặn đầu hàng" có thêm lý do (P07 mục 6.2).
- mnbt4k có thể làm W2/W3 đổi; không đo ở đây, P13 đo W1–W3 đầy đủ trên cấu hình cuối.

## Ước giờ GPU

| Bước | Giờ | Cách tính |
|---|---|---|
| Thuê + setup + điểm mốc W2/16 ×3 | 0.35 | 0.25 + 0.10 |
| Khởi động cfg1 | 0.13 | |
| A: tìm k cho cfg1, dự kiến 4 lần chạy | 2.25 | Ví dụ k = 18 đạt, 22 đạt, 28 trượt sớm, 26 đạt hoặc trượt → xong: 2–3 lần đạt × 0.68 + 1–2 lần trượt × 0.37 = 2.1–2.4 |
| B: 4 cấu hình × (đổi cấu hình + một lần chạy) | 2.72 | 4 × (0.13 + 0.55); 0.55 = trung bình đạt 0.68 / trượt 0.40 |
| Kéo kết quả (P11 nối tiếp nên chưa destroy) | 0.05 | |
| **Cộng × 1.1** | **6.0 h ≈ 53k VND** | |

- Tốt nhất 4.6 h: cfg1 không đổi goodput W4 (A chỉ 3 lần chạy, 1.5 h), mọi cấu hình B trượt sớm.
- Xấu nhất 8.3 h: A cần 5 lần chạy, 3 đạt (2.8 h), thêm một lần do kết quả không đơn điệu (+0.2), mọi cấu hình B đạt (0.81 mỗi cái), có bước C (+0.81).

## Chạy tiếp được

Mỗi lần chạy W4 là một đơn vị 40 phút (đã có từ P05). Danh sách cấu hình chạy bằng `scripts/remote_multi.sh` (P08); dừng ở đâu chạy lại cùng lệnh.
Bước B phụ thuộc kết quả bước A (`k_s`): Claude viết cấu hình bước B khi A xong (đang theo dõi phiên, không cần dừng máy).

## Tiêu chí xong

- [ ] Goodput W4 của cfg1 (seed 1, ×1).
- [ ] Bảng so cặp 4 cấu hình với cfg1 ở `k_s`.
- [ ] cfg2 được chọn theo quy tắc, có lý do.

## Báo cáo P10 cần trả lời

1. Tăng KV (cfg1) đẩy goodput W4 lên bao nhiêu? Đợt dồn token prefill còn là nguyên nhân hỏng không?
2. Cờ nào giảm được việc câu dài chặn câu ngắn, giảm bao nhiêu, có thành goodput không?
3. Cơ chế "chặn đầu hàng" có được xác nhận (nofull / prio giảm TTFT câu ngắn) hay nguyên nhân chính là ngân sách token mỗi bước (mnbt4k / lpt1k)?
4. Còn lại gì cho GĐ4 (bộ xếp lịch riêng)?
