# P05 — Phát lại trace tải thật (BurstGPT)

> Cần GPU: có · Ước tính ~1.25 giờ GPU (~13k VND) · Phụ thuộc: P04 (cần biết năng lực để chọn mức tải)

## Mục tiêu

P04 đo với số user cố định. Thực tế request đến **không đều**: có lúc dồn dập, có lúc vắng. P05 kiểm tra
vLLM mặc định giữ SLO được không khi request đến theo thời điểm của một trace sản xuất thật.

## Thiết kế

- **Thời điểm đến:** lấy từ BurstGPT (trace Azure OpenAI thật). Chọn 1 đoạn 30 phút **có đột biến tải rõ** (ghi rõ đoạn nào, tiêu chí chọn).
- **Nội dung request:** lấy từ W2 (workload chính). BurstGPT chỉ dùng phần thời điểm đến.
- **Chỉnh tốc độ:** nhân/chia khoảng cách thời gian giữa các request để tải trung bình bằng:
  - mức A: ~50% năng lực đo được ở P04 (tải bình thường);
  - mức B: ~80% (tải cao — đỉnh đột biến sẽ vượt năng lực).
  Tổng: 2 mức × 30 phút.
- **Công cụ gửi:** kiểm tra `vllm bench serve` phiên bản đã ghim có gửi theo danh sách thời điểm cho trước được không.
  - Có → dùng luôn.
  - Không → viết client nhỏ `bench/replay.py` (Python async, đọc output dạng stream để đo TTFT/TPOT).
    **Kiểm chứng trước khi dùng:** chạy client này ở chế độ số user cố định (W2, 16 user) và so với `vllm bench serve`;
    lệch < 5% mới dùng.

## Các bước

1. Ở local: chuẩn bị file trace (đoạn chọn + 2 mức tốc độ), ghép với mẫu W2; viết/kiểm tra công cụ gửi.
2. Trên L4 (có thể nối tiếp phiên P04): reset prefix cache, chạy mức A, rồi mức B. Ghi `nvidia-smi` và `/metrics` như P03.
3. Rsync, destroy (nếu không làm tiếp P06).
4. Phân tích ở local: chia thời gian thành từng khoảng 1 phút → TTFT p95, TPOT p95, số request đang chờ, % request đạt SLO trong từng phút.

## Kết quả cần có

| Bảng / hình | Nội dung |
|---|---|
| Hình | Theo thời gian: số request đến mỗi phút (trên) và TTFT p95 / TPOT p95 mỗi phút (dưới), kẻ đường SLO |
| Bảng | Mỗi mức tải: % request đạt SLO cả đoạn, % đạt SLO trong các phút cao điểm, thời gian hồi phục sau đỉnh |

## Tiêu chí xong

- [ ] Công cụ gửi theo trace đã kiểm chứng (nếu tự viết).
- [ ] Đủ 2 mức tải, có hình theo thời gian; số tóm tắt ghi measurements.md.

## Báo cáo P05 cần trả lời

1. Ở tải trung bình 50% và 80% năng lực, bao nhiêu % request đạt SLO? Có phút nào trượt nặng không?
2. Khi có đột biến, hàng đợi tăng bao nhiêu và mất bao lâu để trở lại bình thường?
3. Kết quả có khớp dự đoán từ P04 không (ví dụ: đỉnh vượt goodput thì TTFT tăng)? Nếu không khớp, vì sao.
