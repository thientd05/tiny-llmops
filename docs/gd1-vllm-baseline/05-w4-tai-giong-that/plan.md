# P05 — W4: tải giống thật (phiên người dùng BurstGPT)

> Cần GPU: có · Ước tính ~6.5 giờ GPU (~67k VND) → **ước lại sau P04: ~5–5.6 h (~38–43k VND)** · Phụ thuộc: P03 (client phiên), P04 (năng lực để chọn k khởi đầu)
>
> **Từ P04 (2026-10-13):** trần ước tính k ≈ 22 (phép tính từ throughput bão hoà + KV, xem báo cáo P04 mục 6) → khởi đầu nóng k = 16 / 24.
> Trước khi thuê máy: thêm chế độ W4 vào `bench/sweep.py` (đơn vị = khung giờ × k × seed, `done.json`, `--status`, dùng `search_step`).
>
> 2026-10-12: thay thiết kế cũ "phát lại 30 phút BurstGPT với nội dung W2". Lý do và thiết kế đầy đủ:
> [workloads.md — W4](../../workloads.md#w4--tải-giống-thật-bài-test-cuối).

## Mục tiêu

P04 đo các ca "thuần": N người gửi liên tục, không nghỉ, cùng một loại request. Người dùng thật khác hẳn:
họ mở đầu bằng câu ngắn, hội thoại dài dần, đọc rồi nghỉ mới hỏi tiếp, có lúc dán tài liệu dài, lúc đông lúc vắng,
và tỉ lệ request ngắn/dài đổi theo từng giờ. P05 đo **vLLM mặc định phục vụ được bao nhiêu người dùng thật** ở giờ cao điểm.

## Thiết kế (đã có code: `workloads/build_w4.py`, `workloads/w4_session.py`)

- **Nguồn hành vi:** phiên hội thoại (`Conversation log`) của BurstGPT_3: 55,920 phiên thật, 110 ngày. Bỏ API log (chương trình gọi).
- **Khung giờ chọn theo quy tắc** (giờ trong ngày, tải trung bình của trace): **peak = 16h** (208 req/giờ), **peak2 = 10h** (180),
  **normal = 13h** (92). Giờ vắng (5h, 4 req/giờ) không đo, tính bằng công thức.
- **Mỗi lần chạy = 10 phút khởi động + 30 phút đo** (chỉ tính request gửi trong 30 phút đo).
- **Tải = k:** chồng phiên của cùng khung giờ từ k ngày khác nhau (k ngày ≈ k lần số người dùng), giữ nguyên think time.
  k ≤ 110. Ngày lấy theo hoán vị cố định theo seed → kịch bản k nằm trong k' > k, tải tăng đơn điệu theo k.
- **Client phiên (partly-open):** lượt đầu gửi đúng `start_s`; lượt sau gửi khi lượt trước **nhận xong** + `think_s`.
  Prompt = lịch sử thật (gồm câu trả lời server vừa sinh) + tin nhắn mới, qua `w4_session.assemble`; vượt `prompt_budget`
  thì bỏ cặp hỏi–đáp cũ nhất. Stream để đo TTFT/TPOT; `max_tokens` + `ignore_eos` theo trace.
- **Đạt SLO:** TTFT p95 ≤ 10 s và TPOT p95 ≤ 150 ms trên mọi request trong 30 phút đo (SLO chính, như W1–W3).
- **Chạy tiếp được:** mỗi lần chạy (một k, một khung giờ, một seed) là một đơn vị có `done.json`. Không cắt nhỏ hơn,
  vì phiên và hàng đợi phải liên tục. Dừng giữa chừng → chạy lại cùng lệnh, mất tối đa ~40 phút.

## Các bước

1. Ở local: sinh kịch bản cần dùng (`build_w4.py scenario --segment peak --k <k>`), ghi sha256 vào kết quả. Kho nội dung
   (`build_w4.py pool`) đã có (2026-10-12).
2. Trên L4, cùng server mặc định như P04:
   1. **Tìm goodput ở peak:** k khởi đầu ước từ goodput P04 (W1/W2), sau đó nhân đôi / chia đôi rồi tìm nhị phân tới
      `hi − lo ≤ max(2, 10% × lo)` (cùng quy tắc [workloads.md](../../workloads.md#cách-tìm-goodput-lưới-thích-nghi)). Dự kiến ~6 lần chạy.
   2. **Kiểm độ ổn định:** chạy lại k = `lo` với seed 2 (bộ ngày khác) và ở peak2. Goodput khác nhau > 1 bước tìm → ghi vào báo cáo.
   3. **Giờ bình thường:** chạy normal ở k = `lo` → số liệu cho chi phí cả ngày.
3. Rsync, destroy (nếu không làm tiếp P06).
4. Phân tích ở local: TTFT/TPOT p95 theo từng phút; tách theo loại request (ngắn < 1k / dài ≥ 2k, lượt đầu / lượt sau / resumed);
   % prefix cache hit; chi phí VND / 1 triệu token cả ngày = ghép kết quả peak / normal / vắng theo số giờ mỗi loại trong ngày thật của trace.

## Kết quả cần có

| Bảng / hình | Nội dung |
|---|---|
| Goodput W4 | k `lo`/`hi` ở peak; quy ra **req/giờ**, **số phiên đang mở** (người dùng hoạt động cùng lúc) |
| Hình 1 | TTFT p95 và TPOT p95 theo k, kẻ đường SLO |
| Hình 2 | Ở k = `lo`: theo thời gian, số request mỗi phút (trên), TTFT/TPOT p95 mỗi phút (dưới) |
| Bảng | TTFT/TPOT p95 tách theo loại request — request ngắn có bị request dài làm chậm không |
| Bảng | Ổn định: goodput seed 1 vs seed 2 vs peak2 |
| Chi phí | VND / 1 triệu token theo cả ngày (không chỉ giờ cao điểm) |

## Tiêu chí xong

- [x] Client phiên đã kiểm chứng: chạy W2 / 16 user closed-loop bằng client này, lệch < 5% so với `vllm bench serve` (P03: gộp 200 request, lệch ≤ 3.7%).
- [ ] Goodput W4 ở peak dạng khoảng hẹp; đã chạy seed 2, peak2, normal; số tóm tắt ghi measurements.md.

## Báo cáo P05 cần trả lời

1. vLLM mặc định phục vụ được bao nhiêu người dùng thật lúc cao điểm (req/giờ, phiên đang mở)?
2. Request ngắn có bị request dài (tài liệu, hội thoại dài) làm chậm không, chậm bao nhiêu?
3. Prefix cache giúp được bao nhiêu khi người dùng quay lại hội thoại (trúng cache theo think time)?
4. So với goodput closed-loop của P04, con số "người dùng" khác bao nhiêu lần, vì sao (think time, độ dài thật)?
5. Chi phí cả ngày so với chỉ tính giờ cao điểm.
