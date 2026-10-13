# P09 — Tăng chỗ KV: KV FP8 + thêm bộ nhớ cho KV

> Cần GPU: có · **Ước 2.3 h ≈ 20k VND** (2.1–4.1 h) · Phụ thuộc: P08 · Phiên 1 (P12 nối sau nếu điều kiện thoả)

## Vì sao làm trước

Nghi vấn N1 có bằng chứng mạnh nhất và ảnh hưởng mọi loại tải (P07 mục 4.1):
- số request chạy cùng lúc dừng đúng ở mức KV chứa được (~60 / ~15 / ~6 ở W1 / W2 / W3);
- 73–83% TTFT tại goodput là thời gian chờ trong hàng đợi;
- P03: lần khởi động có KV ít hơn 23% → TTFT p95 W2/16 gần gấp đôi;
- cận trên ước (chưa đo): goodput ~88 / ~26 / ~15 nếu throughput giữ nguyên.

Các plan sau (P10–P13) đều chạy trên nền cấu hình thắng ở đây.

## Cấu hình

| Tên | Cờ thêm vào `--max-model-len 16384` | Lý do |
|---|---|---|
| base | (không) | Điểm mốc W2/16 ×3 đầu phiên, so máy với P04 |
| **kv8** | `--kv-cache-dtype fp8 --gpu-memory-utilization 0.95` | Gấp đôi số token KV (FP8 1 byte / giá trị thay vì 2) + ~0.66 GiB nữa cho KV (~+19% so với 0.92; P01: 22.04 GiB khả dụng, nvidia-smi còn trống ~1.8 GiB khi chạy) |
| kv8-a (dự phòng) | như kv8 + `--attention-backend TRITON_ATTN` hoặc `FLASHINFER` | Nếu kv8 không khởi động (issue vLLM #60262: KV FP8 tự chọn FlashInfer rồi lỗi) |
| kv8-b (dự phòng) | như kv8 nhưng `--gpu-memory-utilization 0.94` | Nếu 0.95 hết bộ nhớ lúc khởi động hoặc lúc chạy |
| kv16 (dự phòng) | `--gpu-memory-utilization 0.95` (KV FP16) | Chỉ khi kv8 trượt cổng chất lượng **hai lần** |

Gộp FP8 và 0.95 thành một cấu hình thay vì đo riêng: tách hai hiệu ứng không cần cho quyết định, làm lại được sau.
Đối chứng "đổi backend, giữ KV FP16" cũng hoãn (README mục "Việc đã hoãn").

## Các bước

1. Thuê L4 (ưu tiên machine 116596), setup, điểm mốc W2/16 ×3 với base.
2. Khởi động kv8. Ghi backend attention, KV dtype, số token KV vLLM báo (P08 việc 5). Không khởi động được → kv8-a → kv8-b.
3. **Cổng chất lượng kv8 trước** (teacher forcing, sinh tham chiếu c8, FinQA, MMLU-Pro), để không tốn giờ đo tốc độ cho cấu hình có thể bị loại.
   - So với **cả hai** lần chạy baseline P06.
   - **Dấu hiệu kém đi** (đặt trước khi đo): điểm FinQA hoặc MMLU-Pro thấp hơn lần baseline thấp hơn quá nửa khoảng CI paired (~1.2 / ~1.6 điểm),
     **hoặc** flips > 1.5 × flips tự nhiên (FinQA 4.5%, MMLU-Pro 6.6%), **hoặc** Δp lệch hẳn về âm (p1 âm hơn −0.08 trong khi p99 vẫn ~+0.06).
   - Có dấu hiệu → chạy lại cổng một lần (0.78 h). Lặp lại được → loại kv8, dùng kv16. Không lặp lại → giữ kv8, ghi "dao động".
4. **Tìm goodput W1–W3 ×1** cho kv8 (seed 1001), khởi đầu nóng ở hai mức mỗi workload: W1 [72, 88], W2 [20, 26], W3 [8, 12]
   (mức dưới = mức baseline trượt; mức trên = cận trên ước ở P07). Tìm nhị phân như cũ tới `hi − lo ≤ max(2, 10% × lo)`.
5. Kéo kết quả, destroy (hoặc chuyển sang P12 nếu điều kiện thoả).

## Quy tắc chọn (đặt trước khi đo)

- kv8 qua cổng chất lượng **và** goodput ≥ baseline + 2 bước ở ít nhất một workload, không workload nào thấp hơn baseline → **cfg1 = kv8**.
- kv8 chỉ hơn ≤ 1 bước ở mọi workload → vẫn giữ kv8 nếu qua cổng chất lượng (cơ chế đúng, chi phí bộ nhớ bằng 0), ghi "chưa rõ lợi ích".
- Không cấu hình nào hơn baseline → cfg1 = baseline, ghi vào báo cáo và xem lại N1 trước P10.

## Ước giờ GPU

| Bước | Giờ | Cách tính |
|---|---|---|
| Thuê + setup + 2 lần khởi động base | 0.25 | P06 |
| Điểm mốc W2/16 ×3 | 0.10 | P05 |
| Khởi động kv8 (compile mới + bật lại) | 0.13 | `server_starts.jsonl` |
| Cổng chất lượng ×1 | 0.78 | P06 |
| Tìm goodput W1–W3 ×1, ~4 mức mỗi workload | 0.74 | W1 4 × 3.4 phút + W2 4 × 2.2 + W3 4 × 5.5 |
| Kéo kết quả, destroy | 0.05 | |
| **Cộng × 1.1** | **2.3 h ≈ 20k VND** | |

- Tốt nhất 2.1 h: mỗi workload chỉ 3 mức.
- Xấu nhất 4.1 h: kv8 không khởi động hai lần (+0.26), tìm 6 mức (+0.26), cổng chất lượng phải chạy lại (+0.78), kv16 phải đo thay (+0.87).

## Tiêu chí xong

- [ ] Điểm mốc ghi vào measurements; máy khác 116596 thì ghi lệch bao nhiêu.
- [ ] Số token KV, backend attention thật của kv8 đã ghi.
- [ ] Kết quả cổng chất lượng kv8 so với hai lần P06.
- [ ] Goodput W1 / W2 / W3 của cfg1 (×1, dạng "lo (hi trượt)") + số preemption W1 tại lo (đầu vào cho P12).

## Báo cáo P09 cần trả lời

1. KV FP8 + 0.95 cho bao nhiêu token KV, vLLM chạy backend nào?
2. Goodput mỗi workload tăng bao nhiêu bước; có khớp cận trên ước ở P07 không, nếu lệch thì vì đâu (throughput, TPOT)?
3. TPOT có trở thành chỉ số hỏng trước không (nếu có → xem lại speculative decoding, P07 mục 4.6)?
4. KV FP8 có làm chất lượng kém đi ngoài mức dao động của baseline không?
