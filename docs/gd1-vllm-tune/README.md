# GĐ1 (phần cuối) — Tune vLLM tối đa trên L4

> Lập ngày 2026-10-14, sau báo cáo [P07](../gd1-vllm-baseline/07-tong-hop-baseline/bao-cao.md). Phần baseline (P01–P07):
> [../gd1-vllm-baseline/README.md](../gd1-vllm-baseline/README.md).
> Mục đích: tìm cấu hình vLLM 0.30.0 tốt nhất trên một L4. Cấu hình này là **mốc công bằng** cho GĐ2 (thư viện khác) và GĐ4 (code riêng):
> mọi cải tiến về sau phải so với vLLM đã tune, không so với vLLM mặc định.

## Nguyên tắc đo (chốt 2026-10-14, P07 mục 6.3)

- **Đo rẻ trước, đo bổ sung khi thật sự cần.** Mọi thứ đã ghim (vLLM, model, bộ câu hỏi, seed) và dữ liệu thô từng request đều lưu,
  nên lần đo nào cũng làm lại được sau trên bất kỳ L4 nào. Danh sách việc đã hoãn ở mục cuối.
- **Sàng lọc ở một mức tải cố định, không tìm goodput cho từng cấu hình.** Mức sàng lọc = mức mà cấu hình hiện tốt nhất đã **trượt**,
  cao hơn mức đạt của nó ít nhất 2 bước tìm. Cấu hình mới đạt ở mức đó → cải thiện rõ (≥ 2 bước), vượt nhiễu. Trượt → so thêm các số liên tục
  (TTFT p95, throughput) với cấu hình cũ ở cùng mức, cùng seed; chênh ≤ 1 bước thì ghi "chưa rõ", không kết luận.
- **Cộng dồn:** mỗi plan thử trên nền cấu hình tốt nhất của plan trước. Chỉ cấu hình cuối (P13) được đo goodput ×3 lần lặp và W4 hai bộ ngày.
- **Mỗi cờ đo bằng loại tải nơi nó có tác dụng:** KV và preemption bằng W1–W3; hàng đợi và lịch sử hội thoại bằng W4.
- **Cùng seed với baseline** (W1–W3: 1001–1003; W4: peak seed 1, seed 2 chỉ ở P13) để so được từng cặp.
- **Cổng chất lượng ×1** chỉ cho cấu hình đổi cách tính số (KV FP8) và cho cấu hình cuối. Lần chạy thứ hai chỉ khi lần đầu có dấu hiệu kém đi.
  **Sửa 2026-10-14 (P09):** "kém đi" = điểm thấp hơn lần baseline thấp hơn quá 1.96 × SE paired, hoặc Δp lệch hẳn về âm. Flips > 1.5 × tự nhiên
  là **"khác biệt, phải báo"** (kèm số câu đổi hai chiều), không tự loại cấu hình. Lý do: kv8 có flips gấp ~2 nhưng hai chiều cân bằng và điểm không đổi;
  chạy lại cổng không phân biệt được "khác" với "kém" ([P09 bao-cao](09-tang-cho-kv/bao-cao.md) mục 4.3). Chấp nhận cấu hình "khác" là quyết định của người dùng.
- **Thay đổi phía ứng dụng** (client gán độ ưu tiên, sửa template) báo thành nhóm riêng "vLLM đã tune + chỉnh ứng dụng".
- Chi phí tính theo giá chung **8,770 VND/h** (P07 mục 3.5).
- Đây là mặc định, không phải luật cứng: kết quả giữa chừng cho thấy cách khác tốt hơn thì đổi, kèm một câu lý do trong báo cáo.

## Danh sách plan

| # | Plan | Nghi vấn (P07) | GPU | Giờ GPU ước | VND ước | Phiên | Trạng thái |
|---|---|---|---|---|---|---|---|
| P08 | [Chuẩn bị harness cho tune](08-chuan-bi-harness-tune/plan.md) | — | Không | 0 | 0 | local | **Xong** 2026-10-14 ([báo cáo](08-chuan-bi-harness-tune/bao-cao.md)); việc 2, 6, 7 và phần W4 của việc 4 để trước P10 |
| P09 | [Tăng chỗ KV: KV FP8 + thêm bộ nhớ](09-tang-cho-kv/plan.md) | N1 | Có | **2.3** (2.1–4.1) | ~20k | 1 | **Xong** 2026-10-14: thật 2.03 h ([báo cáo](09-tang-cho-kv/bao-cao.md)). kv8 W1/W2/W3 = 140/26/12; cfg1 = kv8 **tạm**, chờ người dùng quyết chất lượng |
| P10 | [Câu dài chặn câu ngắn ở hàng đợi (W4)](10-hang-doi-w4/plan.md) | N3 | Có | **6.0** (4.6–8.3) | ~53k | 2 | Chưa |
| P11 | [Giữ lịch sử hội thoại ở RAM CPU (W4)](11-kv-ram-cpu/plan.md) | N2 | Có | **0.9** (0.7–1.7) | ~8k | 2 (nối sau P10) | Chưa |
| P12 | [Preemption W1 (có điều kiện)](12-preemption-w1/plan.md) | N4 | Có | **0.55** hoặc 0 | ~5k | 1 (nối sau P09) | **Xong** 2026-10-14: thật 0.40 h ([báo cáo](12-preemption-w1/bao-cao.md)); không nhận cờ nào |
| P13 | [Đo đầy đủ cấu hình tune cuối](13-do-cau-hinh-cuoi/plan.md) | — | Có | **5.0** (4.6–8.4) | ~44k | 3 | Chưa |
| P14 | [Tổng hợp GĐ1 tune, chuẩn bị GĐ2](14-tong-hop-tune/plan.md) | — | Không | 0 | 0 | local | Chưa |
| | **Tổng** | | | **~14.8 h** (12.0–23.1) | **~130k** (105–203k) | 3 phiên |

N5 (giữ khối `<think>` rỗng) **hoãn**: cần bài đo chất lượng hội thoại nhiều lượt chưa có (P07 mục 4.5).

Thứ tự phụ thuộc: P08 → P09 (→ P12) → P10 → P11 → P13 → P14.

**So với ước ở P07 (~31 h):** thấp hơn vì (1) sàng lọc ở một mức tải cố định thay vì tìm goodput cho từng cấu hình;
(2) bỏ đối chứng backend và các lần đo chỉ để giải thích (hoãn, mục cuối); (3) P13 khởi đầu nóng từ kết quả P09–P11, không đo lại lưới 8/16/32/64,
không đo khung giờ 10h/13h (để GĐ5 nếu cần đường cong đầy đủ).

## Đơn giá thời gian dùng để ước (đo thật ở P03–P06)

| Việc | Thời gian | Nguồn |
|---|---|---|
| Thuê + đẩy code + cài đặt + 2 lần khởi động server baseline | 0.25 h | P06 12 phút; P05 20 phút (mạng tải lên chậm) |
| Điểm mốc W2 / 16 user × 3 trên baseline (so máy mới với máy P04) | 0.10 h | P05 6 phút |
| Đổi cấu hình server (compile mới 314–349 s, harness bật lại 97–165 s) | 0.13 h | `server_starts.jsonl` P03, P05, P06 |
| Một đơn vị W1 ở N = 64–80 / 96 / 128 | 2.7–3.3 / 3.8 / 5.0 phút | P04 `units.csv` cột `unit_s` |
| Một đơn vị W2 ở N = 16–24 / 32 | 2.1 / 2.6 phút | như trên |
| Một đơn vị W3 ở N = 4–16 / 32 | 5.3–5.8 / 6.7 phút | như trên |
| Một lần chạy W4 đạt (10 + 30 phút + chờ request dở) | 0.68 h | P05: 40.2–40.7 phút |
| Một lần chạy W4 trượt, dừng sớm | 0.32–0.46 h | P05: 18.9 / 20.3 / 27.7 phút |
| Cổng chất lượng ×1 (teacher forcing + sinh tham chiếu c8 + FinQA + MMLU-Pro) + cài lm-eval | 0.78 h | P06: 1.2 + 19.0 + 9.6 + 13.6 phút + ~3 phút cài |
| Kéo kết quả, kiểm checksum, destroy | 0.05 h | P05, P06 |
| Hệ số máy chờ (thời gian thuê / tổng việc) | ×1.1 | P04 4.44 / 4.35; P05 4.97 / 4.49 |

Mọi số "Giờ GPU ước" trong bảng plan đã nhân ×1.1. Khoảng trong ngoặc: trường hợp tốt nhất / xấu nhất của các nhánh có điều kiện trong plan.
Số giữa là trường hợp dễ xảy ra nhất theo bằng chứng P07; cách tính từng bước nằm trong mục "Ước giờ GPU" của mỗi plan.
Nguồn sai số lớn nhất: goodput dịch bao xa (dịch xa thì tìm nhiều mức hơn) và số lần W4 trượt sớm hay đạt (0.4 h so với 0.68 h).

## Phiên thuê máy

| Phiên | Plan | Giờ | Ghi chú |
|---|---|---|---|
| 1 | P09, rồi P12 nếu điều kiện thoả | ~2.1–4.7 h | Điểm mốc W2/16 đầu phiên |
| 2 | P10 → P11 | ~5.3–10.0 h | Dài nhất. Cấu hình P11 viết ngay trong phiên sau khi P10 xong (Claude đang theo dõi) |
| 3 | P13 | ~4.6–8.4 h | Gồm cổng chất lượng của cấu hình cuối |

Cố gắng thuê lại máy 116596 (máy P03/P04/P06). Khác máy thì điểm mốc đầu phiên cho biết lệch bao nhiêu, và P13 tự đo cặp với baseline trong cùng phiên
khi cần (xem P13).

## Việc đã hoãn (đo bổ sung khi cần, tất cả làm lại được sau)

| Việc | Khi nào cần | Giờ ước |
|---|---|---|
| Đối chứng "đổi backend attention, giữ KV FP16" (tách hiệu ứng của KV FP8 với hiệu ứng đổi kernel) | Khi viết luận điểm "KV là nghẽn" cho hội đồng, hoặc GĐ3 thấy backend attention đáng kể | ~0.4 h |
| W4 baseline chạy lại cùng seed | Khi cấu hình tune chỉ hơn baseline ≤ 1 bước trên W4 (P13 tự làm nếu xảy ra) | 0.7–1.1 h |
| Lặp ×3 các cấu hình sàng lọc | Khi một quyết định phụ thuộc vào chênh ≤ 1 bước | ~0.3–0.7 h mỗi cấu hình |
| Cổng chất lượng lần 2 | Khi lần 1 có dấu hiệu kém đi | 0.78 h |
| GSM8K, IFEval cho cấu hình tune | Khi cần đủ 4 bài cho báo cáo cuối (GĐ5) | ~0.3 h |
| Đường cong 8/16/32/64 và khung giờ 10h/13h cho cấu hình tune | GĐ5 (hình so sánh, chi phí cả ngày chi tiết) | ~3–4 h |
| N5 giữ khối `<think>` rỗng | Khi có bài đo chất lượng nhiều lượt | chưa ước |
| LMCache làm backend RAM CPU | GĐ2 (sàng lọc thư viện bổ trợ) | ~1 h |
| kv16 (KV FP16 + 0.95), phương án thay kv8 | Chỉ khi người dùng không chấp nhận "khác nhưng không kém" của kv8 (P09 mục 6); đo ở đầu phiên P10 | ~0.9 h |
| Cổng chất lượng kv8 lần 2 | Không cần theo quy tắc đã sửa (P09 mục 4.3); làm nếu muốn kiểm "lặp lại" cho hội đồng | 0.78 h |
| Hiệu chuẩn scale KV FP8 (llm-compressor, references F2) | Nếu muốn giảm flips của kv8; cần tìm hiểu trước | chưa ước |
| Cách mở đầu bài đo W1–W3 (N user gửi cùng một giây) hoặc báo thêm cửa sổ "steady" | Trước P13: với kv8, W2/W3 trượt chỉ vì đợt khởi đầu (P09 mục 4.4); tính lại từ dữ liệu sẵn có | 0 (phân tích) |
| `--max-num-seqs` / cỡ CUDA graph nhỏ hơn → +10% KV | P13 (ứng viên, P12 mục 4.6) | trong P13 |
| N6: W2 bị giới hạn bởi prefill chen decode → `--max-num-batched-tokens` | P13 hoặc GĐ2, tuỳ kết quả W4 | ~0.5 h |

## Quy ước thư mục

```
docs/gd1-vllm-tune/
  README.md                   ← file này
  0X-<ten>/plan.md             ← plan
  0X-<ten>/doc-ngay.md, bao-cao.md  ← hai báo cáo theo ../gd1-vllm-baseline/mau-bao-cao.md
results/gd1/<plan>/<run_id>/  ← kết quả thô, rsync từ máy thuê về
bench/configs/tune/           ← cấu hình từng biến thể
```

Khi xong mỗi plan: cập nhật bảng trên, viết hai báo cáo, ghi số vào [../measurements.md](../measurements.md), `vastai show instances`.
