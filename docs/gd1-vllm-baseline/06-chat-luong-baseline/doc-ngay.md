# P06 — Đọc ngay: bản INT4 có trả lời kém đi không, và "sai" nghĩa là gì

> Bản dễ đọc, kể lại từng bước. Bản đầy đủ số liệu: [bao-cao.md](bao-cao.md).

**Một câu:** chạy 1 giờ 47 phút trên L4, tốn khoảng **15,500 VND**. Kết quả:
- Bản INT4 đạt **69.0 điểm MMLU-Pro** và **84.7 điểm IFEval**, gần như đúng bằng số Qwen công bố cho bản gốc (69.1 và 83.7), tức không thấy kém đi.
- Trả lời đúng **59%** câu hỏi số liệu tài chính trên tài liệu (FinQA-RAG).
- Chính vLLM chạy cùng câu hỏi hai lần cũng chỉ cho câu trả lời giống hệt **23%**. Đây là "mức tự lệch" dùng làm ngưỡng để phán các tối ưu sau này có làm sai hay không.

---

## 0. Vì sao cần P06

Mọi bước sau (tune vLLM, đổi thư viện, kernel tự viết ở GĐ4) đều làm model chạy nhanh hơn. Hội đồng sẽ hỏi ngay: **nhanh hơn nhưng có trả lời sai đi không?**
P06 tạo sẵn ba thứ để trả lời câu đó:
1. **Điểm chuẩn:** bản mặc định đạt bao nhiêu điểm trên vài bài kiểm tra chuẩn.
2. **Đáp án mẫu từng chữ:** 1,000 câu hỏi, lưu từng token và xác suất của 20 token cao nhất ở mỗi vị trí.
3. **Mức tự lệch:** chính vLLM chạy lại thì khác bao nhiêu. Thấp hơn mức này thì không thể nói một tối ưu "làm sai".

---

## 1. Bước 1 ở laptop: tìm tài liệu trước, quyết định sau (0 đồng)

Theo yêu cầu "làm gì cũng phải có cơ sở", tôi tải và đọc bản gốc 12 nguồn trước khi chốt cách đo
(lưu ở [references nhóm E](../../references/README.md)). Mấy điểm quyết định chính:

| Quyết định | Vì sao (nguồn) |
|---|---|
| Đo MMLU-Pro và IFEval | Qwen công bố đúng hai số này cho bản gốc ở chế độ tắt thinking → so được INT4 với bản gốc |
| MMLU-Pro chỉ lấy 1,000 / 12,032 câu, chọn ngẫu nhiên theo môn | Nghiên cứu thống kê của Anthropic (Miller 2024): ~1,000 câu đủ phát hiện chênh 3 điểm |
| Thêm FinQA-RAG: hỏi số liệu trên báo cáo tài chính, chấm bằng đáp án gốc, sai lệch ≤ 1% vẫn tính đúng | Đúng bài toán của đồ án. Đáp án lấy từ dataset FinQA gốc vì RAGBench không có đáp án chuẩn; dung sai 1% theo paper FinLFQA |
| Đo cả "độ gần từng token" chứ không chỉ điểm | Microsoft (Dutta 2024): lượng tử hoá giữ nguyên độ chính xác nhưng vẫn đổi đúng/sai ở tới 13.6% câu |
| Chạy lại tham chiếu nhiều lần để đo mức tự lệch | Thinking Machines (2025): Qwen3 ở temperature 0 cho 80 kết quả khác nhau trong 1,000 lần, do batch thay đổi |

Đã viết code `quality/`, thử trọn với server giả. Hai lỗi bắt được trước khi thuê máy:
- Bộ đọc đáp án lấy nhầm số trong trích dẫn "[Document 2]".
- Teacher forcing cho prompt dài có thể làm L4 hết bộ nhớ (đọc code vLLM), nên chỉ làm trên 200 prompt ngắn.

---

## 2. Thuê máy và chạy (21:05)

```bash
vastai create instance 49498406 --image nvidia/cuda:13.0.2-devel-ubuntu22.04 --disk 140 --ssh --direct
```
Còn trống đúng **máy 116596**, máy đã dùng cho P03/P04, và lại rẻ nhất ($0.306/h ≈ 8,000 VND). Đẩy code + dữ liệu (15 MB) mất 10 giây, rồi chạy:

```bash
tmux new -d -s q "bash quality/remote_quality.sh quality/configs/p06.yaml /root/results/p06"
```

| Phần | Nghĩa |
|---|---|
| `remote_quality.sh` | Cài vLLM + model (bỏ qua nếu đã có) → cài lm-eval 0.4.13 (bộ chấm điểm chuẩn, ghim phiên bản) → chạy 9 đơn vị đo |
| `p06.yaml` | Danh sách 9 đơn vị: 3 lần sinh tham chiếu, 2 lần teacher forcing, FinQA, GSM8K, IFEval, MMLU-Pro |

Server khởi động lần đầu bị biên dịch lại từ đầu nên KV chỉ có 28,704 token (lỗi đã biết từ P03). Công cụ tự bật lại, được 37,264, bằng P04.

---

## 3. Kết quả

### 3.1 Điểm (khoảng trong ngoặc = khoảng tin cậy 95%)

| Bài | Đo cái gì | Bản INT4 (đo) | Qwen công bố (bản gốc) |
|---|---|---|---|
| MMLU-Pro (1,000 câu) | kiến thức 14 môn, 10 lựa chọn | **69.0** (66.1–71.9) | 69.1 |
| IFEval (541) | làm đúng chỉ dẫn định dạng ("viết dưới 100 chữ", "3 gạch đầu dòng"…) | **84.7** (81.6–87.7) | 83.7 |
| GSM8K (1,319) | toán lời văn nhiều bước | **91.4** (89.8–92.9) | (chỉ có bản Base: 91.8) |
| FinQA-RAG (1,147) | số liệu tài chính trên tài liệu | **59.2** (56.4–62.0) | — |

**Nghĩa là:** trên hai bài có số công bố, bản INT4 không kém bản gốc ngoài sai số.
Lưu ý: Qwen đo bằng cách lấy mẫu ngẫu nhiên (temperature 0.7), mình đo greedy, nên chỉ nói được "không có chênh lệch đáng kể", không nói được "bằng nhau tuyệt đối".

FinQA-RAG 59% chưa có số nào để so. Để hình dung: chuyên gia tài chính đạt 91%, người thường 51%. 5% số câu model từ chối "không đủ thông tin",
còn lại là lấy nhầm số trong bảng hoặc tính sai phần trăm.

### 3.2 Mức tự lệch: chạy lại có giống không?

Cùng 1,000 câu hỏi, cùng chế độ greedy (luôn chọn token cao nhất), chạy hai lần:

| | Kết quả |
|---|---|
| Câu trả lời giống hệt cả 256 token | **chỉ 23%** |
| Bắt đầu khác nhau từ token thứ (trung vị) | **46** |
| Tại chỗ bắt đầu khác, token đứng đầu hơn token thứ hai (trung vị) | **3.7 điểm %**, tức gần như ngang nhau |
| Xác suất của token đã chọn thay đổi (trước chỗ khác) | trong khoảng **±6 điểm %**, **đối xứng** quanh 0 |

**Vì sao:** GPU cộng các số theo thứ tự khác nhau tuỳ số câu đang chạy cùng lúc, nên kết quả lệch ở chữ số cuối. Khi hai token gần ngang nhau,
chênh lệch nhỏ đó đủ đổi lựa chọn, rồi mọi chữ sau đều khác. Đổi từ 8 lên 64 câu cùng lúc cũng không làm lệch thêm (24.5% so với 23.1%).

**Đo kiểu llama.cpp** (đưa cùng văn bản vào, xem model chọn token nào ở từng vị trí): chạy 1 câu một lúc so với 8 câu một lúc, hai bên
chọn cùng token cao nhất ở **99.1%** vị trí.

**Dùng để làm gì:** sau này một kernel tự viết cho kết quả "khác" thì chưa chắc là sai. Nó chỉ bị nghi sai nếu:
- chọn cùng token thấp hẳn dưới ~99%, hoặc
- xác suất bị kéo lệch về một phía (model kém đi) thay vì dao động đối xứng như ở đây.

---

## 4. Dòng thời gian và chi phí

| Giờ | Việc |
|---|---|
| 21:05 | Thuê máy 116596; < 1 phút máy chạy |
| 21:06–21:19 | Cài đặt + khởi động server (tự bật lại 1 lần) |
| 21:19–22:10 | Sinh tham chiếu 3 lần (19 + 19 + 13 phút) |
| 22:10–22:12 | Teacher forcing 2 lần |
| 22:12–22:51 | FinQA 10 phút, GSM8K 11, IFEval 5, MMLU-Pro 14 |
| 22:52 | Kéo kết quả (kiểm checksum khớp), **destroy** |

**Chi phí:** tín dụng Vast $6.461 → $5.869 = **$0.59 ≈ 15,500 VND** cho 1.78 giờ. Ước trước 22–26k VND; rẻ hơn vì máy rẻ hơn và chạy nhanh hơn ước.

Hai lỗi nhỏ của tôi trong lúc phân tích, không ảnh hưởng số đo:
- Gộp nhầm hai cách chấm của GSM8K (đã sửa; số đúng khớp với bảng của lm-eval).
- Phân tích sớm khi khúc cuối chưa kịp kéo về.

---

## 5. Rút ra điều gì

1. **INT4 không làm model kém đi thấy được** trên MMLU-Pro và IFEval. Lựa chọn model chính (Qwen3-30B-A3B INT4) đứng vững về chất lượng.
2. **"Giống hệt từng chữ" không phải tiêu chí đúng/sai**, vì chính vLLM với chính nó chỉ giống 23%. Tiêu chí đúng: same top-1 ~99% và
   độ lệch xác suất đối xứng.
3. Đã có **cổng chất lượng** cho mọi bước tối ưu sau. Một lần kiểm nhanh ~25 phút GPU, gồm teacher forcing, FinQA và MMLU-Pro so theo từng câu.
4. **Chạy lại lần 2** (14/10, cùng máy, 53 phút, ~8,400 VND): chỉ chạy lại cùng cấu hình mà **1.6–6.6% câu tự đổi đúng ↔ sai**
   (GSM8K 1.6%, FinQA 4.5%, IFEval 5.2%, MMLU-Pro 6.6%). MMLU-Pro lần 2 thấp hơn 1.6 điểm, suýt bị coi là "khác biệt có ý nghĩa"
   dù cùng một cấu hình. Vì vậy, so một cấu hình mới phải so với **khoảng giữa các lần chạy baseline**, và kết luận "kém đi" chỉ khi
   lặp lại được. Điểm baseline nên báo trung bình hai lần: MMLU-Pro 68.2, IFEval 84.1, GSM8K 91.4, FinQA 59.7.

## Bước tiếp theo

- **P07:** tổng hợp P01–P06 (0 giờ GPU), lập plan tune GĐ1. Ưu tiên: KV FP8, `max_num_batched_tokens`, và giữ cache câu trả lời
  trong hội thoại (từ P05). Mỗi cấu hình phải qua cổng chất lượng ở trên.
