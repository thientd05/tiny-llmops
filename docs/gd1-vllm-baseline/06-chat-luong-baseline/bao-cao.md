# Báo cáo P06 — Chất lượng baseline + output tham chiếu

> Ngày làm: 2026-10-13 (bước 1 local chiều tối; đo trên L4 21:05 → 22:52) · Máy: host 506934 / machine 116596 (máy của P03/P04),
> instance 55753483 · vLLM 0.30.0 + `--max-model-len 16384` (không cờ thêm) · lm-eval 0.4.13 ·
> Giờ GPU: 1.78 h (≈ $0.592 ≈ 15,500 VND theo tín dụng bị trừ) + lần 2 (14/10) 0.88 h (≈ $0.320 ≈ 8,400 VND)

## 1. Tóm tắt

- **Không phát hiện suy giảm chất lượng của bản INT4 so với số Qwen công bố cho bản gốc.**
  - MMLU-Pro: 69.0 (CI 95%: 66.1–71.9), Qwen công bố **69.1**.
  - IFEval strict: 84.7 (81.6–87.7), Qwen công bố **83.7**.
  - Hai bên khác cách sinh (greedy so với lấy mẫu T = 0.7) và khác bộ prompt, nên chỉ kết luận được "không có chênh lệch vượt sai số".
- Bài toán chính của đồ án, **FinQA-RAG** (hỏi số liệu trên báo cáo tài chính): 59.2% (56.4–62.0). GSM8K: 91.4%.
- **Mức tự lệch của chính vLLM** (cùng prompt, greedy, chỉ khác lần chạy):
  - chỉ **23–25%** output giống hệt 256 token; trung vị bắt đầu lệch ở **token 46**;
  - Δp đối xứng quanh 0 (p1 / p99 ≈ ±0.06): đây là nhiễu, không phải sai;
  - teacher forcing: **same top-1 99.10%**.
  Đây là ngưỡng để phán các tối ưu sau này (GĐ1 tune, GĐ4 kernel tự viết) có làm sai hay không.
- Đã lưu output tham chiếu 1,000 prompt (token id + top-20 logprob mỗi vị trí) và đáp án từng câu của 4 tác vụ. So sánh sau này
  chạy bằng `quality/analyze.py --compare`.
- Đạt cả 3 tiêu chí xong của plan.

## 2. Đã làm gì và vì sao

**Bước 1 (local, 0 giờ GPU).** Mọi quyết định có tài liệu minh chứng (plan mục "Quyết định đã chốt", [references nhóm E](../../references/README.md)).
Hai chỗ đổi so với plan ban đầu, có lý do:
1. **Không lưu top-20 rồi tính KL bằng teacher forcing cho cả 1,000 prompt.**
   - Lý do: vLLM 0.30 tính logits cho mọi vị trí prompt khi bật `prompt_logprobs`. Với prompt RAG dài, phần này tốn ~2.5 GB bộ nhớ tạm, có rủi ro hết bộ nhớ trên L4.
   - Thay bằng: so khi sinh tự do (chính) + teacher forcing trên 200 prompt ngắn.
2. **Thêm FinQA-RAG và IFEval** để có bài toán đúng mục tiêu đồ án và có thêm một số công bố để so.

**Bước 2 (L4).**
- Máy 116596, cùng máy P03/P04. Lần khởi động đầu biên dịch lại từ đầu (KV 28,704 token); harness tự bật lại, được KV 37,264 (bằng P04).
- Chạy 9 đơn vị theo `quality/configs/p06.yaml`, không lỗi, không phải chạy lại đơn vị nào.
- Kéo kết quả về (kiểm checksum khớp), destroy.

## 3. Kết quả

### 3.1 Điểm tác vụ

CI 95% = p ± 1.96·√(p(1−p)/n), theo Miller 2024 (E3).

| Tác vụ | n | Điểm (%) | CI 95% | Qwen công bố (bản gốc, non-thinking) |
|---|---|---|---|---|
| MMLU-Pro (5-shot CoT, tập con ngẫu nhiên 1,000 / 12,032) | 1,000 | **69.0** | 66.1–71.9 | 69.1 |
| IFEval prompt-level strict | 541 | **84.7** | 81.6–87.7 | 83.7 |
| GSM8K flexible-extract (5-shot) | 1,319 | **91.4** | 89.8–92.9 | — (bản Base: 91.81) |
| FinQA-RAG, dung sai 1% | 1,147 | **59.2** | 56.4–62.0 | — |

Chỉ số phụ:
- IFEval loose 86.9, inst-level strict 89.6.
- GSM8K strict-match 90.4.
- FinQA dung sai 0.1% / 5%: 53.6 / 61.1.

MMLU-Pro theo môn (n mỗi môn 32–112, sai số lớn): math 90.2, biology 90.0, economics 84.3 … engineering 50.6, **law 34.1**.
Số chi tiết: [measurements.md](../../measurements.md) mục "2026-10-13 — P06".

### 3.2 Mức tự lệch (sinh tự do, 1,000 prompt, greedy, 256 token)

| So sánh | Giống hệt 256 token | Vị trí bắt đầu lệch p10 / p50 / p90 | Δp trên đoạn chung p1 / p99 | KL≈ mean / p99 | Chênh top1−top2 tại điểm lệch (p50) |
|---|---|---|---|---|---|
| c8 r1 vs c8 r2 (cùng mức tải) | 23.1% | 5 / 46 / 143 | −0.056 / +0.057 | 0.0015 / 0.026 | 0.037 |
| c8 r1 vs c64 r1 (khác mức tải) | 24.5% | 5 / 46 / 147 | −0.061 / +0.058 | 0.0016 / 0.029 | 0.036 |

- Δp = xác suất token đã chọn ở lần sau trừ lần đầu, chỉ tính trên đoạn trước điểm lệch (cùng ngữ cảnh). Có 83,505 vị trí.
- KL≈ tính từ top-20 + phần đuôi; độ phủ top-20 trung bình 0.999, nên sai số của phép xấp xỉ nhỏ.

### 3.3 Teacher forcing (200 prompt ngắn, 44,976 vị trí, cùng văn bản tham chiếu)

| So sánh | Same top-1 | Δp mean / RMS / p1 / p99 | KL≈ mean / p99 |
|---|---|---|---|
| c1 vs c8 | **99.10% ± 0.09** | 0.0002 / 0.017 / −0.047 / +0.049 | 0.0014 / 0.017 |

### 3.4 Thời gian và chi phí

| Phần | Thời gian |
|---|---|
| Thuê → máy chạy | < 1 phút |
| Đẩy code + dữ liệu (~15 MB) | 10 s |
| Cài đặt (model 158 s, vLLM 192 s, lm-eval) + 2 lần khởi động server | ~12 phút |
| Sinh tham chiếu c8 r1 / c8 r2 / c64 r1 | 19.0 / 19.0 / 13.0 phút |
| Teacher forcing c1 / c8 | 1.2 / 0.8 phút |
| FinQA / GSM8K / IFEval / MMLU-Pro | 9.6 / 10.8 / 5.3 / 13.6 phút |
| **Tổng thuê** | **1.78 h ≈ 15,500 VND** (ước trước 2.2–2.6 h) |

## 4. Giải thích kết quả

### 4.1 INT4 so với bản gốc (câu hỏi 1 của plan)

**Đo được:** MMLU-Pro 69.0 so với 69.1; IFEval 84.7 so với 83.7. Cả hai số công bố đều nằm trong CI 95% của số mình đo.

**Không so tuyệt đối được, vì ba lý do:**
1. Qwen sinh bằng lấy mẫu (T = 0.7, top-p 0.8, presence penalty 1.5; E1 mục 4.6), mình dùng greedy.
2. Qwen dùng bộ prompt và harness riêng; mình dùng lm-eval 0.4.13.
3. MMLU-Pro của mình là tập con 1,000 câu.

**Suy ra:** không có bằng chứng bản GPTQ-INT4 kém bản gốc trên hai tác vụ này. Chênh lệch thật, nếu có, nhỏ hơn khoảng ±3 điểm.
Kết luận này khớp với Dutta 2024 (E4): độ chính xác của bản lượng tử thường chỉ chênh 0–2%. Nhưng Dutta cũng cho thấy flips có thể lớn
dù độ chính xác không đổi. Muốn biết bản INT4 trả lời **giống** bản gốc tới đâu thì phải chạy bản gốc, mà bản BF16 (~61 GB) không vừa L4.
Ngoài phạm vi.

### 4.2 Mức tự lệch: vì sao greedy mà vẫn ra khác nhau (câu hỏi 2)

**Đo được:**
- Hai lần chạy cùng mức tải chỉ giống hệt ở 23% prompt.
- Điểm lệch xảy ra khi hai token đầu bảng gần ngang nhau: chênh xác suất trung vị chỉ 0.037.
- Trên đoạn chung, Δp đối xứng: p1 −0.056, p99 +0.057, trung bình ≈ 0.

**Cơ chế (E7):** kernel không bất biến theo kích thước batch. Mỗi bước, số request trong batch khác nhau nên thứ tự cộng số thực khác nhau,
xác suất lệch ở bit cuối. Khi hai token gần ngang nhau, chênh lệch nhỏ đó đủ đổi lựa chọn, và từ đó mọi token sau đều khác.

**Chi tiết đáng chú ý:**
- Đổi mức tải 8 → 64 **không làm lệch thêm** (24.5% so với 23.1%). Ngay ở 8 request đồng thời, batch đã thay đổi liên tục vì request
  bắt đầu và kết thúc lệch nhau. Vậy chạy lại hai lần là đủ để đo mức tự lệch.
- Tài liệu dài (CUAD) lệch muộn hơn (trung vị token 77) nhưng ít giống hệt nhất (13%). Output dài, nhiều câu, nhiều chỗ "gần ngang nhau".

**Hệ quả cho GĐ1 / GĐ4:**
- **"Output giống hệt" không dùng làm tiêu chí đúng/sai được**, vì chính baseline với chính nó cũng chỉ giống 23%.
- Tiêu chí dùng được là **so phân bố với ngưỡng tự lệch:**
  - teacher forcing same top-1 ≈ **99.1%** (một kernel đúng phải cho mức tương đương; thấp hẳn hơn là dấu hiệu sai);
  - Δp phải **đối xứng** với độ rộng p1/p99 cỡ **±0.05–0.06** (llama.cpp E5: lệch về phía âm = model kém đi);
  - KL≈ trung bình cỡ **0.0014–0.0016**.

### 4.3 Chênh bao nhiêu mới là khác biệt thật (câu hỏi 3)

- **So hai điểm riêng lẻ** (không theo từng câu), nửa khoảng CI 95% là:
  MMLU-Pro ±2.9 · IFEval ±3.0 · GSM8K ±1.5 · FinQA ±2.8 điểm.
  Hai cấu hình chênh ít hơn khoảng √2 × mức này là chưa phân biệt được.
- **So theo từng câu (paired, E3)** nhạy hơn nhiều: SE ≈ √(tỉ lệ câu đổi kết quả / n).
  - Ví dụ: hai cấu hình khác kết quả ở 5% của 1,000 câu → SE ≈ 0.7 điểm → CI ±1.4 điểm.
  - `quality/analyze.py --compare` tính sẵn paired SE và flips.
- **Hạn chế còn lại:** P06 chỉ chạy mỗi tác vụ **một lần**, nên chưa biết **flips tự nhiên** (baseline với chính nó) trên tác vụ.
  Cần số này để biết một cấu hình mới có flips "bình thường" hay không. Đề xuất ở mục 6.

### 4.3b Flips tự nhiên (đo bổ sung 2026-10-14, cùng máy 116596)

Chạy lại 4 bài chấm điểm y nguyên cấu hình (53 phút, ~8,400 VND; [measurements](../../measurements.md) mục "P06 lần 2").

| Bài | Điểm lần 1 → lần 2 | Flips tự nhiên | Hiệu, CI 95% paired |
|---|---|---|---|
| FinQA (1%) | 59.2 → 60.1 | 4.5% | +0.9 (−0.4 … +2.1) |
| GSM8K | 91.4 → 91.4 | 1.6% | +0.1 (−0.6 … +0.8) |
| IFEval strict | 84.7 → 83.5 | 5.2% | −1.1 (−3.0 … +0.8) |
| MMLU-Pro | 69.0 → 67.4 | 6.6% | **−1.6 (−3.2 … −0.0)** |

**Đo được:**
- Chỉ chạy lại cùng cấu hình mà 1.6–6.6% câu đã đổi đúng ↔ sai.
- Ở MMLU-Pro, hiệu −1.6 điểm có CI 95% paired vừa không chứa 0.

**Suy ra:**
1. **Flips tự nhiên là ngưỡng thật** cho các bước sau. Ví dụ, một cấu hình tune có ~6–7% flips MMLU-Pro là bình thường; phải cao rõ hơn
   mới là dấu hiệu đổi hành vi.
2. **So một lần chạy với một lần chạy là chưa đủ**, kể cả khi dùng paired SE: baseline với chính nó đã cho một kết quả "có ý nghĩa" giả.
   Lý do: paired SE coi lần chạy như cố định, nhưng ở đây chính lần chạy cũng là nguồn ngẫu nhiên (lệch do batch).
   Quy tắc đề xuất cho cổng chất lượng: **hiệu của cấu hình mới phải nằm ngoài khoảng hiệu giữa các lần chạy baseline**, và kết luận
   "kém đi" chỉ khi lặp lại được ở lần chạy thứ hai.
3. Điểm baseline nên báo là **trung bình hai lần**: MMLU-Pro 68.2, IFEval 84.1, GSM8K 91.4, FinQA 59.7. Số công bố (69.1, 83.7) vẫn
   nằm trong khoảng → kết luận ở 4.1 không đổi.

### 4.4 Về FinQA-RAG 59.2%

- Đây không phải số so được với công bố nào: prompt do mình dựng (system prompt W2 + tài liệu RAGBench + yêu cầu dòng Answer).
  Nó là **mốc nội bộ** cho các bước tối ưu.
- Để hình dung độ khó (FinQA paper, E9): chuyên gia tài chính đạt 91%, người không chuyên 51%, mô hình tốt nhất trong paper 65%.
- **Phân tích lỗi:**
  - 60 câu (5%) model **từ chối** ("not enough information"), theo luật 1 của system prompt W2. Prompt không bị cắt (dài nhất 3,183 token),
    nên đây là lựa chọn của model, hoặc RAGBench đã cắt mất tài liệu chứa đáp án khi chọn 10 tài liệu (chưa kiểm).
  - Các lỗi xem mẫu: lấy nhầm số trong bảng (ví dụ lấy giá trị tích luỹ 248.36 thay vì phần tăng 148.36), tính phần trăm sai.

## 5. Vấn đề gặp phải

| Vấn đề | Xử lý |
|---|---|
| Bước 1: bộ đọc đáp án FinQA lấy nhầm số trong trích dẫn "[Document 2]" | Test `quality/test_analyze.py` bắt được trước khi thuê máy; đã sửa |
| `analyze.py` gộp nhầm hai bộ lọc của GSM8K (strict 91.4 thay vì 90.4) | Do file mẫu của lm-eval có mỗi câu một dòng cho mỗi bộ lọc. Sửa: tách theo trường `filter`; số khớp với bảng của lm-eval |
| Phân tích sớm giữa phiên thấy c64 chỉ có 900 prompt | `done.json` được kéo về trước khúc cuối. Lần kéo sau đủ; số cuối cùng tính trên đủ 1,000 |
| `pkill -f <script>` tự tắt luôn shell đang chạy lệnh (mẫu nằm trong dòng lệnh) | Không ảnh hưởng số đo; lần sau dùng `pgrep` lấy PID rồi `kill` |

## 6. Ảnh hưởng tới bước sau

**Cổng chất lượng cho GĐ1 tune và GĐ4** (mỗi cấu hình mới chạy `quality/run.py` với tập con các đơn vị, rồi `analyze.py --compare` với P06):

| Kiểm tra | Thời gian GPU | Ngưỡng đạt (đề xuất, từ số P06) |
|---|---|---|
| Teacher forcing 200 prompt | ~1 phút | same top-1 ≥ ~99% và Δp đối xứng (p1 / p99 trong khoảng ±0.06) |
| Sinh tham chiếu 1,000 prompt | ~13–19 phút | vị trí lệch và KL cỡ như bảng 3.2 |
| FinQA + MMLU-Pro (paired) | ~23 phút | hiệu không âm ngoài CI paired; flips cỡ flips tự nhiên |

- **Đã làm (2026-10-14):** chạy lại 4 bài chấm điểm → flips tự nhiên 1.6–6.6% (mục 4.3b). Cổng chất lượng so cấu hình mới với
  **khoảng giữa hai lần chạy baseline**, không với một lần.
- **KV FP8** (hướng tune số 1 từ P04/P05) đổi số học của attention, nên chắc chắn phải qua cổng này.
- **Kernel GĐ4:** unit test so với PyTorch (quy tắc 12) đo sai số từng phép. Cổng trên đo ảnh hưởng tới output của cả model. Cần cả hai.

## 7. Thuật ngữ dùng trong báo cáo này

| Từ | Nghĩa |
|---|---|
| Greedy | Luôn chọn token xác suất cao nhất (temperature 0) |
| CI 95% | Khoảng tin cậy 95% của điểm: p ± 1.96·√(p(1−p)/n) |
| Paired | So hai cấu hình theo từng câu hỏi chung, không so hai điểm tổng |
| Flips | % câu đổi đúng ↔ sai giữa hai cấu hình |
| Teacher forcing | Đưa nguyên văn bản tham chiếu vào làm prompt rồi đọc xác suất từng vị trí; mọi cấu hình so trên cùng ngữ cảnh |
| Same top-1 | % vị trí hai cấu hình chọn cùng token cao nhất |
| Δp | Xác suất token tham chiếu ở cấu hình B trừ cấu hình A |
| KL≈ | KL divergence tính xấp xỉ từ top-20 xác suất + phần đuôi |
| Độ phủ top-20 | Tổng xác suất của 20 token cao nhất |
| Mức tự lệch | Mức khác nhau giữa hai lần chạy của chính vLLM mặc định (do batch thay đổi) |
| Batch invariance | Kernel cho cùng kết quả bất kể kích thước batch; vLLM mặc định không có tính chất này |
