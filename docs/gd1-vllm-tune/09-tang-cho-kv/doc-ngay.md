# P09 (+ P12) — Đọc ngay: lưu bộ nhớ ngắn hạn của model bằng 8 bit

**Một câu:** cho vLLM lưu "bộ nhớ ngắn hạn" (KV cache) bằng số 8 bit thay vì 16 bit và dành thêm 3% bộ nhớ GPU cho nó.
Số user phục vụ được trên một L4 **tăng gần gấp đôi** (chat 68 → 140, RAG 18 → 26, tóm tắt 6 → 12), chi phí mỗi triệu token giảm 25–41%.
Điểm chất lượng không giảm, nhưng model đổi đáp án ở nhiều câu hơn bình thường (cả đúng→sai lẫn sai→đúng), **cần bạn quyết có chấp nhận không** (cuối bài).
Phiên thuê 2.43 giờ GPU ≈ **20,040 VND** (ước 2.85 h ≈ 25k).

---

## Bước 0 — KV cache là gì, vì sao tăng nó

Khi model đọc câu hỏi và viết câu trả lời, mỗi token để lại một mẩu dữ liệu (key và value của từng lớp attention) để các token sau dùng lại.
Toàn bộ các mẩu đó là **KV cache**, nằm trong bộ nhớ GPU. Một token của Qwen3-30B-A3B cần 98,304 byte ở 16 bit (48 lớp × 4 đầu × 128 số × 2 × 2 byte).
Với cấu hình mặc định, L4 chỉ còn chỗ cho **37,264 token**, tức ~60 cuộc chat, ~15 câu hỏi RAG, ~6 tài liệu tóm tắt cùng lúc.
P07 đo được: hết chỗ thì request mới phải xếp hàng, và 73–83% thời gian chờ chữ đầu tiên là thời gian xếp hàng.

## Bước 1 — Thuê máy, đo điểm mốc (02:11–02:31)

Thuê đúng máy đã dùng cho baseline (machine 116596, $0.303/giờ). Trước tiên chạy lại cấu hình mặc định ở 16 user RAG, 3 lần với 3 bộ câu hỏi như P04:

| Bộ câu hỏi | Thời gian chờ chữ đầu (p95) P04 → hôm nay | Token / giây |
|---|---|---|
| 1 | 6.30 → 5.79 s | 171 → 172 |
| 2 | 4.87 → 4.94 s | 169 → 169 |
| 3 | 6.03 → 5.88 s | 167 → 167 |

→ Máy hôm nay chạy giống hệt hôm đo baseline, nên so trực tiếp được.

## Bước 2 — Bật KV 8 bit, đo chất lượng trước (02:31–03:23)

```bash
vllm serve ... --max-model-len 16384 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95
#                                    ↑ KV lưu bằng số 8 bit   ↑ cho vLLM dùng 95% bộ nhớ GPU (mặc định 92%)
```

vLLM báo chứa được **81,328 token** (×2.18), và tự đổi bộ kernel attention từ FlashAttention sang **FlashInfer**
(FlashAttention bản dùng trên L4 không đọc được KV 8 bit).

**Lỗi đầu tiên:** bài đo chất lượng đầu tiên (teacher forcing: đưa văn bản có sẵn vào, xin xác suất ở mọi vị trí) làm vLLM chết sau 4 giây:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 298.00 MiB ... 151.06 MiB is free
```

Nghĩa là GPU hết bộ nhớ. Xin xác suất ở mọi vị trí thì vLLM phải tính bảng 151,936 từ cho từng token của câu hỏi. Phần này không được dự trù lúc khởi động,
mà ở 95% thì chỉ còn 151 MiB trống. Phục vụ chat bình thường chỉ tính bảng đó cho token cuối, nên không bị. Sửa: chạy riêng bài này ở 92%,
vẫn KV 8 bit, vì dung lượng vùng KV không đổi cách tính. Ba bài còn lại chạy ở 95%.

**Kết quả chất lượng**, so với hai lần chạy baseline (chính baseline chạy hai lần đã lệch nhau):

| Bài | Baseline lần 1 / lần 2 | KV 8 bit | Câu đổi đúng↔sai | Mức tự nhiên (baseline với chính nó) |
|---|---|---|---|---|
| FinQA (1,147 câu hỏi số liệu tài chính trên tài liệu) | 59.2 / 60.1 | **59.0** | 7.7% (45 đúng→sai, 43 sai→đúng) | 4.5% |
| MMLU-Pro (1,000 câu kiến thức) | 69.0 / 67.4 | **67.8** | 11–12% (68 / 56) | 6.6% |

- **Điểm không giảm**: 59.0 và 67.8 nằm trong khoảng hai lần chạy baseline.
- **Nhưng model đổi đáp án nhiều hơn bình thường**, gần gấp đôi mức tự nhiên, và số câu đổi hai chiều gần bằng nhau.
- Đọc từng token: cùng một văn bản, KV 8 bit chọn token khác baseline ở 2.9% vị trí (baseline với chính nó: 0.9%).
  Sinh 256 token tự do, chỉ 5.4% câu ra giống hệt (baseline chạy lại: 23.1%).

Quy tắc đặt trước có 3 dấu hiệu "kém đi". Hai dấu hiệu về điểm và về lệch một chiều: **không**. Dấu hiệu "câu đổi nhiều hơn 1.5 lần mức tự nhiên": **có**.
Plan bảo chạy lại cổng chất lượng (~47 phút). **Tôi không chạy lại**: phép đọc từng token trên 45,000 vị trí đã cho thấy chênh lệch có hệ thống,
không phải ngẫu nhiên, nên chạy lại chắc chắn ra cùng kết quả. Câu hỏi thật là "khác nhưng không kém thì có chấp nhận không", và câu đó bạn quyết.

## Bước 3 — Tìm số user tối đa (03:23–04:13)

Mỗi mức user đo một lần (bộ câu hỏi 1), bắt đầu ở hai mức đoán trước, rồi mở rộng / chia đôi tới khi khoảng hẹp:

| Chat (W1) | 72 ✓ | 88 ✓ | 132 ✓ | **140 ✓** | 148 ✗ | 164 ✗ | 198 ✗ |
|---|---|---|---|---|---|---|---|
| Chờ chữ đầu p95 (s) | 4.5 | 5.2 | 7.6 | 8.4 | 10.2 | 15.3 | 25.2 |

| RAG (W2) | 20 ✓ | **26 ✓** | 28 ✗ | 32 ✗ | 40 ✗ |
|---|---|---|---|---|---|
| Chờ chữ đầu p95 (s) | 6.8 | 9.8 | 10.7 | 11.9 | 14.0 |

| Tóm tắt (W3) | 8 ✓ | **12 ✓** | 14 ✗ | 18 ✗ |
|---|---|---|---|---|
| Chờ chữ đầu p95 (s) | 3.4 | 7.7 | 10.6 | 17.3 |

(✓ = cả "chờ chữ đầu p95 ≤ 10 s" và "thời gian giữa hai token p95 ≤ 150 ms")

| | Baseline | KV 8 bit | VND / 1 triệu token ra |
|---|---|---|---|
| Chat | 68 user | **140** | 3,960 → **2,350** |
| RAG | 18 | **26** | 14,460 → **10,880** |
| Tóm tắt | 6 | **12** | 24,030 → **17,400** |

**Vì sao chat tăng nhiều thế** (P07 chỉ dám ước ~88): GPU sinh được nhiều token/giây hơn hẳn khi chạy nhiều request cùng lúc,
615 → 1,035 token/s. Mỗi bước sinh token, GPU phải đọc gần hết trọng số các "expert" (~15 GB) dù chạy 60 hay 130 request.
Chạy 130 request thì một lần đọc đó phục vụ 130 token thay vì 60. Trước đây KV hết chỗ ở ~60 request nên không tới được vùng này.

**Hai điều mới thấy:**
- Chat lại hết chỗ KV ở ~133 request (173 lần vLLM phải tạm dừng request vì hết chỗ ở mức 140 user). → Bước 4 (P12).
- RAG và tóm tắt trượt chủ yếu do **giây đầu tiên của bài đo**: công cụ đo gửi N câu hỏi cùng một lúc, 28 câu × 2,900 token phải đọc dồn một lượt (~10 s).
  Bỏ phần đầu và cuối đi, RAG ở 28–32 user chỉ chờ 1.7–2.1 s. Người dùng thật không gửi cùng một giây, nên bài đo tải giống thật (W4, P10) sẽ cho câu trả lời đúng hơn.

## Bước 4 — P12: bớt "tạm dừng request" ở chat (04:14–04:36)

Ở 140 user chat, vLLM phải **tạm dừng request đang chạy** 173 lần vì hết chỗ KV ("preemption"): bỏ KV của request đó, lát sau tính lại từ đầu.
Điều kiện chạy P12 (≥ 50 lần) thoả, nên thử hai cờ, mỗi cờ đo ở 140 user (mức đang đạt) và 164 user (mức đang trượt):

```bash
--watermark 0.02        # ← chừa 2% chỗ KV, không nhận request mới vào phần đó
--max-num-seqs 133      # ← không cho quá 133 request chạy cùng lúc (= số request KV chứa được ở 140 user)
```

| | Tạm dừng 140 / 164 user | Chờ chữ đầu p95 140 / 164 (s) | Chờ chữ đầu p50 | Token/s | Đạt 164? |
|---|---|---|---|---|---|
| Không cờ | 173 / 272 | 8.4 / 15.3 | 2.0 / 8.1 s | 1,035 | ✗ |
| watermark | 10 / 38 | 8.3 / 14.4 | chậm hơn 12–24% | +4% | ✗ |
| max-num-seqs | **0 / 0** | 8.0 / 12.8 | chậm hơn 13–17% | +4% | ✗ |

→ Hai cờ đều xoá gần hết việc tạm dừng, GPU làm được thêm 4% việc. Nhưng thời gian chờ chỉ **đổi chỗ**: thay vì được nhận rồi bị dừng,
request mới chờ ngoài hàng. Chờ chữ đầu p95 gần như không đổi, nên vẫn trượt ở 164. Quy tắc đặt trước: **không nhận**.

Một chi tiết hay: `--max-num-seqs 133` làm vLLM dành ít bộ nhớ hơn cho phần ghi sẵn kernel (CUDA graph), nên vùng KV tăng thêm 10% (81,328 → 89,488 token).
Đây là chỗ KV miễn phí, sẽ thử lại ở P13.

## Dòng thời gian và chi phí

| Giờ (VN) | Việc |
|---|---|
| 02:11 | Thuê machine 116596 ($0.303/h niêm yết) |
| 02:11–02:16 | Cài vLLM + tải model 17 GB (163 s) |
| 02:16–02:31 | Bật vLLM mặc định (lần đầu compile → tự bật lại), điểm mốc RAG 16 user × 3 |
| 02:31–02:37 | Bật KV 8 bit; bài đo chất lượng đầu làm vLLM hết bộ nhớ |
| 02:37–02:40 | Sửa (chạy bài đó ở 92%) |
| 02:40–03:23 | Đo chất lượng (4 bài) |
| 03:23–04:13 | Tìm số user tối đa: 16 lần đo |
| 04:14–04:36 | P12: 2 cấu hình × 2 mức |
| 04:36–04:37 | Kéo kết quả (308 file, kiểm mã băm khớp hết), huỷ máy |

**Chi phí theo tín dụng bị trừ: $0.765 ≈ 20,040 VND** cho 2.43 giờ (P09 ~16.7k, P12 ~3.3k). Plan ước 2.85 h ≈ 25k: rẻ hơn vì máy rẻ hơn giá tham chiếu
và P12 chỉ cần 22 phút. Lần bị lỗi bộ nhớ tốn thêm ~10 phút.

## Rút ra

- Tăng chỗ KV là thay đổi có lợi nhất tới giờ: số user gần gấp đôi chỉ bằng hai cờ.
- Sau khi KV không còn chặn, nghẽn chuyển sang chỗ khác: chat lại hết KV ở mức cao hơn, RAG bị giới hạn bởi việc đọc câu hỏi dài chen ngang việc sinh token.
- Công cụ đo kiểu "N user gửi cùng lúc" bắt đầu ảnh hưởng mạnh tới kết luận. Cần bài đo tải thật (W4) và cân nhắc sửa cách bắt đầu bài đo.

## Bạn cần quyết trước P10

**Có chấp nhận KV 8 bit không?** Điểm hai bài không đổi; số user gần gấp đôi; nhưng ~3–6% câu hỏi đổi đáp án so với hiện tại (đổi đúng thành sai và
sai thành đúng gần bằng nhau), và câu chữ khác hẳn baseline. Tôi đề xuất **chấp nhận**: tiêu chí dự án là "không làm giảm chất lượng", điểm không giảm.
Báo cáo cuối sẽ ghi rõ số câu đổi đáp án. Nếu không chấp nhận: đo phương án 16 bit + 95% bộ nhớ (~0.9 h ≈ 8k VND, ước chỉ +19% chỗ KV) ở đầu phiên P10.
