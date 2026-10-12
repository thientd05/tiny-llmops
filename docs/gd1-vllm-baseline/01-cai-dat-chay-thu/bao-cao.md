# Báo cáo P01 — Cài vLLM và chạy thử model trên L4

> Ngày làm: 2026-10-12 · Máy: host 506934 / machine 116596 (Czechia) · vLLM 0.30.0 · Hoá đơn: GPU 0.346 h, tổng $0.126 ≈ 3,300 VND
> Số liệu đầy đủ: [measurements.md](../../measurements.md) mục "P01". Kết quả thô: `results/gd1/01-cai-dat-chay-thu/20261012-p01/`.

## 1. Tóm tắt

- vLLM 0.30.0 chạy được Qwen3-30B-A3B-GPTQ-Int4 trên L4. Trả lời đúng cả câu hỏi ngắn, prompt RAG 3k token và hợp đồng 7.9k token.
- Cấu hình **hoàn toàn mặc định không khởi động được**, vì không đủ bộ nhớ cho độ dài request tối đa 40960 token.
  Baseline của dự án từ đây = mặc định + đúng một cờ `--max-model-len 16384`.
- Phát hiện quan trọng nhất: sau khi nạp model, KV cache chỉ còn **37,264 token** (3.41 GiB).
  Kể cả khi vLLM chỉ cấp KV cache lúc request cần (PagedAttention), KV của prompt RAG ~2.9k token vẫn phải nằm trong cache suốt lúc sinh output.
  Theo tính toán, W2 chỉ có khoảng **15–16 request** chạy cùng lúc, số còn lại phải xếp hàng. Nhiều khả năng đây là giới hạn đầu tiên khi đo 32/64 user; P03 sẽ đo để kiểm chứng.
- Đạt tiêu chí xong của plan: script đã có và chạy lại được, số liệu đã ghi, máy đã destroy.

## 2. Đã làm gì và vì sao

1. **Chọn vLLM 0.30.0** (phát hành 2026-09-22). Bản mới nhất là 0.31.0, nhưng mới ra 1 tuần, chưa đủ tiêu chí "đã phát hành ít nhất 2 tuần".
   Khi đọc release notes 0.30.0 tôi gặp một thay đổi nguy hiểm: vLLM **bỏ hỗ trợ GPTQ act-order** (`g_idx`).
   Act-order là một kiểu lưu weight lượng tử theo thứ tự cột đã đảo; model dùng kiểu này sẽ cho ra kết quả sai nếu chạy trên 0.30.0.
   Tôi đã kiểm tra `config.json` của model: `desc_act: false`, nghĩa là model không dùng act-order, nên chạy đúng.
2. **Đổi image sang `nvidia/cuda:13.0.2-devel-ubuntu22.04`** (trước là 12.8.1). Bản cài vLLM 0.30.0 trên PyPI đi kèm torch biên dịch cho CUDA 13.0.
   Lúc chạy vLLM thì image nào cũng được, vì torch tự mang thư viện CUDA riêng.
   Nhưng ở GĐ4, kernel tự viết phải biên dịch bằng `nvcc` cùng phiên bản CUDA với torch, nên dùng image CUDA 13.0 ngay từ bây giờ.
3. **Làm hết phần không cần GPU ở máy local trước** (P02, các script) rồi mới thuê máy. Trên máy thuê, việc tải model và cài vLLM chạy **song song**.
4. Thuê máy L4 rẻ nhất đang có ($0.283/h, không phải loại interruptible). Chạy checklist, setup, khởi động server, gửi request thử,
   kéo kết quả về rồi **destroy ngay** (tổng ~17 phút).

Khác với plan: phải khởi động server 3 lần thay vì 1 (mục 5).

## 3. Kết quả

### vLLM mặc định tự chọn những gì

| Thành phần | vLLM chọn | Ghi chú |
|---|---|---|
| Nhân ma trận cho lớp thường (attention projection) | Marlin (kernel 4 bit) | Như dự kiến |
| Nhân ma trận cho MoE | Marlin MoE | Không dùng file config JSON → thiếu config cho L4 (library-gaps #4) **không ảnh hưởng** đường này |
| Attention | FlashAttention 2 | FlashAttention 3 chỉ chạy trên Hopper |
| KV cache | float16 | Chưa dùng FP8 |
| Prefix caching, chunked prefill | Bật | Hai tính năng này bật sẵn |
| CUDA graph + torch.compile | Bật | Đây là lý do khởi động mất vài phút |
| Tỉ lệ VRAM được dùng | 92% | |
| Số token tối đa xử lý trong 1 bước | 2048 (suy ra từ log) | Ảnh hưởng tốc độ prefill prompt dài → ứng viên tune |
| Tham số sinh token mặc định | temperature 0.6, top_k 20, top_p 0.95 | Lấy từ file cấu hình của model |

### VRAM được chia như sau

| Phần | GiB |
|---|---|
| VRAM GPU báo khả dụng | 22.04 |
| vLLM được dùng (92%) | 20.27 |
| Weight model + phần ngoài torch | 16.1 |
| Bộ nhớ tạm khi tính (activation) | 0.76 |
| CUDA graph | 0.33 |
| **KV cache** | **3.41 → 37,264 token** |

### Thời gian từ lúc máy chạy đến lúc server sẵn sàng

| Bước | Giây |
|---|---|
| Chạy checklist, copy script | ~60 |
| Cài vLLM + tải model 16 GB (song song) | 193 |
| Khởi động server (khi đã có cache compile) | 235 |

## 4. Giải thích kết quả

**Vì sao mặc định không chạy được.**
KV cache là vùng VRAM lưu kết quả trung gian của từng token đã xử lý, để bước sau không phải tính lại.
Với model này, mỗi token cần 2 (key và value) × 48 lớp × 4 KV head × 128 số × 2 byte = 98,304 byte, tức 96 KiB.
Một request dài 40960 token (giới hạn trong config model) cần 3.75 GiB KV cache.
Sau khi trừ weight, lần khởi động đầu chỉ còn 2.63 GiB, và vLLM từ chối chạy nếu không chứa được ít nhất một request dài tối đa.
Các workload của dự án dài nhất ~8.2k token (W3), nên giới hạn 16384 token không làm mất request nào. Cờ này chỉ đặt giới hạn độ dài, không đổi cách vLLM tính.

**KV cache 37,264 token nghĩa là gì với từng workload** (đây là phép **suy ra**, chưa đo dưới tải).

vLLM cấp KV cache theo từng khối nhỏ 16 token, **khi request cần mới cấp** (cơ chế PagedAttention). Nó không giữ chỗ trước cho độ dài tối đa.
Dòng log `Maximum concurrency for 16,384 tokens per request: 2.27x` là con số bi quan: nó giả định mọi request đều dài 16,384 token.
Bảng dưới tính theo số token **thực tế** mỗi request dùng, tức là đã tính cả lợi ích của việc cấp phát theo nhu cầu.

| Workload | Prompt TB | Output TB | Số request có KV nằm cùng lúc: lúc vừa prefill xong / lúc sinh xong output |
|---|---|---|---|
| W1 | ~407 | ~333 | ~92 / ~50 |
| W2 (622 token đầu dùng chung nhờ prefix caching, chỉ tính 1 lần) | ~2,858 | ~174 | **~16 / ~15** |
| W3 (95 token đầu dùng chung) | ~5,617 | ~299 | **~7 / ~6** |

**Vì sao cấp phát theo nhu cầu không nâng được con số của W2 và W3.** Phần chưa cần cấp chỉ là output.
Còn KV của **toàn bộ prompt** phải được tạo ở bước prefill và nằm trong cache suốt thời gian request sinh token.
Với W2, prompt chiếm ~94% số token của một request, nên lợi ích từ việc chưa cấp cho output chỉ khoảng 1 request (16 so với 15).

**Khi KV cache đầy, vLLM làm gì.** Cần thêm khối mà không còn chỗ thì vLLM tạm dừng request vào sau cùng (preemption):
xoá KV của nó, đưa về hàng chờ, và về sau **prefill lại từ đầu**.
Như vậy có thể nhận request mới sớm hơn, nhưng số request có KV nằm trong cache cùng lúc không vượt được mức ở bảng trên.
Request bị tạm dừng còn tốn thêm một lần prefill.

**Dự đoán (cần kiểm ở P03):** với W2 ở 32 và 64 user, chỉ khoảng 15–16 request được chạy cùng lúc, số còn lại xếp hàng nên TTFT tăng.
W3 bị giới hạn ở khoảng 6–7 request. Với W1, phần output chiếm tỉ lệ lớn nên cấp phát theo nhu cầu có tác dụng rõ, nhưng ngay cả mức thấp (~50) vẫn gần đủ cho 64 user.
P03 sẽ đo trực tiếp qua `/metrics` của vLLM: số request đang chạy tối đa, % KV cache đã dùng, số lần preemption.
Hai hướng tune nhắm thẳng vào giới hạn này: KV cache FP8 (mỗi token tốn một nửa bộ nhớ) và tăng tỉ lệ VRAM cho vLLM.

**Hai lần khởi động cho dung lượng KV cache khác nhau.** Lần đầu (torch.compile chạy từ đầu) được 2.63 GiB, lần sau (dùng lại kết quả compile đã lưu) được 3.41 GiB.
Đây là chênh lệch **đo được**. Nguyên nhân thì **chưa rõ**; tôi phỏng đoán bước compile lần đầu giữ lại một phần bộ nhớ GPU lúc vLLM đo bộ nhớ trống.
Hệ quả: trên máy mới, lần khởi động đầu có thể cho KV cache nhỏ hơn khoảng 23%, kéo theo kết quả đo thấp hơn.
P03 phải ghi dung lượng KV cache mỗi lần khởi động. Nếu đúng như phỏng đoán thì quy định: mỗi phiên khởi động server một lần để tạo cache, tắt đi, rồi mới khởi động bản dùng để đo.

**Chế độ thinking.** Cùng 3 câu hỏi: thinking bật cho output dài 422 / 2345 / 1018 token, tắt chỉ còn 39 / 202 / 54 token, ngắn hơn 10–19 lần.
Nếu bật, độ dài output do model tự quyết, nên lượng tải gửi vào server không kiểm soát được.
Kết luận giữ như P02 đã đề xuất: **tắt thinking** cho mọi workload.

**Tốc độ 1 request** (chỉ để tham khảo, dự án không báo cáo kịch bản 1 user): khoảng 88–92 token/giây khi sinh output.

## 5. Vấn đề gặp phải

| # | Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|---|
| 1 | Server mặc định không khởi động, báo thiếu KV cache | Xem mục 4 | Thêm `--max-model-len 16384` ([challenges.md](../../challenges.md) #2) |
| 2 | Lần 2 chết với lỗi `FileNotFoundError: 'ninja'` | FlashInfer (thư viện kernel vLLM dùng để lấy mẫu token) biên dịch kernel ngay lúc chạy (JIT) bằng chương trình `ninja`. `ninja` có trong venv, nhưng script không đưa venv vào PATH | Sửa `serve_default.sh` ([challenges.md](../../challenges.md) #3) |
| 3 | `ncu` vẫn bị chặn | Host 506934 không cho đọc performance counter (đã biết) | Không ảnh hưởng GĐ1 |

Ba lần khởi động tốn khoảng 8 phút GPU, tương đương ~1,000 VND.

## 6. Ảnh hưởng tới bước sau

- **P03:**
  - Ghi dung lượng KV cache sau mỗi lần khởi động. Kiểm tra giả thuyết "khởi động lần đầu cho KV nhỏ hơn".
  - Quyết định tham số sinh token khi đo. Đề xuất: giữ mặc định của model (temperature 0.6), giống người dùng thật.
    Độ dài output đã cố định bằng `ignore_eos`, nên không cần greedy để kiểm soát tải.
  - `vllm bench serve` cũng xáo trộn thứ tự mẫu theo seed của nó, cần ghi seed lại.
- **Chi phí cố định mỗi phiên thuê:**
  - thời gian khoảng 8–9 phút (setup ~3 phút + khởi động ~4 phút + checklist) ≈ $0.04;
  - phí tải xuống 21 GB ≈ $0.027;
  - Vast còn tính tiền GPU cho một phần thời gian `loading` trước khi máy chạy (hoá đơn ghi 0.346 h GPU, trong khi đo từ lúc `running` chỉ ~17 phút).

  Cộng lại khoảng **$0.07 ≈ 1,800 VND mỗi phiên** → nên gom nhiều plan vào một phiên.
- **Nghi vấn để tune** (đưa vào P07):
  - KV cache nhỏ là giới hạn chính → thử KV FP8 và tăng `gpu_memory_utilization`.
  - Chỉ 2048 token mỗi bước → prefill W2/W3 phải chia nhiều bước, TTFT có thể cao → tune `max_num_batched_tokens`.
  - Xem có backend attention nào nhanh hơn FlashAttention 2 trên L4 không.
- **Đã đổi:** image dùng khi thuê (CLAUDE.md, vast-guide.md). Script: `scripts/setup_node.sh`, `scripts/serve_default.sh`, `scripts/smoke_test.py`.

## 7. Thuật ngữ dùng trong báo cáo này

| Từ | Nghĩa |
|---|---|
| KV cache | Vùng VRAM lưu kết quả trung gian (key, value) của các token đã xử lý; quyết định bao nhiêu request chạy cùng lúc |
| Act-order (`desc_act`, `g_idx`) | Kiểu lượng tử GPTQ xử lý các cột weight theo thứ tự đã đảo; cần kernel riêng để đọc đúng |
| Marlin | Kernel của vLLM nhân ma trận với weight 4 bit |
| FlashAttention 2 / FlashInfer | Hai thư viện kernel tính attention (FlashInfer còn có kernel lấy mẫu token) |
| JIT (biên dịch lúc chạy) | Kernel được biên dịch ngay khi chương trình cần lần đầu, thay vì biên dịch sẵn khi cài |
| `ninja` | Chương trình điều phối biên dịch, FlashInfer dùng khi biên dịch JIT |
| torch.compile | PyTorch biên dịch phần tính toán của model thành code GPU tối ưu hơn; lần đầu mất ~60 s, sau đó dùng lại cache |
| CUDA graph | Ghi lại chuỗi lệnh GPU một lần, chạy lại nhiều lần để giảm thời gian CPU gửi lệnh |
| Thinking mode | Chế độ Qwen3 viết phần suy luận dài trước câu trả lời |
| TTFT | Thời gian từ lúc gửi request tới lúc nhận token đầu tiên |
| PagedAttention | Cách vLLM chia KV cache thành khối nhỏ (16 token) và cấp cho request khi cần, không giữ chỗ trước cho độ dài tối đa |
| Preemption | Khi hết KV cache, vLLM tạm dừng một request: xoá KV của nó, sau này prefill lại từ đầu |
| venv | Môi trường Python riêng, chứa đúng các gói (và phiên bản) dự án cần |
