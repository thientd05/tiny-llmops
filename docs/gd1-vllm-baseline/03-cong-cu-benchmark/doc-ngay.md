# P03 — Đọc ngay: viết công cụ đo tự động, thử trên laptop, rồi đo thử trên L4

> Bản dễ đọc, kể lại từng bước. Bản đầy đủ số liệu: [bao-cao.md](bao-cao.md).

**Một câu:** viết xong bộ công cụ đo tự động và thử kỹ trên laptop (không tốn tiền). Sau đó thuê L4 50 phút, khoảng **7,000 VND**, để đo thử 10 lần.
Công cụ đo đúng. Bốn phát hiện chính:
1. Lần bật vLLM đầu tiên trên máy mới có **bộ nhớ KV ít hơn 23%**, đủ làm kết quả đổi từ "đạt" sang "suýt trượt".
2. W2 (câu hỏi RAG) **hết KV cache ngay từ 16 người dùng**.
3. **Chỉ số TTFT p95 dao động ~20% giữa hai lần đo**, lớn hơn mức 5% mong muốn.
4. Client tự viết cho W4 cho số giống `vllm bench serve` (lệch ≤ 3.7%).

---

## 0. Đo cái gì, bằng gì

- **Server**: vLLM chạy model trên máy thuê, nhận câu hỏi qua HTTP ở cổng 8000.
- **Client**: chương trình gửi câu hỏi tới server và bấm giờ. Client chạy **ngay trên máy thuê**. Nếu gửi từ laptop ở Việt Nam sang Czechia, mỗi câu hỏi bị cộng ~0.2 s đường mạng.
- **Ba con số chính** của mỗi câu hỏi:
  - **TTFT** (time to first token): từ lúc gửi tới lúc nhận chữ đầu tiên, tức thời gian người dùng nhìn màn hình trống.
  - **TPOT** (time per output token): trung bình bao lâu ra một chữ (token) tiếp theo, tức tốc độ chữ chạy.
  - **E2EL**: tổng thời gian từ lúc gửi tới lúc xong.
- **p95**: sắp xếp TTFT của mọi câu hỏi trong một lần đo từ nhanh tới chậm, lấy giá trị ở vị trí 95%. Nghĩa là 95% câu hỏi nhanh hơn số này.
- **SLO** (mức phải đạt): TTFT p95 ≤ 10 s và TPOT p95 ≤ 150 ms.
- **Goodput**: số người dùng cùng lúc lớn nhất mà vẫn đạt SLO. Đây là con số đồ án muốn tăng.
- **"N người dùng" (closed-loop)**: N luồng gửi câu hỏi; mỗi luồng gửi, chờ trả lời xong, gửi câu tiếp ngay. Như vậy server luôn có N câu hỏi.

---

## 1. Kiểm tra công cụ đo của vLLM trước khi viết code (laptop, 0 đồng)

vLLM có sẵn công cụ đo `vllm bench serve`. Plan ghi tên một số tham số theo trí nhớ, nên tôi kiểm lại trước khi dùng:
- danh sách tham số in ra từ máy L4 ở P01;
- mã nguồn đúng bản 0.30.0.

Có hai bẫy, nếu không kiểm thì số đo sai mà không ai biết:

| Bẫy | Nếu không sửa | Cách xử lý |
|---|---|---|
| `--custom-output-len` mặc định là 256 và **ghi đè** độ dài câu trả lời trong file workload | Mọi câu trả lời dài đúng 256 token, sai so với thiết kế P02 | Đặt `--custom-output-len -1` (lấy độ dài từ file) |
| `--num-warmups` (gửi vài câu chạy nóng máy) gửi lại **đúng câu hỏi đầu tiên** của lần đo | Câu đó được server nhớ sẵn phần đầu (prefix cache), trả lời nhanh hơn thật | Không dùng; tự gửi 4 câu khác để chạy nóng, rồi bảo server xoá bộ nhớ đệm (`/reset_prefix_cache`) |

Ngoài ra:
- Công cụ tự lặp lại câu hỏi nếu thiếu mẫu. Tôi tắt bằng `--no-oversample`, vì câu hỏi lặp sẽ trúng bộ nhớ đệm và cho số đẹp giả.
- `temperature` không còn mặc định bằng 0, nên tôi đặt rõ `--temperature 0`.

---

## 2. Viết công cụ (laptop)

Bốn file trong `bench/`:

| File | Làm gì |
|---|---|
| `sweep.py` | Chạy cả loạt đo: bật server, chạy nóng, xoá bộ nhớ đệm, gọi client, ghi kết quả, quyết định đo mức nào tiếp |
| `summarize.py` | Đọc kết quả, in bảng: goodput, nhiễu, so sánh các kiểu tính |
| `session_client.py` | Client tự viết: phát lại phiên người dùng W4 (cho P05), và chế độ N người dùng để so với vLLM |
| `fake_server.py` | **Server giả**: trả lời giống vLLM nhưng không chạy model, để thử công cụ trên laptop |

**Mỗi lần đo** (ví dụ W2, 16 người dùng, lần lặp thứ 2) có tên `w2-c16-r2` và một thư mục riêng. File cuối cùng được ghi là `done.json`.
Nếu bị ngắt giữa chừng thì chạy lại **đúng lệnh cũ**. Thư mục nào có `done.json` thì bỏ qua; thư mục dở dang thì xoá và đo lại.

**Tự tìm goodput.** Không đo một danh sách số người dùng cố định:
1. Đo 8, 16, 32, 64.
2. Nếu 64 vẫn đạt thì đo 128, 256...
3. Nếu 8 đã trượt thì đo 4, 2.
4. Sau đó **chia đôi khoảng** giữa mức đạt lớn nhất và mức trượt nhỏ nhất, tới khi khoảng hẹp hơn 2 người (hoặc 10%).

Bước tiếp theo luôn tính lại được từ các `done.json` đã có, nên dừng ở đâu cũng đi tiếp đúng chỗ.

**Trong lúc đo**, cứ mỗi giây ghi lại:
- từ server (`/metrics`): bao nhiêu câu đang chạy, bao nhiêu câu đang chờ, % bộ nhớ KV đã dùng, số lần bị tạm dừng vì hết KV (preemption);
- từ GPU (`nvidia-smi`): công suất, xung nhịp, nhiệt độ.

---

## 3. Thử công cụ trên laptop với server giả (0 đồng)

Server giả có một "sức chứa K" đặt trước. Nhận quá K câu cùng lúc thì nó cố ý trả lời chậm. Vì vậy điểm gãy đúng **biết trước = K**, và công cụ phải tìm ra K.

```bash
bash bench/test_fake.sh <thư_mục_tạm>
```

| Thử | Mong đợi | Kết quả |
|---|---|---|
| K = 100 (điểm gãy trên 64, phải đo lên) | khoảng chứa 100 | 96 đạt, 104 trượt ✓ |
| K = 5 (dưới 8, phải đo xuống) | chứa 5 | 4 đạt, 6 trượt ✓ |
| K = 40 (giữa lưới) | chứa 40 | 40 đạt, 44 trượt ✓ |
| Ngắt bằng Ctrl+C, rồi `kill -9` (giết ngay), rồi chạy lại cùng lệnh | Chỉ đo phần thiếu | Lần đo dở bị xoá và đo lại; lần đã xong không bị đo lại ✓ |
| Sửa cấu hình (100 → 120 câu mỗi lần đo) | Phải đo lại hết | Đúng ✓ |
| Phát lại W4 (554 phiên, tua nhanh 100 lần) | Đủ số lượt, không lượt nào gửi sớm | 957/957 lượt, 0 gửi sớm ✓ |

Trên laptop tôi cài đúng `vllm bench serve` bản 0.30.0 (bản chạy CPU, chỉ dùng để gửi câu hỏi). Nhờ vậy lúc thử trên laptop là thử đúng công cụ sẽ chạy trên L4.

---

## 4. Thuê máy

```bash
vastai create instance 49498406 --image nvidia/cuda:13.0.2-devel-ubuntu22.04 --disk 60 --ssh --direct --label p03-p04-bench
```

Offer rẻ nhất ($0.268/giờ, tính cả ổ đĩa $0.2833/giờ) **đúng là máy đã dùng ở P01** (machine 116596). Cùng máy nên số liệu so được với P01.
Máy chạy lúc 23:28:55.

Kiểm tra máy (1 phút): băng thông bộ nhớ GPU 231.2 GB/s (giống P01), `ncu` vẫn bị chặn, giới hạn công suất 72 W.

---

## 5. Đẩy code lên và chạy trong `tmux`

```bash
rsync -az --relative bench/*.py bench/configs scripts/*.sh workloads/manifest.json workloads/w4_session.py \
      data/workloads/w1.jsonl data/workloads/w2.jsonl data/workloads/w3.jsonl  root@<IP>:/root/td/
ssh ... "cd /root/td && tmux new -d -s exp 'bash scripts/remote_bench.sh bench/configs/smoke.yaml /root/results/p03'"
```

| Phần | Nghĩa |
|---|---|
| `rsync --relative ...` | Copy code và 3 file workload (~56 MB) lên máy thuê, giữ nguyên cấu trúc thư mục |
| `tmux new -d -s exp '...'` | Chạy lệnh trong một "phiên tmux" tên `exp`. Phiên này sống trên máy thuê, laptop mất mạng thì nó vẫn chạy |
| `remote_bench.sh` | Script mới: cài đặt (bỏ qua bước đã xong) → đo → tổng hợp |
| `smoke.yaml` | Danh sách 10 lần đo thử |

Trên laptop, một vòng lặp kéo kết quả về mỗi 2 phút (`rsync`). Thư mục kết quả trên laptop là bản gốc.

### Hai lỗi lúc bắt đầu (lỗi của tôi)

1. **Cài xong nhưng không đo.**
   - Script ghi log vào `setup/setup.out` trước khi tạo thư mục `setup/`, nên lệnh ghi log báo lỗi.
   - Script đặt chế độ "lỗi một lệnh thì dừng cả script", nên dừng luôn.
   - Sửa: tạo thư mục trước. Máy chạy không ~7 phút, ~900 VND.
2. **`ImportError: Please install vllm[bench]`.**
   - `vllm bench serve` cần thêm thư viện `pandas`. Gói `vllm` thường không có, phải cài `vllm[bench]`.
   - Trên laptop tôi đã tự cài pandas từ trước nên không thấy lỗi này.
   - Sửa `setup_node.sh` để cài `vllm[bench]==0.30.0`. Mất ~1 phút.

Hai lần chạy lại đều dùng **đúng lệnh cũ**. Bước cài đặt tự bỏ qua (4–6 giây), lần đo dở bị xoá và đo lại. Cơ chế chạy tiếp được hoạt động đúng như thiết kế.

---

## 6. Kết quả đo thử

### Phát hiện 1: lần bật đầu tiên có ít bộ nhớ KV hơn

KV cache là phần bộ nhớ GPU lưu thông tin của các token đã đọc. Càng nhiều KV thì càng nhiều câu hỏi chạy cùng lúc.

| Lần bật server | Có biên dịch model từ đầu không | KV cache |
|---|---|---|
| 1 (máy mới) | Có, mất 71 s | **28,704 token** |
| 2 | Không, đọc lại kết quả biên dịch đã lưu | **37,264 token** |
| 3 | Không | 37,264 token |

Cùng máy, cùng tham số, chỉ khác lần đầu phải biên dịch, mà KV chênh 23%.
P01 đã nghi điều này. Nay đã có bằng chứng.

Ảnh hưởng tới kết quả, cùng bài đo W2 với 16 người dùng:
- Lần bật 1: TTFT p95 = **9.9 s** (suýt trượt ngưỡng 10 s).
- Lần bật 2, 3: TTFT p95 = **4.6–5.6 s**.

→ Đã thêm vào công cụ: nếu server vừa biên dịch từ đầu (> 10 s) thì **tự tắt và bật lại** rồi mới đo. Tốn thêm ~2.5 phút mỗi máy mới.

### Phát hiện 2: W2 hết KV cache ngay từ 16 người dùng

| Bài đo | TTFT p95 | TPOT p95 | Chữ ra / giây | Đạt SLO |
|---|---|---|---|---|
| W1 (chat ngắn), 16 người | 1.1 s | 49 ms | 322 | đạt |
| W2 (RAG), 8 người | 2.3 s | 57 ms | 146 | đạt |
| W2, 16 người (KV đủ) | 4.6–5.6 s | 89–93 ms | 168–170 | đạt |
| W2, 32 người | 25 s | 75 ms | 160 | **trượt** |
| W3 (tóm tắt hợp đồng), 8 người | 15.7 s | 49 ms | 96 | **trượt** |

(W1/16, W2/8, W2/32, W3/8 đo ở lần bật 1, khi KV còn nhỏ. Chỉ để xem sơ bộ; P04 đo lại đầy đủ.)

Ở W2 với 16 người: bộ nhớ KV dùng 100%, có lúc 13 câu đang chờ, bị tạm dừng 6–7 lần.

Server tự báo: trong 5.3 s TTFT trung bình thì **4.3 s là chờ xếp hàng**, chỉ 0.8 s là tính toán.
Nghĩa là câu hỏi RAG chậm vì **thiếu chỗ trong bộ nhớ**, không phải vì GPU tính chậm.

Một câu hỏi W2 khi xong chiếm ~3,000 token KV. 37,264 token chỉ chứa trọn ~12 câu.

### Phát hiện 3: GPU luôn chạy ở trần điện 72 W

Lúc đo, GPU dùng 73–74 W (chạm giới hạn) và tự hạ xung nhịp xuống ~1,100 MHz, tức khoảng một nửa mức tối đa 2,040 MHz.
Đây là đặc điểm của L4. Mọi số đo về sau đều trong điều kiện này.

### Bốn kiểm tra công cụ (plan yêu cầu)

| Kiểm tra | Kết quả |
|---|---|
| (a) Server nhận đúng số token như bộ workload P02? | **Khớp đúng từng câu hỏi** |
| (b) Client bấm giờ khớp server tự đo? | Client chậm hơn đúng ~25 ms (thời gian gửi/nhận HTTP), ổn định |
| (c) Xoá bộ nhớ đệm có tác dụng? | Trúng bộ nhớ đệm 21.6%, đúng bằng phần system prompt chung của W2 (622/2,858 = 21.8%) |
| (d) Đo hai lần cùng một điểm lệch bao nhiêu? | Chữ ra/giây lệch 1.2%, TPOT 3.7% ✓. **TTFT p95 lệch 19%** ✗ |

### Phát hiện 4: TTFT p95 dao động ~20% (vấn đề thiết kế, báo bạn)

- Plan đặt mục tiêu nhiễu < 5%.
- Throughput và TPOT đạt mục tiêu này. TTFT p95 thì không.
- Lý do: khi hết bộ nhớ, các câu hỏi chậm đến theo cụm (nhiều câu liền nhau cùng chờ 9–13 s). p95 của 100 câu là khoảng câu chậm thứ 5. Chỉ cần một cụm nhiều hay ít hơn là p95 đổi cả giây.

Tăng số câu mỗi lần đo không giải quyết được:
- Nhiễu giảm theo căn bậc hai số câu. Muốn từ 20% xuống 5% cần gấp **16 lần** số câu.
- Gấp đôi số câu chỉ giảm nhiễu xuống ~14%, nhưng P04 tốn thêm ~5 giờ GPU.

**Cách làm đã chọn** (đã ghi vào docs/workloads.md và challenges.md #5):
- Mỗi mức đo 3 lần, quyết định theo **trung vị**.
- So hai cấu hình bằng **khoảng goodput** (ví dụ "16 đạt, 18 trượt"), không bằng TTFT p95 của một lần đo.
- Ngưỡng 5% chỉ dùng cho throughput và TPOT.

### Câu bạn hỏi: đợt đầu và đoạn cuối có làm lệch p95 không? → có, và vì sao vẫn tính trên mọi câu

So hai cách tính trên cùng dữ liệu:
- **all**: mọi câu.
- **steady**: bỏ N câu gửi đầu (lúc mọi người cùng bắt đầu) và N câu gửi cuối (lúc số câu đang chạy giảm dần).

| Bài đo | TTFT p95 all | TTFT p95 steady | Chênh |
|---|---|---|---|
| W1, 16 người | 1.09 s | 0.31 s | −71% |
| W2, 8 người | 2.35 s | 1.50 s | −36% |
| W2, 16 người (3 lần) | | | −10%, −21%, −11% |
| W2, 32 người | 25.1 s | 23.9 s | −5% |

Đúng là có lệch. Trước khi đo tôi định: lệch quá ~5% thì chuyển sang steady. Nhưng số đo cho thấy ba điều:
1. Steady chỉ còn ~2/3 số câu (68 thay vì 100), nên **nhiễu tăng gần gấp đôi** (31–36% so với 19%), lớn hơn cả phần lệch nó sửa.
2. Chỗ lệch nhiều (−36%, −71%) là lúc TTFT chỉ 0.3–2 s, còn rất xa ngưỡng 10 s, nên không đổi kết luận đạt/trượt. Gần ngưỡng chỉ lệch 5–10%.
3. Tính trên mọi câu thì kết quả xấu hơn thật một chút, tức thận trọng. Đây cũng là cách vLLM và các bài báo báo số. Mọi cấu hình đo cùng một cách nên so với nhau vẫn công bằng.

→ **Giữ cách "all"**. Công cụ lưu số liệu từng câu, nên báo cáo P04 vẫn có thêm goodput theo "steady" mà không cần đo thêm. Nếu hai cách cho kết luận khác nhau, báo cáo sẽ ghi rõ.

### Client tự viết (cho W4) có đáng tin không?

Cùng bài W2 với 16 người, cùng các câu hỏi, chạy xen kẽ với `vllm bench serve` trên cùng server. Gộp 2 lần đo, mỗi bên 200 câu:

| Chỉ số | Client tự viết so với vLLM |
|---|---|
| Chữ ra / giây | −1% |
| TTFT p50 / p95 | +5.3% / **+3.7%** |
| TPOT p95 | +0.4% |
| E2EL trung vị | −0.5% |

Lệch < 5% (TTFT p50 hơi nhỉnh, nằm trong nhiễu) → **dùng được cho P05**.

---

## 7. Ước lượng lại chi phí P04 (trước khi chạy)

Từ thời gian đo thật:
- W1: ~1.2 giây mỗi câu hỏi.
- W2: ~1.2 giây mỗi câu.
- W3: ~3.3 giây mỗi câu (hợp đồng dài, đọc lâu).
- Khởi động lại server: ~2.3 phút.

| Phần | Giờ GPU |
|---|---|
| Lưới 8/16/32/64 × 3 workload × 3 lần | 2.9 |
| Chia đôi tìm điểm gãy: W1 (có thể phải lên 128), W2 (16–32), W3 (chia xuống 4) | ~2.2 |
| **P04** | **~5.1 h ≈ 38k VND** (xấu nhất ~6.1 h ≈ 45k) |

Cả phiên (P03 + P04) khoảng 47–54k VND, dưới mức 100k bạn cho phép, nên tôi chạy P04 luôn trên cùng máy, không tắt máy.

---

## 8. Dòng thời gian

```
22:36–23:27  viết + thử công cụ trên laptop (0 đồng)
23:28:16     thuê máy (cùng máy P01)
23:28:55     máy chạy
23:29:35     bắt đầu cài đặt + đo
23:32:41     cài xong (186 s) → lỗi tee, script dừng           (lỗi 1)
23:40:55     sửa, chạy lại → bật server lần 1 (343 s, KV 28,704)
23:47        lỗi thiếu pandas                                   (lỗi 2)
23:48:24     sửa, chạy lại → bắt đầu đo
00:05        xong lần lặp 1 → khởi động lại server (KV 37,264)
00:17:30     xong 10 lần đo
00:19:15     bắt đầu P04 trên cùng máy
```

**Chi phí phần P03** (tính theo thời gian, vì hoá đơn Vast gộp cả P03 + P04): 50 phút × $0.2833/giờ ≈ $0.24, cộng tải model + phần mềm ~$0.03 → **≈ 7,000 VND**.

---

## 9. Rút ra điều gì

1. **Bộ nhớ KV quyết định tất cả với W2/W3.** Chỉ +30% KV đã làm TTFT p95 giảm một nửa. Hướng tune đầu tiên là lưu KV 8 bit (FP8), gấp đôi chỗ.
2. **Không bao giờ đo trên lần bật server đầu tiên của máy mới.** Công cụ đã tự xử lý.
3. **TTFT p95 của một lần đo không đủ tin để so hai cấu hình chênh nhau vài %.** Phải dùng trung vị 3 lần và khoảng goodput.
4. Công cụ đo đúng (khớp token từng câu, khớp đồng hồ server), chạy tiếp được sau khi bị ngắt, và client W4 tin được.

## Bước tiếp theo

**P04** đang chạy trên cùng máy: đo đầy đủ W1/W2/W3, mỗi mức 3 lần, tự tìm goodput.
Dự đoán từ smoke:
- W2: goodput ~16–24 người.
- W3: dưới 8, phải đo xuống 4.
- W1: có thể trên 64.
