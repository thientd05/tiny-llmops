# P01 — Đọc ngay: thuê máy L4, cài vLLM, chạy thử model

> Bản dễ đọc, kể lại từng bước. Bản đầy đủ số liệu: [bao-cao.md](bao-cao.md).

**Một câu:** thuê một máy có GPU L4 trong 21 phút, tốn khoảng **3,300 VND**. Đã cài vLLM, chạy được model Qwen3-30B-A3B và model trả lời đúng.
Chạy được, nhưng **không phải với cấu hình mặc định nguyên vẹn**: phải thêm một tham số vì GPU thiếu bộ nhớ.

---

## 0. Có hai máy

| | Laptop của bạn ("local") | Máy thuê ("remote") |
|---|---|---|
| Ở đâu | Trước mặt bạn | Trung tâm dữ liệu ở Czechia |
| Có GPU L4? | Không | Có |
| Ai gõ lệnh | Tôi gõ lệnh **trên laptop** | Không ai ngồi trước nó; mọi lệnh được gửi từ laptop sang |

Mọi việc đều xuất phát từ laptop. Lệnh nào cần chạy trên máy thuê thì gửi sang qua **SSH**.
SSH là cách đăng nhập vào máy khác qua mạng: gõ lệnh ở máy mình, lệnh chạy ở máy kia, chữ in ra hiện về máy mình.

Hai thứ đã cài sẵn trên laptop:
- **`vastai`**: chương trình dòng lệnh để làm việc với Vast.ai. Những gì bạn bấm chuột trên trang web Vast (tìm máy, thuê, huỷ) thì nó làm bằng lệnh gõ.
  Nó dùng **API key** (mật khẩu riêng cho chương trình) mà bạn đã lưu trong máy.
- **SSH key `~/.ssh/vast`**: một cặp file dùng thay mật khẩu khi đăng nhập.
  Nửa công khai (`vast.pub`) đã đưa lên Vast; Vast tự chép nó vào mọi máy bạn thuê, nên laptop đăng nhập được mà không cần gõ mật khẩu.

---

## 1. Chuẩn bị trước khi thuê (không tốn tiền)

Máy thuê tính tiền theo giây, nên mọi việc **không cần GPU** đều làm trên laptop trước:

1. **Chọn phiên bản vLLM.** Bản mới nhất là 0.31.0, nhưng mới ra 1 tuần, có thể còn lỗi chưa ai phát hiện.
   Chọn **0.30.0** (đã ra 3 tuần).
   Đọc ghi chú phát hành thì thấy bản này bỏ hỗ trợ một kiểu nén model tên "act-order"; model dùng kiểu đó sẽ cho ra chữ sai.
   Tôi mở file cấu hình của model ra xem: model **không** dùng kiểu này (`desc_act: false`), nên dùng được.
2. **Chọn image** (bộ phần mềm cài sẵn khi máy bật, xem bước 3). vLLM 0.30.0 đi kèm CUDA 13.0 nên chọn image có CUDA 13.0, cho khớp.
   (CUDA là bộ công cụ để chạy chương trình trên GPU NVIDIA.)
3. **Viết 3 script** (file chứa một loạt lệnh, chạy một lần là chạy hết):
   `setup_node.sh` để cài đặt, `serve_default.sh` để bật vLLM, `smoke_test.py` để gửi câu hỏi thử.
4. **Làm xong P02** (bộ câu hỏi dùng để đo), để lấy sẵn một câu hỏi RAG thật đem đi thử.

---

## 2. Tìm máy

```bash
vastai search offers 'gpu_name=L4 num_gpus=1 verified=true reliability>0.98 disk_space>=60 rentable=true' -o 'dph'
```

| Điều kiện | Nghĩa |
|---|---|
| `gpu_name=L4 num_gpus=1` | Có đúng 1 GPU L4 |
| `verified=true` | Máy đã được Vast kiểm tra |
| `reliability>0.98` | Trong quá khứ máy hoạt động ổn định trên 98% thời gian |
| `disk_space>=60` | Còn ít nhất 60 GB ổ đĩa |
| `-o 'dph'` | Sắp theo giá mỗi giờ (dph = dollar per hour), rẻ nhất lên đầu |

Hôm đó có 4 máy, 3 máy thuộc cùng một chủ ở Czechia. Rẻ nhất là `49498406` với giá $0.268/giờ.
Số này là **offer ID**, mã của lời chào cho thuê, chưa phải máy của mình.

---

## 3. Thuê máy

```bash
vastai create instance 49498406 \
  --image nvidia/cuda:13.0.2-devel-ubuntu22.04 \
  --disk 60 --ssh --direct --label p01-vllm-default
```

| Phần | Nghĩa |
|---|---|
| `49498406` | Offer chọn ở bước 2 |
| `--image ...` | Bộ phần mềm cài sẵn khi máy bật: Ubuntu 22.04 + CUDA 13.0 + `nvcc` (trình biên dịch code GPU). Vast tải bộ này về rồi khởi động máy từ nó |
| `--disk 60` | Xin 60 GB ổ đĩa: model 16 GB + phần mềm ~10 GB + phần dư |
| `--ssh --direct` | Cho phép đăng nhập SSH thẳng vào máy |
| `--label` | Tên gợi nhớ, chỉ để mình đọc |

Kết quả: `"new_contract": 55518236`. Đây là **instance ID**, mã của máy mình vừa thuê. Các lệnh sau (xem, huỷ) đều dùng số này.

Đây là loại thuê **không bị lấy lại giữa chừng**. Có loại rẻ hơn gọi là "interruptible": nếu người khác trả giá cao hơn thì máy bị lấy mất ngay.
Dự án không dùng loại đó, vì đang đo mà mất máy thì kết quả hỏng.

---

## 4. Chờ máy bật

```bash
vastai show instance 55518236     # hỏi lại mỗi 10 giây
```

```
14:44     loading  Pulling from nvidia/cuda     ← máy đang tải image về
14:46     loading  Setting up python3...        ← Vast cài thêm phần mềm SSH
14:46:51  running                               ← máy chạy, chắc chắn đã tính tiền
```

---

## 5. Lấy địa chỉ để đăng nhập

```bash
vastai ssh-url 55518236
# → ssh://root@185.101.222.111:17309
```

| Phần | Nghĩa |
|---|---|
| `root` | Tên tài khoản trên máy thuê (tài khoản toàn quyền) |
| `185.101.222.111` | Địa chỉ của máy thuê trên Internet (địa chỉ IP) |
| `17309` | Cổng. Nhiều máy thuê dùng chung một địa chỉ IP, mỗi máy được một số cổng riêng |

Từ đây, mọi lệnh gửi sang máy thuê đều có dạng:

```bash
ssh -i ~/.ssh/vast -p 17309 root@185.101.222.111 'lệnh muốn chạy'
#   └ dùng key nào  └ cổng   └ tài khoản@địa chỉ    └ lệnh chạy BÊN KIA, chữ in ra hiện về laptop
```

**Kiểm tra máy** (checklist, 1 phút):

| Kiểm tra | Kết quả | Ý nghĩa |
|---|---|---|
| `nvidia-smi` | NVIDIA L4, 23 GB bộ nhớ GPU | Đúng loại GPU |
| Đo tốc độ đọc/ghi bộ nhớ GPU | 230.8 GB/s | ~77% mức tối đa 300 GB/s của L4, giống lần đo trước |
| Thử `ncu` (công cụ xem chi tiết bên trong GPU) | Bị chặn | Chủ máy này không cho dùng, đã biết từ trước. Chưa cần ở giai đoạn này |

---

## 6. Đẩy code lên

```bash
scp -i ~/.ssh/vast -P 17309 \
  scripts/setup_node.sh scripts/serve_default.sh scripts/smoke_test.py smoke_samples.jsonl \
  root@185.101.222.111:/root/work/
```

`scp` là lệnh copy file, chạy qua SSH. Đọc từ trái sang phải: copy 4 file này từ laptop vào thư mục `/root/work/` trên máy thuê.
(`scp` ghi cổng bằng `-P` viết hoa, `ssh` ghi `-p` viết thường; đó là quy ước riêng của từng lệnh.)

Chỉ đẩy đúng file cần dùng. Code gốc luôn nằm trên laptop và trong Git; bản trên máy thuê mất cũng không sao.

---

## 7. Cài đặt: `setup_node.sh`

```bash
ssh ... 'cd /root/work && nohup bash setup_node.sh /root/results/p01 > /root/results/p01/setup.out 2>&1 &'
```

- `nohup ... &`: chạy **ở chế độ nền** và **không bị dừng khi SSH ngắt**. Nếu thiếu hai thứ này, laptop mất mạng là việc cài đặt trên máy thuê dừng theo.
- `> setup.out 2>&1`: mọi chữ script in ra được ghi vào file `setup.out` thay vì in lên màn hình, để xem lại sau.

Script làm 4 việc:

```bash
# 1. Cài uv: công cụ cài gói Python, nhanh hơn pip nhiều
curl -LsSf "https://astral.sh/uv/0.12.19/install.sh" | sh

# 2. Tải model 16 GB từ HuggingFace (kho model công khai), CHẠY NỀN — dấu & cuối khối ngoặc...
( uvx ... hf download Qwen/Qwen3-30B-A3B-GPTQ-Int4 --revision 9b534e43... --local-dir /root/models/... ) &

# 3. ...trong lúc model đang tải thì cài vLLM 0.30.0 vào một môi trường Python riêng (/root/venv)
uv venv /root/venv
uv pip install vllm==0.30.0 --torch-backend=cu130

# 4. Chờ tải model xong (wait), rồi ghi thông tin máy + phiên bản mọi gói vào env.txt
wait $DL_PID
```

Bước 2 và 3 chạy **cùng lúc**: một bên tải model từ HuggingFace, một bên tải gói Python.
Kết quả, laptop đọc từ file `setup_times.txt` trên máy thuê:

```
uv_ready          1s
model_downloaded  168s     ← 16 GB trong 168 giây ≈ 95 MB/giây
vllm_installed    193s
all_done          193s     ← cả hai xong sau 3 phút 13 giây
```

Nếu làm lần lượt thì mất khoảng 6 phút. Máy tính tiền theo giây nên tiết kiệm được ~3 phút mỗi lần thuê máy mới.

**Ghim phiên bản** (`0.12.19`, `0.30.0`, mã `9b534e43...` của model): ghi chính xác phiên bản thay vì "bản mới nhất".
Như vậy lần sau cài lại vẫn ra đúng phần mềm này, và kết quả đo so sánh được với nhau.

---

## 8. Bật vLLM: `serve_default.sh`

```bash
ssh ... 'bash /root/work/serve_default.sh /root/results/p01/server.log'
```

Script chạy `vllm serve <model>` ở nền, rồi cứ 2 giây hỏi địa chỉ `localhost:8000/health` một lần:
- server trả lời OK → in thời gian khởi động rồi thoát; server vẫn tiếp tục chạy trên máy thuê;
- tiến trình server chết trước đó → in 40 dòng cuối của log để biết lỗi gì.

`localhost` nghĩa là "chính máy này". Lệnh chạy trên máy thuê nên `localhost:8000` là server vLLM trên máy thuê.

### Lần 1 — mặc định nguyên vẹn → chết

Log báo:

```
To serve at least one request with the model's max seq len (40960),
3.75 GiB KV cache is needed, which is larger than the available KV cache memory (2.63 GiB).
```

Giải thích:
- **Bộ nhớ GPU có 23 GB.** Riêng phần số của model (weight) đã chiếm 15.7 GB.
- **KV cache** là phần bộ nhớ lưu kết quả tính toán của từng chữ (token) đã đọc, để khi sinh chữ tiếp theo không phải tính lại từ đầu.
  Với model này, **mỗi token tốn 96 KB** KV cache.
- Theo file cấu hình của model, một câu hỏi được phép dài tới **40,960 token**, cần 40,960 × 96 KB = **3.75 GB** KV cache.
- Sau khi trừ weight và các phần khác, chỉ còn **2.63 GB**.
  vLLM có quy định: nếu không chứa nổi dù chỉ một câu hỏi dài tối đa thì không khởi động.

Cách sửa: thêm tham số `--max-model-len 16384`, tức giới hạn mỗi câu hỏi tối đa 16,384 token.
Câu hỏi dài nhất của dự án (hợp đồng W3) chỉ ~8,200 token, nên giới hạn này không loại câu hỏi nào.
Tham số này cũng **không đổi cách vLLM tính toán**, chỉ đặt giới hạn độ dài.
→ Từ nay "vLLM mặc định" của dự án = mặc định + đúng 1 tham số này.

### Lần 2 — thêm `--max-model-len 16384` → chết vì lý do khác

```
FileNotFoundError: No such file or directory: 'ninja'
```

vLLM dùng một thư viện tên FlashInfer. Thư viện này **biên dịch một phần code GPU ngay lúc chạy**, cần chương trình `ninja` để điều khiển việc biên dịch.
`ninja` đã được cài trong `/root/venv`, nhưng script không cho hệ thống biết tìm trong thư mục đó.
Sửa script: thêm một dòng báo cho hệ thống tìm chương trình trong `/root/venv/bin`. Đây là lỗi script của tôi, không liên quan cấu hình vLLM.

### Lần 3 → chạy

```
ready_after 235s
memory.used 21057 MiB / 23034 MiB
```

Server sẵn sàng sau 235 giây. Bốn phút này gồm:

| Việc | Giây | Là gì |
|---|---|---|
| Đọc model 16 GB từ ổ đĩa vào GPU | 53 | |
| Biên dịch model (torch.compile) | 1 | Lần 1 mất 62 giây; từ lần 2 dùng lại kết quả đã lưu |
| Ghi sẵn chuỗi lệnh GPU (CUDA graph) | 27 | Lưu sẵn để mỗi bước chạy sau nhanh hơn |
| FlashInfer biên dịch code (phỏng đoán) | ~110 | Khoảng trống trong log, ngay chỗ lần 2 bị lỗi `ninja` |
| Còn lại (khởi tạo, kiểm tra bộ nhớ...) | ~44 | |

**Bộ nhớ GPU chia ra sao:**

```
22.5 GB bộ nhớ GPU (nvidia-smi báo 23,034 MiB)
├── 15.7  weight model
├──  0.4  phần ngoài (thư viện CUDA...)
├──  0.8  bộ nhớ tạm khi tính
├──  0.3  chuỗi lệnh GPU ghi sẵn
├──  3.4  KV cache  → chứa được 37,264 token
└── ~1.9  để trống (vLLM cố ý không dùng hết, chừa khoảng 8% phòng khi thiếu)
```

**Điều lạ:** lần 1 KV cache chỉ được 2.63 GB, lần 3 được 3.41 GB, khác nhau 0.78 GB.
Khác biệt duy nhất giữa hai lần là lần 1 phải biên dịch model từ đầu. Tôi đoán bước biên dịch giữ lại một phần bộ nhớ, nhưng chưa chứng minh được.
Sẽ kiểm tra ở P03. Nếu đúng thì mỗi lần thuê máy mới phải bật server một lần cho nó biên dịch, tắt đi, rồi mới bật lại để đo.

---

## 9. Hỏi thử: `smoke_test.py`

```bash
ssh ... '/root/venv/bin/python /root/work/smoke_test.py smoke_samples.jsonl /root/results/p01/smoke.json'
```

Script này chạy **trên máy thuê** và gửi câu hỏi tới server trên chính máy đó.
Không gửi từ laptop, vì đường mạng Việt Nam → Czechia mất ~0.2 giây mỗi chiều, sẽ làm sai số đo thời gian.

| Câu hỏi | Chữ vào (token) | Chữ ra (token) | Thời gian |
|---|---|---|---|
| 3 câu ngắn, **bật** chế độ suy nghĩ | 22–36 | 422 / 2345 / 1018 | 4.6 / 26.8 / 11.1 giây |
| Cùng 3 câu, **tắt** chế độ suy nghĩ | 26–40 | 39 / 202 / 54 | 0.5 / 2.2 / 0.6 giây |
| Câu hỏi RAG (tài liệu kỹ thuật IBM + câu hỏi) | 2,997 | 150 | 2.3 giây |
| Tóm tắt hợp đồng dài nhất | 7,879 | 265 | 5.3 giây |

- **Chế độ suy nghĩ (thinking)** của Qwen3: model tự viết một đoạn suy luận dài trước khi trả lời.
  Bật lên thì câu trả lời dài hơn **10–19 lần**, và dài bao nhiêu là do model tự quyết.
  Khi đo tốc độ, mình cần biết trước mỗi câu trả lời dài bao nhiêu, nên **tắt** chế độ này cho mọi bài đo.
- Số token vào (2,997 và 7,879) **khớp đúng** với số P02 đã đếm trên laptop → bộ câu hỏi P02 dùng được.
- Câu trả lời đọc vào có nghĩa: câu RAG trả lời đúng theo quy tắc trong system prompt, câu hợp đồng tóm tắt đủ 8 mục.

---

## 10. Kéo kết quả về rồi huỷ máy

```bash
# Kéo kết quả: máy thuê → laptop
rsync -az -e "ssh -i ~/.ssh/vast -p 17309" \
  root@185.101.222.111:/root/results/p01/  results/gd1/01-cai-dat-chay-thu/20261012-p01/
```

`rsync` cũng copy qua SSH giống `scp`, nhưng chỉ copy phần mới hoặc đã thay đổi; chạy lại nhiều lần không phải copy lại file cũ.
Chiều ở đây ngược với bước 6: máy thuê → laptop.

```bash
vastai destroy instance 55518236 -y     # huỷ máy; -y = không hỏi lại "có chắc không?"
vastai show instances                    # → "0 instances": không còn máy nào tính tiền
vastai show invoices                     # xem từng khoản bị trừ tiền
```

`destroy` xoá máy cùng **mọi thứ trên đó** (model, phần mềm, log). Vì vậy luôn kéo kết quả về trước rồi mới huỷ.
Cách còn lại là `stop`: tắt máy nhưng giữ ổ đĩa, vẫn trả tiền ổ đĩa. Dự án **không dùng** `stop`.
Lúc máy dừng, GPU được trả về cho người khác thuê. Thị trường L4 rất ít máy, nên lúc quay lại dễ thấy GPU đã bị chiếm, vẫn phải thuê máy mới và cài lại,
cộng thêm tiền ổ đĩa đã trả. Cài lại chỉ mất ~7 phút, ~1,600 VND. Quy tắc: quay lại trong ~10 phút thì để máy chạy, lâu hơn thì `destroy`.

---

## 11. Toàn bộ phiên theo thời gian

```
14:43:36  thuê máy
14:46:51  máy chạy
14:47     đẩy script lên, bắt đầu cài, kiểm tra máy
14:51     cài xong (3 phút 13 giây)
14:51     bật vLLM mặc định                     → chết: không đủ bộ nhớ cho câu hỏi 40,960 token
14:55     bật lại, giới hạn 16,384 token        → chết: thiếu ninja
14:58     sửa script, bật lại                   → chạy sau 235 giây
15:02     hỏi thử 8 câu (1 phút)
15:03     kéo kết quả về laptop
15:03:58  huỷ máy, kiểm tra còn 0 máy
```

**Chi phí** (theo hoá đơn Vast):

```
GPU      0.346 giờ   $0.092
Ổ đĩa    0.394 giờ   $0.007
Tải về   21 GB       $0.027   ← model 16 GB + phần mềm 5 GB; Vast tính phí theo GB tải về
Tổng                 $0.126 ≈ 3,300 VND
```

Hoá đơn tính 0.346 giờ (~21 phút), trong khi máy ở trạng thái "running" chỉ ~17 phút. Vậy một phần thời gian máy đang bật cũng bị tính tiền.

---

## 12. Rút ra điều gì

1. **Model lượng tử 4 bit chạy được trên L4.** Nhưng bộ nhớ còn lại cho KV cache rất ít: **37,264 token**.
2. **Bao nhiêu câu hỏi chạy cùng lúc được** (phép tính, chưa đo):
   - vLLM không giữ chỗ trước cho câu hỏi; câu hỏi cần thêm chỗ thì mới cấp thêm.
   - Nhưng toàn bộ phần tài liệu của một câu hỏi RAG (~2,900 token) phải nằm trong KV cache suốt thời gian model trả lời.
   - Vì vậy W2 chỉ chạy cùng lúc được khoảng **15–16 câu hỏi**; W3 (hợp đồng dài) khoảng **6–7**. Người thứ 17 trở đi phải xếp hàng chờ.
   - P03 sẽ đo thật để kiểm tra.
3. **Hướng cải thiện đầu tiên khi tune:** lưu KV cache bằng 8 bit thay vì 16 bit (FP8). Mỗi token chỉ tốn một nửa bộ nhớ, nên chứa được gấp đôi số câu hỏi.
4. **Mỗi lần thuê máy mới tốn cố định khoảng 1,800 VND** (cài đặt, bật server, tải 21 GB), chưa tính lúc đo.
   Vì vậy nên làm P03 và P04 trong cùng một lần thuê.

## Bước tiếp theo

**P03:** viết công cụ đo tự động, thuê máy, đo thử vài điểm.
Mục đích: kiểm tra công cụ đo đúng, và kiểm tra hai dự đoán trên (giới hạn 15–16 câu hỏi; KV cache nhỏ hơn ở lần bật đầu).
