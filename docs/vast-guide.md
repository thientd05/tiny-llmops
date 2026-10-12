# Hướng dẫn thuê & bật/tắt L4 trên Vast.ai

> CLI đã cài tại `~/.local/bin/vastai`. SSH key: `~/.ssh/vast` (private, **không chia sẻ**) / `~/.ssh/vast.pub` (đã thêm lên Vast).

---

## 0. Việc làm 1 lần

1. **Nạp tiền**: [cloud.vast.ai/billing](https://cloud.vast.ai/billing/) → *Add Credit* ($10 ≈ 35 giờ L4).
   Bật **low-balance email alert** (ví dụ ở mức $3).
2. **API key cho CLI**: [cloud.vast.ai/manage-keys](https://cloud.vast.ai/manage-keys/) → copy key → chạy trong terminal của bạn:
   ```bash
   vastai set api-key <API_KEY>
   vastai show user          # kiểm tra: thấy email + balance là OK
   ```
3. Kiểm tra SSH key đã có trên tài khoản: `vastai show ssh-keys`

---

## 1. Ba trạng thái của máy — hiểu cái này để không mất tiền oan

| Trạng thái | Tính tiền GPU | Tính tiền ổ đĩa | Dữ liệu | Lưu ý |
|---|---|---|---|---|
| **Running** | ✅ theo giây | ✅ | Còn | Đang dùng |
| **Stopped** | ❌ | ✅ (vẫn tính!) | Còn | Bật lại có thể phải **chờ** nếu người khác đang thuê GPU đó |
| **Destroyed** | ❌ | ❌ | **Mất hết** | Không khôi phục được |

**Quy tắc chọn stop hay destroy** (ví dụ máy 140GB, phí đĩa $0.20/GB/tháng ≈ $0.04/giờ khi stop; dựng lại máy mới ≈ $0.12):

- Nghỉ **< 3 giờ** (ăn trưa, họp) → **stop**
- Nghỉ **qua đêm / nhiều ngày** → **destroy**

Hết tiền → máy bị stop; **nếu không lưu thẻ, máy và dữ liệu bị xoá**. Vì vậy: code luôn ở Git, kết quả luôn rsync về máy local.

---

## 2. Thuê máy (mỗi phiên làm việc)

### Cách A — Web (dễ nhất)

1. [cloud.vast.ai/create](https://cloud.vast.ai/create/) → chọn template → bấm *Edit*:
   - **Image**: `nvidia/cuda:13.0.2-devel-ubuntu22.04` (có `nvcc`, cần cho custom CUDA sau này)
   - **Launch mode**: *SSH* (bật *Use direct SSH connection*)
   - **Disk**: `140` GB
2. Bộ lọc bên trái: GPU = **L4**, số GPU = 1, *Verified*, Reliability ≥ 98%.
3. Không chọn máy **Interruptible** (bị lấy máy giữa chừng → hỏng benchmark).
4. So sánh giá **trọn gói**: rê chuột lên giá để xem phí ổ đĩa + băng thông của host.
5. Bấm **Rent** → vào trang [Instances](https://cloud.vast.ai/instances/), chờ trạng thái *Running* (lúc *Loading* không tính tiền).

### Cách B — CLI

```bash
# Tìm máy L4, rẻ nhất lên đầu
vastai search offers 'gpu_name=L4 num_gpus=1 verified=true reliability>0.98 disk_space>=140' -o 'dph'

# Thuê theo ID ở cột đầu tiên
vastai create instance <OFFER_ID> \
  --image nvidia/cuda:13.0.2-devel-ubuntu22.04 \
  --disk 140 --ssh --direct

vastai show instances       # lấy INSTANCE_ID, chờ status = running
```

---

## 3. Kết nối

```bash
vastai ssh-url <INSTANCE_ID>      # ra dạng ssh://root@IP:PORT
```

**Terminal:**
```bash
ssh -i ~/.ssh/vast -p <PORT> root@<IP> -L 8000:localhost:8000
```
`-L 8000:...` → API server trên máy thuê hiện ra ở `localhost:8000` trên máy bạn (không cần mở port ra internet).

**VS Code (Remote-SSH)** — sửa `~/.ssh/config`, mỗi máy mới chỉ cần đổi `HostName` và `Port`:
```
Host vast-l4
  HostName <IP>
  Port <PORT>
  User root
  IdentityFile ~/.ssh/vast
  LocalForward 8000 localhost:8000
  StrictHostKeyChecking accept-new
```
Sau đó: `Ctrl+Shift+P` → *Remote-SSH: Connect to Host* → `vast-l4`.

> Máy mới trùng IP:port với máy cũ có thể báo lỗi "host key changed" → chạy `ssh-keygen -R "[<IP>]:<PORT>"`.

---

## 4. Checklist 10 phút đầu — không đạt thì destroy ngay

```bash
nvidia-smi                          # đúng L4, 24GB, driver hỗ trợ CUDA ≥ 13.0 (driver ≥ 580)
df -h / && free -g && nproc         # còn ~140GB, RAM ≥ 32GB
nvcc --version                      # có compiler CUDA

# Tốc độ tải (cần ≥ 50 MB/s để tải model ~17GB trong vài phút)
curl -o /dev/null -w 'speed: %{speed_download} B/s\n' \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/resolve/main/config.json

# Thử ncu (performance counter) — ghi lại kết quả (host_id, có/không) vào nhật ký dự án
cat > /tmp/t.cu <<'EOF'
__global__ void k(float *x){ x[threadIdx.x] += 1.f; }
int main(){ float *x; cudaMalloc(&x, 1024); k<<<1,256>>>(x); cudaDeviceSynchronize(); }
EOF
nvcc -arch=sm_89 /tmp/t.cu -o /tmp/t && ncu /tmp/t 2>&1 | tail -5
#  → thấy bảng metric        : host cho phép profiling 🎉 (ghi lại host_id!)
#  → thấy ERR_NVGPUCTRPERM   : bình thường với container; dùng phương án dự phòng
```

> Image `nvidia/cuda:13.0.2-devel` có sẵn `nvcc` 13.0 và `ncu` (2025.3.1) nhưng **không có `nsys`** — cài thêm khi cần.
> Kết quả các lần test máy: xem [measurements.md](measurements.md). Host 506934 (Czechia) **chặn ncu**.

---

## 5. Kết thúc phiên

```bash
# Trên máy local: kéo kết quả về
rsync -avz -e "ssh -i ~/.ssh/vast -p <PORT>" root@<IP>:/root/results/ ./results/

# Nghỉ ngắn (< 3h)
vastai stop instance <INSTANCE_ID>
vastai start instance <INSTANCE_ID>     # bật lại

# Nghỉ dài / xong việc (CLI sẽ hỏi xác nhận; thêm -y để bỏ qua)
vastai destroy instance <INSTANCE_ID>
```

Trên web: trang [Instances](https://cloud.vast.ai/instances/) có nút ⏹ (stop), ▶ (start), 🗑 (destroy).

**Trước khi đi ngủ luôn chạy:** `vastai show instances` → đảm bảo không còn máy nào *running* ngoài ý muốn.

---

## 6. Theo dõi chi tiêu

```bash
vastai show user            # số dư hiện tại
vastai show invoices        # lịch sử trừ tiền
```
Web: [Billing](https://cloud.vast.ai/billing/) → tab *Charges* (chi tiết GPU / disk / băng thông theo từng máy).

---

## 7. Lỗi hay gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `Permission denied (publickey)` | Thiếu `-i ~/.ssh/vast`, hoặc key thêm **sau** khi tạo máy → destroy, tạo lại |
| Bấm start nhưng máy đứng ở *scheduling* | GPU đang bị người khác thuê → chờ, hoặc destroy và thuê máy khác |
| Tải model rất chậm | Host mạng yếu → destroy, chọn máy `inet_down` cao hơn |
| `ERR_NVGPUCTRPERM` khi chạy `ncu` | Container không có quyền đọc counter → xem phương án dự phòng (L4 VM trên AWS g6) |
| Số dư tụt dù không dùng | Còn máy *stopped* đang tính phí ổ đĩa → destroy |
