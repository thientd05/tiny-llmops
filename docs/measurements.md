# Nhật ký đo đạc

> Ghi **mọi** số liệu đo được, kèm ngày, máy (host/machine ID), phần mềm và lệnh tái lập.
> Mục mới thêm lên **đầu** file. Không sửa số cũ — nếu đo lại thì thêm mục mới.

---

## 2026-10-12 — Test máy L4 đầu tiên (Vast.ai)

### Thông tin máy

| Trường | Giá trị |
|---|---|
| Thời gian | 13:31 → 13:34 (+07), ~4 phút chạy |
| Instance / offer | 55510860 / offer 35835963 (đã destroy) |
| **Host / machine** | **host 506934 / machine 77226** — Czechia, CZ, verified, reliability 99.75% |
| Giá | GPU $0.3333/h, tổng $0.3407/h (gồm disk 40GB) |
| Phí ổ đĩa | $0.1333 / GB / tháng |
| Phí băng thông | tải lên $0.0013/GB, tải xuống $0.0013/GB (= $1.3/TB) |
| Image | `nvidia/cuda:12.8.1-devel-ubuntu22.04`, launch mode SSH direct |
| GPU | NVIDIA L4, compute cap 8.9, **23034 MiB** khả dụng, power limit 72 W |
| Driver | 615.71.09 (CUDA tối đa 13.4) |
| CPU | Intel Xeon Silver 4114, 40 luồng thấy được, **20 vCPU được cấp** |
| RAM | 128 GB trên host (container thấy 125 GB) |
| Mainboard | 0W23H8 (Dell) |
| **PCIe** | **Gen3 x16**, đo được host↔GPU **8.4 GB/s** (L4 hỗ trợ Gen4 — host này giới hạn) |
| Ổ đĩa | INTEL SSDSC2BA40 (SATA SSD), 2277 MB/s theo Vast |
| Mạng (Vast đo) | down 943.7 Mbps / up 97.1 Mbps |
| OS host | Ubuntu 24.04 |
| Vast DLPerf | 25.3 · total FLOPS 24.2 TFLOPS (FP32) |

### Kết quả đo

| Phép đo | Kết quả | Ghi chú |
|---|---|---|
| Băng thông VRAM (cudaMemcpy D2D 1 GiB × 10, đếm đọc+ghi) | **231.2 GB/s** | ≈ 77% của 300 GB/s lý thuyết. Vast tự đo: 220.7 GB/s |
| Tải từ HuggingFace (500 MB đầu `model.safetensors`) | **80.9 MB/s** (6.48 s) | → model ~17 GB tải ≈ 3.5 phút |
| `nvcc` | 12.8 (`cuda_12.8.r12.8/compiler.35583870_0`) | có sẵn trong image devel |
| `ncu` | Nsight Compute 2025.1.1 có sẵn, **chạy lỗi `ERR_NVGPUCTRPERM`** | host chặn performance counter → xem `challenges.md` |
| `nsys` | **Không có** trong `/usr/local/cuda/bin` của image này | cần cài thêm nếu dùng |
| Model `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | Tồn tại trên HF, 11 file, 1 file `model.safetensors` | |

### Chi phí

- Số dư trước: $10.0000 → sau: $9.99615 → **tốn $0.0038 (~100 VND)**

### Lệnh tái lập

```bash
# Thông tin máy
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader
df -h / ; free -g ; nproc ; nvcc --version
vastai show instance <ID> --raw          # metadata host/machine/pcie/giá

# Tốc độ tải HF
curl -sL -r 0-524287999 -o /dev/null -w 'speed=%{speed_download}B/s\n' \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4/resolve/main/model.safetensors

# Băng thông VRAM
cat > /tmp/bw.cu <<'CU'
#include <cstdio>
int main(){ size_t n=1ull<<30; void *a,*b; cudaMalloc(&a,n); cudaMalloc(&b,n);
 cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
 cudaMemcpy(b,a,n,cudaMemcpyDeviceToDevice); cudaEventRecord(s);
 for(int i=0;i<10;i++) cudaMemcpy(b,a,n,cudaMemcpyDeviceToDevice);
 cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e);
 printf("D2D copy: %.1f GB/s (read+write)\n", 2.0*10*n/ms/1e6); }
CU
nvcc -O2 -arch=sm_89 /tmp/bw.cu -o /tmp/bw && /tmp/bw

# Thử ncu
cat > /tmp/t.cu <<'CU'
__global__ void k(float *x){ x[threadIdx.x] += 1.f; }
int main(){ float *x; cudaMalloc(&x, 1024); k<<<1,256>>>(x); cudaDeviceSynchronize(); return 0; }
CU
nvcc -arch=sm_89 /tmp/t.cu -o /tmp/t && ncu --metrics gpu__time_duration.sum,dram__bytes.sum /tmp/t
```

---

## 2026-10-12 — Ảnh chụp thị trường Vast.ai (~13:00 +07)

### L4 dạng container (1 GPU)

| Offer ID | Nơi | $/h tổng | Reliability | Disk max | Tải xuống $/TB | Disk $/GB/tháng | Host |
|---|---|---|---|---|---|---|---|
| 49498406 | Czechia | 0.268 | 99.8% | 147 GB | 1.3 | 0.20 | — (đã hết khi thuê) |
| 49883396 | Utah, US | 0.323 | 99.0–99.2% | 182 GB | 3.9 | 0.47 | 323419 |
| 35835963 | Czechia | 0.334 | 99.8% | 354 GB | 1.3 | 0.13 | 506934 ← đã test |
| 36579964 | Czechia | 0.334 | 99.9% | 522 GB | 1.3 | 0.13 | 506934 |
| 54285094 | Czechia | 0.335 | 99.2% | — | 2.6 | 0.20 | 506934 |
| 55313287 | Utah, US | 0.337 | 92.8% | 50 GB | 3.9 | 0.60 | — |

→ Nguồn cung L4 rất mỏng (~5 máy), phần lớn từ **cùng host 506934** — host này **chặn ncu**.

### Máy dạng VM (`vms_enabled=true`, 1 GPU)

- **L4 dạng VM: 0 máy.**
- Ada (sm_89) dạng VM: 13 máy

| GPU | Số máy | Giá thấp nhất | Nơi | Offer |
|---|---|---|---|---|
| RTX 4000 Ada (AD104, cùng die với L4) | 1 | $0.201/h | Hungary | 36173713 |
| RTX 4060 Ti | 2 | $0.225/h | Quebec, CA | 49744628 |
| RTX 4080 | 1 | $0.243/h | Florida, US | 54308059 |
| RTX 4090 | 9 | $0.415/h | Denmark | 51927459 |

- Tổng VM mọi loại GPU: 32 máy (nhiều nhất RTX 4090: 9).

### Lệnh tái lập

```bash
vastai search offers 'gpu_name=L4 num_gpus=1 verified=true reliability>0.98 disk_space>=40' -o 'dph'
vastai search offers 'gpu_name=L4 vms_enabled=true' -o 'dph'
vastai search offers 'vms_enabled=true compute_cap=890 num_gpus=1' -o 'dph'
```

---

## Tham khảo giá ngoài Vast (tra cứu 2026-10-12, nguồn bên thứ ba)

| Nhà cung cấp | Máy | Giá |
|---|---|---|
| AWS | g6.xlarge (1×L4, VM) | spot ~$0.36–0.55/h (interruption 10–15%), on-demand $0.80/h (us-east-1) |
| GCP | g2-standard-4 (1×L4, VM) | on-demand ~$0.71/h, spot ~$0.62/h (us-central1); free trial **không** dùng được GPU |
| RunPod | L4 | Community ~$0.44/h, Secure ~$0.49/h |
| Tỷ giá dùng để tính | | ~26,200 VND/USD (tháng 8/2026) |
