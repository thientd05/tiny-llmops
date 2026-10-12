# Nhật ký đo đạc

> Ghi **mọi** số liệu đo được, kèm ngày, máy (host/machine ID), phần mềm và lệnh tái lập.
> Mục mới thêm lên **đầu** file. Không sửa số cũ — nếu đo lại thì thêm mục mới.

---

## 2026-10-12 — P01: cài vLLM 0.30.0 và chạy thử Qwen3-30B-A3B-GPTQ-Int4 trên L4

Báo cáo: [gd1-vllm-baseline/01-cai-dat-chay-thu/bao-cao.md](gd1-vllm-baseline/01-cai-dat-chay-thu/bao-cao.md).
Kết quả thô: `results/gd1/01-cai-dat-chay-thu/20261012-p01/`.

### Máy & phần mềm

| Trường | Giá trị |
|---|---|
| Thời gian chạy (tính tiền) | 14:46:51 → 15:03:58 (+07), ~17 phút |
| Instance / offer | 55518236 / 49498406 (đã destroy) |
| Host / machine | **host 506934 / machine 116596** — Czechia (khác machine 77226 lần test trước) |
| Giá | $0.2833/h tổng (disk 60 GB) |
| Image | `nvidia/cuda:13.0.2-devel-ubuntu22.04` (nvcc 13.0.88, ncu 2025.3.1) |
| Driver | 595.84 |
| CPU / RAM | Xeon Silver 4114, 30 luồng thấy được / 94 GB |
| PCIe | Gen3 x16 |
| Phần mềm | vLLM **0.30.0**, torch 2.13.0 (cu130), flashinfer-python 0.6.18.post1, Python 3.12.14 (uv 0.12.19) |
| Model | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441` (desc_act=false, group 128, 15.77 GiB) |

### Checklist máy

| Phép đo | Kết quả |
|---|---|
| Băng thông VRAM (cudaMemcpy D2D 1 GiB × 10) | **230.8 GB/s** (lần trước machine 77226: 231.2) |
| `ncu` | `ERR_NVGPUCTRPERM` — host 506934 vẫn chặn |
| `nsys` | không có trong image |

### Thời gian (giây)

| Bước | Thời gian |
|---|---|
| Cài uv | 1 |
| Tải model 16 GB từ HF (song song với cài vLLM) | 168 (~95 MB/s) |
| Cài vLLM + torch (uv) | 193 tính từ đầu |
| **Setup tổng** | **193** |
| Khởi động lần 1 (mặc định, compile từ đầu) — lỗi | nạp weight 47, torch.compile 62, capture CUDA graph 12 → lỗi KV |
| Khởi động thành công (`--max-model-len 16384`, đã có cache compile) | **235** tới `/health` OK: nạp weight 53, torch.compile 1.2 (cache), capture CUDA graph 11 + 16, khoảng trống ~110 (nghi FlashInfer JIT biên dịch kernel sampling) |

### Cấu hình mặc định vLLM 0.30.0 tự chọn

| Mục | Giá trị |
|---|---|
| Kernel linear GPTQ | `MarlinLinearKernel` |
| MoE | backend `MARLIN` WNA16, `MarlinExperts` (không có dòng "default MoE config" — đường Marlin không dùng config JSON Triton) |
| Attention | `FLASH_ATTN` (FlashAttention 2), chọn từ FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION |
| Sampling top-k/top-p | FlashInfer |
| dtype / KV dtype | float16 / auto (=float16) |
| Prefix caching / chunked prefill | bật / bật |
| torch.compile / CUDA graph | VLLM_COMPILE / FULL_AND_PIECEWISE, max capture size 512 |
| Model runner | V2 |
| `gpu_memory_utilization` | 0.92 |
| `max_model_len` | 40960 (theo config model) → **không khởi động được**, phải đặt 16384 |
| max_num_batched_tokens | 2048 (suy từ "compile range (1, 2048)", chưa thấy in trực tiếp) |
| Sampling mặc định | lấy từ `generation_config.json` của model: temperature 0.6, top_k 20, top_p 0.95 |

### Bộ nhớ (GiB, lần khởi động thành công)

| Phần | GiB |
|---|---|
| VRAM khả dụng / vLLM được dùng (0.92) | 22.04 / 20.27 |
| Weight + non-torch | 16.1 (weight 15.68) |
| Activation đỉnh | 0.76 |
| CUDA graph | 0.33 thực tế (ước tính 0.64) |
| **KV cache** | **3.41 → 37,264 token** (≈ 96 KiB/token, khớp 2×48×4×128×2 byte) |
| nvidia-smi khi chạy | 21,057 / 23,034 MiB |

Lần khởi động đầu (compile từ đầu, max_model_len 40960): KV chỉ **2.63 GiB → 28,688 token**. Chênh 0.78 GiB chưa rõ
nguyên nhân (nghi bộ nhớ do torch.compile chiếm khi compile lần đầu) — kiểm ở P03.

### Smoke test (1 request mỗi lần, chỉ để kiểm tra đúng/sai — không phải số benchmark)

| Request | Prompt tok | Output tok | Thời gian | Ghi chú |
|---|---|---|---|---|
| 3 câu hỏi ngắn, thinking **bật** | 22–36 | 422 / 2345 / 1018 | 4.6 / 26.8 / 11.1 s | ~88–92 tok/s |
| cùng câu, thinking **tắt** | 26–40 | 39 / 202 / 54 | 0.5 / 2.2 / 0.6 s | output ngắn hơn 10–19× |
| W2 `techqa_TRAIN_Q041` (completions, greedy) | **2997** (= số đếm P02) | 150 | 2.3 s | trả lời đúng định dạng system prompt |
| W3 hợp đồng dài nhất (completions, greedy) | **7879** (= số đếm P02) | 265 | 5.3 s | tóm tắt đúng 8 mục |

### Chi phí

Theo hoá đơn Vast (`vastai show invoices`): GPU 0.346 h = $0.092 · ổ đĩa 0.394 h = $0.007 · **tải xuống 21 GB = $0.027**
→ **tổng $0.126 ≈ 3,300 VND**. Vast tính GPU 0.346 h (~21 phút), nhiều hơn ~17 phút đo từ lúc trạng thái `running`
→ một phần thời gian `loading` cũng bị tính. Phí tải xuống (model 16 GB + gói Python ~5 GB) ≈ $0.027 **mỗi phiên máy mới**.

### Lệnh tái lập

```bash
vastai create instance <OFFER> --image nvidia/cuda:13.0.2-devel-ubuntu22.04 --disk 60 --ssh --direct
scp scripts/{setup_node.sh,serve_default.sh,smoke_test.py} <smoke_samples.jsonl> root@<IP>:/root/work/
bash /root/work/setup_node.sh /root/results/p01
bash /root/work/serve_default.sh /root/results/p01/server_default.log                       # mặc định → lỗi KV
bash /root/work/serve_default.sh /root/results/p01/server.log --max-model-len 16384          # chạy được
/root/venv/bin/python /root/work/smoke_test.py smoke_samples.jsonl /root/results/p01/smoke.json
```

---

## 2026-10-12 — P02: bộ workload W1/W2/W3 (local, không GPU)

Báo cáo: [gd1-vllm-baseline/02-bo-sinh-workload/bao-cao.md](gd1-vllm-baseline/02-bo-sinh-workload/bao-cao.md).
Chi tiết + sha256: `workloads/manifest.json`. Đếm bằng tokenizer Qwen3, **tính cả chat template**.

| Workload | Số mẫu | Prompt token min / TB / p50 / p95 / max | Output token min / TB / p50 / p95 / max | Phần đầu chung | Nguồn |
|---|---|---|---|---|---|
| W1 | 2000 | 200 / 407 / 368 / 726 / 800 | 200 / 333 / 272 / 600 / 600 | 3 token | ShareGPT (lượt hỏi đầu, ≥95% ký tự ASCII) |
| W2 | 1253 | 2000 / 2858 / 2785 / 3887 / 3999 | 150 / 174 / 150 / 287 / 400 | **622 token** | RAGBench train+val: techqa 704, finqa 326, delucionqa 127, tatqa 80, emanual 16 |
| W3 | 343 | 4008 / 5617 / 5577 / 7483 / 7879 | 250 / 299 / 301 / 344 / 350 | 95 token | CUAD (mọi split), 343 hợp đồng khác nhau, 286 bị cắt tại ranh giới đoạn |

Build lại lần 2 → sha256 trùng (tái lập OK). Thời gian build ~100 s trên laptop.

```bash
.venv/bin/python workloads/build.py           # build
.venv/bin/python workloads/build.py --check   # kiểm tra tái lập
```

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
