# Nhật ký khó khăn & cách khắc phục

> Ghi ngay khi gặp: bối cảnh → triệu chứng → nguyên nhân → phương án đã cân nhắc → quyết định → chi phí/đánh đổi.
> Dùng làm tư liệu cho chương "Khó khăn và giải pháp" khi bảo vệ.

---

## #3 — FlashInfer không biên dịch được kernel vì thiếu `ninja` trong PATH (2026-10-12, P01)

**Bối cảnh.** Khởi động vLLM 0.30.0 lần 2 trên L4. **Triệu chứng.** Server chết ở bước warmup:
`FileNotFoundError: 'ninja'` trong `flashinfer/jit/cpp_ext.py`.
**Nguyên nhân.** FlashInfer biên dịch kernel sampling lúc chạy (JIT) bằng `ninja`. `ninja` có trong venv, nhưng script gọi
`/root/venv/bin/vllm` bằng đường dẫn tuyệt đối, không kích hoạt venv → `ninja` không có trên PATH.
**Quyết định.** `scripts/serve_default.sh` thêm `export PATH="$VENV/bin:$PATH"`. Không đổi cấu hình vLLM.
**Chi phí.** ~3 phút GPU (~150 VND). Bài học: bước biên dịch JIT này còn tốn ~2 phút mỗi máy mới.

## #2 — vLLM mặc định không khởi động được với Qwen3-30B-A3B INT4 trên L4 24 GB (2026-10-12, P01)

**Bối cảnh.** Đo baseline "vLLM mặc định" (chỉ `vllm serve <model>`).
**Triệu chứng.** `ValueError: To serve at least one request with the model's max seq len (40960), 3.75 GiB KV cache is needed,
which is larger than the available KV cache memory (2.63 GiB)`.
**Nguyên nhân.** Weight chiếm 15.7 GiB trên 22 GiB khả dụng; vLLM chỉ dùng 92% → còn 2.63 GiB cho KV cache (~28.7k token,
96 KiB/token). vLLM đòi chứa trọn ít nhất 1 request dài tối đa (40960 token theo config model) → từ chối khởi động.
**Phương án.** (a) tăng `gpu_memory_utilization` — đổi cách chia bộ nhớ, là việc của tune; (b) KV FP8 — cũng là tune;
(c) giảm `max_model_len` — chỉ giới hạn độ dài request, không đổi cách chạy.
**Quyết định.** (c) `--max-model-len 16384` (đủ cho W3: prompt ≤ 8k + output ≤ 350). "Baseline mặc định" của dự án
= mặc định + đúng 1 cờ này. Đã dự đoán trước trong plan P01.
**Chi phí.** ~3 phút GPU. Ý nghĩa cho đồ án: trên GPU 24 GB, model MoE INT4 ~16 GB để lại rất ít chỗ cho KV cache → đây là
giới hạn chính của L4 với workload RAG (xem báo cáo P01).

---

## #1 — Không đọc được GPU performance counter (`ncu`) trên L4 thuê (2026-10-12)

**Bối cảnh.** Dự án cần `ncu` (Nsight Compute) để chẩn đoán vì sao kernel chậm (stall reason, occupancy, cache hit, số transaction bộ nhớ) trước khi viết custom CUDA.

**Triệu chứng.** Trên L4 thuê ở Vast.ai (host 506934, dạng container):
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

**Nguyên nhân.** Từ driver R418+, đọc performance counter cần quyền admin (`CAP_SYS_ADMIN`) hoặc host đặt
`NVreg_RestrictProfilingToAdminUsers=0`. Container cho thuê không được cấp `CAP_SYS_ADMIN`
(RunPod cũng chặn tương tự), người thuê không đổi được cấu hình driver của host.

**Ràng buộc thị trường.** Cùng ngày, Vast có ~5 máy L4, đa số thuộc **cùng host 506934**, và **0 máy L4 dạng VM**
(VM mới có root thật → `ncu` chạy được).

**Phương án đã cân nhắc.**

| Phương án | Chính xác (đúng L4)? | Chi phí | Nhược điểm |
|---|---|---|---|
| Chờ L4 dạng VM trên Vast | ✅ | 0 | Không biết bao giờ có |
| L4 VM trên AWS g6.xlarge | ✅ | ~$0.36–0.80/h × 10–20 h ≈ 100–400k VND | Phải xin quota G-instance trước (vài giờ–vài ngày) |
| L4 VM trên GCP g2-standard-4 | ✅ | ~$0.62–0.71/h | Free trial cấm GPU; tài khoản cá nhân hay kẹt quota |
| RTX 4000 Ada VM trên Vast (cùng die AD104) | ❌ gần đúng | $0.20/h | 48 SM vs 58, 360 vs 300 GB/s → chỉ để khám phá, không báo cáo |
| Không dùng `ncu`, chỉ đo thời gian | ✅ | 0 | Biết "chậm bao nhiêu", không biết "vì sao" |

**Quyết định.** Tách **đo lường** và **chẩn đoán**:
1. Đo lường + tune + mọi số liệu báo cáo: **L4 container giá rẻ**, dùng roofline từ thời gian
   (bytes / time so với 300 GB/s), `nsys` trace, `-Xptxas -v`, `cuobjdump -sass`, occupancy API.
2. Chẩn đoán sâu bằng `ncu`: **L4 VM trên AWS g6** chỉ trong số giờ cần thiết (ước 10–20 h).
3. Theo dõi Vast để bắt L4 dạng VM nếu xuất hiện.

**Trạng thái.** Đang chờ: tạo tài khoản AWS và xin quota G-instance.
