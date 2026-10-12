# Nhật ký khó khăn & cách khắc phục

> Ghi ngay khi gặp: bối cảnh → triệu chứng → nguyên nhân → phương án đã cân nhắc → quyết định → chi phí/đánh đổi.
> Dùng làm tư liệu cho chương "Khó khăn và giải pháp" khi bảo vệ.

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
