# Nhật ký khó khăn & cách khắc phục

> Ghi ngay khi gặp: bối cảnh → triệu chứng → nguyên nhân → phương án đã cân nhắc → quyết định → chi phí/đánh đổi.
> Dùng làm tư liệu cho chương "Khó khăn và giải pháp" khi bảo vệ.

---

## #5 — TTFT p95 dao động ~20% giữa hai lần đo cùng một điểm (2026-10-13, P03)

**Bối cảnh.** P03 cần mức nhiễu < 5% để quy tắc "kém hơn > 5% thì loại" (CLAUDE.md quy tắc 10) có nghĩa.
**Triệu chứng.** W2 / 16 user, hai lần đo (r2, r3) trên cùng máy, cùng KV 37,264 token, 100 request mỗi lần:
TTFT p95 4.58 s → 5.55 s (lệch 19%). Trong khi đó throughput ra lệch 1.2%, TPOT p95 3.7%. Tính riêng phần ổn định
(bỏ N request đầu và cuối, còn 68 mẫu) thì TTFT p95 lệch còn nhiều hơn (31%).
**Nguyên nhân (suy ra).** Ở 16 user, KV cache đầy 100%, có 6–7 lần preemption và tới 13 request phải chờ. TTFT của một request
phụ thuộc nó rơi vào lúc hàng đợi dài hay ngắn, nên các request chậm đi theo cụm (thấy trong chuỗi TTFT theo thứ tự gửi).
p95 của 100 mẫu là khoảng request chậm thứ 5. Chỉ cần một cụm preemption nhiều hay ít hơn là p95 đổi vài trăm ms đến vài giây.
**Phương án.**

| Phương án | Ưu | Nhược |
|---|---|---|
| Tăng số request mỗi lần đo (ví dụ ×2) | Giảm nhiễu | Nhiễu p95 chỉ giảm ~√2 (~20% → ~14%); giờ GPU P04 ×2 (~+5 h); W3 chỉ có 343 mẫu |
| Giữ max(100, 4N), quyết định đạt/trượt theo **trung vị 3 lần** (đã có trong thiết kế) | Không tốn thêm | Một lần đo đơn lẻ vẫn nhiễu ~20% |
| So cấu hình bằng **khoảng goodput** thay vì TTFT p95 một lần đo | Đúng chỉ số mục tiêu | Độ phân giải bị giới hạn bởi bước tìm (2 user hoặc 10%) |

**Quyết định.** Giữ max(100, 4N) và trung vị 3 lần lặp. Hai cấu hình được coi là khác nhau khi khoảng goodput không chồng nhau
(docs/workloads.md). Ngưỡng 5% chỉ áp dụng cho throughput và TPOT, vì hai chỉ số này đạt (≤ 3.7%). Với TTFT p95, ghi rõ nhiễu ~20%
ở chế độ bão hoà KV.
**Chi phí.** 0 giờ GPU thêm.
**Cập nhật sau P04 (22 mức × 3 lần lặp).** Nhiễu (max − min) / trung vị: throughput trung vị 1.6% (max 5.1%), TPOT p95 3.6% (max 5.9%),
**TTFT p95 15.7% (p75 23.5%, max 47.8%)** → xác nhận nhận định trên. Hệ quả thấy được: W2 / 20 user có TTFT p95 11.07 / 8.76 / 10.06 s
(trung vị cách ngưỡng 0.6%) → goodput W2 = 18 nhưng 20 nằm ngay trên ranh giới; theo cửa sổ steady thì W2 = 20 (24 trượt).
Kết luận cho các so sánh về sau: chênh goodput ≤ 1 bước tìm (2 user ở W2) chưa đủ để nói là khác nhau.

## #4 — Lần khởi động vLLM đầu tiên trên máy mới có KV cache ít hơn 23% (2026-10-12, P03)

**Bối cảnh.** P01 thấy hai lần khởi động cho KV khác nhau (2.63 vs 3.41 GiB) nhưng hai lần đó khác cả `max_model_len`.
**Triệu chứng.** P03, cùng máy (machine 116596), cùng cờ `--max-model-len 16384`:
lần 1 (compile torch.compile từ đầu, 71 s) → **28,704 token**; lần 2 và 3 (dùng cache compile, 0.6–1.2 s) → **37,264 token**.
Hệ quả đo được: W2 / 16 user có TTFT p95 **9.9 s** ở lần 1 và **4.6–5.6 s** ở lần 2–3. Kết luận đạt/trượt SLO đổi theo lần khởi động.
**Nguyên nhân (phỏng đoán, chưa chứng minh).** vLLM đo bộ nhớ đỉnh lúc chạy thử (profile) để tính phần còn lại cho KV cache.
Khi compile lần đầu, tiến trình còn giữ thêm ~0.78 GiB (bộ nhớ của quá trình compile / Inductor autotune) đúng lúc đo,
nên phần chia cho KV nhỏ hơn. Lần sau nạp kết quả compile từ đĩa nên không có phần này.
**Phương án.** (a) luôn khởi động hai lần trên máy mới, chỉ đo trên lần thứ hai; (b) đặt cứng dung lượng KV (`--kv-cache-memory`)
— là một cờ tune, không dùng cho baseline mặc định.
**Quyết định.** (a), tự động trong harness: `server.recompile_restart_s: 10` — nếu dòng `init engine ... (compilation: X s)` cho X > 10 s
thì tắt và bật lại server trước khi đo. Mọi lần khởi động được ghi KV vào `server_starts.jsonl`. Các điểm r1 của smoke P03 đo trên
lần khởi động 1 chỉ dùng để minh hoạ hiệu ứng này, không dùng làm số baseline.
**Chi phí.** ~2.5 phút GPU mỗi máy mới (~300 VND). Ý nghĩa cho đồ án: kích thước KV cache quyết định goodput W2/W3; một thay đổi
nhỏ về bộ nhớ (0.78 GiB) đổi TTFT p95 gần gấp đôi → hướng tune đầu tiên là tăng chỗ cho KV (KV FP8, `gpu_memory_utilization`).

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
