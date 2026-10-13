# Báo cáo P12 — Preemption W1

P12 chạy cùng phiên với P09 (2026-10-14 04:14–04:36, 0.40 h ≈ 3.3k VND). Báo cáo viết chung:
[../09-tang-cho-kv/bao-cao.md](../09-tang-cho-kv/bao-cao.md) mục 3.5 và 4.6; đọc ngay: [../09-tang-cho-kv/doc-ngay.md](../09-tang-cho-kv/doc-ngay.md) bước 4.
Số liệu: [measurements.md](../../measurements.md) mục "P12".

Tóm tắt: điều kiện thoả (173 preemption ở W1/140 với kv8). `--watermark 0.02` và `--max-num-seqs 133` giảm preemption 86–100%, throughput +4%,
nhưng không đạt ở 164 user → **không nhận**. `--max-num-seqs 133` cho thêm 10% chỗ KV (CUDA graph nhỏ hơn) → ứng viên P13.
