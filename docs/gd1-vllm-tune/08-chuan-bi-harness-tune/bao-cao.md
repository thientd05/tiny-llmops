# Báo cáo P08 — Chuẩn bị harness cho tune

> Ngày làm: 2026-10-13 (phiên trước, phần đầu) → 2026-10-14 01:50–02:15 (phần cuối) · Laptop, server giả · Giờ GPU: 0 (0 VND)
> Plan: [plan.md](plan.md) · Tiếp theo: [P09](../09-tang-cho-kv/bao-cao.md)

## 1. Tóm tắt

- Harness đo được **nhiều cấu hình nối tiếp trong một phiên thuê**, mỗi cấu hình chỉ ghi phần khác so với file chung, và **chạy tiếp được**
  khi bị giết giữa chừng (thử bằng SIGKILL cả nhóm tiến trình với server giả: đơn vị dở bị xoá và đo lại, đơn vị xong không bị đo lại).
- Mỗi lần khởi động server ghi lại **backend attention, kiểu dữ liệu KV, backend MoE, `max_num_batched_tokens`** mà vLLM thật sự chọn
  (đã kiểm regex trên log thật của P04: `FLASH_ATTN`, `auto`, `MARLIN`, `2048`).
- Có **cổng chất lượng cho biến thể** (`quality/analyze.py --gate`), so với **cả hai** lần chạy baseline P06 theo quy tắc "kém đi" đặt trước,
  và cấu hình cổng cho kv8 (`quality/configs/tune-kv8.yaml`).
- Có `bench/compare.py` so cặp từng đơn vị cùng seed (W1–W3), `--status` ước giờ bằng đơn giá đo thật ở P04.
- **Hoãn tới trước P10** (không cần cho P09/P12): W4 đo ở mức cố định (việc 2), metric `prompt_tokens_by_source` (việc 6),
  client gán độ ưu tiên (việc 7), phần W4 của `compare.py` (việc 4). Lý do: tiết kiệm thời gian đêm nay để kịp chạy P09 + P12 trong một phiên;
  các việc này sẽ thử bằng server giả trước phiên P10, không tốn GPU.

## 2. Đã làm gì và vì sao

| # | Việc (plan) | Trạng thái | File | Đã thử thế nào |
|---|---|---|---|---|
| 1 | Cấu hình biến thể: `include:` + gộp sâu | Xong | `bench/common.py` (`deep_merge`, `read_config_tree`) | Biến thể ghi `khoá: null` để bỏ khoá của file nền (ví dụ bỏ W1, W3 trong điểm mốc). `--dry-run` của `p09-anchor.yaml`, `p09-kv8.yaml` ra đúng đơn vị; `quality/configs/tune-kv8.yaml` gộp đúng với `p06.yaml` |
| 2 | W4 ở mức cố định | **Hoãn tới trước P10** | — | — |
| 3 | Nhiều cấu hình nối tiếp, chạy tiếp được | Xong | `scripts/remote_multi.sh` | `bench/test_fake.sh … multi`: 2 cấu hình bench (A chế độ grid, B chế độ search khác cờ server) + 1 cổng chất lượng. Lần 1 bị SIGKILL cả nhóm tiến trình khi B đang đo đơn vị thứ hai; lần 2 cùng lệnh: A bỏ qua, B xoá đơn vị dở `w1-c52-r1` rồi đo lại, đơn vị đã xong không đổi thời gian sửa file; B ra goodput 48 (52 trượt), đúng sức chứa server giả 48 |
| 4 | So cặp `compare.py` | Xong phần W1–W3; **phần W4 hoãn** | `bench/compare.py` | Chạy trên P03 so với P04: thấy đúng đơn vị P03 r1 KV ít (TTFT p95 W2/16 9.91 → 6.30 s) |
| 5 | Ghi cấu hình thật vLLM chọn | Xong | `bench/sweep.py` (`Server.start`) | Regex thử trên `results/gd1/04-.../server/start-002.log`: attention `FLASH_ATTN`, KV `auto`, MoE `MARLIN`, `max_num_batched_tokens` 2048, KV 37,264 token |
| 6 | Metric `prompt_tokens_by_source` | **Hoãn tới trước P10** | — | — |
| 7 | `--priority-by-length` | **Hoãn tới trước P10** | — | — |
| 8 | Cấu hình cổng chất lượng biến thể + so với cả hai lần P06 | Xong | `quality/configs/tune-kv8.yaml`, `quality/analyze.py --gate`, `quality/run.py` (`source_path`) | Baseline so với chính nó → đạt; server giả (số vô nghĩa) → đủ 5 dấu hiệu "kém đi" được bật, tức `--gate` đọc đúng tên đơn vị của biến thể |
| 8b | Hệ số mở rộng `search.grow`, thêm `search.shrink` | Xong | `bench/common.py` | `bench/test_search.py` vẫn qua; bảng ở mục 3.2 |
| 9 | Ước giờ trong `--status` | Xong | `bench/sweep.py` (`DEFAULT_S_PER_REQ`) | `p09-kv8 --status` khi chưa đo gì: ~1.05 h (gồm ước 4 mức thêm mỗi workload) |
| 10 | Thử với server giả | Xong | `bench/test_fake.sh` (ca `multi`), `bench/configs/test-fake-multi-{a,b}.yaml`, `quality/configs/test-fake-tune.yaml` | Như việc 3 |
| — | Theo dõi phiên dài từ laptop | Mới | `scripts/watch_remote.sh` | Kéo kết quả mỗi 60 s, đọc dòng log mới từ vị trí lưu trong file, thoát khi có đơn vị xong / lỗi / 20 phút |
| — | Kiểm cú pháp cờ vLLM 0.30.0 | Xong | — | Trong `vllm/engine/arg_utils.py` và `vllm/config/cache.py`, `config/scheduler.py` (references F3): `--kv-cache-dtype` nhận `fp8`; `--gpu-memory-utilization`; `--watermark` (float, 0 ≤ x < 1, mặc định 0, chú thích "tránh preemption lặp"); `--max-num-seqs`; `--attention-backend` |

**Quyết định tự đưa ra:**
- **Thêm `search.shrink`** (mở rộng xuống khi mọi mức khởi đầu nóng đều trượt). Cũ: chia đôi (W1 72 trượt → đo 36). Với khởi đầu nóng ở
  ngay sát goodput dự kiến, chia đôi tốn thêm một mức đo (~3–5 phút). `p09-kv8.yaml` dùng `grow 1.5`, `shrink 0.75`; mặc định giữ ×2 / ×0.5 như cũ.
- **Teacher forcing của biến thể chạy trên văn bản tham chiếu của baseline** (`source_path`), không trên văn bản biến thể tự sinh: mọi cấu hình
  được chấm trên cùng một chuỗi token, so trực tiếp được với `ref-tf-c8-r1` của baseline.
- **Cổng chất lượng không chạy GSM8K, IFEval** (giữ như plan): FinQA (đúng bài toán RAG) + MMLU-Pro (có số công bố) + độ gần token đủ cho quyết định;
  hai bài còn lại để cấu hình cuối (README "Việc đã hoãn").

## 3. Kết quả

### 3.1 Lệnh chạy P09 và danh sách đơn vị

```bash
# local: đẩy code + dữ liệu + văn bản tham chiếu baseline P06 (chunks chỉ có ở local, 28 MB)
rsync -az --relative bench/*.py bench/configs scripts/*.sh quality/*.py quality/*.sh quality/configs quality/manifest.json \
  workloads/*.py workloads/manifest*.json workloads/prompts data/workloads/w{1,2,3}.jsonl data/quality \
  results/gd1/06-chat-luong-baseline/20261013-p06/units/ref-gen-c8-r1 root@<IP>:/root/td/
# máy thuê, trong tmux:
bash scripts/remote_multi.sh bench/configs/tune/p09-list.txt
```

| Bước (`p09-list.txt`) | Cấu hình | Đơn vị |
|---|---|---|
| 1 | `bench/configs/tune/p09-anchor.yaml` (mặc định) | `w2-c16-r1..r3` |
| 2 | `quality/configs/tune-kv8.yaml` | `ref-tf-c8-r1`, `ref-gen-c8-r1`, `finqa-c32-r1`, `lm-mmlu-pro` |
| 3 | `bench/configs/tune/p09-kv8.yaml` | đợt đầu `w1-c72`, `w1-c88`, `w2-c20`, `w2-c26`, `w3-c8`, `w3-c12` (r1), rồi tìm nhị phân |

### 3.2 Bước tìm khi khởi đầu nóng (p09-kv8, `grow 1.5`, `shrink 0.75`)

| Khởi đầu nóng | Kết quả hai mức | Mức đo tiếp |
|---|---|---|
| W1 [72, 88] | trượt / trượt | 54 (cũ: 36) |
| W1 [72, 88] | đạt / đạt | 132 (cũ: 176) |
| W2 [20, 26] | đạt / đạt | 40 (cũ: 52) |
| W2 [20, 26] | đạt / trượt | 22 (nhị phân) |
| W3 [8, 12] | trượt / trượt | 6 (cũ: 4) |

## 4. Giải thích

- Đơn giá `--status` (giây / request của một đơn vị, gồm warmup và chờ request dở; trung vị P04 `units.csv`): W1 0.63 [0.58–1.67],
  W2 1.25 [1.11–1.47], W3 3.19 [3.02–3.51]. Đơn vị có N lớn hơn 25 (W1) gửi 4N request, nên thời gian tăng theo N.
- Server giả không có backend attention nên trường `attention_backend` trong `server_starts.jsonl` để trống khi thử; với vLLM thật đã kiểm trên log P04.

## 5. Vấn đề gặp phải

- `remote_multi.sh` ghi cứng đường dẫn máy thuê (`/root/venv`, `setup_node.sh`) nên không thử được ở local. Sửa: biến môi trường
  `MULTI_PY`, `MULTI_SKIP_SETUP`, `MULTI_SETUP_OUT`, `MULTI_LMEVAL_VENV` (mặc định giữ đường dẫn máy thuê).
- Thử ngắt bằng `timeout -s KILL bash remote_multi.sh` chỉ giết bash, tiến trình `sweep.py` con vẫn chạy. Máy chết thật thì mọi tiến trình
  cùng chết, nên test dùng `setsid` + `kill -KILL -- -PGID` để giống thật.
- `quality/run.py`: thông báo lỗi "thiếu prompt_ids" dùng `u['source']`, sẽ lỗi KeyError khi đơn vị chỉ có `source_path`. Đã sửa.

## 6. Ảnh hưởng tới bước sau

- Trước phiên P10 (local, không GPU): làm việc 2, 6, 7 và phần W4 của `compare.py`; thử bằng `bench/test_fake.sh w4` + ca mới.
- P12 dùng chế độ `grid` trên nền cfg1 của P09 (cấu hình viết trong phiên, sau khi biết goodput W1 của cfg1).

## 7. Thuật ngữ

- **Gộp sâu (deep merge):** ghép file biến thể lên file chung; khoá lồng nhau được ghép từng tầng, danh sách thay hẳn.
- **Khởi đầu nóng (warm start):** bắt đầu tìm goodput ở hai mức chọn trước gần kết quả dự kiến, thay vì lưới 8/16/32/64.
- **Cổng chất lượng:** bộ đo chất lượng ngắn mà cấu hình đổi cách tính số phải qua trước khi được đo tốc độ / được nhận.
