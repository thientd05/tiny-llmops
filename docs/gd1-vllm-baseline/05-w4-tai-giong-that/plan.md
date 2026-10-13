# P05 — W4: tải giống thật (phiên người dùng BurstGPT)

> Cần GPU: có · Ước tính ~6.5 giờ GPU (~67k VND) → **ước lại sau P04: ~5–5.6 h (~38–43k VND)** · Phụ thuộc: P03 (client phiên), P04 (năng lực để chọn k khởi đầu)
>
> **Từ P04 (2026-10-13):** trần ước tính k ≈ 22 (phép tính từ throughput bão hoà + KV, xem báo cáo P04 mục 6) → khởi đầu nóng k = 16 / 24.
> Trước khi thuê máy: thêm chế độ W4 vào `bench/sweep.py` (đơn vị = khung giờ × k × seed, `done.json`, `--status`, dùng `search_step`).
> **Xong 2026-10-13.** Phần local: mục "Chuẩn bị local" cuối file. Đo trên L4 14:28–19:26 (4.97 h, ≈ 49,200 VND): mục "Kết quả chạy".
> Báo cáo: [đọc ngay](doc-ngay.md) · [đầy đủ](bao-cao.md).
>
> 2026-10-12: thay thiết kế cũ "phát lại 30 phút BurstGPT với nội dung W2". Lý do và thiết kế đầy đủ:
> [workloads.md — W4](../../workloads.md#w4--tải-giống-thật-bài-test-cuối).

## Mục tiêu

P04 đo các ca "thuần": N người gửi liên tục, không nghỉ, cùng một loại request. Người dùng thật khác hẳn:
họ mở đầu bằng câu ngắn, hội thoại dài dần, đọc rồi nghỉ mới hỏi tiếp, có lúc dán tài liệu dài, lúc đông lúc vắng,
và tỉ lệ request ngắn/dài đổi theo từng giờ. P05 đo **vLLM mặc định phục vụ được bao nhiêu người dùng thật** ở giờ cao điểm.

## Thiết kế (đã có code: `workloads/build_w4.py`, `workloads/w4_session.py`)

- **Nguồn hành vi:** phiên hội thoại (`Conversation log`) của BurstGPT_3: 55,920 phiên thật, 110 ngày. Bỏ API log (chương trình gọi).
- **Khung giờ chọn theo quy tắc** (giờ trong ngày, tải trung bình của trace): **peak = 16h** (208 req/giờ), **peak2 = 10h** (180),
  **normal = 13h** (92). Giờ vắng (5h, 4 req/giờ) không đo, tính bằng công thức.
- **Mỗi lần chạy = 10 phút khởi động + 30 phút đo** (chỉ tính request gửi trong 30 phút đo).
- **Tải = k:** chồng phiên của cùng khung giờ từ k ngày khác nhau (k ngày ≈ k lần số người dùng), giữ nguyên think time.
  k ≤ 110. Ngày lấy theo hoán vị cố định theo seed → kịch bản k nằm trong k' > k, tải tăng đơn điệu theo k.
- **Client phiên (partly-open):** lượt đầu gửi đúng `start_s`; lượt sau gửi khi lượt trước **nhận xong** + `think_s`.
  Prompt = lịch sử thật (gồm câu trả lời server vừa sinh) + tin nhắn mới, qua `w4_session.assemble`; vượt `prompt_budget`
  thì bỏ cặp hỏi–đáp cũ nhất. Stream để đo TTFT/TPOT; `max_tokens` + `ignore_eos` theo trace.
- **Đạt SLO:** TTFT p95 ≤ 10 s và TPOT p95 ≤ 150 ms trên mọi request trong 30 phút đo (SLO chính, như W1–W3).
- **Chạy tiếp được:** mỗi lần chạy (một k, một khung giờ, một seed) là một đơn vị có `done.json`. Không cắt nhỏ hơn,
  vì phiên và hàng đợi phải liên tục. Dừng giữa chừng → chạy lại cùng lệnh, mất tối đa ~40 phút.

## Các bước

1. Ở local: sinh kịch bản cần dùng (`build_w4.py scenario --segment peak --k <k>`), ghi sha256 vào kết quả. Kho nội dung
   (`build_w4.py pool`) đã có (2026-10-12).
2. Trên L4, cùng server mặc định như P04:
   1. **Tìm goodput ở peak:** k khởi đầu ước từ goodput P04 (W1/W2), sau đó nhân đôi / chia đôi rồi tìm nhị phân tới
      `hi − lo ≤ max(2, 10% × lo)` (cùng quy tắc [workloads.md](../../workloads.md#cách-tìm-goodput-lưới-thích-nghi)). Dự kiến ~6 lần chạy.
   2. **Kiểm độ ổn định:** chạy lại k = `lo` với seed 2 (bộ ngày khác) và ở peak2. Goodput khác nhau > 1 bước tìm → ghi vào báo cáo.
   3. **Giờ bình thường:** chạy normal ở k = `lo` → số liệu cho chi phí cả ngày.
3. Rsync, destroy (nếu không làm tiếp P06).
4. Phân tích ở local: TTFT/TPOT p95 theo từng phút; tách theo loại request (ngắn < 1k / dài ≥ 2k, lượt đầu / lượt sau / resumed);
   % prefix cache hit; chi phí VND / 1 triệu token cả ngày = ghép kết quả peak / normal / vắng theo số giờ mỗi loại trong ngày thật của trace.

## Kết quả cần có

| Bảng / hình | Nội dung |
|---|---|
| Goodput W4 | k `lo`/`hi` ở peak; quy ra **req/giờ**, **số phiên đang mở** (người dùng hoạt động cùng lúc) |
| Hình 1 | TTFT p95 và TPOT p95 theo k, kẻ đường SLO |
| Hình 2 | Ở k = `lo`: theo thời gian, số request mỗi phút (trên), TTFT/TPOT p95 mỗi phút (dưới) |
| Bảng | TTFT/TPOT p95 tách theo loại request — request ngắn có bị request dài làm chậm không |
| Bảng | Ổn định: goodput seed 1 vs seed 2 vs peak2 |
| Chi phí | VND / 1 triệu token theo cả ngày (không chỉ giờ cao điểm) |

## Tiêu chí xong

- [x] Client phiên đã kiểm chứng: chạy W2 / 16 user closed-loop bằng client này, lệch < 5% so với `vllm bench serve` (P03: gộp 200 request, lệch ≤ 3.7%).
- [x] Goodput W4 ở peak dạng khoảng hẹp; đã chạy seed 2, peak2, normal; số tóm tắt ghi measurements.md. (2026-10-13: **18 (20 trượt)**; seed 2: 16 (18 trượt))

## Báo cáo P05 cần trả lời

1. vLLM mặc định phục vụ được bao nhiêu người dùng thật lúc cao điểm (req/giờ, phiên đang mở)?
2. Request ngắn có bị request dài (tài liệu, hội thoại dài) làm chậm không, chậm bao nhiêu?
3. Prefix cache giúp được bao nhiêu khi người dùng quay lại hội thoại (trúng cache theo think time)?
4. So với goodput closed-loop của P04, con số "người dùng" khác bao nhiêu lần, vì sao (think time, độ dài thật)?
5. Chi phí cả ngày so với chỉ tính giờ cao điểm.

## Chuẩn bị local (xong 2026-10-13, 0 giờ GPU)

### Đã làm

| Việc | File |
|---|---|
| Chế độ `w4` trong harness: đơn vị `w4-<khung>-s<seed>-k<k>-r<lần>` có `done.json` (hash gồm sha256 kịch bản + danh sách ngày), tìm k bằng `search_step` của P04, xong tự chạy các kịch bản kiểm tra ở k = lo; `--status` in cả tiến độ của lần chạy đang dở (`progress.json`, ghi mỗi 30 s) | `bench/sweep.py` |
| Client phiên: lọc theo ngày, dừng sớm, chờ request dở có giới hạn, ghi gửi trễ + token trúng cache theo từng request | `bench/session_client.py` |
| Tổng hợp W4: goodput k (SLO chính, phụ, độ nhạy TTFT), từng lần chạy, tách theo loại request, request ngắn bị request dài chen, prefix cache theo think time | `bench/summarize.py` |
| Hình 1, Hình 2 + chi phí cả ngày (cần trace, chạy ở local) | `bench/w4_report.py` |
| Kịch bản gốc k=110: peak2-s1, normal-s1, peak-s2 (peak-s1 có từ 2026-10-12); sha256 ghi vào `workloads/manifest_w4.json` | `data/workloads/w4/` |
| Cấu hình đo thật / thử với server giả | `bench/configs/w4.yaml`, `bench/configs/test-fake-w4.yaml` |
| Server giả có prefix cache giả (khối 64 ký tự, LRU) để thử phần phân tích cache | `bench/fake_server.py` |
| Test đầu-cuối có ngắt giữa chừng (SIGKILL) | `bash bench/test_fake.sh <dir> w4` |

### Quyết định phương pháp (chốt trước khi đo, không đổi sau khi thấy kết quả)

1. **Kịch bản k ngày = kịch bản gốc k=110 lọc theo k ngày đầu của hoán vị**, thay vì sinh riêng từng k bằng `build_w4.py scenario --k`.
   - Lý do: tìm nhị phân không biết trước sẽ cần k nào. Sinh riêng thì phải mang trace + kho nội dung (~400 MB) lên máy thuê,
     hoặc sinh sẵn mọi k (~1.2 GB). Kịch bản gốc chỉ 8–21 MB mỗi khung giờ.
   - Hệ quả tốt: kịch bản k nằm trọn trong k' > k **kể cả nội dung tin nhắn** (sinh riêng từng k thì cùng phiên có thể nhận
     tin nhắn khác nhau). Số phiên, số lượt, thời điểm, độ dài prompt/output theo trace không đổi so với sinh riêng.
   - Đã kiểm: lọc = sinh riêng ở k = 1/10/40 và 3 khung giờ (cùng phiên, thời điểm, think time, độ dài trace, max_tokens).
     Chỉ khác nội dung tin nhắn và 1–2% lượt có độ dài lấy ngẫu nhiên (client gốc đã cắt lịch sử).
   - Bước kiểm này phát hiện lỗi phiên nhiều ngày của `build_w4.py` (challenges.md #6), đã sửa và sinh lại mọi kịch bản.
2. **Mỗi mức k chạy 1 lần** (40 phút). Không lặp 3 lần như P04 vì quá tốn. Độ ổn định kiểm bằng seed 2 (bộ ngày khác) ở k = lo.
   Nếu kết quả không đơn điệu (k nhỏ trượt, k lớn đạt), `search_step` đòi chạy lại 2 lần mỗi mức đó; đạt/trượt theo đa số, hoà → trượt.
3. **Dừng sớm khi chắc chắn trượt.**
   - Điều kiện: số request trong cửa sổ đo **chắc chắn** trượt TTFT (hoặc TPOT) ≥ 5% × (số lượt tối đa của kịch bản) + 2.
   - "Chắc chắn trượt TTFT" = đã xong với TTFT > 10 s, hoặc chưa có token đầu mà đã chờ > 10 s.
     "Chắc chắn trượt TPOT" = đã xong với TPOT > 150 ms, hoặc đang sinh mà đã quá 150 ms × (max_tokens − 1) sau token đầu
     (ignore_eos nên số token ra cố định).
   - Số request trong cửa sổ không thể vượt số lượt của kịch bản, nên khi điều kiện đúng thì p95 cả lần chạy chắc chắn > SLO.
     Kết quả đạt/trượt giống hệt chạy đủ 30 phút. Mức quá tải tốn ~10–25 phút thay vì ~50.
   - Hạn chế: TTFT/TPOT p95 của lần dừng sớm là của phần đã chạy. Bảng độ nhạy chỉ dùng lần dừng sớm cho SLO **chặt hơn**
     (với SLO lỏng hơn, ví dụ TTFT 20 s, không biết kết quả → bỏ mức đó).
4. **Chờ request dở sau cửa sổ tối đa 10 s + 4096 × 150 ms + 30 s = 654 s.** Request đạt SLO chắc chắn xong trước mốc này
   (output tối đa 4096 token). Request bị cắt chắc chắn trượt, ghi là lỗi (∞). Không có giới hạn này thì một lần chạy quá tải
   có thể kéo dài hàng giờ để chờ hàng đợi rút hết.
5. **Thêm cờ `--enable-prompt-tokens-details`** vào server. Cờ này chỉ gắn thêm `prompt_tokens_details.cached_tokens` vào
   `usage` của câu trả lời (`vllm/entrypoints/openai/completion/serving.py` dòng 451 và 607, bản 0.30.0). Số token trúng cache
   vẫn được tính dù có cờ hay không, nên không đổi tốc độ. Nhờ cờ này, câu hỏi 3 (cache theo think time) được trả lời theo
   từng request, thay vì chỉ có tỉ lệ trúng chung của server.
6. **Điểm mốc W2 / 16 user × 3 lần** chạy đầu phiên (~6 phút), so với P04 (TTFT p95 4.9–6.3 s, TPOT p95 89.6–92.7 ms,
   167–171 tok/s ra) để biết máy mới có nhanh/chậm hơn máy P04 không (thi-nghiem-chay-tiep.md nguyên tắc 9).
7. **Chi phí cả ngày tính theo từng giờ của trace**: VND / 1 triệu token ra = 24 × giá/giờ ÷ (k × Σ 24 giờ token ra trung bình
   của trace). Đây là bản chính xác của "ghép peak / normal / quiet theo số giờ mỗi loại": thay 3 loại giờ bằng đủ 24 giờ.
   Các lần chạy peak / normal ở k = lo dùng để kiểm: server có thật sự phục vụ đạt SLO ở các mức tải đó, và lượng token
   đo được có khớp trace × k không.
8. Quyết định đạt/trượt chỉ trên request **gửi trong 30 phút đo**. Request của 10 phút khởi động không tính.

### Kiểm tra ở local (server giả, thời gian co 100 lần)

Số liệu ở [measurements.md](../../measurements.md) mục "2026-10-13 — P05 chuẩn bị local". Tóm tắt:
- Tìm k hội tụ đúng: 16 đạt, 24 trượt (dừng sớm), 20 đạt, 22 trượt (dừng sớm) → 20 (22 trượt); sau đó tự chạy peak-s2 và normal ở k = 20.
- Ngắt bằng SIGKILL giữa lần chạy k=22, chạy lại cùng lệnh: thư mục dở bị xoá và chạy lại, 4 đơn vị đã xong không bị chạy lại.
- `--scenario-check`: 0 lượt gửi sớm hơn "nhận xong + think time"; lượt đầu đúng `start_s`.
- Tổng hợp P04 chạy lại bằng `common.py` mới ra `summary.md` giống hệt (trừ dòng đường dẫn).
- Tải CPU client: co thời gian 10 lần (12 request/giây, ~10 lần tải thật ở k=24) → gửi trễ p99 0.08 s, max 0.31 s → client không phải nghẽn.

### Chạy trên L4 (khi thuê máy)

```bash
# local: đẩy code + workload + 4 kịch bản gốc (~70 MB) lên
rsync -az --relative bench/*.py bench/configs scripts/*.sh workloads/manifest.json workloads/manifest_w4.json \
      workloads/w4_session.py data/workloads/w2.jsonl data/workloads/w4/*-k110-s*.jsonl \
      data/workloads/w4/*-k110-s*.summary.json root@<IP>:/root/td/
ssh ... "cd /root/td && tmux new -d -s exp 'BENCH_MACHINE=\"host=... machine=... instance=...\" \
      bash scripts/remote_bench.sh bench/configs/w4.yaml /root/results/p05'"
# local: kéo kết quả mỗi 2 phút; xem tiến độ (cả lần chạy đang dở)
.venv/bin/python bench/sweep.py --config bench/configs/w4.yaml --out results/gd1/05-w4-tai-giong-that/<run> --status
# sau khi destroy: tổng hợp + hình + chi phí
.venv/bin/python bench/summarize.py --config bench/configs/w4.yaml --out <run>
.venv/bin/python bench/w4_report.py --config bench/configs/w4.yaml --out <run> \
      --fig-dir docs/gd1-vllm-baseline/05-w4-tai-giong-that/hinh --price-vnd-h <giá thật>
```

### Ước giờ GPU (cập nhật)

| Phần | Thời gian |
|---|---|
| Cài đặt + tải model + khởi động server (máy mới) | ~0.3 h |
| Điểm mốc W2/16 × 3 | ~0.1 h |
| Tìm k: ~2–3 lần đạt × ~42 phút + ~1–2 lần trượt (dừng sớm) × ~15–25 phút | ~1.9–2.9 h |
| Kiểm tra: peak-s2, peak2, normal × ~42 phút (nếu trượt thì dừng sớm, ngắn hơn) | ~2.1 h |
| **Tổng** | **~4.4–5.4 h ≈ 33–41k VND** (đơn giá thực P04 ~7,570 VND/h) |

## Kết quả chạy (2026-10-13, L4 host 506934 / machine 150535)

- Tìm k (peak seed 1): 16 ✓ → 24 ✗ (dừng sớm) → 20 ✗ (dừng sớm) → 18 ✓ → **goodput 18 (20 trượt)**.
- Kiểm tra ở k = 18: peak seed 2 ✗ (dừng sớm), peak2 10h ✓, normal 13h ✓.
- **Khác plan: thêm 1 lần chạy peak seed 2 ở k = 16** (✓).
  - Lý do: seed 2 trượt ở 18. Chỉ một điểm thì không biết bộ ngày thứ hai lệch 1 bước hay nhiều hơn.
  - Giới hạn đặt trước khi chạy: chỉ 1 lần, trượt thì ghi "< 16".
  - Cấu hình: `w4.extra` trong `bench/configs/w4.yaml`.
  - Goodput chính vẫn theo seed 1 như plan. Seed 2: 16 (18 trượt) → lệch đúng 1 bước tìm, ghi vào báo cáo theo quy tắc của plan.
- Giờ GPU thực tế 4.97 h (ước 4.4–5.4 h); chi phí ≈ 49,200 VND (ước 33–41k): giá máy $0.372/h cao hơn đơn giá P04 dùng để ước, cộng 1 lần chạy thêm.
- Số liệu: [measurements.md](../../measurements.md) mục "2026-10-13 — P05". Phân tích: [bao-cao.md](bao-cao.md).
