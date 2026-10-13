# Workload, dataset & chỉ số đánh giá

> Thiết kế ngày 2026-10-12. Không đo kịch bản 1 user (không phải mục tiêu dự án).

## Bốn kịch bản

| Kịch bản | Prompt / output (token) | Tải | Vai trò / gap chính kiểm tra (xem [library-gaps.md](library-gaps.md)) |
|---|---|---|---|
| **W1 Chatbot nội bộ** | 200–800 / 200–600 | Closed-loop N user | Ca thuần, chẩn đoán: MoE decode (#1) |
| **W2 RAG tài liệu doanh nghiệp** | 2–4k / 150–400, system prompt 622 token dùng chung | Closed-loop N user | Ca thuần, chẩn đoán: MoE decode (#1), attention FP8 KV (#3), TTFT |
| **W3 Tóm tắt tài liệu dài** (hợp đồng, tóm tắt theo yêu cầu, người dùng chờ xem kết quả) | 4–8k / ~300 | Closed-loop N user | Ca thuần, chẩn đoán: prefill FP8/W4A8 (#2) |
| **W4 Tải giống thật** (**bài test cuối**) | Trộn ngắn → dài, đổi theo giờ (theo trace) | Phiên người dùng thật (BurstGPT), chồng k ngày | Tổng hợp: request dài làm chậm request ngắn, prefix cache theo phiên, đột biến |

W1/W2/W3 là các ca đặc biệt (100% một loại request), dùng để biết nghẽn đến từ đâu. W4 là tình huống tổng quát,
dùng cho kết luận cuối. Doanh nghiệp cụ thể = chỉnh tương đối hai núm của W4 (số người dùng = k; độ dài).

Concurrency W1–W3: **lưới gốc 8 / 16 / 32 / 64, tự mở rộng + tìm nhị phân điểm gãy SLO** (mục "Cách tìm goodput" bên dưới).
W4: tìm nhị phân trên k (mục "W4").

## Cách tìm goodput (lưới thích nghi)

> Chốt ngày 2026-10-12. Lý do: lưới cố định 8/16/32/64 chỉ ghi được "goodput ≥ 64" hoặc "< 8" nếu mọi mức cùng đạt / cùng trượt.
> Khi đó hai cấu hình đều "≥ 64" sẽ không so được với nhau. Đây là **hiệu ứng trần**, che mất đúng phần cải thiện mà đồ án cần chứng minh.
> Ước lượng trước (chưa đo, từ KV cache P01): W1 dễ đạt cả 64 (nhất là sau tune, ví dụ KV FP8), còn W2/W3 dễ trượt sớm.

**Kiểu tải:** closed-loop. N client, mỗi client gửi request, chờ xong rồi gửi tiếp ngay, không có think time.
Như vậy "N user" = N request luôn có mặt trong hệ thống. Tải này nặng hơn N người thật, nên kết quả thiên về bảo thủ.

**Một mức N "đạt"** khi cả TTFT p95 ≤ 10 s **và** TPOT p95 ≤ 150 ms (SLO chính, mục "SLO" bên dưới).
Nếu đo nhiều lần lặp thì xét theo **trung vị** các lần.

**p95 tính trên mọi request của lần đo (cửa sổ "all")** — chốt ở P03 (2026-10-13). Lúc bắt đầu, N request được gửi cùng lúc
(đợt đầu, prefill dồn), và cuối lần đo số request đang chạy giảm dần. Bỏ hai đoạn này (cửa sổ "steady": bỏ N request gửi đầu và N cuối)
làm TTFT p95 thấp hơn 5–21% khi gần ngưỡng (tới 71% khi tải nhẹ) nhưng chỉ còn ~2N mẫu, nhiễu gần gấp đôi (31–36% so với 19%).
"all" lệch về phía thận trọng và giống cách `vllm bench serve` báo số. Harness lưu số liệu từng request nên goodput theo "steady"
luôn được tính kèm, không cần đo thêm. Lý do và số liệu: báo cáo P03.

**Nhiễu đo (P03):** cùng điểm W2 / 16 user, hai lần đo: throughput 1.2%, TPOT p95 3.7%, **TTFT p95 ~19%** (KV đầy, preemption theo cụm).
Vì vậy ngưỡng "khác nhau > 5%" chỉ dùng cho throughput / TPOT; với TTFT/goodput, so bằng **khoảng goodput** (trung vị 3 lần lặp).
Xem [challenges.md](challenges.md) #5.

**Không đo trên lần khởi động compile từ đầu:** lần đầu trên máy mới cho KV cache ít hơn 23% (28,704 vs 37,264 token), đổi kết luận
đạt/trượt. Harness tự khởi động lại server khi compile > 10 s ([challenges.md](challenges.md) #4).

**Các bước, làm riêng cho từng workload:**

1. **Lưới gốc:** đo 8, 16, 32, 64. Mức này luôn đo đủ, để mọi cấu hình có cùng đường cong so sánh được.
2. **Mở rộng lên** nếu 64 vẫn đạt: nhân đôi 128, 256, 512… đến mức trượt đầu tiên.
   Bước này chắc chắn dừng: khi N vượt số request server chạy cùng lúc được (giới hạn bởi KV cache và `max_num_seqs`),
   phần dư phải xếp hàng, TTFT tăng tuyến tính theo N và sớm muộn vượt ngưỡng TTFT.
3. **Mở rộng xuống** nếu 8 đã trượt: 4, rồi 2. Nếu 2 vẫn trượt thì ghi goodput **"< 2"** và dừng.
   Không đo 1 user (ngoài mục tiêu dự án). Kết quả này tự nó đã là một phát hiện, ví dụ prefill prompt dài đã vượt TTFT SLO.
4. **Tìm nhị phân** giữa `lo` (mức đạt lớn nhất) và `hi` (mức trượt nhỏ nhất): đo `N = (lo + hi) / 2`, làm tròn về số chẵn,
   rồi cập nhật `lo` hoặc `hi`. Dừng khi `hi − lo ≤ max(2, 10% × lo)`. Từ một khoảng gấp đôi, thường cần thêm 3–4 điểm.
5. **Kiểm tra đơn điệu:** nếu thấy một mức đạt nằm **trên** một mức trượt (ví dụ 16 trượt nhưng 32 đạt), thì không tìm tiếp.
   Đo lặp lại hai mức đó. Nếu vẫn ngược thì ghi vào báo cáo và [challenges.md](challenges.md), vì đây là dấu hiệu nhiễu lớn
   hoặc hiện tượng lạ (ví dụ preemption).

**Báo cáo goodput dạng khoảng:** `goodput = lo`, kèm `hi`. Ví dụ "goodput W2 = 20 (22 trượt)".
Hai cấu hình chỉ được coi là khác nhau khi khoảng của chúng không chồng nhau, và chênh lệch vượt mức nhiễu đo ở P03.

**Số lần lặp:**

| Loại sweep | Lần lặp | Ghi chú |
|---|---|---|
| Đầy đủ (P04, cấu hình tune cuối GĐ1, GĐ2 thư viện sống sót, GĐ5) | ×3 cho mọi điểm | Quyết định đạt/trượt theo trung vị 3 lần |
| Rút gọn (dò tham số khi tune, sàng lọc thư viện) | ×1 | **Khởi đầu nóng:** bỏ lưới gốc, đo ngay `lo` và `hi` của cấu hình trước. Nếu `hi` cũng đạt thì nhân đôi lên; nếu `lo` cũng trượt thì chia đôi xuống; sau đó tìm nhị phân. Thường mất ~4 điểm |

**Chạy tiếp được:** điểm đo tiếp theo phụ thuộc kết quả các điểm trước. Nhưng nó luôn **tính lại được từ các `done.json`
đã có**, vì quy tắc tìm là tất định. Vì vậy chạy lại cùng lệnh sẽ tự đi tiếp đúng chỗ.
ID điểm đo vẫn tính từ cấu hình (`w2-c20-r1`), như [thi-nghiem-chay-tiep.md](thi-nghiem-chay-tiep.md).

**Giới hạn số mẫu:** mỗi lần đo cần `max(100, 4 × N)` request khác nhau (đề xuất P03). Ở N lớn có thể vượt số mẫu của workload
(W1 2000, W2 1253, W3 343). Gửi lặp lại cùng một prompt sẽ trúng prefix cache, làm kết quả đẹp giả.
Vì vậy harness phải **báo lỗi** chứ không được tự lặp mẫu. Gặp trường hợp này thì sinh thêm mẫu (P02) hoặc ghi rõ trần trong báo cáo.

## Dataset

### A. Thời điểm đến & độ dài (trace production thật, không có nội dung)

| Dataset | Nội dung | Ghi chú |
|---|---|---|
| [Azure LLM Inference Trace 2023/2024](https://github.com/Azure/AzurePublicDataset) | timestamp, ContextTokens, GeneratedTokens (Splitwise, ISCA'24) | Trace hội thoại 2024 có thể đã bị gỡ → kiểm tra repo |
| [BurstGPT](https://github.com/HPMLL/BurstGPT) | ~5.29M request, 121 ngày, Azure OpenAI, API vs conversation | Tải đột biến — kiểm SLO lúc cao điểm |
| Mooncake trace (Kimi, FAST'25) | Hash prefix, context dài | Mô phỏng prefix cache |

### B. Nội dung prompt

| Dataset | Dùng cho |
|---|---|
| **[RAGBench](https://huggingface.co/datasets/rungalileo/ragbench)** ([arXiv 2407.11005](https://arxiv.org/pdf/2407.11005)): ~100k mẫu RAG chuẩn hoá từ 12 dataset, **context có sẵn** (100–11k token). Domain doanh nghiệp: hợp đồng (CUAD), hỗ trợ khách hàng / tài liệu kỹ thuật (TechQA, EManual, DelucionQA), tài chính (FinQA, TAT-QA) | W2 — RAG; W3 — CUAD (hợp đồng dài) |
| ShareGPT, LMSYS-Chat-1M, WildChat | W1 — ShareGPT là chuẩn benchmark vLLM, so được với paper |

**Đã build (P02, 2026-10-12):** `workloads/build.py` → `data/workloads/w{1,2,3}.jsonl` (2000 / 1253 / 343 mẫu), manifest +
sha256 ở `workloads/manifest.json`, thống kê ở [measurements.md](measurements.md). Quyết định: **thinking tắt** cho cả 3 workload
(P01: bật làm output dài 10–19×); prompt dựng sẵn theo chat template Qwen3, gửi `/v1/completions`; system prompt W2 thực tế
**622 token** dùng chung; RAGBench train+val cho benchmark, **test để dành đo chất lượng**.

**Ghép:** timestamp + độ dài từ (A), nội dung từ (B) khớp độ dài.
Prompt RAG = system prompt chung + context của RAGBench + câu hỏi → không cần tự dựng hệ retrieval/embedding.

### C. Chất lượng (tối ưu không làm hỏng model)

| Cấp | Cách đo |
|---|---|
| Kernel | Unit test vs PyTorch reference (max abs/rel error) |
| Model | 1,000 prompt greedy (W1/W2/W3-like): % output giống hệt, vị trí bắt đầu lệch, Δp và KL xấp xỉ (top-20) trên đoạn chung; teacher forcing trên 200 prompt ngắn: same top-1. So với **mức tự lệch** của chính vLLM (cùng / khác mức tải) |
| Tác vụ | MMLU-Pro (tập con ngẫu nhiên 1,000 câu), IFEval (541), GSM8K (1,319), **FinQA-RAG** (1,147 câu RAGBench finqa test + đáp án gốc FinQA, dung sai 1%). Mọi điểm kèm CI 95%; so hai cấu hình bằng paired SE + flips |

Chốt 2026-10-13 (P06 bước 1), lý do + bằng chứng: [P06 plan](gd1-vllm-baseline/06-chat-luong-baseline/plan.md) mục "Quyết định đã chốt",
[references](references/README.md) nhóm E. Code: `quality/`.

⚠️ Kiểm tra license từng dataset (WildChat, LMSYS, RAGBench và các dataset thành phần) — chỉ dùng cho nghiên cứu.

## Chỉ số

- **SLO:** xem mục "SLO" bên dưới (chính: TTFT p95 ≤ 10 s, TPOT p95 ≤ 150 ms; phụ: 2 s / 100 ms)
- **Goodput:** số user đồng thời tối đa vẫn đạt SLO / 1 L4, tìm theo mục "Cách tìm goodput" và báo cáo dạng khoảng (`lo`, kèm `hi`)
- **Chi phí:** VND / 1 triệu token (đầu ra)
- **Điểm hoà vốn với API** (GĐ5): vẽ chi phí self-host theo mức tải (request/giờ) cho từng biến thể
  (vLLM mặc định → đã tune → thư viện tốt nhất GĐ2 → custom CUDA), so với giá API thương mại của
  cùng model (tính cả token vào lẫn ra, ghi nguồn + ngày tra giá). Chỉ số: mức tải tối thiểu để
  self-host rẻ bằng API, và tối ưu dời điểm đó bao nhiêu. Lý do: doanh nghiệp tải thấp thuê API rẻ hơn;
  đồ án nhắm tới bên phải/nên tự host (ràng buộc dữ liệu, nhà cung cấp inference, SaaS tải cao).
- Phụ: throughput (tok/s), TTFT/TPOT p50/p95/p99, VRAM, % roofline từng kernel

## SLO

> Chốt ngày 2026-10-12, thay cho SLO cũ (TTFT p95 < 2 s, TPOT p95 < 100 ms) vốn không có nguồn dẫn.
> Bằng chứng đầy đủ (trích dẫn, con số, hạn chế): [references/README.md](references/README.md) nhóm A, B.

**Nhận định nền:** người dùng thấy chữ đang chạy thì yên tâm tới cuối. Họ chỉ bất an khi **màn hình trống quá lâu**.
Vì vậy SLO giới hạn **thời gian tới token đầu** (TTFT) và yêu cầu **chữ chạy kịp tốc độ đọc** (TPOT). Tổng thời gian (E2EL)
không bị giới hạn.

| | SLO chính (dùng tìm goodput) | Căn cứ |
|---|---|---|
| **TTFT p95** | **≤ 10 s** | Nielsen/Miller: 10 s là giới hạn giữ chú ý (A2, A3, A4). CHI '26 (A1): TTFT 9 s không làm giảm hài lòng, tin tưởng, hay tăng bực bội; còn được đánh giá hữu ích hơn 2 s. 20 s vẫn chưa hại rõ nhưng 81.5% nhận ra trễ và một phần nghĩ hệ thống lỗi → chọn 10 s, để biên an toàn. Trong A1 mọi người chờ đúng 9 s; ở đây **p95** ≤ 10 s nghĩa là 95% người chờ ít hơn → chặt hơn thí nghiệm |
| **TPOT p95** | **≤ 150 ms** | Brysbaert 2019 (B1): đọc thầm 238 từ/phút (175–300). Người đọc nhanh 300 từ/phút ≈ 6.7 token/s ≈ 150 ms/token → chữ chạy kịp cả người đọc nhanh |
| E2EL | không giới hạn | Chữ đã chạy ổn định thì người dùng đọc theo; ai không cần ngay thì đi làm việc khác (A2: chờ > 10 s là lúc người dùng chuyển việc) |

- **Áp dụng cho W1, W2, W3, W4** như nhau. A1 là tác vụ chat nhiều lượt với GPT-4o, nên áp dụng trực tiếp cho W1/W2.
  Với W3 (tóm tắt), chưa tìm được nghiên cứu riêng; dùng cùng SLO và ghi rõ hạn chế này trong báo cáo.
- **Không chọn 4 s** (A6): nghiên cứu đó là hội thoại **giọng nói** thời gian thực, người dùng không thấy chữ chạy.
- **SLO phụ (chặt): TTFT p95 ≤ 2 s, TPOT p95 ≤ 100 ms.** Đây là SLO cũ, giữ để so sánh và để phân tích độ nhạy.
  Báo cáo thêm goodput theo SLO phụ, tính từ các điểm đã đo, **không đo thêm**. Vì tìm nhị phân chạy theo SLO chính,
  goodput theo SLO phụ chỉ có độ phân giải của các điểm đã có (thường là lưới gốc).
- **Phân tích độ nhạy (0 giờ GPU):** vẽ goodput theo ngưỡng TTFT 2 / 5 / 10 / 20 s (TPOT 150 ms) từ cùng dữ liệu.
  Hội đồng chọn ngưỡng khác thì vẫn đọc được kết quả.

## W4 — tải giống thật (bài test cuối)

> Chốt ngày 2026-10-12. Code: `workloads/build_w4.py` (sinh kịch bản), `workloads/w4_session.py` (ghép prompt lúc chạy),
> `workloads/analyze_burstgpt.py` (phân tích trace). Plan đo: [P05](gd1-vllm-baseline/05-w4-tai-giong-that/plan.md).
> Số liệu trace: [measurements.md](measurements.md) mục "Phân tích trace BurstGPT_3". Bằng chứng: [references](references/README.md) nhóm D.

**Vì sao cần:** người dùng thật không gửi liên tục một loại request. Họ mở đầu bằng câu ngắn (trace: trung vị 44 token),
hội thoại dài dần (trung vị 314 → 600 → 894 → 1188 token ở lượt 2–5), đọc rồi nghỉ (think time trung vị 124 s), có lúc dán tài liệu dài,
và **tỉ lệ request dài đổi theo từng giờ** (request ≥ 2k token chiếm 5–25% tuỳ giờ). Không có bộ tỉ lệ ngắn/trung bình/dài cố định nào
đại diện được → không trộn W1/W2/W3 theo tỉ lệ, mà **phát lại hành vi người dùng thật**.

**Nguồn:** phiên hội thoại (`Conversation log`, có `Session ID`) của BurstGPT_3 (110 ngày, 55,920 phiên).
Bỏ API log (95.6% request, chương trình gọi tự động, không có phiên). Nhịp tải của phần hội thoại giống giờ hành chính
(cao điểm 9–11h và 14–17h, giảm lúc 12h, gần như không có ai 3–7h).

**Mô hình tải: partly-open** (references D1). Người dùng **đến** theo trace (không phụ thuộc server). Trong phiên, lượt sau gửi khi lượt trước
**nhận xong** + think time của trace (người chờ trả lời rồi mới hỏi tiếp).

**Nội dung:** trace chỉ có số token, không có chữ. Tin nhắn mới < 1500 token lấy từ ShareGPT (tin nhắn người dùng thật), ≥ 1500 token
ghép tài liệu RAGBench (train+val), độ dài khớp trace. Prompt lượt sau = lịch sử thật (câu trả lời server vừa sinh) + tin nhắn mới,
nên prefix cache hoạt động như chat thật. Lịch sử vượt độ dài trace → bỏ cặp hỏi–đáp cũ nhất (client trong trace cũng cắt lịch sử).
Mỗi tin nhắn / tài liệu mở đầu chỉ dùng một lần → không trúng cache "ăn may" giữa các phiên.

**Khung giờ (chọn theo quy tắc, không chọn tay):** theo tải trung bình mỗi giờ trong ngày của trace:
**peak = 16h** (208 req/giờ, giờ đông nhất), **peak2 = 10h** (180, cao điểm buổi sáng, kiểm độ ổn định),
**normal = 13h** (92, gần trung vị), quiet = 5h (4, không đo, tính bằng công thức).
Mỗi lần chạy: **10 phút khởi động + 30 phút đo**. Phiên đã bắt đầu trước cửa sổ (người đang chat dở) vẫn được đưa vào;
lượt đầu trong cửa sổ mang đủ độ dài lịch sử theo trace, như người quay lại hội thoại cũ.

**Tăng tải = chồng phiên, không tua nhanh:** lấy cùng khung giờ của **k ngày khác nhau** phát chồng lên nhau → k lần số người dùng,
giữ nguyên think time và cấu trúc phiên. Tua nhanh sẽ co think time lại, sai hành vi thật (references D4: phải giữ phân bố thời gian giữa các lượt).
k ≤ 110 (số ngày của trace). Ngày lấy theo hoán vị cố định theo seed → kịch bản k nằm trong kịch bản k' > k, tải tăng đơn điệu.
Kiểm tra (2026-10-12): req/giờ đo trong kịch bản sát k × trung bình trace (peak k=40: 8,130 vs 8,333; k=110: 22,130 vs 22,917).
Một Session ID có thể xuất hiện ở nhiều ngày (người dùng quay lại hội thoại cũ vài ngày sau): mỗi (phiên, ngày) là một phiên riêng,
lần sau là phiên "quay lại" mang đủ lịch sử (sửa 2026-10-13; sau sửa: peak k=40 8,278 vs 8,334, k=110 22,800 vs 22,917).
Khi đo (P05), kịch bản k ngày = kịch bản gốc k=110 lọc theo k ngày đầu của hoán vị (giống hệt sinh riêng về phiên, thời điểm, độ dài).

**Goodput W4** = k lớn nhất ở peak mà mọi request trong 30 phút đo đạt SLO chính (TTFT p95 ≤ 10 s, TPOT p95 ≤ 150 ms);
tìm nhị phân trên k như mục "Cách tìm goodput". Báo cáo quy ra **req/giờ** và **số phiên đang mở** (người dùng hoạt động cùng lúc).
**Chi phí cả ngày** = ghép kết quả peak / normal / quiet theo số giờ mỗi loại trong ngày thật của trace (không chỉ giờ cao điểm).

**Scale cho doanh nghiệp (tương đối):** số người dùng → k; dùng nhiều RAG/tóm tắt hơn → dịch phân bố độ dài về phía dài (làm ở GĐ5 nếu cần).

**Hạn chế (ghi vào báo cáo):**
- Trace từ dịch vụ ChatGPT/GPT-4 trên Azure (tổ chức không công bố), không phải chatbot nội bộ của một doanh nghiệp cụ thể.
- Token trong trace đếm bằng tokenizer của OpenAI; dùng thẳng làm số token Qwen3 (văn bản tiếng Anh, chênh lệch nhỏ).
- Nội dung các lượt trong một phiên ghép từ nguồn khác nhau → không liền mạch về nghĩa; chỉ độ dài và prefix là thật.
  Với tốc độ GPU, đây là hai yếu tố quyết định.
- Lượt có độ dài giảm (client gốc cắt lịch sử) không suy ra được độ dài tin nhắn mới → lấy ngẫu nhiên từ phân bố tin nhắn mới của trace.

## W3b — batch ban đêm: đã bỏ (2026-10-12)

Tóm tắt hàng loạt ban đêm hiếm gặp. Phần lớn người dùng tóm tắt tài liệu rải rác trong ngày, và trường hợp đó đã nằm trong W3 và W4.
Thiết kế đã thảo luận (makespan, n_max(T) theo cửa sổ 8 h / 24 h, quét `--max-num-seqs`) còn trong lịch sử Git
nếu sau này cần (references nhóm C).
