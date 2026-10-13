# Tài liệu tham khảo (bằng chứng cho đồ án)

> Mọi paper, báo cáo, trang tài liệu đã tra để ra quyết định thiết kế. Mỗi mục ghi: trích dẫn, link, ngày tra,
> **con số / câu dùng làm bằng chứng**, mức kiểm chứng, và quyết định nào trong dự án dựa vào nó.
>
> Bản lưu (PDF/HTML) nằm ở `files/` **chỉ ở local** (gitignore), vì repo public trên GitHub và các file này có bản quyền
> của bên thứ ba. Nên sao lưu thêm `files/` ra chỗ khác (Drive). Mất file thì tải lại theo link bên dưới.

**Mức kiểm chứng:**
- ✅ đã đọc và đối chiếu con số trong bản gốc (PDF/HTML ở `files/`);
- ⚠️ chỉ đọc qua tóm tắt hoặc nguồn thứ cấp, chưa đối chiếu bản gốc → không dùng làm luận cứ chính.

Thêm tài liệu mới: thêm một mục vào đúng nhóm, lưu bản sao vào `files/` (nếu tải được), và ghi rõ quyết định nào dùng nó.

---

## A. Người dùng chịu chờ bao lâu (dùng cho SLO TTFT / TPOT)

### A1. Tan, Messerschmidt, Yin & Nov (2026). *The Impact of Response Latency and Task Type on Human-LLM Interaction and Perception.* CHI '26, Barcelona. ✅

- Link: <https://arxiv.org/abs/2604.06183> · Bản lưu: `files/tan2026-chi-latency-task-type.pdf` · Tra: 2026-10-12
- **Thiết kế:** 240 người (Prolific, Mỹ), giao diện chat GPT-4o, TTFT cố định **2 / 9 / 20 s**, sau token đầu stream đều
  **25 tok/s (40 ms/token)**. Tác vụ: Creation (slogan, brainstorm, viết điền đoạn) và Advice (gợi ý, memo đánh đổi, sửa email).
- **Kết quả dùng làm bằng chứng:**
  - TTFT **không ảnh hưởng có ý nghĩa** tới hài lòng (expectation gap), tin tưởng, bực bội / tải nhận thức (NASA-TLX),
    và hành vi (prompt mới, sửa, tạo lại, new chat).
  - "Thoughtfulness": F(2,234) = 5.12, p = .007. 2 s bị đánh giá kém chín chắn hơn 9 s và 20 s.
    "Usefulness": F(2,234) = 3.61, p = .028; 9 s hơn 2 s.
  - Tỉ lệ **nhận ra có trễ**: ≈ 26.6% (2 s) → 66.3% (9 s) → 81.5% (20 s).
  - Định tính: ở trễ dài một phần người dùng nghĩ hệ thống lỗi; số người nói bực/bất an tăng theo trễ.
- **Hạn chế (tác giả nêu):** chỉ 3 mức TTFT, không thử > 20 s; không có hậu quả thật; mẫu Mỹ; giao diện không có chỉ báo tiến độ;
  **không có tác vụ tóm tắt**.
- **Dùng cho:** SLO TTFT p95 ≤ 10 s cho W1/W2/W3a ([workloads.md](../workloads.md) mục "SLO").

### A2. Nielsen, J. (1993). *Usability Engineering*, chương 5 — "Response Times: The 3 Important Limits" (NN/g). ✅

- Link: <https://www.nngroup.com/articles/response-times-3-important-limits/> · Bản lưu: `files/nngroup-response-times-3-limits.html` · Tra: 2026-10-12
- Gốc từ Miller (1968) và Card, Robertson & Mackinlay (1991), xem A3.
- **Câu trích:** "10 seconds is about the limit for keeping the user's attention focused on the dialogue."
  "Delays of longer than 10 seconds are only acceptable during natural breaks in the user's work, for example when switching tasks."
- **Dùng cho:** mốc 10 s của SLO TTFT; lập luận người dùng chờ lâu hơn thì chuyển việc khác (nhóm "gửi rồi đi làm việc khác").

### A3. Miller, R. B. (1968). *Response time in man-computer conversational transactions.* AFIPS Fall Joint Computer Conference, 267–277. · Card, S. K., Robertson, G. G., & Mackinlay, J. D. (1991). *The information visualizer.* CHI '91, 181–188. ⚠️

- Chưa tải (sau tường phí). Trích qua A2. Dùng khi cần dẫn nguồn gốc của các mốc 0.1 / 1 / 10 s.

### A4. Nielsen, J. (2024). *Time Scales of UX: From 0.1 Seconds to 100 Years.* UX Tigers / Substack. ✅

- Link: <https://jakobnielsenphd.substack.com/p/time-scale-ux> · Bản lưu: `files/nielsen2024-time-scales-ux.html` · Tra: 2026-10-12
- **Ý dùng:** các mốc thời gian phản hồi do nhận thức con người quyết định nên không đổi theo thời gian.
  Trả lời câu hỏi "số từ 1968/1993 còn đúng không".

### A5. Nah, F. F.-H. (2004). *A study on tolerable waiting time: how long are Web users willing to wait?* Behaviour & Information Technology 23(3), 153–163. ⚠️

- Link: <https://ink.library.smu.edu.sg/sis_research/9983> (bị chặn tải tự động; bản gốc trên Taylor & Francis, có phí) · Tra: 2026-10-12
- **Ý dùng (từ abstract):** có **phản hồi** (feedback) thì người dùng chịu chờ lâu hơn. Stream token là một dạng phản hồi.
- Các con số chi tiết hay được trích (median 22.6 s có thanh tiến độ vs 9 s không có; ~38 s) **chỉ thấy qua nguồn thứ cấp**,
  chưa đối chiếu, không dùng làm luận cứ chính.

### A6. Maslych et al. (2025). *Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents.* arXiv 2507.22352. ✅

- Link: <https://arxiv.org/abs/2507.22352> · Bản lưu: `files/arxiv2507.22352-llm-iva-response-delays.pdf` · Tra: 2026-10-12
- **Thiết kế:** hội thoại **bằng giọng nói** với agent ảo (VR), trễ 1.5 / 4.0 / 6.5 s.
- **Kết quả:** "latency above 4 seconds degrades quality of experience"; trích nghiên cứu hội thoại người–người: trên 2 s thấy không tự nhiên, im lặng > 4 s làm đứt mạch.
- **Vì sao KHÔNG dùng cho SLO dự án:** đây là hội thoại giọng nói thời gian thực, người dùng không thấy chữ chạy.
  Chatbot văn bản (W1/W2) gần với A1 hơn. Giữ lại để trả lời nếu hội đồng hỏi "sao không chọn 4 s".

### A7. Gnewuch, Morana, Adam & Maedche (2018). *"Faster Is Not Always Better": Understanding the Effect of Dynamic Response Delays in Human-Chatbot Interaction.* ECIS 2018. · Gnewuch et al. (2022). *Opposing Effects of Response Time in Human–Chatbot Interaction.* BISE 64(6). ⚠️

- Link: <https://aisel.aisnet.org/ecis2018_rp/113/> · <https://aisel.aisnet.org/bise/vol64/iss6/5>
  · Bản lưu (trang abstract): `files/gnewuch2018-ecis-faster-not-better.html`, `files/gnewuch2022-bise-opposing-effects.html` · Tra: 2026-10-12
- **Ý dùng:** trả lời tức thì không phải lúc nào cũng tốt hơn; trễ vừa phải (trung bình 2.3 s, tới ~4 s) làm tăng cảm giác "người thật"
  và hài lòng (2018); trễ có tác động trái chiều tuỳ kinh nghiệm người dùng (2022, N = 202). Củng cố ý: không cần TTFT cực thấp.
- Chatbot kịch bản (không phải LLM stream) → chỉ dùng làm bằng chứng phụ.

## B. Tốc độ đọc (dùng cho SLO TPOT)

### B1. Brysbaert, M. (2019). *How many words do we read per minute? A review and meta-analysis of reading rate.* Journal of Memory and Language 109, 104047. ✅

- Link: <https://doi.org/10.1016/j.jml.2019.104047> · Bản lưu: `files/brysbaert2019-reading-rate.pdf` · Tra: 2026-10-12
- **Kết quả:** 190 nghiên cứu, 18,573 người: đọc thầm tiếng Anh trung bình **238 từ/phút** (phi hư cấu), 260 (hư cấu);
  đa số người lớn **175–300 từ/phút** (phi hư cấu).
- **Tính ra (dự án tự suy):** 238 từ/phút ≈ 4.0 từ/s; với ~0.75 từ/token → **≈ 5.3 token/s ≈ 190 ms/token**.
  Người đọc nhanh (300 từ/phút) ≈ 6.7 token/s ≈ 150 ms/token.
- **Dùng cho:** SLO TPOT p95 ≤ 150 ms: text chạy kịp cả người đọc nhanh (300 từ/phút).

## C. Batch / khối lượng tài liệu (cho W3b — đang tạm hoãn)

### C1. OpenAI Batch API — `completion_window`. ✅

- Link: <https://developers.openai.com/api/reference/cli/resources/batches/methods/create/index.md> · Bản lưu: `files/openai-batch-api-create.md` · Tra: 2026-10-12
- **Câu trích:** "The time frame within which the batch should be processed. Currently only `24h` is supported."
- **Ý dùng:** chuẩn ngành cho batch LLM là xong trong **24 h** → một mốc cửa sổ T khi tính n_max(T).

### C2. UW–Milwaukee HR — EPM Nightly Batch Processing Schedule. ✅

- Link: <https://uwm.edu/hr/?p=13971> · Bản lưu: `files/uwm-epm-nightly-batch-window.html` · Tra: 2026-10-12
- **Ý dùng:** ví dụ một cửa sổ batch ban đêm thật: **20:00 → 07:00 (~11 h)**. Chỉ là một tổ chức, không phải chuẩn chung.

### C3. NCMA / WorldCC (2025). *CCM Institute Report — Contract Management Capacity Metrics* (3 Apr 2025). ⚠️

- Link: <https://ncmahq.org/common/Uploaded%20files/About/AEWG/CCM%20Institute%20Report%20-%20CM%20capacity%20metrics%203%20Apr%202025.pdf>
  (bị chặn tải tự động) · Tra: 2026-10-12
- **Ý dùng (qua tóm tắt tìm kiếm):** mỗi người quản lý hợp đồng xử lý khoảng **3–25 hợp đồng** cùng lúc tuỳ ngành.
  **Không** có số hợp đồng mỗi công ty (số đó nằm trong WorldCC Benchmark Report, cần trả phí).
  Kết luận: không có dữ liệu công khai về "một công ty gửi bao nhiêu tài liệu" → đổi sang đo năng lực n_max(T).

## D. Mô phỏng tải giống thật: phiên người dùng, hội thoại nhiều lượt, đột biến (cho W4)

### D1. Schroeder, Wierman & Harchol-Balter (2006). *Open Versus Closed: A Cautionary Tale.* NSDI '06, 239–252. ✅

- Link: <https://www.usenix.org/conference/nsdi-06/open-versus-closed-cautionary-tale> · Bản lưu: `files/schroeder2006-nsdi-open-vs-closed.pdf` · Tra: 2026-10-12
- **Mô hình partly-open:** người dùng **đến** theo quá trình mở (không phụ thuộc hệ thống); mỗi lần ghé, sau mỗi request
  "with probability p the user stays and makes a followup request (possibly after some think time), and with probability 1 − p
  the user simply leaves" → số request mỗi lần ghé phân phối hình học.
- "real-world applications often fit best into partly-open models"; "counter to intuition, parameters like think time have almost
  no impact on the performance of a partly-open model" (trong điều kiện của bài: web server, không có cache theo phiên).
- **Dùng cho:** khung lý thuyết của W4 (đúng mô tả "người dùng đến, chat nhiều lượt có nghỉ, rồi rời đi").
  Ý "think time gần như không ảnh hưởng" **không tự áp dụng** cho LLM: think time quyết định KV prefix cache của phiên còn hay bị đẩy ra.

### D2. Zhao et al. (2024). *WildChat: 1M ChatGPT Interaction Logs in the Wild.* ICLR 2024. + dataset card `allenai/WildChat-1M`. ✅

- Link: <https://arxiv.org/abs/2405.01470> · <https://huggingface.co/datasets/allenai/WildChat-1M>
  · Bản lưu: `files/wildchat2024-iclr.pdf`, `files/wildchat-1m-dataset-card.md` · Tra: 2026-10-12 · License: **ODC-BY**
- **Có:** nội dung từng lượt + **timestamp từng tin nhắn** (user và assistant; timestamp assistant = lúc server nhận đủ câu trả lời)
  + `turn_identifier` + `hashed_ip` + header trình duyệt. Paper: hashed IP + header "could potentially enable researchers to link
  conversations from the same user" (tác giả không cung cấp liên kết trực tiếp; IP có thể đổi giữa chừng → ghép theo người chỉ gần đúng).
- **Số liệu (paper):** trung bình **2.52 lượt**/hội thoại; ~41% hội thoại nhiều lượt; 3.7% > 10 lượt; 204,736 IP khác nhau.
- **Dùng cho:** đo **thời gian nghỉ giữa các lượt** (timestamp user lượt sau − timestamp assistant lượt trước = đọc + nghỉ),
  số lượt mỗi hội thoại, tốc độ dài ra của ngữ cảnh, số hội thoại mỗi người và khoảng nghỉ giữa các hội thoại.
- **Hạn chế:** chatbot công cộng miễn phí (HF Space), không phải công cụ nội bộ doanh nghiệp.

### D3. Wang et al. (2025). *BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems.* KDD '25. + README repo. ✅

- Link: <https://arxiv.org/abs/2401.17644> · <https://github.com/HPMLL/BurstGPT>
  · Bản lưu: `files/burstgpt2025-kdd.pdf`, `files/burstgpt-readme.md` · Tra: 2026-10-12 · License: **CC-BY-4.0**
- **Có:** `Timestamp`, `Session ID`, `Elapsed time`, `Model`, `Request tokens`, `Response tokens`, `Total tokens`, `Log Type`.
  "`Session ID`: conversation ID, only conversation mode have this". `Elapsed time` = từ lúc gửi tới khi nhận đủ câu trả lời (s).
  121 ngày liên tục (file 1–2, ~5.29M dòng), Azure OpenAI GPT-3.5/GPT-4; tách `Conversation log` và `API log`.
- **Dùng cho:** **nhịp đến của người dùng theo thời gian** (theo ngày/giờ, đột biến), số lượt mỗi phiên và khoảng cách giữa các lượt
  (đối chiếu chéo với D2), độ dài vào/ra theo lượt. Không có nội dung prompt.
- **Đã tự kiểm (2026-10-12, [measurements.md](../measurements.md)):** chỉ file **BurstGPT_3** có `Session ID`/`Elapsed time`;
  `Request tokens` **gồm lịch sử hội thoại** (corr 0.905 với req + resp lượt trước); think time p50 = 124 s (khớp D4 ~100 s);
  95.6% request là API log (không có phiên). Bản lưu dữ liệu: `data/raw/burstgpt/` (không commit).

### D4. Xiang et al. (2026). *ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production.* NSDI '26. ✅

- Link: <https://arxiv.org/abs/2505.09999> · Code: <https://github.com/alibaba/ServeGen> · Bản lưu: `files/servegen2026-nsdi.pdf` · Tra: 2026-10-12
- Dữ liệu thật Alibaba Cloud Model Studio, 4 tháng, 12 model, hàng tỷ request.
- **Kết quả dùng làm bằng chứng:**
  - Request đến **đột biến** (CV ≫ 1), không một quá trình ngẫu nhiên nào khớp mọi trường hợp; tải **theo giờ trong ngày**
    (cao điểm buổi chiều, thấp sáng sớm).
  - **Phân bố độ dài vào/ra thay đổi theo thời gian**, tới 1.63× (vào) và 1.46× (ra), thay đổi độc lập nhau
    → bằng chứng cho nhận định "tỉ lệ ngắn/trung bình/dài không cố định".
  - Hội thoại nhiều lượt (deepseek-r1): 188,986 / 1,964,415 request (~10%), 57,205 hội thoại, trung bình **3.5 lượt**;
    "ITTs concentrate around 100 seconds, with an extremely long tail".
  - Sinh tải ngây thơ (co giãn khoảng cách thời gian) cho tải đột biến sai; phải **giữ phân bố thời gian giữa các lượt** mới thật.
  - Bộ sinh tải ghép theo từng client; công bố tham số đã làm sạch, không công bố dữ liệu thô.
- **Dùng cho:** lý do phải sinh tải theo phiên + đối chiếu think time (~100 s) với số đo từ D2/D3.

### D5. Qin et al. (2025). *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.* FAST '25. + traces. ⚠️

- Link: <https://arxiv.org/abs/2407.00079> · Trace: <https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces>
  (`conversation_trace.jsonl`, `toolagent_trace.jsonl`, `synthetic_trace.jsonl`) · Bản lưu: `files/mooncake2025-fast.pdf`, `files/mooncake-readme.md` · Tra: 2026-10-12
- Trace có `timestamp`, `input_length`, `output_length`, `hash_ids` (băm từng khối 512 token → thấy được phần prefix dùng chung).
  Chưa mở các file FAST25 để kiểm trường dữ liệu.
- **Dùng cho:** đối chiếu tỉ lệ dùng lại prefix (hội thoại nhiều lượt) nếu cần.

### D6. Công cụ benchmark nhiều lượt có sẵn. ✅

- **vLLM `benchmarks/multi_turn`** (`files/vllm-multi-turn-bench-readme.md`): phát lại hội thoại (ShareGPT hoặc sinh tổng hợp),
  `--num-clients`, `--max-active-conversations`; README **không có think time** giữa các lượt.
- **NVIDIA AIPerf** (`files/nvidia-aiperf-multi-turn.md`): `--conversation-turn-delay-mean/-stddev` (ms) mô phỏng think time,
  mặc định 0. Chỉ có trung bình/độ lệch, không theo phân bố thật.
- **GuideLLM issue #1082** (`files/guidellm-issue-1082.html`): đề xuất "conversation-level concurrency" cho tải nhiều lượt thật — đang mở.
- Kết luận: **chưa công cụ nào** có đủ phiên + think time theo phân bố thật + người dùng đến theo trace + nội dung RAG doanh nghiệp.
- Tra: 2026-10-12. Links: <https://github.com/vllm-project/vllm/tree/main/benchmarks/multi_turn> ·
  <https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/multi-turn-conversations> · <https://github.com/vllm-project/guidellm/issues/1082>

---

## E. Đo chất lượng: tác vụ, thống kê, độ gần với baseline, tính tái lập (cho P06 và mọi bước tối ưu sau)

> Tra 2026-10-13 khi chuẩn bị P06. Quyết định dựa trên nhóm này: [P06 plan.md](../gd1-vllm-baseline/06-chat-luong-baseline/plan.md) mục "Quyết định đã chốt".

### E1. Qwen Team (2025). *Qwen3 Technical Report.* arXiv 2505.09388. ✅

- Link: <https://arxiv.org/abs/2505.09388> · Bản lưu: `files/qwen3-2025-technical-report.pdf` · Tra: 2026-10-13
- **Bảng 16 (post-trained, non-thinking), cột Qwen3-30B-A3B:** MMLU-Redux 84.1 · GPQA-Diamond 54.8 · IFEval strict prompt 83.7 ·
  MATH-500 89.8 · AIME'24 32.8 · Arena-Hard 88.0 · BFCL v3 58.6 · LiveCodeBench v5 29.8.
- **Bảng 5 (bản Base, không phải bản chat):** Qwen3-30B-A3B-Base MMLU-Pro 61.49 · GSM8K 91.81 · MMLU 81.38.
  → Báo cáo **không có** MMLU-Pro / GSM8K cho bản post-trained.
- **Cách đánh giá (mục 4.6):** non-thinking dùng **lấy mẫu** temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5;
  max output 32,768 token. Thinking: temperature 0.6, top-p 0.95.
- **Dùng cho:** chọn tác vụ có số công bố để so INT4 với bản gốc (IFEval); biết điểm công bố đo bằng lấy mẫu, không phải greedy
  → chênh lệch với số mình đo một phần do cách sinh.

### E2. Qwen — model card `Qwen/Qwen3-30B-A3B-Instruct-2507` (bảng so sánh). ✅

- Link: <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507> · Bản lưu: `files/qwen3-30b-a3b-instruct-2507-model-card.md` · Tra: 2026-10-13
- Bảng benchmark có cột **"Qwen3-30B-A3B Non-Thinking"** (đúng bản dự án dùng, ở chế độ tắt thinking):
  **MMLU-Pro 69.1** · MMLU-Redux 84.1 · GPQA 54.8 · SuperGPQA 42.2 · IFEval 83.7 · MMLU-ProX 65.1.
- **Dùng cho:** mốc so sánh MMLU-Pro của bản INT4 (P06 câu hỏi 1).

### E3. Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* arXiv 2411.00640 (Anthropic). ✅

- Link: <https://arxiv.org/abs/2411.00640> · Bản lưu: `files/miller2024-error-bars-evals.pdf` · Tra: 2026-10-13
- **Khuyến nghị (mục 1):** (1) sai số chuẩn theo định lý giới hạn trung tâm; (2) sai số chuẩn theo cụm khi câu hỏi theo nhóm;
  (3) giảm phương sai bằng lấy mẫu lại / dùng xác suất token kế tiếp; (4) **so hai model bằng hiệu theo từng câu (paired)**;
  (5) dùng power analysis để biết một eval (hoặc **một tập con ngẫu nhiên**) có đủ để kiểm giả thuyết không.
- **CI 95% = s̄ ± 1.96 × SE** (công thức 3). Paired: Var(A−B) = Var(A) + Var(B) − 2 Cov(A, B) → hai model trả lời giống nhau
  nhiều thì sai số của hiệu rất nhỏ (mục 4.2).
- **Cỡ mẫu (công thức 9, ví dụ mục 5):** phát hiện chênh δ = 0.03 với α = 0.05, power 80%: n ≈ **969** câu →
  "new evals should contain at least 1,000 questions".
- **Dùng cho:** MMLU-Pro lấy **tập con ngẫu nhiên 1,000 câu** (không chạy đủ 12,032); báo mọi điểm kèm CI 95%; so bản tối ưu với
  baseline bằng paired SE.

### E4. Dutta et al. (2024). *Accuracy is Not All You Need.* NeurIPS 2024, arXiv 2407.09141 (Microsoft Research). ✅

- Link: <https://arxiv.org/abs/2407.09141> · Bản lưu: `files/dutta2024-accuracy-not-all-you-need.pdf` · Tra: 2026-10-13
- Các phương pháp lượng tử hoá có độ chính xác chênh **0–2%** so với 16-bit nhưng **flips** (câu đổi đúng↔sai) tới **13.6%**;
  chỉ GPTQ W8A16 có flips không đáng kể (mục 1). MMLU: 8–16% flips với BnB W4A4 (mục 4).
- **Định nghĩa:** flips = % câu đổi đúng→sai hoặc sai→đúng; KL = trung bình KL(P_baseline || P_nén) trên mọi token của đáp án,
  tính trên **toàn bộ từ vựng** (công thức 1). KL và flips tương quan Spearman **0.981** trên MMLU.
- **Kết luận tác giả:** "distance metrics such as flips and KL-divergence are essential for evaluating all optimization methods
  which may change the model outputs".
- **Dùng cho:** mọi bước tối ưu (KV FP8, thư viện khác, kernel GĐ4) phải báo **flips + độ gần (KL/Δp/same-top-1)**, không chỉ độ chính xác.

### E5. llama.cpp — `tools/perplexity/README.md` (mục KL divergence). ✅

- Link: <https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md> (commit README gần nhất `92f7da00b4`, 2026-03-05) ·
  Bản lưu: `files/llamacpp-perplexity-kld-readme.md` · Tra: 2026-10-13
- Cách so bản lượng tử với FP16 trên **cùng văn bản** (teacher forcing): KLD, Mean Δp ("change in 'correct' token probability"),
  RMS Δp, percentiles của Δp, **Same top p** ("Percentage of how often the token was assigned the highest probabilities by both models").
- "If the percentiles are symmetric then the quantization is essentially just adding noise. If the negative values are significantly
  larger than the positive values then this indicates that the model is actually becoming worse".
- File logits đầy đủ rất lớn: **11 GiB (LLaMA 2) / 37 GiB (LLaMA 3)** cho Wikitext-2.
- **Dùng cho:** bộ chỉ số độ gần của P06 (same-top-1, Δp và tính đối xứng, KL); lý do không lưu đủ logits.

### E6. Anshumann et al. (2025). *Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs.* arXiv 2503.16870. ⚠️

- Link: <https://arxiv.org/abs/2503.16870> · Tra: 2026-10-13 (chỉ đọc tóm tắt qua tìm kiếm, chưa lưu bản gốc)
- Cảnh báo: chỉ giữ top-K xác suất của teacher làm ước lượng bị lệch (không thấy phần đuôi).
- **Dùng cho:** ghi rõ KL tính từ top-20 là **xấp xỉ**; P06 đo kèm "độ phủ" (tổng xác suất của top-20) để biết sai số; chỉ số chính
  dùng same-top-1 và Δp (tính chính xác, không cần đủ logits).

### E7. He, H. & Thinking Machines Lab (2025). *Defeating Nondeterminism in LLM Inference.* Thinking Machines blog, 10/9/2025. ✅

- Link: <https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/> · Bản lưu: `files/he2025-thinkingmachines-defeating-nondeterminism.html` · Tra: 2026-10-13
- Nguyên nhân output khác nhau ở temperature 0: forward pass **không bất biến theo kích thước batch** ("our request's output does depend
  on the parallel user requests... because our forward pass lacks 'batch invariance'").
- Thí nghiệm: Qwen3-235B-A22B-Instruct-2507, 1,000 lần sinh ở temperature 0 → **80 kết quả khác nhau**, giống nhau 102 token đầu,
  lệch từ token 103. Bật kernel bất biến theo batch → 1,000/1,000 giống nhau.
- **Dùng cho:** P06 đo "mức tự lệch" của vLLM mặc định ở các mức tải khác nhau làm ngưỡng; không coi mọi lệch token là lỗi.

### E8. vLLM docs — *Reproducibility* (latest, cập nhật 7/10/2026). ✅

- Link: <https://docs.vllm.ai/en/latest/usage/reproducibility.html> · Bản lưu: `files/vllm-docs-reproducibility.html` · Tra: 2026-10-13
- "vLLM does not guarantee the reproducibility of the results by default, for the sake of performance." Online mode chỉ tái lập được khi
  bật batch invariance (`VLLM_BATCH_INVARIANT=1`, beta). "Even with the above settings, vLLM only provides reproducibility when it runs
  on the same hardware and the same vLLM version."
- **Dùng cho:** baseline đo ở chế độ mặc định (không bật batch invariance vì làm chậm); so sánh chất lượng phải cùng máy, cùng phiên bản.

### E9. Chen et al. (2021). *FinQA: A Dataset of Numerical Reasoning over Financial Data.* EMNLP 2021, arXiv 2109.00122. ✅

- Link: <https://arxiv.org/abs/2109.00122> · Bản lưu: `files/finqa2021-emnlp.pdf` · Dữ liệu: <https://github.com/czyssrs/FinQA> (`dataset/test.json`,
  lưu ở `data/raw/finqa/test.json`) · Tra: 2026-10-13
- 8,281 câu hỏi số liệu trên báo cáo tài chính (bảng + văn bản); chia train/dev/test, **test 1,147**. Chỉ số: **execution accuracy**
  (kết quả số đúng). Chuyên gia tài chính đạt 91.16%, người không chuyên 50.68%.
- Kiểm 2026-10-13: RAGBench `finqa` test có 1,147 id / 1,138 câu hỏi khác nhau, **khớp 1,138/1,138** câu hỏi của FinQA test
  → lấy được đáp án gốc (`qa.exe_ans`).
- **Dùng cho:** độ chính xác RAG của P06 (tài liệu tài chính doanh nghiệp, chấm tự động bằng đáp án số).

### E10. Friel et al. (2024). *RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems.* arXiv 2407.11005. ✅

- Link: <https://arxiv.org/abs/2407.11005> · Bản lưu: `files/ragbench2024.pdf` · Tra: 2026-10-13
- 12 dataset thành phần, giữ split gốc; với FinQA "We retain the original splits and generate 2 LLM responses per each context-query
  example"; TAT-QA "generate new responses with LLMs". Cột `response` là **câu trả lời do LLM sinh** (gpt-3.5-turbo-0125...) kèm nhãn
  bám tài liệu, **không phải đáp án đúng**.
- **Dùng cho:** không chấm RAG bằng `response` của RAGBench; lấy đáp án từ dataset gốc (E9).

### E11. Bộ tác vụ chuẩn trong lm-evaluation-harness 0.4.13 (ghim). ✅

- MMLU-Pro — Wang et al. (2024), NeurIPS 2024 D&B, arXiv 2406.01574 (`files/mmlupro2024-neurips.pdf`): **12,032 câu, 14 môn, 10 lựa chọn**;
  "MMLU-Pro necessitates chain-of-thought (CoT)". Task lm-eval `mmlu_pro`: 5-shot CoT, greedy, max 2,048 token, trích "answer is (X)".
- GSM8K — Cobbe et al. (2021), arXiv 2110.14168 (`files/gsm8k2021-cobbe.pdf`): 7.5K train / **1K test** (bản HF `openai/gsm8k` có 1,319 câu test);
  task `gsm8k`: 5-shot, greedy, trích "#### số".
- IFEval — Zhou et al. (2023), arXiv 2311.07911 (`files/ifeval2023-zhou.pdf`): **25 loại** chỉ dẫn kiểm được bằng luật, **541 prompt**;
  task `ifeval`: 0-shot, greedy, max 1,280 token, chấm bằng luật (không cần model giám khảo).
- lm-eval 0.4.13 (PyPI, phát hành 2026-08-31): `--samples` chọn đúng chỉ số câu theo task (tập con ngẫu nhiên có seed), `--use_cache`
  lưu từng câu đã trả lời (chạy tiếp được). Model kiểu API (`local-completions`) **không truyền `enable_thinking`** vào chat template
  (`lm_eval/models/api_models.py` `apply_chat_template`) → P06 dùng lớp con nhỏ để render giống lúc đo tốc độ.

### E12. Cách chấm đáp án số khi dùng LLM cho bài tài chính. ✅

- **Long et al. (2025). *FinLFQA: Evaluating Attributed Text Generation of LLMs in Financial Long-Form Question Answering.*** arXiv 2510.06426.
  Bản lưu: `files/finlfqa2025.pdf`. "a predicted number is considered correct if it matches the ground truth within a relative tolerance
  of 0.01 (e.g., 3.965 is matched to 3.97)", kèm chuẩn hoá đơn vị (scale normalization).
- **de Padua, Qureshi & Karakaplan (2023). *GPT-3 Models are Few-Shot Financial Reasoners.*** arXiv 2307.13617. Bản lưu:
  `files/gpt3-financial-reasoners2023.pdf`. Báo execution accuracy trên FinQA theo dung sai "between 1% to 200% at 1% increments".
- Tra: 2026-10-13. **Dùng cho:** P06 chấm FinQA-RAG đúng nếu lệch tương đối ≤ 1% (chính), báo thêm 0.1% và 5%; chấp nhận dạng phần trăm
  vì đáp án gốc FinQA ghi phần trăm dạng phân số (0.14464).

---

## F. Tune vLLM trên L4: cơ chế bộ xếp lịch, KV cache, ngân sách token, hội thoại nhiều lượt (cho P07 và plan tune GĐ1)

> Tra 2026-10-14 khi làm P07. Dùng cho danh sách nghi vấn tune: [P07 bao-cao.md](../gd1-vllm-baseline/07-tong-hop-baseline/bao-cao.md) mục 4.
> Tài liệu và mã vLLM lấy đúng tag **`v0.30.0`** (phiên bản đã ghim) từ `raw.githubusercontent.com/vllm-project/vllm/v0.30.0/...`
> (docs.vllm.ai trả lỗi 429 hôm đó; bản "latest" có thể khác bản đang dùng).

### F1. vLLM 0.30.0 docs — *Optimization and Tuning* (`docs/configuration/optimization.md`) + *Automatic Prefix Caching* design (`docs/design/prefix_caching.md`). ✅

- Bản lưu: `files/vllm-0.30.0-docs-optimization.md`, `files/vllm-0.30.0-docs-design-prefix-caching.md` · Tra: 2026-10-14
- **Preemption:** "preemption and recomputation can adversely affect end-to-end latency". Gợi ý: tăng `gpu_memory_utilization`;
  giảm `max_num_seqs` hoặc `max_num_batched_tokens`. V1 mặc định RECOMPUTE.
- **Chunked prefill:** lập lịch ưu tiên decode, phần ngân sách còn lại cho prefill. "Smaller values (e.g., 2048) achieve better ITL because
  there are fewer prefills slowing down decodes. Higher values achieve better time to first token (TTFT)"; "For optimal throughput, we recommend
  setting `max_num_batched_tokens > 8192`".
- **`--kv-cache-memory`:** đặt cố định dung lượng KV, bỏ bước đo bộ nhớ lúc khởi động; "a conservative value caps batch concurrency".
  (Trong mã v0.30.0 tên cờ là `--kv-cache-memory-bytes`, xem F3.)
- **Prefix caching:** khối KV của request đã xong vào cuối hàng "free queue"; cần chỗ thì lấy khối đầu hàng, tức **ít dùng gần nhất (LRU)**,
  và khối đó mất khả năng dùng lại.
- **Dùng cho:** P07 N1 (KV), N3 (ngân sách token), N4 (preemption), N2 (vì sao lịch sử bị đẩy ra).

### F2. vLLM 0.30.0 docs — *Quantized KV Cache* (`docs/features/quantization/quantized_kvcache.md`). ✅

- Bản lưu: `files/vllm-0.30.0-docs-quantized-kvcache.md` · Tra: 2026-10-14
- KV FP8 giảm bộ nhớ KV, "store more tokens in memory". `kv_cache_dtype="fp8"` không hiệu chuẩn thì mọi scale = 1.0;
  "[Recommended] Calibration with a dataset (via llm-compressor)". Per-head scale chỉ có ở backend Flash Attention.
- **Dùng cho:** P07 N1. Hạn chế đã biết trên Ada: FA2 không hỗ trợ KV FP8 → backend khác ([library-gaps.md](../library-gaps.md) #3).

### F3. vLLM 0.30.0 — mã nguồn bộ xếp lịch và cấu hình. ✅

- Bản lưu: `files/vllm-0.30.0-v1-scheduler.py` (`vllm/v1/core/sched/scheduler.py`), `files/vllm-0.30.0-config-scheduler.py`,
  `files/vllm-0.30.0-config-cache.py`, `files/vllm-0.30.0-engine-arg-utils.py` · Tra: 2026-10-14 · License Apache-2.0
- **Mặc định cho GPU < 70 GB** (`EngineArgs.get_batch_defaults`, server OpenAI): `max_num_batched_tokens` **2048**, `max_num_seqs` **256**.
- **Nhận request chờ** (`Scheduler.schedule`, phần "Next, schedule the WAITING requests"): duyệt hàng đợi theo `policy` (mặc định `"fcfs"`);
  `allocate_slots(..., full_sequence_must_fit=self.scheduler_reserve_full_isl)`; nếu `new_blocks is None` → `break`
  (request phía sau không được xét trong bước đó).
- `scheduler_reserve_full_isl: bool = True`: "checks whether the full input sequence length fits in the KV cache before admitting a new request,
  rather than only checking the first chunk. Prevents over-admission and KV cache thrashing with chunked prefill."
- `watermark: float = 0.0`: "Fraction of total KV cache blocks to keep free ... when admitting waiting or preempted requests ... helps avoid
  frequent KV cache eviction and the resulting repeated preemption".
- `long_prefill_token_threshold` (0 = tắt), `policy` `"fcfs"` | `"priority"` ("lower value means earlier handling"), `scheduler_cls` (lớp xếp lịch tuỳ biến),
  `max_num_queued_tokens` (từ chối request bằng HTTP 503 khi tồn prefill vượt mức).
- Preemption khi request đang chạy thiếu KV: FCFS chọn `self.running[-1]` (request vào sau cùng); priority chọn request ưu tiên thấp nhất.
- `kv_offloading_size` (GiB, mặc định None = tắt), `kv_offloading_backend` `"native"` | `"lmcache"`: "enable KV cache offloading to CPU".
- Tên cờ CLI: `--scheduling-policy`, `--long-prefill-token-threshold`, `--scheduler-reserve-full-isl`, `--watermark`, `--scheduler-cls`,
  `--kv-offloading-size`, `--kv-offloading-backend`, `--kv-cache-memory-bytes`, `--kv-cache-dtype`.
- **Dùng cho:** P07 N3 (cơ chế chặn đầu hàng), N1, N2, N4; ứng viên GĐ4 bộ xếp lịch cắm qua `scheduler_cls`.

### F4. Agrawal et al. (2024). *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* OSDI '24, arXiv 2403.02310. ✅

- Link: <https://arxiv.org/abs/2403.02310> · Bản lưu: `files/sarathi-serve2024-osdi.pdf` · Tra: 2026-10-14
- Chunked prefill + "stall-free batching": giới hạn số token mỗi bước (token budget) để prefill không làm khựng decode.
- Mục 4.3: chọn token budget là đánh đổi giữa chi phí prefill và độ trễ decode; "setting the token budget to maximum number of tokens that can be packed
  in a batch without violating TBT SLO". Budget quá nhỏ: "lower arithmetic intensity and other fixed overheads". Kích thước khúc không chia hết tile
  có thể làm prefill chậm hơn 32% (257 so với 256).
- **Dùng cho:** P07 N3: TPOT đang dư (82–104 ms so với SLO 150) → có chỗ tăng `max_num_batched_tokens`.

### F5. Gao et al. (2024). *Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention.* USENIX ATC '24, arXiv 2403.19708. ✅ (phần tóm tắt)

- Link: <https://arxiv.org/abs/2403.19708> · Bản lưu: `files/cachedattention2024-atc.pdf` · Tra: 2026-10-14
- Hội thoại nhiều lượt phải tính lại KV của lịch sử; giữ KV ở tầng bộ nhớ rẻ hơn (RAM, đĩa), nạp trước theo lớp, xếp chỗ theo gợi ý của bộ xếp lịch;
  tách mã hoá vị trí để KV còn dùng được khi cắt bớt ngữ cảnh. Kết quả (tóm tắt): "decreases the time to the first token (TTFT) by up to 87%,
  improves the prompt prefilling throughput by up to 7.8× for multi-turn conversations, and reduces the end-to-end inference cost by up to 70%".
- Chỉ mới đối chiếu phần tóm tắt và giới thiệu; chưa đọc phần thí nghiệm.
- **Dùng cho:** P07 N2 (giữ lịch sử ở RAM CPU); ứng viên GĐ4 "giữ KV khi client cắt lịch sử".

### F6. Qwen — model card `Qwen/Qwen3-30B-A3B` (mục Best Practices). ✅

- Link: <https://huggingface.co/Qwen/Qwen3-30B-A3B> · Bản lưu: `files/qwen3-30b-a3b-model-card.md` · Tra: 2026-10-14
- "**No Thinking Content in History**: In multi-turn conversations, the historical model output should only include the final output part and does not
  need to include the thinking content. It is implemented in the provided chat template in Jinja2."
- **Dùng cho:** P07 N5: giữ khối `<think></think>` rỗng trong lịch sử là lệch khỏi cách dùng Qwen khuyến nghị → cần cổng chất lượng trước khi dùng.

---

## G. Phương pháp so sánh hiệu năng (cho mọi phép so ở GĐ1 tune → GĐ5)

### G1. Hoefler, T. & Belli, R. (2015). *Scientific Benchmarking of Parallel Computing Systems: Twelve ways to tell the masses when reporting performance results.* SC '15. ✅

- Link: <https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf> · Bản lưu: `files/hoefler2015-sc-scientific-benchmarking.pdf` · Tra: 2026-10-14
- Quy tắc dùng:
  - Rule 5: "For nondeterministic data, report confidence intervals of the measurement."
  - Rule 6: "Do not assume normality of collected data (e.g., based on the number of samples) without diagnostic checking."
  - Rule 7: "Compare nondeterministic data in a statistically sound way, e.g., using non-overlapping confidence intervals or ANOVA."
  - Rule 8: cân nhắc trung bình / trung vị có phù hợp không; "worst-case latency, may require other percentiles".
  - Rule 9: "Document all varying factors and their levels as well as the complete experimental setup".
- **Dùng cho:** P07 dùng một giá tham chiếu cố định (Rule 9); báo khoảng min–max giữa các lần lặp thay vì CI theo request (Rule 6);
  quy tắc "khoảng goodput không chồng nhau" sẵn có của dự án khớp Rule 7.

### G2. Cawley, G. C. & Talbot, N. L. C. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* JMLR 11, 2079–2107. ✅ (phần tóm tắt)

- Link: <https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf> · Bản lưu: `files/cawley2010-jmlr-selection-bias.pdf` · Tra: 2026-10-14
- Tối ưu theo một tiêu chí đo có phương sai thì cách chọn bị "over-fitting", và số đo của phương án được chọn bị lệch lạc quan; "the effects of this form of
  over-fitting are often of comparable magnitude to differences in performance between learning algorithms". Kết luận áp dụng cho
  "any model selection practice involving the optimisation of a model selection criterion evaluated over a finite sample of data".
- Bài viết cho học máy; dự án dùng nguyên lý chung: chọn cấu hình tune tốt nhất từ nhiều lần đo có nhiễu thì phải đo lại cấu hình thắng trên dữ liệu mới.
- **Dùng cho:** P07 mục 6.1 R3.
