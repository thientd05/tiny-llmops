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
