"""Dựng các bộ câu hỏi cho đo chất lượng (P06). Chạy ở LOCAL, không cần GPU.

    .venv/bin/python quality/build_sets.py        → data/quality/{reference,finqa_rag}.jsonl, mmlu_pro_samples.json,
                                                    sha256 ghi vào quality/manifest.json

Thiết kế + lý do: docs/gd1-vllm-baseline/06-chat-luong-baseline/plan.md mục "Quyết định đã chốt"; bằng chứng: docs/references nhóm E.
- reference.jsonl (1,000 prompt, greedy, 256 token ra): so độ gần token-by-token giữa baseline và mọi bản tối ưu sau này.
    500 W1-like (ShareGPT, KHÔNG nằm trong data/workloads/w1.jsonl), 400 W2-like (RAGBench split TEST, cùng cách dựng W2 nhưng prompt 1–4k token vì tài liệu test ngắn hơn),
    100 W3-like (lấy từ data/workloads/w3.jsonl — W3 đã dùng toàn bộ CUAD nên không còn tài liệu dài tách riêng).
    Cờ `tf`: 200 prompt W1-like ngắn nhất (≤ 300 token) dùng thêm cho teacher forcing (prompt_logprobs; prompt dài thì vLLM
    tính logits cho mọi vị trí prompt → tốn bộ nhớ, xem plan).
- finqa_rag.jsonl: RAGBench `finqa` test (1,147 id) dựng prompt như W2 + yêu cầu dòng cuối "Answer: <số>"; đáp án gốc từ
  FinQA test.json (`qa.exe_ans`), khớp theo câu hỏi.
- mmlu_pro_samples.json: tập con ngẫu nhiên 1,000 câu MMLU-Pro, chia theo tỉ lệ 14 môn, chỉ số theo thứ tự trong từng
  task con của lm-eval (`mmlu_pro_<môn>` = test lọc theo category, giữ thứ tự) → truyền vào `lm-eval --samples`.
"""

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import ijson
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workloads"))
from build import (PROMPTS, RAW, W1_PROMPT, W2_PROMPT, Renderer, clamp, is_english, ragbench_rows,  # noqa: E402
                   sha256, unique_docsets)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/quality"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
SEED = 20261013
N_W1, N_W2, N_W3, N_TF = 500, 400, 100, 200
TF_MAX_PROMPT = 300
REF_MAX_TOKENS = 256
W2_TEST_SUBSETS = ["techqa", "emanual", "delucionqa", "tatqa"]  # finqa test để dành cho độ chính xác RAG
W2_REF_MIN = 1000  # split test của RAGBench có tài liệu ngắn hơn: ≥ 2k token chỉ có 180 câu, ≥ 1k có 472 (2026-10-13)
MMLU_PRO_N = 1000
FINQA_INSTR = ("\n\nAfter your answer, write the final result on the last line in exactly this form: "
               "\"Answer: <value>\", where <value> is a single number (no units, no thousands separators) "
               "or yes/no.")
SOURCES_Q = {
    "mmlu_pro": {"repo": "TIGER-Lab/MMLU-Pro", "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                 "file": "data/test-00000-of-00001.parquet"},
    "finqa": {"repo": "github.com/czyssrs/FinQA", "commit": "0f16e2867b", "file": "dataset/test.json"},
}


def w1_like(R, rng, used):
    rows = []
    with open(RAW / "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json", "rb") as f:
        for conv in ijson.items(f, "item"):
            turns = conv.get("conversations") or []
            if len(turns) < 2 or turns[0]["from"] != "human" or turns[1]["from"] != "gpt":
                continue
            q = turns[0]["value"].strip()
            if f"w1-{conv['id']}" in used or not (300 <= len(q) <= 5000) or not is_english(q):
                continue
            rows.append((conv["id"], q))
    rng.shuffle(rows)
    out = []
    for cid, q in rows:
        row = R.row(f"ref-w1-{cid}", "sharegpt", [{"role": "user", "content": q}], REF_MAX_TOKENS)
        if W1_PROMPT[0] <= row["prompt_tokens"] <= W1_PROMPT[1]:
            out.append(row)
        if len(out) >= N_W1 * 3:  # đủ nhiều để chọn tập ngắn cho teacher forcing
            break
    short = sorted([r for r in out if r["prompt_tokens"] <= TF_MAX_PROMPT], key=lambda r: r["id"])
    rng.shuffle(short)
    tf = {r["id"] for r in short[:N_TF]}
    rest = [r for r in out if r["id"] not in tf]
    chosen = short[:N_TF] + rest[:N_W1 - N_TF]
    for r in chosen:
        r["tf"] = r["id"] in tf
    return chosen


def rag_prompt(R, system, documents, question, suffix=""):
    """Như build_w2: lấy tài liệu theo thứ tự tới khi chạm 4k token."""
    docs, row = [], None
    for d in documents:
        docs.append(d)
        body = "\n\n".join(f"[Document {i + 1}]\n{x.strip()}" for i, x in enumerate(docs))
        user = f"{body}\n\nQuestion: {question.strip()}{suffix}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = R.render(messages)
        n = R.ntok(prompt)
        if n > W2_PROMPT[1] and row is not None:
            break
        row = (messages, prompt, n)
    return row


def w2_like(R, rng):
    system = (PROMPTS / "w2_system.txt").read_text().strip()
    rows = []
    for subset in W2_TEST_SUBSETS:
        for r in unique_docsets(ragbench_rows(subset, ["test"], ["id", "question", "documents"])):
            messages, prompt, n = rag_prompt(R, system, r["documents"], r["question"])
            if W2_REF_MIN <= n <= W2_PROMPT[1]:
                rows.append({"id": f"ref-w2-{r['id']}", "source": subset, "prompt": prompt, "prompt_tokens": n,
                             "output_tokens": REF_MAX_TOKENS, "messages": messages, "tf": False})
    rng.shuffle(rows)
    return rows[:N_W2]


def w3_like(rng):
    rows = [json.loads(x) for x in open(ROOT / "data/workloads/w3.jsonl")]
    rng.shuffle(rows)
    out = []
    for r in rows[:N_W3]:
        r = {**r, "id": "ref-" + r["id"], "output_tokens": REF_MAX_TOKENS, "tf": False}
        out.append(r)
    return out


def finqa_rag(R):
    system = (PROMPTS / "w2_system.txt").read_text().strip()
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    gold = {}
    for d in json.load(open(RAW / "finqa/test.json")):
        gold.setdefault(norm(d["qa"]["question"]), set()).add(json.dumps(d["qa"]["exe_ans"]))
    rows, seen, skipped = [], set(), 0
    for r in ragbench_rows("finqa", ["test"], ["id", "question", "documents"]):
        if r["id"] in seen:
            continue  # RAGBench có 2 câu trả lời LLM cho mỗi ví dụ → 2 dòng cùng id
        seen.add(r["id"])
        g = gold.get(norm(r["question"]))
        if not g or len(g) != 1:
            skipped += 1
            continue
        messages, prompt, n = rag_prompt(R, system, r["documents"], r["question"], FINQA_INSTR)
        rows.append({"id": r["id"], "question": r["question"], "gold": json.loads(next(iter(g))), "prompt": prompt,
                     "prompt_tokens": n, "output_tokens": 768, "messages": messages})
    return rows, skipped


def mmlu_pro_samples(rng):
    t = pq.read_table(RAW / "mmlu_pro/test.parquet", columns=["question_id", "category"]).to_pylist()
    cats = {}
    for i, r in enumerate(t):
        cats.setdefault(r["category"], []).append(i)
    total = len(t)
    # chia theo tỉ lệ (largest remainder) cho đủ đúng MMLU_PRO_N
    quota = {c: MMLU_PRO_N * len(v) / total for c, v in cats.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    for c in sorted(quota, key=lambda c: quota[c] - alloc[c], reverse=True)[:MMLU_PRO_N - sum(alloc.values())]:
        alloc[c] += 1
    samples = {}
    for c in sorted(cats):
        task = "mmlu_pro_" + c.replace(" ", "_")
        samples[task] = sorted(rng.sample(range(len(cats[c])), alloc[c]))  # chỉ số trong task con (đã lọc theo môn)
    return samples, {c: len(v) for c, v in cats.items()}


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = Renderer()
    rng = random.Random(SEED)
    used = {json.loads(x)["id"] for x in open(ROOT / "data/workloads/w1.jsonl")}
    ref = w1_like(R, rng, used) + w2_like(R, rng) + w3_like(rng)
    for i, r in enumerate(ref):
        r["idx"] = i
    write_jsonl(OUT / "reference.jsonl", ref)
    fq, skipped = finqa_rag(R)
    write_jsonl(OUT / "finqa_rag.jsonl", fq)
    samples, cat_sizes = mmlu_pro_samples(random.Random(SEED + 1))
    (OUT / "mmlu_pro_samples.json").write_text(json.dumps(samples, sort_keys=True) + "\n")

    files = {}
    for name in ("reference.jsonl", "finqa_rag.jsonl", "mmlu_pro_samples.json"):
        files[name] = {"path": f"data/quality/{name}", "sha256": sha256(OUT / name)}
    by_src = {}
    for r in ref:
        by_src.setdefault(r["source"], []).append(r["prompt_tokens"])
    stats = {
        "reference": {"n": len(ref), "tf": sum(r["tf"] for r in ref),
                      "by_source": {k: {"n": len(v), "prompt_min": min(v), "prompt_max": max(v),
                                        "prompt_mean": round(sum(v) / len(v))} for k, v in sorted(by_src.items())}},
        "finqa_rag": {"n": len(fq), "skipped_ambiguous_or_missing": skipped,
                      "gold_yes_no": sum(isinstance(r["gold"], str) for r in fq),
                      "prompt_mean": round(sum(r["prompt_tokens"] for r in fq) / len(fq)),
                      "prompt_max": max(r["prompt_tokens"] for r in fq)},
        "mmlu_pro": {"n": sum(len(v) for v in samples.values()), "per_task": {k: len(v) for k, v in samples.items()},
                     "category_sizes": cat_sizes},
    }
    MANIFEST.write_text(json.dumps({"seed": SEED, "files": files, "sources": SOURCES_Q, "stats": stats},
                                   indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
