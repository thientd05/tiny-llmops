"""Sinh bộ request cố định cho W1/W2/W3 (P02). Chạy trên máy local, không cần GPU.

    .venv/bin/python workloads/build.py            # build + ghi manifest
    .venv/bin/python workloads/build.py --check    # build lại, so sha256 với manifest

Mỗi dòng JSONL:
    prompt         prompt đã áp chat template Qwen3 (có system, thinking tắt) -> gửi /v1/completions,
                   `vllm bench serve --dataset-name custom --skip-chat-template`
    output_tokens  số token output yêu cầu (dùng kèm --ignore-eos)
    prompt_tokens  số token của prompt (tokenizer Qwen3)
    messages       dạng chat gốc, cho thư viện chỉ có /v1/chat/completions
    id, source     nguồn mẫu
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import ijson
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
OUT = ROOT / "data/workloads"
PROMPTS = Path(__file__).resolve().parent / "prompts"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"

SEED = 20261012
SOURCES = {
    "tokenizer": {"repo": "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "revision": "9b534e4318b7ebc3c961a839f13eb18b1833f441"},
    "ragbench": {"repo": "galileo-ai/ragbench (trước là rungalileo/ragbench)",
                 "revision": "97808f3e5fd16ede40bbff6c2949af8139b2eb7b", "license": "CC-BY-4.0 (dataset thành phần có license riêng)"},
    "sharegpt": {"repo": "anon8231489123/ShareGPT_Vicuna_unfiltered", "file": "ShareGPT_V3_unfiltered_cleaned_split.json",
                 "revision": "192ab2185289094fc556ec8ce5ce1e8e587154ca", "license": "apache-2.0 (nội dung từ ShareGPT, chỉ dùng nghiên cứu)"},
}

# Khoảng độ dài theo docs/workloads.md (đơn vị: token, tính cả chat template)
W1_PROMPT, W1_OUTPUT, W1_MAX = (200, 800), (200, 600), 2000
W2_PROMPT, W2_OUTPUT = (2000, 4000), (150, 400)
W2_SUBSETS = ["techqa", "emanual", "delucionqa", "finqa", "tatqa"]
W3_PROMPT, W3_OUTPUT = (4000, 8000), (250, 350)

W3_TASK = """Summarize the contract above with these sections:
1. Parties and their roles
2. Purpose and scope
3. Term, renewal and termination
4. Payment and financial obligations
5. Liability, indemnification and insurance
6. Confidentiality and intellectual property
7. Governing law and dispute resolution
8. Unusual clauses or risks the manager should know about"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Renderer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(RAW / "qwen3-tokenizer")

    def render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def ntok(self, text):
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def row(self, id_, source, messages, output_tokens):
        prompt = self.render(messages)
        return {"id": id_, "source": source, "prompt": prompt, "prompt_tokens": self.ntok(prompt),
                "output_tokens": int(output_tokens), "messages": messages}


def clamp(x, lo_hi):
    return max(lo_hi[0], min(lo_hi[1], x))


def is_english(text):
    return sum(c.isascii() for c in text) >= 0.95 * len(text)


def ragbench_rows(subset, splits, columns):
    for split in splits:
        path = RAW / f"ragbench/{subset}/{split}-00000-of-00001.parquet"
        yield from pq.read_table(path, columns=columns).to_pylist()


def unique_docsets(rows):
    """Mỗi bộ tài liệu chỉ dùng 1 lần -> không có cache hit 'ăn may' giữa các request."""
    seen = set()
    for r in rows:
        key = hashlib.sha1("\x00".join(r["documents"]).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            yield r


def build_w1(R, rng):
    rows = []
    with open(RAW / "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json", "rb") as f:
        for conv in ijson.items(f, "item"):
            turns = conv.get("conversations") or []
            if len(turns) < 2 or turns[0]["from"] != "human" or turns[1]["from"] != "gpt":
                continue
            q, a = turns[0]["value"].strip(), turns[1]["value"].strip()
            if not (300 <= len(q) <= 5000) or not a or not is_english(q):  # lọc thô theo ký tự trước khi đếm token
                continue
            messages = [{"role": "user", "content": q}]
            row = R.row(f"w1-{conv['id']}", "sharegpt", messages, clamp(R.ntok(a), W1_OUTPUT))
            if W1_PROMPT[0] <= row["prompt_tokens"] <= W1_PROMPT[1]:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:W1_MAX]


def build_w2(R, rng):
    system = (PROMPTS / "w2_system.txt").read_text().strip()
    rows, dropped = [], {}
    for subset in W2_SUBSETS:
        cols = ["id", "question", "documents", "response"]
        for r in unique_docsets(ragbench_rows(subset, ["train", "validation"], cols)):
            # Lấy tài liệu theo thứ tự (thứ tự retrieval) tới khi chạm ngân sách 4k token, như hệ RAG thật
            docs, row = [], None
            for d in r["documents"]:
                docs.append(d)
                body = "\n\n".join(f"[Document {i + 1}]\n{x.strip()}" for i, x in enumerate(docs))
                user = f"{body}\n\nQuestion: {r['question'].strip()}"
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                cand = R.row(f"w2-{r['id']}", subset, messages, clamp(R.ntok(r["response"] or ""), W2_OUTPUT))
                if cand["prompt_tokens"] > W2_PROMPT[1]:
                    break
                row = cand
            if row and row["prompt_tokens"] >= W2_PROMPT[0]:
                rows.append(row)
            else:
                dropped[subset] = dropped.get(subset, 0) + 1
    rng.shuffle(rows)
    return rows, dropped


def build_w3(R, rng):
    system = (PROMPTS / "w3_system.txt").read_text().strip()
    overhead = R.ntok(R.render([{"role": "system", "content": system},
                                {"role": "user", "content": f"CONTRACT:\n\n\n\n{W3_TASK}"}]))
    rows, truncated = [], 0
    cuad = unique_docsets(ragbench_rows("cuad", ["train", "validation", "test"], ["id", "documents"]))
    for r in cuad:
        text = "\n\n".join(d.strip() for d in r["documents"])
        target = rng.randint(*W3_PROMPT)  # độ dài mục tiêu, ngẫu nhiên có seed
        enc = R.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        budget = target - overhead
        if len(enc.input_ids) + overhead < W3_PROMPT[0]:
            continue
        if len(enc.input_ids) > budget:
            # cắt tại ranh giới đoạn văn gần nhất trước vị trí token thứ `budget`
            cut = text.rfind("\n", 0, enc.offset_mapping[budget][0])
            if cut <= 0:
                continue
            text = text[:cut].rstrip()
            truncated += 1
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"CONTRACT:\n\n{text}\n\n{W3_TASK}"}]
        row = R.row(f"w3-{r['id']}", "cuad", messages, rng.randint(*W3_OUTPUT))
        if W3_PROMPT[0] <= row["prompt_tokens"] <= W3_PROMPT[1]:
            rows.append(row)
    rng.shuffle(rows)
    return rows, truncated


def common_prefix_tokens(R, rows, n=50):
    ids = [R.tok(r["prompt"], add_special_tokens=False).input_ids for r in rows[:n]]
    k = 0
    while all(len(x) > k and x[k] == ids[0][k] for x in ids):
        k += 1
    return k


def stats(R, rows):
    p = np.array([r["prompt_tokens"] for r in rows])
    o = np.array([r["output_tokens"] for r in rows])
    desc = lambda a: {"min": int(a.min()), "mean": round(float(a.mean()), 1), "p50": int(np.median(a)),
                      "p95": int(np.percentile(a, 95)), "max": int(a.max())}
    src = {}
    for r in rows:
        src[r["source"]] = src.get(r["source"], 0) + 1
    return {"n": len(rows), "prompt_tokens": desc(p), "output_tokens": desc(o), "sources": src,
            "shared_prefix_tokens": common_prefix_tokens(R, rows)}


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="build lại và so sha256 với manifest")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    R = Renderer()
    w1 = build_w1(R, random.Random(SEED + 1))
    w2, w2_dropped = build_w2(R, random.Random(SEED + 2))
    w3, w3_truncated = build_w3(R, random.Random(SEED + 3))

    files = {}
    for name, rows in [("w1", w1), ("w2", w2), ("w3", w3)]:
        path = OUT / f"{name}.jsonl"
        write_jsonl(path, rows)
        files[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), **stats(R, rows)}
    files["w2"]["dropped_docsets_out_of_range"] = w2_dropped
    files["w3"]["truncated_contracts"] = w3_truncated

    manifest = {"seed": SEED, "sources": SOURCES, "system_prompts": {
        "w2": {"file": "workloads/prompts/w2_system.txt", "sha256": sha256(PROMPTS / "w2_system.txt")},
        "w3": {"file": "workloads/prompts/w3_system.txt", "sha256": sha256(PROMPTS / "w3_system.txt")}},
        "chat_template": "Qwen3, enable_thinking=False, add_generation_prompt=True", "files": files}

    if args.check:
        old = json.loads(MANIFEST.read_text())
        ok = all(old["files"][k]["sha256"] == files[k]["sha256"] for k in files)
        print("TÁI LẬP OK" if ok else "SHA256 KHÁC MANIFEST")
        sys.exit(0 if ok else 1)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(files, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
