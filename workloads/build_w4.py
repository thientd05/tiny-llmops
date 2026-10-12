"""Sinh kịch bản W4 (tải giống thật) từ phiên hội thoại BurstGPT_3. Chạy local, không cần GPU.

    .venv/bin/python workloads/build_w4.py pool                        # 1 lần: kho nội dung (~vài phút)
    .venv/bin/python workloads/build_w4.py profile                     # tải theo giờ trong ngày, chọn khung giờ
    .venv/bin/python workloads/build_w4.py scenario --segment peak --k 20 [--seed 1]

Thiết kế: docs/workloads.md mục "W4". Tóm tắt:
- Chỉ dùng `Conversation log` (người chat, có Session ID); bỏ API log (chương trình gọi).
- Khung giờ: chọn theo giờ trong ngày (peak / peak2 / normal), cửa sổ = 10 phút khởi động + 30 phút đo.
- Tăng tải = **chồng phiên** của cùng khung giờ từ k ngày khác nhau (k ngày = k lần số người dùng),
  giữ nguyên think time và cấu trúc từng phiên. Ngày lấy theo một hoán vị cố định theo seed → kịch bản k nằm trong k' > k.
- Độ dài lấy từ trace (`Request tokens` đã gồm lịch sử). Nội dung thật: tin nhắn < 1500 token lấy từ ShareGPT,
  dài hơn thì ghép tài liệu RAGBench (train+val). Mỗi tin nhắn / tài liệu mở đầu chỉ dùng một lần → không trúng cache ăn may.

Mỗi dòng của file kịch bản là một phiên:
    start_s            thời điểm gửi lượt đầu (trong cửa sổ), tính từ đầu cửa sổ
    resumed            phiên đã bắt đầu trước cửa sổ (người đang chat dở): lượt đầu mang đủ độ dài lịch sử theo trace,
                       như người quay lại hội thoại cũ sau một lúc nghỉ (KV của lịch sử phải prefill lại)
    turns[k].think_s   lượt k (k ≥ 2) gửi sau khi lượt k−1 **nhận xong** think_s giây (partly-open: người chờ trả lời rồi mới hỏi)
    turns[k].user      tin nhắn mới; prompt thật = lịch sử thật + tin nhắn (workloads/w4_session.assemble)
    turns[k].prompt_budget  độ dài prompt tối đa (theo trace, đã chặn bởi max_model_len); vượt thì bỏ lịch sử cũ
    turns[k].max_tokens     số token output (dùng kèm ignore_eos)
Client chỉ đo request gửi trong [warmup_s, warmup_s + measure_s); hết cửa sổ thì không gửi lượt mới.
"""

import argparse
import bisect
import json
import random
import sys
from pathlib import Path

import ijson
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_burstgpt import SHA256 as BURSTGPT_SHA256, URL as BURSTGPT_URL, load as load_trace  # noqa: E402
from build import RAW, ROOT, SOURCES, Renderer, is_english, sha256  # noqa: E402

OUT = ROOT / "data/workloads/w4"
POOL = ROOT / "data/workloads/w4_pool"
MANIFEST = Path(__file__).resolve().parent / "manifest_w4.json"

SEED = 20261012
MAX_MODEL_LEN = 16384           # cờ duy nhất của baseline (P01)
MAX_OUTPUT = 4096
WARMUP_S, MEASURE_S = 600, 1800
DOC_MIN = 1500                  # tin nhắn từ ngưỡng này trở lên = người dùng dán tài liệu
CHAT_POOL_MAX = DOC_MIN + 100
DOC_SUBSETS = ["techqa", "emanual", "delucionqa", "finqa", "tatqa", "cuad"]
DOC_HEADER = "Please read the document below and help me with it.\n\n"
SOURCES_W4 = {**SOURCES, "burstgpt": {"url": BURSTGPT_URL, "sha256": BURSTGPT_SHA256, "license": "CC-BY-4.0"}}
LEN_BINS = [0, 1000, 2000, 4000, 8000, 10**9]
LEN_LABELS = ["<1k", "1-2k", "2-4k", "4-8k", ">=8k"]


# ---------- kho nội dung ----------

def build_pool(R):
    POOL.mkdir(parents=True, exist_ok=True)
    texts, seen = [], set()
    with open(RAW / "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json", "rb") as f:
        for conv in ijson.items(f, "item"):
            for t in conv.get("conversations") or []:
                q = t["value"].strip()
                if t["from"] == "human" and q and len(q) <= 4 * CHAT_POOL_MAX * 2 and is_english(q) and q not in seen:
                    seen.add(q)
                    texts.append(q)
    chat = []
    for i in range(0, len(texts), 5000):
        batch = texts[i:i + 5000]
        for q, ids in zip(batch, R.tok(batch, add_special_tokens=False).input_ids):
            if len(ids) <= CHAT_POOL_MAX:
                chat.append({"text": q, "tokens": len(ids)})
    random.Random(SEED + 41).shuffle(chat)

    docs, seen = [], set()
    for subset in DOC_SUBSETS:
        for split in ["train", "validation"]:  # test để dành đo chất lượng
            path = RAW / f"ragbench/{subset}/{split}-00000-of-00001.parquet"
            for ds in pq.read_table(path, columns=["documents"]).column("documents").to_pylist():
                for d in ds:
                    d = d.strip()
                    if d and d not in seen:
                        seen.add(d)
                        docs.append({"text": d, "subset": subset})
    for i in range(0, len(docs), 2000):
        batch = docs[i:i + 2000]
        for d, ids in zip(batch, R.tok([d["text"] for d in batch], add_special_tokens=False).input_ids):
            d["tokens"] = len(ids)
    docs = [d for d in docs if d["tokens"] >= 50]
    random.Random(SEED + 42).shuffle(docs)

    files = {}
    for name, rows in [("chat", chat), ("docs", docs)]:
        path = POOL / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        files[name] = {"path": str(path.relative_to(ROOT)), "n": len(rows), "sha256": sha256(path)}
    write_manifest({"pool": files, "sources": SOURCES_W4})
    print(json.dumps(files, indent=2))


def write_manifest(update):
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"seed": SEED}
    m.update(update)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class ChatPicker:
    """Lấy tin nhắn ShareGPT có số token gần mục tiêu nhất, không lặp lại."""

    def __init__(self, rows, rng):
        self.by_len = {}
        for r in rows:
            self.by_len.setdefault(r["tokens"], []).append(r["text"])
        for v in self.by_len.values():
            rng.shuffle(v)
        self.lengths = sorted(self.by_len)

    def pick(self, target):
        i = bisect.bisect_left(self.lengths, target)
        cands = sorted(self.lengths[max(0, i - 40): i + 40], key=lambda n: abs(n - target))
        for n in cands:
            if self.by_len[n]:
                return self.by_len[n].pop(), n
        raise RuntimeError(f"hết tin nhắn quanh {target} token — tăng kho hoặc giảm k")


class DocPicker:
    """Ghép tài liệu RAGBench tới đúng số token; tài liệu mở đầu mỗi tin nhắn không lặp lại."""

    def __init__(self, rows, R, rng):
        self.rows, self.R, self.rng, self.next = rows, R, rng, 0
        self.header = R.ntok(DOC_HEADER)

    def pick(self, target):
        if self.next >= len(self.rows):
            raise RuntimeError("hết tài liệu mở đầu — tăng kho hoặc giảm k")
        parts, total = [self.rows[self.next]], self.rows[self.next]["tokens"]
        self.next += 1
        while total < target:
            d = self.rows[self.rng.randrange(len(self.rows))]
            parts.append(d)
            total += d["tokens"] + 2
        body = "\n\n".join(d["text"] for d in parts)
        ids = self.R.tok(body, add_special_tokens=False).input_ids[:max(1, target - self.header)]
        text = DOC_HEADER + self.R.tok.decode(ids)
        return text, self.R.ntok(text)


# ---------- trace ----------

def conversation_sessions():
    df = load_trace()
    day0 = int(df.ts.min() // 86400)
    c = df[(df.logtype == "Conversation log") & (df.resp > 0)].sort_values(["sid", "ts"], kind="stable")
    sessions = []
    for sid, g in c.groupby("sid", sort=False):
        sessions.append({"sid": sid, "ts": g.ts.to_numpy(), "elapsed": g.elapsed.to_numpy(),
                         "req": g.req.to_numpy(), "resp": g.resp.to_numpy()})
    n_days = int(df.ts.max() // 86400) - day0 + 1
    return sessions, day0, n_days


def hour_profile(sessions, n_days):
    """req trung bình / giờ theo giờ trong ngày, và 3 khung giờ chọn theo quy tắc."""
    counts = np.zeros(24)
    for s in sessions:
        np.add.at(counts, ((s["ts"] // 3600) % 24).astype(int), 1)
    mean = counts / n_days
    peak = int(np.argmax(mean))
    far = [h for h in range(24) if min(abs(h - peak), 24 - abs(h - peak)) >= 4]
    peak2 = max(far, key=lambda h: mean[h])
    normal = int(np.argmin(np.abs(mean - np.median(mean))))
    quiet = int(np.argmin(mean))
    return mean, {"peak": peak, "peak2": peak2, "normal": normal, "quiet": quiet}


def positive_deltas(sessions):
    out = []
    for s in sessions:
        d = s["req"][1:] - (s["req"][:-1] + s["resp"][:-1])
        out.extend(int(x) for x in d if x > 0)
    return np.array(out)


# ---------- kịch bản ----------

def build_scenario(R, segment, k, seed):
    sessions, day0, n_days = conversation_sessions()
    mean, seg = hour_profile(sessions, n_days)
    if k > n_days:
        sys.exit(f"k = {k} > {n_days} ngày có trong trace")
    hour = seg[segment]
    days = np.random.default_rng(seed).permutation(n_days)[:k]
    windows = {int(d): (day0 + int(d)) * 86400 + hour * 3600 - WARMUP_S for d in days}
    span = WARMUP_S + MEASURE_S

    # Lấy mọi phiên có lượt nằm trong cửa sổ, kể cả phiên bắt đầu trước đó (người đang chat dở).
    # Phiên bắt đầu trước: lượt đầu trong cửa sổ gửi kèm đủ độ dài lịch sử theo trace (resumed).
    chosen = []
    for s in sessions:
        for j, t in enumerate(s["ts"]):
            d = int(t // 86400) - day0
            if d in windows and windows[d] <= t < windows[d] + span:
                chosen.append((t - windows[d], d, j, s))
                break
    chosen.sort(key=lambda x: (x[0], x[3]["sid"]))

    rng = random.Random(seed * 1000 + 7)
    chat = ChatPicker(read_jsonl(POOL / "chat.jsonl"), rng)
    docs = DocPicker(read_jsonl(POOL / "docs.jsonl"), R, rng)
    deltas = positive_deltas(sessions)
    ovh_first = R.ntok(R.render([{"role": "user", "content": ""}]))
    ovh_pair = R.ntok(R.render([{"role": "user", "content": ""}, {"role": "assistant", "content": ""},
                                {"role": "user", "content": ""}])) - ovh_first

    out, st = [], {"capped": 0, "doc_msgs": 0, "sampled_delta": 0, "resumed": 0}
    for start, d, first, s in chosen:
        turns, w0 = [], windows[d]
        st["resumed"] += first > 0
        for j in range(first, len(s["ts"])):
            if s["ts"][j] >= w0 + span:
                break
            req, resp = int(s["req"][j]), int(s["resp"][j])
            if j == first:  # lượt đầu, hoặc lượt đầu trong cửa sổ của phiên đang dở: cả lịch sử thành một tin nhắn
                target, think = req - ovh_first, 0.0
            else:
                target = req - int(s["req"][j - 1] + s["resp"][j - 1]) - ovh_pair
                think = max(0.0, float(s["ts"][j] - (s["ts"][j - 1] + s["elapsed"][j - 1])))
                if target < 1:  # client gốc đã cắt lịch sử → không suy ra được độ dài tin nhắn mới
                    target = int(deltas[rng.randrange(len(deltas))])
                    st["sampled_delta"] += 1
            max_tokens = max(1, min(resp, MAX_OUTPUT))
            cap = MAX_MODEL_LEN - max_tokens - 16
            target = max(1, min(target, cap - ovh_first))
            if target >= DOC_MIN:
                text, n = docs.pick(target)
                st["doc_msgs"] += 1
            else:
                text, n = chat.pick(target)
            budget = min(req, cap)
            st["capped"] += budget < req
            turns.append({"think_s": round(think, 3), "user": text, "user_tokens": n,
                          "prompt_budget": max(budget, n + ovh_first), "max_tokens": max_tokens,
                          "trace_t": round(float(s["ts"][j] - w0), 3), "trace_req": req})
        out.append({"sid": s["sid"], "day": d, "start_s": round(float(start), 3), "resumed": bool(first > 0),
                    "turns": turns})

    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{segment}-k{k}-s{seed}"
    path = OUT / f"{name}.jsonl"
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize(out, segment, hour, k, seed, days, st)
    summary["req_per_hour_expected"] = round(float(mean[hour]) * k, 1)  # k × trung bình trace ở giờ này
    summary["sha256"] = sha256(path)
    (OUT / f"{name}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize(out, segment, hour, k, seed, days, st):
    turns = [t for s in out for t in s["turns"]]
    measured = [t for t in turns if WARMUP_S <= t["trace_t"] < WARMUP_S + MEASURE_S]
    budgets = np.array([t["prompt_budget"] for t in measured]) if measured else np.zeros(1)
    bins = np.histogram(budgets, LEN_BINS)[0] / max(1, len(budgets)) * 100
    # số phiên đang mở theo trace (từ lượt đầu tới lượt cuối + 10 phút), lấy mẫu mỗi phút trong cửa sổ đo
    spans = [(s["start_s"], s["turns"][-1]["trace_t"] + 600) for s in out if s["turns"]]
    active = [sum(a <= t < b for a, b in spans) for t in range(WARMUP_S, WARMUP_S + MEASURE_S, 60)]
    return {
        "segment": segment, "hour_of_day": hour, "k_days": k, "seed": seed, "days": [int(d) for d in days],
        "window": {"warmup_s": WARMUP_S, "measure_s": MEASURE_S},
        "sessions": len(out), "requests_total": len(turns), "requests_measured": len(measured),
        "req_per_hour_measured": round(len(measured) * 3600 / MEASURE_S, 1),
        "active_sessions_p50_max": [int(np.median(active)), int(max(active))] if active else [0, 0],
        "prompt_budget_bins_pct": dict(zip(LEN_LABELS, np.round(bins, 1).tolist())),
        "prompt_budget_p50_p90_p99": [int(np.percentile(budgets, q)) for q in (50, 90, 99)],
        "max_tokens_p50_p90": [int(np.percentile([t["max_tokens"] for t in turns], q)) for q in (50, 90)] if turns else [0, 0],
        "doc_messages": st["doc_msgs"], "capped_by_max_model_len": int(st["capped"]),
        "follow_up_len_sampled": st["sampled_delta"], "resumed_sessions": st["resumed"],
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool")
    sub.add_parser("profile")
    sc = sub.add_parser("scenario")
    sc.add_argument("--segment", choices=["peak", "peak2", "normal"], required=True)
    sc.add_argument("--k", type=int, required=True, help="số ngày chồng lên nhau (≈ số lần số người dùng)")
    sc.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.cmd == "profile":
        sessions, _, n_days = conversation_sessions()
        mean, seg = hour_profile(sessions, n_days)
        print("req TB/giờ theo giờ trong ngày:", {h: round(float(m), 1) for h, m in enumerate(mean)})
        print("khung giờ:", seg)
        return
    R = Renderer()
    if args.cmd == "pool":
        build_pool(R)
    else:
        build_scenario(R, args.segment, args.k, args.seed)


if __name__ == "__main__":
    main()
