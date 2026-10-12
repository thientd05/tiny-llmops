"""Client gửi tải tự viết: phát lại phiên người dùng W4 (P05) và chế độ closed-loop để kiểm chứng với vllm bench serve.

    # W4: phát lại kịch bản (workloads/build_w4.py), partly-open
    python bench/session_client.py scenario --scenario data/workloads/w4/peak-k10-s1.jsonl \
        --tokenizer /root/models/Qwen3-30B-A3B-GPTQ-Int4 --base-url http://localhost:8000 \
        --model Qwen/Qwen3-30B-A3B-GPTQ-Int4 --out x.json.gz --ignore-eos --temperature 0

    # closed-loop N user trên file workload P02 (cùng mẫu, cùng thứ tự với vllm bench serve --seed S)
    python bench/session_client.py closed --dataset data/workloads/w2.jsonl --concurrency 16 --num-requests 100 \
        --seed 1001 --base-url ... --model ... --out x.json.gz --ignore-eos --temperature 0

Cách đo giống hệt vllm/benchmarks (bản 0.30.0): gửi /v1/completions có stream; TTFT = từ lúc gửi tới chunk đầu có
`choices`; latency = tới chunk cuối có `choices`; số token ra lấy từ `usage`; TPOT = (latency − TTFT) / (out − 1).
File ra dùng cùng khoá với `vllm bench serve --save-detailed` (ttfts, latencies, output_lens, input_lens, start_times,
itls, errors) để bench/common.py đọc chung; chế độ scenario thêm thông tin phiên.

Kịch bản (scenario): lượt đầu của mỗi phiên gửi đúng `start_s`; lượt sau gửi khi lượt trước NHẬN XONG + `think_s`;
prompt = lịch sử thật (gồm câu trả lời server vừa sinh) + tin nhắn mới qua workloads/w4_session.assemble.
Chỉ request gửi trong [warmup_s, warmup_s + measure_s) được đánh dấu `in_window`; hết cửa sổ thì không gửi lượt mới,
các request đang chạy được chờ xong. `--time-scale` < 1 chỉ dùng để thử ở local với server giả (co mọi mốc thời gian).
"""

import argparse
import asyncio
import codecs
import gzip
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workloads"))


class SSE:
    """Tách luồng SSE thành từng message (giống StreamedResponseHandler của vLLM)."""

    def __init__(self):
        self.buf = ""
        self.dec = codecs.getincrementaldecoder("utf-8")()

    def add(self, b):
        self.buf += self.dec.decode(b)
        msgs = []
        while "\n\n" in self.buf:
            m, self.buf = self.buf.split("\n\n", 1)
            if m.strip():
                msgs.append(m.strip())
        if self.buf.startswith("data: "):
            c = self.buf.removeprefix("data: ").strip()
            if c == "[DONE]":
                msgs.append(self.buf.strip())
                self.buf = ""
            elif c:
                try:
                    json.loads(c)
                    msgs.append(self.buf.strip())
                    self.buf = ""
                except json.JSONDecodeError:
                    pass
        return msgs


async def send(session, url, payload):
    """Một request stream. Trả về dict: start, ttft, latency, itl, text, out, prompt_len, error."""
    r = {"start": None, "ttft": 0.0, "latency": 0.0, "itl": [], "text": "", "out": None, "prompt_len": None,
         "error": ""}
    st = time.perf_counter()
    r["start"] = st
    last = st
    first = False
    try:
        async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
            if resp.status != 200:
                r["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return r
            h = SSE()
            async for chunk in resp.content.iter_any():
                chunk = chunk.strip()
                if not chunk:
                    continue
                for msg in h.add(chunk):
                    if msg.startswith(":"):
                        continue
                    data = msg.removeprefix("data: ")
                    if data == "[DONE]":
                        continue
                    d = json.loads(data)
                    if choices := d.get("choices"):
                        t = time.perf_counter()
                        if not first:
                            first = True
                            r["ttft"] = t - st
                        else:
                            r["itl"].append(t - last)
                        last = t
                        r["text"] += choices[0].get("text") or ""
                    elif usage := d.get("usage"):
                        r["out"] = usage.get("completion_tokens")
                        r["prompt_len"] = usage.get("prompt_tokens")
            if not first:
                r["error"] = "không nhận được chunk nào có token"
            r["latency"] = last - st
    except Exception:
        r["error"] = traceback.format_exc()[-1000:]
    return r


def payload(args, prompt, max_tokens):
    p = {"model": args.model, "prompt": prompt, "repetition_penalty": 1.0, "max_tokens": int(max_tokens),
         "logprobs": None, "stream": True, "stream_options": {"include_usage": True}}
    if args.ignore_eos:
        p["ignore_eos"] = True
    if args.temperature is not None:
        p["temperature"] = args.temperature
    return p


def make_session(limit):
    conn = aiohttp.TCPConnector(limit=limit, limit_per_host=limit, ttl_dns_cache=300, use_dns_cache=True,
                                keepalive_timeout=60, enable_cleanup_closed=True, force_close=False)
    return aiohttp.ClientSession(connector=conn, trust_env=True, timeout=aiohttp.ClientTimeout(total=6 * 3600))


def result_dict(recs, t0, extra=None):
    """Cùng khoá với vllm bench serve --save-detailed; start_times tính từ perf_counter như vLLM."""
    out = {
        "ttfts": [r["ttft"] for r in recs],
        "latencies": [r["latency"] for r in recs],
        "output_lens": [r["out"] if r["out"] is not None else 0 for r in recs],
        "input_lens": [r["prompt_len"] for r in recs],
        "start_times": [r["start"] for r in recs],
        "itls": [[round(x, 6) for x in r["itl"]] for r in recs],
        "errors": [r["error"] for r in recs],
        "generated_texts": [r["text"][:200] for r in recs],
        "t0": t0,
    }
    out["completed"] = sum(not e for e in out["errors"])
    out["failed"] = len(recs) - out["completed"]
    if extra:
        out.update(extra)
    return out


def write_gz(path, obj):
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ---------- closed-loop ----------

def sample_like_vllm(path, n, seed):
    """Chọn mẫu giống vllm/benchmarks/datasets CustomDataset: đọc theo thứ tự file, random.seed(seed), shuffle, lấy n đầu."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    random.seed(seed)
    random.shuffle(rows)
    if n > len(rows):
        sys.exit(f"cần {n} mẫu, file chỉ có {len(rows)} (không lặp mẫu)")
    return rows[:n]


async def run_closed(args):
    rows = sample_like_vllm(args.dataset, args.num_requests, args.seed)
    url = args.base_url.rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(args.concurrency)
    async with make_session(args.concurrency) as session:
        async def one(row):
            async with sem:
                return await send(session, url, payload(args, row["prompt"], row["output_tokens"]))
        t0 = time.perf_counter()
        recs = await asyncio.gather(*[asyncio.create_task(one(r)) for r in rows])
    res = result_dict(recs, t0, {"ids": [r.get("id") for r in rows], "mode": "closed",
                                 "concurrency": args.concurrency, "seed": args.seed})
    res["duration"] = max(r["start"] + r["latency"] for r in recs) - t0
    write_gz(args.out, res)
    print(f"closed: {res['completed']} xong, {res['failed']} lỗi, {res['duration']:.1f} s")


# ---------- kịch bản W4 ----------

async def run_scenario(args):
    from transformers import AutoTokenizer
    from w4_session import assemble

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
    count = lambda msgs: len(tok(render(msgs), add_special_tokens=False).input_ids)

    sessions = [json.loads(line) for line in open(args.scenario) if line.strip()]
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
    summ = Path(str(args.scenario).removesuffix(".jsonl") + ".summary.json")
    win = json.loads(summ.read_text())["window"] if summ.exists() else {"warmup_s": 600, "measure_s": 1800}
    warm, meas = args.warmup_s if args.warmup_s is not None else win["warmup_s"], \
        args.measure_s if args.measure_s is not None else win["measure_s"]
    sc = args.time_scale
    end_s = warm + meas
    url = args.base_url.rstrip("/") + "/v1/completions"
    loop = asyncio.get_running_loop()
    recs = []

    async with make_session(0) as session:
        t0 = time.perf_counter()

        async def at(t_rel):
            await asyncio.sleep(max(0.0, t0 + t_rel * sc - time.perf_counter()))

        async def run_session(s):
            await at(s["start_s"])
            history = []
            for j, turn in enumerate(s["turns"]):
                if j > 0:
                    await asyncio.sleep(turn["think_s"] * sc)
                t_rel = (time.perf_counter() - t0) / sc
                if t_rel >= end_s:
                    break  # hết cửa sổ: không gửi lượt mới
                msgs, dropped = await loop.run_in_executor(
                    None, assemble, history, turn["user"], turn["prompt_budget"], count)
                prompt = await loop.run_in_executor(None, render, msgs)
                r = await send(session, url, payload(args, prompt, turn["max_tokens"]))
                r.update({"sid": s["sid"], "turn": j, "resumed": bool(s.get("resumed")) and j == 0,
                          "t_rel": round(t_rel, 3), "in_window": warm <= t_rel < end_s, "dropped": dropped,
                          "budget": turn["prompt_budget"], "max_tokens": turn["max_tokens"]})
                recs.append(r)
                if r["error"]:
                    break  # lượt lỗi: dừng phiên (không có câu trả lời để làm lịch sử)
                history.append((turn["user"], r["text"]))

        await asyncio.gather(*[asyncio.create_task(run_session(s)) for s in sessions])

    recs.sort(key=lambda r: r["start"])
    extra = {k: [r[k] for r in recs] for k in ("sid", "turn", "resumed", "t_rel", "in_window", "dropped", "budget",
                                                "max_tokens")}
    extra.update({"mode": "scenario", "scenario": str(args.scenario), "window": {"warmup_s": warm, "measure_s": meas},
                  "time_scale": sc})
    res = result_dict(recs, t0, extra)
    write_gz(args.out, res)
    n_in = sum(extra["in_window"])
    print(f"scenario: {len(recs)} request ({n_in} trong cửa sổ đo), {res['failed']} lỗi")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("closed", "scenario"):
        p = sub.add_parser(name)
        p.add_argument("--base-url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--ignore-eos", action="store_true")
        p.add_argument("--temperature", type=float, default=None)
        if name == "closed":
            p.add_argument("--dataset", required=True)
            p.add_argument("--concurrency", type=int, required=True)
            p.add_argument("--num-requests", type=int, required=True)
            p.add_argument("--seed", type=int, required=True)
        else:
            p.add_argument("--scenario", required=True)
            p.add_argument("--tokenizer", required=True)
            p.add_argument("--warmup-s", type=float, default=None)
            p.add_argument("--measure-s", type=float, default=None)
            p.add_argument("--time-scale", type=float, default=1.0, help="chỉ để thử ở local; đo thật luôn = 1")
            p.add_argument("--max-sessions", type=int, default=0, help="chỉ để thử: lấy N phiên đầu")
    args = ap.parse_args()
    asyncio.run(run_closed(args) if args.mode == "closed" else run_scenario(args))


if __name__ == "__main__":
    main()
