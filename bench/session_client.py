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
- `--days d1,d2,...`: chỉ phát các phiên của những ngày này. Kịch bản k ngày = kịch bản gốc k=110 lọc theo k ngày đầu của
  hoán vị (summary `days`), nên kịch bản k nằm trọn trong kịch bản k' > k, kể cả nội dung tin nhắn (P05).
- Có `--slo-ttft-ms` / `--slo-tpot-ms`:
  - chờ request còn dở sau cửa sổ tối đa `drain` = TTFT SLO + max_tokens lớn nhất × TPOT SLO + 30 s. Request đạt SLO
    chắc chắn xong trước mốc này; request bị cắt chắc chắn đã trượt SLO → ghi lỗi (tính như vô cùng, giống request lỗi).
  - `--abort-certain-fail`: dừng sớm khi số request trong cửa sổ **chắc chắn** trượt (TTFT hoặc TPOT) vượt
    5% × (số lượt tối đa của kịch bản) + 2: khi đó p95 của lần chạy chắc chắn vượt SLO, chạy tiếp chỉ tốn tiền GPU.
    Request chắc chắn trượt TTFT = đã xong với TTFT > SLO, hoặc chưa có token đầu mà đã chờ quá SLO.
- Ghi thêm mỗi request: `sched` (thời điểm lẽ ra gửi, thời gian trace), `lag` (gửi trễ bao nhiêu giây so với lịch,
  gồm thời gian ghép prompt), `cached` (token prompt trúng prefix cache, cần server bật --enable-prompt-tokens-details).
"""

import argparse
import asyncio
import codecs
import gzip
import json
import math
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


def new_rec():
    return {"start": None, "ttft": 0.0, "latency": 0.0, "itl": [], "text": "", "out": None, "prompt_len": None,
            "cached": None, "error": "", "first_at": None, "chunks": 0}


async def send(session, url, payload, r=None):
    """Một request stream. Điền vào r (tạo mới nếu không có): start, ttft, latency, itl, text, out, prompt_len, cached,
    error; first_at / chunks cập nhật trong lúc chạy để theo dõi request còn dở."""
    r = new_rec() if r is None else r
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
                            r["first_at"] = t
                        else:
                            r["itl"].append(t - last)
                        last = t
                        r["chunks"] += 1
                        r["text"] += choices[0].get("text") or ""
                    elif usage := d.get("usage"):
                        r["out"] = usage.get("completion_tokens")
                        r["prompt_len"] = usage.get("prompt_tokens")
                        r["cached"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if not first:
                r["error"] = "không nhận được chunk nào có token"
            r["latency"] = last - st
    except asyncio.CancelledError:
        r["error"] = r["error"] or f"bị huỷ: {r.get('cancel_reason', 'cancel')}"
        r["latency"] = last - st
        raise
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
        "cached": [r.get("cached") for r in recs],
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

def certain_violations(recs, now, slo_ttft, slo_tpot):
    """Số request trong cửa sổ chắc chắn trượt TTFT / TPOT tính tới lúc `now` (giây thật, perf_counter)."""
    v_ttft = v_tpot = 0
    for r in recs:
        if not r["in_window"]:
            continue
        done = r.get("done", False)
        if r["error"] and done:
            v_ttft += 1
            v_tpot += 1
            continue
        if r["first_at"] is None:
            v_ttft += (now - r["start"]) > slo_ttft
        else:
            v_ttft += r["ttft"] > slo_ttft
            n = r["max_tokens"]
            if done:
                out = r["out"] or 0
                v_tpot += out > 1 and (r["latency"] - r["ttft"]) / (out - 1) > slo_tpot
            elif n > 1:
                # ignore_eos → request sinh đúng max_tokens; đã chạy quá slo_tpot × (n − 1) sau token đầu thì chắc trượt
                v_tpot += (now - r["first_at"]) > slo_tpot * (n - 1)
    return v_ttft, v_tpot


async def run_scenario(args):
    from transformers import AutoTokenizer
    from w4_session import assemble

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    render = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
    count = lambda msgs: len(tok(render(msgs), add_special_tokens=False).input_ids)

    sessions = [json.loads(line) for line in open(args.scenario) if line.strip()]
    if args.days:
        keep = {int(x) for x in args.days.split(",")}
        sessions = [s for s in sessions if s["day"] in keep]
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
    n_upper = sum(len(s["turns"]) for s in sessions)  # số request tối đa có thể gửi (mọi lượt của kịch bản)
    slo = args.slo_ttft_ms is not None and args.slo_tpot_ms is not None
    slo_ttft, slo_tpot = (args.slo_ttft_ms / 1000, args.slo_tpot_ms / 1000) if slo else (None, None)
    max_tok = max((t["max_tokens"] for s in sessions for t in s["turns"]), default=1)
    drain = slo_ttft + max_tok * slo_tpot + 30 * sc if slo else None  # giây thật
    abort_thr = math.ceil(0.05 * n_upper) + 2
    state = {"aborted": None, "drain_cut": 0}

    async with make_session(0) as session:
        t0 = time.perf_counter()

        async def at(t_rel):
            await asyncio.sleep(max(0.0, t0 + t_rel * sc - time.perf_counter()))

        async def run_session(s):
            await at(s["start_s"])
            history = []
            sched = s["start_s"]
            for j, turn in enumerate(s["turns"]):
                if j > 0:
                    sched = (time.perf_counter() - t0) / sc + turn["think_s"]
                    if sched >= end_s:
                        break  # lượt sau rơi ra ngoài cửa sổ: không chờ think time vô ích
                    await asyncio.sleep(turn["think_s"] * sc)
                t_rel = (time.perf_counter() - t0) / sc
                if t_rel >= end_s:
                    break  # hết cửa sổ: không gửi lượt mới
                msgs, dropped = await loop.run_in_executor(
                    None, assemble, history, turn["user"], turn["prompt_budget"], count)
                prompt = await loop.run_in_executor(None, render, msgs)
                r = new_rec()
                r.update({"sid": s["sid"], "day": s["day"], "turn": j, "resumed": bool(s.get("resumed")) and j == 0,
                          "t_rel": round(t_rel, 3), "sched": round(sched, 3), "in_window": warm <= t_rel < end_s,
                          "dropped": dropped, "budget": turn["prompt_budget"], "max_tokens": turn["max_tokens"],
                          "user_tokens": turn["user_tokens"], "think_s": turn["think_s"] if j > 0 else None})
                recs.append(r)
                await send(session, url, payload(args, prompt, turn["max_tokens"]), r)
                r["done"] = True
                r["lag"] = round((r["start"] - t0) / sc - sched, 4)
                if r["error"]:
                    break  # lượt lỗi: dừng phiên (không có câu trả lời để làm lịch sử)
                history.append((turn["user"], r["text"]))

        def progress(now, final=False):
            v_ttft, v_tpot = certain_violations(recs, now, slo_ttft, slo_tpot) if slo else (None, None)
            p = {"t_trace_s": round((now - t0) / sc, 1), "window_end_s": end_s, "sent": len(recs),
                 "sent_in_window": sum(r["in_window"] for r in recs),
                 "in_flight": sum(not r.get("done") for r in recs), "certain_fail_ttft": v_ttft,
                 "certain_fail_tpot": v_tpot, "abort_threshold": abort_thr, "n_upper": n_upper,
                 "aborted": state["aborted"], "final": final}
            if args.progress:
                write_json_atomic(args.progress, p)
            return p

        tasks = {asyncio.create_task(run_session(s)) for s in sessions}
        pending, last_prog = tasks, 0.0
        while pending:
            _, pending = await asyncio.wait(pending, timeout=max(0.05, min(5.0, 5.0 * sc * 10)))
            now = time.perf_counter()
            reason = None
            if slo and args.abort_certain_fail:
                v_ttft, v_tpot = certain_violations(recs, now, slo_ttft, slo_tpot)
                if max(v_ttft, v_tpot) >= abort_thr:
                    reason = (f"dừng sớm: {v_ttft} request chắc chắn trượt TTFT, {v_tpot} trượt TPOT "
                              f"≥ ngưỡng {abort_thr} (5% × {n_upper} lượt tối đa + 2)")
                    state["aborted"] = {"reason": reason, "t_trace_s": round((now - t0) / sc, 1),
                                        "certain_fail_ttft": v_ttft, "certain_fail_tpot": v_tpot,
                                        "threshold": abort_thr}
            if reason is None and drain is not None and now > t0 + end_s * sc + drain and pending:
                reason = f"hết thời gian chờ {drain:.0f} s sau cửa sổ"
                state["drain_cut"] = sum(not r.get("done") for r in recs)
            if reason:
                for r in recs:
                    if not r.get("done"):
                        r["cancel_reason"] = reason
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                pending = set()
                print(reason, flush=True)
            if now - last_prog > 30 * sc or not pending:
                last_prog = now
                progress(now, final=not pending)

    for r in recs:
        r.setdefault("lag", None)
        if not r.get("done"):  # bị huỷ (dừng sớm / hết thời gian chờ): không có câu trả lời đầy đủ
            r["error"] = r["error"] or f"bị huỷ: {r.get('cancel_reason')}"
    recs.sort(key=lambda r: r["start"])
    keys = ("sid", "day", "turn", "resumed", "t_rel", "sched", "lag", "in_window", "dropped", "budget", "max_tokens",
            "user_tokens", "think_s")
    extra = {k: [r[k] for r in recs] for k in keys}
    extra.update({"mode": "scenario", "scenario": str(args.scenario), "window": {"warmup_s": warm, "measure_s": meas},
                  "time_scale": sc, "days": args.days, "sessions": len(sessions), "n_upper": n_upper,
                  "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms}, "drain_s": drain,
                  "abort_certain_fail": bool(args.abort_certain_fail), "abort_threshold": abort_thr,
                  "aborted": state["aborted"], "drain_cut": state["drain_cut"]})
    res = result_dict(recs, t0, extra)
    write_gz(args.out, res)
    n_in = sum(extra["in_window"])
    print(f"scenario: {len(sessions)} phiên, {len(recs)} request ({n_in} trong cửa sổ đo), {res['failed']} lỗi"
          + (f"; {state['aborted']['reason']}" if state["aborted"] else "")
          + (f"; cắt {state['drain_cut']} request sau thời gian chờ" if state["drain_cut"] else ""))


def write_json_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


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
            p.add_argument("--days", default="", help="chỉ phát phiên của các ngày này (d1,d2,...)")
            p.add_argument("--slo-ttft-ms", type=float, default=None)
            p.add_argument("--slo-tpot-ms", type=float, default=None)
            p.add_argument("--abort-certain-fail", action="store_true",
                           help="dừng sớm khi p95 chắc chắn trượt SLO (cần --slo-*)")
            p.add_argument("--progress", default="", help="file JSON tiến độ, ghi mỗi 30 s")
    args = ap.parse_args()
    asyncio.run(run_closed(args) if args.mode == "closed" else run_scenario(args))


if __name__ == "__main__":
    main()
