"""Chạy đo chất lượng P06 trên server vLLM (máy thuê), chạy tiếp được (docs/thi-nghiem-chay-tiep.md).

    python quality/run.py --config quality/configs/p06.yaml --out /root/results/p06          # chạy / chạy tiếp
    python quality/run.py --config ... --out ... --status                                   # xem tiến độ

Ba loại đơn vị (cấu hình `units`):
- gen:    sinh greedy cho một bộ prompt (reference / finqa_rag) ở mức đồng thời cố định; lưu token id + top-k logprob từng
          vị trí. Chia khúc 100 prompt, mỗi khúc một file (ghi nguyên tử) → dừng giữa chừng mất tối đa 1 khúc.
- tf:     teacher forcing: gửi prompt + output của một đơn vị gen làm prompt, lấy `prompt_logprobs` cho phần output
          (chỉ bộ prompt ngắn, cờ `tf`: prompt dài làm vLLM tính logits cho mọi vị trí prompt → tốn bộ nhớ, xem plan P06).
- lmeval: lm-evaluation-harness 0.4.13 (quality/lmeval_run.py) với `--use_cache` → chạy lại chỉ hỏi câu còn thiếu.
Trước mỗi đơn vị gen/tf: /reset_prefix_cache → mọi lần chạy bắt đầu cùng trạng thái cache (KV lấy từ cache và KV tính
lại có thể khác nhau ở bit cuối, làm lệch token). Không cần warmup cho chất lượng.
Mỗi đơn vị có done.json chứa hash cấu hình (đổi cấu hình → đo lại).
"""

import argparse
import asyncio
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import common as C  # noqa: E402
import sweep as SW  # noqa: E402

CHUNK = 100


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_set(cfg, name, only_tf=False):
    path = C.resolve(cfg["sets"][name])
    rows = [json.loads(x) for x in open(path) if x.strip()]
    if only_tf:
        rows = [r for r in rows if r.get("tf")]
    return path, rows


def unit_spec(cfg, u):
    spec = {k: v for k, v in u.items()}
    spec.update({"server_start": cfg["server"]["start"], "software": cfg["software"],
                 "sampling": {"temperature": 0}})
    if u["kind"] in ("gen", "tf"):
        spec["set_sha256"] = C.sha256_file(C.resolve(cfg["sets"][u["set"]]))
    if u["kind"] == "lmeval":
        spec["lmeval"] = cfg["lmeval"]
        if u.get("samples"):
            spec["samples_sha256"] = C.sha256_file(C.resolve(u["samples"]))
    return spec


# ---------- gen / tf qua HTTP ----------

def parse_tid(s):
    return int(s.split(":", 1)[1]) if isinstance(s, str) and s.startswith("token_id:") else s


async def post(session, url, payload):
    for attempt in range(5):
        try:
            async with session.post(url, json=payload) as resp:
                body = await resp.text()
                if resp.status == 200:
                    return json.loads(body)
                err = f"HTTP {resp.status}: {body[:300]}"
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(err)


def gen_payload(cfg, u, row):
    p = {"model": cfg["server"]["model"], "prompt": row["prompt"], "max_tokens": int(u.get("max_tokens", 256)),
         "temperature": 0, "return_token_ids": True}
    if u.get("logprobs"):
        p.update({"logprobs": int(u["logprobs"]), "return_tokens_as_token_ids": True})
    return p


def gen_record(row, out, t):
    ch = out["choices"][0]
    rec = {"idx": row.get("idx", row.get("id")), "id": row["id"], "text": ch["text"],
           "finish_reason": ch.get("finish_reason"), "prompt_ids": ch.get("prompt_token_ids"),
           "out_ids": ch.get("token_ids"), "latency_s": round(t, 3), "usage": out.get("usage")}
    lp = ch.get("logprobs")
    if lp:
        rec["lp"] = [None if x is None else round(x, 6) for x in lp["token_logprobs"]]
        rec["top"] = [[[parse_tid(k), round(v, 6)] for k, v in (d or {}).items()] for d in lp["top_logprobs"]]
    return rec


def tf_payload(cfg, u, src):
    ids = src["prompt_ids"] + src["out_ids"]
    return {"model": cfg["server"]["model"], "prompt": ids, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": int(u.get("prompt_logprobs", 20))}


def tf_record(src, out, t):
    pl = out["choices"][0]["prompt_logprobs"]
    n0 = len(src["prompt_ids"])
    tgt = src["out_ids"]
    lp, top = [], []
    for j, d in enumerate(pl[n0:n0 + len(tgt)]):
        d = {int(k): v for k, v in (d or {}).items()}
        lp.append(round(d[tgt[j]]["logprob"], 6) if tgt[j] in d else None)
        top.append([[k, round(v["logprob"], 6)] for k, v in sorted(d.items(), key=lambda kv: kv[1]["rank"])
                    if v.get("rank", 99) <= 1000][:21])
    return {"idx": src["idx"], "id": src["id"], "lp": lp, "top": top, "latency_s": round(t, 3)}


async def run_chunk(cfg, u, items, make_payload, make_record):
    import aiohttp
    url = cfg["server"]["base_url"].rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(int(u["concurrency"]))
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=3600)) as session:
        async def one(item):
            async with sem:
                t0 = time.perf_counter()
                out = await post(session, url, make_payload(cfg, u, item))
                return make_record(item, out, time.perf_counter() - t0)
        return await asyncio.gather(*[one(x) for x in items])


def write_gz(path, obj):
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class QRunner:
    def __init__(self, cfg, out, machine):
        self.cfg, self.out, self.machine = cfg, out, machine
        self.units = out / "units"
        self.units.mkdir(parents=True, exist_ok=True)
        self.server = SW.Server(cfg, out)

    def unit_dir(self, u):
        return self.units / u["id"]

    def is_done(self, u):
        return C.is_done(self.unit_dir(u), C.sha256_obj(unit_spec(self.cfg, u)))

    def prepare(self, d, spec_hash):
        """Thư mục có từ lần chạy trước với cấu hình khác → xoá. Cùng cấu hình (spec.json khớp) → giữ khúc đã xong."""
        sp = d / "spec_sha256"
        if d.exists() and (not sp.exists() or sp.read_text().strip() != spec_hash):
            log(f"{d.name}: thư mục cũ khác cấu hình → xoá")
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        sp.write_text(spec_hash + "\n")

    def run(self):
        for u in self.cfg["units"]:
            if self.is_done(u):
                continue
            for dep in [u.get("source")] if u.get("source") else []:
                du = next(x for x in self.cfg["units"] if x["id"] == dep)
                if not self.is_done(du):
                    raise RuntimeError(f"{u['id']} cần {dep} xong trước (xếp {dep} lên trước trong cấu hình)")
            self.run_unit(u)
        log("xong toàn bộ")

    def run_unit(self, u):
        spec = unit_spec(self.cfg, u)
        h = C.sha256_obj(spec)
        d = self.unit_dir(u)
        self.prepare(d, h)
        self.server.ensure()
        srv = self.server.current
        t_begin = time.time()
        log(f"{u['id']}: bắt đầu ({u['kind']})")
        if u["kind"] in ("gen", "tf"):
            self.server.reset_prefix_cache()
            n = self.run_http(u, d)
        elif u["kind"] == "lmeval":
            n = self.run_lmeval(u, d)
        else:
            raise ValueError(u["kind"])
        C.write_json(d / "meta.json", {"unit": u["id"], "spec": spec, "spec_sha256": h, "items": n,
                                       "unit_s": round(time.time() - t_begin, 1), "machine": self.machine,
                                       "hostname": socket.gethostname(), "server": srv,
                                       "finished": SW.now_iso(), "git_commit": SW.git_commit()})
        C.write_json(d / "done.json", {"spec_sha256": h, "finished": SW.now_iso()})
        log(f"{u['id']}: xong {n} mục sau {time.time() - t_begin:.0f} s")

    def run_http(self, u, d):
        if u["kind"] == "gen":
            _, rows = load_set(self.cfg, u["set"])
            items, mk_p, mk_r = rows, gen_payload, gen_record
        else:
            _, rows = load_set(self.cfg, u["set"], only_tf=True)
            want = {r["idx"] for r in rows}
            src = []
            for f in sorted((self.units / u["source"] / "chunks").glob("*.json.gz")):
                for rec in C.read_json(f):
                    if rec["idx"] in want:
                        src.append(rec)
            # prompt id = đúng chuỗi id server đã tokenize ở đơn vị gen nguồn (return_token_ids)
            missing = [s["idx"] for s in src if not s["prompt_ids"]]
            if missing:
                raise RuntimeError(f"{u['id']}: thiếu prompt_ids cho {len(missing)} mục của {u['source']}")
            items, mk_p, mk_r = sorted(src, key=lambda s: s["idx"]), tf_payload, tf_record
        (d / "chunks").mkdir(exist_ok=True)
        total = 0
        for c0 in range(0, len(items), CHUNK):
            f = d / "chunks" / f"{c0 // CHUNK:05d}.json.gz"
            part = items[c0:c0 + CHUNK]
            total += len(part)
            if f.exists():
                continue
            t0 = time.time()
            recs = asyncio.run(run_chunk(self.cfg, u, part, mk_p, mk_r))
            write_gz(f, recs)
            log(f"{u['id']}: khúc {c0 // CHUNK + 1}/{(len(items) + CHUNK - 1) // CHUNK} xong sau {time.time() - t0:.0f} s")
        return total

    def run_lmeval(self, u, d):
        L = self.cfg["lmeval"]
        s = self.cfg["server"]
        margs = (f"model={s['model']},base_url={s['base_url'].rstrip('/')}/v1/completions,"
                 f"tokenizer={C.resolve(self.cfg['client']['tokenizer'])},num_concurrent={u['concurrency']},"
                 f"max_retries=5,tokenized_requests=False,max_length={L['max_length']},timeout=3600,"
                 f"max_gen_toks={L['max_gen_toks']}")  # mặc định 256 của model API cắt lời giải GSM8K (task không tự đặt)
        cmd = [L["python"], str(ROOT / "quality/lmeval_run.py"), "run", "--model", "local-completions-nothink",
               "--model_args", margs, "--tasks", u["tasks"], "--apply_chat_template",
               "--use_cache", str(d / "cache"), "--output_path", str(d / "out"), "--log_samples",
               "--seed", str(L.get("seed", 1234))]
        if u.get("samples"):
            cmd += ["--samples", C.resolve(u["samples"]).read_text().strip()]
        if u.get("limit"):
            cmd += ["--limit", str(u["limit"])]
        with open(d / "lmeval.log", "a") as lf:
            lf.write(f"\n### {SW.now_iso()} {' '.join(cmd[:12])} ...\n")
            lf.flush()
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        if p.returncode != 0:
            raise RuntimeError(f"{u['id']}: lm-eval lỗi (mã {p.returncode}), xem {d / 'lmeval.log'}")
        res = sorted((d / "out").rglob("results_*.json"))
        if not res:
            raise RuntimeError(f"{u['id']}: không thấy file results_*.json")
        return sum(1 for _ in (d / "out").rglob("samples_*.jsonl"))


def status(runner):
    cfg = runner.cfg
    for u in cfg["units"]:
        d = runner.unit_dir(u)
        if runner.is_done(u):
            m = C.read_json(d / "meta.json")
            st = f"xong ({m['items']} mục, {m['unit_s'] / 60:.1f} phút)"
        elif d.exists():
            nchunk = len(list((d / "chunks").glob("*.json.gz"))) if (d / "chunks").exists() else 0
            st = f"dở dang ({nchunk} khúc)" if u["kind"] != "lmeval" else "dở dang (lm-eval có cache)"
        else:
            st = "chưa chạy"
        print(f"{u['id']:<22} {u['kind']:<7} {st}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--machine", default=os.environ.get("BENCH_MACHINE", ""))
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = QRunner(cfg, out, a.machine)
    if a.status:
        status(runner)
        return
    runner.run()


if __name__ == "__main__":
    main()
