"""Chạy loạt đo benchmark (workload × số user đồng thời × lần lặp) trên một server kiểu OpenAI, chạy tiếp được.

    python bench/sweep.py --config bench/configs/full.yaml --out results/gd1/04-do-baseline-day-du   # chạy / chạy tiếp
    python bench/sweep.py --config ... --out ... --status                                              # xem tiến độ

Hai chế độ (mode trong file cấu hình):
- grid:   danh sách điểm đo cho trước (smoke).
- search: tìm goodput theo docs/workloads.md "Cách tìm goodput": lưới gốc → nhân đôi / chia đôi → tìm nhị phân.
          Chạy theo đợt; điểm tiếp theo tính lại từ các done.json đã có (tất định) → dừng rồi chạy lại cùng lệnh là đi tiếp.

Mỗi đơn vị đo (ví dụ w2-c32-r2) nằm ở <out>/units/<id>/; done.json (chứa hash cấu hình) ghi cuối cùng.
Thư mục không có done.json hợp lệ = dở dang → xoá và đo lại. Trước mỗi lần đo: warmup → /reset_prefix_cache.
Trong lúc đo: /metrics và nvidia-smi mỗi 1 s. Không dùng --num-warmups của vllm bench serve vì nó gửi lại chính
request đầu tiên của tập đo (làm nóng prefix cache cho request đó).
"""

import argparse
import concurrent.futures as cf
import gzip
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

UNIT_RE = re.compile(r"^(?P<w>[a-z0-9]+)-c(?P<n>\d+)-r(?P<r>\d+)(?:-(?P<client>[a-z]+))?$")
POLL_METRICS = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                "vllm:prompt_tokens_total", "vllm:generation_tokens_total"]


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def http(url, data=None, timeout=30):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def parse_prom(text, names):
    """Cộng các series cùng tên (bỏ label) của những metric cần theo dõi."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in names:
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


# ---------- server ----------

class Server:
    """Khởi động / tắt server bằng lệnh trong cấu hình (scripts/serve_default.sh, hoặc server giả khi test)."""

    KV_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
    KVMEM_RE = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
    CONC_RE = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
    COMPILE_RE = re.compile(r"\(compilation: ([\d.]+) s\)")
    INIT_RE = re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s")
    LOADW_RE = re.compile(r"Loading weights took ([\d.]+) seconds")

    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, out
        self.base = cfg["server"]["base_url"].rstrip("/")
        self.starts = out / "server_starts.jsonl"
        self.current = None  # thông tin lần khởi động đang phục vụ

    def healthy(self):
        try:
            http(self.base + "/health", timeout=5)
            return True
        except Exception:
            return False

    def start(self, reason):
        (self.out / "server").mkdir(parents=True, exist_ok=True)
        k = (sum(1 for x in self.starts.read_text().splitlines() if x.strip()) if self.starts.exists() else 0) + 1
        logf = self.out / "server" / f"start-{k:03d}.log"
        cmd = self.cfg["server"]["start"].format(log=logf)
        log(f"server start ({reason}): {cmd}")
        t0 = time.time()
        p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        ready_s = time.time() - t0
        if p.returncode != 0 or not self.healthy():
            raise RuntimeError(f"server không khởi động được:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
        reused = "dùng lại" in p.stdout
        info = {"start_index": k, "time": now_iso(), "reason": reason, "cmd": cmd, "ready_s": round(ready_s, 1),
                "reused_running_server": reused, "log": str(logf.relative_to(self.out)) if logf.exists() else None}
        if logf.exists():
            text = logf.read_text(errors="replace")
            if m := self.KV_RE.search(text):
                info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
            if m := self.KVMEM_RE.search(text):
                info["kv_cache_gib"] = float(m.group(1))
            if m := self.CONC_RE.search(text):
                info["max_concurrency_at"] = [int(m.group(1).replace(",", "")), float(m.group(2))]
            if m := self.COMPILE_RE.search(text):
                info["torch_compile_s"] = float(m.group(1))
            if m := self.INIT_RE.search(text):
                info["init_engine_s"] = float(m.group(1))
            if m := self.LOADW_RE.search(text):
                info["load_weights_s"] = float(m.group(1))
        elif reused:  # server còn chạy từ lần chạy trước của sweep → lấy số KV của lần khởi động thật gần nhất
            prev = [json.loads(x) for x in self.starts.read_text().splitlines() if x.strip()] \
                if self.starts.exists() else []
            prev = [x for x in prev if not x.get("reused_running_server")]
            if prev:
                info.update({k2: prev[-1][k2] for k2 in ("kv_cache_tokens", "kv_cache_gib") if k2 in prev[-1]})
                info["same_process_as_start"] = prev[-1]["start_index"]
        with open(self.starts, "a") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        log(f"server ready sau {ready_s:.0f} s, KV cache = {info.get('kv_cache_tokens')} token"
            + (" (dùng lại server đang chạy)" if reused else ""))
        self.current = info
        # P03: lần khởi động compile torch.compile từ đầu (máy mới) cho KV cache nhỏ hơn ~23% (28,704 vs 37,264 token)
        # → không đo trên lần đó: tắt và bật lại (lần sau dùng cache compile). Ngưỡng: compile > recompile_restart_s.
        lim = self.cfg["server"].get("recompile_restart_s")
        if lim and info.get("torch_compile_s", 0) > lim and not reused:
            log(f"server vừa compile từ đầu ({info['torch_compile_s']:.0f} s > {lim} s) → khởi động lại trước khi đo")
            self.stop()
            return self.start("bỏ lần khởi động compile từ đầu")
        return info

    def stop(self):
        cmd = self.cfg["server"].get("stop")
        if cmd:
            log(f"server stop: {cmd}")
            subprocess.run(["bash", "-c", cmd], check=False)
        for _ in range(120):
            if not self.healthy():
                break
            time.sleep(1)
        self.current = None

    def restart(self, reason):
        self.stop()
        return self.start(reason)

    def ensure(self):
        if self.current is None or not self.healthy():
            return self.start("chưa chạy" if self.current is None else "server chết")
        return self.current

    def metrics_text(self):
        try:
            return http(self.base + "/metrics", timeout=10)
        except Exception as e:
            return f"# lỗi đọc /metrics: {e}\n"

    def reset_prefix_cache(self):
        """Trả về True khi server báo xoá xong. Endpoint báo success=false nếu còn block đang bị giữ → thử lại."""
        for _ in range(60):
            try:
                body = http(self.base + "/reset_prefix_cache", data={}, timeout=30)
                ok = json.loads(body).get("success", True) if body.strip().startswith("{") else True
                if ok:
                    return True
            except Exception as e:
                log(f"reset_prefix_cache lỗi: {e}")
            time.sleep(1)
        raise RuntimeError("không xoá được prefix cache sau 60 lần thử")


# ---------- ghi song song trong lúc đo ----------

class Pollers:
    def __init__(self, server, unit_dir, interval=1.0):
        self.server, self.dir, self.interval = server, unit_dir, interval
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._metrics_loop, daemon=True)
        self.smi = None

    def _metrics_loop(self):
        with open(self.dir / "metrics_timeseries.jsonl", "w") as f:
            t0 = time.time()
            while not self.stop_ev.is_set():
                t = time.time()
                vals = parse_prom(self.server.metrics_text(), set(POLL_METRICS))
                f.write(json.dumps({"t": round(t - t0, 3), **vals}) + "\n")
                f.flush()
                self.stop_ev.wait(max(0.0, self.interval - (time.time() - t)))

    def start(self):
        self.thread.start()
        if shutil.which("nvidia-smi"):
            q = ("timestamp,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,"
                 "clocks_event_reasons.active")
            self.smi_f = open(self.dir / "gpu.csv", "w")
            self.smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv", "-lms", "1000"],
                                        stdout=self.smi_f, stderr=subprocess.STDOUT)

    def stop(self):
        self.stop_ev.set()
        self.thread.join(timeout=10)
        if self.smi:
            self.smi.terminate()
            self.smi.wait(timeout=10)
            self.smi_f.close()


# ---------- runner ----------

class Runner:
    def __init__(self, cfg, out, machine, dry=False):
        self.cfg, self.out, self.machine, self.dry = cfg, out, machine, dry
        self.units = out / "units"
        self.units.mkdir(parents=True, exist_ok=True)
        self.server = Server(cfg, out)
        self.block = None  # lần lặp của khối đơn vị đang chạy (để khởi động lại server giữa các lần lặp)
        self.git = git_commit()

    # --- trạng thái từ đĩa ---
    def done_units(self):
        res = {}
        for d in self.units.iterdir():
            m = UNIT_RE.match(d.name)
            if not m or not (d / "done.json").exists():
                continue
            w, n, r, client = m["w"], int(m["n"]), int(m["r"]), m["client"] or "vllm"
            if w not in self.cfg["workloads"]:
                continue
            spec = C.unit_spec(self.cfg, w, n, r, client)
            if C.is_done(d, C.sha256_obj(spec)):
                res[(w, n, r, client)] = d
        return res

    def unit_metrics(self, d):
        return C.read_json(d / "metrics.json")

    def analyze(self, w, done=None):
        """Trạng thái tìm goodput của workload w, tính lại hoàn toàn từ các đơn vị đã xong."""
        done = self.done_units() if done is None else done
        R = self.cfg["repeats"]
        reps = {}
        for (w2, n, r, client), d in done.items():
            if w2 == w and client == "vllm":
                reps.setdefault(n, {})[r] = d
        levels, verdicts, counts = {}, {}, {}
        for n, rr in sorted(reps.items()):
            counts[n] = len(rr)
            if all(r in rr for r in range(1, 2 * R + 1)):
                use = list(range(1, 2 * R + 1))
            elif all(r in rr for r in range(1, R + 1)):
                use = list(range(1, R + 1))
            else:
                use = None
            lv = {"repeats_done": sorted(rr), "verdict": None}
            if use:
                ms = [self.unit_metrics(rr[r])[self.cfg["verdict_window"]] for r in use]
                med = C.median_metrics(ms)
                lv.update({"used_repeats": use, "median": med, "verdict": C.passes(med, self.cfg["slo"])})
                verdicts[n] = lv["verdict"]
            levels[n] = lv
        nsamp = self.cfg["workloads"][w]["samples"]
        status, need = C.search_step(verdicts, counts, self.cfg["search"], R, C.max_fit(self.cfg, nsamp))
        lo, hi = C.interval(verdicts)
        return {"levels": levels, "verdicts": verdicts, "status": status, "need": need, "lo": lo, "hi": hi}

    def plan_batch(self):
        """Các đơn vị còn phải đo của đợt hiện tại, xếp xen kẽ theo lần lặp: r1 mọi điểm, r2 mọi điểm, ..."""
        done = self.done_units()
        todo = []
        if self.cfg["mode"] == "grid":
            for p in self.cfg["points"]:
                for r in range(1, p.get("repeats", self.cfg["repeats"]) + 1):
                    todo.append((r, p["w"], p["n"], p.get("client", "vllm")))
        else:
            for w in self.cfg["workloads"]:
                for n, q in self.analyze(w, done)["need"].items():
                    for r in range(1, q + 1):
                        todo.append((r, w, n, "vllm"))
        todo = [u for u in todo if (u[1], u[2], u[0], u[3]) not in done]
        order = {w: i for i, w in enumerate(self.cfg["workloads"])}
        if self.cfg["mode"] == "search":
            todo.sort(key=lambda u: (u[0], order[u[1]], u[2]))
        else:
            todo.sort(key=lambda u: u[0])  # giữ thứ tự trong cấu hình, xen kẽ theo lần lặp
        return todo

    # --- chạy ---
    def run(self):
        while True:
            todo = self.plan_batch()
            if not todo:
                break
            log(f"đợt mới: {len(todo)} đơn vị: " + ", ".join(C.unit_id(w, n, r, c) for r, w, n, c in todo))
            if self.dry:
                return
            for r, w, n, client in todo:
                self.run_unit(w, n, r, client)
            if self.cfg["mode"] == "grid":
                break
        log("xong toàn bộ")

    def maybe_restart(self, r, n):
        pol = self.cfg["server"].get("restart", "never")
        grid_level = n in (self.cfg["search"].get("warm_start") or self.cfg["search"]["grid"])
        if self.block is not None and r != self.block and (
                pol == "per_repeat" or (pol == "per_repeat_grid" and grid_level)):
            self.server.restart(f"sang lần lặp r{r}")
        self.block = r
        self.server.ensure()

    def run_unit(self, w, n, r, client):
        cfg = self.cfg
        uid = C.unit_id(w, n, r, client)
        spec = C.unit_spec(cfg, w, n, r, client)
        h = C.sha256_obj(spec)
        d = self.units / uid
        if C.is_done(d, h):
            return
        wl = cfg["workloads"][w]
        m = C.num_requests(cfg, n)
        if m > wl["samples"]:
            raise C.SampleCapError(f"{uid}: cần {m} request khác nhau, {w} chỉ có {wl['samples']} mẫu")
        if d.exists():
            log(f"{uid}: thư mục dở dang (không có done.json hợp lệ) → xoá, đo lại")
            shutil.rmtree(d)
        d.mkdir(parents=True)
        self.maybe_restart(r, n)
        srv = self.server.current
        t_begin = time.time()
        log(f"{uid}: N={n}, {m} request, seed {spec['seed']}, client {client}")

        self.warmup(wl)
        self.server.reset_prefix_cache()
        (d / "metrics_before.txt").write_text(self.server.metrics_text())
        pol = Pollers(self.server, d)
        pol.start()
        cmd = self.client_cmd(w, n, m, spec["seed"], client, d, uid)
        t0 = time.time()
        try:
            with open(d / "client.log", "w") as lf:
                p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=cfg.get("unit_timeout_s", 3600))
        finally:
            pol.stop()
        t1 = time.time()
        (d / "metrics_after.txt").write_text(self.server.metrics_text())
        if p.returncode != 0:
            tail = (d / "client.log").read_text(errors="replace")[-3000:]
            raise RuntimeError(f"{uid}: client lỗi (mã {p.returncode}):\n{tail}")
        if client == "vllm":
            compact_vllm_result(d / "bench.json", d / "bench.json.gz")
        reqs = C.load_requests(d)
        met = {"all": C.compute_metrics(reqs, n, "all"), "steady": C.compute_metrics(reqs, n, "steady")}
        C.write_json(d / "metrics.json", met)
        a = met[cfg["verdict_window"]]
        C.write_json(d / "meta.json", {
            "run_id": uid, "spec": spec, "spec_sha256": h, "start": datetime.fromtimestamp(t0).astimezone().isoformat(),
            "end": datetime.fromtimestamp(t1).astimezone().isoformat(), "client_s": round(t1 - t0, 2),
            "unit_s": round(time.time() - t_begin, 2), "machine": self.machine, "hostname": socket.gethostname(),
            "server": srv, "client_cmd": cmd, "dataset": str(wl["path"]), "git_commit": self.git,
            "config": cfg["_path"]})
        C.write_json(d / "done.json", {"spec_sha256": h, "finished": now_iso()})
        log(f"{uid}: xong sau {t1 - t0:.0f} s — TTFT p95 {a['ttft_p95_ms']:.0f} ms, TPOT p95 {a['tpot_p95_ms']:.1f} ms,"
            f" {a['output_tok_s']:.0f} tok/s ra, lỗi {a['failed']}"
            f" → {'ĐẠT' if C.passes(a, cfg['slo']) else 'TRƯỢT'} (cửa sổ {cfg['verdict_window']})")

    def warmup(self, wl):
        """Gửi vài request (vài mẫu đầu file, ít token ra) để server chạy qua mọi đường code trước khi đo.
        Prefix cache bị xoá ngay sau đó nên việc dùng mẫu của workload không làm lợi lần đo."""
        wu = self.cfg["warmup"]
        if not wu.get("requests"):
            return
        rows = []
        with open(wl["path"]) as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= wu["requests"]:
                    break
        url = self.server.base + "/v1/completions"
        body = lambda row: {"model": self.cfg["server"]["model"], "prompt": row["prompt"],
                            "max_tokens": wu.get("max_tokens", 16), "temperature": 0, "ignore_eos": True}
        with cf.ThreadPoolExecutor(len(rows)) as ex:
            list(ex.map(lambda row: http(url, body(row), timeout=600), rows))

    def client_cmd(self, w, n, m, seed, client, d, uid):
        cfg, wl, s = self.cfg, self.cfg["workloads"][w], self.cfg["server"]
        samp = cfg["client"]["sampling"]
        if client == "vllm":
            cmd = shlex.split(cfg["client"]["vllm_cmd"]) + [
                "--backend", "openai", "--endpoint", "/v1/completions", "--base-url", s["base_url"],
                "--model", s["model"], "--tokenizer", str(C.resolve(cfg["client"]["tokenizer"])),
                "--dataset-name", "custom", "--dataset-path", str(wl["path"]), "--custom-output-len", "-1",
                "--skip-chat-template", "--num-prompts", str(m), "--max-concurrency", str(n),
                "--seed", str(seed), "--no-oversample", "--percentile-metrics", "ttft,tpot,itl,e2el",
                "--metric-percentiles", "50,95,99",
                "--goodput", f"ttft:{cfg['slo']['ttft_ms']}", f"tpot:{cfg['slo']['tpot_ms']}",
                "--save-result", "--save-detailed", "--result-dir", str(d), "--result-filename", "bench.json",
                "--disable-tqdm", "--request-id-prefix", f"{uid}-"]
            if samp.get("ignore_eos"):
                cmd.append("--ignore-eos")
            if "temperature" in samp:
                cmd += ["--temperature", str(samp["temperature"])]
            return cmd
        if client == "session":
            cmd = [cfg["client"].get("python", sys.executable), str(C.ROOT / "bench/session_client.py"), "closed",
                   "--dataset", str(wl["path"]), "--concurrency", str(n), "--num-requests", str(m),
                   "--seed", str(seed), "--base-url", s["base_url"], "--model", s["model"],
                   "--out", str(d / "bench.json.gz")]
            if samp.get("ignore_eos"):
                cmd.append("--ignore-eos")
            if "temperature" in samp:
                cmd += ["--temperature", str(samp["temperature"])]
            return cmd
        raise ValueError(client)


def compact_vllm_result(src, dst):
    """bench.json của vllm bench serve (--save-detailed) → bench.json.gz; câu trả lời chỉ giữ 200 ký tự đầu
    (đủ để kiểm tra mắt thường; số token ra đã có trong output_lens) để kết quả thô vừa với Git."""
    raw = json.loads(Path(src).read_text())
    raw["generated_texts"] = [t[:200] for t in raw.get("generated_texts", [])]
    raw["itls"] = [[round(x, 6) for x in xs] for xs in raw.get("itls", [])]
    tmp = Path(str(dst) + ".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(raw, f)
    os.replace(tmp, dst)
    Path(src).unlink()


def git_commit():
    if c := os.environ.get("BENCH_GIT_COMMIT"):
        return c
    try:
        sha = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(C.ROOT), "status", "--porcelain", "bench", "scripts", "workloads"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return None


def prepare_workloads(cfg):
    manifest = json.loads((C.ROOT / "workloads/manifest.json").read_text())["files"]
    for w, wl in cfg["workloads"].items():
        wl["path"] = C.resolve(wl["path"])
        if not wl["path"].exists():
            sys.exit(f"thiếu file workload {wl['path']}")
        wl["sha256"] = C.sha256_file(wl["path"])
        if w in manifest and wl["sha256"] != manifest[w]["sha256"]:
            sys.exit(f"{w}: sha256 {wl['sha256']} khác workloads/manifest.json — build lại bằng workloads/build.py")
        wl["samples"] = C.count_lines(wl["path"])


# ---------- --status ----------

def status(runner):
    cfg = runner.cfg
    done = runner.done_units()
    print(f"cấu hình {cfg['_path']} · chế độ {cfg['mode']} · {len(done)} đơn vị đã xong · cửa sổ quyết định "
          f"{cfg['verdict_window']} · SLO TTFT p95 ≤ {cfg['slo']['ttft_ms']} ms, TPOT p95 ≤ {cfg['slo']['tpot_ms']} ms")
    per_req = {}
    for (w, n, r, client), d in done.items():
        meta = C.read_json(d / "meta.json")
        per_req.setdefault(w, []).append(meta["unit_s"] / meta["spec"]["num_requests"])
    todo = runner.plan_batch()
    rest_s = 0.0
    overhead = cfg.get("unit_overhead_s", 15)
    if cfg["mode"] == "search":
        for w in cfg["workloads"]:
            a = runner.analyze(w, done)
            print(f"\n{w}: {a['status']}, goodput {C.fmt_interval(a['lo'], a['hi'], a['status'])}")
            for n, lv in a["levels"].items():
                med = lv.get("median")
                v = {True: "đạt", False: "trượt", None: "chưa đủ lần lặp"}[lv["verdict"]]
                extra = f"TTFT p95 {med['ttft_p95_ms']:.0f} ms, TPOT p95 {med['tpot_p95_ms']:.1f} ms" if med else ""
                print(f"  N={n:<5} lặp xong {lv['repeats_done']}  {v:16} {extra}")
            # ước lượng các mức còn phải đo sau đợt này (chưa biết trước vì phụ thuộc kết quả)
            if a["status"] == "measuring":
                lo, hi = a["lo"], a["hi"]
                if lo and hi:
                    steps = max(0, math.ceil(math.log2(max((hi - lo) / C.tol(lo, cfg["search"]), 1))) - 1)
                elif hi is None:
                    steps = 4
                else:
                    steps = 3
                future_n = (lo + hi) // 2 if lo and hi else (lo or 8) * 2 if hi is None else 4
                rest_s += steps * cfg["repeats"] * (overhead + C.num_requests(cfg, future_n)
                                                    * statistics_median(per_req.get(w), 1.0))
                print(f"  ước lượng thêm ~{steps} mức sau đợt hiện tại")
    for r, w, n, client in todo:
        rest_s += overhead + C.num_requests(cfg, n) * statistics_median(per_req.get(w), 1.0)
    print(f"\nđợt hiện tại còn {len(todo)} đơn vị: " + ", ".join(C.unit_id(w, n, r, c) for r, w, n, c in todo))
    price = cfg.get("price_usd_per_h", 0.2833)
    hours = rest_s / 3600
    print(f"ước tính còn ~{hours:.2f} giờ GPU (≈ ${hours * price:.2f} ≈ {hours * price * 26200:,.0f} VND)"
          + ("" if per_req else " — chưa có đơn vị nào xong, dùng 1 s/request làm tạm"))


def statistics_median(xs, default):
    if not xs:
        return default
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", action="store_true", help="in tiến độ, không chạy")
    ap.add_argument("--dry-run", action="store_true", help="in đợt kế tiếp, không chạy")
    ap.add_argument("--machine", default=os.environ.get("BENCH_MACHINE", ""),
                    help="ghi vào meta: ví dụ 'host=506934 machine=116596 instance=...'")
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    prepare_workloads(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(cfg, out, args.machine, dry=args.dry_run)
    if args.status:
        status(runner)
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        runner.run()
    except C.SampleCapError as e:
        log(f"DỪNG: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
