"""Phần dùng chung của harness: cấu hình, ID/hash đơn vị đo, đọc kết quả từng request, tính chỉ số, logic tìm goodput.

Thiết kế: docs/gd1-vllm-baseline/03-cong-cu-benchmark/plan.md, docs/workloads.md mục "Cách tìm goodput",
docs/thi-nghiem-chay-tiep.md. Không phụ thuộc GPU/aiohttp → chạy được ở local (summarize, test).
"""

import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ---------- tiện ích file ----------

def atomic_write(path, data, mode="w"):
    """Ghi ra path.tmp rồi đổi tên → không bao giờ có file ghi dở (thi-nghiem-chay-tiep.md nguyên tắc 2)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, mode) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


# ---------- cấu hình ----------

def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_path"] = str(path)
    cfg.setdefault("repeats", 1)
    cfg.setdefault("seed_base", 1000)
    cfg.setdefault("num_requests", {"min": 100, "per_user": 4})
    cfg.setdefault("slo", {"ttft_ms": 10000, "tpot_ms": 150})
    cfg.setdefault("slo2", {"ttft_ms": 2000, "tpot_ms": 100})
    cfg.setdefault("verdict_window", "all")
    s = cfg.setdefault("search", {})
    s.setdefault("grid", [8, 16, 32, 64])
    s.setdefault("min_n", 2)
    s.setdefault("max_n", 1024)
    s.setdefault("abs_tol", 2)
    s.setdefault("rel_tol", 0.10)
    s.setdefault("warm_start", None)
    return cfg


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num_requests(cfg, n):
    """Số request mỗi lần đo: max(min, per_user × N) (đề xuất P03)."""
    nr = cfg["num_requests"]
    return max(int(nr["min"]), int(nr["per_user"]) * int(n))


class SampleCapError(Exception):
    """Lần đo cần nhiều request khác nhau hơn số mẫu của workload. Không tự lặp mẫu (lặp sẽ trúng prefix cache)."""


# ---------- đơn vị đo ----------

def unit_id(w, n, r, client="vllm"):
    uid = f"{w}-c{n}-r{r}"
    return uid if client == "vllm" else f"{uid}-{client}"


def unit_spec(cfg, w, n, r, client="vllm"):
    """Mọi thứ quyết định kết quả của một đơn vị đo. Hash của nó nằm trong done.json:
    đổi cấu hình → hash khác → đơn vị bị đo lại (thi-nghiem-chay-tiep.md nguyên tắc 4)."""
    wl = cfg["workloads"][w]
    m = num_requests(cfg, n)
    return {
        "workload": w,
        "dataset_sha256": wl["sha256"],
        "concurrency": int(n),
        "repeat": int(r),
        "num_requests": m,
        "seed": int(cfg["seed_base"]) + int(r),
        "client": client,
        "sampling": cfg["client"]["sampling"],
        "warmup": cfg["warmup"],
        "server_start": cfg["server"]["start"],
        "software": cfg["software"],
    }


def is_done(unit_dir, spec_hash):
    d = Path(unit_dir) / "done.json"
    if not d.exists():
        return False
    try:
        return read_json(d).get("spec_sha256") == spec_hash
    except (json.JSONDecodeError, OSError):
        return False


# ---------- kết quả từng request ----------

W4_KEYS = ("sid", "day", "turn", "resumed", "t_rel", "sched", "lag", "dropped", "budget", "max_tokens", "user_tokens",
           "think_s", "cached")

def load_requests(unit_dir, raw=None):
    """Đọc kết quả từng request (cùng khoá với `vllm bench serve --save-detailed`), trả về list dict.
    Kết quả W4 (session_client scenario): chỉ giữ request gửi trong cửa sổ đo (`in_window`); các khoá theo phiên
    (sid, turn, resumed, lag, cached, ...) được chép kèm vào từng request."""
    raw = read_json(Path(unit_dir) / "bench.json.gz") if raw is None else raw
    errors = raw.get("errors") or [""] * len(raw["ttfts"])
    extra = [k for k in W4_KEYS if k in raw]
    reqs = []
    for i in range(len(raw["ttfts"])):
        if "in_window" in raw and not raw["in_window"][i]:
            continue
        ok = not errors[i] and raw["ttfts"][i] is not None and raw["ttfts"][i] > 0
        reqs.append({**{k: raw[k][i] for k in extra},
            "start": raw["start_times"][i],
            "ttft": raw["ttfts"][i] if ok else math.inf,
            "latency": raw["latencies"][i] if ok else math.inf,
            "out": raw["output_lens"][i],
            "in": raw["input_lens"][i],
            "ok": ok,
            "error": errors[i],
            "itl": raw["itls"][i] if ok and raw.get("itls") else [],
        })
    return reqs


def tpot_of(r):
    """Giống vllm/benchmarks/serve.py: TPOT = (latency − TTFT) / (output_len − 1), chỉ khi output_len > 1."""
    if not r["ok"]:
        return math.inf
    if r["out"] > 1:
        return (r["latency"] - r["ttft"]) / (r["out"] - 1)
    return 0.0


def window(reqs, n, kind):
    """`all`: mọi request. `steady`: bỏ đợt đầu (N request đầu được gửi cùng lúc lúc t=0) và N request gửi cuối
    (chạy khi số request đang chạy đã giảm dần) → chỉ giữ phần hệ thống có đúng N request."""
    if kind == "all":
        return reqs
    order = sorted(reqs, key=lambda r: r["start"])
    return order[n:len(order) - n] if len(order) > 2 * n else order


def pct(xs, q):
    xs = [x for x in xs]
    if not xs:
        return math.nan
    if any(math.isinf(x) for x in xs):
        # request lỗi tính như vô cùng: np.percentile không xử lý inf ổn định → sắp xếp + nội suy tay
        xs = sorted(xs)
        k = (len(xs) - 1) * q / 100
        lo, hi = math.floor(k), math.ceil(k)
        if math.isinf(xs[hi]):
            return math.inf
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return float(np.percentile(xs, q))  # cùng cách nội suy (linear) với vllm bench serve


def compute_metrics(reqs, n, kind="all"):
    ws = window(reqs, n, kind)
    ok = [r for r in ws if r["ok"]]
    ttft = [r["ttft"] * 1000 for r in ws]
    tpot = [tpot_of(r) * 1000 for r in ws if (not r["ok"]) or r["out"] > 1]
    e2el = [r["latency"] * 1000 for r in ws]
    itl = [x * 1000 for r in ok for x in r["itl"]]
    if ok:
        t0 = min(r["start"] for r in ok)
        t1 = max(r["start"] + r["latency"] for r in ok)
        dur = max(t1 - t0, 1e-9)
    else:
        dur = math.nan
    m = {"n_requests": len(ws), "failed": sum(not r["ok"] for r in ws), "duration_s": dur,
         "output_tok_s": sum(r["out"] for r in ok) / dur, "input_tok_s": sum(r["in"] for r in ok) / dur,
         "req_s": len(ok) / dur,
         "mean_in": float(np.mean([r["in"] for r in ok])) if ok else math.nan,
         "mean_out": float(np.mean([r["out"] for r in ok])) if ok else math.nan}
    for name, xs in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2el", e2el)):
        for q in (50, 95, 99):
            m[f"{name}_p{q}_ms"] = pct(xs, q)
        m[f"{name}_mean_ms"] = float(np.mean([x for x in xs if not math.isinf(x)])) if xs else math.nan
    return m


def passes(metrics, slo):
    return metrics["ttft_p95_ms"] <= slo["ttft_ms"] and metrics["tpot_p95_ms"] <= slo["tpot_ms"]


def median_metrics(list_of_metrics):
    """Trung vị từng chỉ số qua các lần lặp (quyết định đạt/trượt theo trung vị, docs/workloads.md)."""
    keys = list_of_metrics[0].keys()
    return {k: float(np.median([m[k] for m in list_of_metrics])) for k in keys}


# ---------- tìm goodput (tất định, tính lại được từ các done.json) ----------

def max_fit(cfg, samples):
    """N lớn nhất (chẵn) mà num_requests(N) = max(min, per_user × N) ≤ số mẫu."""
    nr = cfg["num_requests"]
    if samples < int(nr["min"]):
        return 0
    return (samples // int(nr["per_user"])) // 2 * 2


def even_mid(lo, hi):
    m = ((lo + hi) // 4) * 2
    if m <= lo or m >= hi:
        m = (lo + hi) // 2
    return m


def tol(lo, s):
    return max(s["abs_tol"], s["rel_tol"] * lo)


def search_step(verdicts, counts, s, repeats, max_fit):
    """Một bước tìm goodput cho một workload.

    verdicts: {N: True/False} các mức đã quyết định (đã đủ số lần lặp yêu cầu).
    counts:   {N: số lần lặp đã xong} (để biết mức nào đã được đo lặp gấp đôi khi kiểm tra đơn điệu).
    s:        cấu hình search. repeats: số lần lặp chuẩn.
    max_fit:  N lớn nhất (chẵn) mà workload còn đủ số mẫu khác nhau cho một lần đo.
    Trả về (status, need) với need = {N: số lần lặp cần có}. status:
      'measuring' | 'done' | 'below_min' | 'above_cap' | 'sample_cap' | 'non_monotonic'
    """
    start = sorted(s["warm_start"]) if s.get("warm_start") else sorted(s["grid"])
    passes_ = sorted(n for n, v in verdicts.items() if v)
    fails = sorted(n for n, v in verdicts.items() if not v)

    # 5. kiểm tra đơn điệu: mức đạt nằm trên mức trượt → đo lặp gấp đôi hai mức đó; vẫn ngược → dừng, báo cáo
    if passes_ and fails and passes_[-1] > fails[0]:
        f, p = fails[0], passes_[-1]
        if counts.get(f, 0) >= 2 * repeats and counts.get(p, 0) >= 2 * repeats:
            return "non_monotonic", {}
        return "measuring", {f: 2 * repeats, p: 2 * repeats}

    # 1. lưới gốc (hoặc lo/hi của cấu hình trước khi khởi đầu nóng) luôn đo đủ
    missing = [n for n in start if n not in verdicts]
    if missing:
        need = {n: repeats for n in missing if n <= max_fit}
        if not need:
            return "sample_cap", {}
        return "measuring", need

    if not fails:  # 2. mở rộng lên; chạm trần an toàn hoặc trần số mẫu thì thử đúng mức trần rồi dừng
        cap = min(s["max_n"], max_fit)
        if passes_[-1] >= cap:
            return ("above_cap" if cap == s["max_n"] else "sample_cap"), {}
        nxt = min(passes_[-1] * 2, cap)
    elif not passes_:  # 3. mở rộng xuống
        if fails[0] <= s["min_n"]:
            return "below_min", {}
        nxt = max(fails[0] // 2, s["min_n"])
    else:  # 4. tìm nhị phân
        lo, hi = passes_[-1], fails[0]
        if hi - lo <= tol(lo, s):
            return "done", {}
        nxt = even_mid(lo, hi)
    if nxt > max_fit:
        return "sample_cap", {}
    return "measuring", {nxt: repeats}


def interval(verdicts):
    """(lo, hi) từ các mức đã quyết định: lo = mức đạt lớn nhất dưới mức trượt nhỏ nhất, hi = mức trượt nhỏ nhất."""
    fails = sorted(n for n, v in verdicts.items() if not v)
    hi = fails[0] if fails else None
    passes_ = sorted(n for n, v in verdicts.items() if v and (hi is None or n < hi))
    lo = passes_[-1] if passes_ else None
    return lo, hi


def fmt_interval(lo, hi, status=None):
    if lo is None and hi is not None:
        return f"< {hi}"
    if hi is None and lo is not None:
        tail = {"sample_cap": " (trần số mẫu)", "above_cap": " (trần an toàn)"}.get(status, "")
        return f"≥ {lo}{tail}"
    if lo is None:
        return "—"
    return f"{lo} ({hi} trượt)"
