"""Tổng hợp kết quả sweep: bảng từng đơn vị, goodput dạng khoảng (SLO chính, SLO phụ, độ nhạy theo ngưỡng TTFT),
nhiễu giữa các lần lặp, so client tự viết với vllm bench serve, kiểm tra server (/metrics). Chạy ở local, không cần GPU.

    python bench/summarize.py --config bench/configs/full.yaml --out results/gd1/04-do-baseline-day-du
        → in Markdown; ghi <out>/summary.md, <out>/summary.json, <out>/units.csv
    python bench/summarize.py ... --compare-clients     # thêm bảng client "session" vs "vllm" (cùng w, N, lần lặp)
    python bench/summarize.py --scenario-check RESULT.json.gz SCENARIO.jsonl   # kiểm tra lần phát lại W4
    python bench/summarize.py --config bench/configs/w4.yaml --out results/gd1/05-...  # W4: goodput k, tách theo loại
        request, request dài chen request ngắn, prefix cache theo think time

Mọi chỉ số tính lại từ số liệu từng request (bench.json.gz) bằng bench/common.py, nên đổi SLO / cửa sổ không cần đo lại.
"""

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import sweep as S  # noqa: E402

WINDOWS = ("all", "steady")
TTFT_SENS_MS = (2000, 5000, 10000, 20000)


def prom_hist(text, name):
    """(sum, count) của histogram Prometheus `name` (cộng mọi label)."""
    vals = S.parse_prom(text, {name + "_sum", name + "_count"})
    return vals.get(name + "_sum", 0.0), vals.get(name + "_count", 0.0)


def server_side(d):
    """Chỉ số phía server trong lúc đo: hiệu /metrics sau − trước, và đỉnh của chuỗi đo mỗi 1 s."""
    b, a = (d / "metrics_before.txt").read_text(), (d / "metrics_after.txt").read_text()
    names = set(S.POLL_METRICS)
    vb, va = S.parse_prom(b, names), S.parse_prom(a, names)
    delta = lambda k: va.get(k, 0.0) - vb.get(k, 0.0)
    out = {"preemptions": delta("vllm:num_preemptions_total"),
           "prefix_queries": delta("vllm:prefix_cache_queries_total"),
           "prefix_hits": delta("vllm:prefix_cache_hits_total"),
           "prompt_tokens": delta("vllm:prompt_tokens_total"),
           "gen_tokens": delta("vllm:generation_tokens_total")}
    out["prefix_hit_rate"] = out["prefix_hits"] / out["prefix_queries"] if out["prefix_queries"] else math.nan
    for name, key in (("vllm:time_to_first_token_seconds", "server_ttft_mean_ms"),
                      ("vllm:request_queue_time_seconds", "server_queue_mean_ms"),
                      ("vllm:request_prefill_time_seconds", "server_prefill_mean_ms")):
        sb, cb = prom_hist(b, name)
        sa, ca = prom_hist(a, name)
        out[key] = (sa - sb) / (ca - cb) * 1000 if ca > cb else math.nan
    ts = d / "metrics_timeseries.jsonl"
    rows = [json.loads(x) for x in ts.read_text().splitlines() if x.strip()] if ts.exists() else []
    for k, key in (("vllm:num_requests_running", "running_max"), ("vllm:num_requests_waiting", "waiting_max"),
                   ("vllm:kv_cache_usage_perc", "kv_usage_max")):
        out[key] = max((r.get(k, 0.0) for r in rows), default=math.nan)
    # trung vị số request đang chạy trong phần giữa lần đo: chỉ các giây có request (client vllm mất ~10 s
    # nạp tokenizer trước khi gửi), bỏ 10% đầu và cuối
    run = [r.get("vllm:num_requests_running", 0.0) for r in rows
           if r.get("vllm:num_requests_running", 0.0) + r.get("vllm:num_requests_waiting", 0.0) > 0]
    mid = run[len(run) // 10: len(run) - len(run) // 10] or run
    out["running_p50_mid"] = float(np.median(mid)) if mid else math.nan
    gpu = d / "gpu.csv"
    if gpu.exists():
        lines = [x.split(", ") for x in gpu.read_text().splitlines()[1:] if x and x[0].isdigit()]
        try:
            out["power_max_w"] = max(float(x[2].split()[0]) for x in lines)
            out["sm_clock_min_mhz"] = min(float(x[3].split()[0]) for x in lines)
            out["temp_max_c"] = max(float(x[5]) for x in lines)
            out["mem_used_max_mib"] = max(float(x[1].split()[0]) for x in lines)
        except (ValueError, IndexError):
            pass
    return out


def collect(runner):
    """Các đơn vị đã xong (đúng hash cấu hình hiện tại) → list dict: w, n, r, client, metrics, server, meta."""
    rows = []
    for (w, n, r, client), d in sorted(runner.done_units().items()):
        meta = C.read_json(d / "meta.json")
        reqs = C.load_requests(d)
        rows.append({"w": w, "n": n, "r": r, "client": client, "dir": d, "meta": meta,
                     "m": {k: C.compute_metrics(reqs, n, k) for k in WINDOWS}, "srv": server_side(d),
                     "in_tokens_dataset": None})
    return rows


def goodput_for(rows, w, slo, win, repeats):
    """Goodput (lo, hi) của workload w theo SLO bất kỳ, từ trung vị các lần lặp đã đo (client vllm)."""
    levels = {}
    for x in rows:
        if x["w"] == w and x["client"] == "vllm":
            levels.setdefault(x["n"], {})[x["r"]] = x["m"][win]
    verdicts = {}
    for n, rr in levels.items():
        use = [rr[r] for r in sorted(rr)] if len(rr) >= repeats else None
        if use:
            verdicts[n] = C.passes(C.median_metrics(use), slo)
    lo, hi = C.interval(verdicts)
    mono = not any(v and any(not verdicts[m] for m in verdicts if m < n) for n, v in verdicts.items())
    return lo, hi, sorted(verdicts), mono


def spread(vals):
    """Độ lệch giữa các lần lặp: (max − min) / trung vị, %."""
    vals = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
    if len(vals) < 2:
        return math.nan
    med = float(np.median(vals))
    return (max(vals) - min(vals)) / med * 100 if med else math.nan


def f(x, nd=0):
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x)):
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:,.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def report(runner, rows, compare_clients):
    cfg = runner.cfg
    R = cfg["repeats"]
    md, js = [], {"config": cfg["_path"], "repeats": R, "slo": cfg["slo"], "slo2": cfg["slo2"],
                  "verdict_window": cfg["verdict_window"], "goodput": {}, "noise": {}, "server_starts": []}
    md.append(f"# Tổng hợp `{runner.out}`\n\nCấu hình `{cfg['_path']}` · {len(rows)} đơn vị đã xong · "
              f"lặp {R} lần/mức · cửa sổ quyết định đạt/trượt: **{cfg['verdict_window']}** "
              "(all = mọi request; steady = bỏ N request gửi đầu và N request gửi cuối)\n")

    # --- server starts ---
    starts = runner.out / "server_starts.jsonl"
    if starts.exists():
        ss = [json.loads(x) for x in starts.read_text().splitlines() if x.strip()]
        js["server_starts"] = ss
        md.append("## Các lần khởi động server\n")
        md.append(md_table(["#", "thời điểm", "lý do", "sẵn sàng (s)", "dùng lại", "KV cache (token)", "KV (GiB)",
                            "torch.compile (s)"],
                           [[s["start_index"], s["time"], s["reason"], s["ready_s"], s["reused_running_server"],
                             f(s.get("kv_cache_tokens")), s.get("kv_cache_gib", "—"), s.get("torch_compile_s", "—")]
                            for s in ss]) + "\n")

    # --- goodput ---
    if cfg["mode"] == "search" or any(x["client"] == "vllm" for x in rows):
        md.append("## Goodput (số user đồng thời lớn nhất đạt SLO, trung vị các lần lặp)\n")
        slos = [("chính", cfg["slo"]), ("phụ", cfg["slo2"])] + \
               [(f"TTFT ≤ {t / 1000:g} s", {"ttft_ms": t, "tpot_ms": cfg["slo"]["tpot_ms"]}) for t in TTFT_SENS_MS]
        hdr = ["workload", "cửa sổ"] + [f"SLO {name} ({s['ttft_ms'] / 1000:g} s / {s['tpot_ms']:g} ms)"
                                         for name, s in slos]
        tab = []
        for w in cfg["workloads"]:
            for win in WINDOWS:
                cells, jw = [w, win], {}
                for name, s in slos:
                    lo, hi, levels, mono = goodput_for(rows, w, s, win, R)
                    cells.append(C.fmt_interval(lo, hi) + ("" if mono else " ⚠ không đơn điệu"))
                    jw[name] = {"lo": lo, "hi": hi, "levels": levels, "monotonic": mono}
                tab.append(cells)
                js["goodput"].setdefault(w, {})[win] = jw
        md.append(md_table(hdr, tab))
        if cfg["mode"] == "search":
            md.append("\nTrạng thái tìm (theo SLO chính, cửa sổ quyết định): " + "; ".join(
                f"{w}: {runner.analyze(w)['status']}" for w in cfg["workloads"]))
        md.append("\nSLO phụ / độ nhạy: chỉ có độ phân giải của các mức đã đo (tìm nhị phân chạy theo SLO chính).\n")

    # --- từng mức (trung vị) ---
    md.append("## Từng mức N (trung vị các lần lặp, cửa sổ all; [steady] trong ngoặc)\n")
    groups = {}
    for x in rows:
        groups.setdefault((x["w"], x["n"], x["client"]), []).append(x)
    tab = []
    for (w, n, client), xs in sorted(groups.items()):
        med = {k: C.median_metrics([x["m"][k] for x in xs]) for k in WINDOWS}
        srv = {k: float(np.nanmedian([x["srv"].get(k, math.nan) for x in xs])) for k in xs[0]["srv"]}
        a, s = med["all"], med["steady"]
        ok = "đạt" if C.passes(a, cfg["slo"]) else "trượt"
        tab.append([w, n, client, len(xs), f"{f(a['ttft_p50_ms'])} / {f(a['ttft_p95_ms'])} [{f(s['ttft_p95_ms'])}]",
                    f"{f(a['tpot_p50_ms'], 1)} / {f(a['tpot_p95_ms'], 1)} [{f(s['tpot_p95_ms'], 1)}]",
                    f(a["itl_p95_ms"], 1), f(a["output_tok_s"]), f(a["input_tok_s"]), ok,
                    f(srv["running_max"]), f(srv["running_p50_mid"]), f(srv["waiting_max"]),
                    f(srv["kv_usage_max"] * 100 if not math.isnan(srv["kv_usage_max"]) else math.nan),
                    f(srv["preemptions"]), f(srv["prefix_hit_rate"] * 100 if not math.isnan(srv["prefix_hit_rate"])
                                            else math.nan, 1),
                    f(a["failed"])])
    md.append(md_table(["w", "N", "client", "lần", "TTFT p50 / p95 (ms) [steady p95]",
                        "TPOT p50 / p95 (ms) [steady p95]", "ITL p95", "tok/s ra", "tok/s vào", "SLO chính",
                        "đang chạy max", "đang chạy p50", "chờ max", "KV dùng max %", "preempt", "trúng prefix %",
                        "lỗi"], tab) + "\n")

    # --- nhiễu ---
    md.append("## Nhiễu giữa các lần lặp ((max − min) / trung vị, %)\n")
    tab = []
    for (w, n, client), xs in sorted(groups.items()):
        if len(xs) < 2:
            continue
        sp = {k: spread([x["m"]["all"][k] for x in xs]) for k in
              ("output_tok_s", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "e2el_p50_ms")}
        sp_st = {k: spread([x["m"]["steady"][k] for x in xs]) for k in ("ttft_p95_ms", "tpot_p95_ms")}
        js["noise"][f"{w}-c{n}-{client}"] = {"n_repeats": len(xs), "all": sp, "steady": sp_st}
        tab.append([w, n, client, len(xs), f(sp["output_tok_s"], 1), f(sp["ttft_p50_ms"], 1),
                    f(sp["ttft_p95_ms"], 1), f(sp_st["ttft_p95_ms"], 1), f(sp["tpot_p50_ms"], 1),
                    f(sp["tpot_p95_ms"], 1), f(sp_st["tpot_p95_ms"], 1), f(sp["e2el_p50_ms"], 1)])
    md.append(md_table(["w", "N", "client", "lần", "tok/s ra", "TTFT p50", "TTFT p95", "TTFT p95 steady",
                        "TPOT p50", "TPOT p95", "TPOT p95 steady", "E2EL p50"], tab) + "\n" if tab
              else "(chưa có mức nào lặp ≥ 2 lần)\n")

    # --- client vs server (kiểm tra b) ---
    md.append("## TTFT client vs server (/metrics, trung bình)\n")
    tab = []
    for x in rows:
        c, s = x["m"]["all"]["ttft_mean_ms"], x["srv"]["server_ttft_mean_ms"]
        tab.append([C.unit_id(x["w"], x["n"], x["r"], x["client"]), f(c, 1), f(s, 1), f(c - s, 1),
                    f((c - s) / s * 100 if s else math.nan, 1)])
    md.append(md_table(["đơn vị", "client (ms)", "server (ms)", "chênh (ms)", "chênh %"], tab) + "\n")

    # --- so cửa sổ all vs steady ---
    md.append("## Cửa sổ all vs steady (p95, chênh % so với all)\n")
    tab = []
    for x in rows:
        a, s = x["m"]["all"], x["m"]["steady"]
        rel = lambda k: (s[k] - a[k]) / a[k] * 100 if a[k] and not math.isinf(a[k]) else math.nan
        tab.append([C.unit_id(x["w"], x["n"], x["r"], x["client"]), a["n_requests"], s["n_requests"],
                    f(a["ttft_p95_ms"]), f(s["ttft_p95_ms"]), f(rel("ttft_p95_ms"), 1),
                    f(a["tpot_p95_ms"], 1), f(s["tpot_p95_ms"], 1), f(rel("tpot_p95_ms"), 1)])
    md.append(md_table(["đơn vị", "req all", "req steady", "TTFT p95 all", "steady", "Δ%", "TPOT p95 all", "steady",
                        "Δ%"], tab) + "\n")

    # --- so hai client ---
    if compare_clients:
        md.append("## Client tự viết (session, closed-loop) vs vllm bench serve\n")
        by = {(x["w"], x["n"], x["r"], x["client"]): x for x in rows}
        tab, js["client_compare"] = [], {}
        for (w, n, r, client), x in sorted(by.items()):
            if client != "session" or (w, n, r, "vllm") not in by:
                continue
            v = by[(w, n, r, "vllm")]
            cells = [f"{w}-c{n}-r{r}"]
            jr = {}
            for k in ("output_tok_s", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "e2el_p50_ms"):
                a, b = x["m"]["all"][k], v["m"]["all"][k]
                d = (a - b) / b * 100 if b else math.nan
                jr[k] = {"session": a, "vllm": b, "diff_pct": d}
                cells.append(f"{f(a, 1)} / {f(b, 1)} ({f(d, 1)}%)")
            js["client_compare"][f"{w}-c{n}-r{r}"] = jr
            tab.append(cells)
        md.append(md_table(["điểm", "tok/s ra", "TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95", "E2EL p50"], tab)
                  + "\n\nMỗi ô: session / vllm (chênh % so với vllm).\n")

    # --- thời gian ---
    md.append("## Thời gian đo\n")
    tab = []
    for x in rows:
        mt = x["meta"]
        tab.append([C.unit_id(x["w"], x["n"], x["r"], x["client"]), mt["spec"]["num_requests"], mt["client_s"],
                    mt["unit_s"], f(mt["unit_s"] / mt["spec"]["num_requests"], 2), x["m"]["all"]["mean_in"].__round__(),
                    f(x["m"]["all"]["mean_out"]), f(x["srv"].get("power_max_w"), 1), f(x["srv"].get("sm_clock_min_mhz")),
                    f(x["srv"].get("temp_max_c"))])
    total = sum(x["meta"]["unit_s"] for x in rows)
    md.append(md_table(["đơn vị", "request", "client (s)", "cả đơn vị (s)", "s/request", "token vào TB",
                        "token ra TB", "W max", "xung SM min", "°C max"], tab))
    md.append(f"\nTổng thời gian các đơn vị: {total / 3600:.2f} h.\n")
    js["total_unit_hours"] = total / 3600
    return "\n".join(md), js


# ---------- W4 (P05) ----------

LEN_GROUPS = [("ngắn < 1k", 0, 1000), ("1–2k", 1000, 2000), ("dài ≥ 2k", 2000, 10**9)]
THINK_BINS = [(0, 30), (30, 120), (120, 300), (300, 900), (900, 10**9)]
ACTIVE_TAIL_S = 600  # phiên coi là còn mở tới 10 phút sau lượt cuối (như build_w4.summarize)


def prompt_len(r):
    return r["in"] if r["in"] is not None else r["budget"]


def lat_stats(rs):
    """n, TTFT p50/p95, TPOT p50/p95 (ms) của một nhóm request (lỗi tính như vô cùng)."""
    if not rs:
        return {"n": 0}
    m = C.compute_metrics(rs, 0, "all")
    return {"n": len(rs), "failed": m["failed"], "ttft_p50_ms": m["ttft_p50_ms"], "ttft_p95_ms": m["ttft_p95_ms"],
            "tpot_p50_ms": m["tpot_p50_ms"], "tpot_p95_ms": m["tpot_p95_ms"]}


def w4_breakdown(reqs):
    """TTFT/TPOT theo độ dài prompt và theo loại lượt (mở phiên / lượt sau / quay lại phiên cũ)."""
    out = {}
    for name, a, b in LEN_GROUPS:
        out[name] = lat_stats([r for r in reqs if a <= prompt_len(r) < b])
    out["lượt đầu phiên mới"] = lat_stats([r for r in reqs if r["turn"] == 0 and not r["resumed"]])
    out["quay lại phiên cũ (resumed)"] = lat_stats([r for r in reqs if r["resumed"]])
    out["lượt sau (≥ 2)"] = lat_stats([r for r in reqs if r["turn"] > 0])
    return out


def w4_interference(reqs, long_min=4000):
    """Request ngắn (< 1k) có bị request dài làm chậm không: so TTFT của request ngắn khi lúc nó chờ token đầu có / không
    có request dài (≥ long_min token) đang chờ token đầu (tức đang prefill hoặc xếp hàng); so TPOT của request ngắn khi
    lúc nó sinh token có / không có request dài đang prefill."""
    ok = [r for r in reqs if r["ok"]]
    longs = [(r["start"], r["start"] + r["ttft"]) for r in ok if prompt_len(r) >= long_min]
    longs.sort()

    def overlaps(a, b):
        return any(x < b and y > a for x, y in longs if x < b)

    shorts = [r for r in reqs if prompt_len(r) < 1000]
    t_with = [r for r in shorts if r["ok"] and overlaps(r["start"], r["start"] + r["ttft"])]
    t_without = [r for r in shorts if r["ok"] and not overlaps(r["start"], r["start"] + r["ttft"])]
    d_with = [r for r in shorts if r["ok"] and r["out"] > 1
              and overlaps(r["start"] + r["ttft"], r["start"] + r["latency"])]
    d_without = [r for r in shorts if r["ok"] and r["out"] > 1
                 and not overlaps(r["start"] + r["ttft"], r["start"] + r["latency"])]
    return {"long_min_tokens": long_min, "n_long": len(longs),
            "ttft_with_long": lat_stats(t_with), "ttft_without_long": lat_stats(t_without),
            "tpot_with_long": lat_stats(d_with), "tpot_without_long": lat_stats(d_without)}


def w4_cache(reqs):
    """Prefix cache theo think time ở lượt sau (không bỏ bớt lịch sử): phần lịch sử dùng lại được trúng cache bao nhiêu.
    Lịch sử dùng lại được ≈ prompt − tin nhắn mới (lệch vài token template). Cần `cached` (--enable-prompt-tokens-details)."""
    fol = [r for r in reqs if r["ok"] and r["turn"] > 0 and not r["dropped"] and r.get("cached") is not None]
    if not fol:
        return None
    rows = []
    for a, b in THINK_BINS:
        g = [r for r in fol if a <= (r["think_s"] or 0) < b]
        if not g:
            rows.append({"think_s": [a, b], "n": 0})
            continue
        reuse = [min(1.0, r["cached"] / max(1, prompt_len(r) - r["user_tokens"])) for r in g]
        rows.append({"think_s": [a, b], "n": len(g), "reuse_hit_mean": float(np.mean(reuse)),
                     "full_hit_share": float(np.mean([x >= 0.9 for x in reuse])),
                     "ttft_p50_ms": lat_stats(g)["ttft_p50_ms"]})
    allc = [r for r in reqs if r["ok"] and r.get("cached") is not None]
    tot = sum(prompt_len(r) for r in allc)
    return {"bins": rows, "cached_token_share": sum(r["cached"] for r in allc) / tot if tot else math.nan}


def w4_active(raw, warm, meas):
    """Số phiên đang mở (từ lượt đầu gửi tới lượt cuối + 10 phút) lấy mẫu mỗi phút trong cửa sổ đo, thời gian trace."""
    spans = {}
    for sid, t in zip(raw["sid"], raw["t_rel"]):
        a, b = spans.get(sid, (t, t))
        spans[sid] = (min(a, t), max(b, t))
    act = [sum(a <= t < b + ACTIVE_TAIL_S for a, b in spans.values()) for t in range(int(warm), int(warm + meas), 60)]
    return [int(np.median(act)), int(max(act))] if act else [0, 0]


def collect_w4(runner):
    rows = []
    for (key, k, r), d in sorted(runner.done_units().items(), key=lambda x: (x[0][0] != runner.w4["search_on"], x[0])):
        raw = C.read_json(d / "bench.json.gz")
        reqs = C.load_requests(d, raw)
        met = C.read_json(d / "metrics.json")
        rows.append({"key": key, "k": k, "r": r, "dir": d, "meta": C.read_json(d / "meta.json"), "met": met,
                     "reqs": reqs, "srv": server_side(d),
                     "active": w4_active(raw, raw["window"]["warmup_s"], raw["window"]["measure_s"])})
    return rows


def report_w4(runner):
    cfg = runner.cfg
    md, js = [], {"config": cfg["_path"], "slo": cfg["slo"], "slo2": cfg["slo2"], "units": {}, "reference": {}}
    a = runner.analyze()
    md.append(f"# Tổng hợp W4 `{runner.out}`\n\nCấu hình `{cfg['_path']}` · tìm k trên **{runner.w4['search_on']}** · "
              f"SLO chính TTFT p95 ≤ {cfg['slo']['ttft_ms'] / 1000:g} s, TPOT p95 ≤ {cfg['slo']['tpot_ms']:g} ms · "
              f"chỉ tính request gửi trong 30 phút đo\n")

    starts = runner.out / "server_starts.jsonl"
    if starts.exists():
        ss = [json.loads(x) for x in starts.read_text().splitlines() if x.strip()]
        md.append("## Các lần khởi động server\n")
        md.append(md_table(["#", "thời điểm", "lý do", "sẵn sàng (s)", "dùng lại", "KV cache (token)", "torch.compile (s)"],
                           [[x["start_index"], x["time"], x["reason"], x["ready_s"], x["reused_running_server"],
                             f(x.get("kv_cache_tokens")), x.get("torch_compile_s", "—")] for x in ss]) + "\n")

    ref = sorted(S.Runner.done_units(runner).items())
    if ref:
        md.append("## Điểm mốc (so máy này với máy P04)\n")
        tab = []
        for (w, n, r, client), d in ref:
            m = C.read_json(d / "metrics.json")["all"]
            js["reference"][C.unit_id(w, n, r, client)] = m
            tab.append([C.unit_id(w, n, r, client), f(m["ttft_p50_ms"]), f(m["ttft_p95_ms"]), f(m["tpot_p50_ms"], 1),
                        f(m["tpot_p95_ms"], 1), f(m["output_tok_s"])])
        md.append(md_table(["đơn vị", "TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95", "tok/s ra"], tab) + "\n")

    rows = collect_w4(runner)
    # --- goodput ---
    md.append("## Goodput W4 (k ngày chồng lên nhau ở khung giờ tìm k)\n")
    slos = [("chính", cfg["slo"]), ("phụ", cfg["slo2"])] + \
           [(f"TTFT ≤ {t / 1000:g} s", {"ttft_ms": t, "tpot_ms": cfg["slo"]["tpot_ms"]}) for t in TTFT_SENS_MS]
    tab, js["goodput"] = [], {}
    search_rows = [x for x in rows if x["key"] == runner.w4["search_on"]]
    for name, slo in slos:
        verdicts = {}
        for x in search_rows:
            m = x["met"]["all"]
            ab = x["met"]["w4"]["aborted"]
            stricter = slo["ttft_ms"] <= cfg["slo"]["ttft_ms"] and slo["tpot_ms"] <= cfg["slo"]["tpot_ms"]
            if ab and not stricter:
                continue  # lần dừng sớm: không biết kết quả với SLO lỏng hơn
            v = C.passes(m, slo) and not ab
            verdicts.setdefault(x["k"], []).append(v)
        verdicts = {k: sum(v) * 2 > len(v) for k, v in verdicts.items()}
        lo, hi = C.interval(verdicts)
        at_lo = next((x for x in search_rows if x["k"] == lo), None)
        js["goodput"][name] = {"lo": lo, "hi": hi, "levels": sorted(verdicts),
                               "req_per_hour_at_lo": at_lo["met"]["w4"]["req_per_hour"] if at_lo else None,
                               "active_sessions_at_lo": at_lo["active"] if at_lo else None}
        tab.append([f"{name} ({slo['ttft_ms'] / 1000:g} s / {slo['tpot_ms']:g} ms)", C.fmt_interval(lo, hi),
                    f(at_lo["met"]["w4"]["req_per_hour"]) if at_lo else "—",
                    f"{at_lo['active'][0]} / {at_lo['active'][1]}" if at_lo else "—"])
    md.append(md_table(["SLO", "goodput k", "req/giờ ở k = lo", "phiên đang mở p50 / max ở k = lo"], tab))
    md.append(f"\nTrạng thái tìm: {a['status']}. SLO phụ / độ nhạy chỉ có độ phân giải của các mức đã đo; "
              "lần dừng sớm (chắc chắn trượt SLO chính) chỉ dùng được cho SLO chặt hơn.\n")

    # --- từng lần chạy ---
    md.append("## Từng lần chạy\n")
    tab = []
    for x in rows:
        m, w, srv = x["met"]["all"], x["met"]["w4"], x["srv"]
        uid = S.w4_uid(x["key"], x["k"], x["r"])
        cache = w4_cache(x["reqs"])
        js["units"][uid] = {"metrics": m, "w4": w, "server": srv, "active": x["active"], "verdict": x["met"]["verdict"],
                            "breakdown": w4_breakdown(x["reqs"]), "interference": w4_interference(x["reqs"]),
                            "cache": cache, "unit_s": x["meta"]["unit_s"]}
        tab.append([uid, w["sessions"], m["n_requests"], f(w["req_per_hour"]), f"{x['active'][0]} / {x['active'][1]}",
                    f"{f(m['ttft_p50_ms'])} / {f(m['ttft_p95_ms'])} / {f(m['ttft_p99_ms'])}",
                    f"{f(m['tpot_p50_ms'], 1)} / {f(m['tpot_p95_ms'], 1)}", f(m["output_tok_s"]), m["failed"],
                    "đạt" if x["met"]["verdict"] else "trượt",
                    f"t = {w['aborted']['t_trace_s']:.0f} s" if w["aborted"] else "—", w["drain_cut"] or "—",
                    f(srv["running_p50_mid"]), f(srv["running_max"]), f(srv["waiting_max"]),
                    f(srv["kv_usage_max"] * 100 if not math.isnan(srv["kv_usage_max"]) else math.nan),
                    f(srv["preemptions"]),
                    f(srv["prefix_hit_rate"] * 100 if not math.isnan(srv["prefix_hit_rate"]) else math.nan, 1),
                    f(cache["cached_token_share"] * 100, 1) if cache else "—",
                    f"{f(w['lag_p50_s'], 2)} / {f(w['lag_p99_s'], 2)} / {f(w['lag_max_s'], 2)}"])
    md.append(md_table(["lần chạy", "phiên", "request đo", "req/giờ", "phiên mở p50 / max",
                        "TTFT p50 / p95 / p99 (ms)", "TPOT p50 / p95 (ms)", "tok/s ra", "lỗi", "SLO chính",
                        "dừng sớm", "cắt sau cửa sổ", "đang chạy p50", "đang chạy max", "chờ max", "KV max %",
                        "preempt", "trúng prefix % (server)", "token trúng cache % (client)",
                        "gửi trễ p50 / p99 / max (s)"], tab) + "\n")

    # --- tách theo loại request ---
    md.append("## Tách theo loại request (TTFT p50 / p95 ms · TPOT p95 ms · số request)\n")
    groups = [g for g, _, _ in LEN_GROUPS] + ["lượt đầu phiên mới", "quay lại phiên cũ (resumed)", "lượt sau (≥ 2)"]
    tab = []
    for x in rows:
        b = js["units"][S.w4_uid(x["key"], x["k"], x["r"])]["breakdown"]
        tab.append([S.w4_uid(x["key"], x["k"], x["r"])] + [
            f"{f(b[g].get('ttft_p50_ms'))} / {f(b[g].get('ttft_p95_ms'))} · {f(b[g].get('tpot_p95_ms'), 1)} · {b[g]['n']}"
            if b[g]["n"] else "—" for g in groups])
    md.append(md_table(["lần chạy"] + groups, tab) + "\n")

    # --- request dài chen request ngắn ---
    md.append("## Request ngắn (< 1k token) có bị request dài (≥ 4k) làm chậm không\n")
    tab = []
    for x in rows:
        it = js["units"][S.w4_uid(x["key"], x["k"], x["r"])]["interference"]
        cell = lambda g, k: f"{f(it[g].get(k), 1 if 'tpot' in k else 0)} ({it[g]['n']})" if it[g]["n"] else "—"
        tab.append([S.w4_uid(x["key"], x["k"], x["r"]), it["n_long"],
                    cell("ttft_with_long", "ttft_p95_ms"), cell("ttft_without_long", "ttft_p95_ms"),
                    cell("tpot_with_long", "tpot_p95_ms"), cell("tpot_without_long", "tpot_p95_ms")])
    md.append(md_table(["lần chạy", "số request dài", "TTFT p95 ngắn: có dài đang prefill", "không có",
                        "TPOT p95 ngắn: có dài đang prefill", "không có"], tab))
    md.append("\nMỗi ô: p95 (số request). \"Có dài đang prefill\" = trong lúc request ngắn chờ token đầu (TTFT) / đang sinh "
              "token (TPOT) có ít nhất một request ≥ 4k token chưa nhận token đầu.\n")

    # --- prefix cache theo think time ---
    md.append("## Prefix cache theo think time (lượt sau, không bỏ bớt lịch sử)\n")
    tab = []
    for x in rows:
        c = js["units"][S.w4_uid(x["key"], x["k"], x["r"])]["cache"]
        if not c:
            continue
        tab.append([S.w4_uid(x["key"], x["k"], x["r"])] + [
            f"{f(b['reuse_hit_mean'] * 100)}% · {f(b['full_hit_share'] * 100)}% · {b['n']}" if b["n"] else "—"
            for b in c["bins"]])
    if tab:
        md.append(md_table(["lần chạy"] + [f"think {a}–{b if b < 10**9 else '∞'} s" for a, b in THINK_BINS], tab))
        md.append("\nMỗi ô: phần lịch sử trúng cache trung bình · % lượt trúng ≥ 90% lịch sử · số lượt.\n")
    else:
        md.append("(không có số token trúng cache theo request — server chưa bật --enable-prompt-tokens-details)\n")

    total = sum(x["meta"]["unit_s"] for x in rows)
    md.append(f"## Thời gian\n\nTổng thời gian các lần chạy W4: {total / 3600:.2f} h "
              f"({len(rows)} lần, trung bình {total / max(1, len(rows)) / 60:.1f} phút).\n")
    js["total_unit_hours"] = total / 3600
    return "\n".join(md), js


def write_csv(rows, path):
    buf = io.StringIO()
    keys = list(rows[0]["m"]["all"].keys()) if rows else []
    skeys = list(rows[0]["srv"].keys()) if rows else []
    wr = csv.writer(buf)
    wr.writerow(["unit", "w", "n", "r", "client", "window"] + keys + skeys + ["unit_s", "start"])
    for x in rows:
        for win in WINDOWS:
            wr.writerow([C.unit_id(x["w"], x["n"], x["r"], x["client"]), x["w"], x["n"], x["r"], x["client"], win]
                        + [x["m"][win][k] for k in keys] + [x["srv"].get(k) for k in skeys]
                        + [x["meta"]["unit_s"], x["meta"]["start"]])
    C.atomic_write(path, buf.getvalue())


def scenario_check(result, scenario):
    """Kiểm tra lần phát lại W4: lượt đầu gửi đúng start_s; lượt sau không sớm hơn (nhận xong lượt trước + think_s)."""
    raw = C.read_json(result)
    sc = raw.get("time_scale", 1.0)
    t0 = raw["t0"]
    sess = {json.loads(x)["sid"]: json.loads(x) for x in open(scenario) if x.strip()}
    if raw.get("days"):  # kịch bản k ngày = kịch bản gốc lọc theo ngày
        keep = {int(x) for x in raw["days"].split(",")}
        sess = {k: v for k, v in sess.items() if v["day"] in keep}
    recs = sorted(zip(raw["sid"], raw["turn"], raw["start_times"], raw["latencies"], raw["errors"]),
                  key=lambda z: (z[0], z[1]))
    late_first, early, late_next, prev = [], 0, [], {}
    for sid, j, st, lat, err in recs:
        s = sess[sid]
        if j == 0:
            late_first.append((st - t0) / sc - s["start_s"])
        else:
            p_end = prev[sid]
            gap = (st - p_end) / sc - s["turns"][j]["think_s"]
            if gap < -1e-3:
                early += 1
            late_next.append(gap)
        prev[sid] = st + lat
    n_in = sum(raw["in_window"])
    exp = sum(1 for s in sess.values() for t in s["turns"]
              if raw["window"]["warmup_s"] <= t["trace_t"] < raw["window"]["warmup_s"] + raw["window"]["measure_s"])
    print(f"request: {len(recs)} (trong cửa sổ đo {n_in}; trace có {exp} lượt trong cửa sổ), lỗi {raw['failed']}")
    print(f"lượt đầu trễ so với start_s (s, thời gian trace): p50 {np.median(late_first):.3f}, "
          f"max {max(late_first):.3f}")
    if late_next:
        print(f"lượt sau: gửi sớm hơn (nhận xong + think) {early} lần; trễ p50 {np.median(late_next):.3f}, "
              f"max {max(late_next):.3f} s")
    dropped = [d for d in raw["dropped"] if d]
    print(f"lượt phải bỏ bớt lịch sử: {len(dropped)}")
    if raw.get("aborted"):  # dừng sớm: request đang chạy bị huỷ → tính là lỗi, không phải lỗi phát lại
        print(f"lần chạy dừng sớm ({raw['aborted']['reason']}) — {raw['failed']} request bị huỷ / lỗi")
        ok = early == 0
    else:
        ok = early == 0 and raw["failed"] == 0
    print("OK" if ok else "CÓ VẤN ĐỀ")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--compare-clients", action="store_true")
    ap.add_argument("--scenario-check", nargs=2, metavar=("RESULT", "SCENARIO"))
    ap.add_argument("--quiet", action="store_true", help="chỉ ghi file, không in")
    args = ap.parse_args()
    if args.scenario_check:
        sys.exit(0 if scenario_check(*args.scenario_check) else 1)
    cfg = C.load_config(args.config)
    cfg.setdefault("workloads", {})
    S.prepare_workloads(cfg)
    if cfg["mode"] == "w4":
        S.prepare_w4(cfg)
        runner = S.W4Runner(cfg, Path(args.out), "")
        rows = []
        md, js = report_w4(runner)
    else:
        runner = S.Runner(cfg, Path(args.out), "")
        rows = collect(runner)
        md, js = report(runner, rows, args.compare_clients)
    C.atomic_write(Path(args.out) / "summary.md", md + "\n")
    C.write_json(Path(args.out) / "summary.json", js)
    if rows:
        write_csv(rows, Path(args.out) / "units.csv")
    if not args.quiet:
        print(md)


if __name__ == "__main__":
    main()
