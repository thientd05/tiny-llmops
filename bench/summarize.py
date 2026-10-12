"""Tổng hợp kết quả sweep: bảng từng đơn vị, goodput dạng khoảng (SLO chính, SLO phụ, độ nhạy theo ngưỡng TTFT),
nhiễu giữa các lần lặp, so client tự viết với vllm bench serve, kiểm tra server (/metrics). Chạy ở local, không cần GPU.

    python bench/summarize.py --config bench/configs/full.yaml --out results/gd1/04-do-baseline-day-du
        → in Markdown; ghi <out>/summary.md, <out>/summary.json, <out>/units.csv
    python bench/summarize.py ... --compare-clients     # thêm bảng client "session" vs "vllm" (cùng w, N, lần lặp)
    python bench/summarize.py --scenario-check RESULT.json.gz SCENARIO.jsonl   # kiểm tra lần phát lại W4

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
    S.prepare_workloads(cfg)
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
