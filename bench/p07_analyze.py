"""P07: phân tích thêm trên kết quả P04/P05 đã có (không cần GPU).

In ra các bảng dùng trong báo cáo P07 (docs/gd1-vllm-baseline/07-tong-hop-baseline/bao-cao.md):
  1. W1–W3: trung vị [min–max] 3 lần lặp theo mức + chi phí VND / 1M token tại goodput (giá tham chiếu + khoảng giá)
  2. Nhiễu (max − min) / trung vị của TTFT TB / p50 / p95, E2EL, throughput
  3. p95 gộp 3 lần lặp + CI bootstrap (coi request độc lập → lạc quan)
  4. Bộ đếm iteration_tokens của vLLM có tính token tính lại sau preemption không
  5. W4: phần prompt là lịch sử dùng lại được (tổng và theo phút)
  6. W4: TTFT request ngắn theo độ dài hàng đợi × có request dài đang chờ token đầu
  7. W4: chi phí cả ngày theo giá

Chạy: python bench/p07_analyze.py [--price 8770]
"""
import argparse
import csv
import gzip
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

P04 = Path("results/gd1/04-do-baseline-day-du/20261013-p04")
P05 = Path("results/gd1/05-w4-tai-giong-that/20261013-p05")
GOODPUT = {"w1": 68, "w2": 18, "w3": 6}
NEAR = [("w1", 64), ("w1", 68), ("w1", 72), ("w2", 16), ("w2", 18), ("w2", 20), ("w2", 24),
        ("w3", 4), ("w3", 6), ("w3", 8)]


def p04_levels():
    rows = [r for r in csv.DictReader(open(P04 / "units.csv")) if r["window"] == "all"]
    g = defaultdict(list)
    for r in rows:
        g[(r["w"], int(r["n"]))].append(r)
    return g


def rng_str(xs, nd=0):
    return f"{st.median(xs):,.{nd}f} [{min(xs):,.{nd}f}–{max(xs):,.{nd}f}]"


def spread(xs):
    m = st.median(xs)
    return (max(xs) - min(xs)) / m if m else 0.0


def section_levels(g, price, lo, hi):
    print("## 1. W1–W3 theo mức (trung vị [min–max] 3 lần lặp)")
    for k in sorted(g):
        v = g[k]
        col = lambda m, sc=1.0: [float(x[m]) * sc for x in v]
        print(f"{k[0]} {k[1]:>3} | TTFT p95 s {rng_str(col('ttft_p95_ms', 1e-3), 2)} | TPOT p95 ms {rng_str(col('tpot_p95_ms'))}"
              f" | ra/s {rng_str(col('output_tok_s'))} | vào/s {rng_str(col('input_tok_s'))} | req/s {rng_str(col('req_s'), 3)}")
    print("\n## 1b. Chi phí tại goodput")
    for w, n in GOODPUT.items():
        v = g[(w, n)]
        out = [float(x["output_tok_s"]) for x in v]
        inp = [float(x["input_tok_s"]) for x in v]
        req = [float(x["req_s"]) for x in v]
        per_m = lambda p, tok: p / (tok * 3600 / 1e6)
        print(f"{w}@{n}: VND/1M ra {per_m(price, st.median(out)):,.0f} [{per_m(price, max(out)):,.0f}–{per_m(price, min(out)):,.0f}]"
              f" | giá {lo}…{hi}: {per_m(lo, st.median(out)):,.0f} … {per_m(hi, st.median(out)):,.0f}"
              f" | VND/1M vào+ra {per_m(price, st.median(out) + st.median(inp)):,.0f}"
              f" | VND/1k request {price / (st.median(req) * 3.6):,.0f}")


def section_noise(g):
    print("\n## 2. Nhiễu (max − min) / trung vị giữa 3 lần lặp: tất cả mức (sát ngưỡng)")
    for m in ["ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "server_queue_mean_ms", "e2el_mean_ms",
              "output_tok_s", "tpot_p95_ms"]:
        allv = sorted(spread([float(x[m]) for x in v]) for v in g.values())
        near = [spread([float(x[m]) for x in g[k]]) for k in NEAR]
        print(f"{m:22s} trung vị {st.median(allv):.1%} p75 {allv[int(0.75 * len(allv))]:.1%} max {max(allv):.1%}"
              f" (sát ngưỡng: trung vị {st.median(near):.1%}, max {max(near):.1%})")


def section_pooled():
    print("\n## 3. TTFT p95 gộp 3 lần lặp, CI bootstrap 2,000 lần")
    rng = np.random.default_rng(0)
    for w, n in NEAR:
        allt, per = [], []
        for r in (1, 2, 3):
            d = json.load(gzip.open(P04 / "units" / f"{w}-c{n}-r{r}" / "bench.json.gz"))
            t = [x for x in d["ttfts"] if x]
            allt += t
            per.append(np.percentile(t, 95))
        a = np.array(allt)
        bs = [np.percentile(rng.choice(a, len(a)), 95) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{w}/{n}: n {len(a)} gộp {np.percentile(a, 95):.2f} s [{lo:.2f}–{hi:.2f}] | trung vị 3 lần {np.median(per):.2f}"
              f" | từng lần {', '.join(f'{p:.2f}' for p in per)}")


def metric(path, name):
    for line in open(path):
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def section_recompute():
    print("\n## 4. iteration_tokens − (prompt − cache hit + generation), theo đơn vị P04")
    for u in sorted((P04 / "units").iterdir()):
        a, b = u / "metrics_after.txt", u / "metrics_before.txt"
        if not a.exists():
            continue
        d = lambda n: metric(a, n) - metric(b, n)
        it = d("vllm:iteration_tokens_total_sum")
        expect = d("vllm:prompt_tokens_total") - d("vllm:prefix_cache_hits_total") + d("vllm:generation_tokens_total")
        print(f"{u.name}: preempt {d('vllm:num_preemptions_total'):.0f} iteration {it:.0f} kỳ vọng {expect:.0f} lệch {it - expect:+.0f}")


def load_w4(unit):
    return json.load(gzip.open(P05 / "units" / unit / "bench.json.gz"))


def history_share(d):
    """Trả về (tổng prompt, phần dùng lại được, phần là câu trả lời trước, số lượt sau, số lượt bị cắt) và theo phút."""
    n = len(d["sid"])
    last = {}
    tot = hist = ans = later = trunc = 0
    per_min = defaultdict(lambda: [0, 0])
    warm = d["window"]["warmup_s"] if isinstance(d["window"], dict) else 600
    for i in sorted(range(n), key=lambda i: d["start_times"][i]):
        if d["errors"][i] or d["ttfts"][i] is None:
            continue
        s, inl = d["sid"][i], d["input_lens"][i]
        h = a = 0
        if s in last:
            later += 1
            pi, po = last[s]
            if inl >= pi + po:
                h, a = pi + po - 4, po
            else:
                trunc += 1
        tot += inl
        hist += h
        ans += a
        last[s] = (inl, d["output_lens"][i])
        m = int((d["t_rel"][i] - warm) // 60)
        if 0 <= m < 30:
            per_min[m][0] += inl
            per_min[m][1] += h
    return tot, hist, ans, later, trunc, per_min


def section_history():
    print("\n## 5. W4: phần prompt là lịch sử dùng lại được")
    for u in ["w4-peak-s1-k16-r1", "w4-peak-s1-k18-r1", "w4-peak-s2-k16-r1", "w4-peak2-s1-k18-r1", "w4-normal-s1-k18-r1"]:
        d = load_w4(u)
        cached = sum(c or 0 for c in d["cached"])
        tot, hist, ans, later, trunc, per_min = history_share(d)
        print(f"{u}: request {len(d['sid'])} | trúng cache {cached / tot:.1%} | dùng lại được {hist / tot:.1%}"
              f" | câu trả lời trước {ans / tot:.1%} | lượt sau {later}, bị cắt {trunc}")
        if u == "w4-peak-s1-k18-r1":
            print("   theo phút: " + "; ".join(f"{m}: {int(p)} tok {h / p:.0%}" for m, (p, h) in sorted(per_min.items())))


def section_hol():
    print("\n## 6. W4: TTFT request < 1k theo hàng đợi × có request ≥ 4k đang chờ token đầu")
    for u in ["w4-peak-s1-k16-r1", "w4-peak-s1-k18-r1", "w4-peak-s2-k16-r1"]:
        d = load_w4(u)
        ts = [json.loads(l) for l in open(P05 / "units" / u / "metrics_timeseries.jsonl")]
        off = next(x["t"] for x in ts if x["vllm:num_requests_running"] > 0) - min(d["t_rel"])
        waiting = {int(round(x["t"])): x["vllm:num_requests_waiting"] for x in ts}
        n = len(d["sid"])
        longs = [(d["t_rel"][i], d["t_rel"][i] + (d["ttfts"][i] or 1e9)) for i in range(n) if d["input_lens"][i] >= 4000]
        rows = []
        for i in range(n):
            if not d["in_window"][i] or d["ttfts"][i] is None or d["input_lens"][i] >= 1000:
                continue
            t = d["t_rel"][i]
            rows.append((waiting.get(int(round(t + off)), float("nan")), any(a <= t < b for a, b in longs), d["ttfts"][i]))
        print(f"{u} (lệch giờ {off:.0f} s)")
        for lo, hi in [(0, 0), (1, 3), (4, 9), (10, 10**6)]:
            for has_long in (False, True):
                x = [r[2] for r in rows if lo <= r[0] <= hi and r[1] == has_long]
                if x:
                    print(f"  chờ {lo}–{hi if hi < 10**6 else '∞'} có dài={has_long}: n {len(x)} p50 {np.median(x):.2f} p95 {np.percentile(x, 95):.2f}")


def section_w4_cost(price, lo, hi):
    print("\n## 7. W4 chi phí cả ngày")
    c = json.load(open(P05 / "cost.json"))
    per_k_tok, per_k_req = c["trace_out_tokens_per_day_per_k"], c["trace_req_per_day_per_k"]
    for p in (price, lo, hi, c["price_vnd_h"]):
        for k in (18, 16):
            day = p * 24
            print(f"giá {p:,.0f} k={k}: {day / (per_k_tok * k / 1e6):,.0f} VND/1M ra | {day / (per_k_req * k / 1e3):,.0f} VND/1k request")
    for u in ["w4-peak-s1-k18-r1", "w4-peak2-s1-k18-r1", "w4-normal-s1-k18-r1", "w4-peak-s2-k16-r1"]:
        d = load_w4(u)
        idx = [i for i in range(len(d["sid"])) if d["in_window"][i] and not d["errors"][i]]
        o = sum(d["output_lens"][i] for i in idx)
        m = sum(d["max_tokens"][i] for i in idx)
        print(f"{u}: token ra {o} / max_tokens {m}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=8770, help="giá tham chiếu VND/h")
    ap.add_argument("--price-lo", type=float, default=7560)
    ap.add_argument("--price-hi", type=float, default=9900)
    a = ap.parse_args()
    g = p04_levels()
    section_levels(g, a.price, a.price_lo, a.price_hi)
    section_noise(g)
    section_pooled()
    section_recompute()
    section_history()
    section_hol()
    section_w4_cost(a.price, a.price_lo, a.price_hi)


if __name__ == "__main__":
    main()
