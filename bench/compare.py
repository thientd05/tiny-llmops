"""So cặp hai lần đo W1–W3 theo từng đơn vị cùng tên (cùng workload, N, lần lặp → cùng seed), P08 việc 4. Chạy ở local.

    .venv/bin/python bench/compare.py RUN_A RUN_B [--window all] [--slo-ttft-ms 10000 --slo-tpot-ms 150]

RUN_A, RUN_B: thư mục kết quả có units.csv (bench/summarize.py ghi). In bảng Markdown: đạt/trượt, TTFT p50/p95, TPOT p95,
token ra/s, preemption, % trúng prefix cache, số request chạy (trung vị phần giữa), và hiệu B so với A (%).
Chỉ so đơn vị có ở cả hai lần (đơn vị riêng của mỗi bên liệt kê ở cuối). W4: chưa hỗ trợ (để trước P10).
"""

import argparse
import csv
import math
from pathlib import Path

COLS = [("ttft_p50_ms", "TTFT p50 (s)", 1e-3, 2), ("ttft_p95_ms", "TTFT p95 (s)", 1e-3, 2),
        ("tpot_p95_ms", "TPOT p95 (ms)", 1, 0), ("output_tok_s", "tok ra/s", 1, 0),
        ("preemptions", "preempt", 1, 0), ("prefix_hit_rate", "trúng cache %", 100, 1),
        ("running_p50_mid", "chạy p50", 1, 0)]


def load(run, window):
    rows = {}
    with open(Path(run) / "units.csv") as f:
        for r in csv.DictReader(f):
            if r["window"] == window:
                rows[r["unit"]] = r
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def fmt(x, scale, nd):
    return "—" if math.isnan(x) else f"{x * scale:,.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--window", default="all")
    ap.add_argument("--slo-ttft-ms", type=float, default=10000)
    ap.add_argument("--slo-tpot-ms", type=float, default=150)
    a = ap.parse_args()
    A, B = load(a.run_a, a.window), load(a.run_b, a.window)
    ok = lambda r: num(r["ttft_p95_ms"]) <= a.slo_ttft_ms and num(r["tpot_p95_ms"]) <= a.slo_tpot_ms and num(r["failed"]) == 0
    key = lambda u: (A.get(u, B.get(u))["w"], int(A.get(u, B.get(u))["n"]), u)
    common = sorted(set(A) & set(B), key=key)
    print(f"A = `{a.run_a}`\nB = `{a.run_b}` (cửa sổ {a.window}; mỗi ô: A → B, hiệu B/A − 1)\n")
    print("| đơn vị | đạt A → B | " + " | ".join(c[1] for c in COLS) + " |")
    print("|---" * (len(COLS) + 2) + "|")
    for u in common:
        cells = []
        for k, _, sc, nd in COLS:
            x, y = num(A[u].get(k)), num(B[u].get(k))
            d = f" ({(y / x - 1) * 100:+.0f}%)" if x and not math.isnan(x) and not math.isnan(y) and k not in ("preemptions",) else ""
            cells.append(f"{fmt(x, sc, nd)} → {fmt(y, sc, nd)}{d}")
        v = lambda r: "đạt" if ok(r) else "trượt"
        print(f"| {u} | {v(A[u])} → {v(B[u])} | " + " | ".join(cells) + " |")
    for name, only in (("chỉ có ở A", sorted(set(A) - set(B), key=key)), ("chỉ có ở B", sorted(set(B) - set(A), key=key))):
        if only:
            print(f"\n{name}: " + ", ".join(f"{u} ({'đạt' if ok((A if name.endswith('A') else B)[u]) else 'trượt'})"
                                             for u in only))


if __name__ == "__main__":
    main()
