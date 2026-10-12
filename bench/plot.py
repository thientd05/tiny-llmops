"""Vẽ hình cho báo cáo sweep (P04): TTFT p95 / TPOT p95 theo số user, goodput theo ngưỡng TTFT. Chạy ở local.

    python bench/plot.py --config bench/configs/full.yaml --out results/gd1/04-do-baseline-day-du/<run> --fig-dir docs/.../hinh

Điểm = trung vị các lần lặp (client vllm, cửa sổ quyết định), vạch dọc = min–max giữa các lần lặp.
Màu: 3 slot đầu của bảng màu tham chiếu (đã kiểm CVD), mỗi workload thêm hình dấu riêng + nhãn trực tiếp.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import summarize as S  # noqa: E402
import sweep as SW  # noqa: E402

SERIES = {"w1": ("#2a78d6", "o", "W1 chat"), "w2": ("#eb6834", "s", "W2 RAG"), "w3": ("#1baf7a", "^", "W3 tóm tắt")}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def per_level(rows, w, key, win):
    lv = {}
    for x in rows:
        if x["w"] == w and x["client"] == "vllm":
            lv.setdefault(x["n"], []).append(x["m"][win][key])
    ns = sorted(lv)
    return ns, [float(np.median(lv[n])) for n in ns], [min(lv[n]) for n in ns], [max(lv[n]) for n in ns]


def metric_fig(rows, cfg, key, scale, ylabel, refs, title, path, logy, mark_goodput=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    win = cfg["verdict_window"]
    for w, (col, mk, name) in SERIES.items():
        if w not in cfg["workloads"]:
            continue
        ns, med, lo, hi = per_level(rows, w, key, win)
        if not ns:
            continue
        med, lo, hi = [np.array(v) / scale for v in (med, lo, hi)]
        ax.vlines(ns, lo, hi, color=col, linewidth=2, alpha=0.45)
        ax.plot(ns, med, color=col, linewidth=2, marker=mk, markersize=7, markeredgecolor=SURF,
                markeredgewidth=1.5, label=name)
        ax.annotate(name, (ns[-1], med[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9,
                    va="center")
        if mark_goodput:
            lo = S.goodput_for(rows, w, cfg["slo"], win, cfg["repeats"])[0]
            if lo in ns:
                ax.plot([lo], [med[ns.index(lo)]], marker="o", markersize=15, markerfacecolor="none",
                        markeredgecolor=col, markeredgewidth=1.5)
                ax.annotate(f"goodput {lo}", (lo, med[ns.index(lo)]), xytext=(-8, 9), textcoords="offset points",
                            ha="right", color=INK, fontsize=8.5)
    for y, txt, ls in refs:
        ax.axhline(y, color=INK2, linewidth=1.2, linestyle=ls)
        ax.annotate(txt, (0, y), xycoords=("axes fraction", "data"), xytext=(4, 3), textcoords="offset points",
                    ha="left", va="bottom", color=INK2, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Số user đồng thời N (closed-loop)", color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def sens_fig(rows, cfg, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    style(ax)
    ths = [t / 1000 for t in S.TTFT_SENS_MS]
    R = cfg["repeats"]
    notes = []
    for w, (col, mk, name) in SERIES.items():
        if w not in cfg["workloads"]:
            continue
        xs, ys, below = [], [], []
        for t in S.TTFT_SENS_MS:
            lo, hi, _, _ = S.goodput_for(rows, w, {"ttft_ms": t, "tpot_ms": cfg["slo"]["tpot_ms"]},
                                        cfg["verdict_window"], R)
            if lo is None:  # mọi mức đã đo đều trượt → chỉ biết "< hi", không vẽ thành 0
                below.append((t / 1000, hi))
            else:
                xs.append(t / 1000)
                ys.append(lo)
        ax.plot(xs, ys, color=col, linewidth=2, marker=mk, markersize=7, markeredgecolor=SURF, markeredgewidth=1.5,
                label=name)
        off, ha = ((9, -3), "left") if w == "w3" else ((0, 7), "center")
        for t, y in zip(xs, ys):
            ax.annotate(f"{y}", (t, y), xytext=off, textcoords="offset points", ha=ha, color=INK, fontsize=8.5)
        for t, hi in below:
            notes.append(f"{name}: ngưỡng {t:g} s → < {hi} (mức nhỏ nhất đã đo vẫn trượt)")
    ax.axvline(10, color=INK2, linewidth=1.2)
    ax.annotate("SLO chính 10 s", (10, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                textcoords="offset points", color=INK2, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks(ths)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if notes:
        ax.text(0.02, 0.62, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", color=INK2, fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_xlabel("Ngưỡng TTFT p95 (TPOT p95 ≤ 150 ms)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Goodput (lo, số user)", color=INK2, fontsize=9.5)
    fig.suptitle("Độ nhạy: goodput theo ngưỡng TTFT (chỉ các mức đã đo)", color=INK, fontsize=11, x=0.02, ha="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", required=True)
    a = ap.parse_args()
    cfg = C.load_config(a.config)
    SW.prepare_workloads(cfg)
    rows = S.collect(SW.Runner(cfg, Path(a.out), ""))
    fd = Path(a.fig_dir)
    fd.mkdir(parents=True, exist_ok=True)
    slo, slo2 = cfg["slo"], cfg["slo2"]
    metric_fig(rows, cfg, "ttft_p95_ms", 1000, "TTFT p95 (s, thang log)",
               [(slo["ttft_ms"] / 1000, f"SLO chính {slo['ttft_ms'] / 1000:g} s", "-"),
                (slo2["ttft_ms"] / 1000, f"SLO phụ {slo2['ttft_ms'] / 1000:g} s", "--")],
               "TTFT p95 theo số user (trung vị 3 lần, vạch = min–max, vòng = goodput)", fd / "h1-ttft-p95.png", True,
               mark_goodput=True)
    metric_fig(rows, cfg, "tpot_p95_ms", 1, "TPOT p95 (ms)",
               [(slo["tpot_ms"], f"SLO chính {slo['tpot_ms']:g} ms", "-"),
                (slo2["tpot_ms"], f"SLO phụ {slo2['tpot_ms']:g} ms", "--")],
               "TPOT p95 theo số user (trung vị 3 lần, vạch = min–max)", fd / "h2-tpot-p95.png", False)
    sens_fig(rows, cfg, fd / "h3-do-nhay-ttft.png")
    metric_fig(rows, cfg, "output_tok_s", 1, "Token ra / giây", [],
               "Throughput ra theo số user (trung vị 3 lần, vạch = min–max)", fd / "h4-throughput.png", False)
    print("đã vẽ:", ", ".join(p.name for p in sorted(fd.glob("h*.png"))))


if __name__ == "__main__":
    main()
