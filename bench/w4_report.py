"""Hình và chi phí cả ngày cho báo cáo W4 (P05). Chạy ở LOCAL (cần trace BurstGPT trong data/raw để tính chi phí).

    .venv/bin/python bench/w4_report.py --config bench/configs/w4.yaml --out results/gd1/05-w4-tai-giong-that \
        --fig-dir docs/gd1-vllm-baseline/05-w4-tai-giong-that/hinh [--price-vnd-h 7570]

Hình 1: TTFT p95 và TPOT p95 theo k (kịch bản tìm k), kẻ đường SLO; lần dừng sớm vẽ điểm rỗng (giá trị là cận dưới).
Hình 2: lần chạy ở k = lo, theo từng phút trong cửa sổ đo: số request gửi (trên), TTFT p95 (giữa), TPOT p95 (dưới).
Chi phí: GPU thuê theo giờ chạy cả ngày. Triển khai cho k người dùng (k = lo) phục vụ trong ngày đúng lượng token của
k ngày trace chồng lên nhau ở từng giờ. VND / 1 triệu token ra (cả ngày) = 24 × giá/giờ ÷ (k × Σ_giờ token ra TB của trace).
So với chỉ tính giờ cao điểm: giá/giờ ÷ (token ra đo được trong 1 giờ cao điểm ở k = lo).
"""

import argparse
import json
import math
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
from plot import INK, INK2, SERIES, SURF, style  # noqa: E402

COL = SERIES["w2"][0]  # một chuỗi mỗi hình → một màu (slot đã kiểm CVD của plot.py)
MAX_OUTPUT = 4096       # như workloads/build_w4.py


def load(cfg_path, out):
    cfg = C.load_config(cfg_path)
    cfg.setdefault("workloads", {})
    SW.prepare_workloads(cfg)
    SW.prepare_w4(cfg)
    runner = SW.W4Runner(cfg, Path(out), "")
    return cfg, runner, S.collect_w4(runner)


def fig_vs_k(cfg, runner, rows, path):
    sr = sorted((x for x in rows if x["key"] == runner.w4["search_on"]), key=lambda x: x["k"])
    lo = runner.analyze()["lo"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, key, scale, unit, slo, slo2 in (
            (axes[0], "ttft_p95_ms", 1000, "TTFT p95 (s, thang log)", cfg["slo"]["ttft_ms"] / 1000,
             cfg["slo2"]["ttft_ms"] / 1000),
            (axes[1], "tpot_p95_ms", 1, "TPOT p95 (ms)", cfg["slo"]["tpot_ms"], cfg["slo2"]["tpot_ms"])):
        style(ax)
        ks = [x["k"] for x in sr]
        vals = [x["met"]["all"][key] / scale for x in sr]
        ab = [bool(x["met"]["w4"]["aborted"]) for x in sr]
        finite = [v for v in vals if math.isfinite(v)] + [slo]
        cap = max(finite) * 2.5 if key.startswith("ttft") else max(finite) * 1.4
        ys = [min(v, cap) for v in vals]  # p95 = ∞ (hơn 5% request lỗi / bị huỷ) → vẽ ở mép trên, nhãn ∞
        ax.plot(ks, ys, color=COL, linewidth=2)
        for k, v, y, a in zip(ks, vals, ys, ab):
            ax.plot([k], [y], marker="^" if v > cap else "o", markersize=8, markeredgewidth=1.5,
                    markeredgecolor=COL if a else SURF, markerfacecolor=SURF if a else COL)
            txt = "∞" if not math.isfinite(v) else f"{v:.3g}"
            ax.annotate(txt + (" (dừng sớm)" if a else ""), (k, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", color=INK, fontsize=8)
        if lo in ks:
            ax.axvline(lo, color=INK2, linewidth=1, linestyle=":")
        for y, txt, ls in ((slo, f"SLO chính {slo:g}", "-"), (slo2, f"SLO phụ {slo2:g}", "--")):
            ax.axhline(y, color=INK2, linewidth=1.2, linestyle=ls)
            ax.annotate(txt, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                        ha="right", color=INK2, fontsize=8.5)
        if key.startswith("ttft"):
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel(unit, color=INK2, fontsize=9.5)
    axes[1].set_xlabel(f"k (số ngày chồng lên nhau, khung {runner.w4['search_on']}); vạch chấm = goodput",
                       color=INK2, fontsize=9.5)
    fig.suptitle("W4: TTFT p95 và TPOT p95 theo k\n(điểm rỗng = dừng sớm vì đã chắc trượt; số là của phần đã chạy)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def fig_timeline(cfg, x, path):
    raw = C.read_json(x["dir"] / "bench.json.gz")
    warm, meas = raw["window"]["warmup_s"], raw["window"]["measure_s"]
    reqs = x["reqs"]
    mins = np.arange(0, meas // 60)
    by = [[r for r in reqs if warm + 60 * m <= r["t_rel"] < warm + 60 * (m + 1)] for m in mins]
    cnt = [len(b) for b in by]
    ttft = [C.compute_metrics(b, 0)["ttft_p95_ms"] / 1000 if b else math.nan for b in by]
    tpot = [C.compute_metrics(b, 0)["tpot_p95_ms"] if b else math.nan for b in by]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, ys, lab, slo in ((axes[0], cnt, "request gửi / phút", None),
                             (axes[1], ttft, "TTFT p95 (s)", cfg["slo"]["ttft_ms"] / 1000),
                             (axes[2], tpot, "TPOT p95 (ms)", cfg["slo"]["tpot_ms"])):
        style(ax)
        if ax is axes[0]:
            ax.bar(mins + 0.5, ys, width=0.8, color=COL)
        else:
            ax.plot(mins + 0.5, [min(y, 1e6) for y in ys], color=COL, linewidth=2, marker="o", markersize=4)
        if slo is not None:
            ax.axhline(slo, color=INK2, linewidth=1.2)
            ax.annotate(f"SLO {slo:g}", (0, slo), xycoords=("axes fraction", "data"), xytext=(4, 3),
                        textcoords="offset points", color=INK2, fontsize=8.5)
        ax.set_ylabel(lab, color=INK2, fontsize=9.5)
    axes[2].set_xlabel("phút trong cửa sổ đo", color=INK2, fontsize=9.5)
    fig.suptitle(f"W4 {SW.w4_uid(x['key'], x['k'], x['r'])}: theo từng phút (p95 của request gửi trong phút đó)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def trace_hour_profile():
    """Token ra trung bình mỗi giờ trong ngày của phần hội thoại BurstGPT (chặn MAX_OUTPUT như kịch bản)."""
    sys.path.insert(0, str(C.ROOT / "workloads"))
    from build_w4 import conversation_sessions
    sessions, _, n_days = conversation_sessions()
    out, req = np.zeros(24), np.zeros(24)
    for s in sessions:
        h = ((s["ts"] // 3600) % 24).astype(int)
        np.add.at(out, h, np.minimum(s["resp"], MAX_OUTPUT))
        np.add.at(req, h, 1)
    return out / n_days, req / n_days


def cost(cfg, runner, rows, price_vnd_h):
    a = runner.analyze()
    lo = a["lo"]
    if lo is None:
        return None
    out_h, req_h = trace_hour_profile()
    day_tok = lo * out_h.sum()
    res = {"k": lo, "price_vnd_h": price_vnd_h, "trace_out_tokens_per_day_per_k": float(out_h.sum()),
           "trace_req_per_day_per_k": float(req_h.sum()),
           "day_out_tokens": float(day_tok), "vnd_per_mtok_day": 24 * price_vnd_h / (day_tok / 1e6),
           "hourly_out_tokens_expected": {h: float(lo * out_h[h]) for h in range(24)}, "measured": {}}
    for x in rows:
        if x["k"] != lo or not x["met"]["verdict"]:
            continue  # chỉ lần chạy đạt SLO ở k = lo mới cho biết lượng token phục vụ được
        reqs = [r for r in x["reqs"] if r["ok"]]
        meas = C.read_json(x["dir"] / "bench.json.gz")["window"]["measure_s"]
        tok_h = sum(r["out"] for r in reqs) * 3600 / meas
        hour = runner.w4["scenarios"][x["key"]]["summary"]["hour_of_day"]
        res["measured"][x["key"]] = {"hour": hour, "out_tokens_per_hour": tok_h,
                                     "expected_from_trace": float(lo * out_h[hour]),
                                     "vnd_per_mtok_if_all_hours_like_this": price_vnd_h / (tok_h / 1e6)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--price-vnd-h", type=float, default=None,
                    help="giá thuê thật / giờ (mặc định: price_usd_per_h của cấu hình × 26,200)")
    ap.add_argument("--no-cost", action="store_true", help="bỏ phần chi phí (không có trace ở máy này)")
    a = ap.parse_args()
    cfg, runner, rows = load(a.config, a.out)
    fd = Path(a.fig_dir)
    fd.mkdir(parents=True, exist_ok=True)
    fig_vs_k(cfg, runner, rows, fd / "h1-ttft-tpot-theo-k.png")
    lo = runner.analyze()["lo"]
    x = next((x for x in rows if x["key"] == runner.w4["search_on"] and x["k"] == lo), None)
    if x:
        fig_timeline(cfg, x, fd / "h2-theo-phut-k-lo.png")
    print("đã vẽ:", ", ".join(p.name for p in sorted(fd.glob("h*.png"))))
    if not a.no_cost:
        price = a.price_vnd_h or cfg.get("price_usd_per_h", 0.30) * 26200
        c = cost(cfg, runner, rows, price)
        C.write_json(Path(a.out) / "cost.json", c)
        if c:
            print(f"k = {c['k']}: {c['day_out_tokens'] / 1e6:.2f} triệu token ra / ngày → "
                  f"{c['vnd_per_mtok_day']:,.0f} VND / 1 triệu token (cả ngày, giá {price:,.0f} VND/h)")
            for key, m in c["measured"].items():
                print(f"  {key} ({m['hour']}h): đo {m['out_tokens_per_hour']:,.0f} token ra/giờ, trace × k "
                      f"{m['expected_from_trace']:,.0f}; nếu cả ngày như giờ này: "
                      f"{m['vnd_per_mtok_if_all_hours_like_this']:,.0f} VND / 1 triệu token")


if __name__ == "__main__":
    main()
