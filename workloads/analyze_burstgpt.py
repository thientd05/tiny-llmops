"""Phân tích trace BurstGPT_3 để thiết kế W4 (tải giống thật). Chạy local, chỉ cần CPU (~10 s).

    .venv/bin/python workloads/analyze_burstgpt.py

Tự tải BurstGPT_3.csv (release v2.0, CC-BY-4.0) vào data/raw/burstgpt/ nếu chưa có.
Chỉ BurstGPT_3 có cột `Session ID` + `Elapsed time`; BurstGPT_1/2 không có.
In ra: tổng quan, độ dài, phiên (lịch sử hội thoại, think time), tải theo giờ, giờ theo phân vị.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data/raw/burstgpt/BurstGPT_3.csv"
URL = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv"
SHA256 = "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"
LEN_BINS = [0, 1000, 2000, 4000, 8000, 10**9]
LEN_LABELS = ["<1k", "1-2k", "2-4k", "4-8k", ">=8k"]


def pct(s, qs):
    return " ".join(f"p{q}={np.percentile(s, q):.0f}" for q in qs)


def bins(s):
    share = pd.cut(s, LEN_BINS, labels=LEN_LABELS, right=False).value_counts(normalize=True)
    return (share.reindex(LEN_LABELS) * 100).round(1).to_dict()


def load():
    if not CSV.exists():
        CSV.parent.mkdir(parents=True, exist_ok=True)
        tmp = CSV.with_suffix(".tmp")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(CSV)
    digest = hashlib.sha256(CSV.read_bytes()).hexdigest()
    if digest != SHA256:
        sys.exit(f"sha256 khác: {digest}")
    df = pd.read_csv(CSV)
    df.columns = ["ts", "sid", "elapsed", "model", "req", "resp", "total", "logtype"]
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def overview(df):
    print("== Tổng quan (mọi request)")
    print("dòng", len(df), "| số ngày", round((df.ts.max() - df.ts.min()) / 86400, 1))
    print("loại log:", df.logtype.value_counts().to_dict())
    print("model:", df.model.value_counts().to_dict())
    print(f"thất bại (resp=0): {(df.resp == 0).mean():.1%}")
    ok = df[df.resp > 0]
    print("req:", pct(ok.req, (50, 90, 99)), "| resp:", pct(ok.resp, (50, 90, 99)))
    print("req theo nhóm %:", bins(ok.req))


def sessions(c):
    print("\n== Phiên (Conversation log)")
    turns = c.groupby("sid").size()
    print("số phiên", len(turns), "| lượt/phiên", pct(turns, (50, 75, 90, 99)),
          f"| TB {turns.mean():.2f} | nhiều lượt {(turns > 1).mean():.1%}")
    ok = c[c.resp > 0]
    print("req theo nhóm %:", bins(ok.req), "| resp:", pct(ok.resp, (25, 50, 75, 90, 99)))
    print("req theo lượt (p50 / p90):")
    for t, s in ok.groupby("turn").req:
        if t in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
            print(f"  lượt {t:>2}: n={len(s):>6} {s.median():>5.0f} / {np.percentile(s, 90):.0f}")

    g = c.groupby("sid")
    f = c.assign(prev_req=g.req.shift(), prev_resp=g.resp.shift(),
                 prev_end=g.ts.shift() + g.elapsed.shift()).dropna(subset=["prev_req"])
    f = f[f.prev_resp > 0]
    hist = f.prev_req + f.prev_resp
    print(f"lượt sau có req lớn hơn lượt trước: {(f.req > f.prev_req).mean():.1%}"
          f" | corr(req, req_trước + resp_trước) = {np.corrcoef(f.req, hist)[0, 1]:.3f}")
    print("think time (s) = ts − (ts_trước + elapsed_trước):", pct(f.ts - f.prev_end, (10, 25, 50, 75, 90, 95)))


def load_profile(c, t0, hours):
    print("\n== Tải hội thoại theo thời gian")
    ph = ((c.ts - t0) // 3600).astype(int).value_counts().reindex(range(hours), fill_value=0)
    pm = ((c.ts - t0) // 60).astype(int).value_counts().reindex(range(hours * 60), fill_value=0)
    print("req/giờ:", pct(ph, (10, 50, 90, 95, 99)), "| max", int(ph.max()),
          f"| p95/p50 {np.percentile(ph, 95) / np.percentile(ph, 50):.1f}")
    print("req/phút:", pct(pm, (50, 90, 99)), "| max", int(pm.max()))
    hod = ((c.ts // 3600) % 24).astype(int).value_counts().sort_index() / (hours / 24)
    print("req TB theo giờ trong ngày:", hod.round(0).astype(int).to_dict())
    long_share = c[c.resp > 0].assign(h=lambda d: ((d.ts - t0) // 3600).astype(int), lg=lambda d: d.req >= 2000)
    per_h = long_share.groupby("h").agg(n=("req", "size"), lg=("lg", "mean"))
    per_h = per_h[per_h.n >= 50]
    print(f"% request ≥2k token mỗi giờ (giờ có ≥50 request, {len(per_h)} giờ):", pct(per_h.lg * 100, (10, 50, 90)))
    print("giờ theo phân vị tải:")
    for q in (10, 50, 95):
        v = np.percentile(ph, q)
        idx = int((ph - v).abs().idxmin())
        print(f"  p{q}: ~{v:.0f} req/giờ, ví dụ giờ #{idx} (ngày {idx // 24}, {idx % 24}h) = {int(ph[idx])} req")


def main():
    df = load()
    overview(df)
    t0 = df.ts.min()
    hours = int((df.ts.max() - t0) // 3600) + 1
    c = df[df.logtype == "Conversation log"].sort_values(["sid", "ts"], kind="stable").copy()
    c["turn"] = c.groupby("sid").cumcount() + 1
    sessions(c)
    load_profile(c, t0, hours)
    api = df[df.logtype == "API log"]
    ph = ((api.ts - t0) // 3600).astype(int).value_counts().reindex(range(hours), fill_value=0)
    print("\n== API log (chương trình gọi, không có phiên): req/giờ", pct(ph, (10, 50, 95, 99)), "| max", int(ph.max()))


if __name__ == "__main__":
    main()
