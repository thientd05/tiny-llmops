"""Kiểm tra logic tìm goodput (bench/common.search_step) không cần server: mô phỏng "đạt ⇔ N ≤ K" cho mọi K.

    .venv/bin/python bench/test_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

S = {"grid": [8, 16, 32, 64], "min_n": 2, "max_n": 1024, "abs_tol": 2, "rel_tol": 0.10, "warm_start": None}


def simulate(oracle, s=S, repeats=3, samples=10**9, per_user=4):
    """oracle(N, r) -> đạt/trượt của lần lặp r. Quyết định theo trung vị (đa số) như sweep.py."""
    runs = {}  # N -> {r: bool}
    mf = C.max_fit({"num_requests": {"min": 100, "per_user": per_user}}, samples)
    for _ in range(200):
        verdicts, counts = {}, {}
        for n, rr in runs.items():
            counts[n] = len(rr)
            use = range(1, 2 * repeats + 1) if all(r in rr for r in range(1, 2 * repeats + 1)) else \
                range(1, repeats + 1) if all(r in rr for r in range(1, repeats + 1)) else None
            if use:
                vals = sorted(rr[r] for r in use)
                verdicts[n] = vals[len(vals) // 2] if len(vals) % 2 else sum(vals) * 2 > len(vals)
        status, need = C.search_step(verdicts, counts, s, repeats, mf)
        if status != "measuring":
            return status, C.interval(verdicts), sorted(runs)
        for n, q in need.items():
            for r in range(1, q + 1):
                runs.setdefault(n, {}).setdefault(r, oracle(n, r))
    raise AssertionError("không dừng")


def main():
    worst = 0
    for K in range(1, 1100):
        status, (lo, hi), levels = simulate(lambda n, r: n <= K)
        if K < 2:
            assert status == "below_min" and lo is None and hi == 2, (K, status, lo, hi)
        elif K >= 1024:
            assert status == "above_cap" and lo == 1024, (K, status, lo)
        else:
            assert status == "done", (K, status)
            assert lo <= K < hi, (K, lo, hi)
            assert hi - lo <= max(2, 0.1 * lo), (K, lo, hi)
            worst = max(worst, len(levels))
            assert set(S["grid"]) <= set(levels)
    print(f"đạt ⇔ N ≤ K, K = 1..1099: luôn tìm đúng khoảng; nhiều nhất {worst} mức đo")

    for K, expect in [(100, (96, 104)), (5, (4, 6)), (40, (40, 44))]:
        status, (lo, hi), levels = simulate(lambda n, r: n <= K)
        print(f"K = {K:3}: {status}, goodput {lo} ({hi} trượt), các mức đã đo {levels}")
        assert (lo, hi) == expect

    # trần số mẫu: W2 có 1253 mẫu → N tối đa 312 (4 × 312 = 1248); K = 500 → đo 312 rồi dừng 'sample_cap'
    status, (lo, hi), levels = simulate(lambda n, r: n <= 500, samples=1253)
    assert status == "sample_cap" and lo == 312 and hi is None, (status, lo, hi)
    print(f"trần mẫu (1253 mẫu, K = 500): {status}, goodput ≥ {lo}, các mức {levels}")
    # K = 280 nằm giữa 256 và 312 → vẫn tìm nhị phân bình thường
    status, (lo, hi), levels = simulate(lambda n, r: n <= 280, samples=1253)
    assert status == "done" and lo <= 280 < hi <= 312, (status, lo, hi)
    print(f"1253 mẫu, K = 280: {status}, goodput {lo} ({hi} trượt)")

    # đơn điệu bị vi phạm: N = 16 trượt ở 2/3 lần đầu (nhiễu), thật ra K = 40 → đo lặp gấp đôi 16 và 32, hết ngược
    noisy = lambda n, r: (n <= 40) and not (n == 16 and r in (1, 2))
    status, (lo, hi), levels = simulate(noisy)
    print(f"nhiễu ở N=16 (r1, r2 trượt): {status}, goodput {lo} ({hi} trượt)")
    # 16 trượt ở r1,r2 nhưng đạt r3..r6 → trung vị 6 lần (4/6 đạt) = đạt → tiếp tục và tìm ra đúng 40
    assert status == "done" and (lo, hi) == (40, 44), (status, lo, hi)

    # ngược thật (16 luôn trượt, 32 luôn đạt) → dừng 'non_monotonic' sau khi đo lặp gấp đôi
    weird = lambda n, r: (n <= 40) and n != 16
    status, _, levels = simulate(weird)
    assert status == "non_monotonic", status
    print(f"ngược thật ở N=16: {status} (dừng, ghi báo cáo)")

    # khởi đầu nóng từ khoảng của cấu hình trước
    ws = dict(S, warm_start=[40, 44])
    status, (lo, hi), levels = simulate(lambda n, r: n <= 70, s=ws, repeats=1)
    assert status == "done" and lo <= 70 < hi, (status, lo, hi)
    print(f"khởi đầu nóng [40, 44], K = 70: goodput {lo} ({hi} trượt), mức đã đo {levels}")
    print("OK")


if __name__ == "__main__":
    main()
