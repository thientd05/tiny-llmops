"""Kiểm tra bộ chấm FinQA và KL xấp xỉ của quality/analyze.py (không cần server).

    .venv/bin/python quality/test_analyze.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import finqa_correct, kl_topk, parse_answer  # noqa: E402

CASES = [  # (câu trả lời, đáp án gốc exe_ans, đúng ở dung sai 1%?)
    ("The net change is 94 [Document 1].\nAnswer: 94", 94.0, True),
    ("...\nAnswer: 14.46%", 0.14464, True),            # phần trăm vs phân số
    ("...\nAnswer: 14.5", 0.14464, True),              # làm tròn, lệch 0.25%
    ("...\nAnswer: 15", 0.14464, False),               # lệch 3.7%
    ("...\nAnswer: 0.1446", 0.14464, True),
    ("...\nAnswer: $1,234.5", 1234.5, True),
    ("...\nAnswer: (35)", -35.0, True),                # số âm kiểu kế toán
    ("...\nAnswer: -0.04365", -0.04365, True),
    ("...\nAnswer: -4.4%", -0.04365, True),
    ("...\nAnswer: Yes", "yes", True),
    ("...\nAnswer: no.", "yes", False),
    ("The result is 94 [Document 2].", 94.0, True),    # không có dòng Answer: → lấy số cuối
    ("Answer: 10\nmore text\nAnswer: 20", 20.0, True), # lấy dòng Answer: cuối
    ("Answer: unknown", 5.0, False),
]


def main():
    bad = 0
    for text, gold, want in CASES:
        pred = parse_answer(text)
        got = finqa_correct(pred, gold, 0.01)
        ok = got == want
        bad += not ok
        print(f"{'OK ' if ok else 'SAI'} pred={pred!r:<12} gold={gold!r:<10} đúng@1%={got}")
    # KL: hai phân phối giống nhau → 0; khác → > 0; độ phủ = tổng xác suất top-k
    p = [[1, math.log(0.7)], [2, math.log(0.2)], [3, math.log(0.05)]]
    kl0, cov = kl_topk(p, p)
    kl1, _ = kl_topk(p, [[1, math.log(0.5)], [2, math.log(0.4)], [3, math.log(0.05)]])
    print(f"KL giống nhau = {kl0:.2e}, khác nhau = {kl1:.4f}, độ phủ = {cov:.2f}")
    bad += not (kl0 < 1e-9 and kl1 > 0 and abs(cov - 0.95) < 1e-9)
    print("OK" if not bad else f"{bad} LỖI")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
