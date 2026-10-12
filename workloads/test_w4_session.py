"""Kiểm tra workloads/w4_session.assemble: lịch sử là prefix của lượt sau, và cắt lịch sử khi vượt ngân sách.

    .venv/bin/python workloads/test_w4_session.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import Renderer  # noqa: E402
from w4_session import assemble  # noqa: E402


def common_prefix(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main():
    R = Renderer()
    ids = lambda msgs: R.tok(R.render(msgs), add_special_tokens=False).input_ids
    count = lambda msgs: len(ids(msgs))

    history, prev = [], None
    for k in range(5):
        msgs, dropped = assemble(history, f"Question {k}: " + "why is the sky blue? " * 20, 10**6, count)
        cur = ids(msgs)
        assert dropped == 0
        if prev is not None:
            shared = common_prefix(prev, cur)
            # prompt lượt trước chỉ khác ở vài token cuối (khối <think></think> rỗng của lượt đang sinh)
            assert shared >= len(prev) - 8, (k, shared, len(prev))
            print(f"lượt {k + 1}: {len(cur)} token, dùng lại prefix {shared}/{len(prev)} của prompt lượt trước")
        history.append((msgs[-1]["content"], f"Answer {k}: " + "because of Rayleigh scattering. " * 30))
        prev = cur

    budget = count(msgs) // 2
    msgs2, dropped = assemble(history, "One more question?", budget, count)
    assert dropped > 0 and count(msgs2) <= budget, (dropped, count(msgs2), budget)
    print(f"ngân sách {budget} token → bỏ {dropped} cặp cũ, còn {count(msgs2)} token")

    msgs3, dropped = assemble(history, "x " * 5000, 100, count)
    assert dropped == len(history) and len(msgs3) == 1
    print("tin nhắn mới tự nó đã vượt ngân sách → bỏ hết lịch sử, vẫn gửi tin nhắn mới")
    print("OK")


if __name__ == "__main__":
    main()
