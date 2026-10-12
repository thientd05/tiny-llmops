"""Ghép prompt cho một lượt của phiên W4 lúc chạy (client benchmark dùng chung hàm này).

Phiên W4 không dựng sẵn prompt: lượt k gồm lịch sử thật (câu hỏi cũ + câu trả lời **server vừa sinh**) + tin nhắn mới,
giống ứng dụng chat thật. Nếu lịch sử vượt `prompt_budget` (độ dài trong trace), bỏ các cặp hỏi–đáp cũ nhất,
giống client trong BurstGPT cắt lịch sử (trace: độ dài prompt ngừng tăng từ lượt ~5).
"""


def assemble(history, user_msg, budget, count_tokens):
    """history: list[(user_text, assistant_text)] theo thứ tự cũ → mới.
    count_tokens(messages) -> số token prompt sau chat template.
    Trả về (messages, số cặp lịch sử đã bỏ)."""
    for drop in range(len(history) + 1):
        messages = []
        for u, a in history[drop:]:
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": user_msg})
        if count_tokens(messages) <= budget or drop == len(history):
            return messages, drop
