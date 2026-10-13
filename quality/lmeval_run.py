"""Chạy lm-evaluation-harness (ghim 0.4.13) vào server vLLM với chat template GIỐNG lúc đo tốc độ (enable_thinking=False).

    python quality/lmeval_run.py run --model local-completions-nothink \
        --model_args model=Qwen/Qwen3-30B-A3B-GPTQ-Int4,base_url=http://localhost:8000/v1/completions,tokenizer=<dir>,... \
        --tasks gsm8k --apply_chat_template --use_cache <db> --output_path <dir> --log_samples

Vì sao cần file này: model kiểu API của lm-eval (`local-completions`) render chat template phía client nhưng không truyền
`enable_thinking` (lm_eval/models/api_models.py, `apply_chat_template`). Không có `enable_thinking=False`, template Qwen3 không
chèn khối <think></think> rỗng → model tự suy luận dài, khác hẳn chế độ dùng khi benchmark (docs/references E11).
Lớp con dưới đây chỉ thay đúng hàm đó; mọi thứ khác (cách gửi request, cache, chấm điểm) là của lm-eval.
"""

import sys

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI


@register_model("local-completions-nothink")
class LocalCompletionsNoThink(LocalCompletionsAPI):
    def __init__(self, tokenizer_backend="huggingface", **kwargs):
        # ép tokenizer phía client (không dùng /tokenize của server) để render template với enable_thinking=False
        super().__init__(tokenizer_backend=tokenizer_backend, **kwargs)

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt, enable_thinking=False)


def main():
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()


if __name__ == "__main__":
    sys.exit(main())
