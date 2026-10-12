"""Gửi vài request thử vào server vLLM đang chạy (P01). Chỉ dùng thư viện chuẩn.

    python smoke_test.py SAMPLES_JSONL OUT_JSON

SAMPLES_JSONL: các dòng của data/workloads/w2.jsonl, w3.jsonl (đã có prompt dựng sẵn).
"""

import json
import sys
import time
import urllib.request

BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
QUESTIONS = [
    "What is the difference between a process and a thread? Answer briefly.",
    "A server costs $0.34 per hour. How much does it cost to run it for 3 days and 7 hours?",
    "Write a short email asking the IT team to reset my VPN password.",
]


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out, round(time.time() - t, 2)


def main():
    samples = [json.loads(l) for l in open(sys.argv[1])]
    results = []

    # 1. Chat ngắn: thinking mặc định (bật) vs tắt; tham số sampling để mặc định của model
    for q in QUESTIONS:
        for thinking in (True, False):
            body = {"model": MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": 4096,
                    "chat_template_kwargs": {"enable_thinking": thinking}}
            out, sec = post("/v1/chat/completions", body)
            results.append({"kind": "chat", "thinking": thinking, "question": q, "seconds": sec,
                            "usage": out["usage"], "answer": out["choices"][0]["message"]["content"],
                            "finish_reason": out["choices"][0]["finish_reason"]})
            print(f"chat thinking={thinking!s:5} {sec:6.1f}s usage={out['usage']}")

    # 2. Prompt dựng sẵn từ workload (RAG ~3k, hợp đồng dài nhất) qua /v1/completions, greedy
    for s in samples:
        body = {"model": MODEL, "prompt": s["prompt"], "max_tokens": s["output_tokens"], "temperature": 0}
        out, sec = post("/v1/completions", body)
        results.append({"kind": "completion", "id": s["id"], "prompt_tokens_expected": s["prompt_tokens"],
                        "seconds": sec, "usage": out["usage"], "answer": out["choices"][0]["text"],
                        "finish_reason": out["choices"][0]["finish_reason"]})
        print(f"{s['id']} {sec:6.1f}s usage={out['usage']} (prompt_tokens dự kiến {s['prompt_tokens']})")

    with urllib.request.urlopen(BASE + "/metrics") as r:
        metrics = r.read().decode()
    json.dump({"results": results, "metrics": metrics}, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    print("->", sys.argv[2])


if __name__ == "__main__":
    main()
