"""Server giả kiểu OpenAI (không cần GPU) để thử harness ở local: sweep.py, session_client.py, chạy tiếp, logic tìm goodput.

    python bench/fake_server.py --port 8123 --mode threshold --capacity 40

Hai kiểu "sức chứa":
- threshold: request đến khi đã có ≥ K request trong hệ thống (tính cả nó > K) thì TTFT bị cộng `--slow-ttft` giây,
  ngược lại TTFT ≈ 0. Closed-loop N user ở trạng thái ổn định luôn có đúng N request → đạt SLO ⇔ N ≤ K.
  Điểm gãy biết chắc = K, dùng để kiểm logic tìm goodput.
- queue: tối đa K request chạy cùng lúc, dư thì xếp hàng FIFO; TTFT = thời gian chờ + prompt × prefill;
  mỗi bước sinh token chậm dần theo số request đang chạy. Giống server thật hơn, dùng để thử client.

Endpoint: /health, /v1/models, /v1/completions (stream / không stream), /reset_prefix_cache, /metrics.
Số token prompt ước bằng số ký tự / 4 (không tokenize). Token ra gửi theo cụm `--chunk-tokens` để Python kịp.
"""

import argparse
import asyncio
import json
import time

from aiohttp import web


class State:
    def __init__(self, a):
        self.a = a
        self.active = 0      # request trong hệ thống (đang chờ + đang chạy)
        self.running = 0
        self.waiting = 0
        self.sem = asyncio.Semaphore(a.capacity) if a.mode == "queue" else None
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.prompt_tokens = 0
        self.gen_tokens = 0


async def completions(request):
    st: State = request.app["st"]
    a = st.a
    body = await request.json()
    prompt = body.get("prompt", "")
    n_in = max(1, len(prompt) // 4)
    n_out = int(body.get("max_tokens") or 16)
    stream = bool(body.get("stream"))
    t_arr = time.perf_counter()
    st.active += 1
    st.prompt_tokens += n_in
    try:
        if a.mode == "threshold":
            delay = a.slow_ttft if st.active > a.capacity else 0.0
            st.running += 1
            await asyncio.sleep(delay + n_in * a.prefill_s)
        else:
            st.waiting += 1
            await st.sem.acquire()
            st.waiting -= 1
            st.running += 1
            await asyncio.sleep(n_in * a.prefill_s)
        st.ttft_sum += time.perf_counter() - t_arr
        st.ttft_count += 1

        def step():
            return a.token_s * (1 + a.slowdown * st.running) if a.mode == "queue" else a.token_s

        if not stream:
            await asyncio.sleep(step() * n_out)
            st.gen_tokens += n_out
            return web.json_response({"choices": [{"text": "x " * n_out}],
                                      "usage": {"prompt_tokens": n_in, "completion_tokens": n_out}})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        async def emit(obj):
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        await emit({"choices": [{"text": "x"}]})  # token đầu ngay sau prefill
        sent = 1
        while sent < n_out:
            k = min(a.chunk_tokens, n_out - sent)
            await asyncio.sleep(step() * k)
            await emit({"choices": [{"text": " x" * k}]})
            sent += k
        st.gen_tokens += n_out
        await emit({"choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": n_out}})
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp
    finally:
        st.active -= 1
        st.running -= 1
        if st.sem is not None:
            st.sem.release()


async def health(request):
    return web.Response(text="")


async def models(request):
    return web.json_response({"data": [{"id": request.app["st"].a.model}]})


async def reset(request):
    st = request.app["st"]
    return web.json_response({"success": st.active == 0})


async def metrics(request):
    st = request.app["st"]
    lab = '{engine="0",model_name="fake"}'
    lines = [
        f"vllm:num_requests_running{lab} {st.running}",
        f"vllm:num_requests_waiting{lab} {st.waiting}",
        f"vllm:kv_cache_usage_perc{lab} {min(1.0, st.running / st.a.capacity)}",
        f"vllm:num_preemptions_total{lab} 0",
        f"vllm:prefix_cache_queries_total{lab} {st.prompt_tokens}",
        f"vllm:prefix_cache_hits_total{lab} 0",
        f"vllm:prompt_tokens_total{lab} {st.prompt_tokens}",
        f"vllm:generation_tokens_total{lab} {st.gen_tokens}",
        f"vllm:time_to_first_token_seconds_sum{lab} {st.ttft_sum}",
        f"vllm:time_to_first_token_seconds_count{lab} {st.ttft_count}",
    ]
    return web.Response(text="\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--mode", choices=["threshold", "queue"], default="threshold")
    ap.add_argument("--capacity", type=int, default=40)
    ap.add_argument("--slow-ttft", type=float, default=0.5, help="threshold: giây cộng vào TTFT khi quá sức chứa")
    ap.add_argument("--prefill-s", type=float, default=0.0, help="giây / token prompt")
    ap.add_argument("--token-s", type=float, default=0.0005, help="giây / token ra")
    ap.add_argument("--slowdown", type=float, default=0.02, help="queue: mỗi request đang chạy làm bước chậm thêm")
    ap.add_argument("--chunk-tokens", type=int, default=25)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-GPTQ-Int4")
    a = ap.parse_args()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["st"] = None

    async def on_start(app):
        app["st"] = State(a)

    app.on_startup.append(on_start)
    app.add_routes([web.post("/v1/completions", completions), web.get("/health", health),
                    web.get("/v1/models", models), web.post("/reset_prefix_cache", reset),
                    web.get("/metrics", metrics)])
    # dòng log giống vLLM để thử phần đọc dung lượng KV cache của sweep.py
    print(f"INFO GPU KV cache size: {a.capacity * 1000:,} tokens", flush=True)
    print(f"INFO Available KV cache memory: {a.capacity / 10:.2f} GiB", flush=True)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
