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
`--prefix-cache-blocks N` (> 0): prefix cache giả kiểu vLLM — prompt chia khối 64 ký tự, băm nối chuỗi, giữ N khối gần nhất
(LRU); số token trúng = số khối đầu đã có × 16, trả trong usage.prompt_tokens_details.cached_tokens (thử phân tích W4).
"""

import argparse
import asyncio
import hashlib
import json
import time
from collections import OrderedDict

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
        self.cache = OrderedDict()  # hash khối → None, LRU
        self.cache_hits = 0

    def lookup(self, prompt):
        """Số token trúng prefix cache giả, rồi đưa mọi khối của prompt vào cache."""
        if self.a.prefix_cache_blocks <= 0:
            return None
        h, hit, miss = b"", 0, False
        for i in range(0, len(prompt) - 63, 64):
            h = hashlib.blake2b(h + prompt[i:i + 64].encode(), digest_size=8).digest()
            if not miss and h in self.cache:
                hit += 1
                self.cache.move_to_end(h)
            else:
                miss = True
                self.cache[h] = None
                if len(self.cache) > self.a.prefix_cache_blocks:
                    self.cache.popitem(last=False)
        self.cache_hits += hit * 16
        return hit * 16


async def completions(request):
    st: State = request.app["st"]
    a = st.a
    body = await request.json()
    prompt = body.get("prompt", "")
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], (list, str)):
        prompt = prompt[0]  # lm-eval gửi batch 1 phần tử
    if isinstance(prompt, list):  # danh sách token id
        prompt_ids = [int(x) for x in prompt]
        prompt = " ".join(map(str, prompt_ids))
        n_in = len(prompt_ids)
    else:
        prompt_ids = None
        n_in = max(1, len(prompt) // 4)
    n_out = int(body.get("max_tokens") or 16)
    stream = bool(body.get("stream"))
    t_arr = time.perf_counter()
    st.active += 1
    st.prompt_tokens += n_in
    cached = st.lookup(prompt) if isinstance(prompt, str) else None
    usage = {"prompt_tokens": n_in, "completion_tokens": n_out}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
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
            await asyncio.sleep((n_in - (cached or 0)) * a.prefill_s)
        st.ttft_sum += time.perf_counter() - t_arr
        st.ttft_count += 1

        def step():
            return a.token_s * (1 + a.slowdown * st.running) if a.mode == "queue" else a.token_s

        if not stream:
            await asyncio.sleep(step() * n_out)
            st.gen_tokens += n_out
            return web.json_response({"choices": [extras(a, body, prompt, prompt_ids, n_out, st.running)],
                                      "usage": usage})
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
        await emit({"choices": [], "usage": usage})
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp
    finally:
        st.active -= 1
        st.running -= 1
        if st.sem is not None:
            st.sem.release()


def fake_ids(text):
    return [int(hashlib.blake2b(w.encode(), digest_size=4).hexdigest(), 16) % 150000 for w in text.split()] or [0]


def fake_dist(ctx, k):
    """Phân phối giả tất định theo ngữ cảnh: top-k (id, logprob) giảm dần."""
    h = hashlib.blake2b(str(ctx).encode(), digest_size=16).digest()
    base = int.from_bytes(h[:4], "big") % 150000
    gap = 0.2 + (h[4] / 255) * 3.0  # độ chênh top1–top2
    out, lp = [], -0.05
    for i in range(k):
        out.append(((base + 7919 * i) % 150000, round(lp, 6)))
        lp -= gap if i == 0 else 0.5
    return out


def extras(a, body, prompt, prompt_ids, n_out, running):
    """Choice không stream kiểu vLLM: text, token_ids, prompt_token_ids, logprobs (top-k), prompt_logprobs.
    --noise p: mỗi bước, với xác suất p × (số request đang chạy / 8) chọn token hạng 2 (giả lập lệch do batch)."""
    import random as _r
    ids = prompt_ids if prompt_ids is not None else fake_ids(prompt)
    k = int(body.get("logprobs") or 0)
    out_ids, lps, tops = [], [], []
    ctx = hash(tuple(ids)) & 0xFFFFFFFF
    rng = _r.Random()
    for j in range(n_out):
        dist = fake_dist((ctx, j, tuple(out_ids[-2:])), max(k, 2))
        pick = 1 if a.noise and rng.random() < a.noise * max(1, running) / 8 else 0
        tid, lp = dist[pick]
        out_ids.append(tid)
        lps.append(lp)
        tops.append({f"token_id:{t}" if body.get("return_tokens_as_token_ids") else str(t): l for t, l in dist[:k]})
    text = " ".join("x" for _ in out_ids)
    if "Answer:" in prompt or "answer" in prompt[-400:]:
        text += "\nAnswer: 42"
    ch = {"index": 0, "text": text, "finish_reason": "length"}
    if body.get("return_token_ids"):
        ch["token_ids"] = out_ids
        ch["prompt_token_ids"] = ids
    if k:
        toks = [f"token_id:{t}" if body.get("return_tokens_as_token_ids") else str(t) for t in out_ids]
        ch["logprobs"] = {"tokens": toks, "token_logprobs": lps, "top_logprobs": tops, "text_offset": list(range(len(toks)))}
    pk = body.get("prompt_logprobs")
    if pk:
        pls = [None]
        for j in range(1, len(ids)):
            dist = fake_dist((hash(tuple(ids[:j])) & 0xFFFFFFFF, "p"), int(pk))
            if a.noise and rng.random() < a.noise * max(1, running) / 8:
                dist = [(t, round(l - 0.01, 6)) for t, l in dist]
            d = {str(t): {"logprob": l, "rank": r + 1, "decoded_token": "x"} for r, (t, l) in enumerate(dist)}
            if str(ids[j]) not in d:
                d[str(ids[j])] = {"logprob": -9.0, "rank": 500, "decoded_token": "x"}
            pls.append(d)
        ch["prompt_logprobs"] = pls
    return ch


async def health(request):
    return web.Response(text="")


async def models(request):
    return web.json_response({"data": [{"id": request.app["st"].a.model}]})


async def reset(request):
    st = request.app["st"]
    if st.active == 0:
        st.cache.clear()
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
        f"vllm:prefix_cache_hits_total{lab} {st.cache_hits}",
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
    ap.add_argument("--prefix-cache-blocks", type=int, default=0, help="> 0: bật prefix cache giả (số khối 64 ký tự)")
    ap.add_argument("--noise", type=float, default=0.0, help="xác suất chọn token hạng 2 mỗi bước (× số request chạy / 8)")
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
