"""Fake OpenAI-compatible streaming server with known, configurable latency.

Used to validate the load generator and run the synthetic demo without a GPU.
Model: prefill takes `prefill_s_per_token * prompt_tokens`, then each output token
takes `tpot_s`. At most `slots` requests are served concurrently; the rest queue,
which creates a latency knee whose position is known analytically. Serves
/v1/completions (token-id or text prompts) and /v1/chat/completions, plus /health and a
vLLM-shaped /metrics so the scraper path is exercised.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import zlib

from aiohttp import web


def make_app(
    prefill_s_per_token: float = 0.0001,
    tpot_s: float = 0.01,
    slots: int = 8,
    salt: str = "",
    fail_after: int | None = None,
) -> web.Application:
    """`fail_after`: emit that many tokens then drop the stream (fault injection)."""
    sem = asyncio.Semaphore(slots)
    state = {"running": 0, "waiting": 0}

    def prompt_info(body: dict) -> tuple[int, str]:
        if "messages" in body:
            text = " ".join(m.get("content", "") for m in body["messages"])
            return max(1, len(text.split())), text
        p = body.get("prompt", "")
        if isinstance(p, list):
            return max(1, len(p)), " ".join(map(str, p[-4:]))
        return max(1, len(str(p).split())), str(p)

    async def generate(request: web.Request) -> web.StreamResponse:
        chat = request.path.endswith("chat/completions")
        body = await request.json()
        prompt_tokens, text = prompt_info(body)
        n_out = int(body.get("max_tokens", 16))
        if not body.get("stream"):  # deterministic function of the prompt (+ salt)
            h = zlib.crc32(f"{text}|{salt}".encode()) % 1000 if salt else 0
            content = " ".join(f"{text[-6:]}{i}{h}" for i in range(n_out))
            choice = {"message": {"content": content}} if chat else {"text": content}
            return web.json_response({"choices": [choice]})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        state["waiting"] += 1
        async with sem:
            state["waiting"] -= 1
            state["running"] += 1
            try:
                await asyncio.sleep(prefill_s_per_token * prompt_tokens)
                for i in range(n_out):
                    if fail_after is not None and i >= fail_after:
                        return resp  # drop mid-stream: no usage, no [DONE]
                    await asyncio.sleep(tpot_s)
                    piece = f"t{i} "
                    choice = {"delta": {"content": piece}} if chat else {"text": piece}
                    await resp.write(f"data: {json.dumps({'choices': [choice]})}\n\n".encode())
            finally:
                state["running"] -= 1
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_out},
        }
        await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def health(_: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def metrics(_: web.Request) -> web.Response:
        return web.Response(
            text=(
                f"vllm:num_requests_running {state['running']}\n"
                f"vllm:num_requests_waiting {state['waiting']}\n"
            )
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", generate)
    app.router.add_post("/v1/completions", generate)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tpot", type=float, default=0.01)
    ap.add_argument("--prefill", type=float, default=0.0001)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--fail-after", type=int, default=None, help="drop streams after N tokens")
    a = ap.parse_args()
    app = make_app(a.prefill, a.tpot, a.slots, fail_after=a.fail_after)
    web.run_app(app, port=a.port, print=None)


if __name__ == "__main__":
    main()
