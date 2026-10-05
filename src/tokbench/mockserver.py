"""Fake OpenAI-compatible streaming server with known, configurable latency.

Used to validate the load generator and run the synthetic demo without a GPU.
Model: prefill takes `prefill_s_per_token * prompt_tokens`, then each output token
takes `tpot_s`. At most `slots` requests are served concurrently; the rest queue,
which creates a latency knee whose position is known analytically. Also serves
/health and a vLLM-shaped /metrics so the scraper path is exercised.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from aiohttp import web


def make_app(
    prefill_s_per_token: float = 0.0001, tpot_s: float = 0.01, slots: int = 8, salt: str = ""
) -> web.Application:
    sem = asyncio.Semaphore(slots)
    state = {"running": 0, "waiting": 0, "slots": slots}

    async def chat(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        text = " ".join(m.get("content", "") for m in body.get("messages", []))
        prompt_tokens = max(1, len(text.split()))
        n_out = int(body.get("max_tokens", 16))
        if not body.get("stream"):  # deterministic function of the prompt (+ salt)
            h = abs(hash((text, salt))) % 1000 if salt else 0
            content = " ".join(f"{text.split()[-1] if text else ''}{i}{h}" for i in range(n_out))
            return web.json_response({"choices": [{"message": {"content": content}}]})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        state["waiting"] += 1
        async with sem:
            state["waiting"] -= 1
            state["running"] += 1
            try:
                await asyncio.sleep(prefill_s_per_token * prompt_tokens)
                for i in range(n_out):
                    await asyncio.sleep(tpot_s)
                    evt = {"choices": [{"delta": {"content": f"t{i} "}}]}
                    await resp.write(f"data: {json.dumps(evt)}\n\n".encode())
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
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tpot", type=float, default=0.01)
    ap.add_argument("--prefill", type=float, default=0.0001)
    ap.add_argument("--slots", type=int, default=8)
    a = ap.parse_args()
    web.run_app(make_app(a.prefill, a.tpot, a.slots), port=a.port, print=None)


if __name__ == "__main__":
    main()
