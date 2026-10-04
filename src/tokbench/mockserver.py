"""Fake OpenAI-compatible streaming server with known, configurable latency.

Used to validate the load generator without a GPU. Model: prefill takes
`prefill_s_per_token * prompt_tokens`, then each output token takes `tpot_s`.
At most `slots` requests are served concurrently; the rest queue, which creates a
latency knee whose position is known analytically.
"""

from __future__ import annotations

import asyncio
import json

from aiohttp import web


def make_app(
    prefill_s_per_token: float = 0.0001, tpot_s: float = 0.01, slots: int = 8
) -> web.Application:
    sem = asyncio.Semaphore(slots)

    async def chat(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        text = " ".join(m.get("content", "") for m in body.get("messages", []))
        prompt_tokens = max(1, len(text.split()))
        n_out = int(body.get("max_tokens", 16))
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        async with sem:
            await asyncio.sleep(prefill_s_per_token * prompt_tokens)
            for i in range(n_out):
                await asyncio.sleep(tpot_s)
                evt = {"choices": [{"delta": {"content": f"t{i} "}}]}
                await resp.write(f"data: {json.dumps(evt)}\n\n".encode())
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_out},
        }
        await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), port=8000)
