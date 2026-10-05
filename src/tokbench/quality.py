"""Quality gates run BEFORE any speedup is trusted.

- greedy_equivalence: two servers (e.g. prefix cache on vs off) must produce identical
  greedy text for the same prompts. Prefix caching must not change outputs.
- gsm8k_accuracy: exact-match accuracy on a fixed GSM8K subset, to put a quality delta
  next to any quantization speedup. Item file is JSONL with {"question", "answer"}
  (the openai/grade-school-math format; the final answer follows '####').
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import aiohttp

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


async def _complete(session: aiohttp.ClientSession, url: str, body: dict) -> str:
    async with session.post(url, json={**body, "stream": False}) as r:
        r.raise_for_status()
        return (await r.json())["choices"][0]["message"]["content"]


@dataclass
class EquivalenceResult:
    n: int
    n_identical: int
    first_diff: int | None

    @property
    def passed(self) -> bool:
        return self.n_identical == self.n


async def greedy_equivalence(url_a: str, url_b: str, bodies: list[dict]) -> EquivalenceResult:
    """Send each body (must be greedy) to both servers and compare completions."""
    if any(b.get("temperature", 1) != 0 for b in bodies):
        raise ValueError("bodies must set temperature=0")
    same, first = 0, None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        for i, b in enumerate(bodies):
            a_txt, b_txt = await _complete(s, url_a, b), await _complete(s, url_b, b)
            if a_txt == b_txt:
                same += 1
            elif first is None:
                first = i
    return EquivalenceResult(len(bodies), same, first)


def last_number(text: str) -> str | None:
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def load_gsm8k(path: str | Path, n: int, seed: int = 0) -> list[dict]:
    """First n items after a seeded shuffle, so the subset is fixed and reproducible."""
    import random

    items = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(items)
    return [
        {
            "question": it["question"],
            "answer": it["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for it in items[:n]
    ]


async def gsm8k_accuracy(url: str, model: str, items: list[dict], max_tokens: int = 384) -> dict:
    correct = 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        for it in items:
            body = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": it["question"] + "\nEnd with: The answer is <number>.",
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            if last_number(await _complete(s, url, body)) == it["answer"]:
                correct += 1
    n = len(items)
    p = correct / n if n else float("nan")
    se = (p * (1 - p) / n) ** 0.5 if n else float("nan")
    return {"n": n, "correct": correct, "accuracy": p, "stderr": se}
