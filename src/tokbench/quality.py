"""Quality gates run BEFORE any speedup is trusted.

- greedy_equivalence: two servers (e.g. prefix cache on vs off) must produce identical
  greedy text for the same prompts. Prefix caching must not change outputs.
- gsm8k_accuracy: exact-match accuracy on a fixed GSM8K subset, to put a quality delta
  next to any quantization speedup. Item file is JSONL with {"question", "answer"}
  (the openai/grade-school-math format; the final answer follows '####').
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import aiohttp

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


async def _complete(session: aiohttp.ClientSession, url: str, body: dict) -> str:
    async with session.post(url, json={**body, "stream": False}) as r:
        r.raise_for_status()
        choice = (await r.json())["choices"][0]
        return choice["message"]["content"] if "message" in choice else choice["text"]


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


async def gsm8k_accuracy(
    url: str, model: str, items: list[dict], max_tokens: int = 1024, concurrency: int = 32
) -> dict:
    """Score every item; returns per-item correctness (needed for a paired comparison).
    max_tokens is generous: truncating a chain of thought would score as wrong and
    masquerade as a quality loss."""
    sem = asyncio.Semaphore(concurrency)

    async def one(session: aiohttp.ClientSession, it: dict) -> bool:
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": it["question"] + "\nEnd with: The answer is <number>."}
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        async with sem:
            return last_number(await _complete(session, url, body)) == it["answer"]

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        correct = await asyncio.gather(*(one(s, it) for it in items))
    n = len(items)
    p = sum(correct) / n if n else float("nan")
    se = (p * (1 - p) / n) ** 0.5 if n else float("nan")
    return {
        "n": n,
        "n_correct": sum(correct),
        "accuracy": p,
        "stderr": se,
        "correct": [bool(c) for c in correct],
    }


def paired_accuracy(base: list[bool], cand: list[bool], margin: float = 0.02) -> dict:
    """Candidate vs baseline accuracy on the SAME items (paired), difference d = cand - base.

    Pairing removes item difficulty from the variance, so a few hundred discordant items
    can resolve a ~2pp gap that unpaired standard errors (~1pp each at n=1319) cannot.
    Verdicts against +/- margin: 'non_inferior' (CI above -margin), 'equivalent' (CI inside
    +/- margin), 'inferior' (CI below -margin), else 'inconclusive'. Exact McNemar p-value.
    """
    if len(base) != len(cand) or not base:
        raise ValueError("need equal-length, non-empty paired outcomes")
    n = len(base)
    n_base_only = sum(b and not c for b, c in zip(base, cand, strict=True))
    n_cand_only = sum(c and not b for b, c in zip(base, cand, strict=True))
    d = (n_cand_only - n_base_only) / n
    disc = n_base_only + n_cand_only
    var = (disc - (n_cand_only - n_base_only) ** 2 / n) / n**2
    se = var**0.5
    lo, hi = d - 1.959964 * se, d + 1.959964 * se
    k = min(n_base_only, n_cand_only)
    p = 1.0 if disc == 0 else min(1.0, 2 * sum(math.comb(disc, i) for i in range(k + 1)) / 2**disc)
    if hi < -margin:
        verdict = "inferior"
    elif lo > -margin and hi < margin:
        verdict = "equivalent"
    elif lo > -margin:
        verdict = "non_inferior"
    else:
        verdict = "inconclusive"
    return {
        "n": n,
        "diff": d,
        "ci": (lo, hi),
        "mcnemar_p": p,
        "verdict": verdict,
        "discordant": disc,
        "margin": margin,
    }
