"""Deterministic, tokenizer-exact prompt generators.

Prompts are lists of token IDs sent to /v1/completions, so input length is exact and
needs no tokenizer (the usual `vllm bench` approach). Random IDs are not natural text:
that is fine for dense-model compute and cache mechanics, but NOT for anything whose
speed depends on text predictability (n-gram speculation); use real text there.

Seeding: every id stream depends on (seed, salt). `salt` must differ per load and per
repeat, otherwise later loads in one server launch replay earlier prompts and hit the
prefix cache, which silently understates TTFT. The shared prefix is salted too, so each
load starts cold and only the first request pays for the prefix.
"""

from __future__ import annotations

import random
from collections.abc import Callable

# Llama-3 vocab is 128256 with special tokens at the top; stay well inside plain text ids.
ID_LO, ID_HI = 1000, 100000


def salt_for(repeat: int, load_idx: int) -> int:
    return repeat * 1009 + load_idx * 101 + 1


def _ids(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(ID_LO, ID_HI) for _ in range(n)]


def make_prompt_fn(
    input_tokens: int,
    output_tokens: int,
    prefix_share: float = 0.0,
    model: str = "model",
    seed: int = 0,
    salt: int = 0,
) -> Callable[[int], dict]:
    """Request i -> /v1/completions body with exactly `input_tokens` prompt tokens.

    The first int(input_tokens*prefix_share) tokens are identical across requests within
    this (seed, salt); the rest are unique per request. Greedy, ignore_eos, so every
    request generates exactly `output_tokens`.
    """
    if not 0.0 <= prefix_share <= 1.0:
        raise ValueError("prefix_share must be in [0, 1]")
    n_shared = int(input_tokens * prefix_share)
    base = seed * 1_000_003 + salt * 7_919
    shared = _ids(random.Random(base), n_shared)

    def make_body(i: int) -> dict:
        unique = _ids(random.Random(base + i + 1_000_000_007), input_tokens - n_shared)
        return {
            "model": model,
            "prompt": shared + unique,
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
        }

    return make_body
