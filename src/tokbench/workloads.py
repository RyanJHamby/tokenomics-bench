"""Deterministic prompt generators. Token counts are approximate (1 word ~ 1 token for
the synthetic text), which is fine for relative comparisons; actual prompt/completion
token counts are taken from the server's usage field.
"""

from __future__ import annotations

import random

_VOCAB = [f"w{i}" for i in range(2000)]


def _text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n_words))


def make_prompt_fn(
    input_tokens: int,
    output_tokens: int,
    prefix_share: float = 0.0,
    model: str = "model",
    seed: int = 0,
):
    """Request i -> chat body. The first `prefix_share` of each prompt is identical
    across requests (a shared system prompt); the remainder is unique per request.
    Greedy decoding (temperature 0) so outputs are comparable across configs.
    """
    if not 0.0 <= prefix_share <= 1.0:
        raise ValueError("prefix_share must be in [0, 1]")
    n_shared = int(input_tokens * prefix_share)
    shared = _text(random.Random(seed), n_shared)

    def make_body(i: int) -> dict:
        unique = _text(random.Random(seed * 1_000_003 + i + 1), input_tokens - n_shared)
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": shared},
                {"role": "user", "content": unique},
            ],
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
        }

    return make_body
