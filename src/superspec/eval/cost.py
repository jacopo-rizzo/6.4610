"""Measuring the draft-to-target cost ratio c of the speedup model.

    S(a, g, c) = (1 - a^(g+1)) / ((1 - a)(g c + 1))

c is the latency of one draft forward step divided by one target forward step,
both at batch size 1 with a warm KV cache. At these sizes it is set mostly by
layer count, kernel launches and the output head, not by parameter count, so
we measure it instead of using the parameter ratio. We also time the target's
verification pass over K + 1 tokens, which is what a speculative round pays.

Each timed call recomputes exactly `num_new` tokens on top of a cached
context (CachedLM always recomputes every position it returns logits for), and
includes the cache crop, as in real decoding.
"""

from __future__ import annotations

import statistics
import time

from superspec.decoding import CachedLM
from superspec.decoding.autoregressive import synchronize


def context_from_texts(texts: list[str], target_vocab, num_tokens: int) -> list[int]:
    """At least `num_tokens` target tokens of real text (texts repeated if short)."""
    ids: list[int] = []
    while len(ids) < num_tokens:
        for text in texts:
            ids.extend(target_vocab.encode(text + "\n\n"))
            if len(ids) >= num_tokens:
                break
    return ids[:num_tokens]


def time_forward_ms(lm: CachedLM, context: list[int], num_new: int, reps: int = 50,
                    warmup: int = 5) -> float:
    """Median milliseconds to compute `num_new` tokens after a cached `context`."""
    if num_new > len(context):
        raise ValueError("context must be at least num_new tokens long")
    lm.reset()
    lm.logits(context, 1)                    # fill the cache
    times = []
    for r in range(warmup + reps):
        # Re-feed the last num_new context tokens: they are recomputed every call.
        synchronize(lm.device)
        start = time.perf_counter()
        lm.logits(context, num_new)
        synchronize(lm.device)
        if r >= warmup:
            times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def measure_cost_ratios(target_lm: CachedLM, target_vocab, drafts: dict, context: list[int],
                        verify_tokens: int = 8, reps: int = 50, warmup: int = 5) -> list[dict]:
    """One row per draft: step times and c = draft step / target step."""
    target_step = time_forward_ms(target_lm, context, 1, reps, warmup)
    target_verify = time_forward_ms(target_lm, context, verify_tokens, reps, warmup)
    rows = []
    for name, draft in drafts.items():
        draft_context = draft.vocab.from_target_ids(context)
        draft_step = time_forward_ms(draft.lm, draft_context, 1, reps, warmup)
        rows.append({
            "draft": name,
            "tokenizer": draft.vocab.name,
            "draft_vocab_size": draft.vocab.vocab_size,
            "context_target_tokens": len(context),
            "context_draft_tokens": len(draft_context),
            "draft_step_ms": round(draft_step, 3),
            "target_step_ms": round(target_step, 3),
            f"target_verify_{verify_tokens}_ms": round(target_verify, 3),
            "c": round(draft_step / target_step, 4),
            "c_vs_verify_pass": round(draft_step / target_verify, 4),
        })
    return rows
