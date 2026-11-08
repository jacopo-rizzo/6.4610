"""Baseline: plain autoregressive decoding with the target model."""

from __future__ import annotations

import time

import torch

from superspec.decoding.kv_cache import CachedLM
from superspec.decoding.result import GenerationResult, count_bytes
from superspec.decoding.sampling import SamplingConfig, sample, to_probs


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autoregressive_generate(
    target: CachedLM,
    target_vocab,
    prompt: list[int],
    max_new_tokens: int,
    sampling: SamplingConfig,
    stop_ids: set[int],
    generator: torch.Generator | None = None,
) -> GenerationResult:
    """One target forward pass per generated token."""
    target.reset()
    synchronize(target.device)
    start = time.perf_counter()

    tokens = list(prompt)
    for _ in range(max_new_tokens):
        logits = target.logits(tokens, 1)[0]
        token = int(logits.argmax()) if sampling.greedy else sample(to_probs(logits, sampling),
                                                                    generator)
        tokens.append(token)
        if token in stop_ids:
            break

    synchronize(target.device)
    new = tokens[len(prompt):]
    return GenerationResult(
        prompt_length=len(prompt),
        tokens=new,
        num_bytes=count_bytes(target_vocab, new, stop_ids),
        wall_time=time.perf_counter() - start,
    )
