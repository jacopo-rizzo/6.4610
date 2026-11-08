"""Turning logits into the distribution we sample from.

Temperature and top-p define the *target* distribution that speculative
decoding must reproduce exactly. The draft may use any distribution, as long
as the verifier is told the one it actually sampled from; we warp the draft's
logits the same way since that tends to maximize acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.0   # 0 means greedy decoding
    top_p: float = 1.0

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def to_probs(logits: torch.Tensor, sampling: SamplingConfig) -> torch.Tensor:
    """Probabilities over the last dimension after temperature and top-p."""
    if sampling.greedy:
        raise ValueError("Greedy decoding has no sampling distribution; use argmax")
    logits = logits.float() / sampling.temperature
    if sampling.top_p < 1.0:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Drop a token if the tokens ranked above it already cover top_p.
        # The top token is always kept.
        drop_sorted = cumulative - sorted_logits.softmax(dim=-1) >= sampling.top_p
        drop = drop_sorted.gather(-1, order.argsort(dim=-1))   # back to vocab order
        logits = logits.masked_fill(drop, float("-inf"))
    return logits.softmax(dim=-1)


def sample(probs: torch.Tensor, generator: torch.Generator | None) -> int:
    return int(torch.multinomial(probs, 1, generator=generator))


def uniform(device: torch.device, generator: torch.Generator | None) -> float:
    return float(torch.rand((), device=device, generator=generator))


def normalize(weights: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """weights / weights.sum(), or `fallback` if the weights are all ~zero
    (happens only when p == q up to rounding, where any choice is fine)."""
    total = weights.sum()
    return weights / total if total > 1e-12 else fallback
