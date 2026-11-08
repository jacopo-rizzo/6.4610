"""What every decoder returns: the generated tokens and timing/acceptance stats."""

from __future__ import annotations

from dataclasses import dataclass, field


def count_bytes(vocab, tokens: list[int], stop_ids: set[int]) -> int:
    """UTF-8 bytes of generated text, not counting stop tokens like <|eot_id|>."""
    return sum(len(vocab.token_bytes(t)) for t in tokens if t not in stop_ids)


@dataclass
class GenerationResult:
    prompt_length: int
    tokens: list[int]                    # generated target token ids (prompt excluded)
    num_bytes: int                       # UTF-8 bytes of the generated text
    wall_time: float                     # seconds, whole generate() call

    # Speculative decoding only (zero for the autoregressive baseline).
    rounds: int = 0                      # number of target verification passes
    draft_steps: int = 0                 # number of draft forward passes
    proposed_tokens: int = 0             # target tokens proposed by the draft
    accepted_tokens: int = 0             # of those, accepted by the verifier
    proposed_bytes: int = 0
    accepted_bytes: int = 0
    draft_time: float = 0.0
    target_time: float = 0.0
    overhead_time: float = 0.0           # re-tokenization, verification bookkeeping
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def bytes_per_second(self) -> float:
        return self.num_bytes / self.wall_time if self.wall_time else 0.0

    @property
    def token_acceptance_rate(self) -> float:
        return self.accepted_tokens / self.proposed_tokens if self.proposed_tokens else 0.0

    @property
    def byte_acceptance_rate(self) -> float:
        return self.accepted_bytes / self.proposed_bytes if self.proposed_bytes else 0.0

    @property
    def mean_accepted_bytes_per_round(self) -> float:
        return self.accepted_bytes / self.rounds if self.rounds else 0.0

    def summary(self) -> dict:
        keys = ["num_bytes", "wall_time", "rounds", "draft_steps", "proposed_tokens",
                "accepted_tokens", "proposed_bytes", "accepted_bytes", "draft_time",
                "target_time", "overhead_time"]
        out = {k: getattr(self, k) for k in keys}
        out.update(bytes_per_second=self.bytes_per_second,
                   token_acceptance_rate=self.token_acceptance_rate,
                   byte_acceptance_rate=self.byte_acceptance_rate,
                   mean_accepted_bytes_per_round=self.mean_accepted_bytes_per_round,
                   num_tokens=len(self.tokens))
        return out
