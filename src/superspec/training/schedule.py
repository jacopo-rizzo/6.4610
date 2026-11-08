"""Learning-rate schedule and batch indexing (plain Python, no torch).

Batches are addressed by index rather than drawn from a stateful iterator:
micro-batch i on process r always contains the same examples. That makes
resuming from a checkpoint exact (skip to the right index) and makes the
standard and SuperBPE runs see the same data order.
"""

from __future__ import annotations

import math


def learning_rate(step: int, total_steps: int, base_lr: float, warmup_frac: float = 0.01,
                  min_lr_frac: float = 0.1, schedule: str = "cosine") -> float:
    """Linear warmup, then cosine (or constant) decay to min_lr_frac * base_lr."""
    warmup = max(1, int(warmup_frac * total_steps))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unknown schedule {schedule!r}")
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def accumulation_steps(examples_per_step: int, micro_batch_size: int, world_size: int) -> int:
    """Gradient accumulation steps so each optimizer step sees examples_per_step examples."""
    per_pass = micro_batch_size * world_size
    if examples_per_step % per_pass:
        raise ValueError(f"examples_per_step={examples_per_step} is not a multiple of "
                         f"micro_batch_size * num_gpus = {per_pass}")
    return examples_per_step // per_pass


def micro_batch_indices(micro_index: int, rank: int, world_size: int,
                        micro_batch_size: int) -> range:
    """Global example indices of micro-batch `micro_index` on process `rank`.

    Consecutive micro-batches across all processes tile 0, 1, 2, ... without
    overlap, so every example is used once per pass over the data.
    """
    start = (micro_index * world_size + rank) * micro_batch_size
    return range(start, start + micro_batch_size)
