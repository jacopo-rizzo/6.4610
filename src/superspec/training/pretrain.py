"""Pretraining data: fixed-length windows of a pretokenized token file.

The train file is cut into consecutive, non-overlapping windows of `seq_len`
tokens, visited in a fixed shuffled order. Windows can span document
boundaries (marked by the end-of-text token), as is usual for pretraining.

Fairness between tokenizers: `seq_len` is derived from a context length in
bytes, so a SuperBPE window holds about as much text as a standard one. With
the same number of sequences per step, both drafts then see about the same
number of bytes per step and take about the same number of steps.
"""

from __future__ import annotations

import numpy as np
import torch

from superspec.data.token_files import TokenFile


def seq_len_for(context_bytes: int, bytes_per_token: float, max_positions: int) -> int:
    return min(max_positions, round(context_bytes / bytes_per_token))


class PretrainWindows:
    def __init__(self, token_file: TokenFile, seq_len: int, seed: int = 0):
        self.tokens = token_file.tokens
        self.seq_len = seq_len
        self.num_windows = len(self.tokens) // seq_len
        self.order = np.random.default_rng(seed).permutation(self.num_windows)

    def batch(self, indices) -> dict:
        """Indices count windows globally; past the end they wrap into another epoch."""
        rows = []
        for i in indices:
            w = self.order[i % self.num_windows]
            rows.append(self.tokens[w * self.seq_len:(w + 1) * self.seq_len])
        ids = torch.from_numpy(np.stack(rows).astype(np.int64))
        return {"input_ids": ids, "labels": ids}     # the model shifts labels itself
