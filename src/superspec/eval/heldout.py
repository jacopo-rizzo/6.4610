"""Held-out loss of a draft model, comparable across tokenizers.

Per-token loss is not comparable between the standard and SuperBPE drafts:
a SuperBPE token covers more text. So we report bits per byte:

    bits_per_byte = (total negative log-likelihood in nats) / (ln 2 * total bytes)

over the same held-out text (the pretokenized `val` split, which both
tokenizers build from the same documents). Both drafts also get the same
amount of context in bytes: the window length in tokens is `context_bytes`
divided by the tokenizer's bytes per token.

Windows are consecutive and non-overlapping, so every held-out token after
the first is predicted exactly once.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from superspec.data.token_files import TokenFile


def token_byte_lengths(vocab) -> np.ndarray:
    """Bytes of each token id. The end-of-document token counts as 0 bytes: it is
    predicted (and costs loss) equally often for both tokenizers."""
    lengths = np.array([len(vocab.token_bytes(i)) for i in range(vocab.vocab_size)], dtype=np.int64)
    lengths[vocab.eos_id] = 0
    return lengths


@torch.no_grad()
def heldout_loss(model, vocab, token_file: TokenFile, context_bytes: int = 8192,
                 batch_size: int = 2, max_windows: int | None = None) -> dict:
    device = next(model.parameters()).device
    max_positions = getattr(model.config, "max_position_embeddings", None) or 10**9
    seq_len = min(max(16, round(context_bytes / token_file.bytes_per_token)), max_positions)

    tokens = token_file.tokens
    num_windows = (len(tokens) - 1) // seq_len
    if max_windows:
        num_windows = min(num_windows, max_windows)
    byte_lengths = token_byte_lengths(vocab)

    total_nll = 0.0
    total_tokens = total_bytes = 0
    model.eval()
    for first in range(0, num_windows, batch_size):
        windows = range(first, min(first + batch_size, num_windows))
        # Window w covers tokens [w*L, w*L + L]: inputs are the first L, targets the last L.
        batch = np.stack([tokens[w * seq_len: w * seq_len + seq_len + 1] for w in windows])
        batch = torch.from_numpy(batch.astype(np.int64)).to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]

        logits = model(input_ids=inputs).logits.float()
        total_nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                     reduction="sum").item()
        total_tokens += targets.numel()
        total_bytes += int(byte_lengths[targets.cpu().numpy()].sum())

    return {
        "tokenizer": vocab.name,
        "seq_len_tokens": seq_len,
        "context_bytes": context_bytes,
        "windows": num_windows,
        "tokens": total_tokens,
        "bytes": total_bytes,
        "nll_per_token": total_nll / total_tokens,
        "bits_per_byte": total_nll / (math.log(2) * total_bytes),
    }
