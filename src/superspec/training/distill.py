"""Distillation data: target responses to chat prompts, for supervised fine-tuning.

Each example is a chat prompt (formatted with the target's chat template) and
the target's own response, both stored as target token ids. A draft re-encodes
them into its own vocabulary (`from_target_ids`), so the standard and SuperBPE
drafts train on exactly the same text. The loss is on response tokens only.

The prompt ends with the template's "\\n\\n" after the assistant header, which
is a barrier token, so superwords never straddle the prompt/response boundary.

Round-start cuts (SuperBPE drafts only). In speculative decoding every round
starts from a context that ends wherever the target's last token landed, often
in the middle of what the full text would merge into one superword: the
context ends with " show" although the text goes on " the" and fully merged
text only ever contains " show the". A draft trained on fully merged text has
never seen " show" followed by " the", so it starts such rounds badly (about
19% of round starts with the OLMo-2 tokenizer). With `cut_rate` > 0 the
training responses are therefore tokenized in randomly cut segments, each
merged on its own, which is exactly what the draft sees and must produce at a
round start: merged(text before the cut) followed by merged(text after it).
Cuts are re-drawn every epoch. Validation examples stay fully merged.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IGNORE = -100   # label value the HF loss ignores


@dataclass
class Example:
    prompt: list[int]       # draft token ids
    response: list[int]     # draft token ids (fully merged), cut to fit max_len
    response_target: list[int] | None = None   # the whole response in target token ids


def segmented_ids(vocab, target_ids: list[int], cut_rate: float, rng: random.Random) -> list[int]:
    """Draft token ids of `target_ids`, tokenized in segments: every boundary
    between two target tokens is a cut with probability `cut_rate`, and no
    superword spans a cut. cut_rate = 0 gives the usual fully merged ids."""
    if cut_rate <= 0:
        return vocab.from_target_ids(target_ids)
    out: list[int] = []
    start = 0
    for i in range(1, len(target_ids)):
        if rng.random() < cut_rate:
            out.extend(vocab.from_target_ids(target_ids[start:i]))
            start = i
    out.extend(vocab.from_target_ids(target_ids[start:]))
    return out


def read_generations(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def split_train_val(rows: list[dict], val_every: int = 100) -> tuple[list[dict], list[dict]]:
    """Deterministic split by the prompt's index: every val_every-th prompt is held out."""
    train = [r for r in rows if r["index"] % val_every != 0]
    val = [r for r in rows if r["index"] % val_every == 0]
    return train, val


def build_examples(rows: list[dict], vocab, max_len: int, min_response: int = 8) -> list[Example]:
    """Re-encode into the draft vocab and truncate to max_len tokens (cutting the
    response). Prompts too long to leave room for a response are skipped."""
    examples = []
    for row in rows:
        prompt = vocab.from_target_ids(row["prompt_ids"])
        response = vocab.from_target_ids(row["response_ids"])
        if len(prompt) + min_response > max_len:
            continue
        examples.append(Example(prompt, response[: max_len - len(prompt)], row["response_ids"]))
    return examples


def collate(examples: list[Example], pad_id: int):
    """Right-padded input_ids / attention_mask / labels (prompt and padding ignored)."""
    import torch

    length = max(len(e.prompt) + len(e.response) for e in examples)
    input_ids = np.full((len(examples), length), pad_id, dtype=np.int64)
    labels = np.full((len(examples), length), IGNORE, dtype=np.int64)
    mask = np.zeros((len(examples), length), dtype=np.int64)
    for row, e in enumerate(examples):
        ids = e.prompt + e.response
        input_ids[row, : len(ids)] = ids
        labels[row, len(e.prompt): len(ids)] = e.response
        mask[row, : len(ids)] = 1
    return {"input_ids": torch.from_numpy(input_ids), "attention_mask": torch.from_numpy(mask),
            "labels": torch.from_numpy(labels)}


class DistillBatches:
    """Examples in a fresh shuffled order each epoch, addressed by global index.

    With `cut_rate` > 0 (and `vocab`, `max_len`), responses are re-tokenized with
    round-start cuts (see the module docstring), drawn from a seed fixed by
    (seed, epoch, example), so resuming a run reproduces the same batches.
    """

    def __init__(self, examples: list[Example], pad_id: int, seed: int = 0, vocab=None,
                 cut_rate: float = 0.0, max_len: int | None = None):
        if cut_rate > 0 and (vocab is None or max_len is None):
            raise ValueError("cut_rate > 0 needs vocab and max_len")
        self.examples = examples
        self.pad_id = pad_id
        self.seed = seed
        self.vocab = vocab
        self.cut_rate = cut_rate
        self.max_len = max_len
        self._orders: dict[int, np.ndarray] = {}

    def _order(self, epoch: int) -> np.ndarray:
        if epoch not in self._orders:
            self._orders[epoch] = np.random.default_rng(self.seed + epoch).permutation(
                len(self.examples))
        return self._orders[epoch]

    def example(self, i: int) -> Example:
        """Example number i of the run (epoch i // n, position i % n)."""
        n = len(self.examples)
        epoch, k = i // n, int(self._order(i // n)[i % n])
        e = self.examples[k]
        if self.cut_rate <= 0 or e.response_target is None:
            return e
        rng = random.Random((self.seed * 1_000_003 + epoch) * 1_000_003 + k)
        response = segmented_ids(self.vocab, e.response_target, self.cut_rate, rng)
        return Example(e.prompt, response[: self.max_len - len(e.prompt)], e.response_target)

    def batch(self, indices) -> dict:
        return collate([self.example(i) for i in indices], self.pad_id)


def response_bits_per_byte(model, examples: list[Example], vocab, batch_size: int = 8,
                           max_examples: int | None = None) -> dict:
    """Loss on held-out target responses, in bits per byte (comparable across vocabs)."""
    import math

    import torch
    import torch.nn.functional as F

    from superspec.eval.heldout import token_byte_lengths

    device = next(model.parameters()).device
    byte_lengths = torch.from_numpy(token_byte_lengths(vocab)).to(device)
    examples = examples[:max_examples] if max_examples else examples
    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = {k: v.to(device) for k, v in
                     collate(examples[start:start + batch_size], vocab.eos_id).items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits.float()
            logits, labels = logits[:, :-1], batch["labels"][:, 1:]
            keep = labels != IGNORE
            total_nll += F.cross_entropy(logits[keep], labels[keep], reduction="sum").item()
            total_tokens += int(keep.sum())
            total_bytes += int(byte_lengths[labels[keep]].sum())
    return {"bits_per_byte": total_nll / (math.log(2) * max(total_bytes, 1)),
            "nll_per_token": total_nll / max(total_tokens, 1)}
