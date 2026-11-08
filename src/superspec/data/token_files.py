"""Pretokenized data on disk: one flat file of token ids per split.

    <dir>/train.bin    uint32 token ids, documents separated by vocab.eos_id
    <dir>/train.json   metadata: token/byte/document counts, tokenizer, eos id
    <dir>/val.bin, <dir>/val.json

uint32 is needed because both vocabularies exceed 65,535 ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DTYPE = np.uint32


class TokenFileWriter:
    """Append tokenized documents to <dir>/<split>.bin, then write the metadata."""

    def __init__(self, directory: str | Path, split: str, vocab):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.vocab = vocab
        self.num_tokens = self.num_bytes = self.num_docs = 0
        self._file = open(self.directory / f"{split}.bin", "wb")

    def write_docs(self, docs: list[list[int]], num_bytes: int) -> None:
        flat = []
        for ids in docs:
            flat.extend(ids)
            flat.append(self.vocab.eos_id)
        np.asarray(flat, dtype=DTYPE).tofile(self._file)
        self.num_tokens += len(flat)
        self.num_bytes += num_bytes
        self.num_docs += len(docs)

    def close(self) -> dict:
        self._file.close()
        meta = {
            "split": self.split,
            "tokenizer": self.vocab.name,
            "vocab_size": self.vocab.vocab_size,
            "eos_id": self.vocab.eos_id,
            "dtype": np.dtype(DTYPE).name,
            "num_tokens": self.num_tokens,
            "num_bytes": self.num_bytes,
            "num_docs": self.num_docs,
            "bytes_per_token": self.num_bytes / max(self.num_tokens, 1),
        }
        (self.directory / f"{self.split}.json").write_text(json.dumps(meta, indent=2))
        return meta


class TokenFile:
    """Read-only, memory-mapped view of one pretokenized split."""

    def __init__(self, directory: str | Path, split: str):
        directory = Path(directory)
        self.meta = json.loads((directory / f"{split}.json").read_text())
        self.tokens = np.memmap(directory / f"{split}.bin", dtype=self.meta["dtype"], mode="r")
        assert len(self.tokens) == self.meta["num_tokens"], "token file and metadata disagree"

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def bytes_per_token(self) -> float:
        return self.meta["bytes_per_token"]

    def random_windows(self, batch_size: int, length: int, rng: np.random.Generator) -> np.ndarray:
        """`batch_size` random contiguous windows of `length` tokens, as int64.

        Windows may cross document boundaries (marked by eos), as usual for
        pretraining. Ask for seq_len + 1 tokens to get inputs and shifted targets.
        """
        starts = rng.integers(0, len(self.tokens) - length, size=batch_size)
        return np.stack([self.tokens[s:s + length] for s in starts]).astype(np.int64)
