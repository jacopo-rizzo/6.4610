"""The SuperBPE draft tokenizer: Llama-3 tokens plus learned superwords.

Ids 0 .. t-1 are exactly the target's tokens; ids t .. T-1 are superwords,
each standing for a fixed sequence of target tokens. Encoding text means
tokenizing it with the target tokenizer, then applying the superword merges
in the order they were learned (standard BPE), chunk by chunk.

Saved layout (written by scripts/train_superbpe.py):
    <dir>/superbpe.json    merges and settings
    <dir>/base/            the target tokenizer, so loading works offline
    <dir>/superwords.tsv   human-readable list of the superwords
"""

from __future__ import annotations

import json
from pathlib import Path

from superspec.tokenization.superword import (
    Pair,
    is_barrier_token,
    merge_pair,
    split_into_chunks,
)
from superspec.tokenization.vocab import StandardVocab

FORMAT_VERSION = 1


class SuperBPETokenizer:
    """Implements the `Vocab` interface (see vocab.py)."""

    def __init__(self, base: StandardVocab, merges: list[Pair], name: str = "llama3_superbpe",
                 max_words_per_token: int = 4, chunk_cache_size: int = 1_000_000):
        self.base = base
        self.name = name
        self.merges = [tuple(m) for m in merges]
        self.max_words_per_token = max_words_per_token

        self.target_vocab_size = base.vocab_size
        self.vocab_size = base.vocab_size + len(self.merges)
        self.bos_id = base.bos_id
        self.eos_id = base.eos_id

        # Merge priority: earlier merges are applied first.
        self.rank: dict[Pair, int] = {pair: k for k, pair in enumerate(self.merges)}

        # expansion[i] = the target token ids that id i stands for.
        self.expansion: list[tuple[int, ...]] = [(i,) for i in range(base.vocab_size)]
        for a, b in self.merges:
            self.expansion.append(self.expansion[a] + self.expansion[b])

        # The first target token of every id. The verifier uses this to turn the
        # draft's distribution over superwords into one over target tokens:
        # P(target token t) = sum of q(w) over all w with first_target_token[w] == t.
        self.first_target_token: list[int] = [e[0] for e in self.expansion]

        self.is_barrier = [
            is_barrier_token(base.token_bytes(i), i in base.special_ids)
            for i in range(base.vocab_size)
        ]

        self._chunk_cache: dict[tuple[int, ...], list[int]] = {}
        self._chunk_cache_size = chunk_cache_size

    # ----- encoding -----

    def encode(self, text: str) -> list[int]:
        return self.from_target_ids(self.base.encode(text))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.from_target_ids(ids) for ids in self.base.encode_batch(texts)]

    def from_target_ids(self, target_ids: list[int]) -> list[int]:
        """Re-tokenize a sequence of target tokens into superword ids."""
        chunks, barriers = split_into_chunks(target_ids, self.is_barrier)
        out: list[int] = []
        for k, chunk in enumerate(chunks):
            out.extend(self._encode_chunk(chunk))
            if k < len(barriers):
                out.append(barriers[k])
        return out

    def _encode_chunk(self, chunk: list[int]) -> list[int]:
        if len(chunk) < 2:
            return chunk
        key = tuple(chunk)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            return cached

        ids = chunk
        while len(ids) >= 2:
            # Apply the highest-priority (earliest-learned) merge present.
            best = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if best not in self.rank:
                break
            ids = merge_pair(ids, best, self.target_vocab_size + self.rank[best])

        if len(self._chunk_cache) < self._chunk_cache_size:
            self._chunk_cache[key] = ids
        return ids

    # ----- decoding -----

    def to_target_ids(self, ids: list[int]) -> list[int]:
        out: list[int] = []
        for i in ids:
            out.extend(self.expansion[i])
        return out

    def expand(self, token_id: int) -> tuple[int, ...]:
        return self.expansion[token_id]

    def decode(self, ids: list[int]) -> str:
        return self.base.decode(self.to_target_ids(ids))

    def token_bytes(self, token_id: int) -> bytes:
        return b"".join(self.base.token_bytes(i) for i in self.expansion[token_id])

    def is_superword(self, token_id: int) -> bool:
        return token_id >= self.target_vocab_size

    # ----- saving / loading -----

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.base.save(directory / "base")
        meta = {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "base_name": self.base.name,
            "base_vocab_size": self.target_vocab_size,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "max_words_per_token": self.max_words_per_token,
            "merges": [list(m) for m in self.merges],
        }
        (directory / "superbpe.json").write_text(json.dumps(meta))

        with open(directory / "superwords.tsv", "w", encoding="utf-8") as f:
            f.write("id\tnum_target_tokens\ttext\n")
            for i in range(self.target_vocab_size, self.vocab_size):
                text = self.token_bytes(i).decode("utf-8", errors="replace")
                f.write(f"{i}\t{len(self.expansion[i])}\t{text!r}\n")

    @classmethod
    def load(cls, directory: str | Path) -> SuperBPETokenizer:
        directory = Path(directory)
        meta = json.loads((directory / "superbpe.json").read_text())
        if meta["format_version"] != FORMAT_VERSION:
            raise ValueError(f"Unsupported SuperBPE format {meta['format_version']}")

        base = StandardVocab.from_pretrained(directory / "base", name=meta["base_name"])
        base.bos_id, base.eos_id = meta["bos_id"], meta["eos_id"]
        if base.vocab_size != meta["base_vocab_size"]:
            raise ValueError("Saved base tokenizer does not match the merges' base vocab size")

        return cls(base, meta["merges"], name=meta["name"],
                   max_words_per_token=meta["max_words_per_token"])
