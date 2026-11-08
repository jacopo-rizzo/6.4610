"""A tiny word-level stand-in for the Llama-3 tokenizer, for tests.

It splits text roughly like Llama-3's pretokenizer (leading space attached
to words, punctuation and newlines separate) and gives each piece an id.
"""

from __future__ import annotations

import re

PIECE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")


class ToyVocab:
    def __init__(self, corpus: list[str]):
        self.name = "toy"
        self.pieces = ["<bos>", "<eos>"]
        self.special_ids = {0, 1}
        self.bos_id, self.eos_id = 0, 1
        for text in corpus:
            for p in PIECE.findall(text):
                if p not in self.pieces:
                    self.pieces.append(p)
        self.index = {p: i for i, p in enumerate(self.pieces)}
        self.vocab_size = self.target_vocab_size = len(self.pieces)

    def encode(self, text):
        return [self.index[p] for p in PIECE.findall(text)]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)

    def token_bytes(self, i):
        return self.pieces[i].encode()

    def to_target_ids(self, ids):
        return list(ids)

    def from_target_ids(self, ids):
        return list(ids)

    def save(self, directory):
        pass
