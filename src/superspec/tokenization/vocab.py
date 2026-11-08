"""A common interface over the standard and SuperBPE tokenizers.

Everything downstream (pretokenization, training, speculative decoding) talks
to a `Vocab`, so it does not need to know which tokenizer a draft uses:

    vocab = load_vocab("tokenizer/llama3_superbpe.yaml")
    ids = vocab.encode("Hello there, how are you?")
    target_ids = vocab.to_target_ids(ids)   # same text in Llama-3 token ids

For the standard vocab the target-id conversions are the identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from superspec.tokenization.byte_level import token_string_to_bytes


class Vocab(Protocol):
    name: str
    vocab_size: int          # size of this tokenizer's vocabulary
    target_vocab_size: int   # size of the target (Llama-3) vocabulary
    bos_id: int
    eos_id: int              # end-of-document token, used between pretraining docs

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def to_target_ids(self, ids: list[int]) -> list[int]: ...
    def from_target_ids(self, target_ids: list[int]) -> list[int]: ...
    def token_bytes(self, token_id: int) -> bytes: ...


class StandardVocab:
    """The target model's own tokenizer (Llama-3 BPE), wrapped as a `Vocab`."""

    def __init__(self, hf_tokenizer, name: str = "llama3",
                 bos_token: str | None = None, eos_token: str | None = None):
        self.hf = hf_tokenizer
        self.name = name
        self.vocab_size = len(hf_tokenizer)
        self.target_vocab_size = self.vocab_size

        # Special tokens (Llama-3's <|...|> and reserved tokens) have no byte-level
        # encoding; we record their literal text instead.
        self.special_ids = set(hf_tokenizer.all_special_ids)
        self.special_ids |= set(getattr(hf_tokenizer, "added_tokens_decoder", {}).keys())

        self._bytes = []
        for i, tok in enumerate(hf_tokenizer.convert_ids_to_tokens(list(range(self.vocab_size)))):
            if i in self.special_ids or tok is None:
                self._bytes.append((tok or "").encode("utf-8"))
            else:
                self._bytes.append(token_string_to_bytes(tok))

        self.bos_id = hf_tokenizer.convert_tokens_to_ids(bos_token or hf_tokenizer.bos_token)
        self.eos_id = hf_tokenizer.convert_tokens_to_ids(eos_token or hf_tokenizer.eos_token)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, name: str = "llama3",
                        bos_token: str | None = None, eos_token: str | None = None):
        from transformers import AutoTokenizer  # imported lazily: heavy dependency

        hf = AutoTokenizer.from_pretrained(str(name_or_path))
        return cls(hf, name=name, bos_token=bos_token, eos_token=eos_token)

    def encode(self, text: str) -> list[int]:
        return self.hf(text, add_special_tokens=False)["input_ids"]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.hf(texts, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.hf.decode(ids)

    def token_bytes(self, token_id: int) -> bytes:
        return self._bytes[token_id]

    def to_target_ids(self, ids: list[int]) -> list[int]:
        return list(ids)

    def from_target_ids(self, target_ids: list[int]) -> list[int]:
        return list(target_ids)

    def save(self, directory: str | Path) -> None:
        self.hf.save_pretrained(str(directory))


def load_vocab(tokenizer_config) -> Vocab:
    """Load a tokenizer from its config (a path under `configs/` or a loaded config).

    A config with a `base_tokenizer` key is a SuperBPE tokenizer, loaded from its
    `output_dir` (run scripts/train_superbpe.py first). Otherwise it is a
    standard HF tokenizer loaded from `hf_name`.
    """
    from superspec.config import load_config
    from superspec.tokenization.superbpe import SuperBPETokenizer

    cfg = load_config(tokenizer_config) if isinstance(tokenizer_config, (str, Path)) \
        else tokenizer_config

    if "base_tokenizer" in cfg:
        return SuperBPETokenizer.load(cfg.output_dir)

    special = cfg.get("special_tokens", {})
    # Pretraining separates documents with <|end_of_text|>, not the chat <|eot_id|>.
    return StandardVocab.from_pretrained(
        cfg.hf_name, name=cfg.name,
        bos_token=special.get("bos"),
        eos_token=special.get("eos_alt", special.get("eos")),
    )
