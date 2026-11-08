"""Loading the target, drafts and prompts for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from superspec.config import load_config
from superspec.decoding import CachedLM
from superspec.models import load_causal_lm
from superspec.tokenization import load_vocab


@dataclass
class Target:
    lm: CachedLM
    vocab: object
    stop_ids: set[int]


@dataclass
class Draft:
    name: str
    lm: CachedLM
    vocab: object


def stop_token_ids(vocab, tokenizer_cfg) -> set[int]:
    """Generation ends at the end-of-turn or end-of-text token
    (Llama-3: <|eot_id|> / <|end_of_text|>; OLMo-2: <|endoftext|> for both)."""
    special = tokenizer_cfg.special_tokens
    return {vocab.hf.convert_tokens_to_ids(special[k]) for k in ("end_of_turn", "end_of_text")}


def chat_prompt_ids(target_vocab, text: str) -> list[int]:
    """A single user turn, formatted with the target's chat template."""
    ids = target_vocab.hf.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):          # newer transformers may return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def load_target(cfg, device: str) -> Target:
    """The target from the config's target profile (cfg.target)."""
    tokenizer_cfg = load_config(cfg.target.tokenizer)
    vocab = load_vocab(tokenizer_cfg)
    model = load_causal_lm(cfg.target.hf_name, dtype=cfg.get("dtype", "bfloat16"), device=device)
    return Target(CachedLM(model), vocab, stop_token_ids(vocab, tokenizer_cfg))


def load_drafts(cfg, device: str, names: list[str] | None = None) -> dict[str, Draft]:
    drafts = {}
    for entry in cfg.drafts:
        if names and entry.name not in names:
            continue
        vocab = load_vocab(entry.tokenizer)
        check_draft_tokenizer(entry.checkpoint, vocab)
        model = load_causal_lm(entry.checkpoint, dtype=cfg.get("dtype", "bfloat16"), device=device)
        drafts[entry.name] = Draft(entry.name, CachedLM(model), vocab)
    return drafts


def check_draft_tokenizer(checkpoint, vocab) -> None:
    """Fail early if a draft was trained with a different tokenizer than configured."""
    from superspec.training.model import read_draft_meta

    meta = read_draft_meta(checkpoint)
    if meta is not None and meta["tokenizer"] != vocab.name:
        raise ValueError(f"{checkpoint} was trained with tokenizer {meta['tokenizer']!r}, "
                         f"but the eval config pairs it with {vocab.name!r}")
