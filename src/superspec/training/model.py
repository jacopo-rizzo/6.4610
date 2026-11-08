"""Building and saving the Llama-style draft models.

A trained draft is saved with `save_pretrained`, so the rest of the code loads
it like any HuggingFace model (superspec.models.load_causal_lm). Next to the
weights we write superspec.json recording which tokenizer it was trained with.
"""

from __future__ import annotations

import json
from pathlib import Path

from superspec.config import load_config

META_FILE = "superspec.json"


def padded_vocab_size(vocab_size: int, multiple: int = 128) -> int:
    """Round the embedding size up for faster matmuls. The extra ids are never
    produced by the tokenizer and the decoders ignore them."""
    return -(-vocab_size // multiple) * multiple


def build_draft_model(model_cfg, vocab_size: int):
    """A randomly initialized LlamaForCausalLM from a configs/model/*.yaml file."""
    from transformers import LlamaConfig, LlamaForCausalLM

    if isinstance(model_cfg, (str, Path)):
        model_cfg = load_config(model_cfg)
    config = LlamaConfig(
        vocab_size=padded_vocab_size(vocab_size),
        hidden_size=model_cfg.hidden_size,
        num_hidden_layers=model_cfg.num_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        intermediate_size=model_cfg.intermediate_size,
        max_position_embeddings=model_cfg.max_position_embeddings,
        rope_theta=model_cfg.rope_theta,
        rms_norm_eps=model_cfg.rms_norm_eps,
        tie_word_embeddings=model_cfg.tie_word_embeddings,
        attn_implementation="sdpa",
    )
    return LlamaForCausalLM(config)


def parameter_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())   # tied weights counted once
    embedding = model.get_input_embeddings().weight.numel()
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding}


def run_name(model_cfg_name: str, vocab_name: str) -> str:
    """e.g. draft_50m_llama3_superbpe; the eval configs expect these folder names."""
    return f"{model_cfg_name}_{vocab_name}"


def save_draft(model, directory: Path, vocab, extra: dict | None = None) -> None:
    """Save weights + config (HF format) and which tokenizer the draft uses."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    meta = {"tokenizer": vocab.name, "vocab_size": vocab.vocab_size,
            "parameters": parameter_counts(model), **(extra or {})}
    (directory / META_FILE).write_text(json.dumps(meta, indent=2))


def read_draft_meta(directory: str | Path) -> dict | None:
    path = Path(directory) / META_FILE
    return json.loads(path.read_text()) if path.exists() else None
