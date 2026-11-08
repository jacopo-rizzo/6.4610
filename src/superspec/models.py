"""Loading the target and draft models."""

from __future__ import annotations

import torch


def load_causal_lm(name_or_path: str, dtype: str = "bfloat16", device: str = "cuda"):
    """A HuggingFace causal LM in eval mode (weights frozen)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=getattr(torch, dtype))
    model.requires_grad_(False)
    return model.to(device).eval()
