"""A HuggingFace causal LM plus its KV cache, with rollback.

Speculative decoding feeds tokens that may later be rejected, so the cache
has to be cut back afterwards. `CachedLM` remembers exactly which tokens its
cache holds. When asked for logits for a token sequence, it keeps the longest
cached prefix that the new sequence shares, crops the rest of the cache, and
runs the model only on the remaining tokens with the correct position ids.

So callers never crop by hand: rejected draft tokens, and draft tokens that
change when accepted text is re-tokenized into superwords, simply stop
matching the cache and are dropped on the next call.
"""

from __future__ import annotations

import inspect

import torch
from transformers import DynamicCache


def common_prefix_length(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class CachedLM:
    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        params = inspect.signature(model.forward).parameters
        # Only compute logits for the positions we need (the vocab is large).
        # The argument was renamed in transformers 4.50.
        self._logits_arg = next((a for a in ("logits_to_keep", "num_logits_to_keep")
                                 if a in params), None)
        self.reset()

    def reset(self) -> None:
        self.cache = DynamicCache()
        self.tokens: list[int] = []        # the tokens whose keys/values are in the cache

    def rollback(self, length: int) -> None:
        """Keep only the first `length` cached tokens."""
        if length >= len(self.tokens):
            return
        if length == 0:
            self.cache = DynamicCache()
        else:
            # A negative count means "remove this many tokens" in every transformers
            # version. A positive count meant "keep this many" before 5.18 and no
            # longer does, which would silently corrupt the cache.
            self.cache.crop(length - len(self.tokens))
            cached = self.cache.get_seq_length()
            if cached != length:
                raise RuntimeError(f"KV cache holds {cached} tokens after rollback, "
                                   f"expected {length}")
        self.tokens = self.tokens[:length]

    @torch.no_grad()
    def logits(self, tokens: list[int], num_positions: int = 1) -> torch.Tensor:
        """Next-token logits at the last `num_positions` positions of `tokens`.

        Row j predicts the token after tokens[len(tokens) - num_positions + j].
        Afterwards the cache holds all of `tokens`.
        """
        if not 1 <= num_positions <= len(tokens):
            raise ValueError(f"num_positions={num_positions} for {len(tokens)} tokens")

        # Every position we need logits for must be recomputed, so keep at most
        # the tokens before them.
        keep = min(common_prefix_length(self.tokens, tokens), len(tokens) - num_positions)
        self.rollback(keep)

        new_tokens = torch.tensor([tokens[keep:]], device=self.device)
        positions = torch.arange(keep, len(tokens), device=self.device)
        extra = {self._logits_arg: num_positions} if self._logits_arg else {}
        out = self.model(
            input_ids=new_tokens,
            position_ids=positions[None],
            cache_position=positions,
            past_key_values=self.cache,
            use_cache=True,
            **extra,
        )
        self.cache = out.past_key_values
        self.tokens = list(tokens)
        return out.logits[0, -num_positions:].float()
