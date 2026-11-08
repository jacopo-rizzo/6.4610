"""The draft model's side of a speculative round: proposing target tokens.

The draft runs autoregressively in its own vocabulary. Each draft token
expands to one target token (standard draft) or one or more (a SuperBPE
superword). A `Proposal` records the resulting target tokens, which draft
token each came from, and the draft distribution the verifier needs for each:

  * the first target token of a draft token gets
        q(t) = sum of q(w) over all draft tokens w whose first target token is t,
    i.e. the draft's probability of starting with t, over the target vocab;
  * later target tokens of the same superword get None, meaning a point mass:
    once the draft picked superword w, those tokens were certain (q = 1).

Both are the true conditional distributions of what the draft proposed, which
is all that lossless rejection sampling requires (see verification.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from superspec.decoding.kv_cache import CachedLM
from superspec.decoding.sampling import SamplingConfig, sample, to_probs


@dataclass
class Proposal:
    target_tokens: list[int] = field(default_factory=list)
    # Per target token: draft probability over the target vocab, or None for a
    # point mass (a non-first token of a superword, or greedy decoding).
    q_rows: list[torch.Tensor | None] = field(default_factory=list)
    draft_tokens: list[int] = field(default_factory=list)
    # Per target token: index into draft_tokens of the draft token it came from.
    # This is the draft-token <-> target-position mapping.
    source: list[int] = field(default_factory=list)
    num_bytes: int = 0
    num_steps: int = 0          # draft forward passes used (can exceed len(draft_tokens) by 1)

    def __len__(self) -> int:
        return len(self.target_tokens)

    def fully_accepted_draft_tokens(self, num_accepted: int) -> int:
        """How many draft tokens lie entirely inside the first `num_accepted` target tokens."""
        if num_accepted == len(self.target_tokens):
            return len(self.draft_tokens)
        return self.source[num_accepted]   # the draft token that got cut is not complete


class Drafter:
    """Wraps a draft model and its vocab (standard or SuperBPE)."""

    def __init__(self, draft: CachedLM, vocab, target_vocab_size: int):
        self.lm = draft
        self.vocab = vocab
        self.target_vocab_size = target_vocab_size
        # first_target[w] = first target token of draft token w, for the marginal above.
        first = [vocab.expand(w)[0] for w in range(vocab.vocab_size)]
        self.first_target = torch.tensor(first, device=draft.device)

    def first_token_marginal(self, q: torch.Tensor) -> torch.Tensor:
        """Draft distribution over draft tokens -> distribution over first target token."""
        out = torch.zeros(self.target_vocab_size, device=q.device, dtype=q.dtype)
        return out.scatter_add_(0, self.first_target, q)

    def propose(self, context: list[int], sampling: SamplingConfig, generator,
                max_bytes: int, max_steps: int, max_target_tokens: int,
                stop_ids: set[int]) -> Proposal:
        """Draft until `max_bytes` bytes are proposed (the draft length is fixed in
        bytes so both tokenizers are compared fairly), or another limit is hit.

        `context` is the current text in draft token ids.
        """
        proposal = Proposal()
        if max_target_tokens <= 0:
            return proposal
        tokens = list(context)
        for _ in range(max_steps):
            logits = self.lm.logits(tokens, 1)[0, : self.vocab.vocab_size]
            proposal.num_steps += 1
            if sampling.greedy:
                w, q_first = int(logits.argmax()), None
            else:
                q = to_probs(logits, sampling)
                w = sample(q, generator)
                q_first = self.first_token_marginal(q)

            expansion = self.vocab.expand(w)
            if len(proposal) + len(expansion) > max_target_tokens:
                break
            index = len(proposal.draft_tokens)
            proposal.draft_tokens.append(w)
            proposal.target_tokens.extend(expansion)
            proposal.q_rows.extend([q_first] + [None] * (len(expansion) - 1))
            proposal.source.extend([index] * len(expansion))
            proposal.num_bytes += len(self.vocab.token_bytes(w))
            tokens.append(w)

            if w in stop_ids or proposal.num_bytes >= max_bytes:
                break
        return proposal
