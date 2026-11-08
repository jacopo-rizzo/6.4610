"""The draft model's side of a speculative round: proposing target tokens.

The draft runs autoregressively in its own vocabulary. Each draft token
expands to one target token (standard draft) or one or more (a SuperBPE
superword). A `Proposal` records the resulting target tokens, which draft
token each came from, and, for each target token, the draft distribution the
verifier checks it against (its "q row"):

  * the first target token of a draft token gets the first-token marginal
        q(t) = sum of q(w) over all draft tokens w whose first target token is t,
    i.e. the draft's probability of starting with t, over the target vocab;
  * target token j >= 1 of superword w gets one of two rows (`rows=`):
      "length_aware" (default): the draft's distribution of token j given that
          it drew *some* superword longer than j that starts like w:
              q_j(x) ∝ sum of q(v) over draft tokens v with
                       len(v) > j, v[:j] == w[:j], v[j] == x;
      "point_mass": None, a point mass on w[j] (the draft is conditioned on
          having drawn exactly w, so the token was certain).

Both rows are lossless (see verification.py). The point mass accepts token j
with probability only p(x), however good the draft is; the length-aware row
accepts with min(1, p(x) / q_j(x)), which is never lower, and it costs no extra
draft forward pass because it reuses the softmax that drew w.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from superspec.decoding.kv_cache import CachedLM
from superspec.decoding.sampling import SamplingConfig, sample, to_probs

# How the drafter fills q rows:
#   length_aware / point_mass  the two lossless rules above
#   wrong_marginal             DELIBERATELY BIASED control for the losslessness
#                              check: the first-token row uses only the base
#                              token's own probability q(t), ignoring superwords
#                              that start with t; later tokens are point masses
#   none                       no rows (exact match needs no draft probabilities)
ROW_RULES = ("length_aware", "point_mass", "wrong_marginal", "none")


@dataclass
class Proposal:
    target_tokens: list[int] = field(default_factory=list)
    # Per target token: draft probability over the target vocab, or None for a
    # point mass (point_mass rule, greedy decoding, or exact match).
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

    def __init__(self, draft: CachedLM, vocab, target_vocab_size: int,
                 rows: str = "length_aware"):
        if rows not in ROW_RULES:
            raise ValueError(f"rows must be one of {ROW_RULES}, got {rows!r}")
        self.lm = draft
        self.vocab = vocab
        self.target_vocab_size = target_vocab_size
        self.rows = rows
        expansions = [vocab.expand(w) for w in range(vocab.vocab_size)]
        # first_target[w] = first target token of draft token w, for the marginal above.
        self.first_target = torch.tensor([e[0] for e in expansions], device=draft.device)
        self._build_continuations(expansions, draft.device)

    def _build_continuations(self, expansions, device) -> None:
        """Index for the length-aware rows: for every prefix of j >= 1 target
        tokens, the draft tokens longer than j that start with it, and their
        (j+1)-th target token. Stored flat: prefix -> slice of two tensors."""
        groups: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
        for w, e in enumerate(expansions):
            for j in range(1, len(e)):
                groups[e[:j]].append((w, e[j]))
        ids, nxt = [], []
        self._span: dict[tuple[int, ...], tuple[int, int]] = {}
        for prefix, members in groups.items():
            start = len(ids)
            ids.extend(w for w, _ in members)
            nxt.extend(t for _, t in members)
            self._span[prefix] = (start, len(ids))
        self._cont_ids = torch.tensor(ids, dtype=torch.long, device=device)
        self._cont_next = torch.tensor(nxt, dtype=torch.long, device=device)

    def first_token_marginal(self, q: torch.Tensor) -> torch.Tensor:
        """Draft distribution over draft tokens -> distribution over first target token."""
        out = torch.zeros(self.target_vocab_size, device=q.device, dtype=q.dtype)
        return out.scatter_add_(0, self.first_target, q)

    def later_token_row(self, q: torch.Tensor, expansion: tuple[int, ...], j: int) -> torch.Tensor:
        """Length-aware row for target token j >= 1 of a draft token with this expansion."""
        start, end = self._span[expansion[:j]]
        members = self._cont_ids[start:end]
        row = torch.zeros(self.target_vocab_size, device=q.device, dtype=q.dtype)
        row.scatter_add_(0, self._cont_next[start:end], q[members])
        # The sum includes q(w) of the drawn token itself, so it is positive.
        return row / row.sum()

    def q_rows_for(self, q: torch.Tensor, expansion: tuple[int, ...]) -> list:
        """The q rows of the target tokens of one drawn draft token."""
        if self.rows == "none":
            return [None] * len(expansion)
        if self.rows == "wrong_marginal":
            first = q[: self.target_vocab_size].clone()
        else:
            first = self.first_token_marginal(q)
        later = [self.later_token_row(q, expansion, j) if self.rows == "length_aware" else None
                 for j in range(1, len(expansion))]
        return [first] + later

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
                w, q = int(logits.argmax()), None
            else:
                q = to_probs(logits, sampling)
                w = sample(q, generator)

            expansion = self.vocab.expand(w)
            if len(proposal) + len(expansion) > max_target_tokens:
                break
            index = len(proposal.draft_tokens)
            proposal.draft_tokens.append(w)
            proposal.target_tokens.extend(expansion)
            proposal.q_rows.extend([None] * len(expansion) if q is None
                                   else self.q_rows_for(q, expansion))
            proposal.source.extend([index] * len(expansion))
            proposal.num_bytes += len(self.vocab.token_bytes(w))
            tokens.append(w)

            if w in stop_ids or proposal.num_bytes >= max_bytes:
                break
        return proposal
