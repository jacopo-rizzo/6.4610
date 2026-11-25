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
    # Per draft token: what it was chosen from (probabilities over the draft vocab when
    # sampling, logits when greedy), kept for Drafter.redraw. None for a token that did
    # not come from a forward pass of this round.
    step_scores: list[torch.Tensor | None] = field(default_factory=list)

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
                 rows: str = "length_aware", superword_bonus: float = 0.0):
        if rows not in ROW_RULES:
            raise ValueError(f"rows must be one of {ROW_RULES}, got {rows!r}")
        self.lm = draft
        self.vocab = vocab
        self.target_vocab_size = target_vocab_size
        self.rows = rows
        # Added to the logits of superword tokens before the draft samples, so it emits
        # more text per forward pass. The verifier gets the distribution the draft really
        # sampled from, so any value keeps the output exact.
        self.superword_bonus = superword_bonus if vocab.vocab_size > target_vocab_size else 0.0
        expansions = [vocab.expand(w) for w in range(vocab.vocab_size)]
        self._exact = {e: w for w, e in enumerate(expansions)}     # expansion -> draft token
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

    def redraw(self, scores: torch.Tensor, greedy: bool, prefix: tuple[int, ...], generator):
        """After a rejection: which draft token would the draft have used, given the target
        tokens `prefix` (the accepted part of the rejected draft token plus the target's
        own token)?

        `scores` is what the rejected draft token was chosen from, so this costs no forward
        pass. We draw again from it, restricted to draft tokens that are `prefix` exactly or
        start with it. If the draw is longer than `prefix`, its remaining target tokens are
        a proposal for the next round: returns (draft token, remaining target tokens, their
        q rows). Returns None if there is nothing to propose.

        Exact for the same reason as the rules in verification.py: the draw is fresh, its
        distribution is known, and the rows are that distribution given that the draw was
        longer than `prefix` (length-aware) or given the drawn token itself (point mass).
        """
        span = self._span.get(prefix)
        if span is None or self.rows not in ("length_aware", "point_mass"):
            return None
        candidates = self._cont_ids[span[0]: span[1]]
        exact = self._exact.get(prefix)
        if exact is not None:
            candidates = torch.cat([candidates, torch.tensor([exact], device=candidates.device)])
        weights = scores[candidates]
        if greedy:
            pick = int(weights.argmax())
        else:
            if float(weights.sum()) <= 0:
                return None
            pick = sample(weights / weights.sum(), generator)
        w = int(candidates[pick])
        if w == exact:
            return None
        expansion = self.vocab.expand(w)
        later = range(len(prefix), len(expansion))
        if greedy or self.rows == "point_mass":
            rows = [None] * len(later)
        else:
            rows = [self.later_token_row(scores, expansion, j) for j in later]
        return w, expansion[len(prefix):], rows

    def propose(self, context: list[int], sampling: SamplingConfig, generator,
                max_bytes: int, max_steps: int, max_target_tokens: int,
                stop_ids: set[int], stop_below: float = 0.0) -> Proposal:
        """Draft until `max_bytes` bytes are proposed (the draft length is fixed in
        bytes so both tokenizers are compared fairly), or another limit is hit.

        `context` is the current text in draft token ids.

        `stop_below` > 0 ends the proposal after the first draft token that the draft
        itself gave a probability below that value: an unsure token is likely to be
        rejected, and everything drafted after a rejection is wasted. The token is still
        proposed, and the decision uses only the draft's own probabilities, so the
        verification rules stay exact.
        """
        proposal = Proposal()
        if max_target_tokens <= 0:
            return proposal
        tokens = list(context)
        for _ in range(max_steps):
            logits = self.lm.logits(tokens, 1)[0, : self.vocab.vocab_size]
            proposal.num_steps += 1
            if self.superword_bonus:
                logits = logits.clone()
                logits[self.target_vocab_size:] += self.superword_bonus
            if sampling.greedy:
                w, q = int(logits.argmax()), None
                confidence = float(logits.float().softmax(-1)[w]) if stop_below > 0 else 1.0
            else:
                q = to_probs(logits, sampling)
                w = sample(q, generator)
                confidence = float(q[w]) if stop_below > 0 else 1.0

            expansion = self.vocab.expand(w)
            if len(proposal) + len(expansion) > max_target_tokens:
                break
            index = len(proposal.draft_tokens)
            proposal.draft_tokens.append(w)
            proposal.step_scores.append(logits if q is None else q)
            proposal.target_tokens.extend(expansion)
            proposal.q_rows.extend([None] * len(expansion) if q is None
                                   else self.q_rows_for(q, expansion))
            proposal.source.extend([index] * len(expansion))
            proposal.num_bytes += len(self.vocab.token_bytes(w))
            tokens.append(w)

            if w in stop_ids or proposal.num_bytes >= max_bytes or confidence < stop_below:
                break
        return proposal
