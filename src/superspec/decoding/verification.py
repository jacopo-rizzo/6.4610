"""The target model's side of a speculative round: accepting draft tokens.

Given the target's next-token logits at each proposed position (K + 1 rows for
K proposed tokens; the last row is for the bonus token after a full accept),
each rule returns (number of accepted tokens, next token). The next token is
either the replacement for the first rejected token or the bonus token.

All rules are lossless: the output has exactly the target's distribution.
tests/test_lossless_math.py checks this exactly on small examples.

Rejection sampling (Leviathan et al., 2023), used for standard and superword
drafts. For a proposed token x drawn from q, accept with probability
min(1, p(x) / q(x)); on rejection, sample from max(p - q, 0), renormalized.
This outputs exactly p for *any* q, as long as q is the distribution x was
really drawn from given everything decided so far. With superwords:
  * at the first token of a draft token, q is the draft's marginal over first
    target tokens (sum of q(w) over superwords w starting with t);
  * at later tokens of a superword, the superword is already decided, so q is
    a point mass: accept with probability p(x), else sample p with x removed.
Conditioning on the chosen superword is allowed because the accept/reject
decisions before it only looked at target tokens, never at which superword
produced them.

String-level exact match (SLEM, Timor et al., 2025), our fallback for
unrelated vocabularies. Sample y from p; accept iff y == x; otherwise output
y. Needs no draft probabilities at all, so the proposal can come from any
tokenizer (decode the draft text, re-encode it with the target tokenizer),
but it accepts less often than rejection sampling.

Greedy decoding (temperature 0) is the same for both: accept while x equals
the target's argmax.
"""

from __future__ import annotations

import torch

from superspec.decoding.drafting import Proposal
from superspec.decoding.sampling import SamplingConfig, normalize, sample, to_probs, uniform


def verify_greedy(target_logits: torch.Tensor, proposal: Proposal) -> tuple[int, int]:
    best = target_logits.argmax(dim=-1).tolist()
    for i, x in enumerate(proposal.target_tokens):
        if best[i] != x:
            return i, best[i]
    return len(proposal), best[len(proposal)]


def verify_rejection(target_logits: torch.Tensor, proposal: Proposal,
                     sampling: SamplingConfig, generator) -> tuple[int, int]:
    if sampling.greedy:
        return verify_greedy(target_logits, proposal)

    p = to_probs(target_logits, sampling)
    for i, (x, q) in enumerate(zip(proposal.target_tokens, proposal.q_rows)):
        p_x = float(p[i, x])
        q_x = 1.0 if q is None else float(q[x])
        if uniform(p.device, generator) < min(1.0, p_x / q_x):
            continue                                       # accepted

        if q is None:                                      # point mass on x
            residual = p[i].clone()
            residual[x] = 0.0
        else:
            residual = (p[i] - q).clamp(min=0.0)
        return i, sample(normalize(residual, fallback=p[i]), generator)

    return len(proposal), sample(p[len(proposal)], generator)


def verify_exact_match(target_logits: torch.Tensor, proposal: Proposal,
                       sampling: SamplingConfig, generator) -> tuple[int, int]:
    if sampling.greedy:
        return verify_greedy(target_logits, proposal)

    p = to_probs(target_logits, sampling)
    for i, x in enumerate(proposal.target_tokens):
        y = sample(p[i], generator)
        if y != x:
            return i, y
    return len(proposal), sample(p[len(proposal)], generator)
