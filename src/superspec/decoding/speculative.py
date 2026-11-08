"""Speculative decoding: the draft proposes, the target verifies in one pass.

Three methods, all lossless:

  "standard"     Leviathan et al. (2023). Draft and target share a vocabulary.
  "superword"    Ours. The draft uses the SuperBPE vocabulary; each superword
                 expands to several target tokens, verified with the first-token
                 marginal / point-mass rule (see verification.py).
  "exact_match"  Timor et al. (2025) string-level exact match, our fallback that
                 works for any draft tokenizer. With retokenize=True the draft's
                 text is re-encoded with the target tokenizer, as in the paper;
                 with retokenize=False the superword expansions are used
                 directly, which isolates the effect of the verification rule.

One round:
  1. Re-tokenize the current text (target ids) into draft ids. For a SuperBPE
     draft this can merge recently accepted tokens into superwords.
  2. The draft proposes target tokens (a fixed number of bytes).
  3. The target scores the context plus all proposed tokens in one forward
     pass, and the verifier accepts a prefix and samples one more token.
Both models' KV caches are rolled back automatically: CachedLM keeps only the
cached prefix that still matches the new sequence (see kv_cache.py).
"""

from __future__ import annotations

import time

import torch

from superspec.decoding.autoregressive import synchronize
from superspec.decoding.drafting import Drafter, Proposal
from superspec.decoding.kv_cache import CachedLM
from superspec.decoding.result import GenerationResult, count_bytes
from superspec.decoding.sampling import SamplingConfig
from superspec.decoding.verification import verify_exact_match, verify_rejection

METHODS = ("standard", "superword", "exact_match")


class SpeculativeDecoder:
    def __init__(
        self,
        target: CachedLM,
        target_vocab,
        draft: CachedLM,
        draft_vocab,
        method: str,
        sampling: SamplingConfig,
        stop_ids: set[int],
        max_draft_bytes: int = 32,
        max_draft_steps: int = 16,
        max_target_tokens_per_round: int = 64,
        retokenize: bool = True,
        time_phases: bool = True,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if method == "standard" and draft_vocab.vocab_size != target_vocab.vocab_size:
            raise ValueError("'standard' needs the draft to use the target vocabulary; "
                             "use 'superword' or 'exact_match' for a SuperBPE draft")
        if method == "superword" and not hasattr(draft_vocab, "merges"):
            raise ValueError("'superword' needs a SuperBPE draft vocabulary")

        self.target = target
        self.target_vocab = target_vocab
        self.draft_vocab = draft_vocab
        self.drafter = Drafter(draft, draft_vocab, target_vocab.vocab_size)
        self.method = method
        self.sampling = sampling
        self.stop_ids = set(stop_ids)
        self.max_draft_bytes = max_draft_bytes
        self.max_draft_steps = max_draft_steps
        self.max_target_tokens_per_round = max_target_tokens_per_round
        self.retokenize = retokenize
        self.time_phases = time_phases   # sync the GPU between phases to time each one

    def _now(self) -> float:
        if self.time_phases:
            synchronize(self.target.device)
        return time.perf_counter()

    def _to_target_strings(self, proposal: Proposal, max_tokens: int) -> Proposal:
        """For exact_match: decode the draft's text and re-encode it with the target
        tokenizer. Exact match needs no draft probabilities, so q_rows are dropped."""
        text = b"".join(self.draft_vocab.token_bytes(w) for w in proposal.draft_tokens)
        target_tokens = self.target_vocab.encode(text.decode("utf-8", errors="replace"))
        target_tokens = target_tokens[:max_tokens]
        return Proposal(target_tokens=target_tokens, q_rows=[None] * len(target_tokens),
                        draft_tokens=proposal.draft_tokens,
                        source=[0] * len(target_tokens),   # no exact mapping after re-encoding
                        num_bytes=proposal.num_bytes, num_steps=proposal.num_steps)

    def generate(self, prompt: list[int], max_new_tokens: int,
                 generator: torch.Generator | None = None) -> GenerationResult:
        """Generate up to `max_new_tokens` target tokens after `prompt` (target ids)."""
        self.target.reset()
        self.drafter.lm.reset()
        result = GenerationResult(prompt_length=len(prompt), tokens=[], num_bytes=0, wall_time=0)
        tokens = list(prompt)
        start = self._now()

        while len(tokens) - len(prompt) < max_new_tokens:
            remaining = max_new_tokens - (len(tokens) - len(prompt))
            # A round outputs up to (proposed + 1) tokens, so propose at most remaining - 1.
            max_tokens = min(self.max_target_tokens_per_round, remaining - 1)

            t0 = self._now()
            draft_context = self.draft_vocab.from_target_ids(tokens)
            t1 = self._now()
            proposal = self.drafter.propose(
                draft_context, self.sampling, generator, max_bytes=self.max_draft_bytes,
                max_steps=self.max_draft_steps, max_target_tokens=max_tokens,
                stop_ids=self.stop_ids)
            t2 = self._now()
            if self.method == "exact_match" and self.retokenize:
                proposal = self._to_target_strings(proposal, max_tokens)

            k = len(proposal)
            target_logits = self.target.logits(tokens + proposal.target_tokens, k + 1)
            t3 = self._now()

            if self.method == "exact_match":
                accepted, next_token = verify_exact_match(target_logits, proposal,
                                                          self.sampling, generator)
            else:
                accepted, next_token = verify_rejection(target_logits, proposal,
                                                        self.sampling, generator)
            new = proposal.target_tokens[:accepted] + [next_token]

            # Stop at the first stop token (it may also be an accepted draft token).
            stopped = False
            for i, t in enumerate(new):
                if t in self.stop_ids:
                    new, stopped = new[: i + 1], True
                    break
            tokens.extend(new)
            t4 = self._now()

            result.rounds += 1
            result.draft_steps += proposal.num_steps
            result.proposed_tokens += k
            result.accepted_tokens += accepted
            result.proposed_bytes += count_bytes(self.target_vocab, proposal.target_tokens,
                                                 self.stop_ids)
            result.accepted_bytes += count_bytes(self.target_vocab,
                                                 proposal.target_tokens[:accepted], self.stop_ids)
            result.accepted_per_round.append(accepted)
            result.draft_time += t2 - t1
            result.target_time += t3 - t2
            result.overhead_time += (t1 - t0) + (t4 - t3)
            if stopped:
                break

        result.wall_time = self._now() - start
        result.tokens = tokens[len(prompt):][:max_new_tokens]
        result.num_bytes = count_bytes(self.target_vocab, result.tokens, self.stop_ids)
        return result
