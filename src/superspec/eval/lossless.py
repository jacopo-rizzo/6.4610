"""Empirical losslessness check, as in the proposal.

For each of a few prompts, sample many outputs with the target alone and with
each speculative method, and compare the distributions of the first
`prefix_bytes` bytes of output (byte-level, since the vocabularies differ).
Per prompt we get a permutation-test p-value (stats.py); if a method is
lossless, its p-values are uniform: about 1% fall below 0.01, and Fisher's
combined p-value is not small. A buggy method shows up as many tiny p-values.

This complements tests/test_lossless_math.py (exact, on toy distributions) by
running the real decoding code on the real models.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from superspec.decoding import SamplingConfig, SpeculativeDecoder, autoregressive_generate
from superspec.eval.common import Draft, Target, chat_prompt_ids
from superspec.eval.prompts import Prompt
from superspec.eval.stats import fisher_combined_p, permutation_test
from superspec.tokenization.vocab import vocab_kind

logger = logging.getLogger(__name__)


def output_prefix(vocab, tokens: list[int], stop_ids: set[int], prefix_bytes: int) -> bytes:
    text = b"".join(vocab.token_bytes(t) for t in tokens if t not in stop_ids)
    return text[:prefix_bytes]


def sample_prefixes(generate, prompt_ids, vocab, stop_ids, num_samples, prefix_bytes,
                    seed, device) -> list[bytes]:
    out = []
    for s in range(num_samples):
        gen = torch.Generator(device=device).manual_seed(seed + s)
        out.append(output_prefix(vocab, generate(prompt_ids, gen).tokens, stop_ids, prefix_bytes))
    return out


def run_lossless_check(cfg, target: Target, drafts: dict[str, Draft], prompts: list[Prompt],
                       device: str) -> dict:
    lc = cfg.lossless_check
    sampling = SamplingConfig(temperature=lc.temperature, top_p=cfg.speculative.top_p)
    n, prefix_bytes, max_new = lc.samples_per_prompt, lc.prefix_bytes, lc.max_new_tokens
    prompts = prompts[: lc.num_prompts]

    def ar(prompt, gen):
        return autoregressive_generate(target.lm, target.vocab, prompt, max_new, sampling,
                                       target.stop_ids, gen)

    methods = {}
    for name, draft in drafts.items():
        names = list(cfg.speculative.methods[vocab_kind(draft.vocab)]) + ["exact_match"]
        for method in names:
            decoder = SpeculativeDecoder(
                target.lm, target.vocab, draft.lm, draft.vocab, method, sampling,
                target.stop_ids, max_draft_bytes=lc.draft_len_bytes, time_phases=False)
            methods[f"{name}/{method}"] = (lambda d: lambda p, g: d.generate(p, max_new, g))(decoder)

    # Control: target vs target with different seeds; its p-values show what
    # "no difference" looks like with this many samples.
    methods["autoregressive (control)"] = ar

    p_values = {m: [] for m in methods}
    rng = np.random.default_rng(cfg.seed)
    for i, prompt in enumerate(prompts):
        ids = chat_prompt_ids(target.vocab, prompt.text)
        reference = sample_prefixes(ar, ids, target.vocab, target.stop_ids, n, prefix_bytes,
                                    seed=10_000 * i, device=device)
        for m, generate in methods.items():
            # Seeds disjoint from the reference so the control is a fair comparison.
            samples = sample_prefixes(generate, ids, target.vocab, target.stop_ids, n,
                                      prefix_bytes, seed=10_000 * i + 5_000, device=device)
            tv, p = permutation_test(reference, samples, lc.num_permutations, rng)
            p_values[m].append(p)
            logger.info("prompt %d %s: TV %.3f, p %.3f", i, m, tv, p)

    return {
        m: {"p_values": ps, "fisher_p": fisher_combined_p(ps),
            "frac_below_0.01": float(np.mean(np.array(ps) < 0.01))}
        for m, ps in p_values.items()
    }
