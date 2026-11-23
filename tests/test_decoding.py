"""Tests for the decoders, using tiny randomly initialized Llama models on CPU.

Needs torch and transformers (skipped otherwise). Run with `pytest tests/`.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from superspec.decoding import (  # noqa: E402
    CachedLM,
    SamplingConfig,
    SpeculativeDecoder,
    autoregressive_generate,
)
from superspec.decoding.drafting import Drafter  # noqa: E402
from superspec.decoding.sampling import to_probs  # noqa: E402
from test_superword import build  # noqa: E402


def tiny_llama(vocab_size, seed, sharpness=40.0):
    """A 2-layer Llama. Scaling the output head makes its distributions peaked,
    so drafts and target disagree often and verification is really exercised."""
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = transformers.LlamaForCausalLM(config).eval()
    model.lm_head.weight.data *= sharpness
    return model


@pytest.fixture(scope="module")
def setup():
    base, superbpe = build()
    target = tiny_llama(base.vocab_size, seed=0)
    std_draft = tiny_llama(base.vocab_size, seed=1)
    super_draft = tiny_llama(superbpe.vocab_size, seed=2)
    prompt = base.encode("One of the best parts of")
    return base, superbpe, target, std_draft, super_draft, prompt


def decoders(setup, sampling, **kwargs):
    base, superbpe, target, std_draft, super_draft, _ = setup
    t = CachedLM(target)
    common = dict(sampling=sampling, stop_ids=set(), max_draft_bytes=16, **kwargs)
    return {
        "standard": SpeculativeDecoder(t, base, CachedLM(std_draft), base, "standard", **common),
        "superword": SpeculativeDecoder(t, base, CachedLM(super_draft), superbpe, "superword",
                                        **common),
        "superword_pointmass": SpeculativeDecoder(t, base, CachedLM(super_draft), superbpe,
                                                  "superword_pointmass", **common),
        "exact_match": SpeculativeDecoder(t, base, CachedLM(super_draft), superbpe,
                                          "exact_match", **common),
        "exact_match_no_retok": SpeculativeDecoder(t, base, CachedLM(super_draft), superbpe,
                                                   "exact_match", retokenize=False, **common),
    }


def test_cache_matches_full_forward(setup):
    base, _, target, *_ = setup
    lm = CachedLM(target)
    rng = np.random.default_rng(0)
    seq = list(rng.integers(0, base.vocab_size, 20))
    # Grow, roll back to a shorter prefix, branch off, and ask for several positions.
    for tokens, n in [(seq[:8], 1), (seq[:15], 3), (seq[:10], 2), (seq[:10] + [3, 4, 5], 4),
                      (seq[:2], 1), (seq, 5)]:
        tokens = [int(t) for t in tokens]
        cached = lm.logits(tokens, n)
        with torch.no_grad():
            full = target(torch.tensor([tokens])).logits[0, -n:].float()
        torch.testing.assert_close(cached, full, atol=1e-4, rtol=1e-4)
        assert lm.tokens == tokens


@pytest.mark.parametrize("rows", ["length_aware", "point_mass"])
def test_superword_proposal_rows(setup, rows):
    _, superbpe, _, _, super_draft, prompt = setup
    drafter = Drafter(CachedLM(super_draft), superbpe, superbpe.target_vocab_size, rows=rows)
    gen = torch.Generator().manual_seed(0)
    seen_superword = False
    for _ in range(20):
        p = drafter.propose(superbpe.from_target_ids(prompt), SamplingConfig(1.0), gen,
                            max_bytes=40, max_steps=8, max_target_tokens=30, stop_ids=set())
        assert superbpe.to_target_ids(p.draft_tokens) == p.target_tokens
        for i, (row, src) in enumerate(zip(p.q_rows, p.source)):
            first_of_draft_token = i == 0 or p.source[i - 1] != src
            if rows == "point_mass":
                assert (row is not None) == first_of_draft_token
            else:
                assert row is not None
            if row is not None:
                assert abs(float(row.sum()) - 1) < 1e-5 and row[p.target_tokens[i]] > 0
        seen_superword |= any(superbpe.is_superword(w) for w in p.draft_tokens)
    assert seen_superword


def test_length_aware_row_matches_definition(setup):
    """q_j(x) ∝ sum of q(v) over v with len(v) > j, v[:j] == w[:j], v[j] == x."""
    _, superbpe, _, _, super_draft, _ = setup
    drafter = Drafter(CachedLM(super_draft), superbpe, superbpe.target_vocab_size)
    q = torch.rand(superbpe.vocab_size, generator=torch.Generator().manual_seed(0))
    q = q / q.sum()
    checked = 0
    for w in range(superbpe.target_vocab_size, superbpe.vocab_size):
        e = superbpe.expand(w)
        for j in range(1, len(e)):
            expected = torch.zeros(superbpe.target_vocab_size)
            for v in range(superbpe.vocab_size):
                ev = superbpe.expand(v)
                if len(ev) > j and ev[:j] == e[:j]:
                    expected[ev[j]] += q[v]
            expected /= expected.sum()
            torch.testing.assert_close(drafter.later_token_row(q, e, j), expected)
            checked += 1
    assert checked > 0


def test_greedy_matches_autoregressive(setup):
    base, _, target, *_, prompt = setup
    sampling = SamplingConfig(temperature=0.0)
    expected = autoregressive_generate(CachedLM(target), base, prompt, 30, sampling, set()).tokens
    for name, decoder in decoders(setup, sampling).items():
        result = decoder.generate(prompt, 30)
        assert result.tokens == expected, name


def exact_target_joint(target, prompt, sampling):
    """Exact distribution of the first two generated tokens."""
    with torch.no_grad():
        p1 = to_probs(target(torch.tensor([prompt])).logits[0, -1], sampling)
        joint = {}
        for y1 in range(len(p1)):
            if p1[y1] < 1e-9:
                continue
            p2 = to_probs(target(torch.tensor([prompt + [y1]])).logits[0, -1], sampling)
            for y2 in range(len(p2)):
                joint[(y1, y2)] = float(p1[y1] * p2[y2])
    return joint


def total_variation(counts, joint, n):
    keys = set(counts) | set(joint)
    return 0.5 * sum(abs(counts.get(k, 0) / n - joint.get(k, 0.0)) for k in keys)


@pytest.mark.parametrize("method", ["standard", "superword", "superword_pointmass",
                                    "exact_match", "exact_match_no_retok"])
def test_sampling_is_lossless(setup, method):
    """Sampled outputs must follow the target's own distribution. We compare the
    first two tokens (generating six, so multi-token superwords get verified)
    against the exact target distribution, with a threshold set by how much
    sampling noise alone gives for the same number of samples."""
    _, _, target, *_, prompt = setup
    sampling = SamplingConfig(temperature=0.8, top_p=0.95)
    joint = exact_target_joint(target, prompt, sampling)
    decoder = decoders(setup, sampling)[method]

    n = 3000
    gen = torch.Generator().manual_seed(123)
    counts = Counter(tuple(decoder.generate(prompt, 6, gen).tokens[:2]) for _ in range(n))
    tv = total_variation(counts, joint, n)

    # Noise level: TV between the exact distribution and n exact samples from it.
    rng = np.random.default_rng(0)
    keys, probs = list(joint), np.array(list(joint.values()))
    noise = [total_variation(dict(zip(keys, rng.multinomial(n, probs / probs.sum()))), joint, n)
             for _ in range(200)]
    threshold = 1.3 * np.quantile(noise, 0.995)
    assert tv < threshold, f"{method}: TV {tv:.4f} vs noise threshold {threshold:.4f}"


def test_cost_ratio_measurement(setup):
    from superspec.eval.common import Draft
    from superspec.eval.cost import context_from_texts, measure_cost_ratios

    base, superbpe, target, std_draft, super_draft, _ = setup
    drafts = {"std": Draft("std", CachedLM(std_draft), base),
              "super": Draft("super", CachedLM(super_draft), superbpe)}
    context = context_from_texts(["One of the best parts of the day is the morning."], base, 40)
    assert len(context) == 40
    rows = measure_cost_ratios(CachedLM(target), base, drafts, context, verify_tokens=4,
                               reps=3, warmup=1)
    assert [r["draft"] for r in rows] == ["std", "super"]
    for r in rows:
        assert r["c"] > 0 and r["draft_step_ms"] > 0
    # SuperBPE packs the same text into fewer draft tokens.
    assert rows[1]["context_draft_tokens"] < rows[0]["context_draft_tokens"]


def test_round_start_statistics(setup):
    """Every proposing round is counted once as clean or mid-superword; a standard draft
    has no superwords, so all of its rounds are clean."""
    *_, prompt = setup
    sampling = SamplingConfig(temperature=0.8)
    for name, decoder in decoders(setup, sampling).items():
        r = decoder.generate(prompt, 40, torch.Generator().manual_seed(0))
        assert r.rounds_clean + r.rounds_mid <= r.rounds, name
        assert r.first_accepted_clean <= r.rounds_clean and r.first_accepted_mid <= r.rounds_mid
        assert r.accepted_tokens_clean + r.accepted_tokens_mid == r.accepted_tokens, name
        if name == "standard":
            assert r.rounds_mid == 0 and r.rounds_clean > 0
        assert "rounds_mid" in r.summary()


@pytest.mark.parametrize("method", ["standard", "superword", "superword_pointmass"])
def test_stop_below_and_draft_temperature_are_lossless(setup, method):
    """Ending proposals at unsure tokens and sampling the draft at its own temperature
    must not change the output distribution."""
    _, _, target, *_, prompt = setup
    sampling = SamplingConfig(temperature=0.8, top_p=0.95)
    joint = exact_target_joint(target, prompt, sampling)
    decoder = decoders(setup, sampling, draft_stop_below=0.3, draft_temperature_scale=0.7)[method]

    n = 3000
    gen = torch.Generator().manual_seed(321)
    counts = Counter(tuple(decoder.generate(prompt, 6, gen).tokens[:2]) for _ in range(n))
    tv = total_variation(counts, joint, n)
    rng = np.random.default_rng(1)
    keys, probs = list(joint), np.array(list(joint.values()))
    noise = [total_variation(dict(zip(keys, rng.multinomial(n, probs / probs.sum()))), joint, n)
             for _ in range(200)]
    assert tv < 1.3 * np.quantile(noise, 0.995), f"{method}: TV {tv:.4f}"


def test_stop_below_shortens_proposals_and_keeps_greedy_output(setup):
    base, _, target, *_, prompt = setup
    sampling = SamplingConfig(temperature=0.0)
    expected = autoregressive_generate(CachedLM(target), base, prompt, 30, sampling, set()).tokens
    plain = decoders(setup, sampling)
    stopping = decoders(setup, sampling, draft_stop_below=0.9)
    for name in ("standard", "superword"):
        a, b = plain[name].generate(prompt, 30), stopping[name].generate(prompt, 30)
        assert b.tokens == expected, name
        assert b.draft_steps / b.rounds < a.draft_steps / a.rounds, name    # shorter proposals
