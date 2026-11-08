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


def test_superword_proposal_rows(setup):
    _, superbpe, _, _, super_draft, prompt = setup
    drafter = Drafter(CachedLM(super_draft), superbpe, superbpe.target_vocab_size)
    gen = torch.Generator().manual_seed(0)
    seen_superword = False
    for _ in range(20):
        p = drafter.propose(superbpe.from_target_ids(prompt), SamplingConfig(1.0), gen,
                            max_bytes=40, max_steps=8, max_target_tokens=30, stop_ids=set())
        assert superbpe.to_target_ids(p.draft_tokens) == p.target_tokens
        for i, (row, src) in enumerate(zip(p.q_rows, p.source)):
            first_of_draft_token = i == 0 or p.source[i - 1] != src
            assert (row is not None) == first_of_draft_token
            if row is not None:
                assert abs(float(row.sum()) - 1) < 1e-5 and row[p.target_tokens[i]] > 0
        seen_superword |= any(superbpe.is_superword(w) for w in p.draft_tokens)
    assert seen_superword


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


@pytest.mark.parametrize("method", ["standard", "superword", "exact_match",
                                    "exact_match_no_retok"])
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
