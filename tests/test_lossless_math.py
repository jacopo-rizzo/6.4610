"""Exact check that the verification rules are lossless (numpy only, no models).

We take a tiny target vocabulary, random target conditionals p(. | prefix), a
draft vocabulary containing superwords, and random draft conditionals
q(. | draft prefix). We then enumerate every possible draft proposal and every
accept/reject outcome of one speculative round, finish the sequence with the
target alone, and compare the exact distribution of the first N tokens with
the target's own. They must match to floating-point precision.

This checks the math of the rules in superspec/decoding/verification.py, not
the torch code itself (tests/test_decoding.py covers that with real models).

Rules checked, per proposed target token x with target distribution p:
  * standard / superword, first token of a draft token:
        q(x) = sum of q(w) over draft tokens w whose first target token is x;
        accept with prob min(1, p(x)/q(x)), else sample from max(p - q, 0).
  * superword, later tokens of the same superword: q is a point mass on x;
        accept with prob p(x), else sample from p with x removed.
  * string-level exact match (Timor et al.): sample y ~ p, accept iff y == x.
"""

import itertools
from collections import defaultdict

import numpy as np

VT = 3                                        # target tokens 0, 1, 2
EXPANSION = [(0,), (1,), (2,), (0, 1), (2, 0), (1, 1, 2)]   # draft ids 3..5 are superwords
N = 4                                         # compare distributions over 4 tokens
DRAFT_STEPS = 2


def random_tables(seed):
    rng = np.random.default_rng(seed)
    target, draft = {}, {}

    def p(prefix):
        if prefix not in target:
            target[prefix] = rng.dirichlet(np.ones(VT) * 0.7)
        return target[prefix]

    def q(prefix, vd):
        if prefix not in draft:
            draft[prefix] = rng.dirichlet(np.ones(vd) * 0.7)
        return draft[prefix]

    return p, q


def target_joint(p, n):
    return {y: np.prod([p(y[:i])[y[i]] for i in range(n)])
            for y in itertools.product(range(VT), repeat=n)}


def first_token_marginal(q_row, expansion):
    out = np.zeros(VT)
    for w, prob in enumerate(q_row):
        out[expansion[w][0]] += prob
    return out


def round_distribution(p, q, expansion, rule):
    """Exact distribution of the tokens one speculative round outputs."""
    vd = len(expansion)
    outputs = defaultdict(float)

    def verify(xs, rows, weight):
        prefix = ()
        for x, row in zip(xs, rows):
            pt = p(prefix)
            if rule == "exact_match":
                for y in range(VT):
                    if y != x:
                        outputs[prefix + (y,)] += weight * pt[y]
                weight *= pt[x]
            else:
                if row is None:                 # point mass on x
                    accept = pt[x]
                    residual = pt.copy()
                    residual[x] = 0
                else:
                    accept = min(1.0, pt[x] / row[x])
                    residual = np.maximum(pt - row, 0)
                if accept < 1:
                    residual = residual / residual.sum()
                    for y in range(VT):
                        outputs[prefix + (y,)] += weight * (1 - accept) * residual[y]
                weight *= accept
            prefix += (x,)
        for y in range(VT):                     # everything accepted: bonus token
            outputs[prefix + (y,)] += weight * p(prefix)[y]

    def draft(dprefix, weight, xs, rows):
        if len(dprefix) == DRAFT_STEPS:
            verify(xs, rows, weight)
            return
        q_row = q(dprefix, vd)
        for w in range(vd):
            if q_row[w] == 0:
                continue
            exp = expansion[w]
            if rule == "wrong_marginal":        # negative control: ignore superwords' mass
                first = np.array([q_row[t] for t in range(VT)])
            else:
                first = first_token_marginal(q_row, expansion)
            draft(dprefix + (w,), weight * q_row[w], xs + list(exp),
                  rows + [first] + [None] * (len(exp) - 1))

    draft((), 1.0, [], [])
    return outputs


def output_joint(p, q, expansion, rule, n):
    """Distribution of the first n tokens: one speculative round, then the target."""
    joint = defaultdict(float)
    for out, prob in round_distribution(p, q, expansion, rule).items():
        if len(out) >= n:
            joint[out[:n]] += prob
            continue
        for rest in itertools.product(range(VT), repeat=n - len(out)):
            y = out + rest
            joint[y] += prob * np.prod([p(y[:i])[y[i]] for i in range(len(out), n)])
    return joint


def max_error(rule, expansion, seed):
    p, q = random_tables(seed)
    ours = output_joint(p, q, expansion, rule, N)
    ref = target_joint(p, N)
    assert abs(sum(ours.values()) - 1) < 1e-9
    return max(abs(ours[y] - ref[y]) for y in ref)


def test_standard_is_lossless():
    for seed in range(5):
        assert max_error("rejection", EXPANSION[:VT], seed) < 1e-12


def test_superword_is_lossless():
    for seed in range(5):
        assert max_error("rejection", EXPANSION, seed) < 1e-12


def test_exact_match_is_lossless():
    for seed in range(5):
        assert max_error("exact_match", EXPANSION, seed) < 1e-12


def test_wrong_marginal_is_not_lossless():
    # Sanity check that this test can fail: using q(t) of the base token alone,
    # instead of summing over superwords that start with t, is biased.
    assert max(max_error("wrong_marginal", EXPANSION, seed) for seed in range(5)) > 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
