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
  * superword, token j >= 1 of superword w, two rules:
      "length_aware": q_j(x) ∝ sum of q(v) over v with len(v) > j,
                      v[:j] == w[:j], v[j] == x; then as above;
      "point_mass":   q is a point mass on x; accept with prob p(x), else
                      sample from p with x removed.
  * string-level exact match (Timor et al.): sample y ~ p, accept iff y == x.
We also check that the length-aware rule accepts at least as many tokens per
round as the point-mass rule.
"""

import itertools
from collections import defaultdict

import numpy as np

VT = 3                                        # target tokens 0, 1, 2
EXPANSION = [(0,), (1,), (2,), (0, 1), (2, 0), (1, 1, 2)]   # draft ids 3..5 are superwords
# Superwords that share prefixes ((0, 1) / (0, 2) / (0, 1, 1); (1, 1, 2) / (1, 1, 0)), so
# the length-aware rows differ from point masses. In EXPANSION every prefix has a
# single continuation, and the two rules coincide.
EXPANSION_SHARED = EXPANSION + [(0, 2), (0, 1, 1), (1, 1, 0)]
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


def later_token_row(q_row, expansion, w, j):
    """Length-aware row for target token j >= 1 of draft token w."""
    out = np.zeros(VT)
    for v, prob in enumerate(q_row):
        e = expansion[v]
        if len(e) > j and e[:j] == expansion[w][:j]:
            out[e[j]] += prob
    return out / out.sum()


def round_distribution(p, q, expansion, rule):
    """Exact distribution of the tokens one speculative round outputs, and the
    expected number of accepted draft-proposed tokens."""
    vd = len(expansion)
    outputs = defaultdict(float)
    accepted = [0.0]

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
            accepted[0] += weight
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
            if rule == "length_aware":
                later = [later_token_row(q_row, expansion, w, j) for j in range(1, len(exp))]
            else:
                later = [None] * (len(exp) - 1)
            draft(dprefix + (w,), weight * q_row[w], xs + list(exp), rows + [first] + later)

    draft((), 1.0, [], [])
    return outputs, accepted[0]


def output_joint(p, q, expansion, rule, n):
    """Distribution of the first n tokens: one speculative round, then the target."""
    joint = defaultdict(float)
    for out, prob in round_distribution(p, q, expansion, rule)[0].items():
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
        assert max_error("length_aware", EXPANSION[:VT], seed) < 1e-12


def test_superword_length_aware_is_lossless():
    for expansion in (EXPANSION, EXPANSION_SHARED):
        for seed in range(5):
            assert max_error("length_aware", expansion, seed) < 1e-12


def test_superword_point_mass_is_lossless():
    for expansion in (EXPANSION, EXPANSION_SHARED):
        for seed in range(5):
            assert max_error("point_mass", expansion, seed) < 1e-12


def test_length_aware_accepts_at_least_as_much():
    gains = []
    for seed in range(5):
        p, q = random_tables(seed)
        aware = round_distribution(p, q, EXPANSION_SHARED, "length_aware")[1]
        p, q = random_tables(seed)
        point = round_distribution(p, q, EXPANSION_SHARED, "point_mass")[1]
        assert aware >= point - 1e-12
        gains.append(aware - point)
    assert max(gains) > 1e-3     # and strictly more on some draws
    print("   accepted tokens/round gained by length-aware rows:",
          [round(float(g), 4) for g in gains])


def test_exact_match_is_lossless():
    for seed in range(5):
        assert max_error("exact_match", EXPANSION, seed) < 1e-12


def test_wrong_marginal_is_not_lossless():
    # Sanity check that this test can fail: using q(t) of the base token alone,
    # instead of summing over superwords that start with t, is biased.
    for expansion in (EXPANSION, EXPANSION_SHARED):
        assert max(max_error("wrong_marginal", expansion, seed) for seed in range(5)) > 1e-3


# ---------------------------------------------------------------------------------------
# Re-draw after a rejection (Drafter.redraw / SpeculativeDecoder(redraw=True)), exactly.
#
# Round 1 is as above. If token i of the proposal is rejected and replaced by the target's
# token y, let w be the draft token it belonged to, j its position inside w, q_row the
# probabilities w was drawn from, and prefix = w[:j] + (y,). A draft token v is drawn again
# from q_row restricted to tokens that equal prefix or start with it. If v is longer than
# prefix, round 2 proposes its remaining target tokens, verified against
#   length_aware: later_token_row(q_row, v, position)      point_mass: a point mass.
# Round 2 drafts nothing more here; after it the target finishes the sequence.

def verify_tail(p, prefix, xs, rows, weight, outputs):
    """Verify proposed tokens xs after `prefix`; add every outcome of the round to outputs."""
    for x, row in zip(xs, rows):
        pt = p(prefix)
        if row is None:
            accept, residual = pt[x], pt.copy()
            residual[x] = 0
        else:
            accept, residual = min(1.0, pt[x] / row[x]), np.maximum(pt - row, 0)
        if accept < 1:
            residual = residual / residual.sum()
            for y in range(VT):
                outputs[prefix + (y,)] += weight * (1 - accept) * residual[y]
        weight *= accept
        prefix += (x,)
    for y in range(VT):
        outputs[prefix + (y,)] += weight * p(prefix)[y]


def redraw_distribution(p, q, expansion, rule, broken=False):
    """Exact distribution of the tokens output by round 1 plus, after a rejection, the
    re-draw round."""
    vd = len(expansion)
    outputs = defaultdict(float)

    def after_rejection(prefix, y, weight, q_row, w, j):
        want = expansion[w][:j] + (y,)
        cands = [v for v in range(vd) if expansion[v][: len(want)] == want and q_row[v] > 0]
        longer = [v for v in cands if len(expansion[v]) > len(want)]
        if not longer:
            outputs[prefix + (y,)] += weight
            return
        total = sum(q_row[v] for v in cands)
        for v in cands:
            wv = weight * q_row[v] / total
            if len(expansion[v]) == len(want):               # drew the prefix itself
                outputs[prefix + (y,)] += wv
                continue
            rest = expansion[v][len(want):]
            if broken:                                       # negative control: wrong rows
                rows = [first_token_marginal(q_row, expansion) for _ in rest]
            elif rule == "length_aware":
                rows = [later_token_row(q_row, expansion, v, k)
                        for k in range(len(want), len(expansion[v]))]
            else:
                rows = [None] * len(rest)
            verify_tail(p, prefix + (y,), rest, rows, wv, outputs)

    def verify(xs, rows, meta, weight):
        prefix = ()
        for i, (x, row) in enumerate(zip(xs, rows)):
            pt = p(prefix)
            if row is None:
                accept, residual = pt[x], pt.copy()
                residual[x] = 0
            else:
                accept, residual = min(1.0, pt[x] / row[x]), np.maximum(pt - row, 0)
            if accept < 1:
                residual = residual / residual.sum()
                q_row, w, j = meta[i]
                for y in range(VT):
                    if residual[y] > 0:
                        after_rejection(prefix, y, weight * (1 - accept) * residual[y], q_row, w, j)
            weight *= accept
            prefix += (x,)
        for y in range(VT):                                  # everything accepted: no re-draw
            outputs[prefix + (y,)] += weight * p(prefix)[y]

    def draft(dprefix, weight, xs, rows, meta):
        if len(dprefix) == DRAFT_STEPS:
            verify(xs, rows, meta, weight)
            return
        q_row = q(dprefix, vd)
        for w in range(vd):
            if q_row[w] == 0:
                continue
            exp = expansion[w]
            first = first_token_marginal(q_row, expansion)
            if rule == "length_aware":
                later = [later_token_row(q_row, expansion, w, j) for j in range(1, len(exp))]
            else:
                later = [None] * (len(exp) - 1)
            draft(dprefix + (w,), weight * q_row[w], xs + list(exp), rows + [first] + later,
                  meta + [(q_row, w, j) for j in range(len(exp))])

    draft((), 1.0, [], [], [])
    return outputs


def redraw_error(rule, expansion, seed, broken=False):
    p, q = random_tables(seed)
    n = N + 2                                                # room for the second round
    joint = defaultdict(float)
    for out, prob in redraw_distribution(p, q, expansion, rule, broken).items():
        if len(out) >= n:
            joint[out[:n]] += prob
            continue
        for rest in itertools.product(range(VT), repeat=n - len(out)):
            y = out + rest
            joint[y] += prob * np.prod([p(y[:i])[y[i]] for i in range(len(out), n)])
    ref = target_joint(p, n)
    assert abs(sum(joint.values()) - 1) < 1e-9
    return max(abs(joint[y] - ref[y]) for y in ref)


def test_redraw_is_lossless():
    for rule in ("length_aware", "point_mass"):
        for seed in range(5):
            assert redraw_error(rule, EXPANSION_SHARED, seed) < 1e-12, (rule, seed)


def test_redraw_with_wrong_rows_is_not_lossless():
    # Sanity check that the re-draw test can fail.
    assert max(redraw_error("length_aware", EXPANSION_SHARED, seed, broken=True)
               for seed in range(5)) > 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
