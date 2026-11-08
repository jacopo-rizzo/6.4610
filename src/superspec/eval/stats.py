"""Statistics for the losslessness check (numpy only).

For one prompt we have outcomes (e.g. the first few output bytes) sampled with
the target alone and with speculative decoding. If speculative decoding is
lossless, both are samples from the same distribution. We test that with a
permutation test on the total variation distance between the two empirical
distributions, then combine the per-prompt p-values with Fisher's method.
"""

from __future__ import annotations

import math

import numpy as np


def total_variation(a: np.ndarray, b: np.ndarray, num_categories: int) -> float:
    pa = np.bincount(a, minlength=num_categories) / len(a)
    pb = np.bincount(b, minlength=num_categories) / len(b)
    return 0.5 * float(np.abs(pa - pb).sum())


def permutation_test(a: list, b: list, num_permutations: int = 2000,
                     rng: np.random.Generator | None = None) -> tuple[float, float]:
    """(observed TV distance, p-value) for "a and b come from the same distribution".

    The p-value is the fraction of random relabelings of the pooled samples with
    a TV distance at least as large as observed (with the usual +1 correction).
    """
    rng = rng or np.random.default_rng(0)
    categories = {x: i for i, x in enumerate(dict.fromkeys(list(a) + list(b)))}
    pooled = np.array([categories[x] for x in list(a) + list(b)])
    n_a, k = len(a), len(categories)

    observed = total_variation(pooled[:n_a], pooled[n_a:], k)
    at_least = 0
    for _ in range(num_permutations):
        perm = rng.permutation(pooled)
        if total_variation(perm[:n_a], perm[n_a:], k) >= observed - 1e-12:
            at_least += 1
    return observed, (at_least + 1) / (num_permutations + 1)


def fisher_combined_p(p_values: list[float]) -> float:
    """Fisher's method: -2 sum(log p) ~ chi-squared with 2k degrees of freedom.

    For even degrees of freedom the chi-squared survival function has a closed
    form, so no scipy is needed: sf(x; 2k) = exp(-x/2) * sum_{i<k} (x/2)^i / i!.
    """
    k = len(p_values)
    half_x = -sum(math.log(p) for p in p_values)    # x / 2
    term, total = 1.0, 1.0
    for i in range(1, k):
        term *= half_x / i
        total += term
    return min(1.0, math.exp(-half_x) * total)
