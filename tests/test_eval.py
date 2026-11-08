"""Tests for the evaluation statistics and summary (no models needed).

Run with `pytest tests/` or, without pytest, `python tests/test_eval.py`.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superspec.eval.stats import fisher_combined_p, permutation_test  # noqa: E402
from superspec.eval.summary import format_table, summarize  # noqa: E402


def test_permutation_test_detects_difference():
    rng = np.random.default_rng(0)
    same_a = list(rng.choice(5, 300, p=[0.4, 0.3, 0.1, 0.1, 0.1]))
    same_b = list(rng.choice(5, 300, p=[0.4, 0.3, 0.1, 0.1, 0.1]))
    diff_b = list(rng.choice(5, 300, p=[0.2, 0.3, 0.3, 0.1, 0.1]))
    _, p_same = permutation_test(same_a, same_b, 500, rng)
    _, p_diff = permutation_test(same_a, diff_b, 500, rng)
    assert p_same > 0.01 and p_diff < 0.01


def test_permutation_p_values_are_uniform_under_null():
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(100):
        a = list(rng.choice(4, 100))
        b = list(rng.choice(4, 100))
        ps.append(permutation_test(a, b, 200, rng)[1])
    assert 0.35 < np.mean(ps) < 0.65
    assert np.mean(np.array(ps) < 0.05) < 0.15


def test_fisher_combined_p():
    assert abs(fisher_combined_p([0.3]) - 0.3) < 1e-12
    # two p-values of 0.5: x = -2 ln(0.25), chi2 with 4 dof: exp(-x/2) * (1 + x/2)
    half_x = -math.log(0.25)
    assert abs(fisher_combined_p([0.5, 0.5]) - math.exp(-half_x) * (1 + half_x)) < 1e-12
    assert fisher_combined_p([1e-4] * 5) < 1e-10


def row(run, method, category, num_bytes, wall_time, draft=None, n_bytes=None, **extra):
    base = dict(run=run, temperature=0.0, method=method, draft=draft, draft_len_bytes=n_bytes,
                prompt_id=str(len(extra)), category=category, num_bytes=num_bytes,
                wall_time=wall_time, rounds=0, draft_steps=0, proposed_tokens=0,
                accepted_tokens=0, proposed_bytes=0, accepted_bytes=0, draft_time=0,
                target_time=0, overhead_time=0, num_tokens=0)
    base.update(extra)
    return base


def test_summary_speedup_and_acceptance():
    rows = [
        row("ar", "autoregressive", "qa", 1000, 10.0),
        row("ar", "autoregressive", "math", 1000, 10.0),
        row("sw", "superword", "qa", 1000, 5.0, "d", 32, rounds=10, proposed_tokens=40,
            accepted_tokens=30, proposed_bytes=200, accepted_bytes=150),
        row("sw", "superword", "math", 1000, 2.5, "d", 32, rounds=10, proposed_tokens=40,
            accepted_tokens=10, proposed_bytes=200, accepted_bytes=50),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "results.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = summarize(path)
        assert (Path(d) / "summary.csv").exists()

    by = {(s["method"], s["category"]): s for s in summary}
    assert by[("superword", "qa")]["speedup"] == 2.0
    assert by[("superword", "math")]["speedup"] == 4.0
    # "all" uses total bytes / total time: 2000 B / 7.5 s vs 2000 B / 20 s
    assert abs(by[("superword", "all")]["speedup"] - 20 / 7.5) < 1e-3
    assert by[("superword", "all")]["token_acceptance"] == 0.5
    assert by[("superword", "all")]["byte_acceptance"] == 0.5
    assert by[("superword", "all")]["accepted_bytes_per_round"] == 10.0
    assert "superword" in format_table(summary)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
