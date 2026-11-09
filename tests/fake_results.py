"""Fake but plausible speed-benchmark results, in the layout scripts/eval/speed.py
writes, for testing the analysis code without running any models.

    python tests/fake_results.py /tmp/fake_results     # then point make_charts.py at it
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

BENCHMARKS = {
    "specbench": ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag"],
    "humaneval": ["humaneval"],
    "mbpp": ["mbpp"],
}
DRAFTS = [("std_50m", "standard", 1.0), ("superbpe_50m", "superbpe", 1.0),
          ("std_150m", "standard", 1.12), ("superbpe_150m", "superbpe", 1.12)]
METHODS = {"standard": ["standard", "exact_match"],
           "superbpe": ["superword", "superword_pointmass", "exact_match"]}


def write_fake_run(root: Path, target: str = "llama3", seed: int = 0,
                   temperatures=(0.0, 0.7, 1.0), lengths=(16, 32, 64), prompts_per_cat=3) -> None:
    rng = random.Random(seed)
    for bench, categories in BENCHMARKS.items():
        run_dir = root / target / bench / "20261109-120000"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {"name": bench, "target": {"name": target},
                  "drafts": [{"name": d, "tokenizer": f"tokenizer/{target}"
                              + ("_superbpe" if k == "superbpe" else "") + ".yaml"}
                             for d, k, _ in DRAFTS]}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config))

        rows = []
        for temp in temperatures:
            specs = [("autoregressive", None, None, None, 1.0)]
            for draft, kind, quality in DRAFTS:
                for method in METHODS[kind]:
                    for n in lengths:
                        specs.append((method, draft, kind, n, quality))
            for method, draft, kind, n, quality in specs:
                for cat in categories:
                    code = bench != "specbench"
                    for p in range(prompts_per_cat):
                        rows.append(_row(rng, temp, method, draft, kind, n, quality, cat,
                                         f"{cat}-{p}", code))
        with open(run_dir / "results.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


def _row(rng, temp, method, draft, kind, n, quality, cat, prompt_id, code):
    num_bytes = rng.randint(2500, 3500)
    base_speed = 110.0                                   # bytes/s autoregressive
    if method == "autoregressive":
        wall = num_bytes / (base_speed * rng.uniform(0.97, 1.03))
        extra = dict(rounds=0, draft_steps=0, proposed_tokens=0, accepted_tokens=0,
                     proposed_bytes=0, accepted_bytes=0)
    else:
        accept = (0.78 - 0.18 * temp) * quality - (0.06 if code else 0.0)
        if kind == "superbpe":
            accept -= 0.03 if method == "superword" else 0.07
        if method == "exact_match":
            accept -= 0.06 * temp
        accept = max(0.2, min(0.95, accept - 0.0015 * n + rng.uniform(-0.02, 0.02)))
        bytes_per_token = 4.3
        tokens_per_step = 1.0 if kind == "standard" else (1.25 if code else 1.55)
        proposed_tokens = n / bytes_per_token
        draft_steps = proposed_tokens / tokens_per_step
        accepted_bytes_round = n * accept * 0.9 + 4
        rounds = max(1, int(num_bytes / accepted_bytes_round))
        # time per round: one target pass + draft steps at cost ratio c
        c = 0.05 * quality
        wall = rounds * (4.3 / base_speed) * (1 + c * draft_steps + 0.03)
        extra = dict(rounds=rounds, draft_steps=int(rounds * draft_steps),
                     proposed_tokens=int(rounds * proposed_tokens),
                     accepted_tokens=int(rounds * proposed_tokens * (accept - 0.02)),
                     proposed_bytes=rounds * n, accepted_bytes=int(rounds * n * accept))
    run = f"T={temp}|{method}|{draft}|{n}"
    return dict(run=run, temperature=temp, method=method, draft=draft, draft_len_bytes=n,
                prompt_id=prompt_id, category=cat, num_bytes=num_bytes, wall_time=wall,
                draft_time=0.0, target_time=0.0, overhead_time=0.0, num_tokens=num_bytes // 4,
                phase_breakdown=False, **extra)


if __name__ == "__main__":
    write_fake_run(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_results"))
