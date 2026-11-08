"""Measure the draft-to-target cost ratio c for every draft in an eval config.

    python scripts/eval/cost_ratio.py --config eval/specbench.yaml
    python scripts/eval/cost_ratio.py --config eval/specbench.yaml target_profile=targets/olmo2.yaml

c = (one draft forward step) / (one target forward step), batch size 1, warm
KV cache, on a context of real benchmark text. Run it on the same GPU as the
speed benchmark. See superspec/eval/cost.py.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from superspec.config import load_config
from superspec.eval.common import load_drafts, load_target
from superspec.eval.cost import context_from_texts, measure_cost_ratios
from superspec.eval.prompts import load_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="eval/specbench.yaml")
    parser.add_argument("--drafts", nargs="*", help="only these draft names (default: all)")
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--verify-tokens", type=int, default=8,
                        help="tokens in the timed target verification pass (K + 1)")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    texts = [p.text for p in load_prompts(cfg.data, cfg.name)]
    target = load_target(cfg, args.device)
    drafts = load_drafts(cfg, args.device, args.drafts)
    context = context_from_texts(texts, target.vocab, args.context_tokens)

    rows = measure_cost_ratios(target.lm, target.vocab, drafts, context,
                               verify_tokens=args.verify_tokens, reps=args.reps)
    out = Path(cfg.output_dir) / cfg.name / "cost_ratio.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    verify_key = f"target_verify_{args.verify_tokens}_ms"
    print(f"{'draft':<20}{'draft ms':>10}{'target ms':>11}{'verify ms':>11}{'c':>8}")
    for r in rows:
        print(f"{r['draft']:<20}{r['draft_step_ms']:>10.2f}{r['target_step_ms']:>11.2f}"
              f"{r[verify_key]:>11.2f}{r['c']:>8.3f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
