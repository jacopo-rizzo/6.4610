"""Check empirically that speculative decoding samples from the target's distribution.

    python scripts/eval_lossless.py --config eval/specbench.yaml
    python scripts/eval_lossless.py --config eval/specbench.yaml lossless_check.num_prompts=5

Settings are under `lossless_check` in configs/eval/default.yaml. A lossless
method has Fisher p not small and ~1% of per-prompt p-values below 0.01,
like the autoregressive control.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from superspec.config import load_config
from superspec.eval.common import load_drafts, load_target
from superspec.eval.lossless import run_lossless_check
from superspec.eval.prompts import load_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="eval/specbench.yaml")
    parser.add_argument("--drafts", nargs="*")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    prompts = load_prompts(cfg.data, cfg.name)
    target = load_target(cfg, args.device)
    drafts = load_drafts(cfg, args.device, args.drafts)
    report = run_lossless_check(cfg, target, drafts, prompts, args.device)

    out = Path(cfg.output_dir) / cfg.name / "lossless.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"{'method':<36}{'fisher p':>10}{'frac p<0.01':>13}")
    for method, r in report.items():
        print(f"{method:<36}{r['fisher_p']:>10.4f}{r['frac_below_0.01']:>13.3f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
