"""Benchmark decoding speed and acceptance (Spec-Bench, HumanEval, MBPP).

    python scripts/eval/speed.py --config eval/specbench.yaml
    python scripts/eval/speed.py --config eval/humaneval.yaml --drafts std_50m superbpe_50m
    python scripts/eval/speed.py --config eval/specbench.yaml target_profile=targets/olmo2.yaml

    # a subset of the sweep, e.g. to split work across GPUs:
    python scripts/eval/speed.py --config eval/specbench.yaml "speculative.temperatures=[0.0]"

    # resume an interrupted run / re-summarize:
    python scripts/eval/speed.py --config eval/specbench.yaml --output-dir <run folder>
    python scripts/eval/speed.py --summarize <run folder>/results.jsonl

Writes results.jsonl (one row per run and prompt) and summary.csv.
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from superspec.config import load_config
from superspec.eval.summary import format_table, summarize

logger = logging.getLogger("eval_speed")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="eval/specbench.yaml")
    parser.add_argument("--drafts", nargs="*", help="only these draft names (default: all)")
    parser.add_argument("--output-dir", default=None, help="default: <output_dir>/<name>/<time>")
    parser.add_argument("--summarize", default=None, help="only summarize this results.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. seed=1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.summarize:
        print(format_table(summarize(Path(args.summarize))))
        return

    # Imported here so --summarize works without the model dependencies.
    from superspec.eval.common import load_drafts, load_target
    from superspec.eval.prompts import load_prompts
    from superspec.eval.speed import run_benchmark

    cfg = load_config(args.config, args.overrides)
    output_dir = Path(args.output_dir or Path(cfg.output_dir) / cfg.name
                      / time.strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    prompts = load_prompts(cfg.data, cfg.name)
    target = load_target(cfg, args.device)
    drafts = load_drafts(cfg, args.device, args.drafts)
    logger.info("%d prompts, drafts: %s", len(prompts), ", ".join(drafts))

    results = run_benchmark(cfg, target, drafts, prompts, output_dir, args.device)
    print(format_table(summarize(results)))
    logger.info("Results in %s", output_dir)


if __name__ == "__main__":
    main()
