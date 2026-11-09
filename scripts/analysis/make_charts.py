"""Make the report charts from speed benchmark results (scripts/eval/speed.py).

    python scripts/analysis/make_charts.py                      # everything under $RESULTS_DIR
    python scripts/analysis/make_charts.py results/llama3 --draft-len 32
    python scripts/analysis/make_charts.py results/ --methods all --formats png

Looks for every results.jsonl under the given folders; runs of the same
target and benchmark (e.g. split by temperature) are merged. For each target
it writes to --out-dir:

    <target>_<benchmark>_speedup_vs_draft_len.{png,pdf}
    <target>_speedup_by_category_<L>B.{png,pdf}           (all benchmarks' categories)
    <target>_<benchmark>_acceptance_by_temperature_<L>B.{png,pdf}
    chart_data.csv                                        (every number plotted)

By default each draft is shown with its main method (standard drafts:
"standard", SuperBPE drafts: "superword"); --methods all adds the ablations
(superword_pointmass, exact_match) and makes one set of charts per draft size
(file names end in _50M, _150M, ...) so each chart stays readable.
"""

import argparse
import csv
import logging
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from superspec.analysis import charts
from superspec.analysis.results import load_table

logger = logging.getLogger("make_charts")

COLUMNS = ["target", "benchmark", "category", "temperature", "label", "kind", "size", "method",
           "draft", "draft_len_bytes", "prompts", "speedup", "bytes_per_second",
           "token_acceptance", "byte_acceptance", "accepted_bytes_per_round",
           "target_tokens_per_draft_step", "draft_steps_per_round"]


def default_draft_len(rows: list[dict]) -> int:
    """32 bytes if it was run (the middle of the default sweep), else the most common."""
    lengths = Counter(r["draft_len_bytes"] for r in rows if r["kind"])
    return 32 if 32 in lengths else lengths.most_common(1)[0][0]


def save(fig, out_dir: Path, name: str, formats: list[str]) -> None:
    if fig is None:
        logger.warning("No data for %s, skipped", name)
        return
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", dpi=200)
    plt.close(fig)
    logger.info("Wrote %s.{%s}", out_dir / name, ",".join(formats))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="*", help="results folders or results.jsonl files "
                        "(default: $RESULTS_DIR or ./results)")
    parser.add_argument("--out-dir", default=None, help="default: <first results folder>/charts")
    parser.add_argument("--draft-len", type=int, default=None,
                        help="draft length (bytes) for the per-category and acceptance charts "
                             "(default: 32 if present)")
    parser.add_argument("--methods", choices=["primary", "all"], default="primary")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    paths = args.results or [os.environ.get("RESULTS_DIR", "./results")]
    table = load_table(paths)
    if args.methods == "primary":
        table = [r for r in table if r["primary"]]
    out_dir = Path(args.out_dir or Path(paths[0]) / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "chart_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r for r in table if r["kind"])

    charts.apply_style()
    for target in sorted({r["target"] for r in table}):
        target_rows = [r for r in table if r["target"] == target]
        draft_len = args.draft_len or default_draft_len(target_rows)
        # All methods for every size would be too many series: split by draft size.
        sizes = sorted({r["size"] for r in target_rows if r["kind"]})
        groups = ([("", target_rows)] if args.methods == "primary" else
                  [(f"_{size}", [r for r in target_rows if r["size"] == size]) for size in sizes])
        for suffix, rows in groups:
            for benchmark in sorted({r["benchmark"] for r in rows}):
                bench_rows = [r for r in rows if r["benchmark"] == benchmark]
                save(charts.speedup_vs_draft_len(bench_rows, target, benchmark), out_dir,
                     f"{target}_{benchmark}_speedup_vs_draft_len{suffix}", args.formats)
                save(charts.acceptance_by_temperature(bench_rows, target, benchmark, draft_len),
                     out_dir,
                     f"{target}_{benchmark}_acceptance_by_temperature_{draft_len}B{suffix}",
                     args.formats)
            save(charts.speedup_by_category(rows, target, draft_len), out_dir,
                 f"{target}_speedup_by_category_{draft_len}B{suffix}", args.formats)
    print(f"Charts and chart_data.csv in {out_dir}")


if __name__ == "__main__":
    main()
