"""Collecting benchmark results for analysis.

scripts/eval/speed.py writes one folder per run:

    $RESULTS_DIR/<target>/<benchmark>/<timestamp>/results.jsonl   one row per (run, prompt)
                                                  /config.yaml     the config it ran with

A benchmark may be split over several runs (e.g. one temperature per GPU), so
rows of the same (target, benchmark) are merged before aggregating; if the same
(run, prompt) appears twice, the later file wins. Aggregation is the same as
summary.csv (superspec.eval.summary.aggregate).

Each aggregated row is labeled with what the charts compare: the draft's
tokenizer `kind` ("standard" or "superbpe"), its `size` (e.g. "50M", from the
draft name), and whether its method is the `primary` one for that kind
(standard -> "standard", SuperBPE -> "superword") or an ablation.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from superspec.eval.summary import aggregate

PRIMARY_METHOD = {"standard": "standard", "superbpe": "superword"}
KIND_LABEL = {"standard": "Standard BPE", "superbpe": "SuperBPE"}


@dataclass
class Run:
    path: Path                           # results.jsonl
    target: str
    benchmark: str
    rows: list[dict]
    draft_kinds: dict[str, str] = field(default_factory=dict)   # draft name -> kind


def find_results(paths: list[str | Path]) -> list[Path]:
    """results.jsonl files: given directly, or anywhere under the given folders."""
    found = []
    for p in map(Path, paths):
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob("results.jsonl")))
        else:
            raise FileNotFoundError(p)
    return found


def load_run(results_path: Path) -> Run:
    with open(results_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    config_path = results_path.parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}

    # Fall back to the folder layout <target>/<benchmark>/<timestamp>/ if the config is missing.
    parts = results_path.parent.parts
    benchmark = config.get("name") or (parts[-2] if len(parts) >= 2 else "unknown")
    target = (config.get("target") or {}).get("name") or (parts[-3] if len(parts) >= 3 else "unknown")

    # A draft is SuperBPE if its tokenizer config is a SuperBPE one. The saved config may
    # hold the unresolved "${target.superbpe_tokenizer}", which also contains "superbpe".
    kinds = {d["name"]: "superbpe" if "superbpe" in str(d.get("tokenizer", "")) else "standard"
             for d in config.get("drafts", [])}
    return Run(results_path, str(target), str(benchmark), rows, kinds)


def draft_kind(draft: str, method: str, known: dict[str, str]) -> str:
    if draft in known:
        return known[draft]
    if method.startswith("superword") or "superbpe" in draft:
        return "superbpe"
    return "standard"


def draft_size(draft: str) -> str:
    """'superbpe_150m' -> '150M'; '' if the name has no size."""
    match = re.search(r"(\d+(?:\.\d+)?[mMbB])\b", draft)
    return match.group(1).upper() if match else ""


def series_label(kind: str, size: str, method: str, primary: bool) -> str:
    label = f"{KIND_LABEL[kind]} {size}".strip()
    return label if primary else f"{label} · {method}"


def load_table(paths: list[str | Path]) -> list[dict]:
    """Aggregated rows for every (target, benchmark) found under `paths`.

    Each row is a summary.csv row plus: target, benchmark, kind, size,
    primary, label. Autoregressive baseline rows have kind None.
    """
    runs = [load_run(p) for p in find_results(paths)]
    if not runs:
        raise FileNotFoundError(f"No results.jsonl found under {paths}")

    grouped: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        grouped[(run.target, run.benchmark)].append(run)

    table = []
    for (target, benchmark), group in sorted(grouped.items()):
        merged: dict[tuple[str, str], dict] = {}
        kinds: dict[str, str] = {}
        for run in sorted(group, key=lambda r: r.path.stat().st_mtime):
            kinds.update(run.draft_kinds)
            for row in run.rows:
                merged[(row["run"], row["prompt_id"])] = row
        for row in aggregate(list(merged.values())):
            row = dict(row, target=target, benchmark=benchmark)
            if row["method"] == "autoregressive":
                row.update(kind=None, size="", primary=True, label="Autoregressive")
            else:
                kind = draft_kind(row["draft"], row["method"], kinds)
                size = draft_size(row["draft"])
                primary = row["method"] == PRIMARY_METHOD[kind]
                row.update(kind=kind, size=size, primary=primary,
                           label=series_label(kind, size, row["method"], primary))
            table.append(row)
    return table
