"""Aggregating benchmark rows (results.jsonl) into summary.csv.

Speed is reported in bytes per second (not tokens: the vocabularies differ),
computed as total bytes / total time over all prompts of a run, and as a
speedup over the autoregressive baseline at the same temperature (and, for
per-category rows, the same category).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SUM_FIELDS = ["num_bytes", "wall_time", "rounds", "draft_steps", "proposed_tokens",
              "accepted_tokens", "proposed_bytes", "accepted_bytes", "draft_time",
              "target_time", "overhead_time", "num_tokens"]


def summarize(results_path: Path) -> list[dict]:
    """Aggregate rows per (run, category), plus an 'all' category per run."""
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]

    totals: dict[tuple[str, str], dict] = {}
    for row in rows:
        for category in (row["category"], "all"):
            key = (row["run"], category)
            if key not in totals:
                totals[key] = {k: row[k] for k in ("temperature", "method", "draft",
                                                   "draft_len_bytes")}
                totals[key].update({f: 0.0 for f in SUM_FIELDS}, category=category, prompts=0)
            t = totals[key]
            for field in SUM_FIELDS:
                t[field] += row[field]
            t["prompts"] += 1

    baseline = {(t["temperature"], t["category"]): t["num_bytes"] / t["wall_time"]
                for t in totals.values() if t["method"] == "autoregressive" and t["wall_time"]}

    summary = []
    for t in totals.values():
        bps = t["num_bytes"] / t["wall_time"] if t["wall_time"] else 0.0
        base = baseline.get((t["temperature"], t["category"]))
        rounds = t["rounds"] or 1
        summary.append({
            "temperature": t["temperature"], "category": t["category"], "method": t["method"],
            "draft": t["draft"], "draft_len_bytes": t["draft_len_bytes"], "prompts": t["prompts"],
            "bytes_per_second": round(bps, 2),
            "speedup": round(bps / base, 3) if base else None,
            "token_acceptance": _ratio(t["accepted_tokens"], t["proposed_tokens"]),
            "byte_acceptance": _ratio(t["accepted_bytes"], t["proposed_bytes"]),
            "accepted_bytes_per_round": round(t["accepted_bytes"] / rounds, 2),
            "draft_steps_per_round": round(t["draft_steps"] / rounds, 2),
            "draft_time_frac": _ratio(t["draft_time"], t["wall_time"]),
            "target_time_frac": _ratio(t["target_time"], t["wall_time"]),
            "overhead_time_frac": _ratio(t["overhead_time"], t["wall_time"]),
        })
    summary.sort(key=lambda s: (s["category"] != "all", s["category"], s["temperature"],
                                s["method"] != "autoregressive", str(s["draft"]),
                                s["method"], s["draft_len_bytes"] or 0))

    out = results_path.parent / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def format_table(summary: list[dict], category: str = "all") -> str:
    cols = ["temperature", "method", "draft", "draft_len_bytes", "bytes_per_second", "speedup",
            "token_acceptance", "byte_acceptance", "accepted_bytes_per_round"]
    rows = [s for s in summary if s["category"] == category]
    widths = {c: max([len(c)] + [len(str(r[c])) for r in rows]) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols)]
    lines += ["  ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join(lines)

