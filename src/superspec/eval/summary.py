"""Aggregating benchmark rows (results.jsonl) into summary.csv.

Speed is reported in bytes per second (not tokens: the vocabularies differ),
computed as total bytes / total time over all prompts of a run, and as a
speedup over the autoregressive baseline at the same temperature (and, for
per-category rows, the same category).

Also reported, for the speedup model S = (1 - a^(g+1)) / ((1 - a)(g c + 1)):
  target_tokens_per_draft_step  proposed target tokens / draft forward passes,
                                the m in "g draft steps cover about m g tokens"
                                (1 for a standard draft, > 1 with superwords);
  draft_ms_per_step, target_ms_per_pass, and the time fractions, only for runs
  with timing.phase_breakdown=true. Without GPU syncs between phases those
  times measure kernel launches, not GPU work, so they are left empty.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROUND_START_FIELDS = ["rounds_clean", "rounds_mid", "first_accepted_clean",
                      "first_accepted_mid", "accepted_tokens_clean", "accepted_tokens_mid"]
SUM_FIELDS = ["num_bytes", "wall_time", "rounds", "draft_steps", "proposed_tokens",
              "accepted_tokens", "proposed_bytes", "accepted_bytes", "draft_time",
              "target_time", "overhead_time", "num_tokens"]


def summarize(results_path: Path) -> list[dict]:
    """Aggregate a results.jsonl into summary.csv (next to it) and return the rows."""
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]
    summary = aggregate(rows)
    out = results_path.parent / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate per-prompt rows per (run, category), plus an 'all' category per run."""
    totals: dict[tuple[str, str], dict] = {}
    for row in rows:
        for category in (row["category"], "all"):
            key = (row["run"], category)
            if key not in totals:
                totals[key] = {k: row[k] for k in ("temperature", "method", "draft",
                                                   "draft_len_bytes")}
                totals[key].update(stop_below=row.get("stop_below", 0.0),
                                   temp_scale=row.get("temp_scale", 1.0))
                totals[key].update({f: 0.0 for f in SUM_FIELDS + ROUND_START_FIELDS},
                                   category=category, prompts=0)
            t = totals[key]
            for field in SUM_FIELDS:
                t[field] += row[field]
            for field in ROUND_START_FIELDS:     # rows from before these fields existed: 0
                t[field] += row.get(field, 0)
            t["prompts"] += 1
            # Rows written before this field existed were always phase-synced.
            t["phases_synced"] = t.get("phases_synced", True) and row.get("phase_breakdown", True)

    baseline = {(t["temperature"], t["category"]): t["num_bytes"] / t["wall_time"]
                for t in totals.values() if t["method"] == "autoregressive" and t["wall_time"]}

    summary = []
    for t in totals.values():
        bps = t["num_bytes"] / t["wall_time"] if t["wall_time"] else 0.0
        base = baseline.get((t["temperature"], t["category"]))
        rounds = t["rounds"] or 1
        synced = t["phases_synced"] and t["method"] != "autoregressive"
        summary.append({
            "temperature": t["temperature"], "category": t["category"], "method": t["method"],
            "draft": t["draft"], "draft_len_bytes": t["draft_len_bytes"],
            "stop_below": t["stop_below"], "temp_scale": t["temp_scale"], "prompts": t["prompts"],
            "bytes_per_second": round(bps, 2),
            "speedup": round(bps / base, 3) if base else None,
            "token_acceptance": _ratio(t["accepted_tokens"], t["proposed_tokens"]),
            "byte_acceptance": _ratio(t["accepted_bytes"], t["proposed_bytes"]),
            "accepted_bytes_per_round": round(t["accepted_bytes"] / rounds, 2),
            "draft_steps_per_round": round(t["draft_steps"] / rounds, 2),
            "target_tokens_per_draft_step": _ratio(t["proposed_tokens"], t["draft_steps"]),
            "accepted_tokens_per_round": _ratio(t["accepted_tokens"], t["rounds"]),
            # Round starts: share that began mid-superword, and how often the first proposed
            # token was accepted in each kind (None for rows without these fields).
            "mid_superword_round_frac": _ratio(t["rounds_mid"],
                                               t["rounds_clean"] + t["rounds_mid"]),
            "first_token_accept_clean": _ratio(t["first_accepted_clean"], t["rounds_clean"]),
            "first_token_accept_mid": _ratio(t["first_accepted_mid"], t["rounds_mid"]),
            "accepted_tokens_per_round_clean": _ratio(t["accepted_tokens_clean"],
                                                      t["rounds_clean"]),
            "accepted_tokens_per_round_mid": _ratio(t["accepted_tokens_mid"], t["rounds_mid"]),
            "draft_ms_per_step":
                _ratio(1000 * t["draft_time"], t["draft_steps"]) if synced else None,
            "target_ms_per_pass":
                _ratio(1000 * t["target_time"], t["rounds"]) if synced else None,
            "draft_time_frac": _ratio(t["draft_time"], t["wall_time"]) if synced else None,
            "target_time_frac": _ratio(t["target_time"], t["wall_time"]) if synced else None,
            "overhead_time_frac": _ratio(t["overhead_time"], t["wall_time"]) if synced else None,
        })
    summary.sort(key=lambda s: (s["category"] != "all", s["category"], s["temperature"],
                                s["method"] != "autoregressive", str(s["draft"]),
                                s["method"], s["draft_len_bytes"] or 0, s["stop_below"],
                                s["temp_scale"]))
    return summary


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def format_table(summary: list[dict], category: str = "all") -> str:
    cols = ["temperature", "method", "draft", "draft_len_bytes", "bytes_per_second", "speedup",
            "token_acceptance", "byte_acceptance", "accepted_bytes_per_round",
            "target_tokens_per_draft_step"]
    rows = [s for s in summary if s["category"] == category]
    widths = {c: max([len(c)] + [len(str(r[c])) for r in rows]) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols)]
    lines += ["  ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join(lines)

