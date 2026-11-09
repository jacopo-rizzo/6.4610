"""Report charts from the aggregated benchmark table (see results.py).

Three charts, each comparing standard-BPE drafts against SuperBPE drafts:
  speedup_vs_draft_len         speedup over autoregressive vs bytes drafted per round,
                               one panel per temperature
  speedup_by_category          speedup per prompt category (Spec-Bench's six, plus
                               HumanEval / MBPP), one panel per temperature
  acceptance_by_temperature    token and byte acceptance rates per temperature

Styling: blue = standard draft, orange = SuperBPE draft (a validated
colorblind-safe pair); model size is shown by line style and marker, so color
never carries meaning alone. Ablation methods (--methods all) take further
palette slots in a fixed order. Every chart has a legend, and the plotted
numbers are all in chart_data.csv.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")  # files only, no display needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

# ----- palette and ink (light mode, validated with the dataviz skill's checker) -----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed (kind, method) -> color, in palette order, so a series keeps its color in
# every chart regardless of which other series are present.
SERIES_COLORS = {
    ("standard", "standard"): "#2a78d6",               # blue
    ("superbpe", "superword"): "#eb6834",               # orange
    ("superbpe", "superword_pointmass"): "#1baf7a",     # aqua
    ("standard", "exact_match"): "#eda100",             # yellow
    ("superbpe", "exact_match"): "#e87ba4",             # magenta
}
FALLBACK_COLORS = ["#6250d6", "#e34948"]               # violet, red
SERIES_ORDER = list(SERIES_COLORS)

# Largest model solid with circles, the next dashed with squares, and so on.
SIZE_STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]

CATEGORY_LABELS = {
    "mt_bench": "MT-Bench", "translation": "Translation", "summarization": "Summarization",
    "qa": "QA", "math_reasoning": "Math", "rag": "RAG", "humaneval": "HumanEval", "mbpp": "MBPP",
}
BENCHMARK_LABELS = {"specbench": "Spec-Bench", "humaneval": "HumanEval", "mbpp": "MBPP"}
SPECBENCH_ORDER = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag"]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


# ----- helpers -----

def _size_value(size: str) -> float:
    match = re.match(r"([\d.]+)([MB])", size or "")
    if not match:
        return 0.0
    return float(match.group(1)) * (1e3 if match.group(2) == "B" else 1)


def series_keys(rows: list[dict]) -> list[tuple]:
    """Distinct (kind, method, size, label) of the non-baseline rows, in a fixed order."""
    keys = {(r["kind"], r["method"], r["size"], r["label"]) for r in rows if r["kind"]}

    def order(key):
        kind, method, size, _ = key
        pos = SERIES_ORDER.index((kind, method)) if (kind, method) in SERIES_ORDER else 99
        return (_size_value(size), pos, method)
    return sorted(keys, key=order)


def series_style(key: tuple, all_sizes: list[str]) -> dict:
    kind, method, size, label = key
    color = SERIES_COLORS.get((kind, method))
    if color is None:
        color = FALLBACK_COLORS[hash(method) % len(FALLBACK_COLORS)]
    rank_from_largest = sorted(all_sizes, key=_size_value, reverse=True).index(size)
    linestyle, marker = SIZE_STYLES[rank_from_largest % len(SIZE_STYLES)]
    return {"color": color, "linestyle": linestyle, "marker": marker, "label": label}


def _sizes(rows) -> list[str]:
    return sorted({r["size"] for r in rows if r["kind"]}, key=_size_value)


def _baseline_line(ax) -> None:
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)


def _times_formatter():
    return FuncFormatter(lambda v, _: f"{v:g}×")


def _figure_legend(fig, handles_labels, ncol: int) -> None:
    handles, labels = handles_labels
    ncol = ncol if len(labels) <= 4 else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.0),
               handlelength=3.2, columnspacing=1.6)


def _title(fig, title: str, subtitle: str) -> float:
    """Title and subtitle at fixed distances from the top; returns the top of the plot area
    (a figure fraction) for tight_layout."""
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / height, title, ha="left", va="top", fontsize=11, fontweight="bold")
    fig.text(0.01, 1 - 0.36 / height, subtitle, ha="left", va="top", fontsize=8.5,
             color=INK_SECONDARY)
    return 1 - 0.62 / height


def _bottom(fig, two_legend_rows: bool) -> float:
    """Bottom of the plot area, leaving room for the legend under it."""
    return (0.55 + 0.25 * two_legend_rows) / fig.get_figheight()


# ----- chart 1: speedup vs draft length -----

def speedup_vs_draft_len(rows: list[dict], target: str, benchmark: str):
    """Speedup (all prompts) vs bytes drafted per round; one panel per temperature."""
    rows = [r for r in rows if r["category"] == "all" and r["kind"]
            and r["speedup"] is not None]
    if not rows:
        return None
    temps = sorted({r["temperature"] for r in rows})
    lengths = sorted({r["draft_len_bytes"] for r in rows})
    keys, sizes = series_keys(rows), _sizes(rows)

    fig, axes = plt.subplots(1, len(temps), figsize=(3.1 * len(temps) + 0.6, 3.6),
                             sharey=True, squeeze=False)
    for ax, temp in zip(axes[0], temps):
        _baseline_line(ax)
        for key in keys:
            pts = sorted((r["draft_len_bytes"], r["speedup"]) for r in rows
                         if r["temperature"] == temp and
                         (r["kind"], r["method"], r["size"], r["label"]) == key)
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, linewidth=2, markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1, **series_style(key, sizes))
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths, [str(n) for n in lengths])
        ax.minorticks_off()
        ax.set_title(f"temperature {temp:g}", loc="left")
        ax.set_xlabel("Draft length per round (bytes)")
        ax.yaxis.set_major_formatter(_times_formatter())
    axes[0][0].set_ylabel("Speedup over autoregressive")
    top = max(r["speedup"] for r in rows)
    axes[0][0].set_ylim(0, max(1.2, top * 1.12))
    axes[0][-1].annotate("autoregressive", xy=(1, 1.0), xycoords=("axes fraction", "data"),
                         xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                         fontsize=7.5, color=INK_MUTED)

    top = _title(fig, f"Speedup vs draft length · {BENCHMARK_LABELS.get(benchmark, benchmark)}",
                 f"Target {target}. All prompts; bytes per second relative to the target alone.")
    _figure_legend(fig, axes[0][0].get_legend_handles_labels(), ncol=min(len(keys), 4))
    fig.tight_layout(rect=(0, _bottom(fig, len(keys) > 3), 1, top))
    return fig


# ----- chart 2: speedup by category -----

def _category_columns(rows: list[dict]) -> list[tuple[str, str, str]]:
    """(benchmark, category, label) for the x axis: each benchmark's categories, then
    its overall value if it has several categories."""
    columns = []
    benchmarks = sorted({r["benchmark"] for r in rows},
                        key=lambda b: (b != "specbench", b))
    for bench in benchmarks:
        cats = {r["category"] for r in rows if r["benchmark"] == bench and r["category"] != "all"}
        ordered = [c for c in SPECBENCH_ORDER if c in cats] + sorted(cats - set(SPECBENCH_ORDER))
        columns += [(bench, c, CATEGORY_LABELS.get(c, c)) for c in ordered]
        if len(ordered) > 1:
            columns.append((bench, "all", f"{BENCHMARK_LABELS.get(bench, bench)}\noverall"))
    return columns


def speedup_by_category(rows: list[dict], target: str, draft_len: int):
    """Grouped bars: speedup per category at one draft length; one panel per temperature."""
    rows = [r for r in rows if r["kind"] and r["draft_len_bytes"] == draft_len
            and r["speedup"] is not None]
    if not rows:
        return None
    temps = sorted({r["temperature"] for r in rows})
    columns = _category_columns(rows)
    keys, sizes = series_keys(rows), _sizes(rows)
    width = 0.8 / len(keys)

    fig, axes = plt.subplots(len(temps), 1, figsize=(max(7.5, 0.75 * len(columns) * max(1, len(keys) / 2.5)),
                                                     2.3 * len(temps) + 1.3),
                             sharex=True, squeeze=False)
    lookup = {(r["temperature"], r["benchmark"], r["category"],
               (r["kind"], r["method"], r["size"], r["label"])): r["speedup"] for r in rows}
    top = max(r["speedup"] for r in rows)
    for ax, temp in zip(axes[:, 0], temps):
        _baseline_line(ax)
        for k, key in enumerate(keys):
            style = series_style(key, sizes)
            xs, ys = [], []
            for c, (bench, cat, _) in enumerate(columns):
                value = lookup.get((temp, bench, cat, key))
                if value is not None:
                    xs.append(c - 0.4 + width * (k + 0.5))
                    ys.append(value)
            # Smaller models get a lighter fill so size reads without color.
            alpha = 0.55 if style["linestyle"] == "--" and len(sizes) > 1 else 1.0
            ax.bar(xs, ys, width, color=style["color"], alpha=alpha, label=style["label"],
                   edgecolor=SURFACE, linewidth=1.0, zorder=2)
        ax.set_title(f"temperature {temp:g}", loc="left")
        ax.set_ylabel("Speedup")
        ax.set_ylim(0, max(1.2, top * 1.12))
        ax.yaxis.set_major_formatter(_times_formatter())
        # Separate benchmarks with a faint vertical rule.
        for c in range(1, len(columns)):
            if columns[c][0] != columns[c - 1][0]:
                ax.axvline(c - 0.5, color=GRID, linewidth=1, zorder=1)
    axes[-1, 0].set_xticks(range(len(columns)), [label for _, _, label in columns])
    for ax in axes[:, 0]:
        ax.tick_params(axis="x", length=0)

    top = _title(fig, f"Speedup by prompt category · draft length {draft_len} bytes",
                 f"Target {target}. Bars are bytes per second relative to the target alone "
                 "(dashed line = autoregressive).")
    _figure_legend(fig, axes[0, 0].get_legend_handles_labels(), ncol=min(len(keys), 4))
    fig.tight_layout(rect=(0, _bottom(fig, len(keys) > 3), 1, top))
    return fig


# ----- chart 3: acceptance by temperature -----

def acceptance_by_temperature(rows: list[dict], target: str, benchmark: str, draft_len: int):
    """Token and byte acceptance (all prompts) per temperature, side by side."""
    rows = [r for r in rows if r["category"] == "all" and r["kind"]
            and r["draft_len_bytes"] == draft_len]
    if not rows:
        return None
    temps = sorted({r["temperature"] for r in rows})
    keys, sizes = series_keys(rows), _sizes(rows)
    width = 0.8 / len(keys)
    lookup = {(r["temperature"], (r["kind"], r["method"], r["size"], r["label"])): r
              for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(min(11, max(7.5, 0.55 * len(temps) * len(keys) + 2)),
                                            3.6), sharey=True)
    metrics = [("token_acceptance", "Token acceptance"), ("byte_acceptance", "Byte acceptance")]
    for ax, (field, title) in zip(axes, metrics):
        for k, key in enumerate(keys):
            style = series_style(key, sizes)
            xs, ys = [], []
            for t, temp in enumerate(temps):
                row = lookup.get((temp, key))
                if row is not None and row[field] is not None:
                    xs.append(t - 0.4 + width * (k + 0.5))
                    ys.append(row[field])
            alpha = 0.55 if style["linestyle"] == "--" and len(sizes) > 1 else 1.0
            ax.bar(xs, ys, width, color=style["color"], alpha=alpha, label=style["label"],
                   edgecolor=SURFACE, linewidth=1.0, zorder=2)
        ax.set_xticks(range(len(temps)), [f"T = {t:g}" for t in temps])
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Acceptance rate")

    top = _title(fig, f"Acceptance by temperature · {BENCHMARK_LABELS.get(benchmark, benchmark)}",
                 f"Target {target}. All prompts, draft length {draft_len} bytes. Share of "
                 "proposed target tokens (left) and proposed bytes (right) that were accepted.")
    _figure_legend(fig, axes[0].get_legend_handles_labels(), ncol=min(len(keys), 4))
    fig.tight_layout(rect=(0, _bottom(fig, len(keys) > 3), 1, top))
    return fig
