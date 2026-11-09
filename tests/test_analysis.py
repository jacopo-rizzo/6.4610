"""Tests for the analysis charts, on fake results (tests/fake_results.py).

Needs matplotlib and pyyaml (skipped otherwise). Run with `pytest tests/`.
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_results import write_fake_run  # noqa: E402
from superspec.analysis.results import draft_size, load_table  # noqa: E402


def test_load_table_labels_and_merges_split_runs():
    with tempfile.TemporaryDirectory() as d:
        write_fake_run(Path(d), temperatures=(0.0, 1.0))
        table = load_table([d])
    assert {r["benchmark"] for r in table} == {"specbench", "humaneval", "mbpp"}
    sb = [r for r in table if r["benchmark"] == "specbench" and r["kind"]]
    assert {r["kind"] for r in sb} == {"standard", "superbpe"}
    assert {r["size"] for r in sb} == {"50M", "150M"}
    primary = {(r["kind"], r["method"]) for r in sb if r["primary"]}
    assert primary == {("standard", "standard"), ("superbpe", "superword")}
    assert all(r["speedup"] > 1 for r in sb)               # baseline found for every row


def test_runs_split_by_temperature_are_merged():
    with tempfile.TemporaryDirectory() as d:
        write_fake_run(Path(d) / "a", temperatures=(0.0,))
        write_fake_run(Path(d) / "b", temperatures=(1.0,))
        # Same target/benchmark in two folders -> one table with both temperatures.
        table = load_table([Path(d) / "a", Path(d) / "b"])
    temps = {r["temperature"] for r in table if r["benchmark"] == "specbench"}
    assert temps == {0.0, 1.0}


def test_draft_size():
    assert draft_size("superbpe_150m") == "150M"
    assert draft_size("std_50m") == "50M"
    assert draft_size("mydraft") == ""


@pytest.mark.parametrize("methods", ["primary", "all"])
def test_make_charts_writes_every_chart(methods):
    with tempfile.TemporaryDirectory() as d:
        write_fake_run(Path(d) / "results")
        out = Path(d) / "charts"
        subprocess.run([sys.executable, str(ROOT / "scripts/analysis/make_charts.py"),
                        str(Path(d) / "results"), "--out-dir", str(out), "--methods", methods,
                        "--formats", "png"], check=True, env={"PYTHONPATH": str(ROOT / "src"),
                                                              "PATH": ""})
        names = {p.name for p in out.glob("*.png")}
        suffixes = [""] if methods == "primary" else ["_50M", "_150M"]
        for s in suffixes:
            assert f"llama3_speedup_by_category_32B{s}.png" in names
            for bench in ("specbench", "humaneval", "mbpp"):
                assert f"llama3_{bench}_speedup_vs_draft_len{s}.png" in names
                assert f"llama3_{bench}_acceptance_by_temperature_32B{s}.png" in names
        with open(out / "chart_data.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows and {"speedup", "token_acceptance", "byte_acceptance"} <= set(rows[0])
