"""Tests that every config loads and resolves, for every target profile.

Needs omegaconf (skipped otherwise). Run with `pytest tests/`.
"""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superspec.config import CONFIG_DIR, load_config, resolve_tokenizer  # noqa: E402

TARGETS = ["llama3", "olmo2"]
os.environ.setdefault("CKPT_DIR", "/ckpt")


def all_configs():
    return sorted(str(p.relative_to(CONFIG_DIR)) for p in CONFIG_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("path", all_configs())
@pytest.mark.parametrize("target", TARGETS)
def test_every_config_resolves(path, target):
    cfg = load_config(path, [f"target_profile=targets/{target}.yaml"])
    OmegaConf.to_container(cfg, resolve=True)      # raises on a broken interpolation


@pytest.mark.parametrize("target", TARGETS)
def test_profile_switches_everything(target):
    overrides = [f"target_profile=targets/{target}.yaml"]
    pre = load_config("train/pretrain.yaml", overrides)
    dis = load_config("train/distill.yaml", overrides)
    ev = load_config("eval/specbench.yaml", overrides)

    assert pre.tokenizer == f"tokenizer/{target}.yaml"
    assert resolve_tokenizer(pre, "superbpe") == f"tokenizer/{target}_superbpe.yaml"
    assert target in dis.generate.output_path and dis.train.tokenizer == pre.tokenizer
    assert all(target in d.checkpoint for d in ev.drafts)
    assert ev.output_dir.endswith(target)
    for path in (pre.target.tokenizer, pre.target.superbpe_tokenizer):
        assert (CONFIG_DIR / path).exists()


@pytest.mark.parametrize("target", TARGETS)
def test_smoke_configs_use_smoke_tokenizer(target):
    overrides = [f"target_profile=targets/{target}.yaml"]
    for path in ("smoke/pretrain.yaml", "smoke/distill.yaml", "smoke/eval.yaml"):
        cfg = load_config(path, overrides)
        assert resolve_tokenizer(cfg, "superbpe") == f"smoke/{target}_superbpe.yaml", path
        assert "/smoke/" in OmegaConf.to_container(cfg, resolve=True).__repr__()
    tok = load_config(f"smoke/{target}_superbpe.yaml")
    assert tok.name == f"{target}_superbpe" and tok.num_superword_merges == 2000
    assert tok.base_tokenizer == f"tokenizer/{target}.yaml"


def test_tokenizer_configs_have_special_tokens():
    for target in TARGETS:
        special = load_config(f"tokenizer/{target}.yaml").special_tokens
        assert {"end_of_turn", "end_of_text", "pad"} <= set(special)
