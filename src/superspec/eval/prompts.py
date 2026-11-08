"""Benchmark prompts: Spec-Bench, HumanEval and MBPP.

Each benchmark config has a `data` section (see configs/eval/). Prompts are
returned as plain text; the caller applies the target's chat template.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from superspec.config import REPO_ROOT


@dataclass
class Prompt:
    id: str
    category: str
    text: str


def load_prompts(data_cfg, benchmark: str) -> list[Prompt]:
    if "path" in data_cfg:
        prompts = _load_specbench(data_cfg)
    else:
        prompts = _load_hf(data_cfg, benchmark)

    per_category = data_cfg.get("max_prompts_per_category")
    if per_category:
        kept, seen = [], defaultdict(int)
        for p in prompts:
            if seen[p.category] < per_category:
                kept.append(p)
                seen[p.category] += 1
        prompts = kept
    if data_cfg.get("max_prompts"):
        prompts = prompts[: data_cfg.max_prompts]
    return prompts


def _load_specbench(data_cfg) -> list[Prompt]:
    path = REPO_ROOT / data_cfg.path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see configs/eval/specbench.yaml for the "
                                "download command")
    categories = set(data_cfg.get("categories") or [])
    prompts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            category = row[data_cfg.get("category_field", "category")]
            if categories and category not in categories:
                continue
            turns = row[data_cfg.get("prompt_field", "turns")]
            prompts.append(Prompt(str(row["question_id"]), category, turns[0]))
    return prompts


def _load_hf(data_cfg, benchmark: str) -> list[Prompt]:
    from datasets import load_dataset

    ds = load_dataset(data_cfg.hf_name, name=data_cfg.get("subset"), split=data_cfg.split)
    instruction = data_cfg.get("instruction", "")
    prompts = []
    for i, row in enumerate(ds):
        text = row[data_cfg.prompt_field]
        if instruction:
            text = f"{instruction}\n\n{text}"
        prompts.append(Prompt(str(row.get("task_id", i)), benchmark, text))
    return prompts
