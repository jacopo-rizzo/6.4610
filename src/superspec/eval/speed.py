"""Decoding speed and acceptance benchmark.

For every temperature, we run the autoregressive baseline and every
(draft, method, draft length in bytes) combination over the same prompts.
Each (run, prompt) gives one row in results.jsonl. Rows are appended as they
finish, so an interrupted run resumes where it stopped when given the same
output folder. `summarize` then aggregates rows into summary.csv.

Aggregation into speedups and acceptance rates is in summary.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from superspec.decoding import SamplingConfig, SpeculativeDecoder, autoregressive_generate
from superspec.eval.common import Draft, Target, chat_prompt_ids
from superspec.eval.prompts import Prompt
from superspec.tokenization.vocab import vocab_kind

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSpec:
    temperature: float
    method: str                      # autoregressive | a METHODS entry | exact_match_noretok
    draft: str | None = None
    draft_len_bytes: int | None = None
    stop_below: float = 0.0          # speculative.draft_stop_below
    temp_scale: float = 1.0          # speculative.draft_temperature_scale

    @property
    def key(self) -> str:
        key = f"T={self.temperature}|{self.method}|{self.draft}|{self.draft_len_bytes}"
        # Appended only when set, so keys of earlier results stay valid.
        if self.stop_below:
            key += f"|stop={self.stop_below}"
        if self.temp_scale != 1.0:
            key += f"|tscale={self.temp_scale}"
        return key


def build_run_specs(cfg, drafts: dict[str, Draft]) -> list[RunSpec]:
    spec_cfg = cfg.speculative
    specs = []
    for temperature in spec_cfg.temperatures:
        if cfg.get("include_autoregressive_baseline", True):
            specs.append(RunSpec(float(temperature), "autoregressive"))
        for name, draft in drafts.items():
            methods = list(spec_cfg.methods[vocab_kind(draft.vocab)])
            if spec_cfg.get("also_run_exact_match", False):
                methods.append("exact_match" if spec_cfg.get("exact_match_retokenize", True)
                               else "exact_match_noretok")
            stop = float(spec_cfg.get("draft_stop_below", 0.0))
            scale = float(spec_cfg.get("draft_temperature_scale", 1.0))
            for method in methods:
                for n_bytes in spec_cfg.draft_len_bytes:
                    specs.append(RunSpec(float(temperature), method, name, int(n_bytes), stop,
                                         scale))
    return specs


def make_generate_fn(spec: RunSpec, cfg, target: Target, drafts: dict[str, Draft]):
    """A function (prompt_ids, generator) -> GenerationResult for one run."""
    sampling = SamplingConfig(temperature=spec.temperature, top_p=cfg.speculative.top_p)
    max_new = cfg.generation.max_new_tokens

    if spec.method == "autoregressive":
        return lambda prompt, gen: autoregressive_generate(
            target.lm, target.vocab, prompt, max_new, sampling, target.stop_ids, gen)

    draft = drafts[spec.draft]
    method = "exact_match" if spec.method.startswith("exact_match") else spec.method
    decoder = SpeculativeDecoder(
        target.lm, target.vocab, draft.lm, draft.vocab, method, sampling, target.stop_ids,
        max_draft_bytes=spec.draft_len_bytes,
        max_draft_steps=cfg.speculative.get("max_draft_steps", 16),
        max_target_tokens_per_round=cfg.speculative.get("max_target_tokens_per_round", 64),
        retokenize=spec.method != "exact_match_noretok",
        time_phases=cfg.timing.get("phase_breakdown", False),
        draft_stop_below=spec.stop_below,
        draft_temperature_scale=spec.temp_scale,
    )
    return lambda prompt, gen: decoder.generate(prompt, max_new, gen)


def _done_keys(results_path: Path) -> set[tuple[str, str]]:
    if not results_path.exists():
        return set()
    with open(results_path) as f:
        return {(r["run"], r["prompt_id"]) for r in map(json.loads, f)}


def run_benchmark(cfg, target: Target, drafts: dict[str, Draft], prompts: list[Prompt],
                  output_dir: Path, device: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    done = _done_keys(results_path)
    prompt_ids = [chat_prompt_ids(target.vocab, p.text) for p in prompts]
    specs = build_run_specs(cfg, drafts)
    logger.info("%d runs x %d prompts -> %s", len(specs), len(prompts), results_path)

    for k, spec in enumerate(specs):
        todo = [i for i, p in enumerate(prompts) if (spec.key, p.id) not in done]
        if not todo:
            continue
        logger.info("[%d/%d] %s (%d prompts)", k + 1, len(specs), spec.key, len(todo))
        generate = make_generate_fn(spec, cfg, target, drafts)

        # Warm up kernels and caches; these generations are not recorded.
        for i in range(min(cfg.timing.get("warmup_prompts", 0), len(prompts))):
            generate(prompt_ids[i], torch.Generator(device=device).manual_seed(0))

        with open(results_path, "a") as f:
            for i in todo:
                # Same seed per prompt across runs, so runs differ only in method.
                gen = torch.Generator(device=device).manual_seed(cfg.seed + i)
                result = generate(prompt_ids[i], gen)
                row = {"run": spec.key, **asdict(spec), "prompt_id": prompts[i].id,
                       "category": prompts[i].category,
                       "phase_breakdown": bool(cfg.timing.get("phase_breakdown", False)),
                       **result.summary()}
                if cfg.get("save_outputs", True):
                    row["output"] = target.vocab.decode(result.tokens)
                f.write(json.dumps(row) + "\n")
                f.flush()
    return results_path
