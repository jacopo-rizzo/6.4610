"""The training loop shared by pretraining and distillation.

Uses HF Accelerate, so the same code runs on one GPU (`python scripts/...`)
or several (`accelerate launch scripts/...`), with bf16 mixed precision.

The caller supplies `get_micro_batch(micro_index)`, returning the inputs for
`model(**batch)` (including `labels`), and optionally `evaluate(model)`. One
optimizer step = `accum_steps` micro-batches per process. Checkpoints (model,
optimizer, RNG, step) go to <run_dir>/checkpoints/step_N and the run resumes
from the latest one automatically; the final model is saved in <run_dir>.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch

from superspec.training.schedule import learning_rate

logger = logging.getLogger(__name__)


def make_optimizer(model, optim_cfg) -> torch.optim.Optimizer:
    """AdamW; no weight decay on norms and biases (1-D parameters)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": optim_cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=optim_cfg.lr, betas=tuple(optim_cfg.betas),
                             fused=torch.cuda.is_available())


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*"),
                         key=lambda p: int(p.name.split("_")[1]))
    return checkpoints[-1] if checkpoints else None


class MetricsLog:
    """Appends metrics to <run_dir>/metrics.jsonl and, if enabled, to wandb."""

    def __init__(self, run_dir: Path, run_name: str, config: dict, wandb_mode: str, enabled: bool):
        self.enabled = enabled
        self.path = run_dir / "metrics.jsonl"
        if enabled:
            run_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        if enabled and wandb_mode not in (None, "disabled"):
            import os

            import wandb

            self.wandb = wandb.init(project=os.environ.get("WANDB_PROJECT", "superspec"),
                                    name=run_name, config=config, mode=wandb_mode, resume="allow")

    def log(self, step: int, metrics: dict) -> None:
        if not self.enabled:
            return
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.wandb:
            self.wandb.log(metrics, step=step)


def train(
    model,
    accelerator,
    optimizer,
    get_micro_batch: Callable[[int], dict],
    total_steps: int,
    accum_steps: int,
    optim_cfg,
    run_dir: Path,
    metrics: MetricsLog,
    evaluate: Callable | None = None,
    log_every: int = 20,
    eval_every: int = 1000,
    save_every: int = 2000,
    keep_checkpoints: int = 2,
) -> None:
    """Run (or resume) training up to `total_steps` optimizer steps."""
    start_step = 0
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is not None:
        accelerator.load_state(str(checkpoint))
        start_step = json.loads((checkpoint / "step.json").read_text())["step"]
        logger.info("Resumed from %s (step %d)", checkpoint, start_step)

    model.train()
    window_start, window_loss = time.time(), 0.0
    for step in range(start_step, total_steps):
        lr = learning_rate(step, total_steps, optim_cfg.lr, optim_cfg.get("warmup_frac", 0.01),
                           optim_cfg.get("min_lr_frac", 0.1), optim_cfg.get("schedule", "cosine"))
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        for micro in range(accum_steps):
            batch = get_micro_batch(step * accum_steps + micro)
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            last = micro == accum_steps - 1
            # Only all-reduce gradients on the last micro-batch of the step.
            with nullcontext() if last else accelerator.no_sync(model):
                loss = model(**batch).loss
                accelerator.backward(loss / accum_steps)
            step_loss += loss.item() / accum_steps

        grad_norm = accelerator.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        window_loss += step_loss
        done = step + 1

        if done % log_every == 0 or done == total_steps:
            elapsed = time.time() - window_start
            n = log_every if done % log_every == 0 else done % log_every
            metrics.log(done, {"train/loss": window_loss / n, "train/lr": lr,
                               "train/grad_norm": float(grad_norm),
                               "train/sec_per_step": elapsed / n})
            logger.info("step %d/%d  loss %.4f  lr %.2e  %.2fs/step", done, total_steps,
                        window_loss / n, lr, elapsed / n)
            window_start, window_loss = time.time(), 0.0

        if evaluate is not None and (done % eval_every == 0 or done == total_steps):
            if accelerator.is_main_process:
                model.eval()
                with accelerator.autocast():
                    results = evaluate(accelerator.unwrap_model(model))
                metrics.log(done, {f"eval/{k}": v for k, v in results.items()})
                logger.info("step %d eval: %s", done, results)
                model.train()
            accelerator.wait_for_everyone()

        if done % save_every == 0 and done < total_steps:
            _save_checkpoint(accelerator, run_dir, done, keep_checkpoints)


def _save_checkpoint(accelerator, run_dir: Path, step: int, keep: int) -> None:
    path = run_dir / "checkpoints" / f"step_{step}"
    accelerator.save_state(str(path))
    if accelerator.is_main_process:
        (path / "step.json").write_text(json.dumps({"step": step}))
        old = sorted((run_dir / "checkpoints").glob("step_*"),
                     key=lambda p: int(p.name.split("_")[1]))[:-keep]
        for p in old:
            shutil.rmtree(p)
    accelerator.wait_for_everyone()
    logger.info("Saved checkpoint %s", path)
