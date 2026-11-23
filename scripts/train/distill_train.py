"""Distillation step 2: fine-tune a pretrained draft on the target's responses.

    python scripts/train/distill_train.py --model model/draft_50m.yaml --tokenizer standard
    python scripts/train/distill_train.py --model model/draft_50m.yaml --tokenizer superbpe
    python scripts/train/distill_train.py --tokenizer superbpe target_profile=targets/olmo2.yaml

Starts from <train.init_from>/<model>_<tokenizer> (the pretraining output) and
saves to <train.output_dir>/<model>_<tokenizer>, which is where the eval
configs look for drafts. Re-running resumes from the latest checkpoint.
"""

import argparse
import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

from superspec.config import load_config, resolve_tokenizer
from superspec.tokenization import load_vocab
from superspec.tokenization.vocab import vocab_kind
from superspec.training.distill import (
    DistillBatches,
    TeacherStore,
    build_examples,
    distill_loss,
    read_generations,
    response_bits_per_byte,
    split_train_val,
)
from superspec.training.loop import MetricsLog, make_optimizer, train
from superspec.training.model import read_draft_meta, run_name, save_draft
from superspec.training.schedule import accumulation_steps, micro_batch_indices

logger = logging.getLogger("distill_train")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train/distill.yaml")
    parser.add_argument("--model", default=None, help="model config (default: train.model)")
    parser.add_argument("--tokenizer", default=None,
                        help="standard | superbpe (the target's tokenizers) or a tokenizer config")
    parser.add_argument("overrides", nargs="*", help="e.g. train.optim.lr=5e-5")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    full_cfg = load_config(args.config, args.overrides)
    cfg = full_cfg.train
    model_cfg = load_config(args.model or cfg.model)
    vocab = load_vocab(resolve_tokenizer(full_cfg, args.tokenizer) if args.tokenizer
                       else cfg.tokenizer)
    name = run_name(model_cfg.name, vocab.name)
    init_dir = Path(cfg.init_from) / name
    run_dir = Path(cfg.output_dir) / name

    meta = read_draft_meta(run_dir)
    if meta and meta.get("stage") == "distill":
        logger.info("%s is already distilled; delete it to retrain", run_dir)
        return
    init_meta = read_draft_meta(init_dir)
    if init_meta is None or init_meta["tokenizer"] != vocab.name:
        raise ValueError(f"{init_dir} is not a finished pretraining run for tokenizer {vocab.name}")

    accelerator = Accelerator(mixed_precision=cfg.get("precision", "bf16"))
    set_seed(cfg.seed)
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(full_cfg, run_dir / "config.yaml")

    train_rows, val_rows = split_train_val(read_generations(cfg.data_path), cfg.val_every)
    max_len = min(cfg.max_seq_len, model_cfg.max_position_embeddings)
    train_examples = build_examples(train_rows, vocab, max_len)
    val_examples = build_examples(val_rows, vocab, max_len)

    accum = accumulation_steps(cfg.examples_per_step, cfg.micro_batch_size,
                               accelerator.num_processes)
    total_steps = int(cfg.epochs * len(train_examples) // cfg.examples_per_step)
    logger.info("%s: %d train / %d val examples, %d steps x %d examples, accum %d", name,
                len(train_examples), len(val_examples), total_steps, cfg.examples_per_step, accum)

    model = AutoModelForCausalLM.from_pretrained(init_dir, torch_dtype=torch.float32)
    optimizer = make_optimizer(model, cfg.optim)
    model, optimizer = accelerator.prepare(model, optimizer)
    # Round-start cuts only change SuperBPE tokenization (see training/distill.py).
    cut_rate = cfg.get("superword_cut_rate", 0.0) if vocab_kind(vocab) == "superbpe" else 0.0
    logger.info("superword cut rate: %g", cut_rate)
    # Train on the target's own next-token distribution as well (see training/distill.py).
    kl_weight = float(cfg.get("kl_weight", 0.0))
    teacher = TeacherStore(cfg.teacher_dir) if kl_weight > 0 else None
    logger.info("kl weight: %g%s", kl_weight, f" (teacher top-{teacher.k})" if teacher else "")
    batches = DistillBatches(train_examples, pad_id=vocab.eos_id, seed=cfg.seed, vocab=vocab,
                             cut_rate=cut_rate, max_len=max_len, teacher=teacher)
    first_target = None
    if teacher is not None and vocab_kind(vocab) == "superbpe":
        first_target = torch.tensor(vocab.first_target_token, device=accelerator.device)

    def loss_fn(m, batch):
        return distill_loss(m, batch, vocab.vocab_size, first_target, kl_weight)[0]

    def get_micro_batch(i):
        return batches.batch(micro_batch_indices(i, accelerator.process_index,
                                                 accelerator.num_processes, cfg.micro_batch_size))

    def evaluate(m):
        return response_bits_per_byte(m, val_examples, vocab, batch_size=cfg.micro_batch_size,
                                      max_examples=cfg.get("eval_examples"))

    log = cfg.logging
    metrics = MetricsLog(run_dir, f"{name}_distill", OmegaConf.to_container(full_cfg, resolve=True),
                         log.wandb, enabled=accelerator.is_main_process)
    train(model, accelerator, optimizer, get_micro_batch, total_steps, accum, cfg.optim, run_dir,
          metrics, evaluate=evaluate, log_every=log.every_steps,
          eval_every=log.eval_every_steps, save_every=cfg.save_every_steps,
          loss_fn=loss_fn if teacher is not None else None)

    if accelerator.is_main_process:
        save_draft(accelerator.unwrap_model(model), run_dir, vocab,
                   extra={"stage": "distill", "init_from": str(init_dir), "steps": total_steps,
                          "superword_cut_rate": cut_rate, "kl_weight": kl_weight})
        logger.info("Saved %s", run_dir)


if __name__ == "__main__":
    main()
