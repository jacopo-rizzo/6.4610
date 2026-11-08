"""Pretrain a draft model on a pretokenized data mix.

    python scripts/pretrain.py --model model/draft_50m.yaml --tokenizer tokenizer/llama3.yaml
    python scripts/pretrain.py --model model/draft_50m.yaml --tokenizer tokenizer/llama3_superbpe.yaml

    # 150M: larger data budget (pretokenize it first) and lower lr
    python scripts/pretrain.py --model model/draft_150m.yaml --tokenizer tokenizer/llama3.yaml \
        data.train_bytes=13_200_000_000 optim.lr=3e-4

    # several GPUs
    accelerate launch scripts/pretrain.py --model ... --tokenizer ...

Saves the final model to <output_dir>/<model>_<tokenizer> (e.g. draft_50m_llama3),
loadable with superspec.models.load_causal_lm. Re-running the same command
resumes from the latest checkpoint.
"""

import argparse
import logging
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from superspec.config import load_config
from superspec.data.token_files import TokenFile
from superspec.eval.heldout import heldout_loss
from superspec.tokenization import load_vocab
from superspec.training.loop import MetricsLog, make_optimizer, train
from superspec.training.model import (
    build_draft_model,
    parameter_counts,
    read_draft_meta,
    run_name,
    save_draft,
)
from superspec.training.pretrain import PretrainWindows, seq_len_for
from superspec.training.schedule import accumulation_steps, micro_batch_indices

logger = logging.getLogger("pretrain")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train/pretrain.yaml")
    parser.add_argument("--model", default=None, help="model config (default: from --config)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer config (default: from --config)")
    parser.add_argument("overrides", nargs="*", help="e.g. optim.lr=3e-4")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    model_cfg = load_config(args.model or cfg.model)
    vocab = load_vocab(args.tokenizer or cfg.tokenizer)
    name = run_name(model_cfg.name, vocab.name)
    run_dir = Path(cfg.output_dir) / name

    meta = read_draft_meta(run_dir)
    if meta and meta.get("stage") == "pretrain":
        logger.info("%s is already pretrained; delete it to retrain", run_dir)
        return

    accelerator = Accelerator(mixed_precision=cfg.get("precision", "bf16"))
    set_seed(cfg.seed)
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, run_dir / "config.yaml")

    data_dir = Path(cfg.data.pretokenized_dir) / vocab.name / f"{cfg.data.train_bytes / 1e9:g}GB"
    train_file, val_file = TokenFile(data_dir, "train"), TokenFile(data_dir, "val")
    seq_len = seq_len_for(cfg.context_bytes, train_file.bytes_per_token,
                          model_cfg.max_position_embeddings)
    windows = PretrainWindows(train_file, seq_len, seed=cfg.seed)

    batch_cfg = cfg.batch
    accum = accumulation_steps(batch_cfg.sequences_per_step, batch_cfg.micro_batch_size,
                               accelerator.num_processes)
    total_steps = int(cfg.get("epochs", 1) * windows.num_windows // batch_cfg.sequences_per_step)

    model = build_draft_model(model_cfg, vocab.vocab_size)
    counts = parameter_counts(model)
    logger.info("%s: %.1fM non-embedding params (%.1fM total), vocab %d, seq_len %d "
                "(%.2f bytes/token), %d steps x %d sequences, accum %d", name,
                counts["non_embedding"] / 1e6, counts["total"] / 1e6, vocab.vocab_size, seq_len,
                train_file.bytes_per_token, total_steps, batch_cfg.sequences_per_step, accum)

    optimizer = make_optimizer(model, cfg.optim)
    model, optimizer = accelerator.prepare(model, optimizer)

    def get_micro_batch(i):
        return windows.batch(micro_batch_indices(i, accelerator.process_index,
                                                 accelerator.num_processes,
                                                 batch_cfg.micro_batch_size))

    def evaluate(m):
        result = heldout_loss(m, vocab, val_file, context_bytes=cfg.context_bytes,
                              batch_size=batch_cfg.micro_batch_size,
                              max_windows=cfg.logging.get("eval_windows", 100))
        return {"bits_per_byte": result["bits_per_byte"], "nll_per_token": result["nll_per_token"]}

    log = cfg.logging
    metrics = MetricsLog(run_dir, name, OmegaConf.to_container(cfg, resolve=True), log.wandb,
                         enabled=accelerator.is_main_process)
    train(model, accelerator, optimizer, get_micro_batch, total_steps, accum, cfg.optim, run_dir,
          metrics, evaluate=evaluate, log_every=log.every_steps,
          eval_every=log.eval_every_steps, save_every=cfg.save_every_steps)

    if accelerator.is_main_process:
        save_draft(accelerator.unwrap_model(model), run_dir, vocab,
                   extra={"stage": "pretrain", "seq_len": seq_len, "steps": total_steps})
        logger.info("Saved %s", run_dir)


if __name__ == "__main__":
    main()
