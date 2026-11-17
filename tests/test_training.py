"""Smoke tests for pretraining/distillation with a tiny model on CPU.

Needs torch, transformers and accelerate (skipped otherwise). Run with `pytest tests/`.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
accelerate = pytest.importorskip("accelerate")
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from superspec.data.token_files import TokenFile, TokenFileWriter  # noqa: E402
from superspec.eval.heldout import heldout_loss  # noqa: E402
from superspec.models import load_causal_lm  # noqa: E402
from superspec.training.distill import (  # noqa: E402
    DistillBatches,
    build_examples,
    collate,
    response_bits_per_byte,
)
from superspec.training.loop import MetricsLog, make_optimizer, train  # noqa: E402
from superspec.training.model import build_draft_model, read_draft_meta, save_draft  # noqa: E402
from superspec.training.pretrain import PretrainWindows  # noqa: E402
from superspec.training.schedule import micro_batch_indices  # noqa: E402
from test_superword import CORPUS, build  # noqa: E402

TINY = OmegaConf.create(dict(hidden_size=32, num_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, intermediate_size=64,
                             max_position_embeddings=128, rope_theta=10000.0,
                             rms_norm_eps=1e-5, tie_word_embeddings=True))
OPTIM = OmegaConf.create(dict(lr=3e-3, betas=[0.9, 0.95], weight_decay=0.1, grad_clip=1.0,
                              warmup_frac=0.1, min_lr_frac=0.1, schedule="cosine"))


def run_training(model, get_batch, steps, run_dir, evaluate=None, save_every=1000):
    acc = accelerate.Accelerator(cpu=True)
    opt = make_optimizer(model, OPTIM)
    model, opt = acc.prepare(model, opt)
    metrics = MetricsLog(run_dir, "test", {}, "disabled", enabled=True)
    train(model, acc, opt, get_batch, steps, accum_steps=2, optim_cfg=OPTIM, run_dir=run_dir,
          metrics=metrics, evaluate=evaluate, log_every=5, eval_every=steps,
          save_every=save_every)
    return acc.unwrap_model(model), acc


def test_pretrain_learns_and_saves():
    _, tok = build()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for split, docs in (("train", CORPUS), ("val", CORPUS[:5])):
            w = TokenFileWriter(d / "data", split, tok)
            w.write_docs([tok.encode(t) for t in docs], sum(len(t.encode()) for t in docs))
            w.close()
        train_file, val_file = TokenFile(d / "data", "train"), TokenFile(d / "data", "val")
        windows = PretrainWindows(train_file, seq_len=32)

        model = build_draft_model(TINY, tok.vocab_size)
        before = heldout_loss(model, tok, val_file, context_bytes=100, batch_size=2)
        model, _ = run_training(model, lambda i: windows.batch(micro_batch_indices(i, 0, 1, 4)),
                                40, d / "run", save_every=20)
        after = heldout_loss(model, tok, val_file, context_bytes=100, batch_size=2)
        assert after["bits_per_byte"] < 0.7 * before["bits_per_byte"]
        assert (d / "run" / "checkpoints" / "step_20" / "step.json").exists()
        assert len((d / "run" / "metrics.jsonl").read_text().splitlines()) >= 8

        save_draft(model, d / "run", tok, extra={"stage": "pretrain"})
        assert read_draft_meta(d / "run")["tokenizer"] == tok.name
        loaded = load_causal_lm(str(d / "run"), dtype="float32", device="cpu")
        again = heldout_loss(loaded, tok, val_file, context_bytes=100, batch_size=2)
        assert abs(again["bits_per_byte"] - after["bits_per_byte"]) < 1e-4


def test_resume_continues_from_checkpoint():
    _, tok = build()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        w = TokenFileWriter(d / "data", "train", tok)
        w.write_docs([tok.encode(t) for t in CORPUS], 1)
        w.close()
        windows = PretrainWindows(TokenFile(d / "data", "train"), seq_len=32)
        get = lambda i: windows.batch(micro_batch_indices(i, 0, 1, 4))  # noqa: E731

        run_training(build_draft_model(TINY, tok.vocab_size), get, 10, d / "run", save_every=5)
        lines = (d / "run" / "metrics.jsonl").read_text().splitlines()
        # Pretend the run was killed after step 5, then resume to 15 steps.
        (d / "run" / "metrics.jsonl").write_text("\n".join(lines[:1]) + "\n")
        run_training(build_draft_model(TINY, tok.vocab_size), get, 15, d / "run", save_every=5)
        logged = (d / "run" / "metrics.jsonl").read_text().splitlines()
        steps = [json.loads(x)["step"] for x in logged]
        assert steps == [5, 10, 15]       # resumed at step 5, did not start over


def test_distill_masks_prompt_and_learns():
    base, tok = build()
    prompt = base.encode("Tell me about the day.\n\n")
    rows = [{"index": i, "prompt_ids": prompt, "response_ids": base.encode(t.strip("\n"))}
            for i, t in enumerate(CORPUS)]
    examples = build_examples(rows, tok, max_len=64)

    batch = collate(examples[:2], pad_id=tok.eos_id)
    n_prompt = len(examples[0].prompt)
    assert (batch["labels"][0, :n_prompt] == -100).all()
    assert batch["labels"][0, n_prompt] == examples[0].response[0]

    with tempfile.TemporaryDirectory() as d:
        model = build_draft_model(TINY, tok.vocab_size)
        before = response_bits_per_byte(model, examples[:10], tok)
        batches = DistillBatches(examples, pad_id=tok.eos_id)
        model, _ = run_training(model, lambda i: batches.batch(micro_batch_indices(i, 0, 1, 4)),
                                30, Path(d) / "run")
        after = response_bits_per_byte(model, examples[:10], tok)
        assert after["bits_per_byte"] < 0.7 * before["bits_per_byte"]


def test_draft_model_sizes():
    from superspec.config import load_config
    from superspec.training.model import parameter_counts

    for name, expected in (("draft_50m", 50.3e6), ("draft_150m", 151.0e6)):
        model = build_draft_model(load_config(f"model/{name}.yaml"), 128256)
        counts = parameter_counts(model)
        assert abs(counts["non_embedding"] - expected) / expected < 0.01, (name, counts)
        assert model.config.vocab_size % 128 == 0

