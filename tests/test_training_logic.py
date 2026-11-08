"""Tests for training logic that needs no torch: schedule, batching, examples.

Run with `pytest tests/` or, without pytest, `python tests/test_training_logic.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from superspec.training.distill import DistillBatches, build_examples, split_train_val  # noqa: E402
from superspec.training.schedule import (  # noqa: E402
    accumulation_steps,
    learning_rate,
    micro_batch_indices,
)
from test_superword import build  # noqa: E402


def test_learning_rate_schedule():
    lr = [learning_rate(s, 1000, 1e-3, warmup_frac=0.1, min_lr_frac=0.1) for s in range(1000)]
    assert abs(lr[0] - 1e-5) < 1e-12                 # warmup starts small
    assert abs(lr[99] - 1e-3) < 1e-12                # warmup ends at the peak
    assert all(a >= b for a, b in zip(lr[100:], lr[101:]))   # then only decays
    assert abs(lr[-1] - 1e-4) < 1e-6                 # down to min_lr_frac * lr


def test_micro_batches_tile_the_data():
    world, micro, accum = 3, 4, 2
    seen = []
    for step in range(5):
        for m in range(accum):
            for rank in range(world):
                seen.extend(micro_batch_indices(step * accum + m, rank, world, micro))
    assert sorted(seen) == list(range(5 * accum * world * micro))   # no gaps, no repeats


def test_accumulation_steps():
    assert accumulation_steps(256, 4, 2) == 32
    try:
        accumulation_steps(256, 3, 1)
        raise AssertionError("should reject a non-divisible batch")
    except ValueError:
        pass


def test_superwords_do_not_cross_prompt_response_boundary():
    # Chat prompts end with "\n\n" (a barrier), so encoding prompt and response
    # separately equals encoding them together, and the loss mask is exact.
    base, tok = build()
    prompt = base.encode("Tell me about the day.\n\n")
    response = base.encode("One of the best parts of the day is the morning.")
    assert (tok.from_target_ids(prompt + response)
            == tok.from_target_ids(prompt) + tok.from_target_ids(response))


def test_build_examples_truncates_and_skips():
    base, tok = build()
    prompt = base.encode("Tell me about the day.\n\n")
    response = base.encode("One of the best parts of the day is the morning. " * 5)
    rows = [{"index": 0, "prompt_ids": prompt, "response_ids": response},
            {"index": 1, "prompt_ids": prompt * 20, "response_ids": response}]
    examples = build_examples(rows, tok, max_len=40)
    assert len(examples) == 1                         # the long prompt is skipped
    e = examples[0]
    assert len(e.prompt) + len(e.response) == 40
    assert tok.to_target_ids(e.response) == response[: len(tok.to_target_ids(e.response))]


def test_split_and_epoch_orders():
    rows = [{"index": i} for i in range(1000)]
    train, val = split_train_val(rows, val_every=100)
    assert len(val) == 10 and len(train) == 990
    batches = DistillBatches(list(range(50)), pad_id=0, seed=0)
    first, second = batches._order(0), batches._order(1)
    assert sorted(first) == sorted(second) == list(range(50))
    assert list(first) != list(second)                # reshuffled each epoch


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
