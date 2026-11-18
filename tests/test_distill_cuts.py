"""Tests for round-start cuts in distillation data (numpy only, no models).

Run with `pytest tests/` or `python tests/test_distill_cuts.py`.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from superspec.training.distill import (  # noqa: E402
    DistillBatches,
    build_examples,
    segmented_ids,
)
from test_superword import CORPUS, build  # noqa: E402


def rows_for(base):
    prompt = base.encode("Tell me about the day.\n\n")
    return [{"index": i, "prompt_ids": prompt, "response_ids": base.encode(t.strip("\n"))}
            for i, t in enumerate(sorted(set(CORPUS)))]


def test_segmented_ids_keep_the_text_and_never_merge_across_a_cut():
    base, tok = build()
    for row in rows_for(base):
        ids = row["response_ids"]
        assert segmented_ids(tok, ids, 0.0, random.Random(0)) == tok.from_target_ids(ids)
        for seed in range(5):
            cut = segmented_ids(tok, ids, 0.3, random.Random(seed))
            assert tok.to_target_ids(cut) == ids                      # same text
            # Replay the same random draws to recover the cut positions.
            rng = random.Random(seed)
            cuts = {i for i in range(1, len(ids)) if rng.random() < 0.3}
            ends, pos = set(), 0
            for t in cut:
                pos += len(tok.expand(t))
                ends.add(pos)
            assert cuts <= ends                                       # no token spans a cut


def test_cuts_expose_contexts_that_merged_text_never_shows():
    base, tok = build()
    ids = base.encode("One of the best parts of the day is the morning")
    full = tok.from_target_ids(ids)
    # Cut inside a superword: merged text has it as one token; the cut text has what a
    # round starting there would see (its first part) and produce (the rest).
    word = next(t for t in full if len(tok.expand(t)) >= 2)
    k = sum(len(tok.expand(t)) for t in full[: full.index(word)]) + 1   # one token into it
    before, after = tok.from_target_ids(ids[:k]), tok.from_target_ids(ids[k:])
    assert full[: len(before)] != before or full[len(before):] != after   # differs from merged
    assert word not in (before[-1:] + after[:1])
    assert tok.to_target_ids(before + after) == ids


def test_batches_with_cuts_are_deterministic_and_change_by_epoch():
    base, tok = build()
    examples = build_examples(rows_for(base), tok, max_len=64)
    n = len(examples)
    plain = DistillBatches(examples, pad_id=tok.eos_id, seed=0)
    a = DistillBatches(examples, pad_id=tok.eos_id, seed=0, vocab=tok, cut_rate=0.3, max_len=64)
    b = DistillBatches(examples, pad_id=tok.eos_id, seed=0, vocab=tok, cut_rate=0.3, max_len=64)

    differs_from_plain = differs_by_epoch = False
    for i in range(n):
        e0, e1 = a.example(i), a.example(i + n)          # same position, epochs 0 and 1
        assert e0.response == b.example(i).response       # resuming reproduces the batch
        assert e0.prompt == plain.example(i).prompt       # prompts are never cut
        assert len(e0.prompt) + len(e0.response) <= 64
        text = tok.to_target_ids(e0.response)
        assert text == e0.response_target[: len(text)]
        differs_from_plain |= e0.response != plain.example(i).response
    for k in range(n):                                    # same example in both epochs
        i0 = next(i for i in range(n) if a._order(0)[i] == k)
        i1 = next(i for i in range(n) if a._order(1)[i] == k)
        differs_by_epoch |= a.example(i0).response != a.example(n + i1).response
    assert differs_from_plain and differs_by_epoch


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
