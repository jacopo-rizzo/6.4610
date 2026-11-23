"""Tests for training on the target's distribution (teacher top-k store and loss).

Needs torch and transformers (skipped otherwise).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from superspec.training.distill import (  # noqa: E402
    DistillBatches,
    TeacherStore,
    build_examples,
    collate,
    distill_loss,
    target_positions,
)
from test_superword import CORPUS, build  # noqa: E402


def tiny(vocab_size, seed):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return transformers.LlamaForCausalLM(config).eval()


def write_teacher(directory, rows, target, k):
    """What scripts/train/distill_teacher.py writes, computed with a full forward pass."""
    lengths = [len(r["response_ids"]) for r in rows]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    idx = np.memmap(directory / "idx.bin", dtype=np.uint32, mode="w+", shape=(int(offsets[-1]), k))
    logp = np.memmap(directory / "logp.bin", dtype=np.float16, mode="w+",
                     shape=(int(offsets[-1]), k))
    full = []
    for i, r in enumerate(rows):
        ids = torch.tensor([r["prompt_ids"] + r["response_ids"]])
        with torch.no_grad():
            lp = target(ids).logits[0, len(r["prompt_ids"]) - 1: -1].float().log_softmax(-1)
        top = lp.topk(k, dim=-1)
        idx[offsets[i]: offsets[i + 1]] = top.indices.numpy()
        logp[offsets[i]: offsets[i + 1]] = top.values.numpy()
        full.append(lp)
    idx.flush()
    logp.flush()
    np.save(directory / "offsets.npy", offsets)
    (directory / "meta.json").write_text(json.dumps({"k": k}))
    return full


@pytest.fixture(scope="module")
def setup():
    base, tok = build()
    prompt = base.encode("Tell me about the day.\n\n")
    rows = [{"index": i, "row": i, "prompt_ids": prompt, "response_ids": base.encode(t.strip("\n"))}
            for i, t in enumerate(sorted(set(CORPUS)))]
    target = tiny(base.vocab_size, seed=0)
    d = Path(tempfile.mkdtemp())
    full = write_teacher(d, rows, target, k=base.vocab_size)       # k = whole vocab: exact p
    return base, tok, rows, target, TeacherStore(d), full


def test_teacher_rows_line_up_with_draft_tokens(setup):
    base, tok, rows, _, teacher, full = setup
    examples = build_examples(rows, tok, max_len=64)
    batches = DistillBatches(examples, pad_id=tok.eos_id, seed=0, vocab=tok, cut_rate=0.3,
                             max_len=64, teacher=teacher)
    for i in range(len(examples)):
        e = batches.example(i)
        batch = collate([e], tok.eos_id, teacher, tok)
        positions = target_positions(tok, e.response)
        last = positions[-1]
        assert tok.to_target_ids(e.response)[: last + 1][-1] == e.response_target[last]
        for j, pos in enumerate(positions):
            at = len(e.prompt) + j
            assert batch["teacher_mask"][0, at]
            # The teacher row for draft token j is the target's distribution at the target
            # position where that draft token starts.
            got = torch.zeros(base.vocab_size).scatter(0, batch["teacher_idx"][0, at],
                                                       batch["teacher_logp"][0, at])
            torch.testing.assert_close(got, full[e.row][pos], atol=2e-2, rtol=0)
            # ... and its most likely token there is scored against the right label.
            assert tok.expand(int(batch["labels"][0, at]))[0] == e.response_target[pos]
        assert not batch["teacher_mask"][0, : len(e.prompt)].any()


def test_kl_is_zero_when_the_draft_equals_the_target(setup):
    base, _, rows, target, teacher, _ = setup
    examples = build_examples(rows, base, max_len=64)
    batch = collate(examples[:3], base.eos_id, teacher, base)
    with torch.no_grad():
        _, ce, kl = distill_loss(target, batch, base.vocab_size, None, kl_weight=1.0)
        other = tiny(base.vocab_size, seed=5)
        _, _, kl_other = distill_loss(other, batch, base.vocab_size, None, 1.0)
    assert float(kl) < 2e-3 and float(kl_other) > 10 * float(kl) and float(ce) > 0


def test_superbpe_kl_uses_the_first_token_marginal(setup):
    base, tok, rows, _, teacher, _ = setup
    examples = build_examples(rows, tok, max_len=64)
    batch = collate(examples[:3], tok.eos_id, teacher, tok)
    model = tiny(tok.vocab_size, seed=3)
    first_target = torch.tensor(tok.first_target_token)
    with torch.no_grad():
        loss, ce, kl = distill_loss(model, batch, tok.vocab_size, first_target, kl_weight=1.0)
        # Brute force for every teacher position.
        logits = model(input_ids=batch["input_ids"]).logits[:, :-1, : tok.vocab_size].float()
        mask = batch["teacher_mask"][:, 1:]
        q = logits[mask].softmax(-1)
        expected = []
        for n in range(q.size(0)):
            marginal = torch.zeros(base.vocab_size)
            for w in range(tok.vocab_size):
                marginal[tok.expand(w)[0]] += q[n, w]
            logp = batch["teacher_logp"][:, 1:][mask][n]
            idx = batch["teacher_idx"][:, 1:][mask][n]
            p = torch.zeros(base.vocab_size).scatter(0, idx, logp.exp())
            keep = p > 0
            expected.append(float((p[keep] * (p[keep].log() - marginal[keep].log())).sum()))
    assert abs(float(kl) - float(np.mean(expected))) < 2e-3
    assert abs(float(loss) - float(ce) - float(kl)) < 1e-5


def test_training_with_kl_brings_the_draft_closer_to_the_target(setup):
    """The stand-in target here is untrained (nearly uniform) while the response text is
    real, so cross-entropy alone moves the draft away from it. With the KL term the draft
    must end much closer to the target than without it."""
    base, tok, rows, _, teacher, _ = setup
    examples = build_examples(rows, tok, max_len=64)
    batches = DistillBatches(examples, pad_id=tok.eos_id, seed=0, vocab=tok, cut_rate=0.2,
                             max_len=64, teacher=teacher)
    first_target = torch.tensor(tok.first_target_token)
    val = collate(examples, tok.eos_id, teacher, tok)

    def final_kl(kl_weight):
        model = tiny(tok.vocab_size, seed=7).train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for step in range(60):
            batch = batches.batch(range(step * 4, step * 4 + 4))
            distill_loss(model, batch, tok.vocab_size, first_target, kl_weight)[0].backward()
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            return float(distill_loss(model, val, tok.vocab_size, first_target, 1.0)[2])

    assert final_kl(20.0) < 0.5 * final_kl(0.0)
