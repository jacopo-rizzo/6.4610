"""Distillation step 1b: store the target's top-k next-token probabilities at every
response position, for training drafts on the target's distribution (train.kl_weight).

    python scripts/train/distill_teacher.py --config train/distill_v2.yaml
    python scripts/train/distill_teacher.py target_profile=targets/olmo2.yaml

Reads train.data_path (written by distill_generate.py) and writes train.teacher_dir in
the layout read by superspec.training.distill.TeacherStore. One forward pass of the
target over all rows; re-running resumes.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from superspec.config import load_config
from superspec.models import load_causal_lm
from superspec.tokenization import load_vocab
from superspec.training.distill import read_generations

logger = logging.getLogger("distill_teacher")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train/distill_v2.yaml")
    parser.add_argument("--tokens-per-batch", type=int, default=24000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    full_cfg = load_config(args.config, args.overrides)
    cfg, target = full_cfg.train, full_cfg.target
    k, max_tokens = cfg.get("teacher_top_k", 32), cfg.get("teacher_max_tokens", 2560)
    vocab = load_vocab(target.tokenizer)
    rows = read_generations(cfg.data_path)

    # Response positions stored per row (long rows are cut to max_tokens in total).
    lengths = np.array([max(0, min(len(r["response_ids"]), max_tokens - len(r["prompt_ids"])))
                        for r in rows], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    total = int(offsets[-1])
    out = Path(cfg.teacher_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = {"k": k, "rows": len(rows), "positions": total, "max_tokens": max_tokens,
            "target": target.hf_name, "data_path": str(cfg.data_path), "complete": False}
    old = json.loads(meta_path.read_text()) if meta_path.exists() else None
    resume = bool(old) and all(old.get(f) == meta[f] for f in ("k", "rows", "positions"))
    mode = "r+" if resume else "w+"
    idx = np.memmap(out / "idx.bin", dtype=np.uint32, mode=mode, shape=(total, k))
    logp = np.memmap(out / "logp.bin", dtype=np.float16, mode=mode, shape=(total, k))
    done_path = out / "done.npy"
    done = np.load(done_path) if resume and done_path.exists() else np.zeros(len(rows), dtype=bool)
    np.save(out / "offsets.npy", offsets)
    meta_path.write_text(json.dumps(meta))
    logger.info("%d rows, %d positions, top-%d -> %s (%d rows already done)", len(rows), total, k,
                out, int(done.sum()))

    model = load_causal_lm(target.hf_name, device=args.device)
    pad = vocab.eos_id
    # Longest rows first: similar lengths per batch, and memory problems show up at once.
    order = sorted((i for i in range(len(rows)) if not done[i] and lengths[i] > 0),
                   key=lambda i: -(len(rows[i]["prompt_ids"]) + lengths[i]))
    start, seen, last_save = time.time(), 0, time.time()
    at = 0
    while at < len(order):
        width = len(rows[order[at]]["prompt_ids"]) + int(lengths[order[at]])
        batch = order[at: at + max(1, args.tokens_per_batch // width)]
        at += len(batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for b, i in enumerate(batch):
            seq = rows[i]["prompt_ids"] + rows[i]["response_ids"][: lengths[i]]
            ids[b, : len(seq)], mask[b, : len(seq)] = torch.tensor(seq), 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(args.device),
                           attention_mask=mask.to(args.device)).logits
        for b, i in enumerate(batch):
            first = len(rows[i]["prompt_ids"]) - 1     # predicts response token 0
            lp = logits[b, first: first + lengths[i], : vocab.vocab_size].float()
            lp = lp.log_softmax(-1)
            top = lp.topk(k, dim=-1)
            idx[offsets[i]: offsets[i + 1]] = top.indices.cpu().numpy()
            logp[offsets[i]: offsets[i + 1]] = top.values.cpu().numpy()
            done[i] = True
            seen += int(lengths[i])
        if time.time() - last_save > 300 or at >= len(order):
            idx.flush()
            logp.flush()
            np.save(done_path, done)
            last_save = time.time()
            logger.info("%d / %d rows, %.0f positions/s", int(done.sum()), len(rows),
                        seen / (time.time() - start))
    meta["complete"] = True
    meta_path.write_text(json.dumps(meta))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
