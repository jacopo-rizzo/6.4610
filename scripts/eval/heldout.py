"""Held-out loss (bits per byte) of every draft in an eval config.

    python scripts/eval/heldout.py --config eval/default.yaml
    python scripts/eval/heldout.py --drafts std_50m superbpe_50m heldout.max_windows=200
    python scripts/eval/heldout.py target_profile=targets/olmo2.yaml

Uses the pretokenized `val` split of each draft's tokenizer (run
scripts/train/pretokenize.py first). The proposal requires the standard and
SuperBPE drafts of the same size to have roughly matching loss, so that speed
differences come from the tokenizer and not from draft quality.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from superspec.config import load_config
from superspec.data.token_files import TokenFile
from superspec.eval.heldout import heldout_loss
from superspec.models import load_causal_lm
from superspec.tokenization import load_vocab

logger = logging.getLogger("eval_heldout")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="eval/default.yaml")
    parser.add_argument("--drafts", nargs="*", help="only these draft names (default: all)")
    parser.add_argument("--checkpoint-key", default="checkpoint",
                        help="draft field to load, e.g. a pretrain-only checkpoint field")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    h = cfg.heldout
    results = []
    for entry in cfg.drafts:
        if args.drafts and entry.name not in args.drafts:
            continue
        vocab = load_vocab(entry.tokenizer)
        val = TokenFile(Path(h.pretokenized_dir) / vocab.name / h.budget, "val")
        model = load_causal_lm(entry[args.checkpoint_key], dtype=cfg.get("dtype", "bfloat16"),
                               device=args.device)
        result = {"draft": entry.name, **heldout_loss(
            model, vocab, val, context_bytes=h.context_bytes, batch_size=h.batch_size,
            max_windows=h.get("max_windows"))}
        logger.info("%s: %.4f bits/byte (%.3f nats/token, %d tokens)", entry.name,
                    result["bits_per_byte"], result["nll_per_token"], result["tokens"])
        results.append(result)
        del model
        torch.cuda.empty_cache()

    out = Path(cfg.output_dir) / "heldout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"{'draft':<16}{'bits/byte':>10}{'nats/token':>12}{'seq_len':>9}")
    for r in results:
        print(f"{r['draft']:<16}{r['bits_per_byte']:>10.4f}{r['nll_per_token']:>12.4f}"
              f"{r['seq_len_tokens']:>9}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
