"""Tokenize the pretraining data mix into token files, once per tokenizer.

    python scripts/train/pretokenize.py --tokenizer standard
    python scripts/train/pretokenize.py --tokenizer superbpe

    # 150M draft budget / OLMo-2 target:
    python scripts/train/pretokenize.py --tokenizer standard data.train_bytes=13_200_000_000
    python scripts/train/pretokenize.py --tokenizer superbpe target_profile=targets/olmo2.yaml

Both tokenizers read exactly the same documents. Output goes to
<data.pretokenized_dir>/<tokenizer name>/<train budget>/{val,train}.{bin,json}.
"""

import argparse
import logging
from pathlib import Path

from superspec.config import load_config, resolve_config_path, resolve_tokenizer
from superspec.data.pretokenize import pretokenize_split
from superspec.tokenization import load_vocab

logger = logging.getLogger("pretokenize")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train/pretrain.yaml")
    parser.add_argument("--tokenizer", default=None,
                        help="standard | superbpe (the target's tokenizers) or a tokenizer config")
    parser.add_argument("--splits", nargs="+", default=["val", "train"], choices=["val", "train"])
    parser.add_argument("overrides", nargs="*", help="e.g. data.train_bytes=1_000_000")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    tokenizer_path = str(resolve_config_path(resolve_tokenizer(cfg, args.tokenizer)))
    vocab = load_vocab(tokenizer_path)
    data = cfg.data

    output_dir = Path(data.pretokenized_dir) / vocab.name / f"{data.train_bytes / 1e9:g}GB"
    logger.info("Tokenizer %s (vocab %d) -> %s", vocab.name, vocab.vocab_size, output_dir)

    # The held-out split is read first; training text starts right after it.
    budgets = {"val": (data.held_out_bytes, 0), "train": (data.train_bytes, data.held_out_bytes)}
    for split in args.splits:
        total_bytes, skip_bytes = budgets[split]
        pretokenize_split(data.mix, vocab, tokenizer_path, output_dir, split,
                          total_bytes=total_bytes, skip_bytes=skip_bytes,
                          num_workers=data.get("num_workers", 8))


if __name__ == "__main__":
    main()
