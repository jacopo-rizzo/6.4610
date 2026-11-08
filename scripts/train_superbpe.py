"""One-time: learn the SuperBPE draft tokenizer's superword merges.

    python scripts/train_superbpe.py --config tokenizer/llama3_superbpe.yaml

Tokenizes the configured text with the target (Llama-3) tokenizer, learns
`num_superword_merges` superwords on top of it, and saves the result to the
config's `output_dir`. Load it afterwards with
`load_vocab("tokenizer/llama3_superbpe.yaml")`.
"""

import argparse
import logging
import time
from collections import Counter

from superspec.config import load_config
from superspec.data.sources import batched, iter_mixture_texts
from superspec.tokenization import SuperBPETokenizer, load_vocab
from superspec.tokenization.superword import (
    count_chunks,
    is_barrier_token,
    learn_superword_merges,
)

logger = logging.getLogger("train_superbpe")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="tokenizer/llama3_superbpe.yaml")
    parser.add_argument("overrides", nargs="*", help="e.g. num_train_bytes=50_000_000")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.overrides)
    base = load_vocab(cfg.base_tokenizer)
    logger.info("Base tokenizer %s, vocab size t = %d", base.name, base.vocab_size)

    is_barrier = [is_barrier_token(base.token_bytes(i), i in base.special_ids)
                  for i in range(base.vocab_size)]

    # 1. Tokenize the training text with the base tokenizer and count chunks.
    start = time.time()
    chunk_counts: Counter = Counter()
    num_bytes = num_tokens = 0
    texts = (text for _, text in iter_mixture_texts(cfg.train_data, cfg.num_train_bytes,
                                                    skip_bytes=cfg.get("skip_bytes", 0)))
    for batch in batched(texts, 1000):
        ids = base.encode_batch(batch)
        chunk_counts.update(count_chunks(ids, is_barrier))
        num_bytes += sum(len(t.encode("utf-8")) for t in batch)
        num_tokens += sum(map(len, ids))
    logger.info("Tokenized %.1f MB into %d tokens (%.2f bytes/token), %d distinct chunks, %.0fs",
                num_bytes / 1e6, num_tokens, num_bytes / num_tokens, len(chunk_counts),
                time.time() - start)

    # 2. Learn the superword merges.
    start = time.time()
    token_bytes = [base.token_bytes(i) for i in range(base.vocab_size)]
    merges = learn_superword_merges(
        chunk_counts, token_bytes, cfg.num_superword_merges,
        max_words_per_token=cfg.max_words_per_token,
        min_pair_count=cfg.get("min_pair_count", 2),
    )
    logger.info("Learned %d merges in %.0fs", len(merges), time.time() - start)

    # 3. Save, and report how much the superwords compress the training text.
    tokenizer = SuperBPETokenizer(base, merges, name=cfg.name,
                                  max_words_per_token=cfg.max_words_per_token)
    tokenizer.save(cfg.output_dir)
    num_super = sum(len(tokenizer._encode_chunk(list(c))) * n for c, n in chunk_counts.items())
    num_in_chunks = sum(len(c) * n for c, n in chunk_counts.items())
    logger.info("Vocab size T = %d. Inside chunks, %d base tokens -> %d superword tokens "
                "(%.1f%% fewer)", tokenizer.vocab_size, num_in_chunks, num_super,
                100 * (1 - num_super / num_in_chunks))
    logger.info("Saved to %s (see superwords.tsv to inspect)", cfg.output_dir)


if __name__ == "__main__":
    main()
