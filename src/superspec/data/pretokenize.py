"""Tokenize a pretraining data mix once and store it as token files.

Run through scripts/pretokenize.py. Tokenizing is split across worker
processes; each worker loads the tokenizer once.
"""

from __future__ import annotations

import logging
import os
from multiprocessing import Pool
from pathlib import Path

from superspec.data.sources import batched, iter_mixture_texts
from superspec.data.token_files import TokenFileWriter

logger = logging.getLogger(__name__)

_worker_vocab = None


def _init_worker(tokenizer_config_path: str) -> None:
    global _worker_vocab
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # we parallelize with processes instead
    from superspec.tokenization import load_vocab

    _worker_vocab = load_vocab(tokenizer_config_path)


def _encode(texts: list[str]) -> tuple[list[list[int]], int]:
    num_bytes = sum(len(t.encode("utf-8")) for t in texts)
    return _worker_vocab.encode_batch(texts), num_bytes


def pretokenize_split(mix, vocab, tokenizer_config_path: str, output_dir: Path, split: str,
                      total_bytes: int, skip_bytes: int = 0, num_workers: int = 8,
                      batch_size: int = 256, log_every_batches: int = 200) -> dict:
    """Tokenize `total_bytes` of the mix (after skipping `skip_bytes`) into one split."""
    texts = (text for _, text in iter_mixture_texts(mix, total_bytes, skip_bytes=skip_bytes))
    writer = TokenFileWriter(output_dir, split, vocab)

    with Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_config_path,)) as pool:
        # imap keeps batches in order, so the output is deterministic.
        for k, (docs, num_bytes) in enumerate(pool.imap(_encode, batched(texts, batch_size))):
            writer.write_docs(docs, num_bytes)
            if k % log_every_batches == 0:
                logger.info("%s: %.2f / %.2f GB, %d tokens", split, writer.num_bytes / 1e9,
                            total_bytes / 1e9, writer.num_tokens)

    meta = writer.close()
    logger.info("%s done: %d tokens, %d docs, %.3f bytes/token", split, meta["num_tokens"],
                meta["num_docs"], meta["bytes_per_token"])
    return meta
