"""Learning superword merges on top of an existing BPE vocabulary.

SuperBPE (Liu et al., 2025) trains a tokenizer in two stages:
  1. ordinary BPE, where pretokenization splits text at whitespace, up to a
     vocab size t;
  2. BPE *without* whitespace splitting, so new merges can join tokens across
     spaces ("superwords" such as " of the"), up to a vocab size T.

Stage 1 is the target's own tokenizer (Llama-3, t = 128,256), so we only learn
stage 2. We run it directly on sequences of target token ids: each new merge
joins two existing ids (base tokens or earlier superwords) into a new id
t, t+1, ... . Every superword therefore expands to a fixed sequence of target
tokens, which is what the speculative-decoding verifier relies on.

Stage 2 still respects some boundaries. "Barrier" tokens (newlines, sentence-
ending punctuation, special tokens) are never merged with anything, which cuts
each document into chunks; merges happen only inside a chunk. Superwords are
also capped at `max_words_per_token` whitespace-separated words.

This is plain Python so it is easy to read and check; it is fast enough for a
few hundred MB of training text on a machine with a decent amount of RAM.
"""

from __future__ import annotations

import heapq
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

logger = logging.getLogger(__name__)

Pair = tuple[int, int]

# A token made only of sentence-ending punctuation (optionally padded with spaces).
SENTENCE_END = re.compile(rb"^\s*[.!?]+\s*$")


def is_barrier_token(token: bytes, is_special: bool) -> bool:
    """Tokens that superwords may never contain."""
    return (
        is_special
        or len(token) == 0
        or b"\n" in token
        or SENTENCE_END.match(token) is not None
    )


def split_into_chunks(ids: list[int], is_barrier: list[bool]) -> tuple[list[list[int]], list[int]]:
    """Split a token sequence at barrier tokens.

    Returns (chunks, barriers): the maximal runs of non-barrier tokens, and the
    barrier tokens between them, interleaved as
        chunks[0], barriers[0], chunks[1], barriers[1], ..., chunks[-1].
    So len(chunks) == len(barriers) + 1, and chunks may be empty.
    """
    chunks: list[list[int]] = [[]]
    barriers: list[int] = []
    for i in ids:
        if is_barrier[i]:
            barriers.append(i)
            chunks.append([])
        else:
            chunks[-1].append(i)
    return chunks, barriers


def count_words(token: bytes) -> int:
    return len(token.split())


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, scanning left to right."""
    a, b = pair
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class _TokenInfo:
    """Per-token facts needed to enforce the max-words rule, grown as we merge."""

    def __init__(self, token_bytes: list[bytes]):
        self.bytes = list(token_bytes)
        self.words = [count_words(t) for t in token_bytes]
        self.starts_with_space = [t[:1].isspace() for t in token_bytes]
        self.ends_with_space = [t[-1:].isspace() for t in token_bytes]

    def add(self, token: bytes) -> None:
        self.bytes.append(token)
        self.words.append(count_words(token))
        self.starts_with_space.append(token[:1].isspace())
        self.ends_with_space.append(token[-1:].isspace())

    def merged_word_count(self, a: int, b: int) -> int:
        # If neither side has whitespace at the join, the last word of `a` and the
        # first word of `b` fuse into one word.
        joined = (
            self.words[a] > 0 and self.words[b] > 0
            and not self.ends_with_space[a] and not self.starts_with_space[b]
        )
        return self.words[a] + self.words[b] - int(joined)


def learn_superword_merges(
    chunk_counts: Counter[tuple[int, ...]],
    token_bytes: list[bytes],
    num_merges: int,
    max_words_per_token: int = 4,
    min_pair_count: int = 2,
    log_every: int = 1000,
) -> list[Pair]:
    """Learn up to `num_merges` superword merges with BPE.

    Args:
        chunk_counts: how often each chunk (a tuple of base token ids, no
            barriers) occurs in the training text. Deduplicating saves memory.
        token_bytes: bytes of every base token; new ids start at len(token_bytes).
        num_merges: number of merges to learn (T - t).
        max_words_per_token: cap on whitespace-separated words per superword.
        min_pair_count: stop once the most frequent pair occurs fewer times.

    Returns:
        merges in the order learned; merge k creates id len(token_bytes) + k.
    """
    base_vocab_size = len(token_bytes)
    info = _TokenInfo(token_bytes)

    def allowed(pair: Pair) -> bool:
        return info.merged_word_count(*pair) <= max_words_per_token

    words = [list(chunk) for chunk in chunk_counts if len(chunk) >= 2]
    freqs = [chunk_counts[tuple(w)] for w in words]
    logger.info("Learning merges over %d distinct chunks", len(words))

    # Count pairs. Pairs of existing tokens can only become rarer as we merge, so
    # pairs already below min_pair_count can never be chosen and are not tracked.
    # That keeps memory down: most pairs in real text occur only once.
    pair_counts: Counter[Pair] = Counter()
    for w, f in zip(words, freqs):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += f
    pair_counts = Counter({p: c for p, c in pair_counts.items() if c >= min_pair_count and allowed(p)})

    # Which chunks contain each tracked pair, so a merge only revisits those.
    where: dict[Pair, set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        for pair in zip(w, w[1:]):
            if pair in pair_counts:
                where[pair].add(wi)

    # Max-heap of (-count, pair). Entries go stale as counts change; we push a
    # fresh entry whenever a count rises and fix stale ones when popped.
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(merges) < num_merges and heap:
        neg_count, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count == 0:
            continue
        if count != -neg_count:            # stale entry: re-queue with the true count
            heapq.heappush(heap, (-count, pair))
            continue
        if count < min_pair_count:
            break

        new_id = base_vocab_size + len(merges)
        merges.append(pair)
        info.add(info.bytes[pair[0]] + info.bytes[pair[1]])

        increased: set[Pair] = set()
        for wi in where.pop(pair):
            old = words[wi]
            new = merge_pair(old, pair, new_id)
            f = freqs[wi]
            old_pairs = list(zip(old, old[1:]))
            new_pairs = list(zip(new, new[1:]))

            # Add the chunk's new pairs before removing its old ones, so a pair that
            # appears in both is never mistaken for an untracked one.
            for p in new_pairs:
                if p in pair_counts or (new_id in p and allowed(p)):
                    pair_counts[p] += f
                    where[p].add(wi)
                    increased.add(p)
            for p in old_pairs:
                if p in pair_counts:
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
            for p in set(old_pairs) - set(new_pairs):
                if p in where:
                    where[p].discard(wi)
            words[wi] = new

        pair_counts.pop(pair, None)
        for p in increased:
            if p in pair_counts:
                heapq.heappush(heap, (-pair_counts[p], p))

        if log_every and len(merges) % log_every == 0:
            logger.info("merge %d: %r (count %d)", len(merges), info.bytes[new_id], count)

    if len(merges) < num_merges:
        logger.warning("Stopped early at %d merges (no pair occurs >= %d times)",
                       len(merges), min_pair_count)
    return merges


def count_chunks(
    id_sequences: Iterable[list[int]], is_barrier: list[bool]
) -> Counter[tuple[int, ...]]:
    """Count the mergeable chunks in a stream of tokenized documents."""
    counts: Counter[tuple[int, ...]] = Counter()
    for ids in id_sequences:
        chunks, _ = split_into_chunks(ids, is_barrier)
        for chunk in chunks:
            if len(chunk) >= 2:
                counts[tuple(chunk)] += 1
    return counts
