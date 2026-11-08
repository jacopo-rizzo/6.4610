"""Tests for superword learning and the SuperBPE tokenizer.

Run with `pytest tests/` or, without pytest, `python tests/test_superword.py`.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from superspec.tokenization.superbpe import SuperBPETokenizer  # noqa: E402
from superspec.tokenization.superword import (  # noqa: E402
    count_chunks,
    is_barrier_token,
    learn_superword_merges,
    merge_pair,
)
from toy_vocab import ToyVocab  # noqa: E402

CORPUS = [
    "One of the best parts of the day is the morning. It is quiet.\n",
    "Most of the time, the cat sits in the sun. One of the dogs barks!\n",
    "In the morning, the sun is warm. It is one of the best things.\n",
    "As a matter of fact, most of the people in the city like the sun.\n",
    "The end of the day is calm? It is. The rest of the day is busy.\n",
] * 20


def build(num_merges=50, max_words=4):
    base = ToyVocab(CORPUS)
    is_barrier = [is_barrier_token(base.token_bytes(i), i in base.special_ids)
                  for i in range(base.vocab_size)]
    counts = count_chunks((base.encode(t) for t in CORPUS), is_barrier)
    token_bytes = [base.token_bytes(i) for i in range(base.vocab_size)]
    merges = learn_superword_merges(counts, token_bytes, num_merges,
                                    max_words_per_token=max_words, log_every=0)
    return base, SuperBPETokenizer(base, merges, max_words_per_token=max_words)


def test_learns_cross_word_merges():
    _, tok = build()
    superwords = [tok.token_bytes(i) for i in range(tok.target_vocab_size, tok.vocab_size)]
    assert b" of the" in superwords
    assert any(len(s.split()) >= 3 for s in superwords)


def test_first_merge_is_most_frequent_pair():
    base, tok = build(num_merges=1)
    is_barrier = tok.is_barrier
    pair_counts = Counter()
    for text in CORPUS:
        ids = base.encode(text)
        for a, b in zip(ids, ids[1:]):
            if not is_barrier[a] and not is_barrier[b]:
                pair_counts[(a, b)] += 1
    assert pair_counts[tok.merges[0]] == max(pair_counts.values())


def test_round_trip():
    base, tok = build()
    for text in CORPUS[:5]:
        ids = tok.encode(text)
        assert tok.to_target_ids(ids) == base.encode(text)
        assert tok.decode(ids) == text


def test_compresses_text():
    base, tok = build()
    n_base = sum(len(base.encode(t)) for t in CORPUS)
    n_super = sum(len(tok.encode(t)) for t in CORPUS)
    assert n_super < 0.8 * n_base


def test_barriers_never_merged():
    _, tok = build()
    for i in range(tok.target_vocab_size, tok.vocab_size):
        assert not any(tok.is_barrier[t] for t in tok.expansion[i])
        assert b"\n" not in tok.token_bytes(i)
        assert b"." not in tok.token_bytes(i)


def test_max_words_respected():
    _, tok = build(num_merges=200, max_words=2)
    for i in range(tok.target_vocab_size, tok.vocab_size):
        assert len(tok.token_bytes(i).split()) <= 2


def naive_bpe(chunk_counts, token_bytes, num_merges, max_words, min_pair_count=2):
    """Slow reference BPE: recount every pair from scratch at each step."""
    words = {tuple(c): n for c, n in chunk_counts.items()}
    tb = list(token_bytes)
    merges = []
    for _ in range(num_merges):
        counts = Counter()
        for w, n in words.items():
            for p in zip(w, w[1:]):
                if len((tb[p[0]] + tb[p[1]]).split()) <= max_words:
                    counts[p] += n
        if not counts:
            break
        count, pair = min((-c, p) for p, c in counts.items())
        if -count < min_pair_count:
            break
        new_id = len(tb)
        merges.append(pair)
        tb.append(tb[pair[0]] + tb[pair[1]])
        merged = Counter()
        for w, n in words.items():
            merged[tuple(merge_pair(list(w), pair, new_id))] += n
        words = merged
    return merges, words


def random_corpus(n_docs=300, seed=0):
    rng = random.Random(seed)
    vocab = "the of and a to in is it that one most best day sun cat dog city rest".split()
    docs = []
    for _ in range(n_docs):
        sentences = []
        for _ in range(rng.randint(1, 3)):
            words = [rng.choice(vocab) for _ in range(rng.randint(3, 10))]
            sentences.append(" ".join(words).capitalize() + rng.choice([".", "!", ",", ";"]))
        docs.append(" ".join(sentences) + "\n")
    return docs


def test_matches_reference_bpe():
    corpus = random_corpus()
    base = ToyVocab(corpus)
    is_barrier = [is_barrier_token(base.token_bytes(i), i in base.special_ids)
                  for i in range(base.vocab_size)]
    counts = count_chunks((base.encode(t) for t in corpus), is_barrier)
    token_bytes = [base.token_bytes(i) for i in range(base.vocab_size)]

    merges = learn_superword_merges(counts, token_bytes, 150, max_words_per_token=3, log_every=0)
    ref_merges, ref_words = naive_bpe(counts, token_bytes, 150, max_words=3)
    assert merges == ref_merges
    assert len(merges) > 50

    # The encoder must turn every training chunk into exactly what training did.
    tok = SuperBPETokenizer(base, merges, max_words_per_token=3)
    for chunk, n in counts.items():
        assert tuple(tok.from_target_ids(list(chunk))) in ref_words


def test_first_target_token():
    _, tok = build()
    for i in range(tok.vocab_size):
        assert tok.first_target_token[i] == tok.expansion[i][0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
