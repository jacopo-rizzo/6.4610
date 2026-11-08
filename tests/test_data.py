"""Tests for token files and data-mix byte budgets.

Run with `pytest tests/` or, without pytest, `python tests/test_data.py`.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from superspec.data import sources  # noqa: E402
from superspec.data.token_files import TokenFile, TokenFileWriter  # noqa: E402
from test_superword import CORPUS, build  # noqa: E402


def test_token_file_round_trip():
    _, tok = build()
    docs = [tok.encode(t) for t in CORPUS[:5]]
    with tempfile.TemporaryDirectory() as d:
        writer = TokenFileWriter(d, "train", tok)
        writer.write_docs(docs[:3], num_bytes=100)
        writer.write_docs(docs[3:], num_bytes=50)
        meta = writer.close()

        f = TokenFile(d, "train")
        assert meta["num_docs"] == 5 and meta["num_bytes"] == 150
        assert len(f) == sum(len(x) + 1 for x in docs)       # +1 eos per doc

        # Splitting on eos gives back the documents.
        flat = list(f.tokens)
        recovered, cur = [], []
        for i in flat:
            if i == tok.eos_id:
                recovered.append(cur)
                cur = []
            else:
                cur.append(int(i))
        assert recovered == docs
        assert max(flat) >= tok.target_vocab_size        # superword ids survive uint32

        windows = f.random_windows(4, 10, np.random.default_rng(0))
        assert windows.shape == (4, 10) and windows.dtype == np.int64


def test_mixture_budgets():
    # Fake sources: each "document" is 10 bytes, tagged with its index.
    def fake_iter_source_texts(source, skip_bytes=0, max_bytes=None):
        i, skipped, taken = 0, 0, 0
        while True:
            text = f"{source.hf_name[:1]}{i:09d}"
            i += 1
            if skipped < skip_bytes:
                skipped += 10
                continue
            if max_bytes is not None and taken >= max_bytes:
                return
            taken += 10
            yield text

    original = sources.iter_source_texts
    sources.iter_source_texts = fake_iter_source_texts
    try:
        mix = [SimpleNamespace(hf_name="web", weight=0.9), SimpleNamespace(hf_name="math", weight=0.1)]
        val = [t for _, t in sources.iter_mixture_texts(mix, total_bytes=1000)]
        train = [t for _, t in sources.iter_mixture_texts(mix, total_bytes=1000, skip_bytes=1000)]
    finally:
        sources.iter_source_texts = original

    assert sum(t.startswith("w") for t in train) == 90
    assert sum(t.startswith("m") for t in train) == 10
    assert not set(val) & set(train)                     # held-out never reused


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
