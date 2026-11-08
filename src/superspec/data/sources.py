"""Streaming raw text from the HuggingFace datasets in a config's data mix.

A mix is a list of sources, each with a `weight`. Given a byte budget, each
source contributes `weight / sum(weights)` of it, read in the dataset's own
order. Reading is deterministic, so the standard and SuperBPE runs see exactly
the same documents, and the held-out split (read first) never overlaps the
training split (read after it).
"""

from __future__ import annotations

from collections.abc import Iterator


def iter_source_texts(source, skip_bytes: int = 0, max_bytes: int | None = None) -> Iterator[str]:
    """Yield documents from one source, skipping the first `skip_bytes` of text
    and stopping once `max_bytes` (UTF-8) have been yielded."""
    from datasets import load_dataset  # imported lazily: heavy dependency

    ds = load_dataset(source.hf_name, name=source.get("subset"), split=source.split,
                      streaming=True)
    field = source.get("text_field", "text")
    skipped = taken = 0
    for row in ds:
        text = row[field]
        n = len(text.encode("utf-8"))
        if skipped < skip_bytes:
            skipped += n
            continue
        if max_bytes is not None and taken >= max_bytes:
            return
        taken += n
        yield text


def iter_mixture_texts(mix, total_bytes: int, skip_bytes: int = 0) -> Iterator[tuple[str, str]]:
    """Yield (source_name, text) for `total_bytes` of text split across the mix.

    `skip_bytes` is split across sources with the same weights, so skipping the
    held-out budget lands each source right after its held-out documents.
    """
    total_weight = sum(s.weight for s in mix)
    for source in mix:
        share = source.weight / total_weight
        for text in iter_source_texts(source, skip_bytes=int(share * skip_bytes),
                                      max_bytes=int(share * total_bytes)):
            yield source.hf_name, text


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
