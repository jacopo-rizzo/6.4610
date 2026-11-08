"""Recover the raw bytes behind each token of a byte-level BPE tokenizer.

Byte-level BPE tokenizers (GPT-2, Llama-3, OLMo-2) store tokens as printable strings
in which each byte is mapped to a unicode character (e.g. a space becomes
"Ġ"). Superword learning needs the actual bytes of each token, for example to
count how many words a candidate superword spans.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte -> printable-character table used by byte-level BPE."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return dict(zip(printable, map(chr, chars)))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def token_string_to_bytes(token: str) -> bytes:
    """Convert a byte-level token string (e.g. "Ġthe") to bytes (b" the")."""
    table = unicode_to_bytes()
    return bytes(table[c] for c in token)
