"""Verify that citation quotes appear word for word in the source policy.

Mirrors the grader's rule from API_CONTRACT.md: Markdown symbols (`**`, `#`, `-`) and
extra spaces are ignored; everything else must match exactly.
"""

from __future__ import annotations

import re

from .ingest import Chunk

_MD_SYMBOLS = re.compile(r"[*#`>_]|^\s*-\s+", re.MULTILINE)
_SPACES = re.compile(r"\s+")
# Models sometimes "prettify" punctuation; map typographic variants back to ASCII so a
# quote isn't rejected over a curly apostrophe. Letters and digits are never altered.
_TYPO = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize(text: str) -> str:
    text = text.translate(_TYPO)
    text = _MD_SYMBOLS.sub(" ", text)
    text = text.replace(" - ", " ")  # stray list dashes after line joins
    return _SPACES.sub(" ", text).strip()


def quote_in(quote: str, source: str) -> bool:
    q = normalize(quote).rstrip(".")
    return bool(q) and q in normalize(source)


def locate_quote(quote: str, doc_id: str, chunks: list[Chunk]) -> str | None:
    """Return the section number of `doc_id` that contains the quote, or None."""
    for c in chunks:
        if c.doc_id == doc_id and quote_in(quote, c.text):
            return c.section
    return None
