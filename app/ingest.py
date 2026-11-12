"""Load the policy Markdown files and split them into one chunk per top-level section.

Why split at `## N.` headings: the API contract cites `doc_id` + top-level section number,
and every policy section is short (one topic, 1-6 lines). One chunk per section gives a
1:1 mapping from retrieved chunk to citation, and no chunk ever straddles two sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SECTION_RE = re.compile(r"^##\s+(\d+)\.\s*(.+?)\s*$")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    doc_id: str          # file name without .md, e.g. "04_business_travel_policy"
    doc_title: str       # e.g. "Business Travel Policy"
    section: str         # top-level section number as a string, e.g. "4"
    section_title: str   # e.g. "Accommodation"
    text: str            # raw Markdown body of the section (without the heading)

    @property
    def label(self) -> str:
        return f"{self.doc_id}#{self.section}"

    @property
    def search_text(self) -> str:
        """Text used for retrieval: titles give the section context its body may lack."""
        return f"{self.doc_title}. {self.section_title}. {self.text}"


def parse_policy(path: Path) -> list[Chunk]:
    doc_id = path.stem
    doc_title = doc_id
    chunks: list[Chunk] = []
    current: tuple[str, str] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None:
            text = "\n".join(body).strip()
            if text:
                chunks.append(Chunk(doc_id, doc_title, current[0], current[1], text))

    for line in path.read_text(encoding="utf-8").splitlines():
        if m := SECTION_RE.match(line):
            flush()
            current, body = (m.group(1), m.group(2)), []
        elif current is None and (m := TITLE_RE.match(line)):
            doc_title = m.group(1)
        elif current is not None:
            body.append(line)
    flush()
    return chunks


def load_corpus(policy_dir: Path) -> tuple[list[Chunk], dict[str, str]]:
    """Return all chunks plus the full raw text of each document (used for quote checks)."""
    files = sorted(policy_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No policy .md files found in {policy_dir}")
    chunks: list[Chunk] = []
    docs: dict[str, str] = {}
    for f in files:
        chunks.extend(parse_policy(f))
        docs[f.stem] = f.read_text(encoding="utf-8")
    return chunks, docs
