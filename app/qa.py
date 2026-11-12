"""The question-answering pipeline: retrieve -> generate -> verify citations."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from .generator import Generator
from .grounding import Judge, ground
from .ingest import Chunk
from .quotes import locate_quote
from .retrieval import Hit

log = logging.getLogger("policy_qa")

TOP_K = int(os.getenv("TOP_K", "6"))
REFUSAL_TEXT = "The policy documents do not contain information about this."


@dataclass
class Answer:
    question: str
    answered: bool
    answer: str
    citations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answered": self.answered,
            "answer": self.answer,
            "citations": self.citations,
        }


def verify_citations(raw: list[dict], chunks: list[Chunk]) -> tuple[list[dict], list[dict]]:
    """Keep only citations whose quote really appears in the policy text.

    A quote found in a different section (or document) than the model claimed is
    re-attributed to where it actually is, since the quote is the evidence.
    Returns (valid, rejected).
    """
    doc_ids = {c.doc_id for c in chunks}
    valid: list[dict] = []
    rejected: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for cit in raw or []:
        doc_id = str(cit.get("doc_id", "")).removesuffix(".md").strip()
        quote = str(cit.get("quote", "")).strip()
        claimed_section = str(cit.get("section", "")).strip().rstrip(".")
        section = locate_quote(quote, doc_id, chunks) if doc_id in doc_ids else None
        if section is None:  # wrong doc_id? look for the quote in every document
            for other in sorted(doc_ids):
                if (section := locate_quote(quote, other, chunks)) is not None:
                    doc_id = other
                    break
        if section is None:
            rejected.append(cit)
            continue
        if section != claimed_section:
            log.info("re-attributed quote %r: section %s -> %s", quote[:40], claimed_section, section)
        key = (doc_id, section, quote)
        if key not in seen:
            seen.add(key)
            valid.append({"doc_id": doc_id, "section": section, "quote": quote})
    return valid, rejected


class PolicyQA:
    def __init__(self, retriever, generator: Generator, chunks: list[Chunk], judge: Judge | None = None):
        self.retriever = retriever
        self.generator = generator
        self.chunks = chunks
        self.judge = judge  # None disables the sentence-level grounding check (v1 behaviour)

    def ask(self, question: str) -> Answer:
        hits: list[Hit] = self.retriever.search(question, k=TOP_K)
        log.info("retrieved: %s", ", ".join(f"{h.chunk.label}({h.dense:.2f})" for h in hits))

        result = self.generator.generate(question, hits)
        if not result.get("answered"):
            return Answer(question, False, result.get("answer") or REFUSAL_TEXT, [])

        valid, rejected = verify_citations(result.get("citations", []), self.chunks)
        if rejected:
            log.warning("dropped %d unverifiable citation(s): %s", len(rejected), rejected)
        if not valid:
            # An answer we cannot back with a verbatim quote is treated as unsupported.
            log.warning("no verifiable citation left; returning a refusal instead")
            return Answer(question, False, REFUSAL_TEXT, [])

        answer = result.get("answer", "").strip()
        if self.judge is not None:
            answer, valid, removed = ground(question, answer, valid, self.judge)
            for r in removed:
                log.warning("removed unsupported sentence %r (%s)", r["sentence"], r["reason"])
            if not answer:
                log.warning("no supported sentence left; returning a refusal instead")
                return Answer(question, False, REFUSAL_TEXT, [])
        return Answer(question, True, answer, valid)
