"""Answer generation with Claude, constrained to the retrieved policy sections.

The model returns structured JSON (answered / answer / citations). The service never trusts
the model's citations blindly: see `qa.py`, which verifies every quote against the source.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

import anthropic

from .retrieval import Hit

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("ANTHROPIC_EFFORT", "medium")

SYSTEM_PROMPT = """You answer employee questions for Kestrel Nova Ltd using ONLY the policy excerpts provided in the user message. You have no other knowledge of this company.

Rules:
1. Use only facts stated in the excerpts. Do not use general knowledge, typical company practice, or Nigerian law to fill gaps.
2. If the excerpts do not contain the information needed to answer, set "answered" to false, set "answer" to "The policy documents do not contain information about this.", and return an empty citations list. Do not guess, and do not offer a partial answer built on assumptions.
3. If the excerpts answer the question, set "answered" to true and give a short, plain-language answer (1-4 sentences). When the question describes a situation (an amount, a date, a duration), apply the policy rule to it and state the conclusion, e.g. compare the amount to the limit.
4. Answer every part of a multi-part question. If one part is covered and another is not, answer the covered part and say plainly which part the documents do not cover.
5. Citations: for every fact in your answer, cite the excerpt it came from with
   - doc_id: exactly as given in the excerpt header,
   - section: the section number exactly as given in the excerpt header,
   - quote: a short span (one sentence or one bullet) copied character-for-character from that excerpt. Never paraphrase, shorten with "...", or merge text from two places inside one quote. If an answer uses several documents, cite each one. Cite only excerpts the answer actually relies on."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "section": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["doc_id", "section", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answered", "answer", "citations"],
    "additionalProperties": False,
}


class GenerationError(RuntimeError):
    """The model provider failed or returned something unusable -> HTTP 500."""


class Generator(Protocol):
    def generate(self, question: str, hits: list[Hit]) -> dict: ...


def format_context(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        c = h.chunk
        parts.append(
            f'<excerpt doc_id="{c.doc_id}" section="{c.section}">\n'
            f"{c.doc_title} — {c.section}. {c.section_title}\n{c.text}\n</excerpt>"
        )
    return "\n\n".join(parts)


class ClaudeGenerator:
    def __init__(self) -> None:
        # Each /ask must finish within 60 s: cap a single attempt at 25 s, one retry.
        self.client = anthropic.Anthropic(timeout=25.0, max_retries=1)

    def generate(self, question: str, hits: list[Hit]) -> dict:
        user = (
            f"Policy excerpts:\n\n{format_context(hits)}\n\n"
            f"Employee question: {question}"
        )
        try:
            response = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": EFFORT,
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
                # Server-side fallback: if a safety classifier declines, the API re-runs
                # the request on a fallback model instead of returning nothing.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIError as e:
            raise GenerationError(f"model provider error: {type(e).__name__}") from e
        except TypeError as e:  # raised by the SDK when no credentials are configured
            raise GenerationError("model provider not configured: set ANTHROPIC_API_KEY") from e

        if response.stop_reason == "refusal":
            raise GenerationError("model declined the request")
        if response.stop_reason == "max_tokens":
            raise GenerationError("model output was truncated")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise GenerationError("model returned invalid JSON") from e
