"""Sentence-level grounding check (added in v2, after the v1 evaluation's Q2 failure).

v1 verified every *quote*, but not every *sentence of the answer*, so an unsupported
sentence could ride along with a valid quote for another fact. This module closes that gap:
the answer is split into sentences and a second model call judges each one against the
verified quotes only. Unsupported sentences are removed, and citations that no remaining
sentence relies on are dropped.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

import anthropic

from .generator import GenerationError

MODEL = os.getenv("GROUNDING_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"))
EFFORT = os.getenv("GROUNDING_EFFORT", "low")

# Split after . ! ? when followed by whitespace and an uppercase letter/quote/digit.
# Amounts like "NGN 80,000." and emails don't trigger a split mid-sentence.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(0-9])")

JUDGE_PROMPT = """You check whether each sentence of an answer is supported by the evidence.

Evidence = the numbered policy quotes below (each labelled with the policy it comes from and its section), plus facts the employee stated about their own situation in the question. Naming the policy or section a quote comes from is supported.

A sentence is SUPPORTED if every factual claim in it is stated in the quotes, or follows directly from them by simple logic or arithmetic (for example comparing an amount in the question to a limit in a quote, or "more than 2 days" implying "2 days or fewer is not covered by the rule").
A sentence is also SUPPORTED if it only says that the policy documents do not cover something.
A sentence is NOT SUPPORTED if it adds any fact, rule, number, name or generalisation that is not in the quotes, even if it sounds plausible or is true in general. Judge strictly: one unsupported claim makes the whole sentence unsupported.

For each sentence, return its index, whether it is supported, the indexes of the quotes it relies on (empty if none), and a short reason.
For a sentence that is NOT supported, also return "rewrite": the same sentence with ONLY the unsupported words removed, keeping its supported content and wording, so it still reads as a complete sentence. Do not add any new information. If nothing in the sentence is supported, return an empty string. For supported sentences, return an empty string."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "supported": {"type": "boolean"},
                    "quotes_used": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string"},
                    "rewrite": {"type": "string"},
                },
                "required": ["index", "supported", "quotes_used", "reason", "rewrite"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sentences"],
    "additionalProperties": False,
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


class Judge(Protocol):
    def judge(self, question: str, sentences: list[str], quotes: list[str]) -> list[dict]: ...
    # `quotes` entries are already labelled with their source, e.g. "(04_business_travel_policy, section 4) ..."


class ClaudeJudge:
    def __init__(self) -> None:
        # Tighter budget than generation so a full /ask stays under the 60 s contract limit.
        self.client = anthropic.Anthropic(timeout=12.0, max_retries=0)

    def judge(self, question: str, sentences: list[str], quotes: list[str]) -> list[dict]:
        content = (
            "Quotes:\n" + "\n".join(f"[{i}] {q}" for i, q in enumerate(quotes))
            + f"\n\nQuestion: {question}\n\nAnswer sentences:\n"
            + "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
        )
        try:
            response = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=JUDGE_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIError as e:
            raise GenerationError(f"grounding check failed: {type(e).__name__}") from e
        except TypeError as e:
            raise GenerationError("model provider not configured: set ANTHROPIC_API_KEY") from e
        if response.stop_reason in ("refusal", "max_tokens"):
            raise GenerationError(f"grounding check failed: {response.stop_reason}")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text)["sentences"]
        except (json.JSONDecodeError, KeyError) as e:
            raise GenerationError("grounding check returned invalid JSON") from e


def _label(c: dict) -> str:
    return f"({c['doc_id']}, section {c['section']}) {c['quote']}"


def ground(question: str, answer: str, citations: list[dict], judge: Judge) -> tuple[str, list[dict], list[dict]]:
    """Return (grounded_answer, kept_citations, removed_sentences).

    For each unsupported sentence the judge proposes a minimal-deletion rewrite. Rewrites are
    checked again in one second judge call and used only if they pass; otherwise the sentence
    is removed. Fails closed: a sentence without an explicit "supported" verdict is removed.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return "", [], []
    quotes = [_label(c) for c in citations]
    verdicts = {v.get("index"): v for v in judge.judge(question, sentences, quotes)}

    final: list[str | None] = [None] * len(sentences)
    used: set[int] = set()
    removed: list[dict] = []
    rewrites: dict[int, str] = {}
    for i, s in enumerate(sentences):
        v = verdicts.get(i) or {}
        if v.get("supported"):
            final[i] = s
            used.update(q for q in v.get("quotes_used", []) if 0 <= q < len(citations))
        elif (rw := (v.get("rewrite") or "").strip()) and rw != s:
            rewrites[i] = rw
        else:
            removed.append({"sentence": s, "reason": v.get("reason", "no verdict")})

    if rewrites:
        idx = list(rewrites)
        second = {v.get("index"): v for v in judge.judge(question, [rewrites[i] for i in idx], quotes)}
        for j, i in enumerate(idx):
            v = second.get(j) or {}
            if v.get("supported"):
                final[i] = rewrites[i]
                used.update(q for q in v.get("quotes_used", []) if 0 <= q < len(citations))
                removed.append({"sentence": sentences[i], "reason": "rewritten to: " + rewrites[i]})
            else:
                removed.append({"sentence": sentences[i], "reason": "rewrite also unsupported: " + v.get("reason", "")})

    kept = [s for s in final if s]
    # Keep only citations a remaining sentence relies on. If the judge listed none at all
    # (e.g. every kept sentence was a pure inference), keep the verified citations rather
    # than return an answer with no evidence.
    kept_cits = [c for i, c in enumerate(citations) if i in used] or (citations if kept else [])
    return " ".join(kept), kept_cits, removed
