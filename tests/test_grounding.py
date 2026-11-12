"""Offline tests for the sentence-level grounding check (judge is faked)."""

from pathlib import Path

from fastapi.testclient import TestClient

from app import main
from app.grounding import ground, split_sentences
from app.ingest import load_corpus
from app.qa import PolicyQA
from app.retrieval import Hit

CHUNKS, _ = load_corpus(Path(__file__).resolve().parent.parent / "data" / "policies")

CITS = [
    {"doc_id": "01_annual_leave_policy", "section": "5", "quote": "Up to **5 unused days** may be carried over into the next calendar year."},
    {"doc_id": "02_sick_leave_policy", "section": "1", "quote": "Unused sick days cannot be carried over or paid out."},
]


class FakeJudge:
    """Marks a sentence unsupported if it contains any of the given words."""

    def __init__(self, bad_words=(), quotes_used=None, rewrites=None):
        self.bad, self.quotes_used, self.rewrites = bad_words, quotes_used, rewrites or {}

    def judge(self, question, sentences, quotes):
        out = []
        for i, s in enumerate(sentences):
            ok = not any(w in s for w in self.bad)
            used = self.quotes_used.get(i, []) if self.quotes_used is not None else list(range(len(quotes)))
            rw = self.rewrites.get(s, "") if not ok else ""
            out.append({"index": i, "supported": ok, "quotes_used": used if ok else [], "reason": "fake", "rewrite": rw})
        return out


def test_split_sentences_keeps_amounts_and_emails():
    text = "The limit is NGN 80,000. Email it-support@kestrelnova.example within 24 hours. Done."
    assert split_sentences(text) == ["The limit is NGN 80,000.", "Email it-support@kestrelnova.example within 24 hours.", "Done."]


def test_unsupported_sentence_removed_and_orphan_citation_dropped():
    answer = "You can carry over up to 5 days. Unused sick days and study days cannot be carried over."
    text, cits, removed = ground("q", answer, CITS, FakeJudge(bad_words=["study"], quotes_used={0: [0]}))
    assert text == "You can carry over up to 5 days."
    assert cits == [CITS[0]]  # the sick-leave citation only backed the removed sentence
    assert removed[0]["sentence"].startswith("Unused sick days")


def test_missing_verdict_fails_closed():
    class SilentJudge:
        def judge(self, question, sentences, quotes):
            return [{"index": 0, "supported": True, "quotes_used": [0], "reason": ""}]

    text, _, removed = ground("q", "First. Second.", CITS, SilentJudge())
    assert text == "First." and len(removed) == 1


class FakeRetriever:
    def search(self, query, k=6):
        return [Hit(c, 1.0, 1.0) for c in CHUNKS[:k]]


class FakeGenerator:
    def generate(self, question, hits):
        return {"answered": True, "answer": "Pets are welcome on Fridays.", "citations": [CITS[0]]}


def test_answer_with_no_supported_sentence_becomes_refusal():
    main.app.state.qa = PolicyQA(FakeRetriever(), FakeGenerator(), CHUNKS, FakeJudge(bad_words=["Pets"]))
    body = TestClient(main.app).post("/ask", json={"question": "Dogs?"}).json()
    assert body["answered"] is False and body["citations"] == []


def test_rewrite_keeps_supported_part():
    answer = "You can carry over up to 5 days through the Travel Desk."
    judge = FakeJudge(bad_words=["Travel Desk"], quotes_used={0: [0]},
                      rewrites={answer: "You can carry over up to 5 days."})
    text, cits, removed = ground("q", answer, CITS, judge)
    assert text == "You can carry over up to 5 days."
    assert cits == [CITS[0]] and removed[0]["reason"].startswith("rewritten")


def test_rewrite_that_fails_recheck_is_removed():
    answer = "Pets are welcome on Fridays."
    judge = FakeJudge(bad_words=["Pets", "Dogs"], rewrites={answer: "Dogs are welcome."})
    text, cits, removed = ground("q", answer, CITS, judge)
    assert text == "" and cits == [] and "rewrite also unsupported" in removed[0]["reason"]
