"""Contract tests for API_CONTRACT.md. The LLM and embedding model are faked, so these run
offline and need no API key."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.generator import GenerationError
from app.ingest import load_corpus
from app.qa import PolicyQA
from app.retrieval import Hit

CHUNKS, _ = load_corpus(Path(__file__).resolve().parent.parent / "data" / "policies")


class FakeRetriever:
    def search(self, query, k=6):
        return [Hit(c, 1.0, 1.0) for c in CHUNKS[:k]]


class FakeGenerator:
    def __init__(self, result=None, exc=None):
        self.result, self.exc = result, exc

    def generate(self, question, hits):
        if self.exc:
            raise self.exc
        return self.result


def client_with(generator) -> TestClient:
    main.app.state.qa = PolicyQA(FakeRetriever(), generator, CHUNKS)
    return TestClient(main.app)


HOTEL_ANSWER = {
    "answered": True,
    "answer": "Yes, the domestic limit is NGN 80,000 per night.",
    "citations": [
        {"doc_id": "04_business_travel_policy", "section": "4",
         "quote": "Domestic hotels: maximum NGN 80,000 per night."}
    ],
}


def test_health():
    r = client_with(FakeGenerator(HOTEL_ANSWER)).get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_answered_shape():
    r = client_with(FakeGenerator(HOTEL_ANSWER)).post("/ask", json={"question": "Hotel limit?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"question", "answered", "answer", "citations"}
    assert body["question"] == "Hotel limit?"
    assert body["answered"] is True
    assert body["citations"] == [HOTEL_ANSWER["citations"][0]]


def test_refusal_has_empty_citations():
    gen = FakeGenerator({"answered": False, "answer": "", "citations": [{"doc_id": "x", "section": "1", "quote": "y"}]})
    body = client_with(gen).post("/ask", json={"question": "Can I bring my dog?"}).json()
    assert body["answered"] is False and body["citations"] == [] and body["answer"]


def test_wrong_section_is_reattributed():
    gen = FakeGenerator({**HOTEL_ANSWER, "citations": [{**HOTEL_ANSWER["citations"][0], "section": "2"}]})
    body = client_with(gen).post("/ask", json={"question": "q"}).json()
    assert body["citations"][0]["section"] == "4"


def test_fabricated_quote_is_dropped_and_answer_refused():
    gen = FakeGenerator({**HOTEL_ANSWER, "citations": [
        {"doc_id": "04_business_travel_policy", "section": "4", "quote": "Hotels are unlimited for managers."}]})
    body = client_with(gen).post("/ask", json={"question": "q"}).json()
    assert body["answered"] is False and body["citations"] == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 1001}, {"question": 5}])
def test_bad_requests_return_422_with_error(payload):
    r = client_with(FakeGenerator(HOTEL_ANSWER)).post("/ask", json=payload)
    assert r.status_code == 422 and "error" in r.json()


def test_provider_failure_is_500_not_refusal():
    r = client_with(FakeGenerator(exc=GenerationError("provider down"))).post("/ask", json={"question": "q"})
    assert r.status_code == 500 and r.json() == {"error": "provider down"}
