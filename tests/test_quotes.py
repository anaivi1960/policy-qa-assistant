from pathlib import Path

from app.ingest import load_corpus
from app.quotes import locate_quote, quote_in

CHUNKS, DOCS = load_corpus(Path(__file__).resolve().parent.parent / "data" / "policies")


def test_corpus_sections():
    assert len(DOCS) == 10
    travel = [c for c in CHUNKS if c.doc_id == "04_business_travel_policy"]
    assert [c.section for c in travel] == ["1", "2", "3", "4", "5", "6"]


def test_markdown_and_spaces_ignored():
    assert quote_in("Domestic hotels: maximum NGN 80,000 per night.", DOCS["04_business_travel_policy"])
    assert quote_in("Domestic  hotels:   maximum **NGN 80,000 per night**", DOCS["04_business_travel_policy"])


def test_paraphrase_rejected():
    assert not quote_in("Domestic hotels cost at most NGN 80,000 a night.", DOCS["04_business_travel_policy"])


def test_locate_section():
    assert locate_quote("Carried-over days must be used by 31 March", "01_annual_leave_policy", CHUNKS) == "5"
