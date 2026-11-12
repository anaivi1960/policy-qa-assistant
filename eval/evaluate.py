"""Run the 15 test questions against a running service and write results.csv.

    python eval/evaluate.py [--url http://localhost:8000] [--out results.csv]

Grading has two layers:
1. Automatic check against eval/answer_key.json (written from the policy text before any run):
   refusal vs answer, required facts present, required doc#section cited, and every quote
   re-verified word for word against the source file (the same check the graders run).
2. Manual review: eval/review.json overrides `outcome` and `notes` per question id, but only
   when the answer is identical to the one that was reviewed (answers vary between runs).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.ingest import load_corpus  # noqa: E402
from app.quotes import quote_in  # noqa: E402

EVAL = Path(__file__).resolve().parent


def auto_grade(resp: dict, key: dict, docs: dict[str, str]) -> tuple[str, list[str]]:
    notes: list[str] = []
    answered = resp.get("answered")
    cites = resp.get("citations", [])

    bad_quotes = [c for c in cites if not quote_in(c.get("quote", ""), docs.get(c.get("doc_id"), ""))]
    if bad_quotes:
        notes.append(f"{len(bad_quotes)} quote(s) not found verbatim in source")

    if key["refuse"]:
        if not answered:
            return "correct_refusal", notes
        return "missed_refusal", notes + ["answered a question the documents do not cover"]
    if not answered:
        return "wrong_refusal", notes + ["refused although the answer is in " + ", ".join(key["cites"])]

    text = " " + resp.get("answer", "").lower() + " "
    missing = [g[0] for g in key["facts"] if not any(alt in text for alt in g)]
    cited = {f"{c['doc_id']}#{c['section']}" for c in cites}
    uncited = [c for c in key["cites"] if c not in cited]
    if missing:
        notes.append("answer is missing: " + ", ".join(missing))
    if uncited:
        notes.append("did not cite: " + ", ".join(uncited))
    if not missing and not uncited and not bad_quotes:
        return "correct", notes
    if len(missing) == len(key["facts"]):
        return "incorrect", notes
    return "partially_correct", notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default=str(ROOT / "results.csv"))
    ap.add_argument("--questions", default=str(EVAL / "test_questions.csv"))
    ap.add_argument("--key", default=str(EVAL / "answer_key.json"))
    ap.add_argument("--review", default=str(EVAL / "review.json"))
    ap.add_argument("--raw", default=str(EVAL / "raw_responses.json"))
    args = ap.parse_args()

    _, docs = load_corpus(ROOT / "data" / "policies")
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    review_path = Path(args.review)
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    questions = list(csv.DictReader(Path(args.questions).open(encoding="utf-8")))

    httpx.get(f"{args.url}/health", timeout=10).raise_for_status()
    rows, raw = [], {}
    with httpx.Client(timeout=70) as client:
        for q in questions:
            t0 = time.perf_counter()
            r = client.post(f"{args.url}/ask", json={"question": q["question"]})
            elapsed = time.perf_counter() - t0
            if r.status_code != 200:
                resp = {"answered": False, "answer": f"HTTP {r.status_code}: {r.text}", "citations": []}
                outcome, notes = "incorrect", [f"request failed with HTTP {r.status_code}"]
            else:
                resp = r.json()
                outcome, notes = auto_grade(resp, key[q["id"]], docs)
            raw[q["id"]] = {**resp, "seconds": round(elapsed, 1), "auto_outcome": outcome, "auto_notes": notes}

            override = review.get(q["id"], {})
            note_text = "; ".join(notes)
            if override and override.get("reviewed_answer") == resp.get("answer"):
                outcome, note_text = override["outcome"], override["notes"]
            elif override:
                note_text = ("AUTO-GRADED ONLY: answer differs from the manually reviewed one; "
                             "re-review needed. " + note_text).strip()
            rows.append({
                "id": q["id"],
                "question": q["question"],
                "answered": str(bool(resp.get("answered"))).lower(),
                "answer": resp.get("answer", ""),
                "citations": ";".join(dict.fromkeys(f"{c['doc_id']}#{c['section']}" for c in resp.get("citations", []))),
                "outcome": outcome,
                "notes": note_text,
            })
            print(f"[{q['id']:>2}] {outcome:<17} {elapsed:4.1f}s  {rows[-1]['citations']}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    Path(args.raw).write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    failures = sum(v for k, v in counts.items() if k not in ("correct", "correct_refusal"))
    print(f"\nCorrect answers: {counts.get('correct', 0)} | Correct refusals: {counts.get('correct_refusal', 0)} "
          f"| Failures: {failures}  {counts}")
    print(f"Wrote {args.out} and {args.raw}")


if __name__ == "__main__":
    main()
