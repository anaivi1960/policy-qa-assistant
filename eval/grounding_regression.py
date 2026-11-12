"""Replay the archived v1 answers through the v2 grounding check.

    python eval/grounding_regression.py [--runs 3]

v1's one failure (Q2: an unsupported claim about study days) appeared in only 1 of 4 runs,
so a fresh eval run cannot tell whether the grounding check fixes it. This script feeds the
exact v1 answers (eval/v1/raw_responses.json) to the real judge and reports, per question,
which sentences it removes. That measures both the fix (Q2) and the cost (any sentence
wrongly removed from the 11 answers that were already correct).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from app.grounding import ClaudeJudge, ground  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=3)
args = ap.parse_args()

v1 = json.loads((ROOT / "eval" / "v1" / "raw_responses.json").read_text(encoding="utf-8"))
questions = {r.split(",", 1)[0]: r.split(",", 1)[1].strip().strip('"')
             for r in (ROOT / "eval" / "test_questions.csv").read_text(encoding="utf-8").splitlines()[1:]}
# The two answers the first grounding version (v2a) damaged, rebuilt exactly from
# eval/v2a/server_log.txt (removed sentence + the text that remained).
TRAVEL_Q = {"doc_id": "04_business_travel_policy", "section": "2"}
cases = {qid: r for qid, r in v1.items() if r["answered"]}
cases["v2a-Q5"] = {"answer": "International travel must be requested at least 21 days before departure, through the Travel Desk. "
                             "If you request later than that, the request may be refused or may need Director approval.",
                   "citations": [{**TRAVEL_Q, "quote": "International travel must be requested at least **21 days** before departure."},
                                 {**TRAVEL_Q, "quote": "Late requests may be refused or may need Director approval."},
                                 {**TRAVEL_Q, "quote": "All flights and hotels are booked through the Travel Desk (travel@kestrelnova.example)."}]}
cases["v2a-e4"] = {"answer": "No. Gym costs count as personal entertainment under the Expense Reimbursement Policy, and personal entertainment is not reimbursable.",
                   "citations": [{"doc_id": "05_expense_reimbursement_policy", "section": "5",
                                  "quote": "Personal entertainment (films, minibar, gym) is not reimbursable."}]}
questions["v2a-Q5"] = questions["5"]
questions["v2a-e4"] = "Can I claim my gym membership as an expense?"

judge = ClaudeJudge()
report = {}
for qid, resp in cases.items():
    report[qid] = []
    for run in range(args.runs):
        text, cits, removed = ground(questions[qid], resp["answer"], resp["citations"], judge)
        report[qid].append({"removed": removed, "citations_kept": len(cits), "citations_before": len(resp["citations"])})
        flag = ("CHANGED " + " | ".join(f"{r['sentence']}  =>  {r['reason']}" for r in removed)) if removed else "all sentences kept"
        print(f"Q{qid} run {run + 1}: {flag}")
out = ROOT / "eval" / "grounding_regression.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out}")
