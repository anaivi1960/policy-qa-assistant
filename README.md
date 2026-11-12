# Company Policy Q&A Assistant (RAG with citations)

A small REST service that answers employee questions using **only** the 10 Kestrel Nova
policy documents in `data/policies/`. Every answer cites the document and section it came
from, with a quote that is verified word for word against the source file. If the
documents don't contain the answer, the service says so instead of guessing.

It implements `API_CONTRACT.md`: `GET /health` and `POST /ask` on port 8000.

## Quick start

Requires Python 3.11+ (developed on 3.14) and an Anthropic API key.

```bash
git clone <this repo> && cd policy-qa-assistant
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your key in .env: ANTHROPIC_API_KEY=...
python run.py                 # listens on http://localhost:8000 (override with PORT=...)
```

The first start downloads the embedding model (about 65 MB, cached in `.cache/`), which
takes about a minute. Later starts take a few seconds.

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How much paternity leave do I get?"}'
```

```json
{
  "question": "How much paternity leave do I get?",
  "answered": true,
  "answer": "You get 10 working days of paternity leave at full pay. You must take it within 3 months of your child's birth or adoption, and you can split it into up to two blocks.",
  "citations": [
    {"doc_id": "08_parental_leave_policy", "section": "2",
     "quote": "Fathers and partners receive **10 working days** of paternity leave at full pay."},
    {"doc_id": "08_parental_leave_policy", "section": "2",
     "quote": "It must be taken within **3 months** of the child's birth or adoption and may be taken in up to two blocks."}
  ]
}
```

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Read from the environment or `.env`. Never committed. |
| `PORT` | `8000` | Port to listen on |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Answering model |
| `ANTHROPIC_EFFORT` | `medium` | Reasoning effort |
| `TOP_K` | `6` | Number of sections retrieved per question |

Errors follow the contract:
- `422 {"error": ...}` when `question` is missing, blank, longer than 1000 characters, or the body isn't JSON.
- `500 {"error": ...}` when the model provider fails or no key is set. A failure never comes back as a `200` refusal.

### Tests and evaluation

```bash
pytest -q                          # 15 offline contract tests (LLM and retriever mocked, no key needed)
python eval/evaluate.py            # with the service running: runs the 15 test questions -> results.csv
```

## Models used

| Role | Model | Why |
|---|---|---|
| Answer generation | **Claude Opus 5.5** (`claude-opus-5-5`) via the Anthropic API, effort `medium`, JSON-schema structured output | Reliable at following "only use these excerpts" and at copying quotes exactly. Structured output guarantees the response parses. |
| Embeddings | **BAAI/bge-small-en-v1.5** (384-dim), run locally with `fastembed` (ONNX) | Free, runs on CPU, about 65 MB, strong on short English retrieval. No second API key and no PyTorch install. |
| Keyword retrieval | BM25 (`rank-bm25`) | Exact-token matching for amounts and policy terms. |

The vector store is an in-memory NumPy matrix. With 48 chunks, a dedicated vector database would add complexity without benefit.

## Design choices

**1. How documents are split: one chunk per top-level section (`## N. Title`).**
The contract cites `doc_id` plus the *top-level section number*, so a section is the natural
unit: each retrieved chunk maps 1:1 to a citation, and no chunk mixes two sections. The
policy sections are short (1–6 lines, one topic each), so fixed-size or overlapping chunks
would only cut sections in half. The document title and section title are added to the text
used for retrieval (for example "Business Travel Policy. Accommodation. …") because many
section bodies don't name their own topic. This gives 48 chunks from 10 documents.

**2. Retrieval: hybrid (dense + BM25), fused with Reciprocal Rank Fusion.**
Embeddings handle paraphrase ("I lost my work laptop" → *Loss or theft*). BM25 handles exact
tokens that embeddings blur, such as `NGN 50,000`, `probation` and `anchor days`. RRF combines
the two rankings by rank instead of by score, so their score scales never need calibrating.

**3. How many chunks are retrieved: `TOP_K = 6`.**
Some questions need two documents (Q11: travel + expenses, Q12: learning + annual leave).
In a retrieval-only check, the lowest-ranked required section was 5th (Q11's expense-approval
section), so 6 leaves a margin of one. Six short sections are about 600 tokens, so recall
costs almost nothing and the model ignores irrelevant excerpts.

**4. How refusal works (two layers).**
1. *Prompted refusal.* The model sees only the retrieved excerpts and is told to answer only
   from them. If they don't contain the answer, it returns `answered: false` with a fixed
   sentence and no citations. It's also told not to fill gaps with general knowledge, and to
   answer a partly covered question while stating which part isn't covered.
2. *Evidence check in code (`app/qa.py`, `app/quotes.py`).* The service doesn't trust the
   model's citations. Every quote is checked against the source file using the graders' rule
   (Markdown symbols and extra whitespace ignored, everything else exact).
   - A quote found under a different section or document than claimed is re-attributed to where it actually is.
   - A quote that isn't in the documents is dropped.
   - If an answer has **no** verifiable quote left, it's turned into a refusal, because an answer we can't back with evidence is treated as a guess.

   `answered: false` always forces `citations: []`.

I deliberately did **not** add a retrieval-similarity threshold for refusal. Top-k always
returns *something*, and on this corpus the similarity scores of unanswerable questions
(≈0.55–0.65) overlap with those of answerable ones such as Q4 (≈0.60). A threshold would have
caused wrong refusals, so the refusal decision rests with the model reading the actual text.

**5. Staying within the 60-second limit.** One LLM call per question, with a 25 s timeout and
one retry. Measured latency was 1.6–7 s per question.

## Evaluation results (the 15 provided test questions)

| | Count |
|---|---|
| **Correct answers** | **11** of 12 answerable questions |
| **Correct refusals** | **3** of 3 unanswerable questions (Q13 pension, Q14 dog, Q15 stock options) |
| **Failures** | **1** (Q2: partially correct). No incorrect answers, wrong refusals or missed refusals. |

**The failure, Q2 (carry-over of leave): partially correct.** The core answer is right and
properly cited: up to 5 days, used by 31 March (`01_annual_leave_policy#5`). But the model added
an unrequested "related" fact, *"unused study days ... cannot be carried over"*, which **no policy
states**. The learning policy only says unused learning *budget* doesn't carry over. The sentence
has no citation, so it breaks the contract rule that every fact must be cited.
- **Why it got through:** the quote check guarantees every *quote* is real, but it doesn't check that every *answer sentence* is backed by a quote.
- **Why the automatic grader missed it:** it checks that required facts are present, not that nothing extra was added. I caught it on manual review.
- **It's intermittent:** the same question answered cleanly in the 3 earlier runs.

Full per-question results are in [`results.csv`](results.csv), and raw API responses in
[`eval/raw_responses.json`](eval/raw_responses.json).

**How these outcomes were graded:**
- `eval/answer_key.json` was written from the policy text **before** the assistant was run.
  It lists the required facts and the doc#section each answer must cite.
- `eval/evaluate.py` auto-grades against that key and re-checks every quote verbatim against the source file.
- I then **read every answer and quote by hand**. Manual grades and notes are in `eval/review.json`.
  - The auto-grader scored the final run 15/15. Manual review downgraded Q2 (above).
  - Q3 and Q11 are graded correct but carry notes:
    - Q3 adds an inference the policy implies but doesn't state.
    - Q11's second document was retrieved at rank 5 of 6, a real fragility.
- Because the test set is small, I also checked for luck:
  - **Repeat runs:** the 15 questions were run 4 times in total. Refusals were identical in all 4 runs, and the required citations were present every time. But answers vary: the *extra* sections cited changed between runs (Q1, Q2, Q8), and Q2's unsupported sentence appeared in 1 of the 4 runs. `results.csv` is the final run, reported as is rather than picking the best run.
  - **Extra robustness set** (`eval/extra_questions.csv`, written by me, **not** part of the official 15): near-miss unanswerables (bereavement leave, mileage rate, dress code, resignation notice period, salary increase), a prompt-injection attempt, trick questions that *are* answerable (carrying over sick days → no; gym membership → not reimbursable; a grade-6 overtime question → TOIL), and a half-answerable question (allowance amount → yes, taxation → not covered). **Result: 9/9 answered correctly and 6/6 correctly refused.** Results are in `eval/extra_results.csv`. A caveat: I wrote these questions myself, after seeing the system work, so they are a sanity check, not an independent benchmark.

**Honest assessment:** retrieval and refusal were reliable on this set. The weakness that
actually showed up is **over-answering**: the model adds nearby facts the user didn't ask
for, and these aren't always supported. The corpus is tiny (10 short documents), and every
answerable question is covered by one or two clearly worded sections, so a strong result here
doesn't show how it would do on a larger, messier corpus. Expected failure modes and fixes are
listed below and in `WRITEUP.md`.

### What changed after the first run

To be transparent: the system prompt, retrieval settings and `TOP_K` were **not** changed after
seeing any evaluation result. Changes made after the first run were limited to the evaluation
script (removing duplicate doc#section entries in the `citations` column, CLI flags for the
extra set) and a clearer error message when no API key is configured.

## Known limitations

- **Answer sentences aren't individually verified (observed, Q2).** Quotes are verified, but an
  unsupported extra sentence can ride along with a valid quote for another fact in the same answer.
  The fix would be a sentence-level grounding check (see `WRITEUP.md`).
- **Refusal depends on the model's judgement.** There's no independent "is this answerable?" check
  besides the quote verification.
- **Small amounts of common knowledge slip in.** For example, it treats Ghana as "international
  travel" and Spain as "outside Nigeria". These are harmless here, but the model isn't strictly
  closed-book.
- **Fixed top-k.** A question needing three or more documents, or a section ranked below 6, would lose
  evidence. Q11 already relies on rank 5.
- **Quote verification is strict and case-sensitive.** A correct answer whose only quote has a small
  copying error would be turned into a refusal (a "wrong refusal"). This didn't happen in any run,
  but it's the main trade-off of failing closed.
- **No conversation memory, no access control, English only.** Each question is answered on its own.
- **Needs internet** for the Anthropic API, and once for the embedding model download.
- **Non-deterministic output.** Repeated runs give different wording, different extra citations,
  and occasionally (Q2, 1 of 4 runs) an unsupported extra sentence.

## Project layout

```
app/
  ingest.py      split policies into one chunk per "## N." section
  retrieval.py   hybrid BM25 + bge-small embeddings, RRF fusion
  generator.py   Claude call, system prompt, JSON schema
  quotes.py      verbatim quote check (graders' normalisation rules)
  qa.py          retrieve -> generate -> verify citations -> refuse if unsupported
  main.py        FastAPI app: /health, /ask, contract-shaped errors
run.py           entry point (PORT env var)
data/policies/   the 10 policy documents
eval/            test questions, answer keys, evaluate.py, manual review, raw responses
tests/           offline contract and quote-check tests
results.csv      final results for the 15 test questions
```

## Summary

- **Correct answers: 11**
- **Correct refusals: 3**
- **Failures: 1**: Q2 partially correct (an unsupported, uncited extra claim about study days). 0 incorrect, 0 wrong refusals, 0 missed refusals.
