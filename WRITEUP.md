# Write-up: where it failed, what I'd improve, known limitations

## Results
| Version | Official 15 questions | What changed |
|---|---|---|
| v1 | 11 correct, 3 refusals, **1 partially correct (Q2)** | Quote verification only |
| v2a | 11 correct, 3 refusals, **1 incorrect (Q5)** | + grounding check that deleted unsupported sentences |
| **v2b (final)** | **12 correct, 3 refusals, 0 failures** | + judge sees quote sources; minimal-deletion rewrites, checked again |

No version ever wrongly refused an answerable question or answered an unanswerable one, and
every quote returned was verbatim from the source.

## Where it failed

### v1, Q2: an unsupported extra fact (partially correct)
Asked how many leave days can be carried over, the model answered correctly (5 days, used by
31 March) and then added *"unused sick days and unused study days are separate and cannot be
carried over."* The sick-days part is true and cited. The **study-days part is in no policy**:
the model merged "unused learning *budget* doesn't carry over" with "study days" into something
plausible. Three causes:
1. **Over-answering.** The model adds related facts nobody asked for.
2. **A verification gap.** v1 checked every *quote* but not every *sentence*, so one valid quote
   gave cover to a sentence that went further than it.
3. **Non-determinism.** It happened in 1 of 4 runs, so a single run finds it largely by chance.

My auto-grader missed it too, because it checks that required facts are present, not that nothing
unsupported was added. I caught it by reading every answer by hand.

### v2a, Q5: the fix removed the answer (incorrect)
The first grounding check deleted any unsupported sentence. In Q5 the judge rightly objected to
international trips being "requested *through the Travel Desk*", since the policy says trips are
*booked* there. But deleting the sentence also deleted "21 days", the actual answer, leaving
*"If you request later than that…"*. In my extra set it also cut a gym-membership answer down to a
bare "No.", because it rejected naming the "Expense Reimbursement Policy": the document title
wasn't part of the quote text. The lesson: **a check that can only delete trades one failure for
another.** Precision against fabrication means nothing if it destroys correct content.

### v2b: the design that worked
- **Source labels for the judge.** It sees which policy and section each quote comes from, so
  naming a source is no longer "unsupported".
- **Rewrite instead of delete.** For an unsupported sentence the judge returns a minimal-deletion
  rewrite, which is checked again in a second call and used only if it passes. Otherwise the
  sentence is dropped.

**Evidence that it works:**
- **Replay test.** Replaying the exact archived failing answers through the judge 3 times each:
  - v1's Q2 sentence became *"Note that unused sick days cannot be carried over"* in 3/3 runs.
  - v2a's Q5 and gym answers were repaired in 3/3 runs.
  - The 11 already-correct v1 answers were left untouched in 33/33 checks.
- **Live runs.** It trimmed exactly the unsupported words in 3 of 60 requests and nothing else.

### Near-misses that remain
- **Q11, retrieval fragility.** The expense-approval section ranks 5th of the 6 retrieved
  sections; with `TOP_K` ≤ 4 the answer would lose its second document.
- **Common knowledge.** Answers assume Ghana is "international" and Spain is "outside Nigeria",
  and the judge accepts these as obvious. That's harmless, but the system isn't strictly closed-book.

## Honest caveats about the final number
- v2b was designed **after** seeing v2a fail on the same 15 questions. The changes are generic and
  nothing references a specific question, but this is still iteration on the test set. The replay
  test and repeat runs reduce, but don't remove, the risk of overfitting.
- My 15-question extra set (v2b: 9 answered, 6 refused, all correct) was written after seeing the
  system work, so it's a sanity check, not a benchmark.
- The grounding check costs a second model call. Answered questions now take 4.6–10.5 s instead of
  2.7–7.0 s, and cost roughly double.

## What I would improve next (in priority order)
1. **An independent test set** written by someone who hasn't seen the system, scored as
   pass rates over several runs instead of one run's grades.
2. **Prompt against over-answering** ("answer only what was asked"). The grounding check now
   *catches* unsupported extras; this would make the model write fewer of them, saving rewrites and
   latency.
3. **Automatic grading of extra claims.** Extend the auto-grader to flag sentences with no
   supporting quote, so a failure like Q2 is caught without relying on manual review.
4. **Adaptive retrieval for multi-part questions.** Split compound questions ("…and who approves?")
   into sub-queries to remove the Q11 fragility.
5. **Calibrate the judge.** Measure its false-removal and miss rates on a labelled set of
   supported and unsupported sentences, and compare a cheaper judge model or an NLI model on cost
   and latency.

## Known limitations
- The grounding judge is itself a model. It can miss an unsupported claim or over-correct, and its
  verdicts vary between runs. There's no third check of its rewrites.
- Refusal depends on the model's judgement plus the evidence checks. I deliberately avoided a
  similarity threshold, because answerable and unanswerable questions had overlapping retrieval
  scores (≈0.55–0.65) on this corpus.
- Fixed top-k = 6. A question needing three or more documents could lose evidence.
- The quote check is strict and case-sensitive, so a mis-copied only quote turns a correct answer
  into a refusal. This was never observed, but it's the trade-off of failing closed.
- No conversation memory and no awareness of the user's role (for example, grade-dependent overtime
  rules rely on the user stating their grade). English only. Needs internet for the API.
- The corpus is tiny and cleanly written. These results don't show how it would do on long,
  overlapping, contradictory or versioned policies.
