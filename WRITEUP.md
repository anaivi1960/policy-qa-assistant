# Write-up: where it failed, what I'd improve, known limitations

## Result in one line
On the 15 provided questions, the assistant got 11 correct answers and 3 correct refusals, with
1 failure (Q2, partially correct). No question was wrongly refused, and no unanswerable question
was answered. Every quote it returned is verbatim from the source.

## Where it failed

### Q2: an unsupported extra fact (partially correct)
*"How many unused leave days can I carry over, and when do I have to use them?"*

The core answer is right: up to 5 days, used by 31 March, citing `01_annual_leave_policy#5`. In the
final run, though, the model added *"unused sick days and unused study days are separate and cannot
be carried over."* The sick-days part is true and cited (`02#1`). The **study-days part is not in
any policy**. The learning policy says unused learning *budget* doesn't carry over and says nothing
about carrying over study days. The model merged two nearby facts into one that sounds plausible.

Three things combined to let this through:
1. **Over-answering.** The model tries to be helpful by adding related facts nobody asked for. Every
   extra sentence is another chance to be wrong, and the prompt doesn't discourage it.
2. **A gap in the verification layer.** My code verifies every *quote* against the source, but not
   every *sentence of the answer*. One valid quote (sick days) gave cover to a sentence that went
   further than it.
3. **Non-determinism.** The same question answered cleanly in 3 of 4 runs. A single evaluation run
   would have found this failure, or missed it, largely by chance.

My automatic grader didn't catch it either: it checks that required facts are *present*, not that
nothing unsupported was *added*. I found it on manual review, which is why I read every answer
rather than trusting the auto-grade.

### Near-misses (graded correct, with notes)
- **Q11, retrieval fragility.** The expense-approval section, which is needed for "who approves my
  claim", ranked only 5th of 6 retrieved sections. With `TOP_K` ≤ 4, the answer would have lost
  its second document and been partially correct.
- **Q3, inference.** "No certificate needed for 2 days or fewer" follows logically from the rule,
  but isn't stated.
- **Robustness set, common knowledge.** Answers assumed Ghana is "international" and Spain is
  "outside Nigeria". These are trivially true, but they show the model isn't strictly closed-book.

## What I would improve (in priority order)

1. **Sentence-level grounding check (fixes Q2).** After generation, split the answer into claims
   and check each one against the cited quotes, using a cheap second model call ("Is this sentence
   fully supported by these quotes? yes/no") or an NLI model. Unsupported sentences would be removed,
   or the answer regenerated. This turns "every fact must be cited" from an instruction into an
   enforced rule.
2. **Prompt against over-answering.** Tell the model to answer only what was asked and to leave out
   related policies unless they change the answer. This is cheap and probably fixes most cases, but
   I'd still want (1), because a prompt rule isn't a guarantee.
3. **Evaluation that measures variance.** Run each question N times and report the pass rate per
   question instead of one run's grade. Extend the auto-grader to flag answer sentences with no
   supporting quote, so the Q2 failure is caught automatically.
4. **Adaptive retrieval for multi-part questions.** Split compound questions ("…and who approves?")
   into sub-queries and retrieve for each, rather than relying on one fixed top-k. That removes the
   Q11 fragility.
5. **A larger, independently written test set.** My 15-question robustness set (9 answered, 6
   refused, all correct) was written by me after seeing the system work, so it's a sanity check
   rather than a benchmark.

## Known limitations
- The answer text isn't verified sentence by sentence; only quotes are (see Q2).
- Refusal depends on the model's judgement plus the evidence check. There's no separate
  answerability classifier, and I deliberately avoided a similarity threshold because unanswerable
  and answerable questions had overlapping retrieval scores (≈0.55–0.65) on this corpus.
- Fixed top-k = 6. A question needing three or more documents could lose evidence.
- The quote check is strict and case-sensitive by design. If the model mis-copied its only quote, a
  correct answer would become a refusal. This was never observed, but it's the trade-off of failing closed.
- No conversation memory, no user or role awareness (for example, grade-dependent overtime rules rely
  on the user stating their grade), English only.
- Needs internet for the Anthropic API, and once for the embedding-model download.
- The corpus is tiny and cleanly written. Results here don't show how it would do on long, overlapping,
  contradictory or versioned policies, which is where retrieval and citation get much harder.
