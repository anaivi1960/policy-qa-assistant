# Manual review of the v2a run (archived)

v2a was the first grounding check: it **deleted** any sentence the judge marked as unsupported.
I read all 30 answers here (15 official + 15 extra) in full and agree with the auto-grades in
`results.csv` / `extra_results.csv`, with one adjustment:

- **Q5: incorrect (agree).** The judge rightly flagged "requested … through the Travel Desk"
  (the policy says trips are *booked* via the Travel Desk), but deleting the whole sentence also
  removed the 21-day rule. The answer left was "If you request later than that, the request may be
  refused or may need Director approval." The actual answer is gone.
- **e4: I'd grade this partially_correct, not incorrect.** "No." is the right conclusion, but the
  reason (gym counts as personal entertainment, which isn't reimbursable) was deleted because the
  judge treated naming the "Expense Reimbursement Policy" as unsupported: the document title isn't
  in the quote text. Either way it counts as a failure.
- **Q3:** the inference "for 1 or 2 days the certificate requirement does not apply" passed the
  judge. That's intended: the judge prompt allows simple logical consequences of a quote.

Both failures led to the v2b changes: source labels on quotes, and a minimal-deletion rewrite that
is checked again instead of deleting whole sentences. See the README's evaluation section.
