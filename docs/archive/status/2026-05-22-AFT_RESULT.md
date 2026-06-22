# Advocate Feasibility Test Result - 2026-05-22

## Summary

The local advocate feasibility run completed on a sanitized PR-decision corpus.
This run does not justify real local fine-tuning yet.

## Inputs

- Corpus: `docs/status/generated/aft/20260522T231408Z/pr_decision_corpus.jsonl`
- Examples: 200 total, deterministic 150 train / 50 holdout split
- Holdout split seed: `aft-v0`
- Data class: sanitized GitHub PR metadata plus settlement receipt summaries
- Path signal: 197/200 examples include sanitized changed-file paths; 10 examples
  include workflow/security/auth/secrets-sensitive paths.
- No raw transcripts, comments, credentials, or secrets were included.

## Holdout Results

| Arm | Accuracy | Correct | Examples | Mock/stubbed predictions | Avg latency ms | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rules | 0.14 | 7 | 50 | 0/50 | 0.0029 | 0.00 |
| frontier_prompt | 0.14 | 7 | 50 | 50/50 (stubbed) | 0.0001 | 0.00 |
| local_advocate | 0.14 | 7 | 50 | 50/50 (stubbed) | 0.0036 | 0.00 |

## Verdict

- `local_advocate_minus_rules`: `0.0`
- `local_advocate_minus_frontier_prompt`: `0.0`
- AFT threshold was not met.
- Do not train a real local advocate model from this corpus yet.
- The `frontier_prompt` arm in this run was fixture-free and fully stubbed; it
  is not evidence against a live frontier model baseline yet.
- The `local_advocate` arm is the mock interface implementation, not a trained
  local model.

## Next Action

Improve the corpus and label rubric before model work. The current result shows that
the mock local advocate interface composes with the benchmark, but it does not yet
produce measurable decision-quality lift.
