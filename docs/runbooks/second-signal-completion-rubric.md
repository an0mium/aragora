# Second-Signal Completion Rubric

Use this rubric when a Tier 0-2 PR is blocked only because the merge-quorum
gate needs one more model family. The goal is to collect a countable exact-head
signal without widening the conductor cycle into settlement, repair, CI reruns,
or operator-only work.

## Required preflight

Before any reviewer is run, verify all of the following on live GitHub state:

- The PR is open, non-draft, and at the same head SHA used for the evidence
  run.
- Mergeability is stable enough for settlement work, and all required checks
  other than `aragora-merge-quorum` are green.
- The merge packet resolves to Tier 0, Tier 1, or Tier 2.
- `settle_one_pr.py --json` shows model quorum as the only blocker, or the
  remaining blocker is focused dogfood that the evidence path can genuinely
  satisfy.
- `identify_lane_owner.py` and operator steering show no active owner or unread
  blocking instruction for the PR, branch, or lane.
- Current-head comments contain no `CHANGES-REQUESTED`, `[P0]`, `[P1]`, or
  concrete `[P2]` dissent.

If any item fails, do not run reviewers. Record the exact blocker and rotate.

## Label provenance

Labels are restrictions, not permissions. Treat `operator-review-required` as a
stop sign until its provenance is understood.

- Query issue events and comments to find who added the label and why.
- If the label is a live operator override restricting all agent action, do not
  post evidence.
- If the label is stale draft-era automation, or it restricts only settlement
  and merge, evidence collection may proceed but final settlement remains
  operator-gated.
- Include the provenance finding in the cycle report.

## Reviewer choice

Prefer the missing independent family, not a broad reviewer sweep.

- For a missing non-OpenAI signal, use `claude` first with explicit
  `--reviewer-timeout 600 --overall-timeout 900`.
- If Claude returns no usable body, retry once with `grok` and the documented
  OpenRouter fallback environment when required.
- Never use an interactive Claude CLI path for this conductor cycle.
- Do not retry the same family on the same head after concrete dissent.

## Dry-run gates

Always start with a dry run and save the JSON artifact under `/tmp`.

The dry-run body must satisfy all of these checks before any apply attempt:

- It is grounded on the exact live head SHA.
- The reviewer returned genuine output.
- `evidence-lint` reports `would_count=true` for the prepared body.
- The body contains no `CHANGES-REQUESTED`, `[P0]`, `[P1]`, or concrete `[P2]`.
- `dissenting_families` is empty.

Run explicit body lint against the exact prepared body before applying.

## Apply rule

Use only the prepared-json apply path for auto-posted evidence.

The apply path is allowed only after a fresh re-check confirms the same head,
same Tier 0-2 status, unchanged owner/steering state, and unchanged required
check shape. If the tool returns prepare-only, do not hand-post the body and do
not edit the artifact to make it pass.

Important trap: a single-family artifact can lint as countable while still not
posting through `--prepared-json --apply` because the apply path also requires
the prepared artifact itself to satisfy supportive quorum. In that case, record
the blocker and switch classes; do not bypass the apply path with a raw GitHub
comment.

## Repeat-blocker class switch

If a conductor cycle reaches the same head with the same apply-path blocker,
do not spend the next cycle retrying the same reviewer command. Choose a
different legal progress class:

- a settlement-stable PR whose prepared artifact can satisfy quorum by itself,
- a current-head blocker report on a different PR,
- a harvest disposition,
- a narrow runbook/tooling clarification in a draft PR, or
- a human-authorization handoff when policy requires it.

The next cycle should have a different external-progress target than the failed
second-signal attempt.
