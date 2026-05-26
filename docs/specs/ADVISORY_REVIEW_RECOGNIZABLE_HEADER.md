# Advisory-Review Workflow: Recognizable Per-Model Header (Tier 4 Pre-Approval)

**Status**: design doc / pre-approval artifact for a Tier 4 merge-authority-
adjacent change. **Not implemented in this PR.** This file + the failing
governance tests in
`tests/governance/test_advisory_review_recognizable_header.py` are the
pre-approval artifact for the future implementation PR, per
`docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`.

## Problem statement

`aragora-merge-quorum.yml` requires *recognized* model-review signals to
satisfy the Tier 0–2 quorum. Signals are recognized via
`aragora/cli/commands/review_queue.py::_infer_model_reviewer_from_text`,
which scans a PR comment's first markdown heading (falling back to the
first 200 characters) for one of seven family markers:
`claude / codex / tesla / harvey / factory / grok / gemini`.

The advisory `aragora-review-gate.yml` workflow currently posts comments
headed `## Aragora Code Review` (see lines 280 and 291 of
`.github/workflows/aragora-review-gate.yml` as of `main` at the time of
writing). None of the recognizer's seven markers appear in that header.
The recognizer therefore maps every advisory review comment to
`unknown_model_reviewer`, which `_known_model_reviewer_id` neutralizes
at counting time.

This is a *silent quorum-suppression bug*: the advisory review *does*
run, and it *does* invoke real model agents (`--agents anthropic-api,
openai-api`, see line 162 of the same workflow), but the resulting
comment is invisible to the recognizer. Two PRs are blocked on this
specific gap as of 2026-05-26:

- **#7450** (model-quorum family-expansion spec, Tier 0): non-draft, all
  required checks green; `aragora-merge-quorum` returns
  `needs_model_review_quorum: 0/1 signal(s)`. The advisory review *did*
  post; it just doesn't count.
- **#7451** (model-family bench harness scaffold, Tier 1): same root
  cause.

A FOCUS.md operator dogfood note on #7451 documents this gap from the
operator side; this file is the design-side companion.

## Non-goals

The following changes are explicitly **out of scope** of the
implementation PR this design authorizes. Each is independently Tier 4
and would need its own pre-approval artifact:

- **No change to `_infer_model_reviewer_from_text`.** The recognizer
  stays at exactly the seven markers listed above. (Expanding the marker
  set is what PR #7450 separately authorizes.)
- **No change to `_normalize_model_reviewer_id`.** The 12-family
  normalization table stays as-is.
- **No change to `aragora-merge-quorum.yml`.** The merge-quorum
  evaluator is untouched. The current quorum-counting and dissent rules
  apply unchanged to the new per-family comments.
- **No change to the Tier-eligibility table or jurisdictional payload
  rules** in `docs/REVIEW_AUTHORITY_PRINCIPLES.md`. The advisory review
  already routes only PR title+diff (a payload-class explicitly
  permitted for Western families in the principles doc), and that does
  not change.
- **No widening of which families count at which Tier.** A Tier 0–2 PR
  that previously needed N recognized signals still needs N recognized
  signals; this PR just makes the existing advisory output recognizable
  so it can be one of them.

## Proposed contract

The advisory `aragora-review-gate.yml` workflow currently posts one
comment per PR (header `## Aragora Code Review`) summarizing the
combined findings of all participating model agents. The proposed
contract changes the *comment shape*, not the review behavior.

For each *participating model agent* that produced a per-agent finding
list in `review.json`, the workflow emits one PR comment whose:

- **first markdown heading** is exactly
  `## <Family> independent semantic review on head <full-SHA>`
  where `<Family>` is the recognizer-eligible family marker
  corresponding to the agent (e.g. `Claude` for `anthropic-api`,
  `OpenAI` … see "Agent-to-family mapping" below), and `<full-SHA>`
  is `$GITHUB_SHA` for the workflow run (the exact head the review
  scanned).
- **body** contains *only that agent's* findings (or an explicit "no
  findings" line). The summary preamble and severity formatting from
  the existing comment template are reused per agent.
- **first 200 characters** still include the family name (the
  recognizer's fallback path) — this is automatically satisfied by the
  heading-line contract above.
- **trailing footer** explicitly notes "Advisory-only — does not bypass
  the merge-quorum check; counted only when the merge-quorum evaluator
  resolves a recognized family marker on the exact head SHA."

If `review.json` does not contain per-agent attribution (the current
JSON shape merges findings), the implementation MUST add per-agent
attribution as a precondition. A workflow that fabricates a family-name
header without underlying per-agent provenance is a regression and
explicitly disallowed by this design.

If a participating agent produced *no* findings, the workflow still
emits the per-agent comment with a "no issues found" body, headed
identically. This makes the recognizer count it as one signal — which
is the correct accounting under Tier 0 (`1 independent model review
or dogfood note`) and Tier 1+ (`N model signals, at least one
adversarial`).

### Agent-to-family mapping

| `--agents` value | Recognized family name in heading |
| --- | --- |
| `anthropic-api` | `Claude` |
| `openai-api` | `OpenAI` *(not yet recognized — pending #7450)* |
| `gemini-api` | `Gemini` |
| `grok-api` | `Grok` |
| `codex-cli` | `Codex` |

Until #7450 lands, `openai-api`-produced comments are headed `OpenAI` but
still resolve to `unknown_model_reviewer` (because `openai` is not in
the recognizer's seven-marker tuple yet). The advisory comment is still
posted — visibility is preserved — it just does not count toward the
quorum until #7450 expands the recognizer.

This is the correct ordering: this PR makes per-family attribution
possible *first*, and #7450 (separately pre-approved by its own design
doc + governance tests) makes more of those families *count* second.
Reversing the order would expand the recognizer without ensuring real
attribution is plumbed through — exactly the silent-attribution-bug
class this whole gate is designed to prevent.

## Tier classification

Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change
governance` ("Loosening any of these constraints in CI … requires the
same preapproval discipline as the original addition"), this change
**is Tier 4** even though it does not modify Python code: it changes
what gets counted toward the model-quorum requirement that gates
`main`-protected merges. The implementation PR will need:

- this design doc, merged,
- the failing governance tests in
  `tests/governance/test_advisory_review_recognizable_header.py`,
  merged in this PR (they pin the current state as the regression floor),
- explicit operator Tier 4 preapproval in the implementation PR before
  the workflow change lands.

## Risk dimensions

| # | Risk | Mitigation in the proposed contract |
| --- | --- | --- |
| 1 | Mis-attribution: workflow stamps `## Claude …` on a comment that aggregates other agents' findings. | Per-agent provenance MUST come from `review.json`'s per-agent attribution; aggregation is forbidden. |
| 2 | Fabricated SHA: workflow stamps a SHA that wasn't actually reviewed (e.g., stale checkout). | Heading SHA MUST be `$GITHUB_SHA` for the workflow run (already-checked-out commit). The recognizer's `_is_comment_grounded_on_head` cross-checks against the actual head SHA at quorum-evaluation time; a mismatch is silently dropped. |
| 3 | Header injection: a model output includes literal text matching the recognized header pattern within its body. | The recognizer scans **only** the *first* line that starts with `#`. It never re-scans subheadings or body text; the 200-char fallback fires **only** when the body contains no `#` heading at all. The workflow always emits a structured first heading, so neither subheadings nor body text — including diff-quoted family names — can displace it. Pinned by `test_recognizer_only_scans_first_heading` and `test_diff_text_containing_family_name_in_body_does_not_resolve`. Existing behavior; no new exposure. |
| 4 | Counting the same advisory comment as multiple signals (re-runs, edits, force-pushes). | The merge-quorum evaluator already dedupes via `_known_model_reviewer_id` (one signal per family per head SHA). A force-push changes the head SHA and drops all prior signals — including the new ones — which is correct per the head-bound settlement principle. |
| 5 | Workflow-output spoofing by a malicious PR diff that includes a fake "## Claude independent semantic review on head abc …" in `pr.diff`. | The recognizer is run on PR-comment bodies, never on `pr.diff` contents. Diff text is review *input*, not review *output*. No new exposure. |
| 6 | "No findings" comment from a non-participating agent (e.g., agent crashed but the workflow still emits its per-agent comment). | The workflow MUST only emit a per-agent comment when `review.json` records that agent as having produced output (success or empty-findings). A crashed agent produces no comment — accurately reflecting that the agent did not actually review. |

## Failing governance tests (regression floor)

`tests/governance/test_advisory_review_recognizable_header.py` pins the
current state as the regression floor:

- The literal `## Aragora Code Review` header used by the current
  workflow resolves to `unknown_model_reviewer` (proves the gap is
  real, today).
- The proposed `## Claude independent semantic review on head <sha>`
  header resolves to `claude` (proves the recognizer already supports
  the proposed contract — *no recognizer change is needed*; this PR is
  pure workflow-side plumbing).
- Header injection in body text past the first heading is ignored
  (proves the recognizer is not loosened by this design).
- Empty / arbitrary-prose bodies stay unknown (proves the recognizer
  is not loosened by this design).

After the implementation PR lands and the workflow emits the new
headers, the "pinned current state" assertions become regression
guards: a regression that reverts the workflow to the `## Aragora Code
Review` shape would resurface the silent-quorum-suppression bug, and
the governance tests would flag it.

## Implementation PR plan (out of scope of this PR)

A future PR will:

1. Audit `aragora.cli.review review --output-format json` to confirm
   per-agent attribution is present in `review.json`. If absent, add
   it (this is itself a Tier 2 change in the CLI, separately scoped).
2. Modify `.github/workflows/aragora-review-gate.yml` to emit one
   comment per participating agent with the contract above.
3. Update the existing comment-update path (currently keyed by
   `startswith("## Aragora Code Review")`) to be keyed per family per
   head SHA (e.g., `startswith("## Claude independent semantic review
   on head <head-sha-prefix>")`) so re-runs update rather than
   duplicate.
4. Convert the failing governance tests in this PR's `tests/governance`
   to passing tests (or move them to a positive-recognition suite),
   per the regression-floor pattern described in
   `tests/governance/test_model_quorum_recognizer_gaps.py`.
5. Verify on a real PR (e.g., dogfood the implementation on the
   implementation PR itself) that the new comments count toward the
   merge-quorum and the quorum-evaluator log shows `1/1 signal(s)
   satisfied` instead of `0/1`.

## Operator preapproval requested

Approving this PR (and the failing governance tests) constitutes
preapproval to *draft* the implementation PR described above. The
implementation PR will require a *second* preapproval at the
implementation step per Tier 4 discipline.

This PR contains no live workflow change. Until the implementation PR
lands, the advisory review continues to post the current header and
continues to be counted as `unknown_model_reviewer`. PRs that need a
counted signal today still need to obtain one from a non-author
identity by another means (e.g., a focused dogfood post from a bot
identity, or running the recognized-family review CLI from a separate
GitHub login).
