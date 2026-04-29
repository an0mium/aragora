# PR #6795 — independent recommendation

**Date:** 2026-04-28
**PR:** [#6795 feat(review-queue): add model quorum merge packets](https://github.com/synaptent/aragora/pull/6795)
**Head SHA:** `d032987fa18b`
**Author:** `an0mium`  |  **Branch:** `codex/model-review-quorum-settlement`
**Diff:** +983 / -7  |  **Required checks:** all SUCCESS  |  **Mergeable:** BLOCKED on REVIEW_REQUIRED

## Recommendation: **HOLD.**  Do not authorize admin-squash yet.

I agree with Codex's analysis and add four further findings below.  The PR
is the right shape and direction, but it ships a quorum function that
**doesn't enforce the heterogeneity property the PR itself defines.**
Merging it as-is encodes a loophole into the review-authority mechanism
that this PR is meant to harden.

I traced the quorum logic against #6795's *own* comment thread.  The
result is consistent with Codex's reading.

---

## Independent reproduction of Codex's two findings

### Finding 1 (Codex): quorum counts raw signal length, not distinct models

In `_build_model_review_quorum()`:

```python
signal_count = len(reviewer_signals) + len(dogfood_evidence)
quorum_satisfied = (
    signal_count >= requirement["required_model_signals"] and has_required_dogfood
)
```

Two `Codex review` comments produce `signal_count == 2`.  This satisfies
Tier 2's "2 heterogeneous model signals" requirement even though the
underlying set of distinct models is `{codex}`.  The function never
projects the signals through `set(reviewer_id)`.  **Confirmed.**

### Finding 2 (Codex): `unknown_model_reviewer` dogfood inflates quorum

`_model_review_signals_from_comments` *does* skip
`unknown_model_reviewer` (line ~503 of the diff).  But
`_dogfood_evidence_from_comments` does **not** apply the same filter —
it stores `unknown_model_reviewer` as a valid `reviewer_id` in the
returned evidence list, which then contributes to `signal_count` via
`len(dogfood_evidence)`.  **Confirmed.**

### Trace against #6795's own comment thread

| # | Author | Has dogfood marker? | Has review marker? | Inferred model | Counted as |
|---|--------|---------------------|--------------------|----------------|------------|
| 0 | `an0mium` | yes (`dogfood`) | no | `unknown_model_reviewer` | dogfood_evidence (would be excluded by Codex's fix) |
| 1 | `github-actions` | no | no (`Aragora Code Review` header but body doesn't trigger) | n/a | filtered |
| 2 | `an0mium` | yes | yes (`independent semantic review`) | `codex` | reviewer_signals + dogfood_evidence (double-counted) |
| 3 | `an0mium` | yes (`post-rebase dogfood`) | no | `codex` (matched on the branch name `codex/...`) | dogfood_evidence |

`reviewer_signals = [codex]` (length 1).
`dogfood_evidence = [unknown_model_reviewer, codex, codex]` (length 3).
`signal_count = 4`, requirement is 2 → **`quorum_satisfied = True`** →
**`verdict = "admin_squash_allowed"`** for this PR.

Distinct known reviewer IDs across all signals: `{codex}`.  That is
**one** heterogeneous model, not two.  The PR's `Tier 2` table in
`docs/REVIEW_AUTHORITY_PRINCIPLES.md` explicitly requires "2
heterogeneous model signals".  The algorithm's verdict contradicts the
PR's own principles file.

---

## Four additional findings

### Finding 3 (new): a single comment is double-counted

Comment[2] above contains both an "independent semantic review" marker
and a "dogfood" marker.  It is added to `reviewer_signals` *and* to
`dogfood_evidence`.  `signal_count = len(reviewer_signals) +
len(dogfood_evidence)` therefore counts that one comment twice.

A natural reading of the principle is that a single comment proves at
most one signal class.  Recommended fix: union-by-(reviewer_id, source)
before counting.

### Finding 4 (new): `_infer_model_reviewer_from_text` over-matches

```python
def _infer_model_reviewer_from_text(text: str) -> str:
    lower = text.lower()
    for name in ("claude", "codex", "tesla", "harvey", "factory", "grok", "gemini"):
        if name in lower:
            return name
    return "unknown_model_reviewer"
```

Substring match against the entire body.  Comment[3] is a rebase note
(no model review at all) but the algorithm tagged it as a `codex`
dogfood **because the branch name `codex/model-review-quorum-settlement`
appears in the body**.  Any PR with `codex/` or `claude-code/` branch
references in its comments inherits a phantom `codex`/`claude` signal.

Recommended fix: require the model name to appear in a structured
header, a known prefix line ("**Reviewer:** codex"), or constrain
matching to the first 200 characters of the comment.

### Finding 5 (new): no SHA-grounding on comments

`docs/REVIEW_AUTHORITY_PRINCIPLES.md` (line 728 of the diff) lists
"grounded in the current head SHA" as a heterogeneity requirement.  The
implementation never checks comment timestamps against
`pr['headRefOid']`.  A force-push that re-writes the diff after the
review comments were posted leaves those comments counting toward
quorum.  This is the same class of trust failure that the head-SHA-pin
on the merge command itself is meant to prevent.

Recommended fix: only count comments whose `createdAt` is at or after
the most recent push that produced the current head SHA, OR require
comments to explicitly cite the head SHA prefix.

### Finding 6 (new): self-review circularity for #6795

The PR's own self-packet uses `_build_model_review_quorum` against
`#6795`'s comment thread.  The two engineering authors contributing to
that thread are `an0mium` (PR author) and the Codex CLI driving the
reviews.  Codex's adversarial review of #6795 (the message I'm
reviewing right now) is a *separate* signal from the inline comment
thread — but the PR's algorithm has no way to ingest it because it
walks `pr.get("comments")` only.

Even after the quorum fix, before merge we should have one comment
on the PR itself from a non-Codex independent reviewer (`grok`,
`claude`, or a human reviewer) that posts to the PR thread so the
fixed quorum logic actually sees a heterogeneous set.

---

## What I would require before authorizing admin-squash

1. **Quorum based on `distinct` known reviewer IDs.**  Project signals
   through `set(s["reviewer_id"] for s in signals if reviewer_id !=
   "unknown_model_reviewer")` and require `len(distinct) >=
   required_model_signals`.
2. **Exclude `unknown_model_reviewer` from dogfood evidence.**  Codex's
   recommendation; one-line change in `_dogfood_evidence_from_comments`.
3. **De-double-count single comments.**  A comment should contribute
   one signal class at most.
4. **Tighten `_infer_model_reviewer_from_text`.**  Either constrain the
   match window or require a structured "Reviewer:" / "## Codex
   review" header.
5. **Optional but recommended: SHA-ground the comments** — only count
   comments posted after the most recent force-push.
6. **Tests proving the failure modes**:
   - duplicate `Codex review` comments do not satisfy Tier 2;
   - `unknown_model_reviewer` dogfood does not satisfy Tier 0;
   - `codex` dogfood + `grok` review *does* satisfy Tier 2;
   - same-comment double-counting cannot satisfy quorum.
7. **One non-Codex independent comment on #6795 itself** (so the fixed
   quorum logic sees `{codex, grok}` or `{codex, claude}` on its own
   PR thread).

After those land, I would support admin-squash for #6795.  The
direction (codified principles + machine-checkable packets) is correct
and the rest of the diff (CLI surface, docs, tests) is well structured.

---

## What I am *not* recommending

- I am **not** recommending closing #6795.  The PR is the right shape
  and is close to landing.
- I am **not** recommending a rebase-and-redo cycle.  The fix list is
  small and additive — a follow-up commit on the same branch is
  sufficient.
- I am **not** disputing that `Quality Gates`, `Epistemic Hygiene +
  Settlement Gate`, `V1 Scope Lock Gate`, and the test-fast matrix all
  pass.  They do.  This is a *semantic* gap in the merge-authority
  algorithm that the gates aren't designed to catch.

## Bottom line

**Hold #6795.**  Patch the quorum logic per Codex's plan plus the four
additional findings above, gather one non-Codex independent review or
dogfood comment, then admin-squash.  Merging as-is would ship a
self-defeating merge gate.
