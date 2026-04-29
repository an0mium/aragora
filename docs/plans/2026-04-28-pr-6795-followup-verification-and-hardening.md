# 2026-04-28 — PR #6795 follow-up: verification and hardening

**Branch:** `claude/merge-authority-followup-20260428`
**Author:** Claude (this run)
**Scope:** independent recheck of the parallel session's HOLD recommendation
on PR #6795 against the actually-merged head SHA, plus two follow-up
hardenings of the model-quorum gate.

This artifact dogfoods the new `_build_model_review_quorum` machinery on the
PR that introduced it: the parallel session's findings are re-walked against
the merged code, four are confirmed addressed pre-merge, two are extended into
this PR.

---

## Background

PR #6795 (`feat(review-queue): add model quorum merge packets`) merged at
commit `6bd234b93` on origin/main. A parallel session writing under
`docs/plans/2026-04-28-pr-6795-recommendation.md` had recommended **HOLD**
based on six findings against the head SHA `d032987f` it observed at the time
of writing. Between `d032987f` and the merged head `9420ea66`, additional
commits landed that addressed Findings 1, 2, 3, 4, and 5 directly.

This artifact:

1. Verifies each of the six findings against the merged code.
2. Closes Findings 2 and 6 with code changes (the remaining gaps).
3. Records the verification trail so a future reviewer can see the merge was
   safe even though the HOLD was filed against a stale snapshot.

---

## Verification matrix

For each finding, the merged-code citation and current status:

| # | Original claim (parallel session) | Status in `6bd234b93` | Citation |
|---|---|---|---|
| 1 | `signal_count = len(reviewer_signals) + len(dogfood_evidence)` counts raw signals, not distinct models | **Fixed pre-merge** | `aragora/cli/commands/review_queue.py:1194-1196`: `counted_reviewer_ids = _counted_model_reviewer_ids(...); signal_count = len(counted_reviewer_ids)`; `_counted_model_reviewer_ids` (line 1370) builds `set[str]` |
| 2 | `_dogfood_evidence_from_comments` does not skip `unknown_model_reviewer` (signals path does) | **Partial: neutralised at counting boundary, but inconsistent with signals path; cleaned in this PR** | `_known_model_reviewer_id` (line 1411) returns `""` for unknowns, so they don't increment `counted_reviewer_ids`. But the dogfood evidence list itself still contained them. Source-side filter added in this PR. |
| 3 | A single comment matching both signal and dogfood markers is double-counted | **Fixed pre-merge** | Same set-dedup as #1; a single reviewer ID contributes one entry regardless of how many lists carry it |
| 4 | `_infer_model_reviewer_from_text` substring-matches the entire body, so `codex/...` branch references in quoted text count | **Fixed pre-merge** | `_infer_model_reviewer_from_text` (line 1522) restricts the match to the comment's first markdown heading or first 200 chars |
| 5 | No SHA-grounding on comments; force-push leaves stale comments in the quorum | **Fixed pre-merge** | `_is_comment_grounded_on_head` (line 1487) requires either `head_sha[:7]` citation in body or `createdAt >= head_committed_at`. Both `_dogfood_evidence_from_comments` and `_model_review_signals_from_comments` call it before counting |
| 6 | Self-review circularity: a PR modifying the merge-authority logic is gated by the version of the gate it tries to land | **Real, unfixed; closed in this PR** | The merged code routes `aragora/cli/commands/review_queue.py` through Tier 2, which permits admin-squash on quorum + dogfood. This PR elevates it to Tier 4 (human preapproval). |

The HOLD recommendation was therefore correct *at the SHA it observed*, and
five of the six findings were closed by commits between `d032987f` and
`9420ea66`. The merge of #6795 was safe modulo Findings 2 (cosmetic at the
counting boundary, real for downstream consumers) and 6 (real). Both are
addressed in this PR.

---

## Changes in this PR

### 1. Finding 6 — merge-authority self-modification → Tier 4

`aragora/cli/commands/review_queue.py` is now in `TIER_4_PREFIXES`. Any future
PR that modifies the model-quorum gate code itself routes to
`tier_4_preapproval_required`, which requires explicit human approval before
merge.

```python
TIER_4_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "deploy/",
    "docker/",
    "k8s/",
    # Merge-authority self-modification: when a PR changes the code that
    # enforces model-quorum settlement gates, that PR's own quorum is
    # evaluated by the version of the gate it is trying to land. A bug or
    # weakening introduced in the diff would let the diff itself through.
    # Elevate to Tier 4 (human preapproval) so the human chain-of-trust is
    # not delegated to the artifact under review.
    "aragora/cli/commands/review_queue.py",
)
```

The principles file (`docs/REVIEW_AUTHORITY_PRINCIPLES.md`) is updated to
make the rule explicit in the Tier 4 row.

### 2. Finding 2 — symmetric source-side filter on `_dogfood_evidence_from_comments`

The function now mirrors `_model_review_signals_from_comments`:

- Skips comments where `_infer_model_reviewer_from_text` returns
  `unknown_model_reviewer`.
- Skips comments authored by `github-actions`.

Before this change the unknowns were neutralised at counting time (because
`_known_model_reviewer_id` returned `""` for them), but they still appeared
in `dogfood_evidence` — misleading downstream consumers reading the packet's
`dogfood_evidence` field as the canonical list of focused-adversarial work.

### 3. Tests

Four new regression tests in `tests/cli/commands/test_review_queue.py` under
`TestModelReviewQuorum`:

- `test_review_queue_self_modification_classified_tier_four` — `review_queue.py` → Tier 4
- `test_tier_four_review_queue_blocks_admin_squash_even_with_full_quorum` — full quorum + dogfood still blocks; verdict is `tier_4_human_preapproval_required`
- `test_dogfood_with_unknown_model_is_excluded_at_source` — Finding 2 fix
- `test_dogfood_from_github_actions_is_excluded_at_source` — Finding 2 fix

Existing tests that used `aragora/cli/commands/review_queue.py` as a generic
Tier 2 fixture have been migrated to `aragora/cli/commands/swarm.py`, which
remains a Tier 2 path. The tier-classification parametrize table now
explicitly asserts `review_queue.py → 4`.

---

## What this PR does *not* do

- It does not retroactively reopen #6795. The merged code is correct on
  Findings 1, 3, 4, 5; the gaps closed here are additive hardening.
- It does not change the heterogeneity contract for any other tier.
- It does not introduce a new mechanism for "self-modifying merge-authority
  PRs" — it reuses the existing Tier 4 path. Future PRs touching
  `review_queue.py` can still land; they just route through the human
  preapproval contract that already exists for workflows/deploy/k8s.

---

## Settlement guidance for this PR

This PR itself modifies `aragora/cli/commands/review_queue.py`. **Under the
rule it introduces, this PR's own classification is Tier 4: human preapproval
required.** That is intentional. The first PR to elevate the file to Tier 4
must itself be settled by a human; that establishes the chain of trust that
all subsequent merge-authority PRs will follow.

The packet for this PR will report `verdict = tier_4_human_preapproval_required`
even with full model quorum and dogfood, exactly as the new test
`test_tier_four_review_queue_blocks_admin_squash_even_with_full_quorum`
asserts.
