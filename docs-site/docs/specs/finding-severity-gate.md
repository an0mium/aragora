---
title: Finding-Severity Dissent Gate (Tier 4 Pre-Approval)
description: Finding-Severity Dissent Gate (Tier 4 Pre-Approval)
---

# Finding-Severity Dissent Gate (Tier 4 Pre-Approval)

**Status:** design doc / pre-approval artifact for a Tier 4 merge-authority
self-modification. This file and
`tests/governance/test_finding_severity_gate.py` are the pre-approval artifact
required by
`docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`
(a change to *what blocks a merge at a given Tier* is a Tier 4 self-modification by
the same rule that governs which family counts at which Tier).

**Enforcement is opt-in and default-OFF.** With the flag OFF the gate behaves
byte-identically to today (proven by the flag-OFF characterization tests). The
behavior is revertible WITHOUT a code change — unset the environment flag. The flag
is the in-tree audit point for the operator's approval.

## Problem Statement

The model-quorum gate (`aragora-merge-quorum`) blocks a merge whenever a recognized
model-review comment carries a *blocking-or-negative* verdict. Today
`has_blocking_or_negative_verdict(body)` returns True for any of three triggers:

1. a real `[P0]`/`[P1]` finding line (head-before-colon NOT in `_NO_FINDING_HEADS`);
2. a negative `Verdict:/Decision:/Recommendation:` line
   (`fail`/`block`/`reject`/`changes-requested`/…);
3. a populated `Blocking finding(s):/Blocker(s):` label.

The severity scanner **already ignores `[P2]`/`[P3]` finding lines** — only `[P0]`
and `[P1]` lines (trigger 1) are recognized as findings. So the *only* thing that
lets a nitpick-only dissent block a merge is the bare negative verdict word
(trigger 2).

### The gap

A thorough reviewer who writes `Verdict: CHANGES-REQUESTED` while listing only
`[P2]`/`[P3]` nits blocks a merge **as hard as a `[P0]` defect**. Because the gate
re-checks the head SHA, any new commit invalidates prior evidence, so a
nitpick-only dissent on an otherwise-clean PR produces an endless head-drift
treadmill: the reviewer keeps re-raising the same low-severity nits against each new
head, and the PR can never settle. The negative *word* — not the finding *severity*
— is what blocks.

## Proposed behavior (opt-in)

Behind `ARAGORA_ENABLE_SEVERITY_GATED_DISSENT` (default OFF), a CHANGES-REQUESTED
comment promotes a **blocking** dissent only when it carries a real `[P0]`/`[P1]`
finding **or** a populated Blocker label. A `[P2]`/`[P3]`-only or finding-free
CHANGES-REQUESTED becomes **advisory**:

- **non-blocking** — it no longer trips `unresolved_dissent` (review-queue gate) and
  no longer flips the evidence collector to prepare-only (`quorum_evidence`); and
- **non-counting** — it does NOT satisfy the quorum. Advisory ≠ supportive: a
  downgraded comment is excluded from the supportive set, so it cannot help a PR
  settle either. It just stops blocking.

The advisory comment **still posts and stays visible** on the PR, and is recorded in
the merge packet (`advisory_views` field + a non-blocking `reasons` note like
`"advisory finding from <family>: no blocking [P0]/[P1] finding — not blocking
(severity-gated dissent)"`). Review quality is preserved and auditable; only the *blocking* effect
of a low-severity dissent is removed.

## Invariants

1. **Flag OFF = zero severity-gating behavior change, plus one disclosed security
   fix.** With the flag unset/falsey the severity-gating semantics are byte-identical
   to today — the existing review_queue / quorum_evidence suites stay green and the
   flag-OFF characterization tests in `tests/governance/test_finding_severity_gate.py`
   pin it. The **one** deliberate flag-independent change is a security fix to the
   pre-existing Blocker-label scanner: a populated Blocker label whose finding text
   begins with a bare "no" (e.g. `Blockers: no authentication on admin endpoint`)
   previously matched the non-blocking prefix `"no"` and was silently demoted to
   advisory — a merge-gate bypass for common security phrasing. It now correctly
   blocks (flag ON or OFF), while genuine no-finding declarations — including
   closed-allowlist adjective hedges (`no major concerns`, `no significant issues`,
   `no remaining blockers`) — stay non-blocking. The no-finding match is deliberately
   fail-closed: any value whose head word is a real subject (`no authentication`,
   `no validation`) or that carries a substantive finding (`no blocking on the auth
   path but SQLi`) blocks. The negative-verdict
   (`Verdict:`/`Decision:`/`Recommendation:`) scanner is unaffected: a *positive*
   verdict such as `Verdict: no concerns` remains non-blocking on the flag-OFF path.
2. **`[P0]`/`[P1]` and populated Blocker labels ALWAYS block**, flag ON or OFF.
3. **Advisory = non-blocking AND non-counting.** A downgraded dissent does not
   satisfy the quorum; `supportive` is unchanged, so it stays non-counting.
4. **The two gate halves stay in lockstep.** `review_queue._dissenting_views_from_comments`
   and `quorum_evidence.EvidenceItem.dissenting` consult the same shared helpers
   (`has_blocking_finding_or_label` / `highest_blocking_severity`), so the live merge
   gate and the auto-settle / evidence path cannot drift on the same comment body.

## Implementation

Four changes; flag OFF ⇒ identical behavior:

1. **Flag** `ARAGORA_ENABLE_SEVERITY_GATED_DISSENT` + helper
   `severity_gated_dissent_enabled(env=None)` in
   `aragora/swarm/quorum_evidence.py`, mirroring `tiered_merge_gate_enabled`
   exactly (same truthy set `{1, true, yes, on}`, default OFF).
2. **Shared helpers** in `aragora/cli/commands/review_queue_comment_verdicts.py`:
   - `highest_blocking_severity(body) -> "P0" | "P1" | None` — reuses the exact
     `[P0]`/`[P1]` detection (and `_NO_FINDING_HEADS`) that
     `has_blocking_or_negative_verdict` uses.
   - `has_blocking_finding_or_label(body) -> bool` — a real `[P0]`/`[P1]` finding OR
     a populated Blocker label; i.e. everything `has_blocking_or_negative_verdict`
     blocks on EXCEPT a bare negative verdict line with no finding/label.
   - `has_blocking_or_negative_verdict` is behavior-unchanged on the negative-verdict
     path (its internal helper constants/functions were lifted to module scope so the
     new helpers share the single source of regex/`_NO_FINDING_HEADS` truth). The only
     intended return-value change is the disclosed Blocker-label security fix above:
     `_NO_FINDING_NO_PHRASE` ("no issues"/"no blockers"/…) is consulted **only** by the
     Blocker-label non-blocking check (`_starts_with_phrase(..., match_no_finding=True)`),
     never by the negative-verdict check, so positive verdicts like `Verdict: no
     concerns` are not promoted to blocking dissents.
3. **`_dissenting_views_from_comments`** (review_queue) consults
   `has_blocking_finding_or_label` when the flag is ON (dropping the bare-negative-
   verdict trigger) and `has_blocking_or_negative_verdict` when OFF. Downgraded
   comments are recorded in `advisory_views` and surfaced as a non-blocking
   `reasons` note.
4. **`EvidenceItem.dissenting`** (quorum_evidence) is `dissenting` only when
   `verdict == "changes_requested" AND has_blocking_finding_or_label(body)` while the
   flag is ON — i.e. a real `[P0]`/`[P1]` finding OR a populated Blocker label (the
   latter blocks even without a `[P0]`/`[P1]` marker, matching the review-queue half;
   this is strictly broader than `highest_blocking_severity(body) is not None`, which
   only detects `[P0]`/`[P1]`). OFF ⇒ `dissenting = (verdict == "changes_requested")`
   as today. The flag is captured once at construction (`severity_gated` field) and —
   like `CollectOutcome.tiered_gate` — is serialized into prepared artifacts and
   reconciled `effective = prepared AND live` at apply, so a gate decision stays
   deterministic and cannot be relaxed by flipping the env between prepare and apply.
   `supportive` is unchanged.

## Rollout

Phase 1 of the nitpick-vs-real plan: land the opt-in flag (default OFF) plus
pre-approval artifacts, so the operator can enable it per the governance contract.
No CI default is changed by this PR.
