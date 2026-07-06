---
title: Tiered Merge-Gate Quorum Policy (Tier 4 Pre-Approval)
description: Tiered Merge-Gate Quorum Policy (Tier 4 Pre-Approval)
---

# Tiered Merge-Gate Quorum Policy (Tier 4 Pre-Approval)

**Status:** design doc / pre-approval artifact for the Tier 4 merge-authority
self-modification on PR #8507. This file and
`tests/governance/test_tiered_merge_gate_quorum_policy.py` are the pre-approval
artifact required by
`docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`
(a change to *which family counts at which Tier* is Tier 4).

## Problem Statement

The model-quorum gate (`aragora-merge-quorum`) decides, per Tier, how many
distinct model-review families must support a PR before it can settle. Today that
decision is encoded in **three** places that have drifted apart, and **none** of
them implements the jurisdiction rules that
`docs/REVIEW_AUTHORITY_PRINCIPLES.md::Tier-eligibility for quorum counting`
already mandates:

1. `aragora/cli/commands/review_queue.py::_tier_requirement` /
   `_build_model_review_quorum` — the live CI merge gate.
2. `aragora/swarm/quorum_evidence.py::tier_quorum_rule` /
   `CollectOutcome.has_supportive_quorum` — the auto-settle / evidence path.
3. `aragora/swarm/merge_quorum_reconcile.py::TIER_REQUIREMENTS` — a read-only
   diagnostic table (non-gating) that renders the "next action" hint.

### Gaps this change inverts

- **G1 — Tier 3-4 are not Western-only-counted.** The spec requires the *entire*
  counted quorum at Tier 4 (and at Tier 3) to come from Western families
  (Anthropic, OpenAI, Google, xAI, Mistral, Nous Hermes). The current code counts
  any two distinct families, so `claude + deepseek` (one Western, one Chinese-routed)
  wrongly satisfies a Tier 3-4 merge. `quorum_evidence.py` even documents this as
  intended ("lets cheap, distinct families (e.g. claude + deepseek/qwen/kimi) form
  a 2-family quorum"). That is correct for Tier 0-1, wrong for Tier 3-4.
- **G2 — Tier 2 does not require at least one Western family.** The spec requires
  Tier 2 to have "at least one of the two required signals … a Western family." The
  current rule is any two distinct.
- **G3 — Three independent encodings.** The diagnostic table and the two gating
  paths can drift; reviewers repeatedly flag this as apparent inconsistency.
- **G4 — `tiered_gate: bool` is a lossy artifact regime marker.** A prepared
  evidence artifact stores only a boolean, and the apply path reconciles it against
  the live flag via `min(prepared, live)` while the live merge gate reads the live
  flag directly. The asymmetry is *fail-safe* (apply is never more permissive than
  the live gate) but undocumented and easy to misread.

The Tier 1-2 single-western-frontier-signal relaxation that PR #8507 introduces is
**retained** and is the feature this change ships; it is the opt-in, default-OFF
`ARAGORA_ENABLE_TIERED_MERGE_GATE` behavior. This redesign places that relaxation
inside one jurisdiction-aware policy object alongside the spec's Tier 2-4 rules so
the relaxation cannot be read in isolation from the constraints around it.

## Counting Contract

A single canonical `QuorumPolicy` (in `aragora/swarm/quorum_evidence.py`, the
module both gating paths already import) is the **only** source of the per-Tier
bar. `review_queue`, `quorum_evidence`, and `merge_quorum_reconcile` all derive
from it; the diagnostic table is computed from it, not hand-maintained.

### Family jurisdiction sets

- `WESTERN_FAMILIES = {claude, openai, gemini, grok, mistral, hermes}` — the
  spec's Western lineages (Anthropic, OpenAI, Google, xAI, Mistral, Nous Hermes).
- `WESTERN_FRONTIER_FAMILIES = {claude, openai}` — a strict **subset** of
  `WESTERN_FAMILIES`: the frontier labs whose single signal may solo-authorize a
  Tier 1-2 relaxed merge. "Frontier" (who can settle Tier 1-2 alone) and "Western"
  (who counts at Tier 3-4) are distinct concepts; the subset relation is asserted
  by a governance test so they cannot diverge into incompatibility.
- Chinese-routed families (`deepseek, qwen, kimi, glm, minimax, yi`) are everything
  in the recognizer that is not Western. They always post and remain readable; they
  are **advisory-only (not counted)** at Tier 3-4 and do not satisfy the
  at-least-one-Western condition at Tier 2.

### Per-Tier policy (matches REVIEW_AUTHORITY_PRINCIPLES.md)

| Tier | `tiered_gate` ON | required signals | western-only counted | ≥1 Western | frontier single-signal |
| --- | --- | --- | --- | --- | --- |
| ≤0 | — | 1 | no | no | no |
| 1 | OFF | 2 | no | no | no |
| 1 | ON | 1 | no | no | **yes** |
| 2 | OFF | 2 | no | **yes** | no |
| 2 | ON | 1 | no | no | **yes** |
| 3 | any | 2 | **yes** | (implied) | no |
| 4 | any | 2 | **yes** | (implied) | no |
| unknown/None | any | 2 | **yes** (fail-safe) | (implied) | no |

Satisfaction predicate over the supportive families `S`:

```
counted = S minus Chinese families            if western_only_counted else S
if requires_western_frontier and not (counted ∩ WESTERN_FRONTIER_FAMILIES): fail
if requires_at_least_one_western and not (counted ∩ WESTERN_FAMILIES):      fail
return len(counted) >= required_signals
```

The Tier 1-2 relaxation (`tiered_gate` ON) never lowers the bar below "one
*western-frontier* signal," so a Chinese-routed family can never solo-authorize a
merge at any Tier, and the Tier 3-4 western-only rule is independent of the flag.

### Prepared-artifact regime + version stamp

A prepared evidence artifact stores the boolean `tiered_gate` regime it was collected
under, plus a `policy_version` stamp (`QUORUM_POLICY_VERSION`). At apply time the
effective regime is the **stricter** of the prepared and live regimes — for the single
boolean relaxation dimension this is exactly `effective = prepared.tiered_gate AND
live_gate`: relaxation requires *both* regimes to permit it. This is fail-safe —
evidence insufficient under the effective regime degrades to "prepare", never a hard
error — and reaches the same monotonic-restrictive result the live merge gate would,
removing the apparent live-vs-apply divergence. A legacy artifact lacking the field
fails closed to the strict regime and logs.

The `tiered_gate` regime carries no authority the live flag does not already grant: a
forged `tiered_gate=true` cannot relax a merge while the live flag is OFF, and an
artifact is trusted only after it is matched to the live exact-head SHA and re-linted.
`policy_version` is a **forward-compat audit stamp** — it records which policy encoding
produced the artifact so a future migration can detect a stale one; it is not itself an
apply-time gate today (the boolean `tiered_gate` reconciliation is). The jurisdiction
rules (Tier 2 ≥1-Western, Tier 3-4 Western-only) are not part of the prepared regime:
they are unconditional and re-derived live from the PR's tier at both collect and
apply, so they cannot go stale.

## Implementation Plan (as shipped)

1. Add `WESTERN_FAMILIES`, `is_western_family`, and the jurisdiction fields +
   `is_satisfied_by`/`counted_families` predicate to `TierQuorumRule` (aliased as
   `QuorumPolicy`) in `quorum_evidence.py`. `tier_quorum_rule` remains the single
   source of truth and encodes the full per-Tier table.
2. Route `review_queue._tier_requirement` / `_build_model_review_quorum` and
   `CollectOutcome.has_supportive_quorum` through that policy (the gate reads the
   jurisdiction fields; `WESTERN_FAMILIES` is re-exported, not duplicated).
3. Keep `merge_quorum_reconcile.TIER_REQUIREMENTS` a literal (a module-load
   derivation would be a circular import via `merge_quorum_io`) and pin it to the
   policy with `test_reconcile_diagnostic_matches_policy` so it cannot drift.
4. Keep the serialized boolean `tiered_gate` regime and add a `policy_version`
   forward-compat stamp; the apply path reconciles via the stricter regime
   (`effective = prepared.tiered_gate AND live_gate`) — functionally the
   monotonic-restrictive `stricter(prepared, live)`, expressed for the single
   boolean dimension. (A full serialized `policy_snapshot` object was considered
   and rejected as redundant: the jurisdiction rules are unconditional and
   re-derived from tier, so only the relaxation flag needs to travel on the artifact.)
5. Governance + unit tests green; re-collect Western quorum; Tier-4 human settlement.

## Governance Test Mapping

`tests/governance/test_tiered_merge_gate_quorum_policy.py` is the regression target.
It first **characterizes** the inverted gaps (G1/G2 — that the new policy refuses a
`claude+deepseek` Tier-4 quorum and requires ≥1 Western at Tier 2), pins the
single-source-of-truth (G3 — all three paths agree per Tier), the frontier⊂Western
subset invariant, and the `stricter(prepared, live)` regime semantics (G4). Per
`REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`, these tests are
the machine-checkable floor for this Tier 4 change and must accompany it through
human settlement.
