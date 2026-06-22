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

### Prepared-artifact policy snapshot (replaces `tiered_gate: bool`)

A prepared evidence artifact stores a versioned `policy_snapshot` (policy version +
the resolved flag regime) rather than a bare boolean. At apply time the effective
policy is the **stricter** of the prepared snapshot and the live policy
(`stricter(prepared, live)`): relaxation requires *both* regimes to permit it.
This is fail-safe — evidence insufficient under the effective policy degrades to
"prepare", never a hard error — and it is the same monotonic-restrictive rule the
live merge gate would reach, removing the apparent live-vs-apply divergence. A
legacy artifact lacking the snapshot fails closed to the strict regime and logs.

`policy_snapshot` carries no authority the live flag does not already grant: a
forged snapshot cannot relax a merge while the live flag is OFF, and an artifact is
trusted only after it is matched to the live exact-head SHA and re-linted.

## Implementation Plan

1. Add `WESTERN_FAMILIES`, jurisdiction helpers, and `QuorumPolicy` (with the
   satisfaction predicate and a `stricter()` combinator) to `quorum_evidence.py`.
   Keep `tier_quorum_rule` as a thin shim returning the new policy for back-compat.
2. Route `review_queue._tier_requirement` / `_build_model_review_quorum` and
   `CollectOutcome.has_supportive_quorum` through `QuorumPolicy`.
3. Derive `merge_quorum_reconcile.TIER_REQUIREMENTS` from `QuorumPolicy` (or replace
   its lookups) so the diagnostic cannot drift.
4. Replace the serialized `tiered_gate` with `policy_snapshot` (back-compat read of
   the legacy boolean); apply path uses `stricter(prepared, live)`.
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
