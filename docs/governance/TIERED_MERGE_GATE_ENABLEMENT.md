# Tiered Merge Gate Enablement

This note records that the **tiered merge gate** is enabled in CI and describes its
effect. It complements the design in
[`docs/specs/TIERED_MERGE_GATE_QUORUM_POLICY.md`](../specs/TIERED_MERGE_GATE_QUORUM_POLICY.md)
and the jurisdiction rules in
[`docs/REVIEW_AUTHORITY_PRINCIPLES.md`](../REVIEW_AUTHORITY_PRINCIPLES.md).

## What is enabled

The `Evaluate merge quorum` step of
[`.github/workflows/aragora-merge-quorum.yml`](../../.github/workflows/aragora-merge-quorum.yml)
sets, alongside the existing `ARAGORA_ENABLE_SEVERITY_GATED_DISSENT: "1"`:

```yaml
ARAGORA_ENABLE_TIERED_MERGE_GATE: "1"
```

The flag is read live by `tier_quorum_rule(...)` in
`aragora/swarm/quorum_evidence.py` (see `tiered_merge_gate_enabled`), which is the single
source of truth for the live merge gate, the auto-settle collector, and the reconcile
diagnostic. No gate logic changed; this enablement only sets the input flag.

## Effect (Tier 1-2 only)

| Tier | Flag OFF (default)                                   | Flag ON (this enablement)                              |
|------|-----------------------------------------------------|--------------------------------------------------------|
| 1    | 2 distinct counting families (any family)           | **1 western-frontier signal** (`claude`/`openai`)      |
| 2    | 2 distinct families, at least one Western           | **1 western-frontier signal** (`claude`/`openai`)      |

The "western-frontier" set is the strict subset `{claude, openai}` of the Western family
set, so a cheaper model can never solo-authorize a merge.

## What is NOT affected

Tier 0, Tier 3, and Tier 4 are **unchanged** by this flag:

- **Tier 0** already requires one signal of any family; unchanged.
- **Tier 3-4** continue to require two distinct **Western** families (Western-only counted
  quorum) **plus** human settlement. The flag's relaxation branch is bounded to tiers 1-2;
  tiers 3-4 fall through to the Western-only fail-safe rule. The Tier 2 "at least one
  Western" and Tier 3-4 "Western-only counted" jurisdiction tightenings are applied
  unconditionally regardless of this flag.

Concretely, P4b/P5 server-handler PRs (Tier 3) still require per-PR operator human
settlement; this enablement does not change that.

## Why it is enabled

The P4a structural-refactor shim/move PRs hit a structurally unreachable Tier-1 model
quorum: the repo-grounded western-frontier reviewer (`claude`) PASSes with `would_count=true`
plus dogfood, but the only sanctioned second families decline illegitimately on the
intentional deprecation shims (an out-of-scope `[P2]` "migrate consumers off the shim", a
provably-false `[P1]` on byte-identical verbatim-moved code, or a non-converging hardening
loop). Severity-gated dissent does not help when those verdicts are `CHANGES-REQUESTED`, and
a human settlement is inert at Tier-1 `needs_model_review_quorum`. The operator chose
**Option A (enable the tiered merge gate)** on 2026-06-26 to let Tier 1-2 settle on the
genuine lone western-frontier signal. Full analysis: the mission's
`research/tier1-shim-gate-mechanism.md`.

## Reversibility

Default is OFF. The global flag is the revocation control: flipping the workflow env back to
OFF (or removing the line) restores the two-distinct-family bar for Tier 1-2 everywhere, with
no code change. Both enabling and reverting are Tier-4 merge-authority edits to the protected
workflow file and require operator settlement.
