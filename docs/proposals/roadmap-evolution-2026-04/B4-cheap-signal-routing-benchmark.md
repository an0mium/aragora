# B4 extension — Cheap-Signal-to-Verification Routing as Benchmarked Decision Policy

**Status:** Planning truth only. **Not `boss-ready`** until the proof-first Foreman gate opens per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md).
**Labels:** `track:B` `area:benchmark` `area:routing` `type:benchmarked-decision-policy` `boss-ready:no-until-foreman-gate`

## What this is

A narrow **benchmarked decision-policy experiment** under existing [Track B4 (Multi-Host Soak and Unattended Criteria)](../../plans/ARAGORA_EVOLUTION_ROADMAP.md). Not a new track, subsystem, or philosophy — it is evaluated on the **existing bounded corpus** the Track B stack already uses.

## Question

On the existing bounded-autonomy benchmark corpus, does routing by cheap signal (single-agent inference + disagreement-feature heuristics) and escalating to full adversarial debate only on high-value or high-disagreement inputs produce:

(a) measured cost reduction vs. always-debate,
(b) no regression in rescue rate or verification pass rate vs. always-debate, and
(c) an auditable routing-decision receipt for every cheap-route call?

If (a) holds with (b) and (c) intact, cheap-signal routing becomes a legitimate production policy for low-stakes decisions. If not, the policy is rejected and the benchmark artifact documents why.

## Scope (bounded)

1. Labelled comparison harness under `benchmarks/b4_cheap_signal_routing/` using the existing Track B corpus.
2. Three arms: `always_debate` (baseline), `cheap_only` (lower bound on accuracy cost), `routed` (cheap + escalate on high-disagreement).
3. Receipt schema extension for the routing decision itself (question, cheap signal, disagreement features, routing decision, rationale). Landed only in the benchmark harness, **not** in the production admission gate.

## Acceptance criteria

- [ ] Harness + corpus-loader land under `benchmarks/b4_cheap_signal_routing/`
- [ ] Benchmark artifact: `benchmarks/b4_cheap_signal_routing/REPORT_<date>.md` with (a) cost, (b) rescue rate + verification pass rate, (c) a sample of routing-decision receipts
- [ ] No production-path changes to `aragora/debate/orchestrator.py`, `aragora/workflow/`, or the admission gate
- [ ] Report concludes with one of: **all three conditions met → propose a narrow production wiring behind a feature flag under B4**; **any condition fails → close and document**

## Explicitly out of scope

- Any production admission-gate change (the retired H3/A6 surface)
- Any surrogate training corpus (the retired H1 surface)
- Any surrogate model zoo
- Any out-of-distribution guardrail as a new subsystem
- Any per-domain surrogate ceiling governance surface

## Foreman-gate posture

Per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md), **the current obligation is operationalizing the proof-first loop, not adding new roadmap scope**. This issue is a **benchmark-only experiment** whose artifact is a report. It **must not** carry `boss-ready` or modify any production admission path until the proof-first Foreman gate opens.

## Related

- [Track B4 in ARAGORA_EVOLUTION_ROADMAP.md](../../plans/ARAGORA_EVOLUTION_ROADMAP.md)
- [NEXT_STEPS_CANONICAL.md proof-first gate](../../status/NEXT_STEPS_CANONICAL.md)
- Existing bounded-autonomy benchmark corpus used by Track B
