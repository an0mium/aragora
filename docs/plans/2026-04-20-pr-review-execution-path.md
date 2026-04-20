# PR Review Execution Path

Last updated: 2026-04-20

Status: design doc for issue #6306

Related:
- [docs/plans/2026-04-19-pr-intelligence-brief.md](2026-04-19-pr-intelligence-brief.md)
- Issue #6306 — Implement PRReviewProtocol for heterogeneous settlement briefs
- Merged foundation PRs: #6355, #6359, #6353, #6328

## Purpose

This document defines the execution path for turning the existing
`metadata_heuristic` PR review packet into a real heterogeneous settlement
brief without weakening the human settlement gate.

The immediate goal is not "automated review comments." The goal is a bounded,
auditable, SHA-bound brief that the founder can settle from the browser or CLI
with explicit evidence, dissent, cost, and recommendation class.

## What Already Exists

Four prerequisite slices are already merged:

1. `PRReviewProtocol` packet scaffold and provider-slot vocabulary (`#6355`)
2. review policy, budget, and cost meter schemas (`#6359`)
3. brief receipt and settlement linkage schemas (`#6353`)
4. browser review-queue surface for human settlement (`#6328`)

`#6306` should compose these assets. It should not replace them.

## Scope

`#6306` should implement the first real reviewer execution path that:

- consumes the existing review queue packet inputs
- resolves provider slots to the latest strongest configured model per family
- runs a bounded heterogeneous review round-trip
- emits recommendation, findings, dissent, confidence, evidence, and cost
- serializes into the existing `PRReviewProtocol` shape
- writes receipt-compatible artifacts keyed to `(repo, pr_number, base_sha, head_sha)`

## Non-Goals

`#6306` must not:

- post GitHub approvals or request-changes automatically
- merge PRs or bypass the settlement gate
- create a new UI surface outside `/review-queue`
- introduce unconstrained multi-round debates by default
- treat one model family as a compliance oracle
- train on settlement outcomes or mutate prompts from production data

## Design Principles

1. Bounded by default.
   Default execution should finish in minutes, not tens of minutes.

2. Heterogeneous by family, not merely agent count.
   The panel should prefer materially different model families over replicas of
   one provider.

3. SHA-bound outputs.
   Every brief and receipt must bind to exact `repo`, `pr_number`, `base_sha`,
   and `head_sha` so stale packets are never silently reused.

4. Advisory only.
   The execution path can prepare settlement, not perform settlement.

5. Explicit disagreement.
   Dissent is a first-class output, not a hidden implementation detail.

## Default Execution Topology

### Default protocol

Phase A execution should use a bounded two-round protocol:

1. Round 1: independent findings
2. Round 2: cross-critique and revision
3. Synthesis: single primary synthesis into the packet

This corresponds to the "B" protocol in the prior planning work.

### Escalation path

Escalate from bounded review to deeper adversarial review only when risk rules
fire. Initial escalation triggers:

- protected or high-consequence paths
- broad blast radius paths
- large diff plus multiple subsystems together
- manual `escalate-pdb` label
- low confidence or high dissent from the bounded pass
- stale brief on new head SHA or repeated flaky/failing CI

Escalation should deepen review, not widen autonomy.

## Reviewer Model Topology

### Provider slots

The execution path should resolve provider slots rather than hardcoding exact
model names in the protocol implementation.

Initial slots:

- `claude_core`
- `gpt_core`
- `gemini_heterodox`
- `grok_heterodox`
- `deepseek_heterodox`
- `kimi_heterodox`
- `qwen_heterodox`
- `mistral_regulatory`
- `synthesizer_primary`
- `synthesizer_secondary` (escalation only)

### Slot policy

- `claude_core` and `gpt_core` are always included
- `mistral_regulatory` is framed as a regulatory or European lens, not as a
  compliance oracle
- `synthesizer_primary` defaults to the strongest configured Claude-family slot
- `synthesizer_secondary` defaults to the strongest configured GPT-family slot
  and runs only for escalated reviews

### Missing slots

If one or more non-core slots are unavailable, execution should degrade
gracefully and record the reduced roster in the packet and receipt. Missing
core slots should hard-fail execution unless the policy explicitly allows a
reduced panel.

## Reviewer Output Contract

Each reviewer should emit structured findings, not prose-only summaries.

Minimum reviewer payload:

- `recommendation_class`
- `confidence` in `[0.0, 1.0]`
- `top_findings[]`
- `evidence_refs[]`
- `risk_flags[]`
- `open_questions[]`
- `summary`

`top_findings[]` entries should include:

- `category`: logic, security, maintainability, skeptic, validation
- `severity`
- `claim`
- `evidence`
- `files[]`

## Reconciliation and Dissent

### Packet-level recommendation

Packet recommendation should not be a naive majority vote. Use this order:

1. hard blockers dominate
2. otherwise synthesis on the full findings panel
3. if synthesis materially diverges from reviewer majority, surface dissent
   explicitly instead of hiding the split

### Dissent rules

The packet should preserve:

- dissenting providers
- dissenting categories
- whether dissent is about severity, correctness, or uncertainty

At minimum, packet dissent should answer:

- who disagreed
- about what
- whether the disagreement blocks approval

### Escalated dual synthesis

For escalated reviews only, run a second synthesis through
`synthesizer_secondary`. If the two syntheses materially diverge on
recommendation class or top blocker, mark the packet `needs_human_attention`
and preserve both summaries in the receipt.

## Storage and Artifact Model

### Cache key

Execution outputs should be stored at a cache key equivalent to:

- `repo`
- `pr_number`
- `base_sha`
- `head_sha`

### Artifacts

Write three durable artifacts:

1. reviewer findings artifact
2. synthesized brief artifact
3. receipt extension artifact

The review queue UI and CLI should read the synthesized brief artifact. The
receipt path should read the receipt extension artifact.

### Reuse policy

If a matching artifact exists for the same `(repo, pr_number, base_sha, head_sha)`
and the policy version matches, reuse it. Any head SHA change invalidates the
brief for settlement purposes.

## Budget and Rate Limits

`#6306` must respect the merged policy and budget layer from `#6359`.

Budget controls should apply at:

- per PR
- per session or morning batch
- per provider family
- escalation eligibility

Minimum execution policy:

- refuse execution when policy budget is exhausted
- record partial execution if the roster is truncated by budget
- never silently downgrade from real heterogeneous review to a fake
  `metadata_heuristic` brief without marking the packet as degraded

## Failure Handling

Failure modes must be explicit:

- unavailable provider slot
- timeout
- partial panel completion
- synthesis failure
- stale packet reuse attempt

On partial failure:

- preserve completed reviewer outputs
- mark execution as degraded
- avoid inventing false consensus
- prefer `needs_human_attention` over false confidence

## UI and CLI Integration

The execution path should integrate with the already-merged review queue
surface, not create a second settlement path.

Expected behavior:

- CLI packet generation can request real execution instead of heuristic-only
- browser review queue reads the same packet and brief fields
- recommendation classes remain:
  - `approve_candidate`
  - `needs_human_attention`
  - `repair_first`

## Implementation Order

Recommended order for `#6306`:

1. provider-slot resolver and availability report
2. reviewer output schema and validation
3. bounded reviewer executor for one round
4. cross-critique round and aggregation
5. primary synthesis into `PRReviewProtocol`
6. cache and artifact persistence keyed by SHA
7. policy and budget gating
8. escalated dual-synthesis path
9. UI/CLI read-path hookup for real execution status

## Acceptance Criteria

`#6306` is done when:

- a real heterogeneous packet can be produced for Aragora PRs
- the packet binds to exact base and head SHAs
- dissent is explicit and preserved
- budget and degradation states are visible
- the review queue can display the resulting brief without changing the
  settlement boundary

## Open Questions

1. What is the threshold for "material divergence" between primary and
   secondary synthesis on escalated reviews?
2. Which provider-slot failures should hard-fail versus degrade?
3. Should batch execution reserve budget for escalations up front or consume on
   demand?
4. Which evidence refs are mandatory in the first real reviewer payload:
   file paths only, or file-plus-hunk-level anchors?
