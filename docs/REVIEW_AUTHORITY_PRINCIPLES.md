# Review Authority Principles

Review authority in Aragora rests on four factors: competence, independence, accountability, and stake. An approval matters only when the approver can judge the change, is not merely echoing the system being judged, bears responsibility for the outcome, and is exposed to the consequences of being wrong.

This document describes the principles underlying current review authority. It does not authorize bot-only GitHub approvals. It does authorize a narrower distinction that is now operationally important: a heterogeneous model quorum can provide the technical review evidence for routine, reversible PRs, while the human/operator remains the accountable risk settler for strategy, high-impact, irreversible, or low-confidence decisions.

Competence in this repo has two forms. Object-level competence is direct judgment about code, tests, and failure modes in a specific diff. Governance competence is judgment about the evidence around a diff: whether the scope is bounded, whether validation is credible, whether competing analyses agree, and whether the claimed risk matches the actual change. Aragora uses both forms. The system prepares evidence, but settlement still depends on a reviewer who can cash that evidence out into a real decision.

The founder role in this workflow is governance competence plus accountability and stake, not object-level competence on every line of every PR. That role is still real. It is the place where bounded evidence, receipts, and competing analysis become an accepted or rejected risk. A founder settlement can therefore be batch-level risk acceptance when the technical packet is strong enough; it does not need to be a duplicative line-by-line code review.

Current AI reviewers in this codebase have not demonstrated sufficient calibration to replace human risk settlement for escalated classes. They are, however, the right technical reviewers for routine bounded work when they are heterogeneous, independent of the authoring lane, grounded in the current head SHA, and adversarial enough to preserve dissent. The packet must show the quorum, not merely assert it.

## Model Review Quorum

A model review quorum is satisfied only when the review artifact records:

- the exact PR head SHA reviewed
- reviewer/model identities or provider families
- whether the reviewer was independent from the authoring lane
- the recommendation and any dissent
- concrete validation or dogfood evidence
- the merge tier and resulting settlement requirement

Machine reviews remain advisory in GitHub terms: they do not become bot `APPROVE` reviews. For low-risk tiers, a receipt-backed model quorum can make admin squash a documented settlement action instead of an ad hoc bypass. For high-risk tiers, the same quorum prepares the risk packet, but the human still explicitly accepts or rejects the risk before merge.

## Merge Tiers

| Tier | Class | Requirement | Settlement |
| --- | --- | --- | --- |
| 0 | Docs-only, tests-only, status/report PRs | Green required checks plus 1 independent model review or dogfood note | Admin squash allowed |
| 1 | Additive internal code with no live caller and no persistence/security/public API effect | Green checks plus 2 model signals, at least one adversarial or dogfood signal | Admin squash allowed |
| 2 | Live automation, CLI, observability, retry/cache behavior | Green checks plus 2 heterogeneous model signals, focused dogfood, and no unresolved dissent | Admin squash allowed |
| 3 | Semantic correctness, persistence, reputation, security/RBAC/auth, public API, SDK, migrations | Model quorum plus explicit human risk settlement | Human risk acceptance required |
| 4 | Secrets, deployment, workflow policy, destructive operations, legal/compliance, irreversible data changes, **merge-authority self-modifications** (changes to the model-quorum gate code itself) | Human approval before implementation and before merge | Human preapproval required |

A change to `aragora/cli/commands/review_queue.py` is treated as Tier 4 because the model-quorum logic that gates the change *is* the code being changed. Without the elevation, a bug or weakening introduced in the diff would be evaluated by the version of the gate it is trying to land — the artifact under review would be its own arbiter. Tier 4 keeps the human in the chain of trust for these PRs specifically.

These principles fit the repo's existing pillars rather than adding new ones. Receipts bind decisions to the exact reviewed state. Evidence-first review requires concrete validation and current-head truth before a merge decision is meaningful. Bounded scope keeps approvals legible enough that an approver can understand what is being accepted. The goal is not symbolic human presence. The goal is a reviewer with enough competence, independence, accountability, and stake to make the approval mean something.
