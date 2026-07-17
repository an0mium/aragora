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

## Enforcement on GitHub

Two workflows express this policy on GitHub, and they are deliberately separate.

`aragora-review-gate.yml` ("Aragora Code Review") is advisory. It runs the heterogeneous model review, posts findings as a PR comment, and never fails a PR. `scripts/check_aragora_review_gate_policy.py` guards it against drift and keeps it advisory on purpose.

`aragora-merge-quorum.yml` ("aragora-merge-quorum") is enforcing. It is the required status check on `main`. It builds the read-only merge-authorization packet (`review-queue merge-packet`) for the PR's exact head SHA and fails the check unless the packet authorizes the merge. For Tier 0-2 it passes when the model-quorum verdict is `admin_squash_allowed` with no unresolved dissent. For Tier 3-4 it passes only when, in addition to the model quorum, a head-SHA-bound human settlement signal is recorded — the `aragora/human-settlement` commit status, set by the operator after the local settlement receipt is written — and it fails closed until then.

Branch protection on `main` therefore requires status checks (CI plus `aragora-merge-quorum`) and does not require a human approving review. This is intentional. A human `APPROVE` review by the author, or by any second GitHub identity the author controls, is a symbolic approval with no independent competence behind it: it satisfies the mechanism while defeating the four factors above. The model quorum supplies the technical review; the operator's recorded risk settlement supplies accountability and stake. Neither is a bot `APPROVE`, and neither pretends a second person reviewed the diff.

A second GitHub account operated by the same person as the PR author MUST NOT be used to satisfy any review requirement. It fails the independence factor this document is built on, and in the audit trail it is indistinguishable from a genuine independent review — which makes it worse than no approval at all. Review authority on this repo comes from the model quorum and the operator's recorded settlement, never from a second login.

These principles fit the repo's existing pillars rather than adding new ones. Receipts bind decisions to the exact reviewed state. Evidence-first review requires concrete validation and current-head truth before a merge decision is meaningful. Bounded scope keeps approvals legible enough that an approver can understand what is being accepted. The goal is not symbolic human presence. The goal is a reviewer with enough competence, independence, accountability, and stake to make the approval mean something.

## Model family eligibility by Tier and jurisdiction

The model-quorum gate counts signals from a known set of model families. The classification is explicit and **total** — every recognized family belongs to exactly one of three classes (pinned by governance tests against `aragora/swarm/quorum_evidence.py`):

- **Western families** (Anthropic, OpenAI, xAI, Mistral, Nous Hermes) — count at every tier, including the Tier 3-4 Western-only quorums.
- **Chinese-routed families** (DeepSeek, Qwen, Kimi, GLM, MiniMax, Yi) — count at Tier 0-2 (Tier 2 additionally requires at least one Western family alongside); advisory-only at Tier 3-4.
- **Advisory-only families** (currently Google/gemini) — never count for **or against** any tier's quorum: a PASS never counts toward a quorum and a CHANGES-REQUESTED never creates blocking dissent, at any tier. Their reviews still post, parse, and lint as evidence comments and remain readable on the PR.

The gate must distinguish *which families count for which Tier* and *which payloads may be routed to which jurisdictions*. Without this distinction, expanding the recognizer would silently weaken Tier 3+ review (calibration concerns) and could leak PII across jurisdictional boundaries.

Google (gemini) was demoted to advisory-only by the 2026-07-16 founder roster directive after a repeat fabricated-claim pattern in merge-quorum reviews (evidence: [`docs/governance/records/20260716T2200Z-gemini-reviewer-reliability-record.md`](governance/records/20260716T2200Z-gemini-reviewer-reliability-record.md), whose mandate is "gemini dissent is NOT to be counted anywhere"). Reinstatement requires a founder Tier-4 settlement reversing that record.

### Tier-eligibility for quorum counting

| Tier | Western families count? | Chinese-routed families count? |
| --- | --- | --- |
| 0 — docs, status, tests-only | ✓ | ✓ |
| 1 — additive internal, no live caller | ✓ | ✓ |
| 2 — live automation, CLI, observability, retry, cache | ✓ | ✓; **at least one of the two required signals must be a Western family** |
| 3 — semantic correctness, persistence, security, public API, migrations | ✓ | **Advisory-only — not counted toward quorum**; may still post evidence comments |
| 4 — secrets, deployment, workflow policy, destructive ops, merge-authority self-mod | ✓ — **Western-only counted quorum required** | **Advisory-only — not counted** |

Advisory-only families (currently gemini) count at **no** tier in this table, for or against: their signals are excluded from supportive quorum counting and from blocking-dissent computation at every row.

Rationale: mixed quorums realize the cost advantage of open-weight Chinese families on routine Tier 0-2 work; Western-only counted quorums for Tier 3+ preserve the calibration-trained-on-Western-OSS prior that high-stakes governance and security review depend on. At Tier 4, the requirement is strictly Western-only (not just "at least one Western signal") because merge-authority self-modifications are the highest-stakes class of change in this repo and the entire counted quorum must originate from training-data lineages with documented Western alignment work. Chinese reviewers are never silenced at any Tier — their comments still post and remain readable — but they do not satisfy the quorum-count condition for Tier 3-4 merges.

### Payload-jurisdiction routing rule

Independent of Tier, the *content being sent to the reviewer* determines which jurisdictions may receive it:

| Payload type | Western families | Chinese-routed families |
| --- | --- | --- |
| Public OSS PR title + diff | ✓ | ✓ |
| Private repo PR title + diff (no PII) | ✓ | ✓ if repo policy permits |
| Inbox triage features (low-information, no body, AFT-style) | ✓ | ✓ |
| Inbox triage *raw email body* | ✓ where AWS Secrets Manager loaded | **✗ never** |
| Customer PII, financials, credentials | ✓ subject to data-residency policy | **✗ never** |
| Secrets, encryption keys, OAuth tokens | ✓ subject to data-residency policy | **✗ never** |
| Private legal material (contracts, settlements, NDAs) | ✓ subject to data-residency policy | **✗ never** |
| Healthcare or regulated data | Vertical-specific allowlist only | **✗ never** |

Advisory-only families are mapped explicitly for payload purposes by their provider's jurisdiction, not by their counting class: gemini (Google, US) takes the **Western families** column above. The advisory-only demotion removes counting authority only — it does not change what gemini may receive.

The payload boundary is hard, not a soft preference. It is enforced at the routing layer (`aragora/agents/api_agents/openrouter.py` and any future provider router), not at the quorum-counting layer. A reviewer that should not see a payload must never receive the payload, regardless of whether it would be counted.

### Family-additive change governance

A change that adds a new family marker to the recognizer in `aragora/cli/commands/review_queue.py::_infer_model_reviewer_from_text`, or changes which family counts at which Tier, is a Tier 4 merge-authority self-modification per the Tier table above. It requires human preapproval before implementation and before merge. The pre-approval artifact for such a change is a design document in `docs/specs/` that enumerates the families being added, their proposed Tier eligibility, their proposed jurisdictional payload constraints, and governance tests in `tests/governance/` that characterize the current gate behavior and pin the gap to be inverted by the implementation.

Removing a family marker, demoting a family to advisory-only, or restricting a family's payload eligibility is Tier 4 by the same rule. Loosening any of these constraints in CI (e.g., counting an advisory family at Tier 3) requires the same preapproval discipline as the original addition. The gemini demotion above is such a Tier-4 change; its pre-approval trail is documented in the committed reliability record ([`docs/governance/records/20260716T2200Z-gemini-reviewer-reliability-record.md`](governance/records/20260716T2200Z-gemini-reviewer-reliability-record.md)).

Reviewer model pins for the OpenRouter-routed lanes (e.g., the kimi lane's `moonshotai/kimi-k2.6`) live in `aragora/swarm/quorum_evidence.py::_OPENROUTER_REVIEWER_MODELS` and can be overridden per-run via the `ARAGORA_OPENROUTER_REVIEWER_MODELS` environment variable (a JSON family→slug map); a bad slug degrades to a non-ok `ReviewerResult` rather than blocking the gate. Per the Tier table above, these Chinese-routed families fully count toward Tier 0-2 quorums (subject to the Tier-2 at-least-one-Western condition) and are advisory-only at Tier 3-4 — a model-pin change alone does not alter any family's counting authority. The kimi K3 upgrade is deferred until K3 has live verification and a catalog entry (`aragora/models/catalog.py`); until then the lane stays on k2.6.

## Why fresh context per round is load-bearing (the gate as a task loop)

The merge gate is a task loop: fresh-context iteration against the same spec until compliance. Each evidence round re-grounds reviewers on the exact PR head with no memory of prior rounds. This deliberate "waste" is the mechanism that lets round N catch errors introduced in round N-1 — including errors introduced by the repairers and conductors themselves (observed in the multi-round histories of PRs #8952 and #8827). Do not optimize away re-review by carrying reviewer context across rounds: context freshness is the feature, not an inefficiency (the "Ralph loop" principle).
