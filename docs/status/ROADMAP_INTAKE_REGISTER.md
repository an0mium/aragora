# Roadmap Intake Register

**Purpose:** a single, durable intake point so that planning outputs — plans, specs,
visions, goals, "make this permanent" findings produced by humans or agents across many
sessions — are **captured, tracked to a terminal state, and reflected in the canonical
docs**, instead of being lost in chat history, scratch worktrees, or one-off files.

This register is the *index and intake gate*. It does not replace the canonical docs; it
feeds them.

## The intake rule (guardrail)

> **Any planning output meant to persist MUST be recorded here with a tracking row, and
> routed to exactly one canonical destination.** A plan that lives only in a chat
> transcript, a detached worktree, or an un-referenced file is considered *not durable* and
> at risk of loss.

When an agent or human is asked to "make this plan permanent / durable":
1. Write the detail to the right place (see destinations below).
2. Add a row to the **Register** table here (title, source, status, destination, tracking).
3. Open or link a GitHub issue/epic if it is executable work.

## Canonical destinations

| Kind of output | Canonical home |
|---|---|
| High-level direction | [`ROADMAP.md`](../../ROADMAP.md) |
| Goals / north stars | [`docs/CANONICAL_GOALS.md`](../CANONICAL_GOALS.md) |
| Near-term execution order | [`docs/status/NEXT_STEPS_CANONICAL.md`](NEXT_STEPS_CANONICAL.md) |
| Tracked epics/milestones | [`docs/status/ACTIVE_EXECUTION_ISSUES.md`](ACTIVE_EXECUTION_ISSUES.md) |
| Feature backlog (P0–P5) | [`docs/FEATURE_GAP_LIST.md`](../FEATURE_GAP_LIST.md) |
| Detailed design specs | `docs/plans/` , `docs/superpowers/specs/` , `docs/superpowers/plans/` |
| Executable work | GitHub issue / epic (linked from the row) |

## Where durable planning outputs already live (search these first)

- **Repo docs:** `ROADMAP.md`, `docs/CANONICAL_GOALS.md`, `docs/FEATURE_GAP_LIST.md`, `docs/status/*.md`, `docs/plans/*.md`, `docs/superpowers/specs/*.md`, `docs/briefs/*`.
- **GitHub:** open issues/epics (500+ open), PR bodies, epic/milestone descriptions.
- **Agent/automation state (lower-trust, may be stale):** `.aragora/automation-receipts/`, `.aragora/automation-outbox-archive/`, `.aragora/review-queue/briefs/`, `aragora/nomic/dev_coordination/`, `.nomic/` plan stores, `.claude/plans/`.
- **At-risk / not durable:** chat transcripts and detached scratch worktrees that were never written into the repo.

## Register — recently captured (2026-06-26 session)

| Item | Source | Status | Destination | Tracking |
|---|---|---|---|---|
| **Sakana Fugu integration** (Fugu as capable agent + resilience backstop + benchmark; never counts as model-family diversity) | this session | Planned | lands via PR [#8640](https://github.com/synaptent/aragora/pull/8640) → `docs/plans/2026-06-26-sakana-fugu-integration.md` (not yet on `main`) | Epic [#8641](https://github.com/synaptent/aragora/issues/8641); F1–F5 [#8642](https://github.com/synaptent/aragora/issues/8642) [#8643](https://github.com/synaptent/aragora/issues/8643) [#8644](https://github.com/synaptent/aragora/issues/8644) [#8645](https://github.com/synaptent/aragora/issues/8645) [#8647](https://github.com/synaptent/aragora/issues/8647) |
| **Reconcile lane** (merge-first reconcile mission on the spine: prune→triage→inspect→harvest→cut→settle→govern; one conductor; real pause/lock) | this session | Spec written | lands via PR [#8656](https://github.com/synaptent/aragora/pull/8656) → `docs/superpowers/specs/2026-06-26-reconcile-lane-design.md` (not yet on `main`) | Epic [#8649](https://github.com/synaptent/aragora/issues/8649) |
| **Native mission engine** (`aragora mission seed/status/run/resume/reconcile`; preserve-first reconcile/admission; live exact-head gate; validation injection; operator receipts) | `codex/native-mission-engine` | **CANONICAL**; in quorum/settlement | `docs/superpowers/plans/2026-06-26-native-mission-engine.md` (on the branch) | PR [#8655](https://github.com/synaptent/aragora/pull/8655) — head advances during settlement; check live `gh pr view 8655` for exact head rather than pinning a SHA |
| **Mission spine (Phase A)** survivable orchestrator (`aragora/missions/` state/ledger/orchestrator/swarm) | `mission/native-orchestrator-spine` | **SUPERSEDED** by `codex/native-mission-engine` | `aragora/missions/` | PR #8628 closed; superseded by PR [#8655](https://github.com/synaptent/aragora/pull/8655) |
| **Workspace reconciliation (2026-06-26)** branches 2,749→489, worktrees 319→1; 1,695 stale codex branches inspected, 4 preserved | this session | Done | (operational) | preserved branches: `codex/review-6887`, `codex/rbac-openapi-coverage-primary-20260615`, `codex/validate-doc-links-anchor-check-r2-20260514`, `codex/disaster-recovery-stat-portability-improver-20260609` — harvest to PRs |
| **Head freeze** — 5 grinder daemons disabled (boss-loop, merge-arbiter, merge-shepherd, overnight-watchdog, publisher); publisher pause-manifest bug confirmed (never read) | this session | Done | (operational) | fixed by reconcile-lane "real pause/lock" (above) |

## Open planning epics (index, 2026-06-26)

These are the durable, tracked roadmap epics. Keep this list current as the intake gate.

| Epic | Title |
|---|---|
| [#8641](https://github.com/synaptent/aragora/issues/8641) | Sakana Fugu integration |
| [#8257](https://github.com/synaptent/aragora/issues/8257) | Codebase health: macro-architecture, packaging, gate-rigor (Factory "Structural Excellence") |
| [#8223](https://github.com/synaptent/aragora/issues/8223) | Open Decision Receipt (ODR) — decision-semantics layer |
| [#8344](https://github.com/synaptent/aragora/issues/8344) | Conveyor hardening — close the six failure classes |
| [#273](https://github.com/synaptent/aragora/issues/273) | Enterprise Assurance Closure |
| [#6303](https://github.com/synaptent/aragora/issues/6303) | Productize PR intelligence brief |
| [#6158](https://github.com/synaptent/aragora/issues/6158) | TCP — Trust-Compound Plan (TCP-1..7) |
| [#6068](https://github.com/synaptent/aragora/issues/6068) | Agent-civilization substrate (AGT-01..06) — planning-only |
| [#6226](https://github.com/synaptent/aragora/issues/6226) | 3-Horizon Execution Roadmap (30/90/365) |
| [#6223](https://github.com/synaptent/aragora/issues/6223) | Dialectical Runtime Synthesis Layer (DIC-23..28) |
| [#6235](https://github.com/synaptent/aragora/issues/6235) / [#6236](https://github.com/synaptent/aragora/issues/6236) / [#6237](https://github.com/synaptent/aragora/issues/6237) | H2 / H3 / deferred maximalist planning-only |

## Pending consolidation actions

1. **Full audit** ([Epic #8650](https://github.com/synaptent/aragora/issues/8650)): sweep all durable sources (docs, GitHub issues/PRs, `.aragora/` state, nomic plan stores) and classify every item as `canonical / implemented / duplicate / superseded / active-roadmap / needs-decision / chat-only`. _(This register is the living home of that audit.)_
2. ~~Reconcile-lane epic~~ → done: [Epic #8649](https://github.com/synaptent/aragora/issues/8649).
3. ~~Mission canonical-branch decision~~ → **resolved**: canonical = `codex/native-mission-engine` (PR #8655); regen fix landed; spine PR #8628 closed. Remaining: clear model-quorum on its live exact head, then settle.
4. **Harvest the 4 preserved branches** into draft PRs (see Register row).

> **Single-register rule:** this file (`docs/status/ROADMAP_INTAKE_REGISTER.md`, PR #8648) is THE intake register. Any agent asked to "create an intake register" must **extend this one**, not create a parallel file — duplicating it reproduces the exact sprawl it exists to prevent.

## Chat-only / not-yet-durable items (capture, do NOT treat as adopted)

These are themes raised in chat/attachments that are **not yet decided** roadmap items. Captured
here so they aren't lost; each needs an explicit founder decision before becoming tracked work.
Do not implement from this section.

| Theme | Origin | Status | Notes |
|---|---|---|---|
| **Sakana Fugu integration** | chat request | **NOW DURABLE** → moved to Register above | Was chat-only; now `docs/plans/2026-06-26-sakana-fugu-integration.md` + Epic #8641 |
| **Automated STRIDE / OWASP security review on every PR** (CWE refs, severity, inline fix) | Factory.ai announcement | needs-decision | Parallels Aragora's `security-review` capability + gauntlet; candidate product surface |
| **Software-factory loop** (signal → triage → build → test → review → ship → monitor → signal) | Factory 2.0 | needs-decision | Aragora analog = mission engine + reconcile lane + receipts; this is the umbrella vision, not a discrete task |
| **Model router / model independence** | Factory Router | partially exists | Aragora already has `aragora/routing/` Pareto optimizer + OpenRouter fallback; Fugu (#8641) extends it; decide how much to productize as a named "router" |
| **Sovereign / self-hosted / air-gapped intelligence** | Factory 2.0 | needs-decision | Aligns with founder's no-standing-key / Secrets-Manager principle; EU/air-gapped deployment story |
| **Continual learning / self-improvement instrumentation** | Factory 2.0 | partially exists | Aragora has Nomic loop + KnowledgeMound cross-cycle learning; decide what to formalize/measure |
| **Missions + operator dashboard + persistent execution** | Factory 2.0 | in progress | Mission engine in PR #8655; dashboard + headless persistent runtime are the remaining gaps |

## Maintenance

- Review this register whenever a planning output is produced or a session ends with a multi-stage plan.
- An item is **done** only when it reaches a terminal state in its canonical destination (merged, closed-superseded, or explicitly parked with a receipt) — mirroring the merge-first / terminal-receipt discipline of the reconcile lane.
