# Aragora Feature Gap List

> **Living document** — tracks features planned, partially built, or in need of hardening. Updated as items are completed or priorities shift.
> **For current execution sequencing** (what to work on right now, what is gated, what is delayed), defer to [docs/status/NEXT_STEPS_CANONICAL.md](status/NEXT_STEPS_CANONICAL.md). Active execution status is tracked in [docs/status/ACTIVE_EXECUTION_ISSUES.md](status/ACTIVE_EXECUTION_ISSUES.md) and linked GitHub issues. **This file is the long-horizon capability and productization backlog** — the P0–P4 tiering expresses intended ordering, not dispatch readiness.
> **For the unified finish-line vision and stage model**, see [docs/CANONICAL_GOALS.md](CANONICAL_GOALS.md). The Decision Integrity Core tranche (crux engine, executable claims, proof-carrying code) is gated on Foreman reliability per [docs/plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md](plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md).
> **For concrete 30/90/365-day timing**, the [3-Horizon Execution Roadmap](plans/2026-04-18-3-horizon-roadmap.md) provides the sprint-level overlay. P0/P1 items map to H1/H2 deliverables; P2/P3 items map to H3; P4 items are the deferred maximalist backlog.
> Last updated: April 18, 2026
> March 2026 priority reframe: product cohesion and PMF proof come before certification. Pentest / SOC 2 stay tracked, but they are no longer the first blocker lane.

## How to Read This List

- **P0**: PMF blockers — close the product loop before certification
- **P1**: Value-prop proof (Q2 2026)
- **P2**: Product hardening and enterprise readiness after PMF
- **P3**: Scale & revenue after PMF
- **P4**: Strategic evolution (2026+)
- **Scaffolding**: Code exists but needs hardening/productization

---

## P0 — PMF Blockers

| Feature | Status | Notes |
|---------|--------|-------|
| Truthful live founder loop | **PROVEN — moved to Completed** | 5/5 consecutive live runs pass (35-62s, March 24, 2026). Test baseline: `71 passed` (focused) / `125 passed` (extended). Receipts persist to store for API/dashboard visibility. All 7 acceptance checklist items pass. |
| Smart provider routing | **Shipped on `main`; live proof pending** | PR #724 shipped the Pareto optimizer and pricing database. Runtime wiring landed on `main` via [#1167](https://github.com/synaptent/aragora/pull/1167), and downstream runtime hints are applied through the debate path. The remaining obligation is to prove that routing behaves well in the live founder loop rather than to debate whether the wiring exists. Historical lineage: [#813](https://github.com/synaptent/aragora/issues/813). |
| Complete one working user journey | **Mocked proof passes; live proof still open** | The relevant slices are on `main`: live settings/API-key wiring ([#1146](https://github.com/synaptent/aragora/pull/1146)), live debate creation ([#1147](https://github.com/synaptent/aragora/pull/1147)), onboarding/get-started ([#1170](https://github.com/synaptent/aragora/pull/1170)), quickstart fail-closed behavior ([#1180](https://github.com/synaptent/aragora/pull/1180)), and structured quickstart receipts ([#1192](https://github.com/synaptent/aragora/pull/1192)). The current gap is one repeatable live proof, not another architecture slice. Historical lineage: [#1046](https://github.com/synaptent/aragora/issues/1046). |
| Knowledge Mound reads enrich debate context | **Shipped on `main`; live read/write proof pending** | Retrieval, precedent loading, and writeback groundwork landed via [#1111](https://github.com/synaptent/aragora/pull/1111), [#1131](https://github.com/synaptent/aragora/pull/1131), [#1132](https://github.com/synaptent/aragora/pull/1132), [#1134](https://github.com/synaptent/aragora/pull/1134), [#1151](https://github.com/synaptent/aragora/pull/1151), [#1168](https://github.com/synaptent/aragora/pull/1168), and [#1176](https://github.com/synaptent/aragora/pull/1176). The remaining question is whether that read/write path is visible and trustworthy in the live founder loop. Historical lineage: [#1048](https://github.com/synaptent/aragora/issues/1048). |
| Debate output quality | **VALIDATED — moved to Completed** | Run 012 (Mar 5): composite 8.38-9.39/10. Diverse benchmark (10 domains): 100% pass, avg composite 0.938. |

---

## P1 — Value-Prop Proof (Q2 2026)

| Feature | Status | Notes |
|---------|--------|-------|
| OpenClaw end-to-end demo | **Core loop shipped** | PR #727: CodeImplementationTask, SpecExtractor, ComputerUseActionBundle, receipt linkage. Production validation with live agents remaining. Tracked in [#814](https://github.com/synaptent/aragora/issues/814). |
| Functional frontend paths (5 that matter) | More truthful on `main`; continuity still gated by live founder-loop proof | Recent merged slices improved the real surface: settings/API-key wiring ([#1146](https://github.com/synaptent/aragora/pull/1146)), debate creation ([#1147](https://github.com/synaptent/aragora/pull/1147)), public status truthfulness ([#1148](https://github.com/synaptent/aragora/pull/1148)), pipeline golden-path visibility ([#1150](https://github.com/synaptent/aragora/pull/1150)), onboarding/get-started ([#1170](https://github.com/synaptent/aragora/pull/1170)), and Wave 2 productization ([#1188](https://github.com/synaptent/aragora/pull/1188)). The remaining gap is continuity across debate creation, results, knowledge, settings, receipts, and dashboard state in a live run. Historical lineage: [#1047](https://github.com/synaptent/aragora/issues/1047). |
| 10+ agent coordinated debates | Scaffolding | Current practical limit: 2-6 agents. Coordination infrastructure exists; scale testing needed after the core loop works. Tracked in [#815](https://github.com/synaptent/aragora/issues/815). |
| Agent-first beta via REST API | **Fleet deployed (12 runners)** | `aragora openclaw watch` polls repos, runs multi-agent review, posts findings. 3 Hetzner + 6 EC2 + 3 local Macs. PR watch daemon on Mac Studio via launchd. Shared operator productization is tracked in [#817](https://github.com/synaptent/aragora/issues/817) and [#819](https://github.com/synaptent/aragora/issues/819). |
| GitHub Actions pre-merge gate | **Workflow created** | `aragora-review-gate.yml` manual-only (workflow_dispatch). Re-enable pull_request trigger when ready. |
| Public demo at aragora.ai/demo | **Truthful proof surface** | `/demo` now runs a canonical live-backed proof when the public playground backend is live and labels any non-live fallback or recorded sample explicitly. `/try` remains the primary beta funnel for user-entered questions, persistence, and sharing. Productization is tracked in [#818](https://github.com/synaptent/aragora/issues/818). |
| EU AI Act compliance package | **Substantially complete (90/100)** | Art. 9/10/11/12/13/14/15/43/49 bundle coverage and customer playbook appendix are shipped. Remaining work is packaging polish, regulator-ready validation, and launch collateral hardening. **Deadline: Aug 2, 2026.** |
| First 2 enterprise pilot engagements | Not started | Closed partnerships — target fintech + healthcare |
| Developer onboarding <10 min | **Contract improved; live timing proof still required** | `aragora quickstart` now fails fast on bad TLS ([#1180](https://github.com/synaptent/aragora/pull/1180)) and supports inline provider keys plus structured receipts ([#1192](https://github.com/synaptent/aragora/pull/1192)). That is better than the earlier behavior, but the actionable target is a repeatable live onboarding run with bounded noise and measured timing, not a blanket "already working" claim. |

---

## P2 — Product Hardening And Enterprise Readiness (After PMF)

| Feature | Status | Notes |
|---------|--------|-------|
| External penetration test | Scope and outreach artifacts ready; vendor selection pending | Kickoff stays warm, but certification is intentionally sequenced after the product loop is usable. Operational status is tracked in `security/pentest/VENDOR_OUTREACH_LOG.md`; work remains tracked in [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), and [#509](https://github.com/synaptent/aragora/issues/509). |
| Semantic convergence (full embedding) | **VALIDATED — moved to Completed** | PR #723 migrated 5 similarity modules from difflib to embedding-based. Remaining difflib usage is exclusively for text diff display, not similarity. |
| ERC-8004 on-chain deployment | Contracts written | Solidity contracts exist; not deployed to any mainnet. Needs chain endpoint config + gas management. Tracked in [#816](https://github.com/synaptent/aragora/issues/816). |
| Decision-Integrity UI Workbench | Partial frontend | Existing workbench pages render, but they do not replace the PMF need for five truthful user-facing paths. Remaining canvas and data wiring work stays secondary to `#1047`. |
| SOC 2 Type II audit engagement | Scope doc ready | 60+ controls implemented (98%); pentest scope doc v3.1.0 finalized; vendor shortlisted (NCC, Bishop Fox, Trail of Bits, Cure53). Blocker: vendor selection + engagement. |
| Enterprise Communication Hub (#293) | **Epic closed** | PR #726: template persistence, router event wiring, E2E tests. Delivery log, retry queue, circuit breakers, event telemetry, user preference UI, Active Triage dashboard, TriageRulesPanel all shipped. Remaining: inbox→debate trigger wiring end-to-end validation, tracked in [#817](https://github.com/synaptent/aragora/issues/817). |

---

## P3 — Scale & Revenue (Q3–Q4 2026)

| Feature | Status | Notes |
|---------|--------|-------|
| Cloud marketplace listings | Not started | AWS Marketplace + Azure Marketplace listings. Infrastructure ready. |
| Vertical packages | Not started | Healthcare (FHIR, HIPAA), Financial (SOX, risk), Legal (contracts, discovery). Guides exist; packages not assembled. |
| Skills Marketplace pilot | Scaffolding | SkillRegistry + SkillMarketplace code exists; no public marketplace endpoint. |
| On-premise deployment productization | Partial | Docker Compose + Helm chart exist; on-prem installer/wizard not built. |
| International expansion / EU data residency | Not started | Data residency controls needed for EU enterprise buyers. |
| Compute escrow mechanism | Not started | Settlement stakes via crypto compute escrow. Design in docs/plans/. |

---

## P4 — Strategic Evolution (2026+)

| Feature | Status | Notes |
|---------|--------|-------|
| Prover-Estimator debate protocol | **Shipped and wired** | 581 LOC engine + consensus handler wired into `consensus_phase.py`. Use `protocol.consensus="prover_estimator"` to activate. 5-stage pipeline: decompose→estimate→challenge→re-estimate→aggregate. 33 unit tests pass. |
| Cross-verification phase (3-pass hallucination detection) | **Shipped and wired** | 395 LOC engine wired as optional post-debate enrichment in `orchestrator_runner.py`. Set `arena.enable_cross_verification=True`. Computes grounding_delta, adversarial_resistance, hallucination_risk. |
| Truth scorer integrated into vote weights | **Shipped and wired** | `TruthScorer` (398 LOC) scores evidence-vs-rhetoric ratio per proposal. `apply_truth_ratio_bonuses()` in `VoteBonusCalculator` rewards high truth ratios. Enable via `protocol.enable_truth_ratio_weighting=True`. |
| Epistemic hygiene + anti-sycophancy | **Shipped and integrated** | ~1,695 LOC across `epistemic_hygiene.py`, `trickster.py`, `trickster_calibrator.py`. Fully integrated into consensus, settlement, prompt assembly, and server. ~3,744 LOC tests. |
| Prompt-to-spec engine | **Shipped** | `aragora spec` CLI command completes in ~23s. Decompose→interrogate→research→specify pipeline. `aragora/prompt_engine/` module (decomposer, interrogator, researcher, spec_builder, conductor). |
| Canvas GUI (8-stage visual DAG) | Partial frontend | Prompt-engine page exists; full 8-stage visual canvas missing. |
| Market resolution mechanism | Design only | Long-horizon settlement claim pricing via prediction market. |
| STOP N-candidate for Nomic Loop | Design only | Multi-plan generation before committing to self-improvement path. |
| Meta-improver for debate protocols | Design only | A/B test protocol variants using Nomic Loop. |
| Obsidian bidirectional sync | **Shipped** | `ObsidianAdapter` with `ReverseFlowMixin` for KM→Obsidian writeback. Forward sync, conflict detection, filesystem watcher. |
| Dialectical Runtime synthesis layer (DIC-23..28) | Planning only | Additive tranche that closes the loop between DIC-20 decay signals, DIC-15 crux-finder, DIC-21 quarantine, and DIC-22 repair proposals. Adds a report-only runtime loop orchestrator (DIC-23 / [#6217](https://github.com/synaptent/aragora/issues/6217)), epistemic genealogy ledger (DIC-24 / [#6218](https://github.com/synaptent/aragora/issues/6218)), adversarial world-state stress-test (DIC-25 / [#6219](https://github.com/synaptent/aragora/issues/6219)), belief coherence monitor (DIC-26 / [#6220](https://github.com/synaptent/aragora/issues/6220)), operator crux arbitration receipts (DIC-27 / [#6221](https://github.com/synaptent/aragora/issues/6221)), and proactive crux gardening (DIC-28 / [#6222](https://github.com/synaptent/aragora/issues/6222)). Epic: [#6223](https://github.com/synaptent/aragora/issues/6223). Activation gated on DIC-20/21/22 production-green plus the proof-first Foreman gate. Design: [docs/plans/2026-04-18-dialectical-runtime-synthesis.md](plans/2026-04-18-dialectical-runtime-synthesis.md). Planning-only labels; no `boss-ready` until the gate opens. |

---

## P5 — Federation (Future)

| Feature | Status | Notes |
|---------|--------|-------|
| Distributed debates across organizations | Design only | Cross-org debate with privacy-preserving knowledge sharing. |
| Cross-organizational knowledge sync | Design only | Federated knowledge graph across org boundaries. |
| Knowledge federation | Design only | Global KM with distributed consensus. |

---

## Scaffolding — Code Exists, Needs Hardening

| Feature | Current State | Gap |
|---------|---------------|-----|
| Self-improving platform quality | Nomic Loop 100% wired; 82 E2E tests; CLB backbone hardened (14/14 issues closed); safety gates + gauntlet gate + evolution audit + golden-path test; **Ralph V14 benchmark validated full autonomous loop** (PRs #1004-#1006) | Diverse benchmark validated (100% pass). Production safety gate requires ENABLE_NOMIC_LOOP=true. Ralph autonomy loop closed for `merge_policy=admin_merge_allowed`. |
| Autonomous self-assessment loop | `IdeaToExecutionPipeline.from_system_metrics()`, `SelfImprovePipeline`, `NomicLoop`, `execute_to_github_issue()`, `plan_from_issue_list()`, worktree autopilot, the canonical assessment compiler, and the pause-refresh shift controller exist on `main` | The remaining truth gap is the control plane: no canonical developer task queue/claim protocol, no universal per-lane run receipt/provenance artifact, and no integrator view tying claims, receipts, heartbeats, and merge readiness together. Tracked in [#1036](https://github.com/synaptent/aragora/issues/1036), [#837](https://github.com/synaptent/aragora/issues/837), [#842](https://github.com/synaptent/aragora/issues/842), [#843](https://github.com/synaptent/aragora/issues/843), and [#990](https://github.com/synaptent/aragora/issues/990). |
| Blockchain receipts | SHA-256 cryptographic hashing works; StakingRegistry (stake/slash/unstake/rewards); ComputeBudgetManager; ReceiptAnchor; SlashEvent model (hollow_consensus, factual_error, calibration_drift) | On-chain storage with ERC-8004 (not deployed); staking/slashing NOT wired into debate outcome loop; agent selection not weighted by compute budget. Tracked in new GitHub issue. |
| Semantic convergence | **Migrated** (PR #723) | All similarity paths use embeddings. Only `unified_diff` (text display) uses difflib. |
| OpenClaw execution | **Core loop shipped** (PR #727) | CodeImplementationTask + SpecExtractor + receipt linkage. Production validation pending. |
| RLM context access | Code complete (92 exports, 27 test files, 15k LOC) | No user-facing guide; integration with default Arena config unclear; training pipeline (buffer/policy/reward/trainer) untested E2E. Tracked in new GitHub issue. |

---

## Completed — Formerly on This List

These items were planned and are now shipped:

| Feature | Shipped |
|---------|---------|
| Nomic Loop end-to-end | Jan 2026 |
| Knowledge Mound Phase A2 (adapter registry + unified memory hardening) | Feb 2026 |
| Unified Memory Gateway | Feb 2026 |
| Retention Gate (Titans/MIRAS) | Feb 2026 |
| RBAC v2 (14 resource types, 8 actions) | Feb 2026 |
| Multi-tenancy (tenant isolation + metering) | Feb 2026 |
| Voice/TTS integration | Feb 2026 |
| Pipeline orchestration (4-stage) | Feb–Mar 2026 |
| Compliance CLI (EU AI Act artifact export) | Mar 2026 |
| Settlement hooks | Mar 2026 |
| Gauntlet receipts (SHA-256 audit trails) | Feb 2026 |
| Article 9 dedicated artifact bundle | Mar 2026 |
| Debate output quality (10-domain benchmark) | Mar 2026 |
| Nomic Loop safety gates (production gate, gauntlet, audit) | Mar 2026 |
| CI setup-python-safe composite action (51 workflows) | Mar 2026 |
| MFA admin enforcement (compliance API, drift alerts, bypass docs) | Mar 2026 |
| Data classification enforcement (runtime, CI PII gate, evidence bundles) | Mar 2026 |
| Closed-loop backbone contracts (IntakeBundle, SpecBundle, DeliberationBundle, ReceiptEnvelope, OutcomeFeedbackRecord) | Mar 2026 |
| Fail-closed spec validation (execution-grade field enforcement) | Mar 2026 |
| Deliberation bundle handoff (dissent + quality gate into planning) | Mar 2026 |
| CI push-to-main noise reduction (35→6 workflows) | Mar 2026 |
| Self-hosted runner fleet (12 runners: 3 Hetzner + 6 EC2 + 3 Mac) | Mar 2026 |
| Decision-Integrity Workbench frontend | Mar 2026 |
| G1 Signed Context Manifests (HMAC-SHA256 + CLI) | Mar 2026 |
| Closed-loop backbone sprint (14 CLB issues) | Mar 2026 |
| ExecutionBundle + VerificationBundle contracts | Mar 2026 |
| Bug-fix loop after verification failure (auto-trigger) | Mar 2026 |
| Receipt envelope normalization (success/fail/blocked) | Mar 2026 |
| Outcome feedback → Nomic goal export pipeline | Mar 2026 |
| Trust-tier + taint propagation across backbone | Mar 2026 |
| External-verifier insertion point (CLB-012) | Mar 2026 |
| Golden-path backbone test (intake → receipt E2E) | Mar 2026 |
| Dogfood backbone profile script | Mar 2026 |
| PR watch daemon fleet (3 Mac machines, 30 reviews/hour) | Mar 2026 |
| Dev swarm coordination layer (lease-aware) | Mar 2026 |
| Swarm supervisor + worker launcher + reconciler (PRs #744-746) | Mar 2026 |
| Inbox trust wedge — receipt-gated email actions (PRs #731-742) | Mar 2026 |
| Session circuit-breaker — auth-state pinning (PRs #736, #740) | Mar 2026 |
| Gmail OAuth setup helper (PR #741) | Mar 2026 |
| Semantic convergence — 5 modules migrated to embeddings (PR #723) | Mar 2026 |
| Smart provider routing Phase 1 — Pareto optimizer + 8-model pricing (PR #724) | Mar 2026 |
| EU AI Act playbook GTM polish + Art. 10/11/43/49 appendix (PR #725) | Mar 2026 |
| Enterprise Comms Hub #293 — template persistence + router wiring (PR #726) | Mar 2026 |
| OpenClaw E2E core loop — CodeImplementationTask + ComputerUseActionBundle (PR #727) | Mar 2026 |
| Stale lease auto-release — PID-based liveness, proactive reaping (PR #1004) | Mar 2026 |
| Reconciler lease reaping — tick_run reaps expired leases (PR #1005) | Mar 2026 |
| Admin merge bypass — autonomous merge when required checks pass (PR #1006) | Mar 2026 |
| Ralph V14 benchmark — full autonomous loop validated (spec→PR→merge, zero intervention) | Mar 2026 |
| Ralph blocker taxonomy — campaign_stalled, needs_human classification (PRs #946-#950) | Mar 2026 |
| **Live founder loop proven repeatable** (5/5 runs, 35-62s, receipts on API/dashboard) | Mar 2026 |
| Prover-Estimator consensus handler wired into debate pipeline | Mar 2026 |
| Cross-verification post-debate enrichment wired into orchestrator | Mar 2026 |
| Truth scorer ratio bonuses wired into vote weight calculation | Mar 2026 |
| Prompt-to-spec CLI (`aragora spec`) — 23s end-to-end | Mar 2026 |
| Inbox trust wedge CLI (`aragora triage auth`, `--dry-run`) | Mar 2026 |
| Quickstart receipt store persistence for API/dashboard visibility | Mar 2026 |
| Embedding rate limit resilience (hash-based fallback on 429) | Mar 2026 |
| Summary preamble cleaning (strip LLM chain-of-thought from CLI output) | Mar 2026 |
| EU AI Act compliance bundle verified end-to-end with real quickstart receipts | Mar 2026 |
