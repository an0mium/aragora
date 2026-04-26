# Aragora 3-Horizon Execution Roadmap (30 / 90 / 365)

> **Status:** active execution plan; operationalizes [ARAGORA_EVOLUTION_ROADMAP.md](ARAGORA_EVOLUTION_ROADMAP.md) into concrete time-boxed deliverables
> **Created:** 2026-04-18
> **Authority:** executes the stage model in [CANONICAL_GOALS.md](../CANONICAL_GOALS.md) (Tool → Teammate → Foreman → Chief of Staff → Organization Substrate)
> **Queue policy:** H1 items may carry `boss-ready` per the current proof-first Foreman gate; H2/H3 items are planning-only until their activation conditions fire; deferred backlog items never carry live-dispatch labels until explicitly re-staged
> **Relationship:** additive to all existing strategic docs; does **not** replace ARAGORA_EVOLUTION_ROADMAP, CANONICAL_GOALS, FEATURE_GAP_LIST, EPISTEMIC_CI_AND_CRUX_ENGINE, or the 3-horizon outcome map already in ARAGORA_EVOLUTION_ROADMAP — it operationalizes them with concrete, bounded work

## Synthesis Frame

The maximalist vision is intact and preserved. All of it:

- **Trustworthy autonomous execution** — vague goals → clarified plans → bounded tasks → code → tests → PRs → merges → learning loops
- **Decision integrity layer** — important decisions debated, challenged, audited, receipted with cryptographic chain of custody
- **Self-improving organization substrate** — memory, knowledge, coordination, governance, monitoring, self-healing compound over time
- **Operating system for agentic work** — not just coding, but roadmap execution, approvals, inbox/workflow actions, multi-domain orchestration
- **Elegant intuitive GUI** — unified DAG structure with optional interactivity at any level, human-legible view of the same runtime truth
- **Leading-edge permissioned memory + broad ingestion** — private/permissioned memory across sources and streams with provenance, trust tiers, and large-context packing
- **Heterogeneous agent parity** — Claude, Codex, Gemini, Grok, DeepSeek, Qwen, Mistral, Llama plus OpenClaw, Nous Hermes, Pi Agent, Anthropic Agent Framework, LangGraph, AutoGen, CrewAI, and future entrants all interoperate with the shared knowledge base and debate substrate
- **SMB operating system** — translates relevant data and ideas to actions with optional detailed control over agents
- **Cryptographic auditable receipts** — including ERC-8004 on-chain attestation when the base safety gates are in place
- **Key decisions debated + stress-tested before consensus, then iteratively broken down and dispatched with high-quality specifications**

The execution path is narrow first, then broad. The efficient sequence is to win the reliability wedge on Aragora itself, use that wedge to unlock design-partner proof, then extend the substrate outward. Every horizon here is a proof ramp, not a scope expansion.

## Stage Targets per Horizon

| Horizon | Stage transition | Proof that matters | Deferred until proof lands |
|---|---|---|---|
| **H1 (Day 1-30)** | Tool → early Teammate | Reliability wedge: autonomous repo maintainer hits ≥50% zero-rescue on bounded corpus; first booster proven | Multi-host expansion, external design partners, new agent types, new verticals |
| **H2 (Day 31-90)** | Teammate → Foreman | External wedge: 2-3 design partners using `aragora review` + decision-integrity backbone live; ≥70% zero-rescue; 12h multi-host soak passes 3× | Chief-of-staff portfolio loops, full canvas GUI, marketplace expansion, live chain writes |
| **H3 (Day 91-365)** | Foreman → Chief of Staff | Organization-substrate foundation: unified DAG canvas, permissioned memory fabric, SMB OS packaging, heterogeneous agent marketplace opt-in, DIC-23..28 in production, first 10 paying customers | Federation, international data residency, compute escrow, Skills Marketplace public launch |
| **Deferred** | Chief of Staff → Organization Substrate | Cross-functional idea-to-execution on one DAG across many organizations with portable reputation | Activates only after H3 exit criteria hold for ≥90 days |

## Horizon 1 (Day 1-30): Prove the Booster

**Single goal:** The autonomous repo maintainer running against Aragora's own backlog hits ≥50% zero-rescue success on a frozen benchmark corpus of bounded software tasks, with 100% of failures classified into canonical buckets.

This is the [Reliability Substrate Spec](../../) already written, executed on a deadline.

### H1 concrete deliverables

| ID | Deliverable | Owner-surface | Acceptance |
|---|---|---|---|
| H1-01 | Benchmark corpus frozen at rev-4 with ≥30 bounded tasks (missing tests, exception narrowing, validation tightening, small refactors, flaky-fix, helper extraction, safe dependency cleanup) | `tests/benchmarks/` + `docs/status/generated/benchmark_scorecards/` | Corpus signed with SHA-256; runnable in CI; scorecard generator emits per-task truth status |
| H1-02 | Daily no-rescue scorecard published to `docs/status/B0_BENCHMARK_TRUTH_STATUS.md` | `scripts/build_benchmark_truth_artifact.py` + daily workflow | Scorecard publishes zero-rescue %, canonical failure-bucket %, and week-over-week delta |
| H1-03 | Phase-4 Task Sanitation Gate shipped | `aragora/swarm/task_sanitizer.py` + `aragora/swarm/boss_validation.py` | Malformed tasks (truncation, contradictory scope, impossible acceptance, missing verifier) rewritten or dropped with audit trail; needs_human drop ≥30% |
| H1-04 | Phase-5 Autonomy Ledger + Self-Heal shipped | `aragora/swarm/autonomy_ledger.py` + `aragora/swarm/self_heal.py` | Ledger mirrors probes/contracts/receipts/outcomes; automatic quarantine for stale auth, permission mismatch, rate limits, publication failures; health/reporting surfaces cut over to ledger truth |
| H1-05 | EU AI Act compliance package: market what exists, not what to build | `docs/compliance/EU_AI_ACT_GUIDE.md` + existing compliance artifact export | Package the compliance artifact bundle, decision receipts, and policy gate documentation into a sellable form for regulated buyers ahead of Aug 2, 2026 deadline — no new enforcement code in H1 |
| H1-06 | Public dogfood cadence: 3 real aragora.ai debates/week + 1 published blog post | Founder time, ~4 hours/week | 12+ debates run over 30 days; one public write-up ("chicken nuggets" pattern) demonstrating heterogeneous consensus; shared on LinkedIn/X |
| H1-07 | Mac Studio boss loop runs overnight with daily checkpoint | Daily 15-minute operator review | Junk issues closed, well-scoped issues relabeled `boss-ready`; PR trends published in `docs/status/PR_MERGE_VELOCITY.md` |
| H1-08 | Dialectical Runtime roadmap docs (PR #6224) merged as foundation for H2 Decision-Integrity backbone | `docs/plans/2026-04-18-dialectical-runtime-synthesis.md` | PR #6224 merged; issues #6217-#6223 remain planning-only until gate opens in H2/H3 |

### H1 exit criteria

- ≥50% zero-rescue on frozen corpus over 5 consecutive days
- 100% of failures classified into canonical terminal-truth buckets
- ≥30% reduction in needs_human rate vs baseline (via sanitizer)
- Ledger-backed status surface is the single source of truth (no conflicting shell heuristics)
- At least 1 design-partner conversation initiated (even if not yet committed)
- EU AI Act package ready to hand to a buyer without further engineering

### H1 explicit non-goals

- No Canvas GUI work
- No new agent type registrations
- No new vertical packaging
- No ERC-8004 live chain writes
- No SMB OS packaging
- No Nomic Loop routing changes beyond the substrate wiring
- No DIC-23..28 implementation (design docs only; PR #6224 is the whole H1 DIC scope)

## Horizon 2 (Day 31-90): Extend the Wedge Externally

**Single goal:** 2-3 external design partners use `aragora review` on their codebases with measurable value; zero-rescue rate on the benchmark corpus crosses 70%; decision-integrity backbone is live across the golden path; EU AI Act packaging is ready for the August 2, 2026 enforcement window.

### H2 concrete deliverables

| ID | Deliverable | Depends on | Acceptance |
|---|---|---|---|
| H2-01 | 2-3 external design partners onboarded with `aragora review` | H1-07 conversations | Each partner runs ≥1 review/week on their own repo; weekly feedback captured; one partner is in a regulated vertical (fintech/healthcare/legal) for EU AI Act dogfood |
| H2-02 | Benchmark corpus ≥70% zero-rescue + 12h multi-host soak passes 3 consecutive times | H1-01, H1-04 | Multi-host soak report published; rescue-class harvest empty for ≥7 days; benchmark rev-5 corpus (≥50 tasks) drives this |
| H2-03 | Decision-integrity backbone: RunLedger threaded through orchestrator/debate/planning; receipts auto-linked to runs; OutcomeFeedbackRecord closes the loop | PR #6224 foundation; `aragora/pipeline/backbone_contracts.py` already merged | `/api/runs/{id}` returns full stage transition history; receipts and feedback auto-persist; dashboards show live ledger state |
| H2-04 | DIC-13..20 in production on ≥3 proof-carrying code units (claim manifests, verification runner, CruxSet emission, truth map, proof-carrying code schema, decay monitor) | H2-02 reliability; DIC-13..20 already shipped | Truth map report published weekly; decay monitor produces integrity scores on 3 real units; no live queue mutation from failed claims |
| H2-05 | ERC-8004 shadow-mode integration: attestation payloads built for receipts and CruxSets; stored in pending-attestation queue; no live wallet writes | H2-03 backbone | Shadow attestation payloads validated against contract ABIs; payload-to-receipt mapping tested; `aragora/blockchain/receipt_anchor.py` emits shadow records only |
| H2-06 | Heterogeneous agent parity proof: Anthropic + OpenAI + Gemini + one OpenRouter model complete the full golden path (intake → debate → plan → execution → receipt) | H2-03 backbone | Each provider runs the full loop on the benchmark corpus; results captured in heterogeneous-parity scorecard; fallback behavior verified |
| H2-07 | EU AI Act compliance packaging finalized for Aug 2, 2026 enforcement; 3+ regulated design partners engaged | H1-05 + H2-01 | Packaged artifact bundle, SDK, runbook, and customer playbook; legal review complete; at least one regulated partner running the full pipeline |
| H2-08 | Prompt-to-spec handoff hardening: vague goals → reviewable specs routinely; `aragora spec` CLI upgrade | Existing `aragora/prompt_engine/` | `aragora spec "<vague request>"` produces a spec ready for debate dispatch in ≤60s; benchmarked against a corpus of 20 real founder requests |
| H2-09 | Operator workbench read surface: `get_run`, `list_runs`, stage timeline viewer | H2-03 backbone | Single-page operator view shows every run's intake → receipt chain; filterable by status/taint/date; exportable as JSON |
| H2-10 | Session circuit-breaker hardening: auth-state pinning production-proven across ≥3 lanes; provider rotation on 401/403 automatic | Existing `aragora/swarm/session_circuit_breaker.py` | Runbook: any 401/403 auto-pins session; OpenRouter fallback triggers on quota errors; captured in ledger |

### H2 exit criteria

- ≥70% zero-rescue benchmark corpus sustained for 7 consecutive days
- 2-3 design partners with weekly cadence and documented value evidence
- 12h multi-host soak passes 3 consecutive runs
- RunLedger is the canonical record for every golden-path run
- EU AI Act package ready for sale
- At least 1 paying or committed customer (even at design-partner rates)

### H2 explicit non-goals

- No live ERC-8004 writes (shadow only)
- No Canvas GUI beyond the operator workbench read surface
- No Chief-of-Staff portfolio loops
- No Nous Hermes / Pi Agent / Anthropic Agent Framework integration (planning only; H3 scope)
- No Skills Marketplace public launch

## Horizon 3 (Day 91-365): Organization Substrate Foundation

**Single goal:** Aragora behaves as a Chief of Staff on bounded domains for at least one design partner; SMB OS packaging ships; unified DAG canvas is live; heterogeneous agent marketplace opt-in is operational; DIC-23..28 synthesis layer is in production; first 10 paying customers.

### H3 concrete deliverables

| ID | Deliverable | Depends on | Acceptance |
|---|---|---|---|
| H3-01 | Unified DAG workbench (Track C3 + C4): full idea → goals → actions → orchestration canvas backed by live runtime truth | H2-09 read surface | One-page canvas shows portfolio across projects; drill-down to any run's receipt chain; intervention controls (pause/retry/quarantine); dissent/evidence surfaced at every handoff |
| H3-02 | Permissioned memory + broad ingestion fabric (Track D complete): memory items carry trust tier, provenance, access boundaries; large-context packing benchmarked; stream ingestion from repos/docs/APIs/chat/receipts | Existing `aragora/memory/`, `aragora/knowledge/` | Memory export roundtrips without provenance loss; large-context pack benchmarked against smaller contexts on decision-quality metric; ingestion supports ≥5 source types with normalized provenance |
| H3-03 | SMB OS packaging: `aragora-core` decoupled from `aragora-enterprise`; <10min onboarding; Docker-compose recipe for founder/SMB deployment | Existing `aragora/` monorepo refactor | `pip install aragora-core && aragora quickstart` succeeds in ≤10min on a fresh machine; no Postgres/Redis/Kafka requirement for the core tier; enterprise tier adds those via separate compose file |
| H3-04 | Heterogeneous agent marketplace opt-in: Claude Code, Codex, Gemini CLI, Grok CLI, OpenClaw, Nous Hermes, Pi Agent, Anthropic Agent Framework, LangGraph, AutoGen, CrewAI all plug-in surfaces | H2-06 parity; existing `aragora/agents/api_agents/` | Each agent type is a registered adapter following the same contract; any user can opt in to a specific agent mix for their workspace; at least 3 novel (non-current) agents land |
| H3-05 | ERC-8004 graduated writes: shadow → low-stakes live writes with multisig for any consequential decision; hot wallet replaced with multisig + offline approval for consequential chain writes | H2-05 shadow proof | Reputation registry writes confirmed on-chain for ≥100 decision receipts; no hot wallet in server for consequential writes; all live writes gated by human approval receipt |
| H3-06 | Nomic Loop adoption: all autonomous self-improvement work routes through the reliability substrate (contracts, preflight, ledger, self-heal); no second reliability stack | H2 substrate proof | Nomic-generated work items carry WorkerContract; success rate matches benchmark corpus; ledger shows Nomic and swarm side by side |
| H3-07 | DIC-23..28 Dialectical Runtime synthesis layer in production: runtime loop (DIC-23) report-only → active; genealogy ledger (DIC-24) live; stress-test (DIC-25) operator-curated; coherence monitor (DIC-26), crux arbitration (DIC-27), crux gardening (DIC-28) operational | DIC-20/21/22 production-green from H2 | All six modules shipping; truth map shows dialectical events, genealogy, coherence, arbitrations, gardening reports; no live queue mutation without explicit flag |
| H3-08 | Chief-of-Staff loop: vague goal → portfolio plan → delegated work → periodic synthesis → human approval checkpoint for bounded domain | H3-01, H3-02, H3-07 | One design partner runs a ≥30-day engagement where Aragora takes vague monthly goals and produces receipts of delegated work with ≥70% approval rate; tradeoffs surfaced explicitly |
| H3-09 | Agent-as-Consumer Substrate (Track G, AGT-01..06): agent registration, capability discovery, compute-budget billing, reputation read/write, productivity metric (VIAH), CruxDetector activation in live debates | Existing Track G planning docs | A2A registration endpoint live; at least 3 external agents registered; VIAH metric published weekly; CruxDetector default-on for production debates |
| H3-10 | First 10 paying or committed design-partner customers | H2 + H3 deliverables | 10 organizations with documented recurring use; ≥5 in regulated verticals; public case studies for ≥3; revenue or pre-revenue LOIs signed |

### H3 exit criteria

- Chief-of-Staff loop works for ≥1 design partner with ≥70% approval rate
- SMB OS core tier has ≥50 self-installed instances across ≥20 organizations
- 10+ paying/committed customers
- DIC-23..28 observable in at least 3 real proof-carrying code units
- Heterogeneous agent marketplace has ≥3 non-incumbent agents in production use

### H3 explicit non-goals

- No cross-organizational federation (P5)
- No international EU data residency (P3 backlog)
- No compute escrow / crypto settlement
- No Skills Marketplace public launch beyond initial pilot
- No SOC 2 Type II audit kickoff (backlog until Chief-of-Staff is stable)

## Deferred / Maximalist Backlog (Preserved, Not Scheduled)

These items are tracked explicitly so the maximalist vision is not lost. They remain planning-only until their prerequisite horizon proves out.

| Item | Canonical home | Activation gate |
|---|---|---|
| Cross-organizational federated debate | FEATURE_GAP_LIST P5; `AGENT_CIVILIZATION_SUBSTRATE.md` | H3 exit + multi-tenant isolation proof |
| SOC 2 Type II audit engagement | FEATURE_GAP_LIST P2 | H3 exit + first enterprise customer commit |
| International EU data residency | FEATURE_GAP_LIST P3 | H3 exit + EU customer demand signal |
| Compute escrow / crypto settlement mechanism | FEATURE_GAP_LIST P3; `contracts/erc8004/` | H3-05 live writes stable for ≥90 days |
| Skills Marketplace public launch | FEATURE_GAP_LIST P3 | H3 pilot + external contributor demand |
| Meta-improver for debate protocols (A/B) | FEATURE_GAP_LIST P4 | H3 exit + DIC-23..28 operational |
| STOP N-candidate for Nomic Loop | FEATURE_GAP_LIST P4 | H3-06 Nomic adoption stable |
| Decision-quality delta benchmark beyond TW-02 | `docs/benchmarks/decision-quality-delta-spec.md` | H3 exit |
| Full prediction-market reputation flow (AGT-03..05 production) | `2026-04-17-prediction-market-validation.md` | H3-09 AGT-01..06 stable |
| On-premise deployment productization | FEATURE_GAP_LIST P3 | H3 + enterprise customer commit |
| 10+ agent coordinated debates at scale | FEATURE_GAP_LIST P1 | H2 parity + H3 marketplace stable |
| Operator-scale Kubernetes via `aragora-operator` | `aragora-operator/` package | H3 + enterprise customer commit + ERC-8004 live stable |

## Weekly / Monthly Cadence

### H1 weekly cadence (Week 1-4)

| Day | Activity | Owner | Time |
|---|---|---|---|
| Mon | Benchmark scorecard review + corpus triage | founder | 30min |
| Tue | Dogfood debate #1 (real business question) | founder | 60min |
| Wed | Substrate PR review + merge | founder | 60min |
| Thu | Dogfood debate #2 (real technical question) | founder | 60min |
| Fri | Weekly check: zero-rescue %, needs_human drop %, ledger surface vs shell disagreement | founder | 45min |
| Sat | Optional blog post drafting | founder | 90min |
| Sun | Autonomous overnight runs continue | Mac Studio | 0 |

Total founder time: ≤6 hours/week

### H2 weekly cadence (Week 5-12)

| Day | Activity |
|---|---|
| Mon | Design-partner check-in #1 |
| Tue | Design-partner check-in #2 |
| Wed | RunLedger / backbone integration review |
| Thu | Heterogeneous parity smoke test |
| Fri | Weekly scorecard + benchmark delta |
| Sat | EU AI Act packaging session |
| Sun | Soak test runs |

### H3 monthly cadence (Month 4-12)

| Week | Activity |
|---|---|
| 1 | Design partners (Chief-of-Staff engagement) |
| 2 | Canvas UI work / Track C |
| 3 | Memory + ingestion fabric / Track D |
| 4 | Agent marketplace onboarding |
| Monthly | Paying customer pipeline review; DIC-23..28 rollout review |

## Integration With Existing Docs

- This doc **operationalizes** ARAGORA_EVOLUTION_ROADMAP.md. That doc's 30/90/365 outcome map and 5 tracks remain authoritative. This doc's tables are the concrete deliverables that satisfy those outcomes.
- The H1/H2/H3 exit criteria align with the stage-exit criteria in CANONICAL_GOALS.md.
- FEATURE_GAP_LIST.md P0-P5 placements remain correct; this doc adds the H1/H2/H3 timing overlay on top of those placements.
- The Dialectical Runtime synthesis layer (DIC-23..28, PR #6224) is explicitly H3-07, staged behind DIC-13..22 production-green in H2.
- EPISTEMIC_CI_AND_CRUX_ENGINE.md's activation gate remains authoritative; this doc schedules it into H2/H3.
- NEXT_STEPS_CANONICAL.md remains the current-tranche source of truth; this doc is the forward-looking overlay.

## Risk Register

| Risk | Horizon | Mitigation |
|---|---|---|
| H1 corpus grows beyond what the founder can triage daily | H1 | Cap corpus at 30 tasks; auto-close untriaged issues after 72h |
| Dogfood debates consume founder time without producing content | H1-H2 | Hard 60min cap per debate; queue drafts for batch-publish |
| EU AI Act deadline slips or vendor selection delays | H2 | Package what exists; certification sequencing already deferred per NEXT_STEPS_CANONICAL |
| Design partners churn before providing feedback | H2 | Target 3+ partners so 1-2 churn is absorbable; document lessons even from churn |
| Canvas UI work balloons before backbone is stable | H3 | Tie C1 (canonical graph model) to live RunLedger; no canvas work until H2-03 lands |
| Heterogeneous agent marketplace fragments the debate contract | H3 | One `AgentContract` interface; non-conforming agents remain experimental |
| ERC-8004 live writes introduce hot-wallet risk | H3 | Multisig-required for consequential writes; offline approval workflow |
| DIC-23..28 becomes a philosophy project instead of product | H3 | First ship is the smallest useful slice (DIC-23 report-only + DIC-24 genealogy view) |
| Nomic Loop generates noise instead of product value | H3 | Nomic work routes through contract admission; rejected admissions count as failures |

## Success Definition

The 365-day success criteria are simple and measurable:

1. **Reliability wedge proven** — ≥70% zero-rescue on the benchmark corpus sustained for ≥30 days
2. **External validation** — ≥10 paying or committed design-partner organizations with documented recurring use
3. **Decision integrity lives** — DIC-13..28 observable across ≥10 proof-carrying code units with live receipts
4. **Substrate is boring** — multi-host soak passes routinely; ledger is the single source of truth; operator intervention is rare and rewarding when it happens
5. **Maximalist vision believable** — the narrow wedge has earned the right to the broader ambition

If the 365-day success criteria are met, the broader organization-substrate ambition becomes credible as the **next** year's work. If they are not met, the diagnosis will be in the ledger, and the roadmap revision will be evidence-driven rather than hope-driven.

## GitHub Tracking

Concrete issues are created for H1 execution. H2 and H3 items are tracked here as the authoritative plan; issues for them will be created as their activation gates open.

- H1 epic: (created in this tranche)
- H1-01..H1-08: (created in this tranche)
- H2 epic: planning only, tracked in this doc
- H3 epic: planning only, tracked in this doc
- Maximalist deferred backlog: planning only, tracked in this doc and FEATURE_GAP_LIST

## Closing Frame

The maximalist vision is the destination. The wedge is the only credible path. Every horizon is a proof ramp that earns the right to the next.

The system is not trying to be everything at once — it is building the boring reliable base so that everything else becomes possible on top of it. That is not a compromise with ambition; it is the operational form of ambition in a world of limited time and bounded skill.
