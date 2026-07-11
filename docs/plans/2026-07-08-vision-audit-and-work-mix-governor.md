# Vision Audit & Work-Mix Governor — Findings, Decisions, and 90-Day Plan

**Date:** 2026-07-08
**Status:** ACTIVE — Phase 1 in flight
**Origin:** Nine-agent audit (six repo readers + thesis critic + web-grounded market scan + plan architect, ~750K tokens), synthesized by Claude Fable 5, amended and approved by founder same day.
**Related:** epic #8762 (Close the Loop), epic #8223 (ODR), issue #9007 (stage-gate freeze violation), issue #8858 (outsider verification run), epic #8972 (harness self-improvement).
**Tier:** this document is docs-only (Tier 0). Child work items carry their own tiers (see §8).

---

## 1. What the audit found

### 1.1 The defensible thesis (verified against code)

> **Heterogeneous frontier models adversarially review a decision; disagreement is
> preserved, not averaged away; the output is a signed, dissent-preserving Decision
> Receipt an auditor can verify offline without trusting Aragora.**

Every word of that sentence is backed by shipped code: the debate engine
(~197 modules under `aragora/debate/`, 10 real consensus modes including working
simplified-PBFT in `byzantine.py`, a 2,322-line calibration-aware team selector), the
receipt stack (`aragora/gauntlet/receipt_models.py` with `ConsensusProof`, dissent
records, provenance, SARIF/PDF/compliance exports, HMAC/RSA/Ed25519 signing backends),
the ELO+Brier calibration flywheel, Knowledge Mound (staleness, contradiction
detection, confidence decay), and the settlement stack in `scripts/` (tiered merge
gates, head-bound quorum evidence, severity-gated dissent, adjudicator, merge executor,
harvest engine).

### 1.2 Repo-state findings (evidence, not vibes)

- **Work-mix imbalance.** Over the Jun 29–Jul 8 window (150 commits): ~46–52% loop/gate
  machinery, ~38% product (half of that internal refactoring), ~10% maintenance. Of the
  last 30 merges, ~2 touched product runtime code. Product PRs are precisely the ones
  that age 1–4 weeks awaiting settlement (#8406, #8519, #8652, #8766, #8809, #8823,
  #9022, #9030, #9033) while docs and gate tooling merge same-day.
- **The repo already detects this**: issue #9007 (`stage-gate-drift`) flags
  "governance-substrate freeze violated" — detection exists, enforcement does not.
- **Surface-to-loop ratio.** 144 top-level modules / ~1.98M LOC; 60% of modules have
  ≤10 Python files; six engines are "built but dormant" (crux detector complete with
  zero external surface; Pareto router shipped but records no rationale; inbox wedge
  CLI-complete but the surface was rejected — retest only as web GUI).
- **Doc conflicts.** Five coexisting positionings across README / WHY_ARAGORA /
  COMMERCIAL_OVERVIEW / EXTENDED_README / CLAUDE.md; RLM described as "compresses
  context" in EXTENDED_README vs "NOT compression" everywhere else; no pricing/SKU
  stated anywhere (WHY_ARAGORA points to COMMERCIAL_OVERVIEW for pricing; it contains
  none); metric drift across docs (35 vs 43 vs 46 agent types).
- **The single highest-leverage code fix outstanding:** the quorum rerun budget in
  `aragora/swarm/merge_quorum_reconcile.py` is keyed by **head SHA**, so every repair
  commit resets it — the mechanical root of the nitpick treadmill (measured: 4 PRs,
  78 commits, 0 merges; #8595 made 16 near-duplicate commits in <2h). Keying it by
  **PR** and routing budget depletion to the M0 adjudicator (#8749) closes the loop.

### 1.3 Market findings (web-verified, July 2026)

- **The integrated loop is rare in the world.** Roundtable/SynthBoard/Opper ship
  debate without cryptographic evidence; EQTY Lab / Agent Receipts / EMILIA ship
  receipts without deliberation; Credo AI/Holistic own compliance workflow with
  neither. Nobody found combines deliberation-quality assurance with tamper-evident
  evidence *of that deliberation*.
- **Model-quorum merge gates for agent-authored PRs do not exist commercially or in
  OSS.** GitHub's own guidance and CodeRabbit's governance frameworks assume human
  merge authority. Aragora's tiered gate running on its own 4,200-file codebase is,
  on public evidence, unique. Nearest research analogue ("Kitchen Loop," 1,094 merged
  PRs) uses test-oracle verification, not adversarial multi-model settlement.
- **Crucible (Roundtable Labs) shuts down Aug 31, 2026** — $19–45/mo adversarial
  decision briefs. Validates demand for the artifact; falsifies the thin-wrapper
  business model. The enterprise version (org-knowledge grounding + receipts +
  channel delivery) is an open hole shaped exactly like Aragora's stack.
- **EU AI Act timing:** the Digital Omnibus (agreed May 7, adopted June 2026) delayed
  high-risk obligations to Dec 2027/Aug 2028, **but GPAI + Article 50 enforcement and
  fines still begin Aug 2, 2026**, and no official conformity templates exist — a
  standard-setting window for the Open Decision Receipt.
- **Threats (12-month):** CodeRabbit flipping multi-model ensemble review from
  advisory to gating (biggest); frontier labs shipping native council+audit-log;
  open receipt specs commoditizing signature plumbing (moat must stay debate quality +
  knowledge grounding + calibration data); EQTY's hardware attestation setting a
  higher proof bar in gov/defense.
- **Mechanism validation:** hollow-consensus detection, prover-estimator, calibrated
  team selection exist elsewhere only as 2026 research papers (RAudit, Elenchus);
  single-LLM-judge gates are documented adversarially fragile ("one-token" attack) —
  heterogeneous quorum is a security property no eval vendor markets.

## 2. Founder decisions (2026-07-08) — binding amendments to the audit recommendations

1. **Focus, don't amputate.** The near-term focus is the defensible receipt thesis
   (§1.1), but the maximalist vision is **retained in full** and built out piece by
   piece, section by section, in integrated fashion. The roadmap/thesis/vision must
   not lose aspects while one aspect carries the near-term claim. Concretely: the
   Tool→Organization-Substrate ladder and the eight foundational pillars move from
   the README to `docs/vision/MAXIMALIST_VISION.md` — moved, not deleted.
2. **Integrate, don't archive.** Modules are removed only when **actively harmful
   AND without future prospect of integration**. The audit's "cut list" is rejected
   as cuts and converted to a staged-integration track: **blockchain/ERC-8004 is
   part of the vision** (skin-in-the-game enforcement for the epistemic engine);
   marketplace, genesis, verticals, broadcast, tournaments remain dormant-by-decision
   with explicit integration criteria. Emphasis is **add, integrate, put emphasis
   on** — not archive or deprecate.
3. **Accepted from the audit:** COMMERCIAL_OVERVIEW's discipline becomes the public
   position; the Nomic Loop is narrated as dogfood evidence — the truthful version
   ("our gate was so genuinely adversarial it deadlocked our own factory for a month;
   here are the receipts") over the success narrative; surface numbers (144 modules /
   223k test functions) are not marketing; the RLM contradiction is fixed (true RLM
   is context-as-REPL-variables, **NOT compression**; compression is only a fallback).

## 3. Strategy restatement

- **Near-term claim (earned, public):** the defensible sentence in §1.1, expressed
  as the CI governance gate + ODR spine (#8223), with the settlement stack as the
  second proven surface ("merge governance for agent fleets").
- **Long-term vision (retained, staged):** `docs/vision/MAXIMALIST_VISION.md` —
  the five-stage ladder, the eight substrate pillars, agent-civilization tracks
  (AGT-01..06), epistemic CI (DIC-*), skin-in-the-game reputation (ERC-8004),
  marketplace/verticals — each gated by capability checkpoints (CP-1..5) and
  integration criteria, never by deletion.
- **The loop is the evidence, not the pillar.** Aragora governing Aragora's own
  development is the demo and the calibration-data flywheel; it is sold as proof,
  not as the product's fifth pillar.

## 4. The work-mix governor (the mechanical fix for "loop producing loop")

A **work-mix governor and throughput ledger** points the existing machinery at the
product. Net-new code is deliberately tiny; everything else is composition of shipped
components.

### 4.1 The daily macro-cycle

| Phase | Existing component | Missing glue (this plan) |
|---|---|---|
| SENSE | `scripts/agent_session_digest.py`, metrics drift gate, `gh` queue snapshot | ledger aggregator → `.aragora/throughput/ledger.jsonl` |
| STEER | `scripts/fable_goal_cycle.py` → `scripts/consult_claude.py`; MetaPlanner risk gating | `scripts/work_mix_gate.py` (advisory first) |
| EXECUTE | `scripts/self_develop.py --auto` → HardenedOrchestrator; boss_loop; lane caps; `aragora/swarm/wip_budget.py` | none — **no new orchestrators** (charter #8927) |
| SETTLE | `collect_quorum_evidence.py` (3 flags on) → `settle_pr.py` → `merge_executor.py` | **PR-keyed rerun budget** fix (§1.2), depletion → adjudicator |
| HARVEST | `harvest_outcomes.py`, park policy (≤3 attempts/head) | schedule weekly under launchd |
| ACCOUNT | founder decision queue, steering conductor | `scripts/weekly_digest.py` |

### 4.2 Work classes and the budget rule

Every merged PR is classified — `product-core` / `product-proof` / `substrate` /
`maintenance` — on **diff content against a checked-in module map** (anti-gaming
baseline), with LLM classification layered on in Phase 2 per the "use real
intelligence, not regex" principle (the deterministic path map is the cross-check,
the LLM is the classifier of record once wired).

Rolling 7-day merged-PR mix must satisfy:

- `product-core` + `product-proof` **≥ 50%**
- `substrate` (gates, conductors, quorum tooling, runbooks/charters about the loop)
  **≤ 25%**
- Exemptions: main-red repair and security fixes are always admissible.

On breach, the gate writes `.aragora/throughput/substrate_freeze.marker`; while it
exists, boss_loop selection and `fable_goal_cycle` refuse substrate-class goals
(parked with label `substrate-frozen`) and the consult packet is force-prefixed with
the standing redirect: *"substrate is frozen; propose only goals from the product
target list."* This mechanizes what #9007 detects but cannot enforce.

**Inventory gate:** if open PRs > 40 or stranded branches > 25, the loop enters
**drain-only mode** (settlement, harvest, and drain passes only; no new goals) —
the 2,749-branch lesson as a standing constraint.

### 4.3 The product target list (allowed goal pool while frozen, ordered)

1. ODR tranche (epic #8223): ODR-2 Ed25519 signing #8225, ODR-3 PyPI verifier #8226,
   ODR-4 crux-finder exposure #8227, ODR-5 calibration API #8229, ODR-6 Art. 14
   attestation #8230, ODR-7 Rekor anchoring #8231.
2. EU AI Act bundle 90→100 before **Aug 2, 2026** (GPAI/Art. 50 enforcement unchanged
   by the Omnibus).
3. #8858 — one real outsider verification run (priority:critical).
4. The stuck product-PR tail, dispositioned by name: #8406, #8519, #8652, #8766,
   #8809, #8823, #9022, #9030, #9033 — settle or park-with-receipt.
5. Dormant-engine **integration** (per founder decision §2.2): crux detector into
   receipts/API; Pareto router recording routing rationale; inbox wedge retest as web
   GUI; staged integration criteria for blockchain/marketplace/verticals (§7).
6. SOC 2 pentest engagement prep (vendor selection is human-owned).

### 4.4 Throughput ledger and kill switches

Single append-only ledger `.aragora/throughput/ledger.jsonl`, fed by merge receipts,
settlement receipts, harvest outcomes, and the nightly SENSE snapshot.

| Metric | Target | Kill switch |
|---|---|---|
| Settlement latency (ready→merged, Tier 0-2, p50) | ≤ 48h | p50 > 7 days → drain-only mode |
| Merge rate | ≥ 10/week | informational (volume is not the constraint) |
| **Product share of merges (7-day)** | ≥ 40% | < 20% two consecutive weeks → substrate freeze |
| **External artifacts published** | ≥ 1 / 14 days | 0 in 30 days → demote to Phase 1, human review |
| Revert rate / main-red | ≤ 2% of merges; MTTR < 24h | main red > 24h → all lanes to repair (halt-file, exists) |
| Evidence efficiency | ≥ 6 PRs settled/day; head-drift invalidations < 10% | > 25% → quiesce rule + single-writer lease (#8852) |
| Park recurrence | ≤ 3 attempts/head SHA (2026-07-07 policy) | breach = queue-selection bug → auto-file issue |
| Self-repair ratio (`fix(...)` PRs to scripts/+swarm/) | < 15% of merges | > 30% → "remove the fragile abstraction" review |

Physical kill switches (all existing patterns): merge-executor `--disarm-file`
(one-way, human-only removal), main-red halt-file (human deletes), substrate-freeze
marker (auto-set on mix breach).

### 4.5 Human touchpoints (exactly these)

1. Tier 3–4 settlement (`settle_tier4_pr.py`, exact-head operator comment) — never
   automated. Includes `scripts/nomic_loop.py`, merge-authority code, CLAUDE.md,
   workflows, branch protection.
2. Arming `merge_executor.py --apply` (per `docs/AGENT_OPERATING_CONTRACT.md`).
3. Founder decision queue — the adjudicator escalates **one named crux per deadlock**.
4. Weekly 30-minute strategy review of the digest: throughput table with WoW deltas,
   work-mix breakdown, external artifacts published + next due, stuck-PR tail with
   named blockers, kill-switch/freeze state, 5 randomly sampled merge classifications
   (anti-gaming spot-audit), top-3 proposed goals with risk classification, charter
   deviations.
5. Monthly: substrate-budget re-tune, charter delta approval, pentest/pilot progress.

## 5. 90-day rollout

**Phase 1 — Harden what runs (days 0–21).**
Ship `aragora/nomic/work_mix.py` + `aragora/nomic/throughput.py` (library, tested),
`scripts/work_mix_gate.py` in **advisory/log-only** mode, `scripts/throughput_ledger.py`,
`scripts/weekly_digest.py`. Confirm all 3 settlement CI flags on. File (Tier-4,
human-preapproved before implementation) the PR-keyed rerun-budget fix. Nightly
pristine-`origin/main` full-shard run writing the halt-file (hidden-red
countermeasure). Disposition the §4.3(4) stuck-PR tail.
*Exit:* 7 consecutive green-main days; ledger producing daily records; product share
measured (baseline, not judged); ≥ 6 of 9 stuck product PRs dispositioned.

**Phase 2 — Expand autonomy (days 21–50).**
Flip `work_mix_gate` to enforcing; wire LLM classification (classifier of record) with
the path map as cross-check; human arms merge_executor `--apply` for Tier 0-2
(max-merges 5/pass); nightly `fable_goal_cycle` → `self_develop --auto` under the
gate; `generate_boss_issues` filtered to product classes; weekly harvest under launchd.
*Exit:* two consecutive weeks of ≥ 10 merges/wk at ≥ 40% product share, revert ≤ 2%,
≤ 1 babysitting intervention/wk, zero substrate-freeze false halts.

**Phase 3 — External proof (days 50–90, ends before Aug 2 + buffer).**
The loop's goals become artifacts: ODR-3 verifier live on PyPI + #8858 outsider run
published; EU AI Act bundle published before Aug 2; one public benchmark or
external-repo gate demo ("Factory with provenance" — run the CI review gate on an
external OSS repo and publish the receipts); inbox wedge web-GUI retest; pentest
kicked off.
*Exit:* ≥ 3 outsider-consumable artifacts in the `docs/artifacts/` registry; one
design-partner demo path timed < 10 min.

## 6. Failure modes → mechanical countermeasures

1. **Loop-producing-loop** → work-mix budget + freeze marker; #9007-style stage gate
   made blocking for substrate-labeled PRs during freeze; classifications
   receipt-logged and spot-audited weekly.
2. **Branch/PR pileup** → inventory gate → drain-only mode; weekly harvest; deletions
   only via `safe_worktree_cleanup.py` / `codex_worktree_autopilot.py cleanup` with
   manifests (~90d recovery); never bulk `rm -rf`.
3. **Reviewer deadlock / nitpick treadmill** → three settlement flags stay on;
   PR-keyed rerun budget; depletion forces adjudicator disposition
   {MERGE_AS_IS, ONE_BOUNDED_ROUND, CLOSE, RESTRUCTURE}; reviewer-failure taxonomy
   (#8869) feeds selection.
4. **Silent main-red (path-gated shards)** → nightly pristine-main full run →
   halt-file; merge executor re-checks main health before every merge (exists).
5. **Metric gaming** → classification from diff paths against a checked-in
   runtime-module map, not labels/messages; docs-only merges counted separately;
   weekly random-sample audit; METRICS 0.5% drift gate (exists).
6. **Head-drift treadmill** → quiesce rule (skip PRs pushed < 30 min ago) + ≤3
   attempts/head park policy + single-writer PR lease (#8852).
7. **Halting/babysitting** → relay-park-don't-halt; bounded per-cycle budgets;
   disarm-file as the only hard stop.
8. **Fragile-abstraction treadmill** → any component receiving ≥ 3 fix-PRs in 30 days
   (detectable from the ledger) auto-freezes further patches and files a Tier-3
   "remove or replace" decision to the founder queue.

## 7. Staged-integration track for dormant modules (integrate, don't archive)

Per founder decision §2.2 each dormant module gets an integration criterion instead
of a cut. None of these are deprecated; none carry `boss-ready` until their gate opens.

| Module | Vision role | Integration criterion (gate to activate) |
|---|---|---|
| `aragora/blockchain/` (ERC-8004) | **Skin-in-the-game enforcement for the epistemic engine**: claims → stakes → resolution → reputation deltas → dispatch eligibility | CP-4 (a reputation delta changes real dispatch); near-term anchoring need is served by ODR-7 Rekor; ERC-8004 activates when staking is wired to debate outcomes on a live network |
| `aragora/marketplace/` + skills marketplace | Distribution surface for templates/protocols once external users exist | First external design partner actively using templates |
| `aragora/verticals/` | Domain specialists (healthcare/legal/financial/accounting) behind the receipt wedge | First vertical pilot signed; activate only that vertical |
| `aragora/genesis/` | Agent-evolution frontier (AGT-track) | CP-2..CP-5 checkpoint progression |
| `aragora/broadcast/` | Post-decision communication surface (podcast/audio) | A paying user asks for it |
| `aragora/tournaments/` | Structured competition layer over ELO | Fold into `aragora/ranking/` when touched next; no standalone investment |
| Crux detector | **Immediate**: ODR-4 #8227 — expose in receipts/API | Already on the product target list (§4.3) |
| Pareto router | **Immediate**: record routing rationale in receipts | Product target list (§4.3) |
| Inbox wedge | SMB operator surface | Retest as polished web GUI only (founder, May 14) |

## 8. Work items

Tracked under the execution epic (see epic issue for live status):

| Item | Tier | Phase |
|---|---|---|
| Plan doc + vision doc + README repositioning + RLM fix (this PR) | 0 | 1 |
| `aragora/nomic/work_mix.py` + `throughput.py` + tests | 1 | 1 |
| `scripts/work_mix_gate.py` / `throughput_ledger.py` / `weekly_digest.py` (advisory) | 1 | 1 |
| PR-keyed rerun budget fix (`merge_quorum_reconcile.py`) + adjudicator routing | **4 — human preapproval before implementation** | 1 |
| Nightly pristine-main full-shard run + halt-file wiring | 2 (launchd install is operator action) | 1 |
| Stuck product-PR tail disposition (9 PRs, §4.3) | per-PR | 1 |
| Enforcing mode + boss_loop/fable_goal_cycle wiring + LLM classification | 2–3 | 2 |
| Weekly harvest under launchd | 2 | 2 |
| External-proof artifacts (ODR-3, #8858, EU AI Act bundle, external-repo demo) | existing epics | 3 |

## 9. What this plan is not

- Not a rewrite: no new orchestrators, stores, or conductors (charter #8927).
- Not a cut list: no module is archived or deprecated by this plan (§2.2).
- Not a bypass: the merge-quorum gate remains the sole settlement authority; Tier 3–4
  remain human-settled; editing merge-authority code remains Tier 4.
