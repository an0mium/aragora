# Hybrid Orchestrator Plan — mode-selecting, Pareto-routing, Fusion-cross-checked

**Status:** design validated (understand → design → adversarial critique, Jun 15 2026). Execution-ready for the Tier-1 foundation; Fusion activation gated on slug/pricing verification.

**End goal (founder):** a hybrid gastown / factory-mission / claude-code-dynamic-workflows orchestrator with (1) multi-model **adversarial cross-checking** via OpenRouter Fusion **and** aragora-native verification, (2) intelligent **automated mode-switching** between dynamic-workflow / goal-anchored / agent-teams patterns, and (3) intelligent **automated Pareto routing** to the optimal model per unit of work.

## Ground-truth anchors (verified)
- **No `aragora/orchestration/` package.** Extend the existing `aragora/routing/unified_router.py::UnifiedDecisionRouter` (Gateway business-criteria → Core engine selection) — via a **new method**, never by mutating `route()`/`_route_auto()` (existing callers depend on it).
- **`fusion_tiebreak.py` is NOT on main** — it ships with open PR-3 (#8446, `50178e6d94`).
- **`FAMILY_PROVIDERS` (12 families) excludes fusion today** — the structural non-quorum exclusion is real. Keep it; add a regression test.
- `FUSION_MODEL="openrouter/fusion"` (`openrouter.py:57`) slug **unverified**; pricing **assumed** `8.00/32.00` per-1M (`usage.py:88-99`, `TODO(fusion-pricing)`).

## Critique corrections folded in (do NOT skip)
- **Cost-aware scoring already exists** (`selection.py:150` + `_score_for_task()` ~`:623`, ~20% weight). Re-scope to *verify/tune* Fusion's 4.5 cost_factor penalty — do NOT add a duplicate weight (would double-count).
- **`aragora/tools/deep_research.py` does not exist** (`tools/` = `code.py`, `__init__.py` only; deep-research is a *skill*, not an importable module). Drop the deep-research half of PR-4 or scope a separate investigation first.
- **Don't break signatures:** `DomainDetector.detect()` returns `list[tuple[str,float]]` — add `estimate_complexity()`/`detect_with_complexity()`. `get_provider_hints()` returns `dict[str,float]` — add `get_provider_hints_rich()`. New methods, not mutations.
- **Kill-switch:** the orchestrator seam (PR #13) ships behind `enable_hybrid_routing` (default OFF).
- **Testable safety:** add a test that no orchestrator/mode-selector/Pareto path can call merge/`settle_tier4`; and PR #11 must snapshot the quorum tally *before* the Fusion call and assert equality after + assert `"fusion" not in FAMILY_PROVIDERS`.
- **Default mode** for ambiguous-not-contested → `dynamic_workflow` (cheapest), reserve `agent_teams` for explicit risk/consensus/contested triggers.

## Architecture (one orchestrator, three never-collapsed decisions)
`UnifiedDecisionRouter` (extended) → (1) **Mode selector** `modes/selector.py` (NEW): pattern ∈ {dynamic_workflow→`workflow/engine.py`, goal_anchored→`nomic/autonomous_orchestrator.py`, agent_teams→`debate/orchestrator.py` Arena}; (2) **Pareto router** `routing/{cost_quality_optimizer,provider_router,selection}.py` via new `TaskRoutingContext`; (3) **Execution mode** `pipeline/execution_mode.py` (AUTONOMOUS|INTERACTIVE by risk tier). → **Adversarial cross-check layer**: native (`consensus.py`, `verification/formal.py`, `review/evidence.py` — **counts** toward quorum) + Fusion (`openrouter.py` FusionAgent, `fusion_tiebreak.py` — **disclosed, NON-counting**). → **Authority: `swarm/quorum_evidence.py` merge-quorum gate, unchanged, sole settlement path.** Every decision receipted (`gauntlet/receipts.py`).

## Tiered PR plan (critical path: 2 → {4} → 1 → {3,5} → {7,8,9} → 13 → 11 → 12 → 14 → 16)
**Start here (pure, zero-risk, no creds):**
- **PR #2** (T1): `TaskRoutingContext` dataclass — `routing/config.py` + tests.
- **PR #4** (T1): `OperationalModeSelector` + `ModeDecisionContext` (heuristic, pure, unwired) — `modes/selector.py` + tests.

**Then (Tier 1):**
- **PR #1**: Fusion slug+pricing **verification** — one live `openrouter/fusion` call (OPENROUTER_API_KEY via Secrets Manager), assert `choices[0].message.content` shape, replace assumed rates, update comment. Unblocks all Fusion activation.
- **PR #3** (re-scoped): verify/tune existing cost term; confirm Fusion penalized. NO new weight.
- **PR #5**: `estimate_complexity()` (new method, no signature break).
- **PR #6a**: `quorum_evidence.py` source comment documenting Fusion exclusion (non-protected, ships freely). **PR #6b** (separate, approval-gated): CLAUDE.md rule line — #11 must NOT depend on #6b.

**Tier 2:**
- **PR #7** budget gate `EstimatedDebateCost` (native-only estimate until D0). **#8** `get_provider_hints_rich()`. **#9** request-time Pareto (bound latency / keep TTL fallback). **#10** wire mode selector into `ArenaBuilder`/`PromptBuilder`. **#11** Fusion tiebreaker (ship `fusion_tiebreak.py` + wire split-quorum as non-counting + the tally guard tests). **#12** Fusion cost-budget enforcement (`cost_tracker.py`). **#13** hybrid orchestrator seam — **new method** on `UnifiedDecisionRouter` behind `enable_hybrid_routing` (OFF), no-regression test on existing AUTO path.

**Needs design settled / human risk acceptance (T2–3):**
- **#14** PR-2 Nomic Fusion opt-in (planning/verify flags; provenance-tag Fusion content, exclude from Nomic consensus). **#15** PR-4 judge only (drop deep-research until a module exists). **#16** activation rollout (gated on PR #1 + #12 + OPENROUTER_API_KEY-in-Secrets-Manager check). **#17–19** DAG mode_sequence, auto-escalation, KM feedback loop.

## Safety / TOS guardrails (binding)
No subscription pooling (Fusion only via OpenRouter metered API). All keys via Secrets Manager, never raw env. Fusion flags default-OFF/BETA; slug verified before any flip. Fusion structurally excluded from `FAMILY_PROVIDERS` — every Fusion output disclosed/advisory/non-counting. merge-quorum gate is the sole settlement authority; orchestrator/router/Fusion cannot settle. Protected-file edits (CLAUDE.md) approval-gated. Every routing decision receipted.

## Codex bridge note (deliverable 1 — DONE)
Local Codex activity is viewable + evaluable now via the merged bridge (`scripts/codex_bridge_digest.py`, #8456) + 15-min launchd digest. 48h evaluation (Jun 13–15): 27 PRs merged, steering directives honored (zero off-limits violations), drain-over-create working. Open follow-ups surfaced: 13 sessions on shared root; 31 stale already-merged ledger entries causing duplicate-prompt loops (conductor lacks merged-PR ledger cleanup); review PR #8454 (dissent-evidence gate).
