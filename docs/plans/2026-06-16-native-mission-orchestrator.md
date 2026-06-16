# Native Mission Orchestrator — one harness, set a goal, walk away, no halting

**Status:** design (grounded in a 3-agent repo investigation, Jun 16 2026). Consolidation, not greenfield.

**Goal (founder):** set a high-level goal in ONE harness and have it orchestrate heterogeneous models — **including Codex** — over hours **without halting** (Factory-mission semantics, native to aragora, minus what makes Factory halt). Acute pain it removes: hand-entering prompts into many Codex Desktop sessions every ~3 minutes.

## The reality this consolidates (four overlapping orchestrators, none cohesive)
- **`aragora/swarm/boss_loop.py`** (~5.3K lines) — GitHub-issue-driven; tick loop + launchd `com.aragora.swarm-boss-loop` KeepAlive + `--max-hours` + keepalive-on-empty-queue. The **non-halting engine** (~99% there). Drives Codex/Claude via subprocess CLI (`codex exec`, `claude -p`) through `boss_worker_lifecycle.py::dispatch_bounded_spec` + `SwarmCommander`.
- **`aragora/nomic/autonomous_orchestrator.py`** — free-text `execute_goal(goal)` → decompose→assign→execute→verify→commit; heterogeneous `AgentRouter` (~:204); approval/stopping/budget gates (~:1041, StoppingRuleEngine, cost cutoff); cross-cycle learning (KM/ELO/telemetry). The **goal front-door** (~70%). Gap: per-goal invocation, **no built-in long-running loop**.
- swarm lane orchestrator (`lane_conductor/supervisor/dispatcher`, `worker_launcher.py`) — PR-queue drainer; spawns Codex/Claude in worktrees (`codex exec -`, `claude -p`).
- `aragora/swarm/agent_bridge/` (`codex_source`/`codex_steer`/`auto_steer`, `broker.py`) — **observe + advise** Codex; cannot drive it headlessly; `broker` turn-loop halts on `needs_human`. **Adjunct, not the answer.**

**Why none cohered:** four different front doors (issue / goal / PR-queue / observe), overlapping but none subsuming the others. nomic has the goal-intake; boss_loop has the non-halting engine; they were never married.

**Why it halts (root cause = the harness/identity layer, NOT the loops):**
1. Model driving via **subscription-authed CLIs** (`codex exec`/`claude -p`) → 402 usage limits every few hours + MCP/chroma wedge.
2. Multiple fleets on one machine/account contend.
3. Hard halts (`needs_human`, consecutive-failures, no-runner) with **no relay-and-resume**.
4. The **outer** harness (Factory/Droid, Claude Code) halts on its own limits.

## Target architecture (marry the pieces; run them around the halting layers)
```
  aragora mission "<goal>" --budget --max-hours --relay slack        ← FRONT DOOR (new, thin)
        │  MissionSpec{goal, acceptance, budget, tier_policy, relay}
        ▼
  nomic/autonomous_orchestrator.execute_goal()  → decompose+plan      ← PLANNER (exists, ~70%)
        │  work items → internal queue (NOT GitHub-issue-gated)
        ▼
  boss_loop tick engine  (keepalive, --max-hours, launchd)            ← NON-HALTING ENGINE (exists, ~99%)
        │  per item: dispatch_bounded_spec → worker
        ▼
  MODEL TRANSPORT = api  (api_agents + Secrets Manager keys)          ← NEW: API by default, not subscription CLI
        │  Codex coding leg = codex exec (bounded; no pure-API equiv) │
        ▼
  merge-quorum gate  (swarm/quorum_evidence.py)  — SOLE merge authority
        ▲
  RELAY-WITH-TIMEOUT (notifications/approvals): on needs_human/repeated-fail →
     notify (Slack/email) + park-this-item + CONTINUE other items; never auto-merge/auto-settle.
  DURABLE STATE: boss_loop session store + nomic cycle telemetry + Convoy/Bead → restart resumes.
  HOST: headless on EC2 (not the laptop) → no MCP/fleet contention.
```
The single change that buys "hours without halting": **API model transport + server host + relay-with-timeout.** Those kill halt-causes 1, 2, and 3; running headless kills 4.

## Honest constraints
- **Codex leg:** there is no pure-API Codex *coding agent*; "orchestrating Codex" means driving the `codex exec` CLI, which has its own credit limits. Treat Codex as a *bounded* heterogeneous leg (with credits funded), while Claude/Grok/etc. run via API. Mission must degrade gracefully when the Codex leg is rate-limited (park Codex items, continue API legs) rather than halt.
- **Funding:** "without halting" requires **API budget**, not pooled subscriptions (see the TOS finding — subscription pooling for automation is non-compliant and is the source of the 402 wall). This is a spend decision, not just code.
- **This is consolidation:** ~99% engine + ~70% goal-intake already exist; the work is wiring + an API-transport path + a relay channel + a thin front door, behind a default-OFF flag.

## Tiered PR plan (smallest safe first; each through the merge-quorum gate)
| # | Tier | PR | Touches | Buildable now? |
|---|---|---|---|---|
| 1 | 1-2 | **`MissionSpec` + `Mission` front-door**: dataclass (goal/acceptance/budget/tier_policy/relay) + a thin runner that calls `execute_goal()` and emits work items to an internal queue (no GitHub-issue requirement). Pure-ish, unit-testable. | `aragora/nomic/` (new `mission.py`) | yes |
| 2 | 2-3 | **API model transport for the runner**: `mission_model_transport={api,cli}` flag; route worker model calls through `api_agents` (Secrets Manager) when `api`. The key non-halting change. | `swarm/worker_launcher.py`, `boss_worker_lifecycle.py`, `config/feature_flags.py` | needs care |
| 3 | 2-3 | **Relay-with-timeout**: wrap hard-halt points (`needs_human`, consecutive-failures) in notify (Slack/email via `notifications/` + `approvals/`) + timeout → park item + continue other work. Never auto-merge/auto-settle Tier-4. | `swarm/boss_loop.py`, `agent_bridge/broker.py`, `notifications/` | needs care |
| 4 | 2 | **`aragora mission` CLI**: `aragora mission "<goal>" --budget --max-hours --relay slack`. | `aragora/cli/` | yes (after #1) |
| 5 | 2 | **Durable mission state + resume**: persist MissionSpec + per-item status; `--resume <mission-id>`. | reuse boss_loop session store + Convoy/Bead | yes |
| 6 | 3 | **Headless EC2 deployment**: systemd/launchd unit + Secrets hydration; runs server-side, not laptop. Ops + checkpointed-for-operator. | deploy config | parked for operator |
| 7 | 2 | **Cross-iteration cost cap + watchdog**: aggregate spend across the whole mission + per-worker hang watchdog (the boss_loop hardening gaps the investigation flagged). | `swarm/boss_loop.py`, `billing/` | yes |

**Critical path:** 1 → 2 → 3 → 4 → 5 → (7) → 6. The mission goes live API-driven + relay-resilient at **#3**; headless server (#6) is the final "walk away while you sleep" step.

## Safety / guardrails (binding)
- **API keys via Secrets Manager only** (`config/secrets.py`); never pooled subscriptions, never raw env. Funding is metered API spend.
- **merge-quorum gate stays the sole settlement authority.** The mission/relay can never auto-merge or auto-settle Tier-4; Tier-3+ items park for human risk acceptance.
- **Relay-with-timeout parks, never bypasses.** On timeout it continues *other* items or stops the *item*, not the gate.
- **Default-OFF flag** `enable_native_mission`; no behavior change until enabled.
- **One fleet at a time.** Running this alongside Factory mission + a boss-loop + hand-driven Codex Desktop on one account reproduces the contention; the mission assumes it is the single fleet.
- agent_bridge/`auto_steer` remains the Codex-coordination **adjunct** (observe/steer), not a driver.

## Relation to other tracks
Engine for [[project_hybrid_orchestrator]]'s mode-switching/Pareto/Fusion (the mission is where those plug in). Uses [[reference_codex_bridge]] for Codex observe/steer. Funding rationale from the subscription-TOS finding.
