# Native Mission Orchestrator — one harness, set a goal, walk away, no halting

**Status:** design (grounded in a 3-agent repo investigation, Jun 16 2026). Consolidation, not greenfield.

**Goal (founder):** set a high-level goal in ONE harness and have it orchestrate heterogeneous models — **including Codex** — over hours **without halting** (Factory-mission semantics, native to aragora, minus what makes Factory halt). Acute pain it removes: hand-entering prompts into many Codex Desktop sessions every ~3 minutes.

## The reality this consolidates (four overlapping orchestrators, none cohesive)
- **`aragora/swarm/boss_loop.py`** (~5.3K lines) — GitHub-issue-driven; tick loop + launchd `com.aragora.swarm-boss-loop` KeepAlive + `--max-hours` + keepalive-on-empty-queue. The **non-halting engine** (~99% there). Drives Codex/Claude via subprocess CLI (`codex exec`, `claude -p`) through `boss_worker_lifecycle.py::dispatch_bounded_spec` + `SwarmCommander`.
- **`aragora/nomic/autonomous_orchestrator.py`** — free-text `execute_goal(goal)` → decompose→assign→execute→verify→commit; heterogeneous `AgentRouter` (~:204); approval/stopping/budget gates (~:1041, StoppingRuleEngine, cost cutoff); cross-cycle learning (KM/ELO/telemetry). The **goal front-door** (~70%). Gap: per-goal invocation, **no built-in long-running loop**.
- swarm lane orchestrator (`lane_conductor/supervisor/dispatcher`, `worker_launcher.py`) — PR-queue drainer; spawns Codex/Claude in worktrees (`codex exec -`, `claude -p`).
- `aragora/swarm/agent_bridge/` (`codex_source`/`codex_steer`/`auto_steer`, `broker.py`) — **observe + advise** Codex; cannot drive it headlessly; `broker` turn-loop halts on `needs_human`. **Adjunct, not the answer.**

**Why none cohered:** four different front doors (issue / goal / PR-queue / observe), overlapping but none subsuming the others. nomic has the goal-intake; boss_loop has the non-halting engine; they were never married.

**Why it halts (root cause = the harness/identity layer, NOT the loops, and NOT the subscriptions themselves):**
1. **The OUTER harness halts on ITS OWN billing limit.** The dominant halt in the Factory transcript is `"5-hour standard usage limit … Switch to Droid Core"` — that's **Droid/Factory's** limit, not Anthropic/OpenAI. Running the orchestrator *inside* Droid imports Droid's wall.
2. **Self-inflicted contention.** Five concurrent `claude` processes + the desktop MCP swarm (chroma-mcp) on one account wedged the `claude -p` CLI. Already mitigated by the MCP-disabled reviewer (#8421).
3. Hard halts (`needs_human`, consecutive-failures, no-runner) with **no relay-and-resume** → the human becomes the relay.
4. GitHub GraphQL rate-limit exhaustion from poll loops.

**A single Claude Max sub and a single Codex Max sub CAN each run for hours** — Codex Max is empirically busy for hours right now; a single uncontended MCP-disabled `claude -p` sits inside the generous weekly Max caps. Subscriptions are NOT the wall; the wall is the outer harness (#1) + contention (#2). One sub *per provider* is single-account use (TOS-clean — distinct from the non-compliant ~15-account *pool*).

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
  MODEL TRANSPORT = subscription CLI (MCP-disabled, 1 sub/provider)   ← DEFAULT: 1 Claude Max + 1 Codex Max
        │  codex exec (Codex Max) + claude -p (Claude Max, MCP off)   │  API = OPTIONAL scale lever
        ▼
  merge-quorum gate  (swarm/quorum_evidence.py)  — SOLE merge authority
        ▲
  RELAY-WITH-TIMEOUT (notifications/approvals): on needs_human/repeated-fail →
     notify (Slack/email) + park-this-item + CONTINUE other items; never auto-merge/auto-settle.
  DURABLE STATE: boss_loop session store + nomic cycle telemetry + Convoy/Bead → restart resumes.
  HOST: headless on EC2 (not the laptop) → no MCP/fleet contention.
```
The changes that buy "hours without halting": **(a) aragora is the orchestrator** (no outer Droid/Factory 5-hour wall — kills halt #1), **(b) single fleet + MCP-disabled reviewer** (kills #2, already shipped #8421), **(c) relay-with-timeout** (kills #3), **(d) headless on server** (kills the laptop-contention side of #2). This works **on subscriptions** (1 Claude Max + 1 Codex Max). API model transport is an **optional scale lever** (more parallelism / beyond weekly caps), not a prerequisite.

## Honest constraints
- **Codex leg:** there is no pure-API Codex *coding agent*; "orchestrating Codex" means driving the `codex exec` CLI, which has its own credit limits. Treat Codex as a *bounded* heterogeneous leg (with credits funded), while Claude/Grok/etc. run via API. Mission must degrade gracefully when the Codex leg is rate-limited (park Codex items, continue API legs) rather than halt.
- **Funding:** runs on **subscriptions by default** — 1 Claude Max + 1 Codex Max (single-account-per-provider, TOS-clean). The 402 wall in the transcript was *Droid/Factory's* limit, not the model subs. **API budget is the optional scale lever** (more parallel legs, or work beyond the weekly Max caps), not a prerequisite. Do NOT pool multiple personal subs (that's the non-compliant pattern).
- **This is consolidation:** ~99% engine + ~70% goal-intake already exist; the work is wiring + an API-transport path + a relay channel + a thin front door, behind a default-OFF flag.

## Tiered PR plan (smallest safe first; each through the merge-quorum gate)
| # | Tier | PR | Touches | Buildable now? |
|---|---|---|---|---|
| 1 | 1-2 | **`MissionSpec` + `Mission` front-door**: dataclass (goal/acceptance/budget/tier_policy/relay) + a thin runner that calls `execute_goal()` and emits work items to an internal queue (no GitHub-issue requirement). Pure-ish, unit-testable. | `aragora/nomic/` (new `mission.py`) | yes |
| 2 | 2-3 | **`mission_model_transport={cli,api}` flag**: default `cli` runs the existing MCP-disabled `claude -p` (#8421) + `codex exec` on subscriptions; `api` routes through `api_agents` (Secrets Manager) as the scale lever. NOT a prerequisite for hours-without-halting. | `swarm/worker_launcher.py`, `boss_worker_lifecycle.py`, `config/feature_flags.py` | needs care |
| 3 | 2-3 | **Relay-with-timeout**: wrap hard-halt points (`needs_human`, consecutive-failures) in notify (Slack/email via `notifications/` + `approvals/`) + timeout → park item + continue other work. Never auto-merge/auto-settle Tier-4. | `swarm/boss_loop.py`, `agent_bridge/broker.py`, `notifications/` | needs care |
| 4 | 2 | **`aragora mission` CLI**: `aragora mission "<goal>" --budget --max-hours --relay slack`. | `aragora/cli/` | yes (after #1) |
| 5 | 2 | **Durable mission state + resume**: persist MissionSpec + per-item status; `--resume <mission-id>`. | reuse boss_loop session store + Convoy/Bead | yes |
| 6 | 3 | **Headless EC2 deployment**: systemd/launchd unit + Secrets hydration; runs server-side, not laptop. Ops + checkpointed-for-operator. | deploy config | parked for operator |
| 7 | 2 | **Cross-iteration cost cap + watchdog**: aggregate spend across the whole mission + per-worker hang watchdog (the boss_loop hardening gaps the investigation flagged). | `swarm/boss_loop.py`, `billing/` | yes |

**Critical path:** 1 → 3 → 4 → 5 → (7) → 6, with **#2 optional** (API is the scale lever, not required). The mission goes live **on subscriptions** + relay-resilient at **#3** (it already has MCP-disabled CLI driving via #8421); headless server (#6) is the final "walk away while you sleep" step. The single biggest immediate win is simply **making aragora the orchestrator instead of Droid/Factory** — that removes the 5-hour outer-harness wall with zero new code.

## Safety / guardrails (binding)
- **Runs on subscriptions by default** (1 Claude Max + 1 Codex Max, single-account-per-provider). If/when the API scale lever is used, keys load via Secrets Manager (`config/secrets.py`), never raw env. Never pool multiple personal subs (the non-compliant pattern).
- **merge-quorum gate stays the sole settlement authority.** The mission/relay can never auto-merge or auto-settle Tier-4; Tier-3+ items park for human risk acceptance.
- **Relay-with-timeout parks, never bypasses.** On timeout it continues *other* items or stops the *item*, not the gate.
- **Default-OFF flag** `enable_native_mission`; no behavior change until enabled.
- **One fleet at a time.** Running this alongside Factory mission + a boss-loop + hand-driven Codex Desktop on one account reproduces the contention; the mission assumes it is the single fleet.
- agent_bridge/`auto_steer` remains the Codex-coordination **adjunct** (observe/steer), not a driver.

## Relation to other tracks
Engine for [[project_hybrid_orchestrator]]'s mode-switching/Pareto/Fusion (the mission is where those plug in). Uses [[reference_codex_bridge]] for Codex observe/steer. Funding rationale from the subscription-TOS finding.
