# Codex↔Claude Supervisory Bridge

> **Status:** deferred future tranche
> **Created:** 2026-04-19
> **Queue policy:** preserve as implementation-ready design; do not treat as the next active build lane
> **Relationship:** additive to the PR-intelligence / PDB track, not a replacement for it
> **Default operating boundary:** autonomous through branch/PR creation, no auto-merge, explicit human gate for judgment calls and policy/risk triggers

## Why This Doc Exists

This plan captures a real operator need:

- eliminate manual copy/paste between Codex and Claude
- keep both harnesses running in their native environments
- let progress continue mostly autonomously
- surface the whole loop in an observable UI
- interrupt the human only when a real judgment call is needed

The design is worth preserving. It is also not the right thing to build ahead of
the currently-active PDB / PR-intelligence tranche.

The current validated bottleneck is human review throughput. The supervisory
bridge may become the next bottleneck-solving layer later, but it should not
preempt the already-sequenced PDB work until the trust and review surfaces are
stable enough to absorb it.

## Defer Rationale

The bridge is deferred for three concrete reasons:

1. **Unvalidated bottleneck**
   Manual relay between Codex and Claude is painful, but it is not yet proven to
   be a larger constraint than review throughput.
2. **Oversight boundary**
   Copy/paste currently acts as a per-message human sanity check. Removing it
   means the system must replace that with explicit policy gates, structured
   footers, and observability strong enough to keep trust intact.
3. **Adapter risk**
   Tmux/log/transcript parsing is feasible, but materially more failure-prone
   than it first appears. The footer contract and broker loop should be treated
   as a real integration project, not a quick shell script.

## Activation Gate

Revisit this plan when all of the following are true:

1. The PDB / PR-intelligence tranche has shipped through its first real dogfood
   loop and is stable enough that it is no longer the dominant founder
   throughput constraint.
2. Review throughput has measurably improved, but manual Codex↔Claude relay is
   still a recurring operator bottleneck.
3. The proof-first / policy-gate surfaces are stable enough to trust
   machine-driven pauses over per-message human inspection.
4. There is clear appetite to keep the bridge local-first and single-operator
   rather than turning it into a generalized multi-tenant platform.

Until those conditions hold, this remains a preserved future tranche.

## Thesis

Build a local-first supervisory broker that runs one real Codex harness and one
real Claude Code harness, keeps their dialogue going automatically, and removes
manual copy/paste without replacing human judgment on irreversible decisions.

The bridge should:

- run Codex and Claude in their real harnesses, not via API impersonation
- persist run state locally under `.aragora/agent_bridge/`
- alternate turns through a structured broker loop
- require a machine-readable footer on every turn
- stop for human input only when both agents request judgment or policy/risk
  rules require it
- optionally escalate unresolved disagreement into Aragora’s many-model debate
  engine
- stop at PR publication, never merge

## Operator Model

The operator experience is:

1. start a bridge run with a bounded task
2. watch Codex and Claude converse in a live web UI
3. inspect current branches, worktrees, changed files, tests, and PR state
4. intervene only when the bridge raises a decision gate
5. choose from structured options, escalate to many-model debate, or provide
   human input
6. let the broker resume and continue autonomously

The bridge is meant to move the human from **message relay** to **decision
settlement**.

## Core Architecture

### 1. Broker domain

Add a broker domain under `aragora/swarm/` with:

- `BridgeRun`
- `BridgeTurn`
- `BridgeDecisionRequest`
- `BridgeOption`
- `BridgeArtifact`
- `BridgeRunStatus`

`BridgeRunStatus` values:

- `booting`
- `running`
- `awaiting_human`
- `awaiting_debate`
- `paused`
- `verifying`
- `publishing`
- `completed`
- `failed`
- `cancelled`

### 2. Persistence

Store bridge state locally under `.aragora/agent_bridge/`:

- one run record per run
- append-only event log per run
- derived artifact snapshots for current state

This is repo-scoped and local-first. Reloading the page or restarting the broker
must not lose the run.

### 3. Harness adapters

Wrap the existing session transport, not clipboard automation.

Each adapter must expose:

- `launch`
- `ready`
- `send_turn`
- `capture_delta`
- `status`
- `worktree_path`
- `branch`

Codex adapter:

- use the existing Codex session / tmux launcher flow

Claude adapter:

- use the existing Claude / tmux flow

Transport rules:

- prompt injection goes through the current session mux / tmux path
- output capture comes from mux logs and transcript tails
- delta capture uses prompt markers
- ANSI and control noise are stripped before parsing

### 4. Footer contract

Every turn must end with a strict machine-readable footer. The canonical JSON
trailer includes:

- `action`
- `summary`
- `next_actor`
- `files_touched`
- `tests_run`
- `needs_human`
- `needs_many_model_debate`
- `done`
- optional `decision_request`

`decision_request` shape:

- `question`
- `recommended_option_id`
- `options[]`

Each option includes:

- `id`
- `title`
- `strengths[]`
- `weaknesses[]`
- `rationale`

If a footer is missing or malformed, the broker sends a repair prompt back to
the same harness and does not advance the baton.

### 5. Baton model

V1 uses one active implementation baton.

- Codex is the default implementer
- Claude is the default reviewer / critic
- Claude only gets a bounded side-lane when the broker explicitly assigns a
  separate scope

This prevents both harnesses from editing the same files by default.

### 6. Ownership and coordination

Each harness runs in its own worktree. The broker uses the existing
coordination/claim primitives to:

- claim files before code-changing turns
- claim PR ownership before publish actions
- surface contested ownership in the UI

### 7. Policy gate

Human input is required when:

- both agents request a human judgment call, or
- a fixed policy/risk rule fires

V1 fixed policy profile:

- `.github/workflows`
- auth / RBAC / security / privacy / compliance paths
- merge / review gate paths
- large multi-subsystem diffs
- repeated verification failure
- stale or conflicting branch state

### 8. Many-model escalation

Many-model debate is explicit and resumptive.

The bridge packages:

- current disagreement
- repo and progress snapshot
- changed files
- latest Codex summary
- latest Claude summary

It then calls the existing Aragora debate engine and attaches the result as a
bridge artifact. The result is fed back into both harnesses as authoritative
tie-break context.

### 9. Publish boundary

The bridge may:

- edit
- test
- commit
- push
- open or update PRs

The bridge may **not** merge.

Any PR publication must pass the existing automation preflight first, and the
result must be recorded in run artifacts.

## Backend Surface

Add `/api/v1/agent-bridge` endpoints:

- `POST /runs`
- `GET /runs`
- `GET /runs/{run_id}`
- `POST /runs/{run_id}/pause`
- `POST /runs/{run_id}/resume`
- `POST /runs/{run_id}/cancel`
- `POST /runs/{run_id}/decisions/{decision_id}`
- `POST /runs/{run_id}/escalate`
- `GET /runs/{run_id}/events`
- `GET /ws/agent-bridge/{run_id}`

`CreateBridgeRunRequest` includes:

- `task`
- `repo_root`
- `base_branch`
- `agents`
- `autonomy_mode`
- `policy_profile`

Default values:

- current repo root
- `main`
- `["codex", "claude"]`
- `through_prs`
- `supervised_local`

## Event Model

Emit at least these run events:

- `run_started`
- `agent_ready`
- `turn_started`
- `turn_completed`
- `artifact_updated`
- `verification_started`
- `verification_result`
- `decision_requested`
- `decision_resolved`
- `escalation_started`
- `escalation_completed`
- `publish_started`
- `pr_opened`
- `run_completed`
- `run_failed`

## UI Surface

The first UI home is the autonomous surface in `aragora/live`, not a sidecar
app and not the command canvas.

Add:

- `/autonomous/bridge`
- `/autonomous/bridge/[runId]`

Run list page should show:

- runs
- current status
- pending decisions
- recent artifacts

Run detail page should show:

- live Codex↔Claude transcript
- current active speaker / baton owner
- worktrees and branches
- changed files
- test status
- PR status
- latest structured summary from each harness
- open decision gates
- full event and audit trail

Reuse:

- spectate stream patterns for live transcript rendering
- approval / intervention panel patterns for human decisions

The bridge UI must be observable, not magical.

## Human Decision Surface

Every decision gate presents exactly three actions:

1. choose one of the synthesized options
2. escalate to many-model debate
3. provide freeform human input

Freeform human input is written back into the run as a structured decision event
and injected into both harnesses.

## Recommended Delivery Sequence

When this plan activates, implement in this order:

1. **SB-1 Broker core**
   dataclasses, persistence, run lifecycle, event log, and broker state machine
2. **SB-2 Footer contract + adapters**
   footer parser/validator, repair loop, Codex adapter, Claude adapter
3. **SB-3 Policy gates + ownership**
   fixed policy profile, claim integration, pause/resume semantics
4. **SB-4 Backend API + streaming**
   run endpoints, event snapshots, websocket stream
5. **SB-5 UI surfaces**
   run list, run detail, transcript, decision cards, event trail
6. **SB-6 Many-model escalation + publish boundary**
   debate packaging, artifact attachment, preflight-gated PR publish flow

Do not invert this order. A UI-first bridge without a stable broker contract
will create false confidence.

## Test Expectations

When this plan activates, the minimum test bar is:

- unit tests for adapter readiness, prompt markers, ANSI stripping, and footer
  parsing
- unit tests for broker alternation, baton retention, malformed footer repair,
  human gates, policy gates, and escalation resume
- API tests for run creation, listing, pause/resume/cancel, decision submission,
  and escalation submission
- UI tests for run list, live transcript, decision cards, option selection,
  escalation, and freeform input
- one fake-harness integration test from run start to PR publication
- one policy-gate integration test proving pause on sensitive path changes
- one restart/resume test proving persistence across broker restarts

## Explicit Non-Goals

- auto-merge
- hidden policy bypass
- OCR / clipboard transport
- replacing Codex or Claude harnesses with direct API impersonation
- multi-tenant productization in the first ship

## Revisit Checklist

When the activation gate is met, the first revisit should answer:

1. Is manual relay still a top-3 founder bottleneck after PDB?
2. Is the trust boundary strong enough to move from per-message review to
   decision-gate review?
3. Should the first live implementation still include the web UI, or is a
   broker-only v0.1 now the better proof step?
4. Does the policy profile still match the current proof-first and review-gate
   boundaries?

If the answer to the first two questions is no, keep this plan deferred.
