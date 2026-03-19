# Tranche Orchestration: Prompt-Driven Multi-Lane Execution

**Date:** 2026-03-19
**Status:** Approved design, pending implementation plan
**Authors:** Claude Opus (design), Codex (review/corrections)

---

## Problem

A human talks to one Claude/Codex session with vague, exploratory intent. That session
needs to translate the conversation into structured work that Aragora can decompose,
dispatch across multiple parallel lanes, cross-model review, and integrate through
PR/merge gates — without the human manually coordinating agents.

Today Aragora has the primitives (Boss-loop dispatch, Campaign decomposition/review,
Tranche inspection, Supervisor/Reconciler execution, DevCoordinationStore leases/receipts)
but lacks the glue that connects a loose intake to durable multi-lane execution with
adaptive autonomy.

---

## Architecture

Four existing components keep their current roles:

| Component | Role | Key Files |
|-----------|------|-----------|
| **Tranche** | Orchestration spine and lifecycle state machine | `aragora/swarm/tranche.py` |
| **Campaign** | Decomposition engine and review machinery | `aragora/swarm/campaign.py` |
| **Boss-loop** | Bounded dispatch engine | `aragora/swarm/boss_loop.py` |
| **DevCoordinationStore** | Durable state (leases, receipts, integration decisions) | `aragora/nomic/dev_coordination.py` |

Supporting components: Supervisor (`supervisor.py`), Reconciler (`reconciler.py`),
WorkerLauncher (`worker_launcher.py`), PullRequestRegistry (`pr_registry.py`),
GhReferenceClient (in `tranche.py`), check partitioning (`aragora/ralph/github_control.py`).

---

## Three Artifact Layers

Central to this design is the distinction between three artifacts that `submit` produces.
Each serves a different purpose and all three are persisted for auditability.

### 1. Intake Bundle (raw)

What the intake Claude/Codex session provided. Loose, possibly incomplete. Preserved
exactly as received for audit trail. This is "the human said this (as translated by the
intake session)."

### 2. Normalized Prompt Bundle

What `submit` inferred, enriched, and structured. Source refs resolved, stale targets
rejected, missing fields filled, decomposition applied where needed. This is "submit
inferred this."

### 3. Tranche Manifest

The compiled, inspected, executable artifact. Gates, lanes, scope, dependencies, policies.
This is "Aragora will execute this."

---

## Lifecycle

### Build order and current status

| # | Command | Status | What It Does |
|---|---------|--------|-------------|
| 0 | `inspect` | On main | Read-only gate, scope, and lane evaluation |
| 0 | `plan` | On branch (not yet on main) | Prompt bundle YAML → tranche manifest YAML. Implemented via `TranchePlanner.plan_from_prompt_bundle()`. |
| 0 | `prepare` | On branch (not yet on main) | Workspace-prep: managed worktree + branch + prepared artifact. Optional — `run` auto-prepares if needed. Does NOT claim leases. |
| 0 | `run` | On branch (not yet on main) | Dispatch lanes through `dispatch_bounded_spec`. Includes conditional inline review via `_review_lane()` when dispatch returns a completed lane with a usable `run_dict`. |
| 1 | `submit` | To build | Intake bundle → enrich → normalize → compile → inspect → persist 3 layers + tranche run state |
| 2 | `design-review` | To build | Bounded proposer/critic/synthesizer challenge loop over the normalized bundle + inspected manifest before any mutating lane starts |
| 3 | `review` | To build | First-class review command with adaptive tiering |
| 4 | `integrate` | To build | Assess-first PR/check state, merge only with `--approve` or permitted autonomy |
| 5 | `TrancheRunState` schema | To build | Durable tranche projection dataclasses (`TrancheRunState`, `LaneRunState`). Prerequisite for `watch`. |
| 5 | `watch`/`list` | To build | Durable projection, observer/driver split, session reattach |

### Prerequisites

1. **Land the existing `codex/tranche-plan-prepare-run` branch.** Current main only has
   `inspect`. The `plan`/`prepare`/`run` surface must be on main before building on top.

2. **Persist `receipt_id` and `lease_id` into tranche artifact metadata.** `integrate`
   depends on `run` writing these fields into `TrancheLaneArtifact.metadata`. This must
   be an explicit part of the phase-0 work, not left implicit.

3. **`inspect` is the required preflight for every mutating step.** `prepare`, `run`,
   `review`, `integrate`, and `watch --driver` all act only against an inspected tranche
   state, not stale manifest data.

### Lifecycle flow

`submit` internally wraps `plan` + `inspect` + persist. The standalone `plan` and
`inspect` commands remain available for manual/debugging use but are not required
in the normal flow.

```
Human (vague) → Intake Session (Claude/Codex)
    │
    ▼
submit ─────── Intake bundle → enrich → normalize
    │            → [plan] compile bundle → manifest
    │            → [inspect] validate gates/scope
    │            → persist 3 layers + run state
    │            Returns: {inspection_status, submission_status, recommended_action}
    ▼
design-review ─ Bounded proposer/critic/synthesizer loop
    │            → challenge weak assumptions against live repo/ref state
    │            → persist findings + revised manifest or unresolved assumptions
    ▼
 prepare ────── (On branch) Workspace-prep, optional (run auto-prepares)
    │
    ▼
  run ─────── (On branch) Dispatch + conditional inline review
    │
    ▼
 review ────── First-class adaptive review (tiers 1/2/3)
    │
    ▼
integrate ──── Assess PR/check state, recommend, merge with --approve
    │
    ▼
 watch ─────── Durable projection, observer/driver, reattach
```

---

## Submit

### Intake Bundle schema

```yaml
# Required
objective: <string>          # What the human wants, in intake session's words

# Optional but valuable
candidate_lanes:
  - lane_id: <string>        # Optional but preferred; submit generates if missing
    title: <string>
    owner_role: <string>     # e.g. critical_path_engineer, ui_engineer, read_only_forensics
    prompt: <string>
    source_refs: [<urls>]
    target_agent: codex|claude
    allowed_write_scope: [<globs>]
    dependencies: [<lane_ids or ref_ids>]
    acceptance_criteria: [<strings>]
    constraints: [<strings>]
    verification_commands: [<strings>]

# Optional context
source_refs:
  - url: <string>
    meaning: <string>

# Execution policy
autonomy_mode: adaptive|fire_and_forget|checkpoint|spectator
risk_tolerance: low|medium|high
suggested_models:
  worker: codex
  reviewer: claude
  planner: claude
constraints: [<strings>]
acceptance_signals: [<strings>]
```

### What submit does

1. **Validates** — `objective` is required, everything else optional.
2. **Enriches** — resolves source refs (GitHub PR/issue refs get live resolution and gate
   checking; docs/local files/notes preserved as context only, not gated). Rejects stale
   targets. Generates `lane_id` where missing.
3. **Decomposes** — explicit triggers:
   - No `candidate_lanes` → full decomposition via `CampaignPlanner`
   - Lane missing `prompt` or `owner_role` → planner augments/rebuilds that lane
   - Lane missing scope/verification/dependencies → inference only, no rebuild
   - Planner output is treated as *proposed*, not authoritative.
4. **Normalizes** — produces normalized prompt bundle with inferred fields.
5. **Compiles** — converts to tranche manifest via `TranchePlanner.plan_from_prompt_bundle()`
   (exists on the `codex/tranche-plan-prepare-run` branch, not yet on main).
6. **Inspects** — runs `TrancheInspector.inspect()`.
7. **Persists** — writes all three artifact layers plus `TrancheRunState`.
8. **Returns dual status:**

| Field | Values | Meaning |
|-------|--------|---------|
| `inspection_status` | `ok` / `blocked` | Factual tranche readiness |
| `submission_status` | `ready_to_prepare` / `awaiting_confirmation` / `blocked` | Policy/autonomy decision |
| `recommended_action` | string | What to do next |

### Submit is non-interactive

It compiles, inspects, persists, and returns. Advancing to execution is a separate
command (`run` or explicit approval).

### CLI

```bash
aragora swarm tranche submit --intake <path|-> \
    [--autonomy adaptive|fire_and_forget|checkpoint|spectator] \
    [--json]
```

---

## Design Review

This stage captures the productive pattern from the Claude/Codex collaboration in this
session: one model proposes, another challenges that proposal against the live repo and
reference state, and a final pass synthesizes the revision before implementation starts.

### Purpose

Catch weak assumptions and role drift **before** `prepare`/`run` launches mutating work.
This is a bounded pre-implementation challenge loop, not open-ended agent chat.

### Shape

- **Proposer:** Presents the normalized prompt bundle and compiled tranche manifest as
  the current execution proposal.
- **Critic:** Challenges the proposal against live repo/code/reference state. Findings
  must be grounded in concrete evidence, not generic objections.
- **Synthesizer:** Produces the revised manifest plus unresolved assumptions from the
  proposer + critic outputs.

### Bounds

- Max **2 rounds**.
- If the synthesizer cannot converge after the bounded rounds, return
  `awaiting_confirmation` or `needs_human`.
- This stage must not mutate tracked repo files or launch workers.

### Persisted outputs

`design-review` persists:

- proposed manifest snapshot
- critique findings
- revised manifest snapshot
- unresolved assumptions
- final recommendation (`approved` / `awaiting_confirmation` / `needs_human`)

### Default policy

- **adaptive** and **checkpoint** modes: run `design-review` by default before the first
  writable lane is prepared.
- **fire_and_forget**: may skip `design-review` only for low-risk, clearly bounded
  tranches.
- **spectator**: defaults to running it, but the driver may explicitly bypass.

### CLI

```bash
aragora swarm tranche design-review --manifest <path> \
    [--rounds 2] \
    [--json]
```

---

## Adaptive Review

Three tiers, selected automatically based on lane risk profile.

### Tier selection (lane-centric first, diff-centric second)

1. Declared write scope breadth
2. Lane/source type (dependency bump vs. cross-system wiring)
3. Verification command presence and result
4. Explicit `risk_tolerance` override
5. Post-run diff size as escalation signal only

### Tier 1: Lightweight Review

- **When:** Low risk — bounded scope, small diff, verifications pass.
- **What:** Single cross-model reviewer via existing `CampaignReviewer.review()`.
- **Outcome:** `passed` → proceed to integrate. `failed` → escalate to Tier 2.

### Tier 2: Multi-Reviewer Consensus (v1)

- **When:** Medium risk, or Tier 1 failure.
- **What (v1):** Two independent reviewers + one synthesizer. Returns the same
  `CampaignReviewGate` contract. NOT full adversarial debate engine in v1.
- **Outcome:** Consensus `passed` → integrate. Consensus `failed` → Tier 3.
  No consensus → `needs_human`.
- **Future:** Full adversarial debate topology as a later upgrade.

### Tier 3: Bounded Retry with Review Findings (v1)

- **When:** High risk, or Tier 2 failure.
- **What (v1):** Reuses campaign retry semantics — `retry_count`,
  `max_retries_per_project`, review findings appended as constraints to retry spec.
  Dispatches through `dispatch_bounded_spec()` with a new worktree (same-worktree
  reuse is a future optimization).
- **Findings format (v1):** Plain-text `findings: list[str]` from `CampaignReviewGate`.
  Structured `ReviewFinding` model (file/line/issue/fix) is a later addition.
- **Bounded:** Max 2 correction attempts, then `needs_human`.

### Public command

```bash
aragora swarm tranche review --manifest <path> \
    [--lane <id>|--all-completed] \
    [--tier auto|1|2|3] \
    [--json]
```

### Adapter: TrancheLane → CampaignProject

`CampaignReviewer.review()` expects a `CampaignProject` and `run_dict`. The review
command (and the existing inline `_review_lane()`) must adapt `TrancheLane` + lane
artifact data into the `CampaignProject` shape. This is a thin projection, not a new
reviewer — the adapter constructs a `CampaignProject` from the lane's title, source
refs, spec, file scope hints, acceptance criteria, and constraints.

### Interaction with run

- The existing inline `_review_lane()` hook stays as a fast path (Tier 1 after dispatch).
- The explicit `review` command can escalate to higher tiers or re-review.
- `run --skip-review` still works for fire-and-forget mode.

---

## Integrate

### Behavior

`integrate` is assess-first by default — it discovers PR state, classifies checks,
computes a merge recommendation, and persists integration metadata. It only executes
a merge when `--approve` is present or autonomy policy permits.

For each completed+reviewed lane:

1. **Discovers PR** — checks lane artifact metadata for `pr_url`, or searches via
   `gh pr list` on the lane's branch.
2. **Registers** — records in `PullRequestRegistry` if not already there.
3. **Monitors checks** — queries via `gh pr checks`. Reuses existing check partition
   logic from `aragora/ralph/github_control.py` rather than inventing a new classifier.
4. **Classifies check results:**
   - Required checks all green → `checks_passed`
   - Required checks failing (diff-caused) → `checks_failed`
   - Non-required/advisory failing → `checks_noise`
   - Noise classification is **advisory only** — never overrides GitHub required checks
     or branch protection in v1.
5. **Records integration decision** — uses `DevCoordinationStore.record_integration_decision()`
   with: `merge` / `cherry_pick` / `request_changes` / `discard` / `salvage`.
   (`cherry_pick` is an existing integration path for partial adoption.)
6. **Executes merge** (only with `--approve` or permitted autonomy):
   - `fire_and_forget` + checks passed + review passed → auto-merge
   - `adaptive` + high confidence → auto-merge
   - `adaptive` + medium/low confidence → `awaiting_confirmation`
   - `checkpoint` → always `awaiting_confirmation`
7. **Updates tranche state** — lane status uses campaign-aligned vocabulary:
   `waiting_for_pr`, `waiting_for_merge`, `completed`, `needs_human`.

### Merge policy (per-lane, with tranche-level default)

```yaml
merge_policy: auto|confirm|manual
# auto:    merge when checks + review pass (gated by autonomy mode)
# confirm: require explicit --approve
# manual:  never auto-merge, just record recommendation
```

### CLI

```bash
# Assess integration readiness
aragora swarm tranche integrate --manifest <path> [--json]

# Approve and merge a specific lane
aragora swarm tranche integrate --manifest <path> --lane <id> --approve

# Approve all mergeable lanes
aragora swarm tranche integrate --manifest <path> --all-mergeable --approve
```

### Prerequisite

`run` must persist `receipt_id` and `lease_id` into `TrancheLaneArtifact.metadata` so
`integrate` can use `DevCoordinationStore.record_integration_decision()` cleanly.

---

## Watch, Attach/Detach, & Durable State

### TrancheRunState

A tranche-level projection persisted as `run_state.yaml` alongside the manifest and lane
artifacts. This is the **tranche summary/control ledger**, not the single source of truth
for lower-level primitives. Authoritative state for run_id, lease_id, PR state, and
worker liveness remains in their respective stores (supervisor runs, DevCoordinationStore,
PRRegistry).

```python
@dataclass
class TrancheRunState:
    manifest_id: str
    status: str          # planned | preparing | running | reviewing |
                         #   integrating | completed | needs_human
    autonomy_mode: str   # adaptive | fire_and_forget | checkpoint | spectator
    created_at: datetime
    updated_at: datetime
    lane_states: dict[str, LaneRunState]
    driver_session: str | None       # session ID of current driver
    driver_heartbeat: datetime | None
    session_history: list[dict]      # [{session_id, attached_at, detached_at, role}]

@dataclass
class LaneRunState:
    lane_id: str
    status: str          # pending | preparing | dispatched | running |
                         #   completed | reviewing | review_passed |
                         #   review_failed | retrying | waiting_for_pr |
                         #   waiting_for_merge | needs_human
                         # Note: "completed" is the canonical terminal success status,
                         # aligned with campaign vocabulary. "merged" is NOT a separate
                         # stored status — a lane whose PR merged transitions to
                         # "completed" with pr_merged=True in metadata.
    run_id: str | None
    receipt_id: str | None
    lease_id: str | None
    worktree_path: str | None
    pr_url: str | None
    retry_count: int
    last_updated: datetime
```

### Session modes

- **Observer**: Read-only, unlimited concurrent sessions. Polls and displays state.
- **Driver**: One session with heartbeat + takeover timeout. Can advance state
  (trigger review, approve merge, etc.). Uses the existing lease/heartbeat model.

### Watch behavior

1. Loads `TrancheRunState` from durable storage.
2. Refreshes from authoritative stores (supervisor runs, leases, receipts, tranche
   artifacts, integration decisions) — not PID polling as primary mechanism.
3. In driver mode with adaptive autonomy, acts on state transitions:
   - Lane completed → triggers review
   - Review passed → triggers integrate assessment
   - All lanes merged → marks tranche `completed`
   - Any lane `needs_human` → pauses and surfaces reason
4. On disconnect: writes `detached_at`, clears driver session. Workers continue running.

### Reattach

A new session runs `watch` on the same manifest. It picks up from durable state.

### CLI

```bash
# Attach as observer (default) or driver
aragora swarm tranche watch --manifest <path> [--driver] [--interval 10] [--json]

# List all known tranches
aragora swarm tranche list [--json]
```

### Deferred from v1

- `abort` — requires a deliberate, safe stop path for supervisor runs/workers. Not
  included until that path is implemented.

---

## Autonomy Modes

Applied at the tranche level, governing how each lifecycle step advances.

| Mode | Behavior |
|------|----------|
| `adaptive` (default) | Confidence-tiered: high-confidence bounded tasks run autonomously; ambiguous or high-risk work pauses for review. System determines confidence from risk tier, review outcome, and check state. |
| `fire_and_forget` | Run all lanes, review, integrate, and merge without human gates. Stop only at hard blockers (test failures, merge conflicts, `needs_human`). |
| `checkpoint` | Pause after plan, after first lane completes, before each merge. Require explicit approval at each gate. |
| `spectator` | Run continuously but allow live observation/intervention. Driver session can redirect, pause, or approve at any point but doesn't have to. |

---

## Non-Goals for v1

- Open-ended multi-agent chat with no bounded stop condition before execution
- Full adversarial debate for code review (Tier 2 is multi-reviewer consensus only)
- Structured `ReviewFinding` model with file/line/issue/fix granularity
- Same-worktree corrective execution in Tier 3
- `abort` command
- Automatic noise classification overriding GitHub required checks
- Lease claiming in `prepare` (supervisor/dispatch path owns runtime enforcement)
