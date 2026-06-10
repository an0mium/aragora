# Loop Control Plane (v1)

> Status: v1, **read-only**. No mutation, no active controller. This is a
> governance *observability* surface, not new autonomy.

Aragora runs several standing loops (the boss loop, the merge arbiter, the
proof-first shift, the codex automation publisher, the worktree autopilot, the
nomic loop). Each was built and operated on its own. The Loop Control Plane
unifies them into a single, budgeted, halt-aware inventory and answers one
governance question per loop:

> **Is this loop safe to keep running right now?**

It is computed entirely from surfaces that already exist; it never becomes a
competing source of truth and it never acts.

---

## 1. The "loops" lesson, mapped to Aragora

This work applies the framing from Matt Van Horn's *"WTF Is a Loop?"*:

| Article idea | Aragora realization |
|---|---|
| A loop is *a scheduler plus a decision-maker* | launchd job (e.g. `com.aragora.swarm-merge-arbiter`) + the loop's in-process decision logic (`aragora/swarm/merge_arbiter.py`) |
| Loops need **hard stops** or they run forever / overspend | max-iteration / max-runtime bounds, no-progress detection, and a spend ceiling (see §2) |
| Loops need **self-verification / a feedback gate** | the model quorum (`aragora-merge-quorum`), proof-first freshness, publisher freshness, worktree health |
| **Skills/tools are the reusable unit** the loop drives | the loop dispatches workers/agents and shells out to repo scripts |
| The dangerous failure is **silent drift** | a loop that keeps "succeeding" while making no real progress, or that conflates *waiting* with *broken* (see §4) |

The point of the article is not "add more loops"; it is "make the loops you
already run *legible* and *stoppable*." v1 does exactly that and no more.

---

## 2. The three hard stops

A loop is only *safe to keep running* when all three of these hold. The plane
audits each loop's guards against them (`audit_halt_readiness`):

1. **Max iteration / runtime.** A bound that guarantees termination
   (e.g. `MergeArbiterConfig.max_runtime_hours = 12`,
   `--max-hours` on the proof-first shift).
2. **No-progress detection that distinguishes fault from waiting.** The loop
   must notice it is not advancing **and** correctly separate an *operational
   fault* (halt, fail closed) from *normal waiting on not-ready work*
   (keep waiting). Getting this wrong is the live `merge_arbiter` defect (§4).
3. **Budget ceiling.** A spend ceiling so a stuck-but-busy loop cannot run up an
   unbounded bill. This is the article's headline risk.

The audit verdict per loop is:

- `ok` - all three present.
- `incomplete` - some present, with specific `gaps` listed.
- `missing` - none present.

### Current findings (curated; see §6)

Every standing loop is currently `incomplete`. Two systemic gaps dominate:

- **No loop has a dollar/budget ceiling.** They are bounded by time/iterations
  only. Wiring real per-loop budgets is the highest-value follow-up.
- **`merge_arbiter`'s no-progress detection does not distinguish an operational
  fault from not-ready PRs** (the #7879 bug class), so its circuit breaker can
  trip on benign waiting and/or spin to `max_runtime` on a real fault.

These are honest, code-referenced findings, not fabricated alarms.

---

## 3. The loop-to-ladder map

Each loop maps to a review tier (`docs/REVIEW_AUTHORITY_PRINCIPLES.md`), a role,
a feedback gate, and a durable-state location. This is the curated registry in
`aragora/swarm/loop_control.py::LOOP_SPECS`.

| Loop | Role | Feedback gate | Durable state | Human gate |
|---|---|---|---|---|
| `boss_loop` | supervisor | quorum | `.aragora/operator_steering` | no |
| `merge_arbiter` | supervisor | quorum | (in-process) | yes (Tier 3-4 settlement) |
| `proof_first_shift` | orchestration | proof_freshness | `.aragora/proof_first_shift` | no |
| `publisher` | publication | publisher_freshness | `.aragora/automation-github-status` | no |
| `worktree_autopilot` | maintenance | worktree_health | `.worktrees` | no |
| `nomic` | self_improvement | (none) | `.aragora_beads` | yes (approval checkpoints) |
| `docs_sync_drift` | maintenance | docs_drift | `.aragora/docs_drift_status.json` | no (its PRs settle through the quorum gate) |

### `docs_sync_drift` (added after PR #8089)

`scripts/docs_sync_drift_detector.py` is a bounded single-shot iteration
(launchd daily via `scripts/install_docs_drift_launchd.sh`) that regenerates
the docs surface against a throwaway worktree of `origin/main` using the exact
commands the `Build Documentation (PR Check)` workflow runs. It exists because
the external run-canceller (`docs/governance/PR_RUN_CANCELLATION_DIAGNOSIS.md`)
can kill that advisory check on a source-doc PR, letting source changes land
without regenerated mirrors (observed escapes: #7829, #7814) - the next
doc-touching PR then inherits a red docs check it did not cause.

Guard design: drift confined to generated mirrors (`docs-site/docs/**`) yields
at most **one** open sync PR (branch namespace `bot/docs-site-sync`), which
settles through the normal model-quorum merge gate - the detector never merges,
approves, or comments. Drift touching anything else (for example `doc_stats`
stamp targets such as protected `CLAUDE.md`) **fails closed to report-only**
(`drift_outside_allowlist`). Waiting on an already-open sync PR is classified
as *waiting*, not a fault - the §4 distinction applied to a brand-new loop.

---

## 4. The fault-vs-waiting distinction (why v1 exists)

The merge arbiter's circuit breaker is meant to *fail closed* on operational
faults. But "made no merge this poll" is **not** the same as "is broken":

- **Waiting:** no PR is mergeable yet (checks pending, quorum not yet reached).
  Correct action: **keep waiting.**
- **Fault:** the GitHub API is failing, auth is broken, or evaluation/merge is
  raising. Correct action: **halt / fail closed.**

Conflating them (PR #7879's defect class) means the loop either trips its breaker
on benign waiting or spins uselessly to `max_runtime` on a real fault. The Loop
Control Plane encodes the correct distinction directly in `classify_loop`:

- a not-ready, *waiting* loop -> state `waiting`, next action `wait`;
- an operational fault (or unrecognized stop reason, fail-closed) -> state
  `blocked`, next action `halt`;
- an unreadable loop -> state `unknown`, next action `report_only` (never an
  implied "continue").

`audit_halt_readiness` then flags `merge_arbiter` as `incomplete` precisely
because its *guard* does not yet make this distinction, so the gap is visible at
the fleet level until the loop itself is fixed.

---

## 5. The `LoopRecord` contract

Every loop is normalized into a `LoopRecord` (`aragora/swarm/loop_control.py`).
The stable, machine-readable fields:

```jsonc
{
  "schema_version": "loop-control/v1",   // forward-compat hook (see §7)
  "loop_id": "merge_arbiter",
  "kind": "merge_arbiter",
  "role": "supervisor",
  "owner": "launchd",
  "state": "running|waiting|blocked|budget_exhausted|halted|human_gated|stale_owner|unknown",
  "ticks": null, "max_ticks": 3,
  "runtime_s": null, "max_runtime_s": 43200.0,
  "last_progress_at": null, "no_progress_ticks": null,
  "budget": {"spend_usd": null, "ceiling_usd": null, "remaining_usd": null,
             "source": "none", "source_status": "unavailable"},
  "feedback_gate": {"kind": "quorum", "status": "quorum"},
  "halt_readiness": {"max_iteration": true, "no_progress": true,
                     "no_progress_distinguishes_fault": false,
                     "budget_ceiling": false,
                     "verdict": "incomplete", "gaps": ["..."], "notes": ["..."]},
  "durability": {"state_path": null, "restart_safe": false},
  "human_gate": {"required": true, "present": false},
  "blocker": null,
  "next_action": "report_only|continue|wait|halt|escalate_human",
  "source_paths": ["scripts/run_merge_arbiter.sh", "aragora/swarm/merge_arbiter.py"],
  "source_status": "ok|degraded|unavailable"
}
```

`source_status` and `source_paths` make every record self-describing: a consumer
can tell *how fresh / trustworthy* a record is and *where it came from* without
re-deriving it. `schema_version` is carried on every record so a future ledger
(§7) can evolve the shape without breaking readers.

---

## 6. Usage and guarantees

```bash
# Human table for the whole fleet
python3 scripts/loop_control_status.py

# JSON (records + fleet summary) for automation
python3 scripts/loop_control_status.py --json

# One loop, offline (skips the network-touching operator snapshot)
python3 scripts/loop_control_status.py --loop merge_arbiter --no-network --json

# Non-zero exit if any loop should halt/escalate or the fleet is unsafe
python3 scripts/loop_control_status.py --exit-nonzero-on-halt
```

**Read-only guarantee.** The IO layer (`aragora/swarm/loop_control_io.py`) is the
only part that touches the world, and it only ever *reads*: `launchctl print`,
`git worktree list --porcelain`, `scripts/publisher_freshness_check.py --json`,
`scripts/agent_bridge.py operator-snapshot --json`, and a file read of
`.aragora/proof_first_shift/runtime_state.json`. It never merges, comments,
reruns, pushes, or passes `--apply`, and it writes nothing. Collectors degrade to
`degraded`/`unavailable` on missing files, rate limits, timeouts, or non-POSIX
hosts rather than raising. The classifier (`loop_control.py`) is pure - no IO at
all - which is what makes the read-only property auditable and the logic
trivially unit-testable.

### Relationship to existing surfaces ("A2, generalized")

- `scripts/settle_status.py` is the **per-PR** read-only settlement view (A2).
  The Loop Control Plane is the **per-loop, fleet-level** generalization of that
  same "derive truth read-only from existing evidence" pattern.
- `scripts/reconcile_merge_quorum.py` / `aragora/swarm/merge_quorum_reconcile.py`
  reconcile merge-quorum settlement (A1). The plane *observes* the loops that
  drive A1; it does not reconcile.
- Feedback gates referenced: `docs/governance/QUORUM_EVIDENCE_RUNBOOK.md`,
  `docs/governance/MERGE_GATE_RECONCILIATION.md`,
  `docs/governance/BOSS_LOOP_MERGE_GATE_PHASE3_DESIGN.md`.

> The halt-readiness guards in `LOOP_SPECS` are a **curated, code-referenced
> audit** of each loop's design as of the cited revision - not auto-derived
> facts. Update them when a loop's guards change. v1 deliberately does not
> attempt to auto-derive guards from source (see §7).

---

## 7. Future LoopLedger Contract (follow-up, NOT in v1)

v1 ships `LoopRecord` and exercises its shape, but it does **not** write any
ledger files and does **not** integrate loop emitters. The following is a
documented intent only, to be designed once `LoopRecord` has proven stable
across all loops.

A future **LoopLedger** would be an append-only, restart-safe history of
`LoopRecord` snapshots, enabling: long-running loop audits, "no progress for N
ticks across restarts" detection, and post-hoc spend/halt accounting.

Proposed (not frozen) contract:

- **Location:** `.aragora/loop_control/ledger.jsonl` (one JSON `LoopRecord`
  envelope per line), mirroring `proof_first_shift/shift_ledger.jsonl`.
- **Envelope:** `{ "schema_version", "recorded_at", "record": <LoopRecord> }`.
  Readers MUST branch on `schema_version` and ignore unknown fields.
- **Producer:** an opt-in emitter that calls `collect_fleet(...)` on the existing
  poll cadence and appends; it MUST remain read-only with respect to the loops
  themselves (it only observes them).
- **Retention/compaction:** TTL + size cap, reusing the worktree-autopilot TTL
  conventions.

**Explicit non-goals for v1 (and until the ledger is designed):**

- No ledger files are written.
- No loop is modified to emit records.
- No active controller: nothing in this plane halts, reruns, reconciles, or
  otherwise changes any loop's behavior. Surfacing `next_action: halt` is a
  *recommendation* for a human or a separately-authorized controller, not an
  action taken here.

When the ledger is built, the only schema change expected is additive; the
`schema_version` field exists so that change does not break existing readers.

---

## 8. Scope summary

| In v1 | Not in v1 |
|---|---|
| Read-only fleet inventory of standing loops | Any mutation of any loop |
| Per-loop `next_action` (continue/wait/halt/escalate/report) | An active controller that performs those actions |
| Curated halt-readiness audit (3 hard stops) with code refs | Auto-derivation of guards from source |
| `LoopRecord` with `schema_version` forward-compat hook | A written LoopLedger or loop-emitter integration |
| Budget *surfacing* (and flagging the missing ceiling) | Wiring real per-loop dollar budgets |
