# Claim-First Lane Dispatch

How to run several merge-advance worker sessions (Codex CLI, Claude Code,
Factory Droid) against the open-PR queue **without** them colliding on the same
PRs — and without hand-pasting ever-growing recursive prompts.

## The problem this solves

The cross-agent lane-lease registry already exists end to end:

- `scripts/claim_active_agent_lane.py` writes atomic claims (lock-file +
  tmp+rename) into `.aragora/agent-bridge/lanes.json`.
- `scripts/identify_lane_owner.py` reads them back with owner-lease liveness
  (lease age, heartbeat, lane-ledger status, stale/terminal assessment).

The chronic failure mode is **not** a missing primitive. As
`claim_active_agent_lane.py`'s own docstring puts it: *"many sessions … never
write a lane claim, so the registry tends to stay empty even when 5-10
concurrent agents are running."* Each session then free-picks "the highest-value
open PR," several converge on the same one, and the duplicate
evidence/settlement work is thrown away when one of them lands first.

Two anti-patterns drive it:

1. **Free-pick** — every worker independently scouts the queue and picks a PR.
2. **Detect-don't-claim** — workers `identify_lane_owner` to *see* an owner but
   never *claim* before working, leaving a race window.

The fix is **claim-first dispatch**: a conductor assigns each unclaimed PR to at
most one worker, and the worker claims it atomically before doing anything.

## The two halves of the loop

| Half | Module | Role |
|------|--------|------|
| **Front** (assign) | `aragora.swarm.lane_dispatcher` | Pick which unclaimed merge-blocked PR each free worker gets; emit its prompt. |
| **Middle** (claim + dispatch) | `aragora.swarm.lane_conductor` | Claim one free lane and drop an inspectable work order for the supervisor. |
| **Back** (launch) | `aragora.swarm.lane_supervisor` | Drain pending work orders through the worker-launcher state machine. |

`lane_dispatcher` is **pure-core**: no GitHub calls, no `lanes.json` writes, no
process spawning. It takes the merge-blocked candidates and the set of PRs that
already have a *live* owner (liveness resolved by `identify_lane_owner`) and
returns assignments. The live wiring — resolve candidates via
`merge_quorum_io`, resolve live owners via `identify_lane_owner`, spawn workers
via `aragora.swarm.worker_launcher` — sits in the CLI/conductor shell so the
decision stays unit-testable.

## Dispatch decision

```bash
python3 scripts/lane_dispatcher.py --json \
    --candidates-json '[{"number":8405,"branch":"codex/a"},{"number":8406,"branch":"codex/b"}]' \
    --live-claims-json '{"8406":"sess-existing"}' \
    --max-workers 3
```

- Candidates are merge-blocked PRs in **priority order**.
- Live claims map a PR to its current live owner (a *stale/terminal* owner must
  not appear, so its lane is reassignable).
- Output: `assignments` (PR + branch + a fresh `owner_session`), `owned` (PRs
  skipped because a live owner holds them), `deferred` (left over once
  `--max-workers` is hit — backpressure instead of unbounded fan-out).

## The worker prompt

Short and **constant** — the guardrails live in the claim/merge-gate tooling and
the lane registry, not in pasted text that grows every turn. Generate it with:

```bash
python3 scripts/lane_dispatcher.py --print-prompt --pr 8405 --branch codex/a
```

It instructs the worker to:

1. **Claim-or-yield** — `identify_lane_owner --pr N`; if a *live* owner that
   isn't this session holds it, print `yielding` and STOP; else
   `claim_active_agent_lane … --release-stale`.
2. **Ground** from live state for that PR only (never trust memory).
3. **Advance one bounded step** toward merge (rerun one stale failed required
   check / collect one exact-head two-family evidence set only if evidence-lint
   `would_count` / one narrow repair in an isolated worktree). Never merge,
   admin-merge, settle Tier-4, touch branch protection, or touch another PR.
4. **Report + release** — print the new head, action, result, single blocker;
   refresh the heartbeat; release the lane if merged or blocked.

It explicitly forbids scouting the queue or switching PRs — assignment comes
from the conductor. This is what ends both the contention **and** the
prompt-bloat: ~15 fixed lines instead of a 200-line recursive prompt that
regenerates and grows itself each turn.

## Wiring the autonomous conductor (no copy-paste)

A long-running loop closes the gap so the operator moves from the message bus to
the escalation channel:

1. Resolve merge-blocked PRs (`merge_quorum_io`) → `candidates`.
2. Resolve live owners (`identify_lane_owner --json`) → `live_claims`.
3. `lane_dispatcher.select_assignments(...)` → assignments (capped at
   `--max-workers`).
4. For each assignment: `claim_active_agent_lane.py` (atomic), then launch the
   worker with `build_worker_prompt(...)` via `aragora.swarm.worker_launcher`.
5. Inspect each worker's completed/failed work order in
   `.aragora/lane_dispatch/{done,failed}/` and route the next action through the
   normal merge-gate or operator-steering path.
6. A higher *validator* model reviews results and injects corrective steering
   into the `.aragora/operator-steering/<session>/` mailbox that
   `identify_lane_owner` already surfaces.

The recursion is intentional and continuous (an always-advancing front); what
the conductor removes is the **uncoordinated** part (collisions) and the
**hand-cranked** part (copy-paste).

### Running the conductor

`aragora.swarm.lane_conductor` implements one *pass* of that loop, and
`scripts/lane_conductor.py` is its CLI. **Dry-run by default; it never merges or
settles** — it only assigns, claims, and drops work orders.

```bash
# Preview the next pass (no claims written, no work orders dispatched):
python3 scripts/lane_conductor.py --json --max-workers 3

# Actually claim the lanes and drop work orders for the supervisor to spawn:
python3 scripts/lane_conductor.py --execute --max-workers 3
```

Under `--execute`, each assignment is claimed via `claim_active_agent_lane.py`
(atomic) and its work order is written to
`.aragora/lane_dispatch/pending/<id>.json`. A work order is self-describing
(`target_agent`, `owner_session_id`, `pr`, `branch`, and the claim-first
`prompt`), and its keys match what `worker_launcher.WorkerLauncher.launch`
reads, so the supervisor adapter that drains the pending dir can hand it
straight to the launcher. The file-drop keeps the conductor decoupled from
worktree provisioning and makes every dispatch inspectable/replayable.

The decision core (`plan_pass`, `build_work_orders`) is pure — the `gh` /
`identify_lane_owner` reads and the claim/dispatch writes are injected
callables, so the whole pass is unit-tested without a network or a worktree.

### Draining work orders (the supervisor)

`aragora.swarm.lane_supervisor` is the back of the handoff — it drains the
work orders the conductor dropped and launches a worker for each, via a
file-state machine that is the double-spawn guard:

```
pending/  --claim (atomic rename)-->  in_progress/  --+--> done/
                                                      +--> failed/
```

The atomic `pending -> in_progress` rename means two concurrent supervisors
never launch the same order: exactly one rename wins, the loser skips. A
launch that raises moves the order to `failed/` with the error recorded and the
drain continues; successes move to `done/`. Everything stays inspectable and
replayable.

```bash
# Preview what the next drain would launch (moves nothing):
python3 scripts/lane_supervisor.py --json

# Claim + launch up to N pending orders:
python3 scripts/lane_supervisor.py --execute --max-launches 3
```

Under `--execute` each order is handed to
`worker_launcher.WorkerLauncher.launch`. That call is async and needs a
provisioned worktree (operator-machine-specific), so the work order must carry a
`worktree`; orders without one fail cleanly into `failed/` rather than launching
unisolated. The drainer state machine is fully unit-tested with an injected fake
launcher; the live `WorkerLauncher` seam is the one piece to validate on your
machine.

## Identity / budget note

Run worker reads through the GitHub App installation token (separate API
budget) so concurrent workers don't starve the operator's shared per-user PAT
quota — see `aragora.swarm.github_app_auth` and the read-routing in
`aragora.swarm.merge_quorum_io`. Reserve the operator PAT (`an0mium`) for the
writes the App token 403s on; keep `scarmani` for human-gate settlement only.
