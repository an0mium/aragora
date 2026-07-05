# The Lease Rule — one ownership truth for all fleets

**Rule: no automated branch push proceeds without holding the corresponding
work lease in the live `aragora.nomic.dev_coordination` store.**

Tracking: issue [#8851](https://github.com/synaptent/aragora/issues/8851)
(acceptance item 2).

## Why

Five fleets modify this repository concurrently — Factory mission workers,
agent-bridge lanes, boss-loop, queue-drain, and conductor sessions — and each
historically tracked branch ownership in a different system (lanes.json,
work orders, tmux session names, nothing at all). The recurring incident
class: two fleets adopt the same branch, one pushes over the other's commits
(foreign-commit contamination), or both implement the same task (duplicate
work). The `dev_coordination` lease store already exists, is SQLite-backed at
the git common dir (`<git-common-dir>/aragora-agent-state/dev_coordination.db`,
overridable via `ARAGORA_DEV_COORDINATION_DB`), and is already used by the
swarm supervisor — so it becomes the single ownership truth.

## The preflight helper

`scripts/check_work_lease.py` is a fast (<1s), dependency-light CLI. The
read-only check touches the SQLite store directly; only mutations import the
full `DevCoordinationStore` (so conflict detection, fleet-claim mirroring,
and event publication keep working).

```bash
# Before pushing: verify you hold the lease, claiming it if free.
python3 scripts/check_work_lease.py <branch> --claim \
    --session-id "$ARAGORA_SESSION_ID" --agent <agent-name> [--pr <n>]

# While working (long sessions): renew the TTL (default 8h).
python3 scripts/check_work_lease.py <branch> --renew

# When done (merged/abandoned): release.
python3 scripts/check_work_lease.py <branch> --release
```

Semantics:

| Situation | Exit | Output |
|---|---|---|
| You hold the active lease (or just claimed it with `--claim`) | 0 | `OK [check|claim] ...` |
| Another session holds it | 1 | one-line owner report (`LEASE CONFLICT: branch '…' leased by <agent>/<session> (lease …, expires …)`) |
| No lease held and no `--claim` | 1 | `no active lease held … run with --claim` |
| Store unreachable | 0 + `WARNING` on stderr (fail-open, v0) | `--strict` makes this exit 1 (fail-closed, v1) |

Branch defaults to the current branch of `--repo` (default: cwd). Session
identity comes from `--session-id`, else `ARAGORA_SESSION_ID` /
`ARAGORA_AGENT_SESSION_ID` / `ARAGORA_SWARM_SESSION_ID`; without any of
these it falls back to a host-level `user@host` identity with a warning —
set `ARAGORA_SESSION_ID` for real per-session ownership. `--json` gives
machine-readable output. `--path`/`--write-scope` pass file scopes through
to `DevCoordinationStore.claim_lease` for scoped claims.

The read-only check opens the store with `mode=ro` and falls back to
`immutable=1` when the WAL sidecars are unavailable (absent `-wal`/`-shm`
after a clean close or a DB copy, or a store directory not writable by the
invoking UID). The immutable read can be slightly stale — acceptable for an
advisory pre-check, since claims go through the store.

### Enforcement scope (precisely what conflicts where)

`DevCoordinationStore.claim_lease` detects conflicts on **file scope**
(`allowed_globs` / `claimed_paths`), not on the `branch` column. The helper
therefore enforces the branch rule in two layers:

- **Helper-mediated claims conflict at the store.** Every `--claim` carries
  a synthetic write-scope entry `.aragora/branch-locks/<branch>`; identical
  literal globs always overlap, so any two helper claims for the same
  branch conflict transactionally inside `claim_lease` — no read-then-write
  race window.
- **Store-direct fleets (swarm supervisor, boss-loop) conflict on file
  scope only.** A store-direct lease for the same branch with a
  non-overlapping file scope does not trip `claim_lease`; the helper
  detects it with a post-claim double-check and backs off (releases the
  just-claimed lease; earliest `created_at` wins). Read-only invocations
  (`check`/`--renew`/`--release`) block on any active foreign branch lease.
- **Full store-level branch uniqueness** (a `branch` conflict predicate
  inside `claim_lease` itself, covering store-direct claimants too) is a
  **v1 item**.

Claims are branch-keyed leases (`task_id` defaults to `branch:<branch>`).
The `--claim` path calls `store.claim_lease` first — the read-only
pre-check never blocks it — so the store reaps expired **and dead-worker /
heartbeat-stale** leases before deciding: a crashed fleet's lease with a
future `expires_at` cannot squat the branch until TTL.

## Adapters per fleet

Each fleet needs exactly one line at claim time and one before push:

- **Conductor / Claude Code / Codex sessions** — before any `git push` of an
  automated branch:

  ```bash
  python3 scripts/check_work_lease.py "$(git branch --show-current)" --claim || exit 1
  ```

- **agent-bridge lanes** — same preflight plus `--record-lane <lane_id>`,
  which writes the lane→lease mapping to the sidecar
  `.aragora/agent-bridge/lane-leases.json` (non-invasive: `LaneRecord`
  itself is unchanged; tooling that wants a lane's lease id reads the
  sidecar keyed by `lane_id`):

  ```bash
  python3 scripts/check_work_lease.py "$BRANCH" --claim \
      --session-id "$LANE_OWNER_SESSION" --record-lane "$LANE_ID"
  ```

- **Factory mission workers** — Factory skills live outside this repo, so
  the pattern is documented here rather than wired in code. At claim time
  (when the worker adopts its branch) and again immediately before push:

  ```bash
  python3 scripts/check_work_lease.py "$BRANCH" --claim \
      --session-id "droid-$FACTORY_MISSION_ID" --agent droid --pr "$PR_NUMBER"
  ```

  A non-zero exit means another fleet owns the branch: pick a new branch
  name or stop — never push over it.

- **boss-loop / queue-drain / swarm** — these already claim leases through
  `DevCoordinationStore` (see `aragora/swarm/supervisor.py`,
  `aragora/swarm/boss_loop.py`); the preflight is a no-op check for them
  and can be added to their push paths as a belt-and-braces guard.

## Rollout

- **v0 (now): fail-open with noise.** The preflight warns and exits 0 when
  the store is unreachable, so no fleet is bricked by a missing DB, a
  broken checkout, or an import failure. Conflicts and missing leases
  still fail (exit 1) — the rule is enforced whenever the store is
  readable.
- **v1: fail-closed.** Once all five fleets carry the preflight and the
  warning rate is ~0, flip invocations to `--strict` so an unreachable
  store blocks pushes too.

## Testing

`tests/scripts/test_check_work_lease.py` covers claim, hold, store-level
branch-lock conflict (owner report), store-direct-lease back-off, release
(including idempotency and non-owner refusal), renew, expired-lease and
heartbeat-stale-lease reclaim, the WAL `immutable=1` fallback (missing
sidecars + read-only store directory), the unreachable-store warn path,
`--strict` fail-closed, the lane sidecar roundtrip, and env-based session
identity:

```bash
pytest tests/scripts/test_check_work_lease.py -v
```
