# Boss Loop One-Shot Verify-Then-Restock — 2026-04-25

One-shot pass over the boss loop: verify health, inspect terminal-class signal,
sanity-check the swarm test suite, and decide whether to restock the queue.

- Triggered by: human one-shot run (no automation handoff)
- Repo state: HEAD pinned at `2cfd9f30 docs: align public scale claims with canonical metrics (#6582)` (matches `origin/main`)
- Environment: shared session container, fresh deps installed for this run
  (pydantic, pydantic-settings, cryptography, defusedxml, aiohttp, numpy,
  idna, pytest, pytest-asyncio). No `.aragora/overnight/` runtime state present.

## Phase 1 — Verify

### 1a. `python scripts/verify_system_health.py`

```
{
  "agent_health": {
    "null_bytes_found": false,
    "timeout_handling_present": true,
    "cli_integration_present": true
  },
  "loop_id_routing": {
    "loop_id_binding_present": true,
    "validation_present": true
  },
  "auth_readiness": {
    "module_exists": false,
    "check_auth_implemented": false,
    "websocket_support": false
  }
}

Prioritized fixes:
- LOW: Add auth module - security scaffolding
```

Result: PASS for agent health and loop-id routing. The auth-readiness section
flags a LOW-priority gap (auth scaffolding missing) — pre-existing, not a
stop-the-world issue.

### 1b. `python scripts/analyze_boss_metrics.py`

The default boss-loop metrics path (`.aragora/overnight/boss_metrics.jsonl`)
does not exist in this environment — no overnight loop has run here. The
analyzer was therefore exercised against the bundled fixture
`benchmarks/fixtures/swarm/sample_boss_metrics.jsonl` purely to confirm the
tool still loads and renders correctly:

```
records: 3
prompt_chars avg: 1166.7
deliverable rate: 33%
terminal-truth no-rescue rate: 33%
terminal-truth meets 30d target: False
terminal-truth actionable failures: 1
publish actions: opened_pr: 1
families: blocked: 1, rescue: 1, success: 1
classes: blocked_not_dispatch_bounded: 1, deliverable_branch_pushed: 1, rescue_worker_crash: 1
```

Status: tool still works end-to-end (loads → scores → renders text report).
**No live boss-metrics dataset is available to evaluate against the
30%-blocked / 60%-no-rescue thresholds in this environment.** The fixture
numbers are not representative and are reported only to prove the analyzer
functions correctly.

### 1c. Last ~50 cycles of `boss_metrics.jsonl`

Cannot be inspected in this environment — see above. Anywhere boss-loop
metrics are persisted (`.aragora/overnight/boss_metrics.jsonl` per
`aragora/swarm/boss_loop.py:333`) is empty/absent. Recommend re-running this
verify pass from the host that owns the live overnight loop, or pointing
`--metrics-file` at a synced snapshot.

### 1d. Boss loop import + swarm test suite

- `python -c "from aragora.swarm.boss_loop import BossLoop"` → imports cleanly.
- `pytest tests/swarm/ -q -x --ignore=tests/swarm/perf` (`tests/swarm/perf`
  does not exist in this checkout, so `--ignore` is a no-op):
  - **321 passed, 1 skipped, 1 failed in 181.20s**
  - Failure (only):
    `tests/swarm/test_boss_loop.py::TestBossLoop::test_specific_issue_number_scope_conflict_reports_overlap_reason`

Failure summary:

```
assert 'overlaps files already owned by open PR or in-flight work'
       in 'Target issue #873 is missing required labels: boss-ready.'
tests/swarm/test_boss_loop.py:1665: AssertionError
```

The test sets up issue #873 with labels `["boss-ready", "priority:critical",
"autonomous"]` and a config requiring those same labels, then expects
`needs_human_reasons[0]` to surface the *scope-conflict* reason (the
issue's only file overlaps the blocked-scope set). The current loop instead
short-circuits with a `missing required labels: boss-ready` message — i.e.
the require-labels gate is firing before the scope-overlap gate, even
though the issue clearly carries the `boss-ready` label. Likely culprits:
either the require-labels normalization treats the explicitly-targeted
issue differently than label-filtered fetches, or the gate ordering was
inverted by a recent refactor. Not stop-the-world (only one targeted
specific-issue path), but the user-facing `needs_human_reasons` text is
wrong here, so worth filing.

## Phase 2 — Restock decision

`aragora/swarm/queue_autofill.py` no longer exposes a dispatcher
(`maybe_autofill_queue` was removed; module is now passive types only —
see the docstring referencing `docs/plans/2026-04-19-batched-pr-review-triage.md`).
Queue depth was therefore measured directly via the GitHub MCP `list_issues`
tool against `synaptent/aragora` with the `boss-ready` label:

- **Open boss-ready issues: 16** (well above the 5-issue threshold)
- Newest titles include:
  - #6549 Open PR for detached worker-launcher stdin mock repair
  - #6548 Open PR for security debate coroutine warning test isolation
  - #6546 Open PR for branch-aware automation outbox handoff dedupe
  - #6545 Open PR for restacked offline publisher dry-run repair
  - #6544 Open PR for offline publisher dry-run repair
  - #6542 Open PR for publishable Codex branch backlog gate
  - #6540 Open PR for automation outbox idempotency dedupe
  - #6487 Publish or confirm `benchmark-truth-publication/24843546388`
  - #6460 Publish or confirm `benchmark-truth-publication/24785358567`
  - #6385 Publish or confirm `benchmark-truth-publication/24729432387`
  - #6349 Export the sandbox route contract
  - #6321 Publish or confirm `benchmark-truth-publication/24606383976`
  - #6298 Fix review-queue parsing for state-based GitHub check rollups
  - #6291 Normalize review auto-detected agent aliases
  - #6289 Harden persisted coordination payload coercion
  - #6187 Make benchmark corpus freshness tests resolve shared overnight metrics

**Action: SKIP restock.** Dispatchable depth is 16 ≫ 5. Running
`scripts/generate_boss_issues.py` would only deepen an already-saturated
queue and increase pressure on the duplicate-detection / PR-conflict
filters. No `--dry-run` was executed.

## Open PR references

Boss-ready queue is dominated by `Open PR for …` items — those titles refer
to *existing* draft PRs that need landing or rebase, not new work. A few
representative numbers from the queue above (#6540, #6542, #6544, #6545,
#6546, #6548, #6549) suggest the bottleneck is on the *publish/merge* side
of the loop, not the *generate* side. That is consistent with the decision
to skip restock.

## Recommended follow-up issues (NOT filed by this run)

1. **`boss-loop scope-overlap reason short-circuited by require-labels gate`** —
   `tests/swarm/test_boss_loop.py::TestBossLoop::test_specific_issue_number_scope_conflict_reports_overlap_reason`
   fails because `needs_human_reasons[0]` reports `missing required labels:
   boss-ready` when the issue actually carries that label. Investigate the
   gate ordering or label-normalization for the `--issue-number`
   single-target path in `aragora/swarm/boss_loop.py`. Suggested labels:
   `boss-ready`, `priority:high`, regression.

2. **`boss-metrics analyzer: support gracefully-empty live metrics path`** —
   `scripts/analyze_boss_metrics.py` requires `--metrics-file`, and there is
   no convenience flag for "use the canonical
   `.aragora/overnight/boss_metrics.jsonl` if present, otherwise emit an
   informative message". Useful if more verify passes will run on hosts
   without the overnight loop's runtime state. Low priority.

3. **`auth scaffolding gap flagged by verify_system_health`** — pre-existing
   LOW priority; tracked here for visibility, not introduced by this run.

## Outcome

- Verify: PASS (one regression-class test failure, otherwise green;
  cannot evaluate live terminal-class distribution from this host).
- Restock: SKIPPED — queue depth 16 ≫ 5 threshold; `queue_autofill` is
  passive-types-only and `generate_boss_issues.py` was not invoked.
- No issues or PRs opened by this run.
