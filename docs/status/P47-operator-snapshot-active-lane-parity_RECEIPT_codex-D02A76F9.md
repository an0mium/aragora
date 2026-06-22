# P47 Receipt: operator-snapshot active lane parity

Session: codex-D02A76F9
Timestamp: 2026-05-18T12:31:27Z
Branch: codex/P47-operator-snapshot-active-lane-parity-20260518-122154
PR: #7324

## Work Completed

Patched `scripts/agent_bridge.py` so lane registry reads merge both the user-level bridge registry and the repo-local `.aragora/agent-bridge/lanes.json` registry. Records are deduped by `lane_id`, preferring the newest `updated_at` and repo-local records when timestamps are missing or tied.

Added a focused regression in `tests/scripts/test_agent_bridge.py` that simulates a user-level registry alongside a repo-local active lane and verifies `operator-snapshot --summary-only` reports the active lane.

## Validation

Passed:

- `python3 -m pytest -q tests/scripts/test_agent_bridge.py::test_operator_snapshot_summary_counts_repo_local_lane_when_user_registry_exists tests/scripts/test_agent_bridge.py::test_operator_snapshot_summary_only_json_omits_records tests/scripts/test_agent_bridge.py::test_operator_snapshot_counts_active_duplicate_pr_lanes_as_conflicts`
- `python3 -m pytest -q tests/scripts/test_agent_bridge.py tests/scripts/test_agent_bridge_sessions.py`
- `python3 scripts/agent_bridge.py operator-snapshot --json --summary-only`
- `ruff check scripts/agent_bridge.py tests/scripts/test_agent_bridge.py tests/scripts/test_agent_bridge_sessions.py`
- `ruff format --check scripts/agent_bridge.py tests/scripts/test_agent_bridge.py tests/scripts/test_agent_bridge_sessions.py`
- `git diff --check`

Known pre-existing nonzero validation:

- `python3 scripts/agent_bridge.py --json health` exited 1 due two pre-existing `prunable_worktree` entries under `/private/var/folders/.../aragora-boss-harvest-*`. This lane did not touch cleanup.

## PR State

PR #7324 is ready for review at head `33e1af6e7554e367e6e47abca5dca14c632a29cf`.

At handoff, GitHub reported no failures, four pending checks (`Release Readiness`, `review`, `Security Scan`, `Baseline Determinism`), and one stale draft-run cancelled `lint-run` entry after the ready transition.

## Deferred

Did not merge. Did not clean worktrees. Did not repair the pre-existing `agent_bridge.py --json health` prunable-worktree warnings.
