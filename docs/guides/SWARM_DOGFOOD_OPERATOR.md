# Supervised Swarm Dogfood Operator Guide

Use this runbook for the first spec-driven supervised swarm run in a local Aragora checkout. Keep work-order file scopes disjoint, use explicit expected tests, and let the reconcile loop move work instead of doing manual worktree surgery.

## 1. Create the spec

You can generate the initial envelope from the CLI:

```bash
python -m aragora.cli.main swarm \
  "Ship the first supervised swarm dogfood run" \
  --skip-interrogation \
  --dry-run \
  --save-spec /tmp/swarm-dogfood.yaml
```

Then edit the spec so each work order owns a narrow `file_scope` and concrete `expected_tests`.

```yaml
raw_goal: Ship the first supervised swarm dogfood run
refined_goal: Ship the first supervised swarm dogfood run
work_orders:
  - work_order_id: docs-lane
    title: Write operator guide
    description: Add the first supervised runbook.
    file_scope:
      - docs/guides/SWARM_DOGFOOD_OPERATOR.md
    expected_tests: []
    target_agent: codex
    reviewer_agent: claude
    risk_level: info
  - work_order_id: swarm-tests
    title: Cover spec and reconcile flow
    description: Add CLI and commander regression coverage.
    file_scope:
      - tests/cli/test_swarm_command.py
      - tests/swarm/test_commander.py
    expected_tests:
      - python -m pytest tests/cli/test_swarm_command.py tests/swarm/test_commander.py -q
    target_agent: claude
    reviewer_agent: codex
    risk_level: review
  - work_order_id: integration-tests
    title: Cover reconcile and integration queue flow
    description: Add reconciler and queue regression coverage.
    file_scope:
      - tests/swarm/test_reconciler.py
      - tests/worktree/test_integration_worker.py
    expected_tests:
      - python -m pytest tests/swarm/test_reconciler.py tests/worktree/test_integration_worker.py -q
    target_agent: codex
    reviewer_agent: claude
    risk_level: review
```

## 2. Provision the run without dispatching workers

Use `--no-dispatch` when you want to confirm the run shape before any worker sessions start:

```bash
python -m aragora.cli.main swarm --spec /tmp/swarm-dogfood.yaml --no-dispatch --json
```

Record the returned `run_id`. This should create the supervisor run and lease available work orders without launching workers.

## 3. Dispatch and reconcile

Use the reconcile loop as the operator control plane:

```bash
python -m aragora.cli.main swarm reconcile \
  --run-id <run_id> \
  --watch \
  --interval-seconds 2 \
  --json
```

Useful variants:

```bash
python -m aragora.cli.main swarm --spec /tmp/swarm-dogfood.yaml --dispatch-only --json
python -m aragora.cli.main swarm status --run-id <run_id> --json
python -m aragora.cli.main swarm reconcile --run-id <run_id>
python -m aragora.cli.main swarm reconcile --all-runs
```

- `--dispatch-only` launches workers and returns immediately.
- `--no-dispatch` provisions supervisor state without launching workers.
- `reconcile --watch` is the operator-friendly path for topping up work, collecting finished workers, and syncing pending integration work.

## 4. Inspect the integration queue

Use the fleet merge queue as the integration surface:

```bash
python -m aragora.cli.main worktree fleet-queue-list
python -m aragora.cli.main worktree fleet-queue-list --status needs_human
python -m aragora.cli.main worktree fleet-queue-list --status blocked
```

Expected queue metadata for dogfood runs includes:

- `receipt_id`
- `changed_paths`
- `tests_run`
- `confidence`

Typical queue states:

- `queued`: ready for validation
- `needs_human`: validated or blocked by an explicit gate
- `blocked`: merge conflict or validation failure
- `merged`: integrated successfully

## 5. Validate or execute the next integration item

Validation only:

```bash
python -m aragora.cli.main worktree fleet-queue-process-next \
  --worker-session-id integrator-1 \
  --json
```

Validate and merge:

```bash
python -m aragora.cli.main worktree fleet-queue-process-next \
  --worker-session-id integrator-1 \
  --execute \
  --json
```

Use validation first on the first dogfood run. Execute only after the queue item shows the expected files, tests, and conflict metadata.

## 6. When the run needs help

Check these fields before intervening:

- `dispatch_error` or `resource_error` on the swarm run
- `reconciler_conflicts` on the queue item
- `merge_error` or `merge_conflicts` on the integration outcome

For the first supervised dogfood run, prefer fixing the narrow blocking issue and re-running the reconciler instead of broad manual cleanup.
