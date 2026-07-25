# Survival Guide — Node 24 Runtime Contract (#9577)

Read this file first after any compaction, restart, handoff, or crash. Then read
`.elves-session.json`, `learnings.md`, the plan, and `execution-log.md`, in
that order.

## Identity

- Session: `elves-node-runtime-contract-9577-20260724`
- Lane: `node-runtime-contract-9577-20260724`
- Branch: `codex/node24-runtime-contract-9577`
- Worktree:
  `$HOME/.codex/worktrees/node24-runtime-contract-9577/aragora`
- Plan:
  `docs/plans/2026-07-24-node-runtime-contract-9577.md`
- Plan SHA-256:
  `4a3adbc45ca347b457b30bd0adb923d77cc1b208b5f7aa33792aadf1c712f071`
- Issue: `#9577`
- Pull request: draft `#9591`
- Staging base: `c7c4681eb08d5e7c7966d10dfba3a1520d671319`
- Lease: `00ea44f4-78d`
- Run mode: finite, two batches, eight-hour lease budget
- Merge owner: user

## Current state

Batch 1 launch is in progress. The launch exact-head tripwire matched
`389556c2217f0ea26965703c17f86c65db5bd177`; no product edit has occurred yet.

The single next action is to re-read steering, renew the lease, verify current
main overlap plus runner/main health, reconcile a clean main advance without
rewriting branch history, capture the baseline, and write the Batch 1
contract.

## Stop Gate

Stopping is currently allowed: **no**.

Reason: the separate launch call has begun and both planned batches remain.
Continue through implementation, exact-runtime validation, independent review,
operational-artifact cleanup, and final parked-draft readiness.

This gate may return to **yes** only when Batch 1 and Batch 2 are complete, the
final draft exact head is validated and independently reviewed, operational
artifacts are removed, and the run has stopped for OWNER handling.

Never interpret generic `proceed` as authority to mark ready, collect evidence,
settle, or merge.

## Non-negotiable scope

The product scope is exactly:

- `aragora/live/package.json`
- `aragora/live/package-lock.json`
- the 13 workflow files listed in the plan, limited to selectors for jobs that
  consume `aragora/live`
- `tests/ci/test_live_node_runtime_workflows.py`

The plan and temporary Elves artifacts are the only additional staging paths.
Do not touch PR #9505, unrelated Node selectors, branch protection, runner
configuration, secrets, deployment settings, dependencies, Next.js,
Dockerfiles, or Docker Compose.

Do not run another Fable/inference consult. Independent review must be local
and non-countable. Do not collect quorum evidence, settle, mark ready, or
merge.

## Restart protocol

Run from the isolated worktree:

```bash
cd "$HOME/.codex/worktrees/node24-runtime-contract-9577/aragora"
git status --short --branch
git rev-parse HEAD
git fetch origin main --quiet
git rev-parse origin/main
```

Read state in this exact order:

```bash
sed -n '1,260p' docs/elves/node-runtime-contract-9577/survival-guide.md
python3 -m json.tool .elves-session.json
sed -n '1,260p' docs/elves/node-runtime-contract-9577/learnings.md
sed -n '1,320p' docs/plans/2026-07-24-node-runtime-contract-9577.md
sed -n '1,320p' docs/elves/node-runtime-contract-9577/execution-log.md
```

Then:

1. re-read `$HOME/.codex/aragora_steering/mailbox.jsonl`;
2. run `scripts/read_operator_steering.py` for the branch and, once known, the
   PR;
3. renew lease `00ea44f4-78d` with session
   `elves-node-runtime-contract-9577-20260724`;
4. refresh lane `node-runtime-contract-9577-20260724`;
5. verify the exact PR head and compare current `origin/main` with the recorded
   base;
6. inspect overlap before any rebase or product edit;
7. verify at least three online `aragora` runners and no sustained protected
   required-check failure on main.

## Lease renewal

Use the same session identity and exact scope recorded in the active lease.
The minimal renewal is:

```bash
ARAGORA_SESSION_ID=elves-node-runtime-contract-9577-20260724 \
ARAGORA_AGENT=codex \
python3 scripts/check_work_lease.py codex/node24-runtime-contract-9577 \
  --repo . --renew --strict --work-id issue:9577 --ttl-hours 8 --json
```

Before every push, verify ownership:

```bash
ARAGORA_SESSION_ID=elves-node-runtime-contract-9577-20260724 \
ARAGORA_AGENT=codex \
python3 scripts/check_work_lease.py codex/node24-runtime-contract-9577 \
  --repo . --verify-only --strict --work-id issue:9577 --json
```

## Commit and push protocol

- Batch 0 commits:
  `[codex/node24-runtime-contract-9577 · Batch 0/2] ...`
- Batch 1 commit:
  `[codex/node24-runtime-contract-9577 · Batch 1/2] ...`
- Batch 2/finalization commits:
  `[codex/node24-runtime-contract-9577 · Batch 2/2] ...`
- Add:
  `Co-authored-by: codex[bot] <codex[bot]@users.noreply.github.com>`
- Before every push:
  - re-read steering;
  - verify lease and exact head;
  - run `git diff --check`;
  - run `bash scripts/automation_pr_preflight.sh origin/main HEAD`.

## Design decisions already made

1. Package consumers get the semver engine range `>=24.18.0 <25`.
2. GitHub Actions jobs executing `aragora/live` get exact `24.18.0`.
3. Workflow files containing both live and SDK/CLI jobs must be edited
   selectively. There is no global `20` to `24.18.0` replacement.
4. The lockfile change is limited to the root package engine field. Any
   dependency-record churn is a hard stop.
5. A focused test owns the 13-file inventory and proves live jobs resolve to
   the runtime contract while leaving unrelated package jobs independent.
6. An existing affected-path live job should provide the actual Node 24 build
   proof. Workflow structure changes require evidence that existing triggers
   are insufficient.
7. This is Tier 4 parked-draft work. The lane stops for OWNER handling after
   implementation, validation, and independent review.

## Known live observations

- The 13 approved workflows currently select Node 20 for jobs that install or
  execute `aragora/live`.
- `release.yml` and `test.yml` also contain Node 20 selectors for non-live
  packages; those selectors are not automatically in scope.
- `live-deploy-mode-gate.yml` already runs `npm ci`, TypeScript checks, and a
  frontend build for live paths, making it the preferred build-proof surface.
- `aragora/live/package.json` and the lockfile root currently have no
  `engines` field.
- Current main's five non-quorum protected required contexts are green. The
  visible uptime-monitor failure is not protected.
- Four `aragora` runners were online at staging preflight, including three
  Linux runners.

## Hard stops

Stop without product edits and ask one exact question if:

- steering, ownership, lease, exact head, current-main overlap, or runner/main
  health is not clean;
- a product change outside the exact scope is needed;
- any dependency or transitive package record changes;
- a live consumer cannot be distinguished safely from an unrelated Node job;
- the real Node 24 build cannot be proved without expanding scope;
- a blocking independent-review finding cannot be fixed in scope.

Do not use destructive git commands. Preserve all unrelated shared-root state.
