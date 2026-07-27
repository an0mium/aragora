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
  `765fc2bb7402405de39a6359b29ff29db21e28f72848403191b5235e13e7d9be`
- Issue: `#9577`
- Pull request: draft `#9591`
- Staging base: `c7c4681eb08d5e7c7966d10dfba3a1520d671319`
- Lease: `8e6a0fa7-5ef`
- Run mode: finite, two batches, eight-hour lease budget
- Merge owner: user

## Current state

Batch 1 is pushed at exact PR head `2433d4975f...`. The exact authorized main
`1cd722cc...` and later non-overlapping main `b0633c5f...` were merged without
rewriting published history. PR #9589's dependency, override, and 1,034
non-root lock records are preserved; this lane adds only the root Node engine
metadata.

The package contract, 21 live workflow selectors, three preserved Node 20
exceptions, and exhaustive focused test are committed at `6a2426923a...`.
Batch 2 exact-tip validation is green: complete `tests/ci` `200 passed, 1
skipped`, focused Ruff checks, workflow policies, diff check, and preflight
PASS, and the exact Node `24.18.0` install/lint/typecheck/228-route build PASS.
All pushed-head non-quorum required checks are green. Final independent review
found one fail-open test-ordering gap; it is repaired and the complete
validation set is green again. Independent re-review found no blocker with
high confidence.

The single next action is to remove session-only operational artifacts,
perform final gates, push, poll the exact head, release the lease, and stop
for OWNER handling.

## Stop Gate

Stopping is currently allowed: **yes**.

Reason: both planned batches, repeated validation, and final independent
re-review are complete with no blocking finding. Only deterministic
operational cleanup, final push/poll, and lease release remain.

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
3. renew lease `8e6a0fa7-5ef` with session
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

- The 13 approved workflows now resolve exact Node `24.18.0` for all 21 jobs
  that install or execute `aragora/live`.
- `release.yml` and `test.yml` also contain Node 20 selectors for non-live
  packages; those selectors are not automatically in scope.
- `live-deploy-mode-gate.yml` already runs `npm ci`, TypeScript checks, and a
  frontend build for live paths, making it the preferred build-proof surface.
- The gate skips draft PRs, so PR #9591 cannot obtain its Node 24 build proof
  from that CI job while remaining draft. Use the exact-Node container proof.
- `aragora/live/package.json` and the lockfile root now declare
  `>=24.18.0 <25`.
- Current main's five non-quorum protected required contexts are green. The
  visible uptime-monitor failure is not protected.
- Three Linux `aragora` runners were online at the latest Batch 1 pre-push
  tripwire.
- The pre-edit exact-Node container baseline passed on Node `24.18.0`: `npm
  ci`, lint, TypeScript, and the complete 228-route Next build.
- PR #9589 subsequently changed `package.json` overrides and regenerated the
  dependency lockfile without adding an engine declaration.

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
