# Learnings — Node 24 Runtime Contract (#9577)

## 2026-07-24 staging

- The shared repository was clean at
  `c7c4681eb08d5e7c7966d10dfba3a1520d671319`; a fresh isolated worktree was
  created from that exact `origin/main`.
- No branch, PR, owner, lease, issue assignee, or issue-comment collision
  existed for issue #9577.
- The first lease attempt was correctly rejected because a broad `tests` path
  overlapped an unrelated active Boundary 2 lane. Narrowing the proposed test
  path to `tests/ci/test_live_node_runtime_workflows.py` eliminated the
  collision. Do not broaden the lease.
- Four self-hosted runners carrying the `aragora` label were online; three
  were Linux runners.
- Protected required contexts are `lint`, `typecheck`, `sdk-parity`,
  `Generate & Validate`, `TypeScript SDK Type Check`, and
  `aragora-merge-quorum`. The five non-quorum contexts were green on staging
  main. A separate uptime-monitor failure was non-required.
- `aragora/live/package.json` and its lockfile root have no engine
  declaration.
- The approved workflow inventory contains mixed-purpose files. In particular,
  `release.yml` and `test.yml` have selectors for SDK/CLI work as well as
  selectors for `aragora/live`. Implement by job ownership, not text
  replacement.
- `live-deploy-mode-gate.yml` already installs, type-checks, and builds
  `aragora/live`; it is the least-invasive candidate for actual Node 24 build
  proof if its trigger/path contract covers the changed package and workflow
  files.
- Staging and implementation are separate Elves calls. This turn must end
  after a docs-only setup commit, pushed draft PR, and launch prompt.

## 2026-07-25 launch

- The complete `tests/ci` baseline is `197 passed, 1 skipped`; checkout
  integrity and required-check priority policy scripts also pass.
- The unmodified frontend installs, lints, type-checks, and completes its
  228-route production build under the exact `node:24.18-alpine` image.
- A production Next build rewrites tracked `aragora/live/next-env.d.ts` from
  the dev route declaration to the production route declaration. Restore that
  generated side effect after validation; it is not part of this lane.
- The 13-workflow inventory contains 24 `setup-node` selectors: 21 serve
  `aragora/live` and should resolve to exact `24.18.0`; three unrelated
  selectors remain Node 20.
- `live-deploy-mode-gate.yml` skips draft PRs. Because #9591 must remain
  draft, exact-Node local/container proof is required and the PR must not claim
  that the draft CI gate performed the build.
- The generic tag `elves/pre-batch-1` was already owned by another historical
  run, so this lane uses
  `elves/pre-batch-1-node-runtime-contract-9577` without rewriting the shared
  tag.
- Current-main overlap is a live tripwire, not a one-time staging check. PR
  #9589 merged during baseline and changed both package metadata files, so the
  lane stopped before product edits as designed.
