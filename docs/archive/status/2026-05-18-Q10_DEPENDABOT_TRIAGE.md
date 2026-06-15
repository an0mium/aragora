# Q10 — Dependabot PR triage receipt

- Session: `droid-20260518-193726-244edc3f`
- Lane: `Q10-dependabot-triage-2026-05-18`
- PRs: 11 (none merged, none closed, none labeled)
- Started: 2026-05-18T19:37:42Z
- Completed: 2026-05-18T19:40:50Z
- Outcome: shipped (triage receipt + 11 PR comments)

## Buckets (per AGENT_OPERATING_CONTRACT)

- **A — merge-eligible (patch-level only):** comment posted noting the PR
  is eligible for operator-approved merge after CI green. **No merge,
  no label.**
- **B — operator-approval-required (major or minor bumps):** comment
  posted explaining only patch-level bumps are always-allowed; minor
  variants spelled out where relevant; mypy 1.x → 2.x flagged
  strongly as the lone strict major bump.
- **C — superseded / close-eligible:** none identified. All 11 PRs are
  0d old, target distinct deps, and none could be proven superseded
  with the existing search surfaces (`pyproject.toml`,
  `aragora/live/package.json`, `sdk/typescript/package.json` all
  pin the older constraints that the PR proposes to bump).

## Verified dep target paths

```text
pyproject.toml
  build>=1.2,<2.0
  fastapi>=0.135.3,<1.0
  uvicorn>=0.44.0,<1.0
  aiokafka>=0.10,<1.0
  mypy>=1.20.2,<2.0

aragora/live/package.json
  @supabase/supabase-js: ^2.105.1
  dompurify:             ^3.4.2
  @next/bundle-analyzer: ^16.2.4
  @playwright/test:      ^1.59.1
  @tailwindcss/postcss:  ^4.2.4

sdk/typescript/package.json
  @types/node:                      ^25.2.3
  @typescript-eslint/eslint-plugin: ^8.56.0
  @typescript-eslint/parser:        ^8.56.0
  @vitest/coverage-v8:              ^4.0.18
  eslint:                           ^10.0.0
```

All 11 PRs target a dep that currently exists at the older version in
the repo — none of the PRs target a removed dep, so bucket C is empty.

## Per-PR classification

| PR | Title (bump) | Category | Bucket | Comment URL |
|----|---|---|---|---|
| #7295 | `@next/bundle-analyzer` 16.2.4 → 16.2.6 (in `/aragora/live`) | patch | **A** | https://github.com/synaptent/aragora/pull/7295#issuecomment-4481351899 |
| #7296 | `@tailwindcss/postcss` 4.2.4 → 4.3.0 (in `/aragora/live`) | minor (within 4.x) | **B** | https://github.com/synaptent/aragora/pull/7296#issuecomment-4481355702 |
| #7297 | `mypy` `<2.0,>=1.20.2` → `>=2.1.0,<3.0` | **MAJOR** (1.x → 2.x) | **B (flagged strongly)** | https://github.com/synaptent/aragora/pull/7297#issuecomment-4481355544 |
| #7298 | `build` floor `>=1.2` → `>=1.5.0` (ceiling unchanged `<2.0`) | minor floor (within 1.x) | **B** | https://github.com/synaptent/aragora/pull/7298#issuecomment-4481355800 |
| #7299 | `dompurify` 3.4.2 → 3.4.4 (in `/aragora/live`) | patch | **A** | https://github.com/synaptent/aragora/pull/7299#issuecomment-4481351979 |
| #7300 | `fastapi` floor `>=0.135.3` → `>=0.136.1` (ceiling `<1.0`) | minor floor (0.x) | **B** | https://github.com/synaptent/aragora/pull/7300#issuecomment-4481355983 |
| #7301 | `aiokafka` floor `>=0.10` → `>=0.14.0` (ceiling `<1.0`) | minor floor (0.x) | **B** | https://github.com/synaptent/aragora/pull/7301#issuecomment-4481356159 |
| #7302 | `uvicorn` floor `>=0.44.0` → `>=0.47.0` (ceiling `<1.0`) | minor floor (0.x) | **B** | https://github.com/synaptent/aragora/pull/7302#issuecomment-4481356371 |
| #7303 | `@supabase/supabase-js` 2.105.1 → 2.105.4 (in `/aragora/live`) | patch | **A** | https://github.com/synaptent/aragora/pull/7303#issuecomment-4481352068 |
| #7304 | group bump `sdk-deps` in `/sdk/typescript` (7 updates) | group (mixed patch+minor) | **B** | https://github.com/synaptent/aragora/pull/7304#issuecomment-4481356540 |
| #7305 | `@playwright/test` 1.59.1 → 1.60.0 (in `/aragora/live`) | minor (within 1.x) | **B** | https://github.com/synaptent/aragora/pull/7305#issuecomment-4481356755 |

### Bucket summary

- **A (merge-eligible, patch-only):** 3 PRs — #7295, #7299, #7303.
- **B (operator-approval-required):** 8 PRs — #7296, #7297, #7298,
  #7300, #7301, #7302, #7304, #7305. Within B, #7297 is the only
  strict major bump (`mypy` 1.x → 2.x); the rest are minor / minor-floor
  / mixed-group bumps that fall outside the "patch-level only" always-allowed
  band of AGENT_OPERATING_CONTRACT.
- **C (superseded / close-eligible):** 0 PRs.

## Rationale per PR

- **#7295** — `16.2.4 → 16.2.6` is a strict patch increment of an existing
  `^16.2.4` constraint; always-allowed per contract.
- **#7296** — `4.2.4 → 4.3.0` raises the minor digit; not always-allowed.
- **#7297** — `mypy 1.x → 2.x` crosses a major-version boundary. Strict
  approval-required per contract. Strongly flagged: mypy 2.0 carries
  type-checking semantic shifts that will cascade across the codebase.
- **#7298** — `build` floor raised from `>=1.2` to `>=1.5.0`; minor floor
  bump, beyond patch-level.
- **#7299** — `3.4.2 → 3.4.4` is a strict patch increment; always-allowed.
- **#7300** — `fastapi` floor raised by a minor digit in 0.x range, where
  semver permits breaking changes on any minor — approval-required.
- **#7301** — `aiokafka` floor jumps four minor digits (0.10 → 0.14) in
  0.x; approval-required.
- **#7302** — `uvicorn` floor jumps three minor digits (0.44 → 0.47) in
  0.x; approval-required.
- **#7303** — `2.105.1 → 2.105.4` is a strict patch increment;
  always-allowed.
- **#7304** — group bump contains a mix of patch and minor bumps
  (eslint 10.3 → 10.4 minor and @types/node 25.6 → 25.8 minor alongside
  patch-only @typescript-eslint/*, vitest, @vitest/coverage-v8). Mixed
  group is not patch-only; approval-required.
- **#7305** — `@playwright/test 1.59.1 → 1.60.0` is a minor bump; not
  always-allowed.

## Mutations performed

- 11 PR comments posted via `gh pr comment <n> --body "..."` (URLs above).
- Zero merges.
- Zero closes.
- Zero label additions or removals.
- Zero ready-for-review flips.
- Zero re-runs of CI.

## R/D compliance

- **R5** — lane claimed before mutation (`scripts/claim_active_agent_lane.py`).
- **R11** — state probed via `gh pr view` (read-only).
- **R19** — no `--amend` of pushed commits (docs-only commit, fresh).
- **R20** — preflight mypy not applicable (no Python source edits).
- **R21** — dependabot PRs treated as operator queue: comment + classify
  only; no merges, no closes, no labels.
- **R25** — no worktree deletion in this lane.

## Lane

`Q10-dependabot-triage-2026-05-18` will be released `status=completed`
at Phase 4.
