# Main-Head Health Runbook

Use this runbook when `origin/main` needs protected-context health evidence but
workflow dispatch is not authorized. The procedure establishes local ground
truth from a pristine exact-head checkout, so settlement and repair decisions do
not depend on noisy shared worktrees or missing GitHub check-runs.

This runbook incorporates the earlier draft on
`origin/codex/main-head-required-checks-runbook-20260707`
(`docs/runbooks/main-head-required-checks.md`) and narrows it to the
no-dispatch verification path.

## No-Dispatch Procedure

Start from the shared root only to fetch and identify the live head:

```bash
git fetch origin --prune
HEAD_SHA=$(git rev-parse origin/main)
git status --short --branch --untracked-files=all
```

Create a pristine detached worktree for the exact head under `/private/tmp`:

```bash
WT=/private/tmp/aragora-main-health-$(date -u +%Y%m%dT%H%M%SZ)
git worktree add --detach "$WT" "$HEAD_SHA"
cd "$WT"
```

Do not diagnose current-main health from the shared root or from an existing
session worktree. Those checkouts can contain user work, branch-specific
dependencies, stale generated files, or partial environment changes that make a
main-health result ambiguous. A detached exact-head worktree keeps the evidence
about `origin/main`, not about the local operator session.

Run the aggregate local proxy for protected checks:

```bash
PATH="$HOME/.pyenv/versions/3.11.11/bin:$PATH" make ci-required
```

Capture the full log outside the worktree, for example:

```bash
LOG=/tmp/aragora-main-health-ci-required-$(date -u +%Y%m%dT%H%M%SZ).log
PATH="$HOME/.pyenv/versions/3.11.11/bin:$PATH" make ci-required >"$LOG" 2>&1
```

## Local Context Map

Read live branch protection before using this mapping:

```bash
gh api repos/synaptent/aragora/branches/main/protection/required_status_checks \
  --jq '{strict, contexts, checks}'
```

`make ci-required` is a local proxy for most, but not all, protected contexts.
It is fail-fast, so later rows are `not reached` when an earlier command fails.

| Required context | Local command | Notes |
| --- | --- | --- |
| `lint` | `ruff check aragora/ tests/ scripts/` | Runs first in `make ci-required`. |
| `typecheck` | `mypy aragora/ --ignore-missing-imports` | Runs second and currently fails on the 2026-07-08 snapshot below. |
| `sdk-parity` | `python scripts/check_version_alignment.py`; `python scripts/check_sdk_parity.py --strict --baseline scripts/baselines/check_sdk_parity.json --budget scripts/baselines/check_sdk_parity_budget.json`; `python scripts/check_sdk_namespace_parity.py --strict --baseline scripts/baselines/check_sdk_namespace_parity.json`; `python scripts/check_cross_sdk_parity.py --strict --baseline scripts/baselines/cross_sdk_parity.json` | Not reached if lint or typecheck fails. |
| `Generate & Validate` | `python scripts/generate_openapi.py --output /tmp/openapi_ci_required.json --format json --quiet`; `python scripts/add_openapi_operation_ids.py --spec /tmp/openapi_ci_required.json`; `python scripts/add_openapi_param_descriptions.py --spec /tmp/openapi_ci_required.json`; `python scripts/add_openapi_descriptions.py --spec /tmp/openapi_ci_required.json`; `python scripts/verify_sdk_contracts.py --strict --baseline scripts/baselines/verify_sdk_contracts.json --extra-spec /tmp/openapi_ci_required.json`; `python scripts/validate_openapi_routes.py --spec /tmp/openapi_ci_required.json --fail-on-missing --baseline scripts/baselines/validate_openapi_routes.json` | Not reached if any earlier command fails. |
| `TypeScript SDK Type Check` | Not covered by `make ci-required` at this head. Use the dedicated SDK workflow or an explicitly authorized local command such as `cd sdk/typescript && npm ci && npx tsc --noEmit` when that context is the target. | This gap should not be interpreted as a pass. |
| `aragora-merge-quorum` | No main-head local proxy. | Treat as PR-only/skipped for `main` unless live tooling proves otherwise. |

## Disposition Rules

- If every local proxy passes and no required GitHub context is red, record the
  exact head, log path, elapsed time, and use the result as main-head legitimacy
  evidence for settlement decisions.
- If any local proxy fails, stop at the first failing command. Do not dispatch
  unrelated workflows to mask the failure. The next cycle should repair the
  named failing surface or prove the failure is an environment/tooling mismatch.
- If a required GitHub context is missing but the local proxy is green, prepare
  an exact-head workflow-dispatch authorization prompt for only that missing
  context. Do not dispatch without authorization.
- If `TypeScript SDK Type Check` is missing, do not infer its state from
  `make ci-required`; run or request the dedicated SDK check path.
- If `aragora-merge-quorum` is skipped on `main`, classify it as main-unsafe or
  PR-only rather than dispatching it.

## Snapshot: 2026-07-08

| Field | Value |
| --- | --- |
| Exact `origin/main` head | `2543ae8b42f6305e3241cfc64e8658a91f1115ea` |
| Commit summary | `docs(m6): canonical consistency sweep across quickstart/verifier/action docs (#9003)` |
| Detached worktree | `/private/tmp/aragora-main-health-20260708T143015Z` |
| Local command | `PATH="$HOME/.pyenv/versions/3.11.11/bin:$PATH" make ci-required` |
| Start / end | `2026-07-08T14:30:36Z` / `2026-07-08T14:31:37Z` |
| Elapsed | 61 seconds |
| Full log | `/tmp/aragora-main-health-ci-required-20260708T143015Z.log` |
| Result | failed at `typecheck` |

Per-check result:

| Local check | Status | Evidence |
| --- | --- | --- |
| Ruff lint | pass | Log lines show `ruff check aragora/ tests/ scripts/` followed by `All checks passed!`. |
| Mypy typecheck | fail | `mypy aragora/ --ignore-missing-imports` reported 2,646 errors in 648 files. |
| Version alignment | not reached | `make ci-required` stopped at mypy. |
| SDK parity trio | not reached | `make ci-required` stopped at mypy. |
| OpenAPI generation and route validation | not reached | `make ci-required` stopped at mypy. |
| TypeScript SDK Type Check | not covered | Not part of `make ci-required` at this head. |
| `aragora-merge-quorum` | main-unsafe / skipped | Live exact-head GitHub checks showed skipped quorum jobs for `main`. |

Minimal failing excerpt:

```text
aragora/server/commands/models.py:75: error: Incompatible types in assignment (expression has type "None", variable has type "list[dict[str, Any] | None]")  [assignment]
aragora/server/commands/models.py:78: error: Incompatible types in assignment (expression has type "None", variable has type "list[dict[str, Any] | None]")  [assignment]
aragora/server/commands/models.py:85: error: Incompatible default for argument "blocks" (default has type "None", argument has type "list[dict[str, Any] | None]")  [assignment]
aragora/pipeline/idea_clusterer.py:226: error: Incompatible types in assignment (expression has type "int | None", target has type "int")  [assignment]
...
Found 2646 errors in 648 files (checked 4248 source files)
make: *** [ci-required] Error 1
```

Disposition: `origin/main` is not locally green under the no-dispatch
`make ci-required` proxy because the typecheck step fails broadly. The next
bounded repair should target the typecheck command or classify why the local
aggregate differs from the protected GitHub `typecheck` context before any
main-health dispatch or settlement decision relies on this snapshot.
