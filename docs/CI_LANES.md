# CI Lanes: R&D and Integrator Model

## Problem

Multiple AI agents working in parallel create many PRs simultaneously. Running
the full 58-workflow CI suite on every draft PR commit saturates the GitHub
Actions queue, creating a bottleneck where good code waits in line behind
speculative iterations.

## Solution: Two Lanes

### R&D Lane (Draft PRs)

- **Who:** Many agents iterating on features, fixes, explorations
- **CI cost:** 6 lightweight checks (~2-3 min total)
- **Workflow:** Create draft PR, iterate freely, mark ready when confident

Checks that run on draft PRs:
| Workflow | Purpose | ~Time |
|----------|---------|-------|
| `lint.yml` | Code style, ruff, formatting | 1 min |
| `sdk-parity.yml` | Python/TS SDK alignment | 2 min |
| `sdk-test.yml` | SDK unit tests | 2 min |
| `pr-debate.yml` | AI review | 1 min |
| `required-check-priority.yml` | Queue management | <1 min |
| `autopilot-worktree-e2e.yml` | Worktree validation | 2 min |

### Integrator Lane (Ready PRs)

- **Who:** PRs marked "Ready for Review" (one at a time ideally)
- **CI cost:** Full 33-workflow suite (~30-60 min)
- **Workflow:** Full test matrix, security scanning, E2E, integration tests

27 additional workflows run only on non-draft PRs:
- Unit tests (full matrix), E2E tests, integration tests
- Security scanning (CodeQL, Bandit, dependency audit)
- Smoke tests, core suites, migration tests
- Benchmarks, coverage, Docker builds
- OpenAPI validation, SDK generation, load tests

## Agent Workflow

```
1. Agent creates worktree + branch
2. Agent creates DRAFT PR           → 6 fast checks only
3. Agent iterates (push commits)    → 6 fast checks only
4. Agent marks PR "Ready"           → Full 33 workflows run
5. Merge when green                 → Main branch protected
```

## Key Principle

**Parallel creativity, serial integration.** Many agents can explore cheaply in
draft PRs. Only one PR at a time goes through the full integrator pipeline,
preventing queue saturation.

## Configuration

Draft gating uses the GitHub Actions job-level condition:
```yaml
if: github.event_name != 'pull_request' || !github.event.pull_request.draft
```

This ensures workflows still run on:
- Push to main (post-merge validation)
- Scheduled runs (nightly/weekly)
- Manual dispatch
- Non-draft pull requests
