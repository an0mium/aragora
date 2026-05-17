# Triage classifier receipt — Stage 1 of Operator Delegation rollout

**Generated:** 2026-05-17T18:06:06Z
**Branch:** `worktree-triage-classifier-20260517`
**Closes:** [#7280](https://github.com/synaptent/aragora/issues/7280) — Stage 1: `scripts/triage_open_prs.py`
**Policy:** `docs/governance/OPERATOR_DELEGATION_POLICY.md` (from PR #7283; will rebase clean once #7283 lands)
**Rollout:** `docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md` (Stage 1 of 5)

## What shipped

`scripts/triage_open_prs.py` — read-only four-bucket classifier that
reproduces today's Bucket A/B/C/D triage table from live `gh pr list`
data. Pure-stdlib (argparse, dataclasses, datetime, json, shutil,
subprocess, sys, pathlib, typing). No `aragora.*` imports. Zero AI-key
consumption. Never mutates GitHub state.

## CLI surface

```
usage: triage_open_prs.py [-h] [--json] [--bucket {A,B,C,D}]
                          [--include-held] [--limit LIMIT]
                          [--from-json FROM_JSON]

Read-only four-bucket PR triage classifier per
docs/governance/OPERATOR_DELEGATION_POLICY.md.

options:
  --json                Emit JSON instead of human table.
  --bucket {A,B,C,D}    Filter to one bucket only.
  --include-held        Always include held PRs (default: yes).
  --limit LIMIT         Max PRs to fetch from gh (default: 100).
  --from-json FROM_JSON Read PR data from JSON file (for tests / offline).
```

## Worked example — live run against current queue

Output of `python3 scripts/triage_open_prs.py` at 2026-05-17T18:06Z:

```
BUCKET A — recommend AUTO-MERGE
  #7251 — MERGE — green CI (17/67), 353 LOC, 3 files, tests present, author=an0mium
  #7259 — MERGE — green CI (17/66), 234 LOC, 2 files, tests present, author=an0mium
  #7262 — MERGE — green CI (16/68), 373 LOC, 2 files, tests present, author=an0mium
  #7263 — MERGE — green CI (14/34), 778 LOC, 2 files, tests present, author=an0mium
  #7276 — MERGE — green CI (16/68), 290 LOC, 2 files, tests present, author=an0mium
  #7278 — MERGE — green CI (5/8), 792 LOC, 5 files, tests present, author=an0mium
  #7279 — MERGE — green CI (17/67), 1185 LOC, 3 files, tests present, author=an0mium
  #7283 — MERGE — green CI (14/34), 372 LOC, 2 files, tests present, author=an0mium

BUCKET B — recommend AUTO-CLOSE
  (none)

BUCKET C — needs operator y/n
  #7173 — STAY HELD — held (#7173 is on the policy hold list)
  #7215 — STAY HELD — held (#7215 is on the policy hold list)
  #7245 — STAY HELD — held (#7245 is on the policy hold list)
  #7252 — STAY HELD — held (#7252 is on the policy hold list)
  #7268 — DECIDE — large diff (1542 LOC > 1500)

BUCKET D — strategic check-in
  (none)

summary: A: 8  B: 0  C: 5  D: 0    total: 13
```

### Sanity-check vs. earlier manual triage

The same triage was produced manually earlier this session. The
mechanical classifier diverged from the manual triage in one place —
and the classifier was right:

- I called **#7251 "held"** based on memory; the canonical hold list
  in the policy doc is `{4990, 7173, 7215, 7240, 7243, 7245, 7249,
  7252}` — `#7251 is NOT on it`. The classifier correctly put it in A.
- The classifier also caught #7283 (the policy PR opened earlier this
  session) and #7268 (Codex's settlement UI — flagged C for large
  diff at 1542 LOC, just above the 1500 cap).
- `#7261` was open during the earlier manual triage but landed
  between then and now — the classifier reflects live state.

This is exactly the operator-delegation premise: the mechanical
classifier reproduces the manual judgment more accurately, faster,
and without the operator (or a frontier-model agent) having to hold
the hold list in working memory.

## Bucket precedence

The classifier evaluates buckets in this order (most-restrictive
wins; first match returns):

1. **C** if `pr_number ∈ HELD_PR_NUMBERS`
2. **C** if any changed file is in `PROTECTED_PATHS`
3. **C** if `additions + deletions > 1500`
4. **B** if CI red AND `updated_at` ≥ 7 days ago
5. **C** if CI red (recent)
6. **C** if any check is `IN_PROGRESS` / `QUEUED`
7. **B** if draft + `created_at ≥ 60d` + `updated_at ≥ 30d`
8. **B** if a newer open PR has ≥80% file overlap + zero CI failures
9. **C** if author ∉ `TRUSTED_AUTHORS`
10. **C** if `mergeable != MERGEABLE`
11. **C** if `mergeStateStatus ∉ {CLEAN, BLOCKED}`
12. **C** if there are code files but no test files
13. **C** if `reviewDecision == CHANGES_REQUESTED`
14. **A** otherwise

Bucket D is reserved for future enhancement — strategic mismatch
with canonical direction is not auto-classifiable from `gh` metadata
alone.

## Tests (38 new, all green)

```
$ python3 -m pytest tests/scripts/test_triage_open_prs.py -q
......................................                                   [100%]
38 passed in 1.26s
```

| Group | Tests | Coverage |
|---|---|---|
| TestBucketA | 2 | Clean additive (CLEAN); clean additive on draft (BLOCKED) |
| TestBucketCTripwires | 13 | Held PR; protected file (CLAUDE.md, automation.toml, aragora/__init__.py); large diff; CI red recent; CI pending; non-trusted author; not mergeable (CONFLICTING); merge state DIRTY; merge state BEHIND; code without tests; pure-docs-doesnt-trip-rule (negative); review CHANGES_REQUESTED |
| TestBucketB | 7 | CI red 7+ days; stale draft over threshold; stale but recent (negative); ready PR not marked stale (negative); supersede by newer clean PR; no supersede when overlap too low (negative); no supersede when newer has CI failure (negative) |
| TestPrecedence | 3 | Held beats all other tripwires; protected beats large diff; CI-red-7d beats supersede |
| TestCliOutput | 8 | Human output; JSON output; bucket filter; missing --from-json file; invalid --from-json JSON; non-array root; no gh on PATH; deterministic output across runs |
| TestEdgeCases | 4 | Empty PR list; PR with zero files; PR with empty author dict; reason capped at 200 chars |

## Validation

```
$ python3 -m pytest tests/scripts/test_triage_open_prs.py -q
38 passed in 1.26s
$ ruff check scripts/triage_open_prs.py tests/scripts/test_triage_open_prs.py
All checks passed!
$ ruff format --check scripts/triage_open_prs.py tests/scripts/test_triage_open_prs.py
2 files already formatted
$ mypy scripts/triage_open_prs.py
Success: no issues found in 1 source file
$ bash scripts/automation_pr_preflight.sh origin/main HEAD
preflight: ok
```

## How this fits the rollout

| Stage | Status |
|---|---|
| Stage 0 — policy doc + rollout doc | shipped as PR #7283 |
| **Stage 1 — this PR (triage_open_prs.py)** | **shipped (this receipt)** |
| Stage 2 — auto_merge_bucket_a.py | tracked as #7281, depends on this |
| Stage 3 — triage_bucket_c.py | tracked as #7282, depends on this |
| Stage 4 — scheduling (LaunchAgent template) | not yet filed |
| Stage 5 — Bucket-D escalation surface | not yet filed |

The Stage 2 + 3 scripts will consume this classifier's `--json`
output rather than re-implement the classification, so this is the
single source of bucket truth going forward.

## Holds respected

- No PR mutation, no labels, draft only.
- Zero AI-key consumption.
- Held PRs (`#7173, #7215, #7240, #7243, #7245, #7249, #7252,
  #4990`) hard-coded in `HELD_PR_NUMBERS`; the classifier puts every
  one of them in Bucket C with reason "held" — never recommends A or B.
- No `automation.toml` edit, no launchd install.
- No protected-file edits (`CLAUDE.md`, `aragora/__init__.py`, `.env`,
  `.envrc`, `scripts/nomic_loop.py`, `docs/AGENT_OPERATING_CONTRACT.md`,
  `automation.toml`).

## Reproduction

```bash
git checkout worktree-triage-classifier-20260517
python3 -m pytest tests/scripts/test_triage_open_prs.py -q
python3 scripts/triage_open_prs.py            # live human table
python3 scripts/triage_open_prs.py --json     # live JSON
python3 scripts/triage_open_prs.py --bucket C # only operator-y/n items
```

## Receipt self-binding

```
shasum -a 256 docs/status/TRIAGE_CLASSIFIER_RECEIPT_20260517T180606Z.md
```

The PR description and final session response print the resulting hex.
