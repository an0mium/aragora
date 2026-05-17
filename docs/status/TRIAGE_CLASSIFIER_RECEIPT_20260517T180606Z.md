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

**2026-05-17T18:50:41Z audit patch:** Bucket A is now exact-head gated in
this classifier, not deferred to Stage 2. Otherwise-eligible candidates
must have a current `aragora review-queue merge-packet --pr <N> --json`
result with `admin_squash_allowed=true`, `not_ready=[]`,
`unresolved_dissent=false`, a matching head SHA, and Tier 3/4 settlement
or preapproval if applicable. Without that proof, the PR remains Bucket C.

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

## Worked example — live run against current queue (tightened policy)

Output of `python3 scripts/triage_open_prs.py` at 2026-05-17T18:11Z,
after the operator tightened the policy doc to require `is_draft == False`
+ `mergeStateStatus == CLEAN` for Bucket A:

```
BUCKET A — recommend AUTO-MERGE
  (none)

BUCKET B — recommend AUTO-CLOSE
  (none)

BUCKET C — needs operator y/n
  #7173 — STAY HELD — held (#7173 is on the policy hold list)
  #7215 — STAY HELD — held (#7215 is on the policy hold list)
  #7245 — STAY HELD — held (#7245 is on the policy hold list)
  #7251 — READY? — draft (policy requires non-draft for Bucket A; 17/67 CI green)
  #7252 — STAY HELD — held (#7252 is on the policy hold list)
  #7259 — READY? — draft (policy requires non-draft for Bucket A; 17/66 CI green)
  #7262 — READY? — draft (policy requires non-draft for Bucket A; 16/68 CI green)
  #7263 — READY? — draft (policy requires non-draft for Bucket A; 14/34 CI green)
  #7268 — DECIDE — large diff (1542 LOC > 1500)
  #7276 — READY? — draft (policy requires non-draft for Bucket A; 16/68 CI green)
  #7278 — READY? — draft (policy requires non-draft for Bucket A; 5/8 CI green)
  #7279 — READY? — draft (policy requires non-draft for Bucket A; 17/67 CI green)
  #7283 — READY? — draft (policy requires non-draft for Bucket A; 14/34 CI green)
  #7284 — READY? — draft (policy requires non-draft for Bucket A; 17/66 CI green)

BUCKET D — strategic check-in
  (none)

summary: A: 0  B: 0  C: 14  D: 0    total: 14
```

### Why Bucket A is empty: the policy tightening

Mid-session, the operator tightened Bucket A criteria in
`docs/governance/OPERATOR_DELEGATION_POLICY.md` (PR #7283) to require:

- `mergeable: MERGEABLE` (existing)
- **`PR is not draft`** (new — drafts always go to C with `READY?`)
- **`mergeStateStatus == CLEAN`** (new — `BLOCKED` no longer qualifies)
- `aragora review-queue merge-packet` reports `admin_squash_allowed=true`,
  `not_ready=[]`, `unresolved_dissent=false` at the **exact** current
  head SHA
- Tier 3 / Tier 4 PRs need explicit risk-settlement at exact head SHA
- Existing: green CI, additive, tests, ≤1500 LOC, trusted author,
  no held PRs, no protected files

The current open queue is 14 PRs deep with ZERO not-draft + CLEAN
items — so the classifier honestly reports Bucket A is empty. The
operator's path forward is one of:

1. Mark some drafts ready (the `READY?` recommendation calls this out),
   which moves them to A on the next pass
2. Run Stage 2 (#7281, `auto_merge_bucket_a.py`) once it ships — it
   will do the deep merge-packet + tier checks before merging

### The classifier corrected my manual triage earlier this session

Earlier I produced a triage manually and called **#7251 "held"** from
memory. The classifier read the canonical hold list
(`HELD_PR_NUMBERS = {4990, 7173, 7215, 7240, 7243, 7245, 7249, 7252}`)
and correctly put #7251 in C with reason `draft` — not `held`. This is
exactly the operator-delegation premise: mechanical classification
beats manual memory.

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
10. **C** if `is_draft` (reason: `draft`, action: `READY?`)
11. **C** if `mergeable != MERGEABLE`
12. **C** if `mergeStateStatus != CLEAN`
13. **C** if there are code files but no test files
14. **C** if `reviewDecision == CHANGES_REQUESTED`
15. **C** if the exact-head merge packet does not authorize admin squash
    or reports `not_ready`, unresolved dissent, head drift, or missing
    Tier 3/4 settlement/preapproval
16. **A** otherwise

Bucket D is reserved for future enhancement — strategic mismatch
with canonical direction is not auto-classifiable from `gh` metadata
alone.

## Tests (46 new, all green)

```
$ python3 -m pytest tests/scripts/test_triage_open_prs.py -q
..............................................                           [100%]
46 passed
```

| Group | Tests | Coverage |
|---|---|---|
| TestBucketA | 9 | Clean additive ready-to-merge → A; draft → C `READY?`; BLOCKED → C; missing merge-packet → C; not_ready → C; admin false → C; head mismatch → C; Tier 3 without settlement → C; Tier 3 with settlement → A |
| TestBucketCTripwires | 14 | Held PR; protected file (CLAUDE.md, automation.toml, aragora/__init__.py); large diff; CI red recent; CI pending; CI cancelled/non-green; non-trusted author; not mergeable (CONFLICTING); merge state DIRTY; merge state BEHIND; code without tests; pure-docs-doesnt-trip-rule (negative); review CHANGES_REQUESTED |
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
