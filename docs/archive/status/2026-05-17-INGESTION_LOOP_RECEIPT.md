# Ingestion-loop receipt — operator-decisions → `gh` applier

**Generated:** 2026-05-17T17:20:01Z
**Branch:** `worktree-operator-decisions-ingestion-20260517`
**Base:** `main` (independent of the #7273 → #7274 → #7277 → #7278 stack)
**Surface:** `scripts/apply_operator_decisions.py`

## The loop, now closed

```
            ┌────────────────────────────────────────────────────────────┐
            │           OPERATOR SETTLEMENT-PACKET LOOP                  │
            └────────────────────────────────────────────────────────────┘

  ┌─────────────────────────┐
  │  Settlement packet      │   (per-PR head pin + tier classification +
  │  receipt JSON, signed   │    recommended action — already exists in
  │  with receipt_sha256    │    docs/receipts/ from the PR #7263 family)
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  /review-queue/packets/ │   (PR #7277 + #7278: file picker → SHA verify
  │  [receiptId] UI         │    → keyboard sign-off → per-decision timing)
  └────────────┬────────────┘
               │
               ▼  Download (signed: payload_sha256 binds the decisions)
  ┌─────────────────────────┐
  │ operator-decisions      │   (schema: aragora-operator-decisions/1.0;
  │ -<utc>.json             │    each entry carries decision, comment,
  │                         │    head_sha, first_focused_at_utc,
  │                         │    decided_at_utc, decision_seconds)
  └────────────┬────────────┘
               │
               ▼  ★ THIS PR — the previously-missing edge
  ┌─────────────────────────┐
  │  scripts/apply_operator │   (verifies payload_sha256 + receipt file
  │  _decisions.py          │    SHA on --apply → HEAD-drift check →
  │                         │    gh pr review/close with binding footer)
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  GitHub PR state moves  │   approve / request-changes / close,
  │  (queue clears)         │   every action audit-trailed to receipt
  └─────────────────────────┘
```

Before this PR, the right-hand half (the download → `gh`) was missing —
the operator could sign off in the UI but nothing applied it. Now the
pipeline is end-to-end actionable in a single shell command.

## CLI surface

```
usage: apply_operator_decisions.py [-h] [--apply]
                                   [--receipt-path RECEIPT_PATH] [--dry-run]
                                   [--json] [--only-pr N]
                                   [--skip-hold-decisions]
                                   decisions_path

Apply a downloaded aragora-operator-decisions/1.0 JSON to GitHub via `gh`.
Defaults to --dry-run.

positional arguments:
  decisions_path        Path to a downloaded operator-decisions-*.json file.

options:
  --apply               Mutate GitHub state. Without this flag, nothing is sent.
  --receipt-path        Original settlement receipt JSON. Required with
                        --apply so the CLI independently verifies
                        receipt_sha256 before mutating GitHub.
  --dry-run             Explicit dry-run (this is the default behaviour when
                        --apply is omitted).
  --json                Emit per-entry results as JSON to stdout.
  --only-pr N           Apply only to the listed PR numbers (repeatable).
  --skip-hold-decisions Always honoured; the hold list is hard-coded and held
                        PRs are skipped regardless of this flag.

exit codes:
  0   all entries either succeeded or were intentionally skipped
  1   any apply step failed (per-entry detail in stderr)
  2   payload signature mismatch / missing file / missing gh / invalid JSON
      (refuse to apply anything)
```

### Decision mapping

| decision | gh action |
|---|---|
| `approve_tier`      | `gh pr review N --repo <receipt_repo> --approve --body <body>` |
| `approve_downgrade` | `gh pr review N --repo <receipt_repo> --approve --body "DOWNGRADED: <body>"` |
| `request_changes`   | `gh pr review N --repo <receipt_repo> --request-changes --body <body>` |
| `reject`            | `gh pr close  N --repo <receipt_repo> --comment <body>` |
| `hold_operator`     | no-op, print `SKIP` |
| `null`              | no-op, print `SKIP (no decision recorded)` |

Every applied body ends with the binding footer so the audit trail
is recoverable from any PR thread:

```
---
Applied from operator-decisions <payload_sha256[:10]> bound to packet <receipt_sha256[:10]>
```

### Held PRs are hard-skipped

The script carries a hard-coded `HELD_PR_NUMBERS = frozenset({4990,
7173, 7215, 7240, 7243, 7245, 7249, 7252})`. These are skipped before
the decision is even inspected, regardless of `--apply`. The operator
may legitimately have recorded a hypothetical decision for one of
these PRs during a packet review pass — the CLI refuses to advance
any of them by even one byte of state change.

## Worked example

Synthetic receipt (decisions for three hypothetical PRs):

```json
{
  "schema_version": "aragora-operator-decisions/1.0",
  "receipt_repo": "synaptent/aragora",
  "receipt_sha256": "abcdef1234567890000000000000000000000000000000000000000000000000",
  "decisions": [
    { "pr_number": 7280, "decision": "approve_tier",
      "comment": "additive only, tests green" },
    { "pr_number": 7281, "decision": "request_changes",
      "comment": "missing test for the drift path" },
    { "pr_number": 7282, "decision": "approve_downgrade",
      "comment": "land at tier 1; tier 2 claim unsupported" }
  ],
  "payload_sha256": "91274c092a57e30daf8a705f9c657070788c6795ce0042d471e5dd6f952f3d48"
}
```

### Human dry-run output (default)

```
$ python3 scripts/apply_operator_decisions.py /tmp/example_operator_decisions.json
WOULD APPLY # 7280  approve_tier          — dry-run
WOULD APPLY # 7281  request_changes       — dry-run
WOULD APPLY # 7282  approve_downgrade     — dry-run

DRY RUN — no PRs were touched. Re-run with --apply to commit.
```

### JSON dry-run output (with `--only-pr 7280`)

```
$ python3 scripts/apply_operator_decisions.py /tmp/example_operator_decisions.json --json --only-pr 7280
{
  "applied": false,
  "payload_sha256": "91274c092a57e30daf8a705f9c657070788c6795ce0042d471e5dd6f952f3d48",
  "receipt_sha256": "abcdef1234567890000000000000000000000000000000000000000000000000",
  "results": [
    {
      "decision": "approve_tier",
      "gh_command": ["gh", "pr", "review", "7280", "--approve", "--body",
                     "--repo", "synaptent/aragora", "--approve", "--body",
                     "additive only, tests green\n\n---\nApplied from operator-decisions 91274c092a bound to packet abcdef1234"],
      "pr_number": 7280,
      "reason": "dry-run",
      "status": "would-apply"
    },
    {
      "decision": "request_changes",
      "gh_command": null,
      "pr_number": 7281,
      "reason": "not in --only-pr filter",
      "status": "skipped"
    },
    ...
  ]
}
```

### What an `--apply` run would do

For PR `7280` the `gh` invocation captured above would actually run.
First a `gh pr view 7280 --repo synaptent/aragora --json headRefOid`
HEAD-drift check; on match,
`gh pr review 7280 --repo synaptent/aragora --approve --body "additive only, tests green\n\n---\n
Applied from operator-decisions 91274c092a bound to packet abcdef1234"`
is executed. On HEAD drift the script prints `DRIFT #7280` and moves on
without applying — drift is expected when the author pushed a new
commit between settlement and apply.

`--apply` also requires `--receipt-path <original-receipt.json>`,
recomputes that file's SHA-256, and derives the GitHub repo from the
receipt `pr_url` before any GitHub mutation. The downloaded
operator-decisions file can prove its own payload hash, but live mutation
requires the original receipt file to prove both packet binding and repo
binding locally.

(The receipt does NOT actually apply against real PRs — that's the
operator's call.)

## Files added

- `scripts/apply_operator_decisions.py` — pure-stdlib CLI (~325 LOC)
- `tests/scripts/test_apply_operator_decisions.py` — 43 fixture-driven tests
- `docs/status/INGESTION_LOOP_RECEIPT_20260517T172001Z.md` — this file

## Validation

```
$ python3 -m pytest tests/scripts/test_apply_operator_decisions.py -q
...........................................                              [100%]
43 passed in 0.98s
$ ruff check scripts/apply_operator_decisions.py tests/scripts/test_apply_operator_decisions.py
All checks passed!
$ ruff format --check scripts/apply_operator_decisions.py tests/scripts/test_apply_operator_decisions.py
2 files already formatted
$ mypy scripts/apply_operator_decisions.py
Success: no issues found in 1 source file
$ bash scripts/automation_pr_preflight.sh origin/main HEAD
preflight: ok
```

## Test coverage

| Test | Behaviour verified |
|---|---|
| sig mismatch → exit 2, no gh calls | Refuses to apply on broken payload binding |
| missing file → exit 2 | Clean error for bad path |
| invalid JSON → exit 2 | Clean error for bad payload |
| no gh on PATH → exit 2 | Bail before processing |
| default dry-run does not call gh | `--apply` opt-in is enforced |
| apply approve_tier → review --approve | Happy path 1 |
| apply approve_downgrade prepends marker | Body convention |
| apply request_changes → review --request-changes | Happy path 2 |
| apply reject → pr close --comment | Happy path 3 |
| apply hold_operator → no gh call | Operator-only |
| null decision skipped | Defensive |
| unknown decision fails closed | No mutation on unsupported decision IDs |
| malformed/type-invalid row fails closed | Later bad rows block earlier mutations |
| `--apply` requires receipt path | Original receipt file must be present before mutation |
| receipt SHA mismatch fails closed | Downloaded payload cannot self-authorize mutation |
| receipt repo mismatch fails closed | `receipt_repo` must match the original receipt `pr_url` repo |
| HEAD drift skips entry, exit 0 | Drift safety |
| `--only-pr 200` touches only #200 | Filter |
| held PR hard-skip on `--apply` | Hold-list enforcement |
| held PR skip in dry-run too | Hold-list enforcement |
| `--json` output shape stable | Machine-readable contract |
| footer carries both SHA prefixes | Audit trail integrity |
| footer present even with empty comment | Audit trail always emitted |
| gh failure returns 1 | Per-entry failure surfaces |

## Holds respected

- No PR mutation, no labels, draft PR only.
- Zero AI-key consumption.
- No `automation.toml` edit, no launchd install.
- No held-PR advancement: PRs `#7173`, `#7215`, `#7240`, `#7243`,
  `#7245`, `#7249`, `#7252`, `#4990` are in `HELD_PR_NUMBERS` and
  hard-skipped by the CLI regardless of `--apply`.
- `#7209 lane` and `BC-12 soak` are non-PR-number holds; the CLI
  only processes PR numbers and does not touch any of those surfaces.

## Reproduction

```bash
git checkout worktree-operator-decisions-ingestion-20260517
python3 -m pytest tests/scripts/test_apply_operator_decisions.py -q
# Then, optionally, against a real downloaded JSON:
python3 scripts/apply_operator_decisions.py \
    ~/Downloads/operator-decisions-2026-05-17T*.json   # dry-run
python3 scripts/apply_operator_decisions.py \
    ~/Downloads/operator-decisions-2026-05-17T*.json \
    --receipt-path docs/receipts/<packet-receipt>.json \
    --apply
```

## Receipt self-binding

SHA-256 of this file is computed after content is finalized via:

```
shasum -a 256 docs/status/INGESTION_LOOP_RECEIPT_20260517T172001Z.md
```

The PR description and final session response print the resulting hex.
