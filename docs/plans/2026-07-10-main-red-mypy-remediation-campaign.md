# Main-Red Mypy Remediation Campaign

**Date:** 2026-07-10
**Status:** draft coordination standard
**Pinned main:** `3fe2e5cf561fc094221008d064040ac84625bd4e`
**Coordination issue:** [#9099](https://github.com/synaptent/aragora/issues/9099)
**Environment blocker:** [#9096](https://github.com/synaptent/aragora/issues/9096)

## Scope

This plan coordinates parallel remediation of the mypy debt exposed by the
truthful main-red investigation. It defines one pinned inventory, a provisional
measurement environment, and a claim protocol for disjoint repair slices.

This document does not authorize edits to CI workflows, `.mypy-baseline`,
branch protection, the merge-executor halt, or any existing repair branch. It
does not settle or merge a PR. The active halt remains in force until a human
reviews a truthful pristine-main green run and separately re-arms the repository.

## Pinned Inventory

A fresh worktree at the pinned main SHA was measured with isolated mypy caches.
The PyJWT-present run used for the partition ledger reports:

- 2,634 errors in 649 files;
- 85 errors in five source files already claimed by #9083 and #9088;
- 2,549 errors in 644 unclaimed files;
- 72 top-level partitions.

The complete per-file snapshot is
[`main-red-mypy-partition.json`](main-red-mypy-partition.json). Error counts in
that file sum to 2,634, and every source file changed by #9083 or #9088 is
marked `CLAIMED`. Their test files are retained in each claim's
`support_files` list because tests are not targets of the full `aragora/` mypy
command.

| Top-level partition | Errors | Files | Claimed errors |
|---|---:|---:|---:|
| `aragora/server` | 1,299 | 268 | 83 |
| `aragora/nomic` | 320 | 23 | 0 |
| `aragora/debate` | 216 | 54 | 0 |
| `aragora/connectors` | 131 | 48 | 0 |
| `aragora/knowledge` | 55 | 25 | 0 |
| `aragora/cli` | 52 | 13 | 2 |
| `aragora/ml` | 34 | 6 | 0 |
| `aragora/services` | 33 | 6 | 0 |
| `aragora/agents` | 31 | 17 | 0 |
| `aragora/storage` | 26 | 12 | 0 |

## Provisional Environment

The environment below reproduces the PyJWT-present GitHub result that exposed
the `aragora/connectors/devices/push.py:462` error. It is a provisional
campaign measurement profile, not the final required-check dependency
contract. Issue #9096 owns the work to make that contract complete and
hermetic.

The same pinned worktree produced different identities when only PyJWT
availability changed:

| Environment | Errors | Files |
|---|---:|---:|
| Without PyJWT | 2,636 | 649 |
| With `PyJWT==2.13.0` | 2,634 | 649 |

Installing PyJWT removed three `no-redef` findings from
`aragora/connectors/chat/jwt_verify.py` and added one `arg-type` finding at
`aragora/connectors/devices/push.py:462`. A net reduction of two is therefore
not evidence of a repair. The machine ledger records both normalized error-set
hashes and the four changed identities.

Reproduce the ledger profile from a fresh worktree:

```bash
PYTHON_312="$(uv python find 3.12.12)"
VENV="$(mktemp -d)/venv"
CACHE="$(mktemp -d)"

uv venv --python "$PYTHON_312" "$VENV"
uv pip install --python "$VENV/bin/python" -e . \
  mypy==2.2.0 \
  mypy-baseline==0.7.4 \
  PyJWT==2.13.0 \
  types-jsonschema==4.26.0.20260518 \
  types-python-dateutil==2.9.0.20260518 \
  types-PyYAML==6.0.12.20260518 \
  types-redis==4.6.0.20241004 \
  types-requests==2.33.0.20260518 \
  types-setuptools==83.0.0.20260706

MYPY_CACHE_DIR="$CACHE" "$VENV/bin/python" -m mypy \
  aragora/ \
  --ignore-missing-imports \
  --show-error-codes \
  --no-color-output \
  --no-pretty \
  --no-error-summary
```

The expected command exit is nonzero while debt remains. Capture its complete
output; do not infer success from a reporting pipeline or a count-only wrapper.
Do not regenerate `.mypy-baseline` from this profile while #9096 is open.

## Existing Claims

The following source and support files are unavailable to new slices while
their PRs remain open:

| PR | Exact head | Source errors | Source files | Support files |
|---|---|---:|---:|---:|
| #9083 | `09cd7dfc1e0575c13732ea6b17fd9dfa038e0878` | 20 | 4 | 3 |
| #9088 | `8104566b718b15b9663feb2816eb0096c88a19a4` | 65 | 1 | 1 |

The JSON ledger is the pinned-base inventory. The comments on #9099 are the
live claim register. A new worker must check both immediately before branching.

## Claim Protocol

Comment on #9099 before opening a remediation PR. A complete claim names an
owner, the pinned base SHA, exact files, expected error count, and an RFC 3339
expiry no more than 24 hours away.

```text
CLAIM
owner: <GitHub login or agent session>
base_sha: 3fe2e5cf561fc094221008d064040ac84625bd4e
files:
  - <exact path>
expected_error_count: <30-80 preferred>
expires_at: <RFC3339 timestamp>
```

Apply these rules:

1. One claim maps to one PR and one disjoint source-file set.
2. Target 30-80 errors per slice. A smaller coherent slice is valid when a
   module boundary makes it safer.
3. Never touch a file in an active issue claim or open repair PR.
4. Re-run the pinned profile with a new isolated cache. Record exact removed
   and added error identities, not only the net count.
5. Link the draft PR and exact head from the claim comment. Refresh the claim
   before expiry if work remains active.
6. Release abandoned claims explicitly. Reclaim an expired file only after a
   new overlap check.
7. Do not include `.mypy-baseline`, workflow, branch-protection, or halt edits
   in a debt slice.
8. Do not treat a smaller error count as merge or settlement authority.

## Slice Verification

For both the pinned base and the candidate head, run the same command in the
same package environment with separate empty caches. Sort lines containing
`: error:` and compare full identities.

A valid slice must show:

- every removed identity belongs to a claimed file;
- no added identity in any file;
- no edit to a file owned by another claim;
- focused tests for each behavior-changing repair;
- the candidate's exact head SHA in the PR evidence;
- no reliance on an unsynced baseline or ambient package state.

If `origin/main` advances, do not silently rewrite the pinned inventory inside
an active slice. Rebase or restack only under the normal ownership rules, then
publish a new ledger generation tied to the new base.

## Exit Criteria

This campaign is complete only when:

- #9096 defines a deterministic required-typecheck dependency closure;
- every ledger entry is repaired or explicitly dispositioned;
- #9084 makes cancellation fail closed;
- #9085 provides a truthful, reviewed baseline-admission path;
- #9086 is resolved independently as SDK parity debt;
- a pristine current-main run passes all truthful required paths; and
- a human separately removes the main-red halt after reviewing that evidence.

Until then, the best next action is the highest-value unclaimed repair slice or
the smallest artifact that removes a concrete blocker. A blocked merge,
workflow edit, or credential path does not block the campaign as a whole.
