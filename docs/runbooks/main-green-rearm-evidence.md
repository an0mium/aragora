# Main-Green Re-Arm Evidence

Use this runbook after `.aragora/merge_executor.halt` has stopped merges because
`origin/main` was classified red. The halt marker is a durable stop signal, not
a live health probe: its recorded SHA may be older than current main, and a fix
landing on main is not by itself proof that every required and full-suite path
is green.

This procedure produces an exact-head evidence packet for a human re-arm
decision. Automated agents may collect the evidence, but they must not delete,
replace, or work around the halt marker.

## Preconditions

1. Work from the canonical checkout whose `.aragora` directory is used by the
   merge executor. Do not use a session worktree's local halt path as a proxy.
2. Keep the halt marker in place while collecting evidence.
3. Use `--no-halt-file` for every diagnostic run. A diagnostic must not rewrite
   an existing incident record.
4. Treat a missing, broken, or below-floor test tool as an infrastructure
   failure, not as evidence that the repository is red.
5. Require the required suite and full suite to test the same `origin/main`
   SHA. If main moves between them, discard the pair and rerun both.

Set the paths once from the canonical checkout:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd -- "$REPO_ROOT"
HALT_FILE="$REPO_ROOT/.aragora/merge_executor.halt"
PRISTINE_DIR="$HOME/.aragora/pristine-main"
```

Record the incident marker before doing anything else:

```bash
test -f "$HALT_FILE"
HALT_SHA256="$(shasum -a 256 "$HALT_FILE" | awk '{print $1}')"
cat "$HALT_FILE"
printf 'halt_sha256=%s\n' "$HALT_SHA256"
```

The marker's `details` explain why the halt was armed. They do not establish
the health of a newer main head.

## 1. Pin The Candidate Head

Fetch with an explicit refspec, then record the full SHA:

```bash
git fetch origin +refs/heads/main:refs/remotes/origin/main
CANDIDATE_SHA="$(git rev-parse origin/main)"
git show --no-patch --format='%H%n%aI%n%s' "$CANDIDATE_SHA"
```

All later evidence must name `CANDIDATE_SHA`. Do not report only a branch name
or a short SHA.

## 2. Prove The Toolchain

`make ci-required` resolves `mypy` from `PATH`. The declared development
dependency currently requires `mypy>=2.1.0,<3.0`; a PATH installation such as
1.19.1 is not a valid required-suite runner even if it can execute.

Capture the interpreter and executable identities:

```bash
python3 --version
python3 -c 'import sys, pytest; print(sys.executable); print(pytest.__version__)'
command -v mypy
mypy --version
git show "${CANDIDATE_SHA}:pyproject.toml" | grep -nE '"mypy[<>=]'
```

Classify any of these outcomes as `infra_error` and stop before running the
suites:

- `pytest` cannot be imported by the interpreter that will launch the full
  suite;
- `mypy` is absent from `PATH`, cannot launch, or its version cannot be parsed;
- PATH `mypy` is below the declared lower bound or outside the declared range;
- the pristine worktree cannot be fetched, created, or safely refreshed;
- a command times out or cannot be launched.

PR #9175 adds this classification to `scripts/pristine_main_health.py`. Until
that exact behavior is present on `CANDIDATE_SHA`, enforce the preflight above
manually. Never interpret a known-invalid toolchain's nonzero result as a new
repository failure.

## 3. Run Required And Full Proofs

Run both suites in report-only mode. Preserve complete stdout, stderr, exit
code, start time, and end time in an operator-owned evidence directory outside
the repository.

```bash
EVIDENCE_PARENT="${TMPDIR:-/tmp}"
EVIDENCE_DIR="$(
  umask 077
  mktemp -d "${EVIDENCE_PARENT%/}/aragora-main-green-${CANDIDATE_SHA}.XXXXXX"
)"

python3 scripts/pristine_main_health.py \
  --repo-root "$REPO_ROOT" \
  --pristine-dir "$PRISTINE_DIR" \
  --halt-file "$HALT_FILE" \
  --suite required \
  --no-halt-file \
  --timeout-minutes 30 \
  >"$EVIDENCE_DIR/required.stdout" \
  2>"$EVIDENCE_DIR/required.stderr"
REQUIRED_EXIT=$?

if [ "$REQUIRED_EXIT" -eq 0 ]; then
  python3 scripts/pristine_main_health.py \
    --repo-root "$REPO_ROOT" \
    --pristine-dir "$PRISTINE_DIR" \
    --halt-file "$HALT_FILE" \
    --suite full \
    --no-halt-file \
    --timeout-minutes 180 \
    >"$EVIDENCE_DIR/full.stdout" \
    2>"$EVIDENCE_DIR/full.stderr"
  FULL_EXIT=$?
else
  FULL_EXIT=not_run
  printf 'full suite not run: required suite exited %s\n' "$REQUIRED_EXIT" \
    >"$EVIDENCE_DIR/full.stdout"
  : >"$EVIDENCE_DIR/full.stderr"
fi

printf 'required_exit=%s\nfull_exit=%s\n' "$REQUIRED_EXIT" "$FULL_EXIT"
```

The proof is green only when both exit codes are zero and both stdout files
report the same 12-character prefix of `CANDIDATE_SHA`; the later full-SHA
guard binds that prefix to the unchanged remote head. A failing
`make ci-required` is a real repository failure only after the toolchain
preflight succeeds. For example, known mypy campaign debt tracked by issue
#9099 remains a real required-suite failure; it is not made green by a passing
focused test or by a draft repair.

Hash the captured evidence:

```bash
shasum -a 256 \
  "$EVIDENCE_DIR/required.stdout" \
  "$EVIDENCE_DIR/required.stderr" \
  "$EVIDENCE_DIR/full.stdout" \
  "$EVIDENCE_DIR/full.stderr"
```

## 4. Recheck Head And Hosted Gates

The local pair is invalid if main moved during collection:

```bash
git fetch origin +refs/heads/main:refs/remotes/origin/main
test "$(git rev-parse origin/main)" = "$CANDIDATE_SHA"
```

Also capture the branch-protection required contexts and every page of check
runs for the same SHA. Reconcile by context name so a required context cannot
disappear merely because it was beyond the API's first page:

```bash
REQUIRED_CONTEXTS_JSON="$(
  gh api repos/synaptent/aragora/branches/main/protection/required_status_checks \
    --jq '[.contexts[], .checks[].context] | unique'
)" || exit 1
CHECK_RUNS_JSON="$(
  gh api --paginate --slurp \
    "repos/synaptent/aragora/commits/$CANDIDATE_SHA/check-runs?filter=latest&per_page=100" \
    | jq '[.[].check_runs[] | {name,status,conclusion,details_url}]'
)" || exit 1

jq -n \
  --argjson required "$REQUIRED_CONTEXTS_JSON" \
  --argjson runs "$CHECK_RUNS_JSON" \
  '[$required[] as $context | {
      context: $context,
      matches: [$runs[] | select(.name == $context)]
    }
    | . + {
        found: (.matches | length > 0),
        conclusions: (.matches | map(.conclusion) | unique)
      }]' \
  | tee "$EVIDENCE_DIR/required-contexts.json"

jq -e 'all(.[]; .found)' "$EVIDENCE_DIR/required-contexts.json"
```

Inspect every row in `required-contexts.json`. A missing context is
`evidence_incomplete`. A main-applicable required context whose latest run did
not complete successfully is `main_red`; a context designed to skip on main
(currently `aragora-merge-quorum`) must remain visible and be identified as an
expected main-only skip in the packet rather than silently treated as green.
An API outage, pagination failure, or rate limit is `evidence_incomplete`, not
green. Retry in a later bounded cycle; do not substitute old PR checks for
current-main checks.

## 5. Build The Human Packet

The re-arm packet must contain:

- full `CANDIDATE_SHA`, commit time, and subject;
- halt-marker body and its pre-run SHA-256;
- Python, pytest, PATH mypy path/version, and declared mypy requirement;
- exact required/full commands, exit codes, timestamps, and log hashes;
- proof that both suites tested the same SHA;
- all branch-protection required contexts and their current-main check URLs;
- the start and end of the observed green interval;
- every anomaly, retry, skipped surface, or unavailable evidence source;
- an explicit statement that no baseline, workflow, branch protection,
  settlement, or halt mutation was used to obtain green.

Use one of these terminal classifications:

| Classification | Meaning | Action |
| --- | --- | --- |
| `main_green_candidate` | Toolchain valid; both local suites and hosted required contexts pass at one unchanged SHA | Ask a human for exact-head re-arm review |
| `main_red` | Toolchain valid; at least one repository check fails | Keep the halt; file or update one exact-head incident |
| `infra_error` | Runner, dependency, launch, timeout, fetch, or worktree failure | Keep the halt; repair infrastructure and rerun |
| `evidence_incomplete` | Head drift, API outage, missing logs, or unmatched SHA | Keep the halt; collect a complete pair later |

## 6. Human Re-Arm Only

The operating contract requires main to remain green for at least one hour
unless the user explicitly waives that interval. A human must review the
packet and authorize the exact `CANDIDATE_SHA`; generic continuation text is
not exact-head re-arm authority.

Immediately before human deletion, run the guards and deletion as one shell
block. Do not split or continue after a failed command:

```bash
set -eu
test "$(shasum -a 256 "$HALT_FILE" | awk '{print $1}')" = "$HALT_SHA256"
git fetch origin +refs/heads/main:refs/remotes/origin/main
test "$(git rev-parse origin/main)" = "$CANDIDATE_SHA"
rm -- "$HALT_FILE"
test ! -e "$HALT_FILE"
```

If either guard fails, preserve the marker and rebuild the packet. Re-arming
does not itself authorize a PR merge, settlement, CI rerun, workflow dispatch,
or branch-protection change; those actions retain their normal gates.
