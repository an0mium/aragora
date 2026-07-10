# Main-red full-suite runner environment diagnosis

Date: 2026-07-10

## Scope

This note records the exact cause of the current pristine-main full-suite halt.
It does not re-arm the merge executor, change the health probe, mutate the
shared Python environment, rerun CI, or make a claim about the outcome of the
test suite.

## Live state

- `origin/main`: `3fe2e5cf561fc094221008d064040ac84625bd4e`
- Halt file: `.aragora/merge_executor.halt`
- Halt SHA-256:
  `ba3004e6a2faa49a024d9231b268e9fe9f1a221dfbcc5234ce4a3a8ac31289f8`
- Halt writer: `scripts/pristine_main_health.py`
- Halt timestamp: `2026-07-10T08:30:12.535152+00:00`
- Halt command:

  ```text
  $ARAGORA_PYTHON -m pytest tests/ -q -p no:cacheprovider --ignore=tests/connectors
  ```

At diagnosis time, all five protected GitHub required contexts were successful
on this exact main SHA: `Generate & Validate`, `TypeScript SDK Type Check`,
`lint`, `sdk-parity`, and `typecheck`. The `aragora-merge-quorum` check was
skipped on the main push, as expected. These results do not replace the local
pristine-main health gate.

## Reproduction

The interpreter preflight fails before test discovery:

```text
$ $ARAGORA_PYTHON -m pytest --version
python3: No module named pytest
$ echo $?
1
```

`scripts/pristine_main_health.py` builds the full-suite command with
`sys.executable`, so a probe launched from this repository `.venv` uses the
same interpreter shown above. The current failure is therefore a runner
environment preflight failure: that interpreter cannot import `pytest`.

The halt detail contains the command and exit code but not the diagnostic
message because the nonzero-result path records only the tail of captured
stdout. Python wrote `No module named pytest` to stderr.

## Isolated evidence

An isolated worktree at the same `origin/main` SHA used Python 3.11.11 and a
new virtual environment. The documented project test extra was installed with:

```text
$EVIDENCE_PYTHON -m pip install -e '.[test]'
```

The interpreter remedy was successful:

```text
$ $EVIDENCE_PYTHON -m pytest --version
pytest 9.1.1
```

The exact halted suite command was then run with a two-hour cap:

```text
$ $EVIDENCE_PYTHON -m pytest tests/ -q -p no:cacheprovider --ignore=tests/connectors
```

Verdict: **INCOMPLETE**. Collection stopped after 244.89 seconds with 84
dependency errors, 4 skips, and 99 warnings; the process exited 2 after 259
seconds total. No test cases executed, so these collection errors are not
evidence that main's code is red.

The 17 missing import families were `asyncpg`, `boto3`, `botocore`,
`cryptography`, `defusedxml`, `fastapi`, `jinja2`, `jsonschema`, `jwt`, `mcp`,
`numpy`, `openai`, `pyotp`, `reportlab`, `watchfiles`, `weaviate`, and `yt_dlp`.
The full log is `/tmp/mainred-evidence-20260710T120908Z.log`.

The repository's fuller test provisioning profile is
`scripts/ci_install_project.sh --extras test`. It declares packages for all of
these imports except the runtime `jsonschema` package; its current declaration
is the `types-jsonschema` stub package. That profile is therefore the correct
starting point for an environment-owner verification, with `jsonschema` added
explicitly unless dependency resolution supplies it transitively.

## Conclusion

The current marker does not prove that tests on `origin/main` failed. No tests
were collected or executed by this invocation. The halt remains binding until
a human verifies main health and re-arms the executor according to the
operating contract.

## Recovery options

### Option A: provision and verify the probe environment

The environment owner captures the current package state, provisions the exact
interpreter used by the scheduled probe, and runs report-only verification:

```text
$PROBE_PYTHON -m pip freeze > /tmp/pristine-probe-before.txt
PATH="$PROBE_ENV_BIN:$PATH" bash scripts/ci_install_project.sh \
  --project-dir "$REPO_ROOT" --extras test --install-mode editable
$PROBE_PYTHON -m pip install 'jsonschema>=4,<5'
$PROBE_PYTHON -m pytest --version
$PROBE_PYTHON scripts/pristine_main_health.py --suite full --no-halt-file
```

Blast radius is limited to the local Python environment used by the scheduled
probe, but dependency upgrades can affect every local job that shares that
environment. Capture a post-install package snapshot and command log. Roll back
by recreating that local environment from the pre-install snapshot rather than
editing repository files or weakening the suite.

Acceptance evidence is a successful `pytest --version`, a zero exit from the
full no-halt-file run on the then-current `origin/main`, and its complete log.
Only after verifying that evidence should a human delete the unchanged halt
marker according to its `re_arm` field.

### Option B: harden the probe

In a separate authorized change, make `scripts/pristine_main_health.py`
preflight `sys.executable -m pytest --version` before running a suite and retain
the tails of both stdout and stderr in the durable result and halt detail. A
preflight failure must remain fail-closed and must be classified separately
from test failure. This edits dev-loop and merge-halt tooling, so it requires
explicit approval before implementation and its own review path.

## Recommendation and requested grant

Use Option A now to establish main's actual full-suite state. Pursue Option B
afterward as durable diagnostic hardening.

Requested grant: **authorize Option A for `origin/main`
`3fe2e5cf561fc094221008d064040ac84625bd4e` and halt SHA-256
`ba3004e6a2faa49a024d9231b268e9fe9f1a221dfbcc5234ce4a3a8ac31289f8`, allowing
the repository environment owner to provision the scheduled probe environment,
run the full no-halt-file verification, and re-arm only if that exact evidence
is green.**

Until one path is completed, keep queue settlement and merge execution halted.
The active mypy remediation lanes are independent and must retain their current
file ownership.
