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

## Conclusion

The current marker does not prove that tests on `origin/main` failed. No tests
were collected or executed by this invocation. The halt remains binding until
a human verifies main health and re-arms the executor according to the
operating contract.

## Recovery paths

1. Environment repair: ensure `pytest` and the repository test dependencies
   are installed in the exact interpreter that launches
   `scripts/pristine_main_health.py`, then run its documented no-halt-file
   verification path before human re-arm.
2. Separately authorized probe hardening: preflight `python -m pytest --version`
   and preserve both stdout and stderr in the durable diagnostic artifact and
   halt detail. This changes governance-tool behavior and must be reviewed as
   its own bounded unit.

Until one path is completed, keep queue settlement and merge execution halted.
The active mypy remediation lanes are independent and must retain their current
file ownership.
