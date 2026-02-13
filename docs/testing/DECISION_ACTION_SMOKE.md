# Decision-to-Action Smoke Scenario

This scenario validates the end-to-end decision pipeline path:

`debate -> plan -> approval -> execution -> completion artifact`

## Local Run

```bash
pytest tests/e2e/test_decision_action_smoke.py -v --tb=short --timeout=120
```

## Artifact Output

The test writes a deterministic artifact JSON at:

`test-results/decision-action-smoke/decision_action_smoke.json`

Override output directory with:

```bash
DECISION_ACTION_SMOKE_ARTIFACT_DIR=/tmp/decision-action-smoke \
pytest tests/e2e/test_decision_action_smoke.py -v --tb=short --timeout=120
```

## CI Wiring

The smoke workflow runs this test in `.github/workflows/smoke.yml` under the
`decision-action-smoke` job and uploads `test-results/decision-action-smoke/`
as an artifact for debugging failures.
