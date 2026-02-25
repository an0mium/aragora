#!/usr/bin/env bash
set -euo pipefail

# Blocking lane for epistemic hygiene + settlement pathways.
# Kept intentionally focused to catch regressions in:
# - hygiene-mode protocol/penalty enforcement
# - settlement extraction and resolution
# - settlement review scheduler calibration writeback
# - debate controller metadata propagation

pytest \
  tests/modes/test_epistemic_hygiene.py \
  tests/debate/test_epistemic_hygiene.py \
  tests/debate/test_settlement.py \
  tests/schedulers/test_settlement_review.py \
  tests/test_debate_controller.py \
  -k "epistemic_hygiene or settlement" \
  -m "not slow and not load and not e2e and not integration and not integration_minimal and not benchmark and not performance" \
  --timeout=120 \
  --tb=short \
  -q

ARTIFACT_DIR="${EPISTEMIC_GATE_ARTIFACT_DIR:-/tmp/epistemic-gate-artifacts}"
mkdir -p "${ARTIFACT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

"${PYTHON_BIN}" scripts/check_epistemic_compliance_regression.py \
  --strict \
  --fixtures scripts/fixtures/epistemic_compliance_fixtures.json \
  --baseline scripts/baselines/epistemic_compliance_regression.json \
  --json-out "${ARTIFACT_DIR}/epistemic-compliance-summary.json" \
  --md-out "${ARTIFACT_DIR}/epistemic-compliance-summary.md"

if [[ ! -s "${ARTIFACT_DIR}/epistemic-compliance-summary.json" ]]; then
  echo "Missing or empty epistemic compliance JSON artifact"
  exit 1
fi

if [[ ! -s "${ARTIFACT_DIR}/epistemic-compliance-summary.md" ]]; then
  echo "Missing or empty epistemic compliance Markdown artifact"
  exit 1
fi

"${PYTHON_BIN}" - "${ARTIFACT_DIR}/epistemic-compliance-summary.json" <<'PY'
import json
import sys
from pathlib import Path

artifact = Path(sys.argv[1])
payload = json.loads(artifact.read_text(encoding="utf-8"))
required = {"generated_at", "fixture_case_count", "models", "cases", "regressions", "passed"}
missing = sorted(required - set(payload))
if missing:
    raise SystemExit(f"Epistemic compliance artifact missing keys: {', '.join(missing)}")
if not isinstance(payload.get("models"), dict) or not payload["models"]:
    raise SystemExit("Epistemic compliance artifact models section is empty")
print("Epistemic gate artifact validation passed")
PY
