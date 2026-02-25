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

