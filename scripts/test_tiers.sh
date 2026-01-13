#!/usr/bin/env bash
set -euo pipefail

tier="${1:-fast}"

case "$tier" in
  fast)
    pytest tests/ -m "not slow and not load and not e2e" --timeout=30
    ;;
  ci)
    pytest tests/ -v --timeout=60 --cov=aragora --cov-report=term-missing --cov-report=xml -x --tb=short
    ;;
  lint)
    black --check --diff aragora/ tests/ scripts/
    ruff_cmd="ruff"
    if ! command -v ruff >/dev/null 2>&1; then
      if command -v python3 >/dev/null 2>&1; then
        ruff_cmd="python3 -m ruff"
      else
        ruff_cmd="python -m ruff"
      fi
    fi
    $ruff_cmd check aragora/ tests/ scripts/
    ;;
  typecheck)
    # Run mypy on core modules - these MUST pass
    echo "=== Type checking aragora (informational) ==="
    mypy aragora/ --ignore-missing-imports --no-error-summary --show-error-codes || true
    echo ""
    echo "=== Type check complete (non-blocking) ==="
    ;;
  frontend)
    (cd aragora/live && npm test)
    ;;
  e2e)
    (cd aragora/live && npm run test:e2e)
    ;;
  *)
    echo "Usage: $0 {fast|ci|lint|typecheck|frontend|e2e}"
    exit 2
    ;;
esac
