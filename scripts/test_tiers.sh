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
    ruff check aragora/ tests/ scripts/
    ;;
  typecheck)
    mypy aragora/ --ignore-missing-imports --no-error-summary --show-error-codes
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
