#!/usr/bin/env bash
# Test Tier Runner for Aragora
#
# Tiers:
#   fast     - Quick unit tests (<5 min) - no slow/load/e2e/integration
#   unit     - Unit tests only with extended timeout
#   ci       - CI tests with coverage (no slow/e2e tests)
#   full     - All tests with extended timeouts
#   slow     - Only slow-marked tests
#   handlers - Only handler tests
#   security - Only security tests
#   lint     - Linting checks
#   typecheck - Type checking
#   frontend - Frontend tests
#   e2e      - End-to-end tests
#
# Usage: ./scripts/test_tiers.sh <tier>
#
set -euo pipefail

tier="${1:-fast}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running test tier: ${tier}${NC}"

case "$tier" in
  fast)
    # Quick tests for local dev - exclude slow, load, e2e, integration
    pytest tests/ -m "not slow and not load and not e2e and not integration" \
      --timeout=30 \
      -q \
      --tb=line
    ;;

  unit)
    # Unit tests with extended timeout - ideal for quick feedback
    pytest tests/ -m "not slow and not load and not e2e and not integration" \
      --timeout=120 \
      -v \
      --tb=short
    ;;

  ci)
    # CI tier - balanced coverage vs speed
    # Skip slow and e2e tests, but include integration tests
    # Coverage threshold: 30% (raised from 19% - Sprint 2.1)
    pytest tests/ \
      -m "not slow and not load and not e2e" \
      --timeout=120 \
      --cov=aragora \
      --cov-report=term-missing \
      --cov-report=xml \
      --cov-report=html \
      --cov-fail-under=30 \
      -v \
      --tb=short \
      -x
    ;;

  full)
    # Full test suite with extended timeouts
    pytest tests/ \
      --timeout=300 \
      --cov=aragora \
      --cov-report=term-missing \
      --cov-report=xml \
      -v \
      --tb=short
    ;;

  slow)
    # Only slow-marked tests
    pytest tests/ -m "slow" \
      --timeout=600 \
      -v \
      --tb=short
    ;;

  handlers)
    # Handler tests only - quick feedback on API changes
    pytest tests/server/handlers/ \
      --timeout=120 \
      -v \
      --tb=short
    ;;

  security)
    # Security-related tests
    pytest tests/security/ tests/server/handlers/test_admin.py tests/server/handlers/test_privacy.py tests/server/middleware/ \
      --timeout=120 \
      -v \
      --tb=short
    ;;

  storage)
    # Storage/database tests
    pytest tests/storage/ tests/ranking/ tests/memory/ \
      --timeout=120 \
      -v \
      --tb=short
    ;;

  privacy)
    # Privacy handler tests only
    pytest tests/server/handlers/test_privacy.py \
      --timeout=60 \
      -v \
      --tb=short
    ;;

  lint)
    echo -e "${YELLOW}Running linting checks...${NC}"
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
    echo -e "${GREEN}Linting passed!${NC}"
    ;;

  typecheck)
    # Run mypy on core modules - informational only
    echo -e "${YELLOW}=== Type checking aragora (informational) ===${NC}"
    mypy aragora/ --ignore-missing-imports --no-error-summary --show-error-codes || true
    echo ""
    echo -e "${GREEN}=== Type check complete (non-blocking) ===${NC}"
    ;;

  frontend)
    (cd aragora/live && npm test)
    ;;

  e2e)
    (cd aragora/live && npm run test:e2e)
    ;;

  help|--help|-h)
    echo "Test Tier Runner for Aragora"
    echo ""
    echo "Usage: $0 <tier>"
    echo ""
    echo "Available tiers:"
    echo "  fast      Quick unit tests, 30s timeout (~2 min)"
    echo "  unit      Unit tests, 120s timeout (~5 min)"
    echo "  ci        CI tests with coverage, excludes slow/e2e (~10 min)"
    echo "  full      All tests, 300s timeout (~30 min)"
    echo "  slow      Only slow-marked tests"
    echo "  handlers  Handler tests only"
    echo "  security  Security-related tests"
    echo "  storage   Storage/database tests"
    echo "  privacy   Privacy handler tests only"
    echo "  lint      Run linting (black, ruff)"
    echo "  typecheck Run type checking (mypy)"
    echo "  frontend  Frontend unit tests"
    echo "  e2e       End-to-end tests"
    echo ""
    echo "Examples:"
    echo "  $0 fast       # Quick local dev feedback"
    echo "  $0 ci         # Run as CI would"
    echo "  $0 handlers   # Test API handlers only"
    ;;

  *)
    echo -e "${RED}Unknown tier: $tier${NC}"
    echo "Run '$0 help' for available tiers"
    exit 2
    ;;
esac

echo -e "${GREEN}Test tier '$tier' completed successfully!${NC}"
