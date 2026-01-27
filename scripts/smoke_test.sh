#!/bin/bash
#
# Aragora Self-Hosted Smoke Test
#
# Verifies a self-hosted Aragora deployment is functioning correctly.
# Run this after deployment to validate the installation.
#
# Usage:
#   ./scripts/smoke_test.sh [BASE_URL]
#
# Example:
#   ./scripts/smoke_test.sh http://localhost:8080
#   ./scripts/smoke_test.sh https://aragora.example.com
#

set -e

BASE_URL="${1:-http://localhost:8080}"
PASSED=0
FAILED=0
TOTAL=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  Aragora Self-Hosted Smoke Test"
echo "  Target: $BASE_URL"
echo "=============================================="
echo ""

# Test function
test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local expected_status="${3:-200}"
    local method="${4:-GET}"

    TOTAL=$((TOTAL + 1))

    printf "Testing: %-40s " "$name"

    status=$(curl -s -o /dev/null -w "%{http_code}" -X "$method" "${BASE_URL}${endpoint}" 2>/dev/null || echo "000")

    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}PASS${NC} (HTTP $status)"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAIL${NC} (expected $expected_status, got $status)"
        FAILED=$((FAILED + 1))
    fi
}

# Test JSON response
test_json_endpoint() {
    local name="$1"
    local endpoint="$2"
    local json_path="$3"

    TOTAL=$((TOTAL + 1))

    printf "Testing: %-40s " "$name"

    response=$(curl -s "${BASE_URL}${endpoint}" 2>/dev/null)

    if echo "$response" | jq -e "$json_path" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAIL${NC} (missing $json_path in response)"
        FAILED=$((FAILED + 1))
    fi
}

echo "=== Core Health Checks ==="
test_endpoint "Health endpoint" "/api/health"
test_endpoint "Liveness probe" "/api/v1/health/live"
test_endpoint "Readiness probe" "/api/v1/health/ready"
echo ""

echo "=== API Endpoints ==="
test_endpoint "Leaderboard (public)" "/api/leaderboard"
test_endpoint "Replays list (public)" "/api/replays"
test_endpoint "Pulse trending (public)" "/api/pulse/trending"
echo ""

echo "=== Authentication ==="
test_endpoint "Auth status (no token)" "/api/v1/auth/status" "401"
echo ""

echo "=== API Response Structure ==="
test_json_endpoint "Health returns status" "/api/health" ".status"
test_json_endpoint "Leaderboard returns rankings" "/api/leaderboard" ".rankings"
echo ""

echo "=== Metrics ==="
test_endpoint "Prometheus metrics" "/metrics"
echo ""

# Summary
echo ""
echo "=============================================="
echo "  Results Summary"
echo "=============================================="
echo -e "  Passed: ${GREEN}$PASSED${NC}"
echo -e "  Failed: ${RED}$FAILED${NC}"
echo "  Total:  $TOTAL"
echo "=============================================="

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
