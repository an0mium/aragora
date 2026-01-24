#!/bin/bash
# ==============================================================================
# Aragora Smoke Test Script
# ==============================================================================
# Verifies a fresh Aragora deployment is working correctly.
# Target: All checks pass in <30 seconds
#
# Usage:
#   ./smoke-test.sh                    # Test localhost:8080
#   ./smoke-test.sh https://api.example.com  # Test custom URL
#   ARAGORA_API_TOKEN=xxx ./smoke-test.sh    # With auth token
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
# ==============================================================================

set -e

# Configuration
BASE_URL="${1:-http://localhost:8080}"
API_TOKEN="${ARAGORA_API_TOKEN:-}"
TIMEOUT="${SMOKE_TEST_TIMEOUT:-10}"
VERBOSE="${SMOKE_TEST_VERBOSE:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
SKIPPED=0

# ==============================================================================
# Helper Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

log_verbose() {
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Make HTTP request with optional auth
do_request() {
    local method="$1"
    local path="$2"
    local data="${3:-}"

    local headers=(-H "Content-Type: application/json")
    if [ -n "$API_TOKEN" ]; then
        headers+=(-H "Authorization: Bearer $API_TOKEN")
    fi

    if [ "$method" = "GET" ]; then
        curl -s -w "\n%{http_code}" \
            --max-time "$TIMEOUT" \
            "${headers[@]}" \
            "${BASE_URL}${path}"
    else
        curl -s -w "\n%{http_code}" \
            --max-time "$TIMEOUT" \
            -X "$method" \
            "${headers[@]}" \
            -d "$data" \
            "${BASE_URL}${path}"
    fi
}

# Check if jq is available
check_jq() {
    if ! command -v jq &> /dev/null; then
        log_info "jq not found, some tests will be limited"
        return 1
    fi
    return 0
}

# ==============================================================================
# Test Functions
# ==============================================================================

test_health() {
    log_info "Testing health endpoint..."

    local response
    response=$(do_request GET "/api/health" 2>/dev/null) || {
        log_fail "Health check - connection failed"
        return 1
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | head -n -1)

    log_verbose "Response: $body"
    log_verbose "Status code: $status_code"

    if [ "$status_code" = "200" ]; then
        log_pass "Health check - status 200"
    else
        log_fail "Health check - got status $status_code"
        return 1
    fi

    # Check for healthy status in body
    if echo "$body" | grep -q '"status".*"healthy"'; then
        log_pass "Health check - status is healthy"
    elif echo "$body" | grep -q '"status"'; then
        log_fail "Health check - status is not healthy: $body"
        return 1
    else
        log_pass "Health check - response received"
    fi
}

test_health_v1() {
    log_info "Testing v1 health endpoint..."

    local response
    response=$(do_request GET "/api/v1/health" 2>/dev/null) || {
        log_skip "v1 Health check - connection failed (may not be implemented)"
        return 0
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ] || [ "$status_code" = "404" ]; then
        log_pass "v1 Health check - endpoint responded"
    else
        log_fail "v1 Health check - unexpected status $status_code"
        return 1
    fi
}

test_agents_list() {
    log_info "Testing agents list endpoint..."

    local response
    response=$(do_request GET "/api/v1/agents" 2>/dev/null) || {
        log_fail "Agents list - connection failed"
        return 1
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | head -n -1)

    log_verbose "Response: $body"

    if [ "$status_code" = "200" ]; then
        log_pass "Agents list - status 200"
    elif [ "$status_code" = "401" ]; then
        log_skip "Agents list - requires authentication"
        return 0
    else
        log_fail "Agents list - got status $status_code"
        return 1
    fi

    # Check for agents array
    if echo "$body" | grep -q '"agents"'; then
        log_pass "Agents list - returned agents array"
    else
        log_fail "Agents list - no agents array in response"
        return 1
    fi
}

test_leaderboard() {
    log_info "Testing leaderboard endpoint..."

    local response
    response=$(do_request GET "/api/v1/leaderboard" 2>/dev/null) || {
        log_skip "Leaderboard - connection failed"
        return 0
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | head -n -1)

    log_verbose "Response: $body"

    if [ "$status_code" = "200" ]; then
        log_pass "Leaderboard - status 200"
    elif [ "$status_code" = "401" ]; then
        log_skip "Leaderboard - requires authentication"
        return 0
    elif [ "$status_code" = "503" ]; then
        log_skip "Leaderboard - ELO system not available (expected on fresh install)"
        return 0
    else
        log_fail "Leaderboard - got status $status_code"
        return 1
    fi
}

test_debates_list() {
    log_info "Testing debates list endpoint..."

    local response
    response=$(do_request GET "/api/v1/debates" 2>/dev/null) || {
        log_skip "Debates list - connection failed"
        return 0
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | head -n -1)

    log_verbose "Response: $body"

    if [ "$status_code" = "200" ]; then
        log_pass "Debates list - status 200"
    elif [ "$status_code" = "401" ]; then
        log_skip "Debates list - requires authentication"
        return 0
    else
        log_fail "Debates list - got status $status_code"
        return 1
    fi
}

test_analytics() {
    log_info "Testing analytics endpoints..."

    local response
    response=$(do_request GET "/api/v1/analytics/disagreements" 2>/dev/null) || {
        log_skip "Analytics - connection failed"
        return 0
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_pass "Analytics disagreements - status 200"
    elif [ "$status_code" = "401" ]; then
        log_skip "Analytics - requires authentication"
        return 0
    elif [ "$status_code" = "429" ]; then
        log_skip "Analytics - rate limited"
        return 0
    else
        log_fail "Analytics disagreements - got status $status_code"
        return 1
    fi
}

test_control_plane() {
    log_info "Testing control plane status..."

    local response
    response=$(do_request GET "/api/v1/control-plane/status" 2>/dev/null) || {
        log_skip "Control plane - connection failed"
        return 0
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_pass "Control plane - status 200"
    elif [ "$status_code" = "401" ]; then
        log_skip "Control plane - requires authentication"
        return 0
    elif [ "$status_code" = "404" ]; then
        log_skip "Control plane - endpoint not found"
        return 0
    else
        log_fail "Control plane - got status $status_code"
        return 1
    fi
}

test_openapi() {
    log_info "Testing OpenAPI spec endpoint..."

    local response
    response=$(do_request GET "/api/docs/openapi.json" 2>/dev/null) || {
        # Try alternative paths
        response=$(do_request GET "/openapi.json" 2>/dev/null) || {
            log_skip "OpenAPI - endpoint not found"
            return 0
        }
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_pass "OpenAPI spec - available"
    elif [ "$status_code" = "404" ]; then
        log_skip "OpenAPI spec - not published yet"
        return 0
    else
        log_fail "OpenAPI spec - got status $status_code"
        return 1
    fi
}

test_websocket() {
    log_info "Testing WebSocket endpoint..."

    # Check if wscat or websocat is available
    if command -v wscat &> /dev/null; then
        local ws_url="${BASE_URL/http/ws}/ws"
        timeout 3 wscat -c "$ws_url" --execute 'ping' &> /dev/null && {
            log_pass "WebSocket - connection successful"
            return 0
        }
    elif command -v websocat &> /dev/null; then
        local ws_url="${BASE_URL/http/ws}/ws"
        echo 'ping' | timeout 3 websocat "$ws_url" &> /dev/null && {
            log_pass "WebSocket - connection successful"
            return 0
        }
    fi

    # Fallback: just check HTTP upgrade is available
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT" \
        -H "Upgrade: websocket" \
        -H "Connection: Upgrade" \
        "${BASE_URL}/ws" 2>/dev/null) || {
        log_skip "WebSocket - connection test skipped (no ws client)"
        return 0
    }

    if [ "$response" = "101" ] || [ "$response" = "200" ] || [ "$response" = "426" ]; then
        log_pass "WebSocket - endpoint responds"
    elif [ "$response" = "404" ]; then
        log_skip "WebSocket - endpoint not found"
    else
        log_skip "WebSocket - got status $response"
    fi
}

test_cors() {
    log_info "Testing CORS headers..."

    local response
    response=$(curl -s -I \
        --max-time "$TIMEOUT" \
        -H "Origin: http://example.com" \
        "${BASE_URL}/api/health" 2>/dev/null)

    if echo "$response" | grep -qi "access-control"; then
        log_pass "CORS - headers present"
    else
        log_skip "CORS - headers not present (may require specific origins)"
    fi
}

test_rate_limiting() {
    log_info "Testing rate limiting headers..."

    local response
    response=$(curl -s -I \
        --max-time "$TIMEOUT" \
        "${BASE_URL}/api/health" 2>/dev/null)

    if echo "$response" | grep -qi "x-ratelimit\|retry-after"; then
        log_pass "Rate limiting - headers present"
    else
        log_skip "Rate limiting - headers not visible"
    fi
}

test_security_headers() {
    log_info "Testing security headers..."

    local response
    response=$(curl -s -I \
        --max-time "$TIMEOUT" \
        "${BASE_URL}/api/health" 2>/dev/null)

    local found=0

    if echo "$response" | grep -qi "x-content-type-options"; then
        found=$((found + 1))
    fi

    if echo "$response" | grep -qi "x-frame-options"; then
        found=$((found + 1))
    fi

    if echo "$response" | grep -qi "content-security-policy"; then
        found=$((found + 1))
    fi

    if [ $found -ge 2 ]; then
        log_pass "Security headers - $found/3 headers present"
    elif [ $found -ge 1 ]; then
        log_skip "Security headers - only $found/3 headers present"
    else
        log_skip "Security headers - not present"
    fi
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo ""
    echo "=============================================="
    echo "        Aragora Smoke Test Suite"
    echo "=============================================="
    echo ""
    echo "Target:  $BASE_URL"
    echo "Timeout: ${TIMEOUT}s per request"
    echo "Auth:    ${API_TOKEN:+configured}${API_TOKEN:-not configured}"
    echo ""
    echo "----------------------------------------------"
    echo ""

    # Core tests (must pass)
    test_health || true
    test_health_v1 || true

    # API endpoint tests
    test_agents_list || true
    test_leaderboard || true
    test_debates_list || true
    test_analytics || true
    test_control_plane || true

    # Documentation
    test_openapi || true

    # Infrastructure tests
    test_websocket || true
    test_cors || true
    test_rate_limiting || true
    test_security_headers || true

    echo ""
    echo "----------------------------------------------"
    echo ""
    echo "Results:"
    echo -e "  ${GREEN}Passed:  $PASSED${NC}"
    echo -e "  ${RED}Failed:  $FAILED${NC}"
    echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All critical tests passed!${NC}"
        echo ""
        exit 0
    else
        echo -e "${RED}Some tests failed. Check the output above.${NC}"
        echo ""
        exit 1
    fi
}

# Run main
main "$@"
