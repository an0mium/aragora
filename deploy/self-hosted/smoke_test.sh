#!/bin/bash
# ==============================================================================
# Aragora Self-Hosted Smoke Test
# ==============================================================================
# Verifies the self-hosted docker compose deployment is working correctly.
# Run this after `docker compose up -d` to validate the deployment.
#
# Usage:
#   ./smoke_test.sh              # Run all tests
#   ./smoke_test.sh --quick      # Run only essential tests
#   ./smoke_test.sh --verbose    # Show detailed output
#
# Exit codes:
#   0 - All critical tests passed
#   1 - One or more critical tests failed
# ==============================================================================

set -e

# Configuration
ARAGORA_PORT="${ARAGORA_PORT:-8080}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9091}"
BASE_URL="http://localhost:${ARAGORA_PORT}"
TIMEOUT="${SMOKE_TEST_TIMEOUT:-10}"
VERBOSE="${1:-}"
QUICK_MODE=false

if [ "$VERBOSE" = "--verbose" ] || [ "$VERBOSE" = "-v" ]; then
    VERBOSE=true
else
    VERBOSE=false
fi

if [ "$1" = "--quick" ] || [ "$2" = "--quick" ]; then
    QUICK_MODE=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
SKIPPED=0
CRITICAL_FAILED=0

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

log_critical_fail() {
    echo -e "${RED}[CRITICAL]${NC} $1"
    ((FAILED++))
    ((CRITICAL_FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

log_section() {
    echo ""
    echo -e "${BLUE}--- $1 ---${NC}"
}

# Make HTTP request
do_request() {
    local method="$1"
    local url="$2"
    local data="${3:-}"

    if [ "$method" = "GET" ]; then
        curl -s -w "\n%{http_code}" \
            --max-time "$TIMEOUT" \
            -H "Content-Type: application/json" \
            "$url" 2>/dev/null
    else
        curl -s -w "\n%{http_code}" \
            --max-time "$TIMEOUT" \
            -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$url" 2>/dev/null
    fi
}

# Check if docker compose services are running
check_docker_services() {
    if ! command -v docker &> /dev/null; then
        log_critical_fail "Docker not found"
        return 1
    fi

    local services
    services=$(docker compose ps --format json 2>/dev/null | grep -c '"running"' || echo "0")

    if [ "$services" -eq 0 ]; then
        # Try alternative check
        services=$(docker compose ps 2>/dev/null | grep -c "Up" || echo "0")
    fi

    if [ "$services" -gt 0 ]; then
        log_pass "Docker services running ($services containers)"
        return 0
    else
        log_critical_fail "No docker compose services running"
        return 1
    fi
}

# ==============================================================================
# Test Functions
# ==============================================================================

test_container_health() {
    log_section "Container Health"

    # Check aragora container
    local aragora_health
    aragora_health=$(docker inspect aragora-api --format='{{.State.Health.Status}}' 2>/dev/null || echo "not found")

    if [ "$aragora_health" = "healthy" ]; then
        log_pass "Aragora container healthy"
    elif [ "$aragora_health" = "starting" ]; then
        log_skip "Aragora container still starting"
    elif [ "$aragora_health" = "not found" ]; then
        log_fail "Aragora container not found (expected: aragora-api)"
    else
        log_fail "Aragora container unhealthy: $aragora_health"
    fi

    # Check postgres container
    local postgres_health
    postgres_health=$(docker inspect aragora-postgres --format='{{.State.Health.Status}}' 2>/dev/null || echo "not found")

    if [ "$postgres_health" = "healthy" ]; then
        log_pass "PostgreSQL container healthy"
    elif [ "$postgres_health" = "not found" ]; then
        log_fail "PostgreSQL container not found"
    else
        log_fail "PostgreSQL container unhealthy: $postgres_health"
    fi

    # Check redis container
    local redis_health
    redis_health=$(docker inspect aragora-redis --format='{{.State.Health.Status}}' 2>/dev/null || echo "not found")

    if [ "$redis_health" = "healthy" ]; then
        log_pass "Redis container healthy"
    elif [ "$redis_health" = "not found" ]; then
        log_fail "Redis container not found"
    else
        log_fail "Redis container unhealthy: $redis_health"
    fi
}

test_liveness_probe() {
    log_section "Liveness Probe"

    local response
    response=$(do_request GET "${BASE_URL}/healthz") || {
        log_critical_fail "Liveness probe - connection failed"
        return 1
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | head -n -1)

    log_verbose "Response: $body"
    log_verbose "Status code: $status_code"

    if [ "$status_code" = "200" ]; then
        log_pass "Liveness probe (/healthz) - status 200"
    else
        log_critical_fail "Liveness probe - got status $status_code"
        return 1
    fi
}

test_readiness_probe() {
    log_section "Readiness Probe"

    local response
    response=$(do_request GET "${BASE_URL}/readyz") || {
        log_fail "Readiness probe - connection failed"
        return 1
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | head -n -1)

    log_verbose "Response: $body"

    if [ "$status_code" = "200" ]; then
        log_pass "Readiness probe (/readyz) - status 200"
    elif [ "$status_code" = "503" ]; then
        log_fail "Readiness probe - service not ready (503)"
        log_verbose "This may indicate database or Redis connectivity issues"
    else
        log_fail "Readiness probe - got status $status_code"
        return 1
    fi
}

test_api_health() {
    log_section "API Health Endpoint"

    local response
    response=$(do_request GET "${BASE_URL}/api/health") || {
        log_fail "API health - connection failed"
        return 1
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | head -n -1)

    log_verbose "Response: $body"

    if [ "$status_code" = "200" ]; then
        log_pass "API health (/api/health) - status 200"

        # Check for database status in response
        if echo "$body" | grep -q '"database"'; then
            if echo "$body" | grep -q '"database".*"connected"'; then
                log_pass "Database connectivity confirmed"
            else
                log_verbose "Database status present but not 'connected'"
            fi
        fi

        # Check for redis status in response
        if echo "$body" | grep -q '"redis"'; then
            if echo "$body" | grep -q '"redis".*"connected"'; then
                log_pass "Redis connectivity confirmed"
            else
                log_verbose "Redis status present but not 'connected'"
            fi
        fi
    elif [ "$status_code" = "401" ]; then
        log_pass "API health - requires auth (expected in production)"
    else
        log_fail "API health - got status $status_code"
    fi
}

test_database_connectivity() {
    log_section "Database Connectivity"

    # Test via docker exec
    local result
    result=$(docker exec aragora-postgres pg_isready -U aragora 2>/dev/null) || {
        log_fail "Database connectivity - pg_isready failed"
        return 1
    }

    if echo "$result" | grep -q "accepting connections"; then
        log_pass "PostgreSQL accepting connections"
    else
        log_fail "PostgreSQL not accepting connections"
    fi
}

test_redis_connectivity() {
    log_section "Redis Connectivity"

    # Test via docker exec - get password from env or use default
    local redis_pass="${REDIS_PASSWORD:-aragora_redis_pass}"
    local result
    result=$(docker exec aragora-redis redis-cli -a "$redis_pass" ping 2>/dev/null) || {
        log_fail "Redis connectivity - ping failed"
        return 1
    }

    if [ "$result" = "PONG" ]; then
        log_pass "Redis responding to PING"
    else
        log_fail "Redis did not respond with PONG"
    fi
}

test_websocket() {
    log_section "WebSocket Endpoint"

    # Check if WebSocket endpoint responds
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT" \
        -H "Upgrade: websocket" \
        -H "Connection: Upgrade" \
        -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
        -H "Sec-WebSocket-Version: 13" \
        "${BASE_URL}/ws" 2>/dev/null) || {
        log_skip "WebSocket - connection test failed"
        return 0
    }

    # 101 = Switching Protocols (WebSocket upgrade)
    # 200 = Endpoint exists but didn't upgrade
    # 426 = Upgrade Required
    if [ "$response" = "101" ]; then
        log_pass "WebSocket - upgrade successful (101)"
    elif [ "$response" = "200" ] || [ "$response" = "426" ]; then
        log_pass "WebSocket - endpoint responds ($response)"
    elif [ "$response" = "404" ]; then
        log_skip "WebSocket - endpoint not found"
    else
        log_skip "WebSocket - got status $response"
    fi
}

test_metrics_endpoint() {
    log_section "Metrics Endpoint"

    # First check if monitoring profile is active
    local prometheus_running
    prometheus_running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c "aragora-prometheus" || echo "0")

    if [ "$prometheus_running" = "0" ]; then
        log_skip "Prometheus not running (--profile monitoring not enabled)"
        return 0
    fi

    local response
    response=$(do_request GET "http://localhost:${PROMETHEUS_PORT}/api/v1/status/config") || {
        log_skip "Prometheus - connection failed"
        return 0
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_pass "Prometheus metrics available"
    else
        log_fail "Prometheus - got status $status_code"
    fi

    # Check Aragora metrics endpoint
    local metrics_response
    metrics_response=$(do_request GET "${BASE_URL}/metrics") || {
        log_skip "Aragora metrics endpoint not available"
        return 0
    }

    local metrics_status
    metrics_status=$(echo "$metrics_response" | tail -n1)

    if [ "$metrics_status" = "200" ]; then
        log_pass "Aragora /metrics endpoint available"
    elif [ "$metrics_status" = "404" ]; then
        log_skip "Aragora /metrics endpoint not found"
    else
        log_verbose "Aragora /metrics returned $metrics_status"
    fi
}

test_api_endpoints() {
    log_section "API Endpoints"

    # Test agents endpoint
    local response
    response=$(do_request GET "${BASE_URL}/api/v1/agents") || {
        log_skip "Agents endpoint - connection failed"
        return 0
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_pass "Agents endpoint (/api/v1/agents) - available"
    elif [ "$status_code" = "401" ]; then
        log_pass "Agents endpoint - auth required (expected)"
    else
        log_verbose "Agents endpoint returned $status_code"
    fi

    # Test debates endpoint
    response=$(do_request GET "${BASE_URL}/api/v1/debates") || {
        log_skip "Debates endpoint - connection failed"
        return 0
    }

    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_pass "Debates endpoint (/api/v1/debates) - available"
    elif [ "$status_code" = "401" ]; then
        log_pass "Debates endpoint - auth required (expected)"
    else
        log_verbose "Debates endpoint returned $status_code"
    fi
}

test_grafana() {
    log_section "Grafana (Optional)"

    local grafana_running
    grafana_running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c "aragora-grafana" || echo "0")

    if [ "$grafana_running" = "0" ]; then
        log_skip "Grafana not running (--profile monitoring not enabled)"
        return 0
    fi

    local grafana_port="${GRAFANA_PORT:-3001}"
    local response
    response=$(do_request GET "http://localhost:${grafana_port}/api/health") || {
        log_skip "Grafana - connection failed"
        return 0
    }

    local status_code
    status_code=$(echo "$response" | tail -n1)

    if [ "$status_code" = "200" ]; then
        log_pass "Grafana available at http://localhost:${grafana_port}"
    else
        log_fail "Grafana - got status $status_code"
    fi
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo ""
    echo "=============================================="
    echo "    Aragora Self-Hosted Smoke Test"
    echo "=============================================="
    echo ""
    echo "Target:     ${BASE_URL}"
    echo "Quick mode: ${QUICK_MODE}"
    echo "Verbose:    ${VERBOSE}"
    echo ""

    # Pre-flight: Check docker services
    check_docker_services || {
        echo ""
        echo -e "${RED}Cannot proceed: Docker services not running${NC}"
        echo "Run: docker compose up -d"
        exit 1
    }

    # Critical tests (must pass)
    test_container_health
    test_liveness_probe
    test_readiness_probe

    # Infrastructure tests
    test_database_connectivity
    test_redis_connectivity

    # API tests
    test_api_health

    if [ "$QUICK_MODE" = false ]; then
        # Extended tests
        test_websocket
        test_api_endpoints
        test_metrics_endpoint
        test_grafana
    fi

    echo ""
    echo "=============================================="
    echo "                 Results"
    echo "=============================================="
    echo ""
    echo -e "  ${GREEN}Passed:  $PASSED${NC}"
    echo -e "  ${RED}Failed:  $FAILED${NC}"
    echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"
    echo ""

    if [ $CRITICAL_FAILED -gt 0 ]; then
        echo -e "${RED}CRITICAL FAILURES: $CRITICAL_FAILED${NC}"
        echo "The deployment is not functional. Check logs with:"
        echo "  docker compose logs aragora"
        echo ""
        exit 1
    elif [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed! Deployment is healthy.${NC}"
        echo ""
        echo "Access points:"
        echo "  API:     ${BASE_URL}"
        echo "  Health:  ${BASE_URL}/healthz"

        # Check for optional services
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "aragora-grafana"; then
            echo "  Grafana: http://localhost:${GRAFANA_PORT:-3001}"
        fi
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "aragora-prometheus"; then
            echo "  Prometheus: http://localhost:${PROMETHEUS_PORT:-9091}"
        fi
        echo ""
        exit 0
    else
        echo -e "${YELLOW}Some non-critical tests failed.${NC}"
        echo "The deployment may be partially functional."
        echo ""
        exit 0
    fi
}

# Run main
main "$@"
