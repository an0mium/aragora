#!/usr/bin/env bash
# Aragora Production Health Check Script
# Verifies all services are running and responding correctly
#
# Usage: ./scripts/healthcheck.sh [options]
#   --verbose    Show detailed output
#   --json       Output as JSON
#   --quiet      Only show failures
#   --url URL    Base URL (default: http://localhost:8080)

set -eo pipefail

# Defaults
VERBOSE=false
JSON=false
QUIET=false
BASE_URL="${ARAGORA_URL:-http://localhost:8080}"
TIMEOUT=5

# Colors (disabled in --quiet or --json mode)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v) VERBOSE=true; shift ;;
        --json|-j) JSON=true; shift ;;
        --quiet|-q) QUIET=true; shift ;;
        --url) BASE_URL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Disable colors for JSON/quiet output
if [[ "$JSON" == "true" ]] || [[ "$QUIET" == "true" ]]; then
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

# Track overall status
OVERALL_STATUS="healthy"
declare -A RESULTS

# Helper functions
log_info() {
    [[ "$QUIET" == "false" ]] && [[ "$JSON" == "false" ]] && echo -e "${BLUE}[INFO]${NC} $1"
}

log_ok() {
    [[ "$QUIET" == "false" ]] && [[ "$JSON" == "false" ]] && echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    [[ "$JSON" == "false" ]] && echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    [[ "$JSON" == "false" ]] && echo -e "${RED}[FAIL]${NC} $1"
}

check_endpoint() {
    local name="$1"
    local endpoint="$2"
    local expected_status="${3:-200}"

    log_info "Checking $name..."

    local start_time=$(date +%s%N)
    local response
    local http_code

    # Make request
    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" "${BASE_URL}${endpoint}" 2>/dev/null) || response="000"

    local end_time=$(date +%s%N)
    local latency_ms=$(( (end_time - start_time) / 1000000 ))

    if [[ "$response" == "$expected_status" ]]; then
        log_ok "$name (${latency_ms}ms)"
        RESULTS["$name"]="ok:${latency_ms}ms"
        return 0
    else
        log_error "$name (got $response, expected $expected_status)"
        RESULTS["$name"]="fail:$response"
        OVERALL_STATUS="unhealthy"
        return 1
    fi
}

check_docker_service() {
    local name="$1"

    log_info "Checking Docker service: $name..."

    if docker compose ps "$name" --format json 2>/dev/null | grep -q '"State":"running"'; then
        log_ok "Docker service $name is running"
        RESULTS["docker_$name"]="ok"
        return 0
    else
        log_error "Docker service $name is not running"
        RESULTS["docker_$name"]="fail"
        OVERALL_STATUS="unhealthy"
        return 1
    fi
}

# Header
[[ "$JSON" == "false" ]] && [[ "$QUIET" == "false" ]] && {
    echo ""
    echo "=========================================="
    echo "  Aragora Production Health Check"
    echo "=========================================="
    echo "  Base URL: $BASE_URL"
    echo "  Time: $(date -Iseconds)"
    echo "=========================================="
    echo ""
}

# Check core API endpoints
log_info "Checking API endpoints..."

check_endpoint "API Health" "/api/health" 200 || true
check_endpoint "API Version" "/api/version" 200 || true
check_endpoint "Metrics" "/api/metrics" 200 || true

# Check optional service endpoints
[[ "$VERBOSE" == "true" ]] && {
    check_endpoint "Database Health" "/api/health/db" 200 || true
    check_endpoint "Redis Health" "/api/health/redis" 200 || true
    check_endpoint "Debates Endpoint" "/api/v1/debates" 200 || true
    check_endpoint "Agents Endpoint" "/api/v1/agents" 200 || true
}

# Check Docker services if docker is available
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    [[ "$QUIET" == "false" ]] && [[ "$JSON" == "false" ]] && echo ""
    log_info "Checking Docker services..."

    check_docker_service "aragora" || true

    [[ "$VERBOSE" == "true" ]] && {
        check_docker_service "postgres" || true
        check_docker_service "redis" || true
        check_docker_service "traefik" || true
    }
fi

# Output results
if [[ "$JSON" == "true" ]]; then
    # JSON output
    echo "{"
    echo "  \"status\": \"$OVERALL_STATUS\","
    echo "  \"timestamp\": \"$(date -Iseconds)\","
    echo "  \"base_url\": \"$BASE_URL\","
    echo "  \"checks\": {"

    first=true
    for key in "${!RESULTS[@]}"; do
        [[ "$first" == "false" ]] && echo ","
        first=false
        echo -n "    \"$key\": \"${RESULTS[$key]}\""
    done

    echo ""
    echo "  }"
    echo "}"
else
    # Text summary
    [[ "$QUIET" == "false" ]] && {
        echo ""
        echo "=========================================="
        if [[ "$OVERALL_STATUS" == "healthy" ]]; then
            echo -e "  Status: ${GREEN}HEALTHY${NC}"
        else
            echo -e "  Status: ${RED}UNHEALTHY${NC}"
        fi
        echo "=========================================="
        echo ""
    }
fi

# Exit code
if [[ "$OVERALL_STATUS" == "healthy" ]]; then
    exit 0
else
    exit 1
fi
