#!/usr/bin/env bash
# ==============================================================================
# Aragora One-Click Docker Demo
# ==============================================================================
# Spins up the full SME stack (app + postgres + redis + observability) via
# docker compose, waits for services to be healthy, seeds demo data, and
# prints access URLs.
#
# Usage:
#   ./deploy/scripts/demo.sh             # Start the demo
#   ./deploy/scripts/demo.sh --down      # Tear down the demo
#   ./deploy/scripts/demo.sh --status    # Check running services
#   ./deploy/scripts/demo.sh --seed      # Re-run seed script only
#   ./deploy/scripts/demo.sh --logs      # Tail service logs
#
# Environment:
#   POSTGRES_PASSWORD       Override default postgres password
#   ARAGORA_PORT            Override API port (default: 8080)
#   GRAFANA_PORT            Override Grafana port (default: 3001)
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.sme.yml"
SEED_SCRIPT="$PROJECT_ROOT/scripts/seed_demo_data.py"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_PORT="${ARAGORA_PORT:-8080}"
GRAFANA_PORT="${GRAFANA_PORT:-3001}"
PG_PASSWORD="${POSTGRES_PASSWORD:-aragora_sme}"
DATABASE_URL="postgresql://aragora:${PG_PASSWORD}@localhost:5432/aragora"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { printf "${BLUE}[demo]${NC} %s\n" "$*"; }
ok()   { printf "${GREEN}[demo]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[demo]${NC} %s\n" "$*"; }
err()  { printf "${RED}[demo]${NC} %s\n" "$*" >&2; }

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
check_docker() {
    if ! command -v docker &>/dev/null; then
        err "Docker is not installed."
        echo "  Install: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info &>/dev/null; then
        err "Docker daemon is not running. Start Docker and try again."
        exit 1
    fi

    if docker compose version &>/dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        err "Docker Compose not found."
        echo "  Docker Compose v2 is included with Docker Desktop."
        echo "  Or install: https://docs.docker.com/compose/install/"
        exit 1
    fi

    ok "Docker and Docker Compose available"
}

check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        err "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Service health checks
# ---------------------------------------------------------------------------
wait_for_postgres() {
    log "Waiting for PostgreSQL to be healthy..."
    local retries=0
    local max_retries=30

    while [ $retries -lt $max_retries ]; do
        if docker exec aragora-sme-postgres pg_isready -U aragora &>/dev/null; then
            ok "PostgreSQL is ready"
            return 0
        fi
        retries=$((retries + 1))
        printf "."
        sleep 2
    done
    echo ""
    err "PostgreSQL did not become healthy after $((max_retries * 2))s"
    return 1
}

wait_for_redis() {
    log "Waiting for Redis to be healthy..."
    local retries=0
    local max_retries=15

    while [ $retries -lt $max_retries ]; do
        if docker exec aragora-sme-redis redis-cli ping 2>/dev/null | grep -q PONG; then
            ok "Redis is ready"
            return 0
        fi
        retries=$((retries + 1))
        printf "."
        sleep 2
    done
    echo ""
    err "Redis did not become healthy after $((max_retries * 2))s"
    return 1
}

wait_for_app() {
    log "Waiting for Aragora application..."
    local retries=0
    local max_retries=60

    while [ $retries -lt $max_retries ]; do
        if curl -sf "http://localhost:${API_PORT}/healthz" &>/dev/null; then
            ok "Aragora application is healthy"
            return 0
        fi
        retries=$((retries + 1))
        printf "."
        sleep 2
    done
    echo ""
    warn "Application health check timed out after $((max_retries * 2))s"
    warn "Check logs with: $0 --logs"
    return 1
}

# ---------------------------------------------------------------------------
# Seed demo data
# ---------------------------------------------------------------------------
run_seed() {
    log "Seeding demo data..."

    if [ ! -f "$SEED_SCRIPT" ]; then
        warn "Seed script not found at $SEED_SCRIPT -- skipping"
        return 0
    fi

    # Try running seed via docker exec (preferred -- no local psycopg2 needed)
    if docker exec aragora-sme python scripts/seed_demo_data.py \
        --database-url "postgresql://aragora:${PG_PASSWORD}@postgres:5432/aragora" 2>/dev/null; then
        ok "Demo data seeded via container"
        return 0
    fi

    # Fallback: run locally if python3 + psycopg2 are available
    if command -v python3 &>/dev/null; then
        if python3 -c "import psycopg2" 2>/dev/null; then
            DATABASE_URL="$DATABASE_URL" python3 "$SEED_SCRIPT" && {
                ok "Demo data seeded locally"
                return 0
            }
        fi
    fi

    warn "Could not seed demo data. Install psycopg2-binary or wait for the app container."
    warn "You can re-run: $0 --seed"
    return 0
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_up() {
    check_docker
    check_compose_file

    echo ""
    echo -e "${CYAN}${BOLD}"
    echo "  Aragora One-Click Demo"
    echo -e "${NC}"

    log "Starting services from $COMPOSE_FILE..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d

    wait_for_postgres
    wait_for_redis
    run_seed
    wait_for_app

    echo ""
    echo -e "${GREEN}${BOLD}=======================================${NC}"
    echo -e "${GREEN}${BOLD}       Aragora Demo is Running!        ${NC}"
    echo -e "${GREEN}${BOLD}=======================================${NC}"
    echo ""
    echo -e "  ${BOLD}API:${NC}        http://localhost:${API_PORT}"
    echo -e "  ${BOLD}Health:${NC}     http://localhost:${API_PORT}/healthz"
    echo -e "  ${BOLD}Grafana:${NC}    http://localhost:${GRAFANA_PORT}  (admin / admin)"
    echo ""
    echo -e "  ${BOLD}Demo login:${NC}"
    echo -e "    Email:    ${CYAN}admin@demo.aragora.ai${NC}"
    echo -e "    Password: ${CYAN}demo123${NC}"
    echo ""
    echo -e "  ${BOLD}Commands:${NC}"
    echo -e "    $0 --logs      Tail logs"
    echo -e "    $0 --status    Check services"
    echo -e "    $0 --down      Tear down"
    echo ""
}

cmd_down() {
    check_docker
    check_compose_file

    log "Stopping and removing demo containers..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" down -v

    ok "Demo torn down. Volumes removed."
}

cmd_status() {
    check_docker
    check_compose_file

    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
}

cmd_logs() {
    check_docker
    check_compose_file

    $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f --tail=100
}

cmd_seed() {
    run_seed
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-}" in
    --down)
        cmd_down
        ;;
    --status)
        cmd_status
        ;;
    --logs)
        cmd_logs
        ;;
    --seed)
        cmd_seed
        ;;
    --help|-h)
        head -18 "$0" | tail -16
        exit 0
        ;;
    ""|--up)
        cmd_up
        ;;
    *)
        err "Unknown option: $1"
        echo "Usage: $0 [--up|--down|--status|--logs|--seed|--help]"
        exit 1
        ;;
esac
