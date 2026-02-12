#!/usr/bin/env bash
# =============================================================================
# Aragora Self-Hosted Installer
# =============================================================================
#
# One-command setup for Aragora production deployment.
#
# Usage:
#   bash scripts/install.sh                  # Interactive setup
#   bash scripts/install.sh --no-start       # Configure only, do not start
#   bash scripts/install.sh --profile monitoring --profile workers
#   bash scripts/install.sh --help
#
# Prerequisites:
#   - Docker 20.10+
#   - Docker Compose v2 (docker compose)
#   - 4 GB RAM minimum
#   - 10 GB free disk space
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy"
COMPOSE_FILE="$DEPLOY_DIR/docker-compose.production.yml"
ENV_TEMPLATE="$DEPLOY_DIR/.env.template"
ENV_FILE="$DEPLOY_DIR/.env"
NGINX_CERTS_DIR="$DEPLOY_DIR/nginx/certs"
BACKUPS_DIR="$DEPLOY_DIR/backups"

MIN_DOCKER_VERSION="20.10"
MIN_RAM_MB=3584  # ~3.5 GB

# ---------------------------------------------------------------------------
# Colors (disabled when not a terminal)
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo -e "${BLUE}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; }
fatal() { error "$@"; exit 1; }

banner() {
    echo ""
    echo -e "${BOLD}============================================${NC}"
    echo -e "${BOLD}    Aragora Self-Hosted Installer${NC}"
    echo -e "${BOLD}============================================${NC}"
    echo ""
}

# Generate a URL-safe random string of the given length.
random_secret() {
    local len="${1:-32}"
    openssl rand -base64 "$((len * 2))" 2>/dev/null \
        | tr -dc 'a-zA-Z0-9' \
        | head -c "$len"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

NO_START=false
PROFILES=()
SKIP_PREREQS=false
FORCE_ENV=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --no-start              Configure only; do not start containers
  --profile <name>        Enable a Docker Compose profile (monitoring, workers, backup)
                          Can be specified multiple times
  --skip-prereqs          Skip prerequisite checks (for CI)
  --force-env             Overwrite existing .env file
  -h, --help              Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") --profile monitoring --profile workers
  $(basename "$0") --no-start
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-start)       NO_START=true; shift ;;
        --profile)        PROFILES+=("$2"); shift 2 ;;
        --skip-prereqs)   SKIP_PREREQS=true; shift ;;
        --force-env)      FORCE_ENV=true; shift ;;
        -h|--help)        usage ;;
        *)                fatal "Unknown option: $1. Use --help for usage." ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Prerequisites
# ---------------------------------------------------------------------------

check_prerequisites() {
    info "Checking prerequisites..."

    # Docker
    if ! command -v docker &>/dev/null; then
        fatal "Docker is not installed. Install it from https://docs.docker.com/get-docker/"
    fi

    # Docker Compose v2
    if ! docker compose version &>/dev/null; then
        fatal "Docker Compose v2 is required. Update Docker or install the compose plugin."
    fi

    # Docker version check
    local docker_version
    docker_version="$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "0.0")"
    local major minor
    major="$(echo "$docker_version" | cut -d. -f1)"
    minor="$(echo "$docker_version" | cut -d. -f2)"
    local req_major req_minor
    req_major="$(echo "$MIN_DOCKER_VERSION" | cut -d. -f1)"
    req_minor="$(echo "$MIN_DOCKER_VERSION" | cut -d. -f2)"

    if [ "$major" -lt "$req_major" ] || { [ "$major" -eq "$req_major" ] && [ "$minor" -lt "$req_minor" ]; }; then
        fatal "Docker $MIN_DOCKER_VERSION+ required (found $docker_version)."
    fi
    ok "Docker $docker_version"

    # Docker daemon running
    if ! docker info &>/dev/null; then
        fatal "Docker daemon is not running. Start Docker and try again."
    fi
    ok "Docker daemon is running"

    # OpenSSL (for secret generation)
    if ! command -v openssl &>/dev/null; then
        fatal "openssl is required for secret generation."
    fi
    ok "openssl available"

    # RAM check (best-effort, works on Linux and macOS)
    local total_ram_mb=0
    if [ -f /proc/meminfo ]; then
        total_ram_mb=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)
    elif command -v sysctl &>/dev/null; then
        total_ram_mb=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%d", $1/1024/1024}')
    fi

    if [ "$total_ram_mb" -gt 0 ]; then
        if [ "$total_ram_mb" -lt "$MIN_RAM_MB" ]; then
            warn "System has ${total_ram_mb} MB RAM. Minimum recommended is 4 GB."
            warn "The deployment may run slowly or fail to start."
        else
            ok "RAM: ${total_ram_mb} MB"
        fi
    fi

    # Disk space check (10 GB)
    local free_disk_gb
    if command -v df &>/dev/null; then
        free_disk_gb=$(df -BG "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
        if [ -z "$free_disk_gb" ]; then
            # macOS df format
            free_disk_gb=$(df -g "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
        fi
        if [ -n "$free_disk_gb" ] && [ "$free_disk_gb" -lt 10 ] 2>/dev/null; then
            warn "Only ${free_disk_gb} GB free disk space. 10 GB recommended."
        elif [ -n "$free_disk_gb" ]; then
            ok "Disk: ${free_disk_gb} GB free"
        fi
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Step 2: Generate .env
# ---------------------------------------------------------------------------

generate_env() {
    if [ -f "$ENV_FILE" ] && [ "$FORCE_ENV" = false ]; then
        info "Existing .env found at $ENV_FILE"
        info "Using existing configuration. Pass --force-env to regenerate."
        return 0
    fi

    info "Generating environment configuration..."

    if [ ! -f "$ENV_TEMPLATE" ]; then
        fatal "Template not found at $ENV_TEMPLATE"
    fi

    cp "$ENV_TEMPLATE" "$ENV_FILE"

    # Generate secrets
    local pg_pass redis_pass jwt_secret receipt_key grafana_pass
    pg_pass="$(random_secret 32)"
    redis_pass="$(random_secret 24)"
    jwt_secret="$(random_secret 44)"
    receipt_key="$(random_secret 32)"
    grafana_pass="$(random_secret 16)"

    # Replace placeholders (portable sed for macOS + Linux)
    if sed --version 2>/dev/null | grep -q GNU; then
        # GNU sed
        sed -i "s|CHANGEME_POSTGRES_PASSWORD|$pg_pass|g"    "$ENV_FILE"
        sed -i "s|CHANGEME_REDIS_PASSWORD|$redis_pass|g"    "$ENV_FILE"
        sed -i "s|CHANGEME_JWT_SECRET|$jwt_secret|g"        "$ENV_FILE"
        sed -i "s|CHANGEME_RECEIPT_KEY|$receipt_key|g"       "$ENV_FILE"
        sed -i "s|CHANGEME_GRAFANA_PASSWORD|$grafana_pass|g" "$ENV_FILE"
    else
        # BSD sed (macOS)
        sed -i '' "s|CHANGEME_POSTGRES_PASSWORD|$pg_pass|g"    "$ENV_FILE"
        sed -i '' "s|CHANGEME_REDIS_PASSWORD|$redis_pass|g"    "$ENV_FILE"
        sed -i '' "s|CHANGEME_JWT_SECRET|$jwt_secret|g"        "$ENV_FILE"
        sed -i '' "s|CHANGEME_RECEIPT_KEY|$receipt_key|g"       "$ENV_FILE"
        sed -i '' "s|CHANGEME_GRAFANA_PASSWORD|$grafana_pass|g" "$ENV_FILE"
    fi

    ok "Generated secrets and wrote $ENV_FILE"

    # Prompt for API key
    if [ -t 0 ]; then
        # Source env to check if key already provided
        set +u
        # shellcheck disable=SC1090
        source "$ENV_FILE" 2>/dev/null || true
        set -u

        if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
            echo ""
            echo -e "${YELLOW}No AI provider API key detected.${NC}"
            echo "At least one key is required for Aragora to run debates."
            echo ""
            read -rp "Anthropic API key (or Enter to skip): " api_key
            if [ -n "$api_key" ]; then
                if sed --version 2>/dev/null | grep -q GNU; then
                    sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$api_key|" "$ENV_FILE"
                else
                    sed -i '' "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$api_key|" "$ENV_FILE"
                fi
                ok "API key saved"
            else
                warn "No API key set. Edit $ENV_FILE before starting."
            fi
        fi
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Step 3: Generate TLS certificates
# ---------------------------------------------------------------------------

generate_certs() {
    mkdir -p "$NGINX_CERTS_DIR"

    if [ -f "$NGINX_CERTS_DIR/server.crt" ] && [ -f "$NGINX_CERTS_DIR/server.key" ]; then
        ok "TLS certificates already exist"
        return 0
    fi

    info "Generating self-signed TLS certificate..."
    info "(Replace with real certificates for production)"

    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$NGINX_CERTS_DIR/server.key" \
        -out "$NGINX_CERTS_DIR/server.crt" \
        -subj "/CN=localhost/O=Aragora/C=US" \
        2>/dev/null

    chmod 600 "$NGINX_CERTS_DIR/server.key"
    chmod 644 "$NGINX_CERTS_DIR/server.crt"

    ok "Self-signed TLS certificate generated"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 4: Create directories
# ---------------------------------------------------------------------------

create_directories() {
    mkdir -p "$BACKUPS_DIR"
    mkdir -p "$DEPLOY_DIR/nginx/certs"
    ok "Data directories created"
}

# ---------------------------------------------------------------------------
# Step 5: Pull / build images
# ---------------------------------------------------------------------------

pull_or_build() {
    info "Pulling container images..."

    local profile_args=""
    for p in "${PROFILES[@]+"${PROFILES[@]}"}"; do
        profile_args="$profile_args --profile $p"
    done

    # Try pulling first; fall back to building
    if ! docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
        $profile_args pull --ignore-buildable 2>/dev/null; then
        info "Some images not available remotely. Building locally..."
    fi

    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
        $profile_args build --quiet 2>/dev/null || true

    ok "Container images ready"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 6: Start services
# ---------------------------------------------------------------------------

start_services() {
    info "Starting Aragora services..."

    local profile_args=""
    for p in "${PROFILES[@]+"${PROFILES[@]}"}"; do
        profile_args="$profile_args --profile $p"
    done

    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
        $profile_args up -d

    ok "Services started"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 7: Health checks
# ---------------------------------------------------------------------------

wait_for_health() {
    info "Waiting for services to become healthy..."

    local max_wait=120
    local interval=5
    local elapsed=0

    # Wait for PostgreSQL
    echo -n "  PostgreSQL: "
    while [ "$elapsed" -lt "$max_wait" ]; do
        if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
            exec -T postgres pg_isready -U "${POSTGRES_USER:-aragora}" -q 2>/dev/null; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        echo -n "."
    done
    if [ "$elapsed" -ge "$max_wait" ]; then
        echo -e "${RED}timeout${NC}"
        warn "PostgreSQL did not become ready within ${max_wait}s"
    fi

    # Wait for Redis
    elapsed=0
    echo -n "  Redis:      "
    # shellcheck disable=SC1090
    set +u; source "$ENV_FILE" 2>/dev/null || true; set -u
    while [ "$elapsed" -lt "$max_wait" ]; do
        if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
            exec -T redis redis-cli -a "${REDIS_PASSWORD:-}" ping 2>/dev/null | grep -q PONG; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        echo -n "."
    done
    if [ "$elapsed" -ge "$max_wait" ]; then
        echo -e "${RED}timeout${NC}"
        warn "Redis did not become ready within ${max_wait}s"
    fi

    # Wait for Aragora API
    elapsed=0
    echo -n "  Aragora:    "
    while [ "$elapsed" -lt "$max_wait" ]; do
        if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
            exec -T aragora curl -sf http://localhost:8080/healthz >/dev/null 2>&1; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        echo -n "."
    done
    if [ "$elapsed" -ge "$max_wait" ]; then
        echo -e "${RED}timeout${NC}"
        warn "Aragora API did not become ready within ${max_wait}s"
        warn "Check logs: docker compose -f $COMPOSE_FILE logs aragora"
    fi

    # Wait for Nginx
    elapsed=0
    echo -n "  Nginx:      "
    while [ "$elapsed" -lt 30 ]; do
        if curl -sf http://localhost/nginx-health >/dev/null 2>&1; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    if [ "$elapsed" -ge 30 ]; then
        echo -e "${RED}timeout${NC}"
        warn "Nginx did not become ready within 30s"
    fi

    echo ""
}

# ---------------------------------------------------------------------------
# Step 8: Print summary
# ---------------------------------------------------------------------------

print_summary() {
    echo -e "${BOLD}============================================${NC}"
    echo -e "${GREEN}${BOLD}  Aragora is running!${NC}"
    echo -e "${BOLD}============================================${NC}"
    echo ""
    echo -e "${BOLD}Access Points:${NC}"
    echo "  API (HTTPS):    https://localhost"
    echo "  API (HTTP):     http://localhost  (redirects to HTTPS)"
    echo "  Health check:   https://localhost/healthz"
    echo "  WebSocket:      wss://localhost/ws"
    echo ""

    # Check if monitoring profile is active
    for p in "${PROFILES[@]+"${PROFILES[@]}"}"; do
        if [ "$p" = "monitoring" ]; then
            echo "  Grafana:        https://localhost/grafana/"
            echo "                  User: ${GRAFANA_USER:-admin}"
            echo ""
            break
        fi
    done

    echo -e "${BOLD}Useful Commands:${NC}"
    echo "  Status:         docker compose -f $COMPOSE_FILE ps"
    echo "  Logs:           docker compose -f $COMPOSE_FILE logs -f aragora"
    echo "  Stop:           docker compose -f $COMPOSE_FILE down"
    echo "  Restart:        docker compose -f $COMPOSE_FILE restart"
    echo "  Upgrade:        git pull && docker compose -f $COMPOSE_FILE up -d --build"
    echo ""
    echo -e "${BOLD}Configuration:${NC}"
    echo "  Env file:       $ENV_FILE"
    echo "  TLS certs:      $NGINX_CERTS_DIR/"
    echo "  Backups:        $BACKUPS_DIR/"
    echo ""
    echo -e "${YELLOW}Note: Self-signed TLS certificate in use.${NC}"
    echo -e "${YELLOW}Replace with real certificates for production:${NC}"
    echo "  $NGINX_CERTS_DIR/server.crt"
    echo "  $NGINX_CERTS_DIR/server.key"
    echo ""
    echo "Full documentation: docs/deployment/SELF_HOSTED.md"
    echo ""
}

print_config_only_summary() {
    echo -e "${BOLD}============================================${NC}"
    echo -e "${GREEN}${BOLD}  Configuration complete!${NC}"
    echo -e "${BOLD}============================================${NC}"
    echo ""
    echo "  Env file:       $ENV_FILE"
    echo "  TLS certs:      $NGINX_CERTS_DIR/"
    echo "  Compose file:   $COMPOSE_FILE"
    echo ""
    echo -e "${BOLD}To start:${NC}"

    local profile_args=""
    for p in "${PROFILES[@]+"${PROFILES[@]}"}"; do
        profile_args="$profile_args --profile $p"
    done

    echo "  docker compose -f $COMPOSE_FILE --env-file $ENV_FILE $profile_args up -d"
    echo ""
    echo "Full documentation: docs/deployment/SELF_HOSTED.md"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    banner

    if [ "$SKIP_PREREQS" = false ]; then
        check_prerequisites
    fi

    generate_env
    generate_certs
    create_directories

    if [ "$NO_START" = true ]; then
        print_config_only_summary
        exit 0
    fi

    pull_or_build
    start_services
    wait_for_health
    print_summary
}

main "$@"
