#!/bin/bash
# ==============================================================================
# Aragora Self-Hosted Deployment Initialization
# ==============================================================================
# Interactive setup script for deploying Aragora with Docker Compose.
# Target: <30 minute setup time from scratch.
#
# Usage:
#   ./init.sh                  # Interactive mode
#   ./init.sh --minimal        # Minimal setup (Redis only)
#   ./init.sh --production     # Production setup with PostgreSQL
#   ./init.sh --full           # Full stack with monitoring
#   ./init.sh --non-interactive # Use defaults/env vars, no prompts
#
# Environment variables (for non-interactive mode):
#   ANTHROPIC_API_KEY, OPENAI_API_KEY - AI provider keys
#   ARAGORA_ENV - Environment (development/production)
#   ENABLE_POSTGRES - Enable PostgreSQL (true/false)
#   ENABLE_MONITORING - Enable Prometheus/Grafana (true/false)
# ==============================================================================

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOY_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
ENV_FILE="$DEPLOY_DIR/.env"
ENV_EXAMPLE="$DEPLOY_DIR/.env.example"
INTERACTIVE=true
SETUP_MODE="minimal"  # minimal, production, full

# ==============================================================================
# Helper Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}==> $1${NC}"
}

prompt_yn() {
    local prompt="$1"
    local default="${2:-n}"

    if [ "$INTERACTIVE" = false ]; then
        [ "$default" = "y" ] && return 0 || return 1
    fi

    local yn_hint="[y/N]"
    [ "$default" = "y" ] && yn_hint="[Y/n]"

    read -p "$prompt $yn_hint: " response
    response="${response:-$default}"
    [[ "$response" =~ ^[Yy] ]]
}

prompt_value() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"

    if [ "$INTERACTIVE" = false ]; then
        echo "${!var_name:-$default}"
        return
    fi

    local hint=""
    [ -n "$default" ] && hint=" [$default]"

    read -p "$prompt$hint: " value
    echo "${value:-$default}"
}

prompt_secret() {
    local prompt="$1"
    local var_name="$2"

    if [ "$INTERACTIVE" = false ]; then
        echo "${!var_name:-}"
        return
    fi

    read -sp "$prompt: " value
    echo
    echo "$value"
}

generate_secret() {
    python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || \
    openssl rand -base64 32 | tr -d '/+=' | head -c 32
}

generate_fernet_key() {
    python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || \
    echo "GENERATE_MANUALLY"
}

# ==============================================================================
# Prerequisite Checks
# ==============================================================================

check_prerequisites() {
    log_step "Checking prerequisites"

    local missing=()

    # Docker
    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        log_success "Docker $docker_version installed"
    else
        missing+=("docker")
        log_error "Docker not found"
    fi

    # Docker Compose
    if docker compose version &> /dev/null; then
        local compose_version=$(docker compose version --short 2>/dev/null || echo "unknown")
        log_success "Docker Compose $compose_version installed"
    elif command -v docker-compose &> /dev/null; then
        local compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
        log_success "Docker Compose $compose_version installed (legacy)"
    else
        missing+=("docker-compose")
        log_error "Docker Compose not found"
    fi

    # Python (for secret generation)
    if command -v python3 &> /dev/null; then
        log_success "Python3 installed"
    else
        log_warn "Python3 not found - some features may be limited"
    fi

    # curl (for health checks)
    if command -v curl &> /dev/null; then
        log_success "curl installed"
    else
        missing+=("curl")
        log_error "curl not found"
    fi

    # Check Docker daemon
    if docker info &> /dev/null; then
        log_success "Docker daemon running"
    else
        log_error "Docker daemon not running"
        missing+=("docker-daemon")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing prerequisites: ${missing[*]}"
        echo ""
        echo "Installation instructions:"
        echo "  Docker: https://docs.docker.com/get-docker/"
        echo "  Docker Compose: Included with Docker Desktop, or install separately"
        exit 1
    fi
}

# ==============================================================================
# Environment Configuration
# ==============================================================================

setup_environment() {
    log_step "Configuring environment"

    # Create .env from example if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$ENV_EXAMPLE" ]; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            log_success "Created .env from .env.example"
        else
            touch "$ENV_FILE"
            log_warn "Created empty .env file"
        fi
    else
        log_info ".env file already exists"
        if prompt_yn "Reset to defaults?"; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            log_success "Reset .env to defaults"
        fi
    fi

    # Load current values
    set -a
    source "$ENV_FILE" 2>/dev/null || true
    set +a
}

configure_ai_providers() {
    log_step "Configuring AI providers"

    echo ""
    echo "Aragora requires at least one AI provider API key."
    echo "Recommended: Anthropic Claude for best debate quality."
    echo ""

    # Anthropic
    local anthropic_key
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        log_info "Anthropic API key already set"
        if prompt_yn "Update Anthropic API key?"; then
            anthropic_key=$(prompt_secret "Anthropic API key (sk-ant-...)")
        else
            anthropic_key="$ANTHROPIC_API_KEY"
        fi
    else
        anthropic_key=$(prompt_secret "Anthropic API key (sk-ant-..., or press Enter to skip)")
    fi

    # OpenAI
    local openai_key
    if [ -n "$OPENAI_API_KEY" ]; then
        log_info "OpenAI API key already set"
        if prompt_yn "Update OpenAI API key?"; then
            openai_key=$(prompt_secret "OpenAI API key (sk-...)")
        else
            openai_key="$OPENAI_API_KEY"
        fi
    else
        openai_key=$(prompt_secret "OpenAI API key (sk-..., or press Enter to skip)")
    fi

    # Validate at least one key provided
    if [ -z "$anthropic_key" ] && [ -z "$openai_key" ]; then
        log_error "At least one AI provider API key is required"
        exit 1
    fi

    # OpenRouter (recommended fallback)
    local openrouter_key
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo ""
        echo "OpenRouter provides automatic fallback when primary APIs fail."
        echo "Highly recommended for production reliability."
        openrouter_key=$(prompt_secret "OpenRouter API key (optional, press Enter to skip)")
    else
        openrouter_key="$OPENROUTER_API_KEY"
    fi

    # Update .env
    update_env_var "ANTHROPIC_API_KEY" "$anthropic_key"
    update_env_var "OPENAI_API_KEY" "$openai_key"
    update_env_var "OPENROUTER_API_KEY" "$openrouter_key"

    log_success "AI providers configured"
}

configure_security() {
    log_step "Configuring security"

    local env_type
    if [ "$SETUP_MODE" = "production" ] || [ "$SETUP_MODE" = "full" ]; then
        env_type="production"
    else
        env_type=$(prompt_value "Environment (development/production)" "development" "ARAGORA_ENV")
    fi

    update_env_var "ARAGORA_ENV" "$env_type"

    if [ "$env_type" = "production" ]; then
        log_info "Generating production security secrets..."

        # JWT Secret
        if [ -z "$ARAGORA_JWT_SECRET" ] || prompt_yn "Regenerate JWT secret?"; then
            local jwt_secret=$(generate_secret)
            update_env_var "ARAGORA_JWT_SECRET" "$jwt_secret"
            log_success "Generated JWT secret"
        fi

        # Encryption key
        if [ -z "$ARAGORA_ENCRYPTION_KEY" ] || prompt_yn "Regenerate encryption key?"; then
            local enc_key=$(generate_fernet_key)
            if [ "$enc_key" = "GENERATE_MANUALLY" ]; then
                log_warn "Could not generate Fernet key - install cryptography package"
                log_info "Generate manually: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            else
                update_env_var "ARAGORA_ENCRYPTION_KEY" "$enc_key"
                log_success "Generated encryption key"
            fi
        fi

        # API Token
        if [ -z "$ARAGORA_API_TOKEN" ]; then
            local api_token=$(generate_secret)
            update_env_var "ARAGORA_API_TOKEN" "$api_token"
            log_success "Generated API token"
            echo ""
            echo -e "${YELLOW}IMPORTANT: Save this API token for client authentication:${NC}"
            echo -e "${BOLD}$api_token${NC}"
            echo ""
        fi
    else
        # Development defaults
        if [ -z "$ARAGORA_API_TOKEN" ]; then
            update_env_var "ARAGORA_API_TOKEN" "dev-token-$(date +%s)"
        fi
        log_info "Using development security settings"
    fi
}

configure_database() {
    log_step "Configuring database"

    local use_postgres=false

    if [ "$SETUP_MODE" = "production" ] || [ "$SETUP_MODE" = "full" ]; then
        use_postgres=true
    elif [ "$SETUP_MODE" = "minimal" ]; then
        use_postgres=false
    else
        echo ""
        echo "Database options:"
        echo "  1. SQLite (development) - Simple, no setup required"
        echo "  2. PostgreSQL (production) - Recommended for reliability"
        echo ""
        if prompt_yn "Use PostgreSQL?" "n"; then
            use_postgres=true
        fi
    fi

    if [ "$use_postgres" = true ]; then
        local pg_password
        if [ -z "$POSTGRES_PASSWORD" ] || [ "$POSTGRES_PASSWORD" = "aragora_secret" ]; then
            pg_password=$(generate_secret | head -c 16)
        else
            pg_password="$POSTGRES_PASSWORD"
        fi

        update_env_var "POSTGRES_PASSWORD" "$pg_password"
        update_env_var "DATABASE_URL" "postgresql://aragora:${pg_password}@postgres:5432/aragora"
        update_env_var "ARAGORA_DB_BACKEND" "postgres"

        DOCKER_PROFILES+=("postgres")
        log_success "PostgreSQL configured"
    else
        update_env_var "ARAGORA_DB_BACKEND" "sqlite"
        log_info "Using SQLite (development mode)"
    fi
}

configure_monitoring() {
    log_step "Configuring monitoring"

    local enable_monitoring=false

    if [ "$SETUP_MODE" = "full" ]; then
        enable_monitoring=true
    elif [ "$SETUP_MODE" = "minimal" ]; then
        enable_monitoring=false
    else
        echo ""
        echo "Monitoring stack includes:"
        echo "  - Prometheus (metrics collection)"
        echo "  - Grafana (dashboards and alerting)"
        echo "  - Jaeger (distributed tracing)"
        echo ""
        if prompt_yn "Enable monitoring stack?" "n"; then
            enable_monitoring=true
        fi
    fi

    if [ "$enable_monitoring" = true ]; then
        # Generate Grafana password
        if [ -z "$GRAFANA_PASSWORD" ] || [ "$GRAFANA_PASSWORD" = "admin" ]; then
            local grafana_pw=$(generate_secret | head -c 12)
            update_env_var "GRAFANA_PASSWORD" "$grafana_pw"
            echo ""
            echo -e "${YELLOW}Grafana admin password:${NC} ${BOLD}$grafana_pw${NC}"
        fi

        update_env_var "METRICS_ENABLED" "true"
        update_env_var "OTEL_ENABLED" "true"

        DOCKER_PROFILES+=("monitoring")
        log_success "Monitoring stack configured"
    else
        update_env_var "METRICS_ENABLED" "true"  # Still enable basic metrics
        log_info "Basic metrics enabled (no full monitoring stack)"
    fi
}

update_env_var() {
    local key="$1"
    local value="$2"

    if [ -z "$value" ]; then
        return
    fi

    # Escape special characters in value for sed
    local escaped_value=$(printf '%s\n' "$value" | sed -e 's/[\/&]/\\&/g')

    if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        # Update existing value
        sed -i.bak "s/^${key}=.*/${key}=${escaped_value}/" "$ENV_FILE"
        rm -f "$ENV_FILE.bak"
    else
        # Add new value
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

# ==============================================================================
# Docker Operations
# ==============================================================================

DOCKER_PROFILES=()

build_and_start() {
    log_step "Building and starting Aragora"

    cd "$DEPLOY_DIR"

    # Build profile arguments
    local profile_args=""
    for profile in "${DOCKER_PROFILES[@]}"; do
        profile_args="$profile_args --profile $profile"
    done

    # Build images
    log_info "Building Docker images (this may take a few minutes)..."
    if docker compose $profile_args build; then
        log_success "Images built successfully"
    else
        log_error "Failed to build images"
        exit 1
    fi

    # Start services
    log_info "Starting services..."
    if docker compose $profile_args up -d; then
        log_success "Services started"
    else
        log_error "Failed to start services"
        exit 1
    fi

    # Wait for health
    log_info "Waiting for services to be healthy..."
    local max_wait=120
    local waited=0

    while [ $waited -lt $max_wait ]; do
        if curl -sf http://localhost:8080/api/health > /dev/null 2>&1; then
            log_success "Backend is healthy"
            break
        fi
        sleep 2
        waited=$((waited + 2))
        printf "."
    done
    echo ""

    if [ $waited -ge $max_wait ]; then
        log_warn "Backend health check timed out - checking logs..."
        docker compose logs backend --tail=50
    fi
}

run_smoke_test() {
    log_step "Running smoke tests"

    if [ -x "$SCRIPT_DIR/smoke-test.sh" ]; then
        # Load API token for smoke test
        source "$ENV_FILE" 2>/dev/null || true
        export ARAGORA_API_TOKEN

        if "$SCRIPT_DIR/smoke-test.sh" http://localhost:8080; then
            log_success "All smoke tests passed"
        else
            log_warn "Some smoke tests failed - check output above"
        fi
    else
        log_warn "Smoke test script not found or not executable"
        # Basic health check
        if curl -sf http://localhost:8080/api/health > /dev/null; then
            log_success "Basic health check passed"
        else
            log_error "Basic health check failed"
        fi
    fi
}

# ==============================================================================
# Summary
# ==============================================================================

print_summary() {
    log_step "Setup complete!"

    echo ""
    echo "========================================"
    echo "         Aragora is running!"
    echo "========================================"
    echo ""
    echo "Access points:"
    echo "  - Web UI:      http://localhost:3000"
    echo "  - API:         http://localhost:8080"
    echo "  - WebSocket:   ws://localhost:8765/ws"
    echo ""

    if [[ " ${DOCKER_PROFILES[*]} " =~ " monitoring " ]]; then
        echo "Monitoring:"
        echo "  - Grafana:     http://localhost:3001 (admin / $GRAFANA_PASSWORD)"
        echo "  - Prometheus:  http://localhost:9090"
        echo "  - Jaeger:      http://localhost:16686"
        echo ""
    fi

    echo "Useful commands:"
    echo "  docker compose logs -f              # View logs"
    echo "  docker compose ps                   # Check status"
    echo "  docker compose down                 # Stop services"
    echo "  ./scripts/smoke-test.sh            # Re-run tests"
    echo ""
    echo "Documentation:"
    echo "  - Quick start: deploy/QUICK_SETUP.md"
    echo "  - Full docs:   docs/ENVIRONMENT.md"
    echo ""

    # API token reminder
    source "$ENV_FILE" 2>/dev/null || true
    if [ -n "$ARAGORA_API_TOKEN" ]; then
        echo -e "${YELLOW}API Token for client authentication:${NC}"
        echo -e "${BOLD}$ARAGORA_API_TOKEN${NC}"
        echo ""
    fi
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo ""
    echo -e "${CYAN}${BOLD}"
    echo "  █████╗ ██████╗  █████╗  ██████╗  ██████╗ ██████╗  █████╗ "
    echo " ██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗██╔══██╗██╔══██╗"
    echo " ███████║██████╔╝███████║██║  ███╗██║   ██║██████╔╝███████║"
    echo " ██╔══██║██╔══██╗██╔══██║██║   ██║██║   ██║██╔══██╗██╔══██║"
    echo " ██║  ██║██║  ██║██║  ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║"
    echo " ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝"
    echo -e "${NC}"
    echo "                Self-Hosted Deployment Setup"
    echo ""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --minimal)
                SETUP_MODE="minimal"
                shift
                ;;
            --production)
                SETUP_MODE="production"
                shift
                ;;
            --full)
                SETUP_MODE="full"
                shift
                ;;
            --non-interactive)
                INTERACTIVE=false
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --minimal          Minimal setup (Redis only)"
                echo "  --production       Production setup with PostgreSQL"
                echo "  --full             Full stack with monitoring"
                echo "  --non-interactive  Use defaults/env vars, no prompts"
                echo "  -h, --help         Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    log_info "Setup mode: $SETUP_MODE"

    check_prerequisites
    setup_environment
    configure_ai_providers
    configure_security
    configure_database
    configure_monitoring
    build_and_start
    run_smoke_test
    print_summary
}

main "$@"
