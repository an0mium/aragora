#!/bin/bash
# Aragora Self-Hosted Setup Wizard
#
# Interactive setup for deploying Aragora on your infrastructure.
# Supports Simple, SME, and Production profiles.
#
# Usage:
#   ./scripts/aragora-setup.sh           # Interactive mode
#   ./scripts/aragora-setup.sh --dry-run # Preview without changes
#   ./scripts/aragora-setup.sh --profile simple # Non-interactive
#
# Requirements:
#   - Docker 20.10+ and Docker Compose 2.0+
#   - At least one AI provider API key (Anthropic, OpenAI, etc.)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DRY_RUN=false
PROFILE=""
SKIP_DOCKER_CHECK=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            --skip-docker-check)
                SKIP_DOCKER_CHECK=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Aragora Self-Hosted Setup Wizard

Usage:
    $0 [OPTIONS]

Options:
    --dry-run           Preview changes without making them
    --profile PROFILE   Select profile: simple, sme, or production
    --skip-docker-check Skip Docker version check
    -h, --help          Show this help message

Profiles:
    simple      SQLite, single container, minimal dependencies
    sme         PostgreSQL + Redis, ideal for teams up to 50
    production  Full HA setup with monitoring

Examples:
    $0                       # Interactive setup
    $0 --profile simple      # Quick start with SQLite
    $0 --dry-run             # Preview what would be done

Documentation:
    See docs/SELF_HOSTED_QUICKSTART.md for quick start guide
    See docs/SELF_HOSTED_COMPLETE_GUIDE.md for full documentation
EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local errors=0

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker 20.10+ first."
        log_info "Visit: https://docs.docker.com/get-docker/"
        errors=$((errors + 1))
    elif [ "$SKIP_DOCKER_CHECK" = false ]; then
        local docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "0.0.0")
        local major=$(echo "$docker_version" | cut -d. -f1)
        if [ "$major" -lt 20 ]; then
            log_warning "Docker version $docker_version detected. Version 20.10+ recommended."
        else
            log_success "Docker $docker_version found"
        fi
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        if ! command -v docker-compose &> /dev/null; then
            log_error "Docker Compose is not installed. Please install Docker Compose 2.0+ first."
            errors=$((errors + 1))
        else
            log_warning "Legacy docker-compose found. Consider upgrading to Docker Compose v2."
        fi
    else
        local compose_version=$(docker compose version --short 2>/dev/null || echo "0.0.0")
        log_success "Docker Compose $compose_version found"
    fi

    # Check available memory
    if command -v free &> /dev/null; then
        local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$mem_gb" -lt 2 ]; then
            log_warning "Less than 2GB RAM available. Aragora may run slowly."
        else
            log_success "${mem_gb}GB RAM available"
        fi
    fi

    # Check disk space
    local disk_available=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ "$disk_available" -lt 5 ]; then
        log_warning "Less than 5GB disk space available."
    else
        log_success "${disk_available}GB disk space available"
    fi

    if [ $errors -gt 0 ]; then
        log_error "Prerequisites check failed. Please fix the errors above."
        exit 1
    fi

    echo ""
}

# Interactive profile selection
select_profile() {
    if [ -n "$PROFILE" ]; then
        log_info "Using profile: $PROFILE"
        return
    fi

    echo ""
    echo "==========================================="
    echo "   Aragora Deployment Profile Selection"
    echo "==========================================="
    echo ""
    echo "1) Simple   - SQLite, single container (5 min setup)"
    echo "             Best for: Personal use, testing, demos"
    echo ""
    echo "2) SME      - PostgreSQL + Redis + Monitoring (30 min setup)"
    echo "             Best for: Teams of 5-50, production-ready"
    echo ""
    echo "3) Production - Full HA stack with Kubernetes (2+ hours)"
    echo "             Best for: Enterprise, 50+ users, high availability"
    echo ""

    read -p "Select profile [1-3] (default: 1): " choice

    case $choice in
        2)
            PROFILE="sme"
            ;;
        3)
            PROFILE="production"
            ;;
        *)
            PROFILE="simple"
            ;;
    esac

    log_info "Selected profile: $PROFILE"
    echo ""
}

# Generate secrets
generate_secret() {
    openssl rand -base64 32 2>/dev/null || cat /dev/urandom | tr -dc 'a-zA-Z0-9' | head -c 32
}

# Configure environment
configure_environment() {
    log_info "Configuring environment..."

    local env_file="$PROJECT_ROOT/.env"

    # Check if .env already exists
    if [ -f "$env_file" ]; then
        read -p ".env file already exists. Overwrite? [y/N]: " overwrite
        if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
            log_info "Keeping existing .env file"
            return
        fi
    fi

    # Start with example
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would copy .env.example to .env"
    else
        cp "$PROJECT_ROOT/.env.example" "$env_file"
    fi

    # Collect API keys
    echo ""
    echo "==========================================="
    echo "   AI Provider Configuration"
    echo "==========================================="
    echo ""
    echo "At least one API key is required."
    echo ""

    read -p "Anthropic API Key (sk-ant-...): " anthropic_key
    read -p "OpenAI API Key (sk-...): " openai_key
    read -p "OpenRouter API Key (optional): " openrouter_key

    if [ -z "$anthropic_key" ] && [ -z "$openai_key" ]; then
        log_error "At least one API key (Anthropic or OpenAI) is required."
        exit 1
    fi

    # Generate secrets
    local secret_key=$(generate_secret)
    local jwt_secret=$(generate_secret)
    local postgres_password=$(generate_secret | head -c 20)

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would configure .env with:"
        log_info "  - ANTHROPIC_API_KEY: ${anthropic_key:+[SET]}"
        log_info "  - OPENAI_API_KEY: ${openai_key:+[SET]}"
        log_info "  - Generated secrets for security"
    else
        # Update .env file
        sed -i.bak "s/^ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$anthropic_key/" "$env_file"
        sed -i.bak "s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" "$env_file"

        if [ -n "$openrouter_key" ]; then
            sed -i.bak "s/^OPENROUTER_API_KEY=.*/OPENROUTER_API_KEY=$openrouter_key/" "$env_file"
        fi

        # Set secrets
        sed -i.bak "s/^ARAGORA_SECRET_KEY=.*/ARAGORA_SECRET_KEY=$secret_key/" "$env_file"
        sed -i.bak "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$postgres_password/" "$env_file"

        # Add JWT secret if not present
        if ! grep -q "ARAGORA_JWT_SECRET" "$env_file"; then
            echo "ARAGORA_JWT_SECRET=$jwt_secret" >> "$env_file"
        else
            sed -i.bak "s/^ARAGORA_JWT_SECRET=.*/ARAGORA_JWT_SECRET=$jwt_secret/" "$env_file"
        fi

        # Set environment mode
        if [ "$PROFILE" != "simple" ]; then
            sed -i.bak "s/^ARAGORA_ENV=.*/ARAGORA_ENV=production/" "$env_file"
        fi

        # Clean up backup files
        rm -f "$env_file.bak"

        log_success "Environment configured in .env"
    fi

    echo ""
}

# Start services
start_services() {
    local compose_file

    case $PROFILE in
        simple)
            compose_file="docker-compose.simple.yml"
            ;;
        sme)
            compose_file="docker-compose.sme.yml"
            ;;
        production)
            compose_file="docker-compose.production.yml"
            ;;
        *)
            compose_file="docker-compose.yml"
            ;;
    esac

    if [ ! -f "$PROJECT_ROOT/$compose_file" ]; then
        log_warning "Compose file $compose_file not found, using default docker-compose.yml"
        compose_file="docker-compose.yml"
    fi

    log_info "Starting Aragora with $compose_file..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run: docker compose -f $compose_file up -d"
    else
        cd "$PROJECT_ROOT"
        docker compose -f "$compose_file" up -d

        log_info "Waiting for services to start..."
        sleep 10

        # Check health
        local max_attempts=30
        local attempt=1

        while [ $attempt -le $max_attempts ]; do
            if curl -s http://localhost:8080/api/health | grep -q "healthy"; then
                log_success "Aragora is healthy!"
                break
            fi

            log_info "Waiting for Aragora to start... ($attempt/$max_attempts)"
            sleep 5
            attempt=$((attempt + 1))
        done

        if [ $attempt -gt $max_attempts ]; then
            log_warning "Health check timed out. Check logs with: docker compose -f $compose_file logs"
        fi
    fi
}

# Show next steps
show_next_steps() {
    echo ""
    echo "==========================================="
    echo "   Setup Complete!"
    echo "==========================================="
    echo ""
    echo "Aragora is running at: http://localhost:8080"
    echo ""
    echo "Quick verification:"
    echo "  curl http://localhost:8080/api/health"
    echo ""
    echo "Run your first debate:"
    echo "  curl -X POST http://localhost:8080/api/debates \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"topic\": \"What is the best way to learn programming?\"}'"
    echo ""

    if [ "$PROFILE" = "sme" ] || [ "$PROFILE" = "production" ]; then
        echo "Monitoring:"
        echo "  Grafana: http://localhost:3001 (admin/admin)"
        echo "  Prometheus: http://localhost:9090"
        echo ""
    fi

    echo "Documentation:"
    echo "  Quick Start: docs/SELF_HOSTED_QUICKSTART.md"
    echo "  Full Guide:  docs/SELF_HOSTED_COMPLETE_GUIDE.md"
    echo ""
    echo "Useful commands:"
    echo "  docker compose logs -f           # View logs"
    echo "  docker compose ps                # Check status"
    echo "  docker compose down              # Stop services"
    echo ""
}

# Main execution
main() {
    echo ""
    echo "==========================================="
    echo "   Aragora Self-Hosted Setup Wizard"
    echo "==========================================="
    echo ""

    parse_args "$@"

    if [ "$DRY_RUN" = true ]; then
        log_warning "Running in dry-run mode - no changes will be made"
        echo ""
    fi

    cd "$PROJECT_ROOT"

    check_prerequisites
    select_profile
    configure_environment
    start_services
    show_next_steps
}

main "$@"
