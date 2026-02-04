#!/usr/bin/env bash
#
# Aragora One-Command Deploy Script
#
# Usage:
#   ./scripts/aragora-deploy.sh                    # Interactive mode
#   ./scripts/aragora-deploy.sh --profile simple   # Quick start
#   ./scripts/aragora-deploy.sh --profile sme      # SME deployment
#   ./scripts/aragora-deploy.sh --profile production --setup
#
# This script wraps the 'aragora deploy start' CLI command, providing
# a simple entry point for deploying Aragora with Docker Compose.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROFILE="simple"
RUN_SETUP=false
DRY_RUN=false
NO_WAIT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile|-p)
            PROFILE="$2"
            shift 2
            ;;
        --setup|-s)
            RUN_SETUP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-wait)
            NO_WAIT=true
            shift
            ;;
        --help|-h)
            echo "Aragora One-Command Deploy"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --profile, -p <profile>  Deployment profile: simple, sme, production (default: simple)"
            echo "  --setup, -s              Run interactive setup before starting"
            echo "  --dry-run                Show what would be done without executing"
            echo "  --no-wait                Don't wait for health checks"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Profiles:"
            echo "  simple      Minimal setup with SQLite (5 min, 1 container)"
            echo "  sme         PostgreSQL + Redis + Monitoring (30 min, 5-50 users)"
            echo "  production  Full HA setup with Traefik TLS (2+ hours, enterprise)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Quick start with simple profile"
            echo "  $0 --profile sme --setup     # SME with interactive setup"
            echo "  $0 --profile production      # Production deployment"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate profile
if [[ ! "$PROFILE" =~ ^(simple|sme|production|dev)$ ]]; then
    echo -e "${RED}Invalid profile: $PROFILE${NC}"
    echo "Valid profiles: simple, sme, production, dev"
    exit 1
fi

# Print banner
echo -e "${BLUE}"
echo "=============================================="
echo "   Aragora One-Command Deploy"
echo "   Profile: $PROFILE"
echo "=============================================="
echo -e "${NC}"

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]] && [[ ! -f "$PROJECT_ROOT/docker-compose.simple.yml" ]]; then
    echo -e "${RED}Error: Could not find docker-compose files${NC}"
    echo "Make sure you're running from the Aragora project directory"
    exit 1
fi

cd "$PROJECT_ROOT"

# Check if aragora CLI is available
if command -v aragora &> /dev/null; then
    # Use the CLI
    DEPLOY_CMD="aragora deploy start --profile $PROFILE"
elif [[ -f "$PROJECT_ROOT/aragora/cli/main.py" ]]; then
    # Use Python module directly
    DEPLOY_CMD="python -m aragora.cli.main deploy start --profile $PROFILE"
else
    # Fallback to direct docker compose
    echo -e "${YELLOW}Aragora CLI not found, using direct Docker Compose...${NC}"

    COMPOSE_FILE="docker-compose.simple.yml"
    case $PROFILE in
        sme) COMPOSE_FILE="docker-compose.sme.yml" ;;
        production) COMPOSE_FILE="docker-compose.production.yml" ;;
        dev) COMPOSE_FILE="docker-compose.dev.yml" ;;
    esac

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Would run: docker compose -f $COMPOSE_FILE up -d"
        exit 0
    fi

    echo "Starting services..."
    docker compose -f "$COMPOSE_FILE" up -d

    echo -e "\n${GREEN}Services started!${NC}"
    echo "Health check: curl http://localhost:8080/api/health"
    exit 0
fi

# Build command arguments
if [[ "$RUN_SETUP" == "true" ]]; then
    DEPLOY_CMD="$DEPLOY_CMD --setup"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    DEPLOY_CMD="$DEPLOY_CMD --dry-run"
fi

if [[ "$NO_WAIT" == "true" ]]; then
    DEPLOY_CMD="$DEPLOY_CMD --no-wait"
fi

# Execute
echo -e "${BLUE}Running: $DEPLOY_CMD${NC}\n"
eval "$DEPLOY_CMD"
