#!/bin/bash
# Aragora Self-Hosted Setup Script
# Usage: ./init.sh [--verify] [--validate]
#
# Options:
#   --verify    Run smoke test after setup completes
#   --validate  Validate configuration after generation
#   --help      Show this help message

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
RUN_VERIFY=false
RUN_VALIDATE=false
for arg in "$@"; do
    case $arg in
        --verify)
            RUN_VERIFY=true
            shift
            ;;
        --validate)
            RUN_VALIDATE=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./init.sh [--verify] [--validate]"
            echo ""
            echo "Options:"
            echo "  --verify    Run smoke test after setup completes"
            echo "  --validate  Validate configuration after generation"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

echo "================================"
echo "  Aragora Self-Hosted Setup"
echo "================================"
echo

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Error: docker is required but not installed."; exit 1; }
command -v docker compose >/dev/null 2>&1 || { echo "Error: docker compose v2 is required."; exit 1; }

# Create .env if not exists
if [ ! -f .env ]; then
    echo "Creating configuration file..."
    cp .env.example .env

    # Generate secrets
    POSTGRES_PASS=$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 24)
    JWT_SECRET=$(openssl rand -base64 32)
    REDIS_PASS=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
    GRAFANA_PASS=$(openssl rand -base64 12 | tr -dc 'a-zA-Z0-9' | head -c 12)

    # Update .env with generated secrets
    sed -i.bak "s/POSTGRES_PASSWORD=CHANGE_ME_STRONG_PASSWORD/POSTGRES_PASSWORD=$POSTGRES_PASS/" .env
    sed -i.bak "s/ARAGORA_JWT_SECRET=CHANGE_ME_GENERATE_RANDOM_SECRET/ARAGORA_JWT_SECRET=$JWT_SECRET/" .env
    sed -i.bak "s/REDIS_PASSWORD=CHANGE_ME_REDIS_PASSWORD/REDIS_PASSWORD=$REDIS_PASS/" .env
    sed -i.bak "s/GRAFANA_PASSWORD=CHANGE_ME_GRAFANA_PASSWORD/GRAFANA_PASSWORD=$GRAFANA_PASS/" .env
    rm -f .env.bak

    echo "Generated secure passwords in .env"
    echo
else
    echo "Using existing .env configuration"
fi

# Prompt for API key if not set
source .env 2>/dev/null || true
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "No AI provider API key found."
    echo -n "Enter your Anthropic API key (or press Enter to skip): "
    read -r api_key
    if [ -n "$api_key" ]; then
        sed -i.bak "s/ANTHROPIC_API_KEY=sk-ant-.../ANTHROPIC_API_KEY=$api_key/" .env
        rm -f .env.bak
        echo "API key saved."
    else
        echo "Warning: No API key set. You'll need to add one to .env before starting."
    fi
fi

echo
echo -e "${GREEN}Setup complete!${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review configuration:     nano .env"
echo "  2. Validate configuration:   ./validate_env.sh"
echo "  3. Start services:           docker compose up -d"
echo "  4. Check status:             docker compose ps"
echo "  5. Verify deployment:        ./smoke_test.sh"
echo
echo -e "${BLUE}Optional profiles:${NC}"
echo "  docker compose --profile monitoring up -d  # Add Grafana/Prometheus"
echo "  docker compose --profile workers up -d     # Add queue workers"
echo "  docker compose --profile backup up -d      # Enable daily backups"
echo
echo -e "${BLUE}Quick start (all services):${NC}"
echo "  docker compose --profile monitoring --profile workers up -d"
echo
echo -e "${BLUE}Access points after startup:${NC}"
echo "  API:        http://localhost:\${ARAGORA_PORT:-8080}"
echo "  Health:     http://localhost:\${ARAGORA_PORT:-8080}/healthz"
echo "  Docs:       http://localhost:\${ARAGORA_PORT:-8080}/docs"
echo "  Grafana:    http://localhost:3000 (if monitoring profile enabled)"
echo
echo -e "${YELLOW}For production: See QUICK_TLS_SETUP.md for HTTPS configuration${NC}"
echo

# Run validation if requested
if [ "$RUN_VALIDATE" = true ]; then
    echo -e "${BLUE}Running configuration validation...${NC}"
    echo
    if [ -f "./validate_env.sh" ]; then
        chmod +x ./validate_env.sh
        ./validate_env.sh || true
    else
        echo "Warning: validate_env.sh not found"
    fi
fi

# Run verification if requested
if [ "$RUN_VERIFY" = true ]; then
    echo "Starting services and running verification..."
    echo

    # Start core services
    docker compose up -d

    # Wait for services to be ready
    echo "Waiting for services to start..."
    sleep 10

    # Run smoke test
    if [ -f "./smoke_test.sh" ]; then
        ./smoke_test.sh
    else
        echo "Warning: smoke_test.sh not found"
    fi
fi
