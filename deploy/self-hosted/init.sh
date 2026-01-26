#!/bin/bash
# Aragora Self-Hosted Setup Script
# Usage: ./init.sh

set -e

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
echo "Setup complete!"
echo
echo "Next steps:"
echo "  1. Review configuration: nano .env"
echo "  2. Start services: docker compose up -d"
echo "  3. Check status: docker compose ps"
echo "  4. View logs: docker compose logs -f aragora"
echo
echo "Optional profiles:"
echo "  docker compose --profile monitoring up -d  # Add Grafana/Prometheus"
echo "  docker compose --profile workers up -d     # Add queue workers"
echo "  docker compose --profile backup up -d      # Enable daily backups"
echo
