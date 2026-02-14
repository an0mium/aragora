#!/usr/bin/env bash
# LiftMode Aragora Setup — Mac Studio M3 Ultra
# One-time bootstrap: checks prerequisites, configures .env, starts services,
# and guides through Gmail OAuth.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!!]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ── Prerequisites ──────────────────────────────────────────────────────
echo "=== LiftMode Aragora Setup ==="
echo ""

command -v docker >/dev/null 2>&1 || fail "Docker not installed. Get it at https://docker.com/products/docker-desktop/"
info "Docker found: $(docker --version | head -1)"

docker compose version >/dev/null 2>&1 || fail "Docker Compose not available. Update Docker Desktop."
info "Docker Compose found"

# ── Environment ────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.template .env
    warn "Created .env from template — you need to fill in credentials"
    echo ""
    echo "Required credentials:"
    echo "  1. ANTHROPIC_API_KEY  — https://console.anthropic.com/settings/keys"
    echo "  2. ARAGORA_API_TOKEN  — run: openssl rand -hex 32"
    echo "  3. GMAIL_CLIENT_ID    — https://console.cloud.google.com/apis/credentials"
    echo "  4. GMAIL_CLIENT_SECRET"
    echo ""
    echo "Gmail OAuth setup:"
    echo "  a) Go to Google Cloud Console → APIs & Services → Credentials"
    echo "  b) Create OAuth 2.0 Client ID (type: Web application)"
    echo "  c) Add authorized redirect URI:"
    echo "     http://localhost:8080/api/auth/oauth/google/callback"
    echo "  d) Enable Gmail API:"
    echo "     https://console.cloud.google.com/apis/library/gmail.googleapis.com"
    echo ""
    read -rp "Press Enter after editing .env, or Ctrl-C to abort..."
fi

# Validate required vars
source .env 2>/dev/null || true
[ -n "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY not set in .env"
[ -n "${ARAGORA_API_TOKEN:-}" ] || fail "ARAGORA_API_TOKEN not set in .env"
[ -n "${GMAIL_CLIENT_ID:-}" ]   || fail "GMAIL_CLIENT_ID not set in .env"
[ -n "${GMAIL_CLIENT_SECRET:-}" ] || fail "GMAIL_CLIENT_SECRET not set in .env"
info "Environment variables validated"

# ── Start Services ─────────────────────────────────────────────────────
echo ""
echo "Starting services..."
docker compose up -d

echo "Waiting for services to be healthy..."
for i in $(seq 1 30); do
    if docker compose ps --format json 2>/dev/null | grep -q '"healthy"' || \
       curl -sf http://localhost:8080/healthz >/dev/null 2>&1; then
        break
    fi
    sleep 2
    printf "."
done
echo ""

# Health check
if curl -sf http://localhost:8080/healthz >/dev/null 2>&1; then
    info "Backend healthy at http://localhost:8080"
else
    warn "Backend not responding yet — check: docker compose logs backend"
    echo "Services may still be starting. Wait a minute and run:"
    echo "  curl http://localhost:8080/healthz"
fi

# ── Gmail OAuth ────────────────────────────────────────────────────────
echo ""
echo "=== Gmail OAuth Setup ==="
echo ""
echo "Open this URL in your browser to authorize Gmail access:"
echo ""
echo "  http://localhost:8080/api/v2/gmail/oauth/authorize"
echo ""
echo "After completing the OAuth consent flow, start the initial sync:"
echo ""
echo "  curl -H 'Authorization: Bearer ${ARAGORA_API_TOKEN}' \\"
echo "       -X POST http://localhost:8080/api/v2/gmail/sync/start"
echo ""
echo "Check sync status:"
echo ""
echo "  curl -H 'Authorization: Bearer ${ARAGORA_API_TOKEN}' \\"
echo "       http://localhost:8080/api/v2/gmail/sync/status"
echo ""

# ── Verify ─────────────────────────────────────────────────────────────
echo "=== Quick Reference ==="
echo ""
echo "  Start:    cd $SCRIPT_DIR && docker compose up -d"
echo "  Stop:     docker compose down"
echo "  Logs:     docker compose logs -f backend"
echo "  Status:   docker compose ps"
echo "  Briefing: python briefing.py --dry-run"
echo ""
echo "  API:      http://localhost:8080"
echo "  Health:   http://localhost:8080/healthz"
echo "  Priority: http://localhost:8080/api/v1/gmail/inbox/priority"
echo ""
info "Setup complete"
