#!/usr/bin/env bash
# LiftMode Aragora Setup — Mac Studio M3 Ultra
# One-time bootstrap: configures AWS Secrets Manager, starts services,
# and guides through Gmail OAuth.
#
# All secrets are stored in AWS Secrets Manager, NOT in .env files.
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

SECRET_NAME="${ARAGORA_SECRET_NAME:-liftmode/production}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# ── Prerequisites ──────────────────────────────────────────────────────
echo "=== LiftMode Aragora Setup ==="
echo ""

command -v docker >/dev/null 2>&1 || fail "Docker not installed. Get it at https://docker.com/products/docker-desktop/"
info "Docker found: $(docker --version | head -1)"

docker compose version >/dev/null 2>&1 || fail "Docker Compose not available. Update Docker Desktop."
info "Docker Compose found"

command -v aws >/dev/null 2>&1 || fail "AWS CLI not installed. Run: brew install awscli"
info "AWS CLI found"

# ── AWS Credentials ───────────────────────────────────────────────────
echo ""
echo "Checking AWS credentials..."
if aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")
    info "AWS authenticated — account $ACCOUNT_ID"
else
    fail "AWS credentials not configured. Run: aws configure"
fi

# ── AWS Secrets Manager Setup ─────────────────────────────────────────
echo ""
echo "Checking AWS Secrets Manager..."

if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    info "Secret '$SECRET_NAME' exists in AWS Secrets Manager"
else
    warn "Secret '$SECRET_NAME' not found — creating it now"
    echo ""
    echo "You need the following credentials. Have them ready:"
    echo ""
    echo "  Required:"
    echo "    ANTHROPIC_API_KEY     — https://console.anthropic.com/settings/keys"
    echo "    ARAGORA_API_TOKEN     — generate: openssl rand -hex 32"
    echo "    GMAIL_CLIENT_ID       — https://console.cloud.google.com/apis/credentials"
    echo "    GMAIL_CLIENT_SECRET   — (same page as above)"
    echo ""
    echo "  Optional:"
    echo "    OPENAI_API_KEY        — https://platform.openai.com/api-keys"
    echo "    OPENROUTER_API_KEY    — https://openrouter.ai/keys"
    echo "    SLACK_WEBHOOK_URL     — https://api.slack.com/messaging/webhooks"
    echo ""
    echo "  Shopify (optional):"
    echo "    SHOPIFY_SHOP_DOMAIN   — your-store.myshopify.com"
    echo "    SHOPIFY_ACCESS_TOKEN  — Settings → Apps → Develop apps → Access token"
    echo ""
    echo "  Zendesk (optional):"
    echo "    ZENDESK_SUBDOMAIN     — your-subdomain.zendesk.com"
    echo "    ZENDESK_EMAIL         — admin email address"
    echo "    ZENDESK_API_TOKEN     — Admin → Channels → API"
    echo ""
    echo "  Gmail OAuth setup:"
    echo "    a) Google Cloud Console → APIs & Services → Credentials"
    echo "    b) Create OAuth 2.0 Client ID (type: Web application)"
    echo "    c) Authorized redirect URI:"
    echo "       http://localhost:8080/api/auth/oauth/google/callback"
    echo "    d) Enable Gmail API:"
    echo "       https://console.cloud.google.com/apis/library/gmail.googleapis.com"
    echo ""

    read -rp "ANTHROPIC_API_KEY: " ANTHROPIC_API_KEY
    read -rp "ARAGORA_API_TOKEN (or press Enter to generate): " ARAGORA_API_TOKEN
    if [ -z "$ARAGORA_API_TOKEN" ]; then
        ARAGORA_API_TOKEN=$(openssl rand -hex 32)
        info "Generated ARAGORA_API_TOKEN: ${ARAGORA_API_TOKEN:0:8}..."
    fi
    read -rp "GMAIL_CLIENT_ID: " GMAIL_CLIENT_ID
    read -rsp "GMAIL_CLIENT_SECRET: " GMAIL_CLIENT_SECRET
    echo ""
    read -rp "OPENAI_API_KEY (optional, Enter to skip): " OPENAI_API_KEY
    read -rp "OPENROUTER_API_KEY (optional, Enter to skip): " OPENROUTER_API_KEY
    read -rp "SLACK_WEBHOOK_URL (optional, Enter to skip): " SLACK_WEBHOOK_URL
    echo ""
    read -rp "SHOPIFY_SHOP_DOMAIN (e.g., liftmode.myshopify.com, Enter to skip): " SHOPIFY_SHOP_DOMAIN
    read -rp "SHOPIFY_ACCESS_TOKEN (Enter to skip): " SHOPIFY_ACCESS_TOKEN
    echo ""
    read -rp "ZENDESK_SUBDOMAIN (e.g., liftmode, Enter to skip): " ZENDESK_SUBDOMAIN
    read -rp "ZENDESK_EMAIL (admin email, Enter to skip): " ZENDESK_EMAIL
    read -rp "ZENDESK_API_TOKEN (Enter to skip): " ZENDESK_API_TOKEN

    # Build JSON secret — passes shell vars as args to avoid injection
    SECRET_JSON=$(python3 -c "
import json, sys
pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
secret = {}
for key, val in pairs:
    if val:
        secret[key] = val
# Mirror Gmail creds as Google creds for OAuth
if 'GMAIL_CLIENT_ID' in secret:
    secret['GOOGLE_CLIENT_ID'] = secret['GMAIL_CLIENT_ID']
if 'GMAIL_CLIENT_SECRET' in secret:
    secret['GOOGLE_CLIENT_SECRET'] = secret['GMAIL_CLIENT_SECRET']
print(json.dumps(secret))
" \
        ANTHROPIC_API_KEY "$ANTHROPIC_API_KEY" \
        ARAGORA_API_TOKEN "$ARAGORA_API_TOKEN" \
        GMAIL_CLIENT_ID "$GMAIL_CLIENT_ID" \
        GMAIL_CLIENT_SECRET "$GMAIL_CLIENT_SECRET" \
        OPENAI_API_KEY "$OPENAI_API_KEY" \
        OPENROUTER_API_KEY "$OPENROUTER_API_KEY" \
        SLACK_WEBHOOK_URL "$SLACK_WEBHOOK_URL" \
        SHOPIFY_SHOP_DOMAIN "$SHOPIFY_SHOP_DOMAIN" \
        SHOPIFY_ACCESS_TOKEN "$SHOPIFY_ACCESS_TOKEN" \
        ZENDESK_SUBDOMAIN "$ZENDESK_SUBDOMAIN" \
        ZENDESK_EMAIL "$ZENDESK_EMAIL" \
        ZENDESK_API_TOKEN "$ZENDESK_API_TOKEN" \
    )

    aws secretsmanager create-secret \
        --name "$SECRET_NAME" \
        --description "LiftMode Aragora production secrets" \
        --secret-string "$SECRET_JSON" \
        --region "$AWS_REGION" >/dev/null

    info "Secret '$SECRET_NAME' created in AWS Secrets Manager"
    echo ""
    echo "  To update secrets later:"
    echo "    aws secretsmanager put-secret-value \\"
    echo "      --secret-id $SECRET_NAME \\"
    echo "      --secret-string '\$(cat updated-secrets.json)' \\"
    echo "      --region $AWS_REGION"
fi

# ── Local .env (non-secret config only) ───────────────────────────────
if [ ! -f .env ]; then
    cp .env.template .env
    info "Created .env with non-secret config (AWS region, DB password)"
fi

# ── Start Services ─────────────────────────────────────────────────────
echo ""
echo "Starting services..."
docker compose up -d

echo "Waiting for services to be healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/healthz >/dev/null 2>&1; then
        break
    fi
    sleep 2
    printf "."
done
echo ""

if curl -sf http://localhost:8080/healthz >/dev/null 2>&1; then
    info "Backend healthy at http://localhost:8080"
else
    warn "Backend not responding yet — check: docker compose logs backend"
    echo "Services may still be starting. Wait a minute and retry."
fi

# ── Auto-connect Shopify + Zendesk ────────────────────────────────
if [ -n "$SHOPIFY_SHOP_DOMAIN" ]; then
    curl -sf -X POST http://localhost:8080/api/v1/ecommerce/connect \
      -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"platform\":\"shopify\",\"credentials\":{\"shop_url\":\"https://$SHOPIFY_SHOP_DOMAIN\",\"access_token\":\"$SHOPIFY_ACCESS_TOKEN\"}}" \
    && info "Shopify connected" || warn "Shopify connection failed — connect manually later"
fi

if [ -n "$ZENDESK_SUBDOMAIN" ]; then
    curl -sf -X POST http://localhost:8080/api/v1/support/connect \
      -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"platform\":\"zendesk\",\"credentials\":{\"subdomain\":\"$ZENDESK_SUBDOMAIN\",\"email\":\"$ZENDESK_EMAIL\",\"api_token\":\"$ZENDESK_API_TOKEN\"}}" \
    && info "Zendesk connected" || warn "Zendesk connection failed — connect manually later"
fi

# ── Gmail OAuth ────────────────────────────────────────────────────────
echo ""
echo "=== Gmail OAuth Setup ==="
echo ""
echo "Open this URL in your browser to authorize Gmail access:"
echo ""
echo "  http://localhost:8080/api/v1/gmail/oauth/authorize"
echo ""
echo "After completing the consent flow, start initial sync:"
echo ""
echo "  curl -H 'Authorization: Bearer <your-token>' \\"
echo "       -X POST http://localhost:8080/api/v1/gmail/sync/start"
echo ""
echo "Check sync status:"
echo ""
echo "  curl -H 'Authorization: Bearer <your-token>' \\"
echo "       http://localhost:8080/api/v1/gmail/sync/status"
echo ""

# ── Install Daily Briefing (launchd) ──────────────────────────────────
echo "=== Daily Briefing Setup ==="
echo ""

PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/com.liftmode.aragora-briefing.plist"

mkdir -p "$PLIST_DIR"

cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.liftmode.aragora-briefing</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>${SCRIPT_DIR}/briefing.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>7</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${SCRIPT_DIR}/logs/briefing.log</string>
    <key>StandardErrorPath</key>
    <string>${SCRIPT_DIR}/logs/briefing-error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ARAGORA_URL</key>
        <string>http://localhost:8080</string>
        <key>ARAGORA_USE_SECRETS_MANAGER</key>
        <string>true</string>
        <key>ARAGORA_SECRET_NAME</key>
        <string>${SECRET_NAME}</string>
        <key>AWS_REGION</key>
        <string>${AWS_REGION}</string>
    </dict>
</dict>
</plist>
PLIST

mkdir -p "${SCRIPT_DIR}/logs"
launchctl load "$PLIST_PATH" 2>/dev/null || true
info "Daily briefing scheduled at 7:00 AM via launchd"
echo "  Plist: $PLIST_PATH"
echo "  Logs:  ${SCRIPT_DIR}/logs/briefing.log"
echo "  Test:  python3 briefing.py --dry-run"
echo ""

# ── Quick Reference ────────────────────────────────────────────────────
echo "=== Quick Reference ==="
echo ""
echo "  Start:    cd $SCRIPT_DIR && docker compose up -d"
echo "  Stop:     docker compose down"
echo "  Logs:     docker compose logs -f backend"
echo "  Status:   docker compose ps"
echo "  Briefing: python3 briefing.py --dry-run"
echo ""
echo "  API:      http://localhost:8080"
echo "  Health:   http://localhost:8080/healthz"
echo "  Priority: http://localhost:8080/api/v1/gmail/inbox/priority"
echo "  Orders:   http://localhost:8080/api/v1/ecommerce/shopify/orders"
echo "  Tickets:  http://localhost:8080/api/v1/support/zendesk/tickets"
echo ""
echo "  Secrets:  aws secretsmanager get-secret-value --secret-id $SECRET_NAME --region $AWS_REGION"
echo ""
info "Setup complete"
