#!/usr/bin/env bash
# Rotate Slack refresh token securely.
# Exchanges the current refresh token, saves the new one to .env and AWS Secrets Manager.
# The old refresh token is invalidated by the exchange.
#
# Usage: ./scripts/rotate_slack_token.sh
#
# Must be run from an IP in the Slack app's allowlist (EC2), or locally
# if IP restrictions are removed. Token exchange itself works from any IP;
# only API calls are restricted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE" >&2
    exit 1
fi

# Read current values
REFRESH=$(grep "^SLACK_REFRESH_TOKEN=" "$ENV_FILE" | cut -d= -f2-)
CLIENT_ID=$(grep "^SLACK_CLIENT_ID=" "$ENV_FILE" | cut -d= -f2-)
CLIENT_SECRET=$(grep "^SLACK_CLIENT_SECRET=" "$ENV_FILE" | cut -d= -f2-)

if [[ -z "$REFRESH" || -z "$CLIENT_ID" || -z "$CLIENT_SECRET" ]]; then
    echo "ERROR: Missing SLACK_REFRESH_TOKEN, SLACK_CLIENT_ID, or SLACK_CLIENT_SECRET in .env" >&2
    exit 1
fi

echo "Exchanging refresh token..."

EXCHANGE=$(curl -s -X POST "https://slack.com/api/oauth.v2.access" \
    -d "client_id=${CLIENT_ID}" \
    -d "client_secret=${CLIENT_SECRET}" \
    -d "grant_type=refresh_token" \
    -d "refresh_token=${REFRESH}")

OK=$(echo "$EXCHANGE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ok',''))")
if [[ "$OK" != "True" ]]; then
    ERROR=$(echo "$EXCHANGE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('error','unknown'))")
    echo "ERROR: Token exchange failed: $ERROR" >&2
    exit 1
fi

NEW_ACCESS=$(echo "$EXCHANGE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('access_token',''))")
NEW_REFRESH=$(echo "$EXCHANGE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('refresh_token',''))")
EXPIRES_IN=$(echo "$EXCHANGE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('expires_in',0))")

# Update .env
sed -i '' "s|^SLACK_REFRESH_TOKEN=.*|SLACK_REFRESH_TOKEN=${NEW_REFRESH}|" "$ENV_FILE"
sed -i '' "s|^SLACK_BOT_TOKEN=.*|SLACK_BOT_TOKEN=${NEW_ACCESS}|" "$ENV_FILE"

echo "Updated .env"

# Update AWS Secrets Manager
if command -v aws >/dev/null 2>&1; then
    python3 - "$NEW_REFRESH" "$NEW_ACCESS" << 'PY'
import json, subprocess, sys

new_refresh, new_access = sys.argv[1], sys.argv[2]
result = subprocess.run(
    ["aws", "secretsmanager", "get-secret-value", "--secret-id", "aragora/production",
     "--query", "SecretString", "--output", "text"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("WARNING: Could not read AWS Secrets Manager", file=sys.stderr)
    sys.exit(0)

secrets = json.loads(result.stdout)
secrets["SLACK_REFRESH_TOKEN"] = new_refresh
secrets["SLACK_BOT_TOKEN"] = new_access
subprocess.run(
    ["aws", "secretsmanager", "put-secret-value", "--secret-id", "aragora/production",
     "--secret-string", json.dumps(secrets)],
    capture_output=True, text=True
)
print("Updated AWS Secrets Manager")
PY
fi

echo "Token rotated successfully."
echo "  Access token expires in: ${EXPIRES_IN}s"
echo "  Old refresh token is now invalid."
echo ""
echo "NOTE: The signing secret must be regenerated manually in the Slack app console."
