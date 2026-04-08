#!/usr/bin/env bash
#
# deploy-prod.sh — Deploy Aragora to production EC2 instances via AWS SSM
#
# Usage:
#   bash scripts/deploy-prod.sh              # Deploy to both instances
#   bash scripts/deploy-prod.sh --dry-run    # Show commands without executing
#
# Requires: aws CLI with SSM permissions configured.

set -euo pipefail

# Production EC2 instance IDs
INSTANCES=("i-0aae2ccd2f68b94d2" "i-092c2d3b4dafc1f24")

# Configuration
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10
SERVICE_NAME="aragora"
DEPLOY_BRANCH="main"
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[deploy]${NC} $*"; }
error() { echo -e "${RED}[deploy]${NC} $*" >&2; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--help]"
            echo ""
            echo "Deploy Aragora to production EC2 instances via AWS SSM."
            echo ""
            echo "Options:"
            echo "  --dry-run   Show commands without executing"
            echo "  --help, -h  Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify AWS CLI is available
if ! command -v aws &>/dev/null; then
    error "aws CLI not found. Install it: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    exit 1
fi

# Deploy commands to run on each instance
DEPLOY_COMMANDS=$(cat <<'CMDS'
set -e
cd /opt/aragora || { echo "ERROR: /opt/aragora not found"; exit 1; }
echo "=== Pulling latest main ==="
git fetch origin main
git checkout main
git reset --hard origin/main
echo "=== Installing dependencies ==="
pip install -q -e . 2>/dev/null || true
echo "=== Restarting aragora service ==="
systemctl restart aragora
echo "=== Waiting for service to start ==="
sleep 5
systemctl is-active aragora
echo "=== Deploy complete ==="
CMDS
)

# Send SSM command to an instance and wait for result
send_deploy_command() {
    local instance_id="$1"

    log "Deploying to ${instance_id}..."

    if [[ "$DRY_RUN" == "true" ]]; then
        warn "[DRY RUN] Would send SSM command to ${instance_id}:"
        echo "$DEPLOY_COMMANDS"
        return 0
    fi

    local command_id
    command_id=$(aws ssm send-command \
        --instance-ids "$instance_id" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[$(echo "$DEPLOY_COMMANDS" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')]" \
        --timeout-seconds 300 \
        --output text \
        --query "Command.CommandId")

    if [[ -z "$command_id" ]]; then
        error "Failed to send command to ${instance_id}"
        return 1
    fi

    log "Command ${command_id} sent to ${instance_id}, waiting for completion..."

    # Wait for the command to complete
    if ! aws ssm wait command-executed \
        --command-id "$command_id" \
        --instance-id "$instance_id" 2>/dev/null; then
        # wait may fail if command failed — check status manually
        true
    fi

    local status
    status=$(aws ssm get-command-invocation \
        --command-id "$command_id" \
        --instance-id "$instance_id" \
        --query "Status" \
        --output text)

    local stdout
    stdout=$(aws ssm get-command-invocation \
        --command-id "$command_id" \
        --instance-id "$instance_id" \
        --query "StandardOutputContent" \
        --output text)

    local stderr
    stderr=$(aws ssm get-command-invocation \
        --command-id "$command_id" \
        --instance-id "$instance_id" \
        --query "StandardErrorContent" \
        --output text)

    if [[ "$status" == "Success" ]]; then
        log "Instance ${instance_id}: deploy succeeded"
        echo "$stdout"
    else
        error "Instance ${instance_id}: deploy failed (status: ${status})"
        echo "$stdout"
        [[ -n "$stderr" ]] && echo "$stderr" >&2
        return 1
    fi
}

# Wait for health check on an instance
wait_for_health() {
    local instance_id="$1"
    local attempt=0

    log "Checking health on ${instance_id}..."

    if [[ "$DRY_RUN" == "true" ]]; then
        warn "[DRY RUN] Would check health on ${instance_id}"
        return 0
    fi

    while (( attempt < HEALTH_CHECK_RETRIES )); do
        attempt=$((attempt + 1))

        local command_id
        command_id=$(aws ssm send-command \
            --instance-ids "$instance_id" \
            --document-name "AWS-RunShellScript" \
            --parameters 'commands=["systemctl is-active aragora && curl -sf http://localhost:8080/health || exit 1"]' \
            --timeout-seconds 30 \
            --output text \
            --query "Command.CommandId" 2>/dev/null) || true

        if [[ -z "$command_id" ]]; then
            warn "Health check attempt ${attempt}/${HEALTH_CHECK_RETRIES} — failed to send command"
            sleep "$HEALTH_CHECK_INTERVAL"
            continue
        fi

        # Wait briefly for the health check command
        sleep 5

        local status
        status=$(aws ssm get-command-invocation \
            --command-id "$command_id" \
            --instance-id "$instance_id" \
            --query "Status" \
            --output text 2>/dev/null) || true

        if [[ "$status" == "Success" ]]; then
            log "Instance ${instance_id}: healthy (attempt ${attempt})"
            return 0
        fi

        warn "Health check attempt ${attempt}/${HEALTH_CHECK_RETRIES} — not yet healthy"
        sleep "$HEALTH_CHECK_INTERVAL"
    done

    error "Instance ${instance_id}: health check failed after ${HEALTH_CHECK_RETRIES} attempts"
    return 1
}

# Main deployment flow
main() {
    log "Starting production deployment to ${#INSTANCES[@]} instances"
    log "Branch: ${DEPLOY_BRANCH}"
    echo ""

    local failed=0

    # Deploy to each instance
    for instance in "${INSTANCES[@]}"; do
        if ! send_deploy_command "$instance"; then
            error "Deployment failed on ${instance}"
            failed=$((failed + 1))
        fi
        echo ""
    done

    if (( failed > 0 )); then
        error "${failed} instance(s) failed deployment — aborting health checks"
        exit 1
    fi

    # Health checks
    log "Running health checks..."
    echo ""

    for instance in "${INSTANCES[@]}"; do
        if ! wait_for_health "$instance"; then
            failed=$((failed + 1))
        fi
    done

    echo ""
    if (( failed > 0 )); then
        error "Deployment completed with ${failed} unhealthy instance(s)"
        exit 1
    fi

    log "All ${#INSTANCES[@]} instances deployed and healthy"
}

main
