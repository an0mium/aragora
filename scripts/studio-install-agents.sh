#!/usr/bin/env bash
# Install Aragora autonomy LaunchAgents on the Mac Studio from this machine.

set -euo pipefail

STUDIO_HOST="${STUDIO_HOST:-10.0.0.62}"
STUDIO_USER="${STUDIO_USER:-armand}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_mac_studio}"
SSH="SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes -i $SSH_KEY $STUDIO_USER@$STUDIO_HOST"
REMOTE_REPO_PATH=""
SKIP_PREFLIGHT=false

usage() {
    cat <<'EOF'
Usage: ./scripts/studio-install-agents.sh [options]

Options:
  --host <host>          Mac Studio hostname or IP (default: 10.0.0.62)
  --user <user>          SSH username on the Mac Studio (default: armand)
  --ssh-key <path>       SSH private key (default: ~/.ssh/id_mac_studio)
  --repo-path <path>     Remote repo path (default: \$HOME/Development/aragora)
  --skip-preflight       Skip remote validate-env and gh auth preflight
  --help                 Show this help

The script forwards selected environment variables for boss-loop and merge-arbiter
configuration if they are set locally (for example BOSS_LABELS, WORKER_MODEL,
REVIEW_MODEL, CLAUDE_RUNNER_PROFILES, BOSS_MAX_HOURS,
ARAGORA_DEV_COORDINATION_DB, BRANCH_PREFIXES).
EOF
}

quote_remote() {
    printf '%q' "$1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            STUDIO_HOST="${2:-$STUDIO_HOST}"
            shift 2
            ;;
        --user)
            STUDIO_USER="${2:-$STUDIO_USER}"
            shift 2
            ;;
        --ssh-key)
            SSH_KEY="${2:-$SSH_KEY}"
            shift 2
            ;;
        --repo-path)
            REMOTE_REPO_PATH="${2:-$REMOTE_REPO_PATH}"
            shift 2
            ;;
        --skip-preflight)
            SKIP_PREFLIGHT=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

SSH="SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes -i $SSH_KEY $STUDIO_USER@$STUDIO_HOST"

FORWARDED_EXPORTS=""
forward_vars=(
    BOSS_REPO
    TARGET_BRANCH
    WORKER_MODEL
    REVIEW_MODEL
    CLAUDE_RUNNER_PROFILES
    BOSS_LABELS
    BOSS_MAX_TICKS
    BOSS_INTERVAL_SECONDS
    BOSS_MAX_HOURS
    BOSS_MAX_CONSECUTIVE_FAILURES
    BOSS_MAX_PARALLEL_DISPATCHES
    BOSS_AUTONOMY_MODE
    BOSS_THROTTLE_SECONDS
    ARAGORA_USER_ID
    ARAGORA_WORKSPACE_ID
    ARAGORA_CLAUDE_PROFILE
    ARAGORA_DEV_COORDINATION_DB
    BRANCH_PREFIXES
    MERGE_ARBITER_INTERVAL_SECONDS
    MERGE_ARBITER_MAX_HOURS
    MERGE_ARBITER_MAX_CONSECUTIVE_FAILURES
    MERGE_ARBITER_THROTTLE_SECONDS
)

for var_name in "${forward_vars[@]}"; do
    if [[ -n "${!var_name-}" ]]; then
        FORWARDED_EXPORTS+="export ${var_name}=$(quote_remote "${!var_name}")"$'\n'
    fi
done

echo "=== Aragora Mac Studio LaunchAgent Install ==="
echo "Host: ${STUDIO_HOST}"
echo "User: ${STUDIO_USER}"
echo "Repo: ${REMOTE_REPO_PATH:-\$HOME/Development/aragora}"

cat <<EOF | eval $SSH "bash -s --"
set -euo pipefail
export PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH'
REMOTE_REPO_PATH=$(quote_remote "${REMOTE_REPO_PATH}")
if [[ -z "\${REMOTE_REPO_PATH}" ]]; then
    REMOTE_REPO_PATH="\$HOME/Development/aragora"
fi
${FORWARDED_EXPORTS}cd "\${REMOTE_REPO_PATH}" 2>/dev/null || { echo 'REPO: NOT FOUND'; exit 1; }
[ -d .venv ] || { echo 'VENV: NOT FOUND'; exit 1; }

if [[ $(quote_remote "${SKIP_PREFLIGHT}") != true ]]; then
    bash scripts/swarm_host_preflight.sh
fi

echo ''
echo '--- Installing boss-loop LaunchAgent ---'
bash scripts/install_boss_loop_launchd.sh

echo ''
echo '--- Installing merge-arbiter LaunchAgent ---'
bash scripts/install_merge_arbiter_launchd.sh

echo ''
echo '--- Installing worktree maintainer LaunchAgent ---'
bash scripts/install_worktree_maintainer_launchd.sh

echo ''
echo '--- Installed LaunchAgents ---'
launchctl list | grep 'com.aragora.' || true
EOF

echo ""
echo "=== Install complete ==="
echo "Next: bash scripts/studio-health.sh"
