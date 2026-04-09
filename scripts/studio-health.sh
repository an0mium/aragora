#!/usr/bin/env bash
# Check aragora health on Mac Studio from the MacBook
# Run: bash scripts/studio-health.sh
set -uo pipefail

STUDIO_HOST="${STUDIO_HOST:-10.0.0.62}"
STUDIO_USER="${STUDIO_USER:-armand}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_mac_studio}"
SSH="SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes -i $SSH_KEY -o ConnectTimeout=5 $STUDIO_USER@$STUDIO_HOST"

echo "=== Aragora Mac Studio Health ==="

# Test connectivity
if ! eval $SSH "echo 'connected'" 2>/dev/null; then
    echo "Mac Studio: UNREACHABLE"
    echo "Try Tailscale: STUDIO_HOST=100.71.253.66 bash scripts/studio-health.sh"
    exit 1
fi

eval $SSH "
    export PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH'
    cd ~/Development/aragora 2>/dev/null || { echo 'REPO: NOT FOUND'; exit 1; }
    source .venv/bin/activate 2>/dev/null || { echo 'VENV: NOT FOUND'; exit 1; }

    echo '--- Processes ---'
    ARBITER=\$(pgrep -f 'swarm.*merge-arbiter' | head -1)
    BOSS=\$(pgrep -f 'swarm.*boss-loop' | head -1)
    echo \"Arbiter: \${ARBITER:-DEAD}\"
    echo \"Boss loop: \${BOSS:-DEAD}\"

    echo ''
    echo '--- Disk ---'
    df -h / | tail -1

    echo ''
    echo '--- Worktrees ---'
    find .worktrees -maxdepth 2 -type d 2>/dev/null | wc -l | tr -d ' '
    echo 'worktrees'

    echo ''
    echo '--- Open PRs ---'
    gh pr list --state open --json number --jq 'length' 2>/dev/null || echo 'gh: not authenticated'
    echo 'open PRs'

    echo ''
    echo '--- Latest logs ---'
    BOSS_LOG=''
    for candidate in .aragora/overnight/boss-loop-launchd.log \$(ls -t .aragora/overnight/boss-loop-*.log 2>/dev/null); do
        [ -f \"\$candidate\" ] || continue
        BOSS_LOG=\"\$candidate\"
        break
    done
    ARB_LOG=''
    for candidate in .aragora/overnight/merge-arbiter-launchd.log \$(ls -t .aragora/overnight/merge-arbiter-*.log 2>/dev/null); do
        [ -f \"\$candidate\" ] || continue
        ARB_LOG=\"\$candidate\"
        break
    done
    echo \"Boss log: \$BOSS_LOG\"
    echo \"Arbiter log: \$ARB_LOG\"
    [ -n \"\$BOSS_LOG\" ] && tail -3 \"\$BOSS_LOG\"
    [ -n \"\$ARB_LOG\" ] && tail -3 \"\$ARB_LOG\"

    echo ''
    echo '--- Git status ---'
    git log --oneline -1
    echo \"Branch: \$(git branch --show-current)\"
"
