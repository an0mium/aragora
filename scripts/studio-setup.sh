#!/usr/bin/env bash
# Setup aragora on Mac Studio for always-on autonomous operation
# Run from the MacBook: bash scripts/studio-setup.sh
set -euo pipefail

STUDIO_HOST="${STUDIO_HOST:-10.0.0.62}"
STUDIO_USER="${STUDIO_USER:-armand}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_mac_studio}"
SSH="SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes -i $SSH_KEY $STUDIO_USER@$STUDIO_HOST"

echo "=== Aragora Mac Studio Setup ==="
echo "Host: $STUDIO_HOST"
echo "User: $STUDIO_USER"

# 1. Test connectivity
echo ""
echo "--- Testing SSH ---"
eval $SSH "echo 'Connected to Mac Studio' && uname -m && python3 --version" || {
    echo "SSH failed. Make sure:"
    echo "  1. Mac Studio is awake (log in at console if needed)"
    echo "  2. SSH key is at $SSH_KEY"
    echo "  3. Try: SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes -i $SSH_KEY $STUDIO_USER@$STUDIO_HOST"
    exit 1
}

# 2. Clone/update repo
echo ""
echo "--- Setting up repo ---"
eval $SSH "
    export PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH'
    if [ -d ~/Development/aragora/.git ]; then
        echo 'Repo exists, pulling latest'
        cd ~/Development/aragora && git pull --ff-only origin main
    else
        echo 'Cloning repo'
        mkdir -p ~/Development
        git clone https://github.com/synaptent/aragora.git ~/Development/aragora
    fi
"

# 3. Create venv and install deps
echo ""
echo "--- Installing dependencies ---"
eval $SSH "
    export PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH'
    cd ~/Development/aragora
    if [ ! -d .venv ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -e . --quiet
    echo 'Dependencies installed'
"

# 4. Copy Claude profiles
echo ""
echo "--- Copying Claude profiles ---"
rsync -av -e "SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes -i $SSH_KEY" \
    ~/.aragora-claude/ $STUDIO_USER@$STUDIO_HOST:~/.aragora-claude/ 2>/dev/null || \
    echo "WARNING: Could not copy Claude profiles. Copy manually if needed."

# 5. Run preflight doctor
echo ""
echo "--- Running preflight ---"
eval $SSH "
    export PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH'
    cd ~/Development/aragora
    source .venv/bin/activate
    python3 -c '
import aragora
print(\"aragora import: OK\")
' 2>/dev/null && echo 'Import check: OK' || echo 'Import check: FAILED'

    # Check gh auth
    gh auth status 2>/dev/null && echo 'GitHub CLI: OK' || echo 'GitHub CLI: needs auth'

    # Check disk
    echo 'Disk:'
    df -h / | tail -1
"

echo ""
echo "=== Setup complete ==="
echo "Next: Install LaunchAgents with: bash scripts/studio-install-agents.sh"
