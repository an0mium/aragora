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

    echo '--- Proof-First Ledger ---'
    python3 - <<'PY'
from pathlib import Path

try:
    from aragora.swarm.shift_ledger import ShiftLedger
except Exception as exc:  # pragma: no cover - operator fallback
    print(f'ledger: unavailable ({exc})')
    raise SystemExit(0)

repo_root = Path.cwd()
ledger_path = repo_root / '.aragora' / 'proof_first_shift' / 'shift_ledger.jsonl'
if not ledger_path.exists():
    print('ledger: missing')
    raise SystemExit(0)

summary = ShiftLedger(path=ledger_path).get_status_summary()
green = summary.get('green_shift') or {}
print(
    'queue={queue} boss={boss} merge={merge} benchmark_fresh={fresh} merged_prs={merged} last_stop={stop}'.format(
        queue=summary.get('current_queue_size'),
        boss=summary.get('current_boss_running'),
        merge=summary.get('current_merge_running'),
        fresh=summary.get('current_benchmark_fresh'),
        merged=summary.get('prs_merged'),
        stop=summary.get('last_stop_reason') or '-',
    )
)
print(
    'green_shift={green} observed_hours={hours} window_complete={window}'.format(
        green=green.get('is_green'),
        hours=green.get('observed_hours'),
        window=green.get('window_complete'),
    )
)
PY

    echo ''
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
