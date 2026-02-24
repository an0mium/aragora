#!/usr/bin/env bash
# Install a launchd job that runs worktree_maintainer.sh on an interval.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.aragora.codex-worktree-maintainer"
INTERVAL_SECONDS=300
BASE_BRANCH="main"
TTL_HOURS="${CODEX_WORKTREE_TTL_HOURS:-24}"
STRATEGY="merge"
LOG_PATH="${REPO_ROOT}/.worktrees/codex-maintainer.log"
KEEP_BRANCHES=true

usage() {
    cat <<'EOF'
Usage: ./scripts/install_worktree_maintainer_launchd.sh [options]

Options:
  --interval-seconds <n>        launchd StartInterval (default: 300)
  --base <branch>               Base branch to integrate from (default: main)
  --ttl-hours <n>               Stale-session TTL in hours (default: 24)
  --strategy <merge|rebase|ff-only|none>
                                Integration strategy (default: merge)
  --delete-branches             Allow cleanup to delete local codex/* branches
  --log-path <file>             Log file path (default: .worktrees/codex-maintainer.log)
  --help                        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval-seconds)
            INTERVAL_SECONDS="${2:-300}"
            shift 2
            ;;
        --base)
            BASE_BRANCH="${2:-main}"
            shift 2
            ;;
        --ttl-hours)
            TTL_HOURS="${2:-24}"
            shift 2
            ;;
        --strategy)
            STRATEGY="${2:-merge}"
            shift 2
            ;;
        --delete-branches)
            KEEP_BRANCHES=false
            shift
            ;;
        --log-path)
            LOG_PATH="${2:-}"
            shift 2
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

case "$STRATEGY" in
    merge|rebase|ff-only|none) ;;
    *)
        echo "Invalid strategy: $STRATEGY" >&2
        exit 2
        ;;
esac

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "interval must be numeric" >&2
    exit 2
fi

PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
mkdir -p "$(dirname "$PLIST_PATH")"
mkdir -p "$(dirname "$LOG_PATH")"

KEEP_FLAG="--no-delete-branches"
if [[ "${KEEP_BRANCHES}" == false ]]; then
    KEEP_FLAG="--delete-branches"
fi

cat >"${PLIST_PATH}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd "${REPO_ROOT}" &amp;&amp; ./scripts/worktree_maintainer.sh --base "${BASE_BRANCH}" --ttl-hours "${TTL_HOURS}" --strategy "${STRATEGY}" --reconcile-only ${KEEP_FLAG}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>${INTERVAL_SECONDS}</integer>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>StandardOutPath</key>
  <string>${LOG_PATH}</string>
  <key>StandardErrorPath</key>
  <string>${LOG_PATH}</string>
</dict>
</plist>
EOF

launchctl unload "${PLIST_PATH}" >/dev/null 2>&1 || true
launchctl load "${PLIST_PATH}"

echo "Installed launchd job: ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Interval: ${INTERVAL_SECONDS}s"
echo "Log: ${LOG_PATH}"
