#!/usr/bin/env bash
# Setup git worktrees for parallel Claude Code sessions.
#
# Each worktree gets its own sibling directory and branch.
# Claude Code sessions run in separate worktrees -- zero file conflicts.
# When done, merge branches back to main with: ./scripts/merge_worktrees.sh
#
# Usage:
#   ./scripts/setup_worktrees.sh [OPTIONS]
#
# Options:
#   --count N        Number of worktrees to create (default: 12)
#   --prefix NAME    Branch prefix (default: dev)
#   --tracks LIST    Comma-separated tracks to create (default: all 6 tracks x2)
#   --base BRANCH    Base branch to fork from (default: main)
#   --dir PATH       Parent directory for worktrees (default: ../aragora-wt-*)
#   --list           List existing worktrees and exit
#   --help           Show this help
#
# Examples:
#   ./scripts/setup_worktrees.sh                              # 12 worktrees, all tracks
#   ./scripts/setup_worktrees.sh --count 4 --tracks sme,qa    # 4 worktrees, 2 tracks
#   ./scripts/setup_worktrees.sh --prefix sprint --count 6     # Sprint-named worktrees
#   ./scripts/setup_worktrees.sh --list                        # Show existing worktrees

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COUNT=12
PREFIX="dev"
BASE_BRANCH="main"
WORKTREE_PARENT="$(dirname "${REPO_ROOT}")"
LIST_ONLY=false
TRACKS_CSV=""

# All available tracks (2 sessions per track = 12 total by default)
ALL_TRACKS=(sme developer qa core security self-hosted)

# Track descriptions for human-readable output
declare -A TRACK_DESC=(
    [sme]="SME features, dashboard, user workspace"
    [developer]="SDKs, APIs, documentation, client packages"
    [qa]="Tests, CI/CD, code quality, coverage"
    [core]="Debate engine, agents, memory (requires approval)"
    [security]="Auth hardening, secrets, vulnerability scanning"
    [self-hosted]="Docker, deployment, backup/restore, ops"
)

usage() {
    head -25 "$0" | tail -23
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --count)     COUNT="$2"; shift 2 ;;
        --prefix)    PREFIX="$2"; shift 2 ;;
        --tracks)    TRACKS_CSV="$2"; shift 2 ;;
        --base)      BASE_BRANCH="$2"; shift 2 ;;
        --dir)       WORKTREE_PARENT="$2"; shift 2 ;;
        --list)      LIST_ONLY=true; shift ;;
        --help|-h)   usage ;;
        *)           echo "Unknown option: $1"; usage ;;
    esac
done

# List mode: show existing worktrees and exit
if ${LIST_ONLY}; then
    echo "=== Existing Worktrees ==="
    git -C "${REPO_ROOT}" worktree list
    echo ""
    # Count dev/* and work/* branches
    DEV_BRANCHES=$(git -C "${REPO_ROOT}" branch --list 'dev/*' 'work/*' 2>/dev/null | wc -l | tr -d ' ')
    echo "Development branches: ${DEV_BRANCHES}"
    exit 0
fi

# Build track list from CSV or defaults
SELECTED_TRACKS=()
if [[ -n "${TRACKS_CSV}" ]]; then
    IFS=',' read -ra SELECTED_TRACKS <<< "${TRACKS_CSV}"
else
    SELECTED_TRACKS=("${ALL_TRACKS[@]}")
fi

# Validate tracks
for track in "${SELECTED_TRACKS[@]}"; do
    found=false
    for valid in "${ALL_TRACKS[@]}"; do
        if [[ "${track}" == "${valid}" ]]; then
            found=true
            break
        fi
    done
    if ! ${found}; then
        echo "Error: Invalid track '${track}'"
        echo "Valid tracks: ${ALL_TRACKS[*]}"
        exit 1
    fi
done

TIMESTAMP="$(date +%m%d)"

echo "=== Aragora Worktree Setup ==="
echo "Count:  ${COUNT}"
echo "Prefix: ${PREFIX}"
echo "Base:   ${BASE_BRANCH}"
echo "Tracks: ${SELECTED_TRACKS[*]}"
echo ""

# Ensure base branch exists and is up to date
if ! git -C "${REPO_ROOT}" rev-parse --verify "${BASE_BRANCH}" >/dev/null 2>&1; then
    echo "Error: Base branch '${BASE_BRANCH}' does not exist"
    exit 1
fi

CREATED=0
for i in $(seq 1 "${COUNT}"); do
    # Cycle through selected tracks
    TRACK_IDX=$(( (i - 1) % ${#SELECTED_TRACKS[@]} ))
    TRACK="${SELECTED_TRACKS[${TRACK_IDX}]}"

    # Session number within this track
    SESSION_NUM=$(( (i - 1) / ${#SELECTED_TRACKS[@]} + 1 ))

    BRANCH_NAME="${PREFIX}/${TRACK}-${SESSION_NUM}"
    DIR_NAME="aragora-wt-${TRACK}-${SESSION_NUM}"
    TREE_DIR="${WORKTREE_PARENT}/${DIR_NAME}"

    DESC="${TRACK_DESC[${TRACK}]:-${TRACK}}"

    if [ -d "${TREE_DIR}" ]; then
        echo "[${i}/${COUNT}] Exists: ${DIR_NAME} (${TRACK})"
        continue
    fi

    echo "[${i}/${COUNT}] Creating: ${DIR_NAME}"
    echo "  Branch: ${BRANCH_NAME}"
    echo "  Track:  ${TRACK} - ${DESC}"

    git -C "${REPO_ROOT}" worktree add -b "${BRANCH_NAME}" "${TREE_DIR}" "${BASE_BRANCH}" 2>/dev/null || {
        # Branch may already exist
        if git -C "${REPO_ROOT}" rev-parse --verify "${BRANCH_NAME}" >/dev/null 2>&1; then
            git -C "${REPO_ROOT}" worktree add "${TREE_DIR}" "${BRANCH_NAME}"
        else
            echo "  FAILED to create worktree"
            continue
        fi
    }

    CREATED=$((CREATED + 1))
    echo "  Done."
    echo ""
done

echo ""
echo "=== Setup Complete ==="
echo "Created: ${CREATED} worktrees"
echo ""

if [ "${CREATED}" -gt 0 ]; then
    echo "Start Claude Code sessions in each worktree:"
    echo ""
    for i in $(seq 1 "${COUNT}"); do
        TRACK_IDX=$(( (i - 1) % ${#SELECTED_TRACKS[@]} ))
        TRACK="${SELECTED_TRACKS[${TRACK_IDX}]}"
        SESSION_NUM=$(( (i - 1) / ${#SELECTED_TRACKS[@]} + 1 ))
        DIR_NAME="aragora-wt-${TRACK}-${SESSION_NUM}"
        TREE_DIR="${WORKTREE_PARENT}/${DIR_NAME}"
        [ -d "${TREE_DIR}" ] && echo "  cd ${TREE_DIR} && claude"
    done
    echo ""
    echo "When done, merge back:"
    echo "  ./scripts/merge_worktrees.sh"
    echo ""
    echo "Clean up:"
    echo "  ./scripts/cleanup_worktrees.sh --merged"
fi
