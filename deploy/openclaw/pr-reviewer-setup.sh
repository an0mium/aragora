#!/usr/bin/env bash
# Aragora PR Reviewer — One-Command Setup
#
# Usage:
#   ./pr-reviewer-setup.sh                   # Dry-run review of an0mium/aragora
#   ./pr-reviewer-setup.sh --repo owner/repo  # Review a specific repo
#   ./pr-reviewer-setup.sh --install-workflow  # Also install the GH Actions workflow
#   ./pr-reviewer-setup.sh --help
#
# Prerequisites: docker, docker compose, gh (GitHub CLI, authenticated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.pr-reviewer.yml"

# Defaults
REPO="${GITHUB_REPO:-an0mium/aragora}"
MODE="once"
DRY_RUN="true"
INSTALL_WORKFLOW=false
PR_URL=""

usage() {
    cat <<EOF
Aragora PR Reviewer Setup

Usage: $(basename "$0") [OPTIONS]

Options:
  --repo OWNER/REPO     GitHub repo to review (default: an0mium/aragora)
  --pr URL              Review a specific PR URL
  --live                Enable live mode (posts comments to PRs)
  --watch               Watch for new PRs and review continuously
  --install-workflow    Copy GitHub Actions workflow into the repo
  --help                Show this help

Environment variables:
  GITHUB_TOKEN          GitHub auth token (or use \`gh auth login\`)
  ANTHROPIC_API_KEY     Required for live mode
  OPENAI_API_KEY        Optional second agent
  OPENROUTER_API_KEY    Optional fallback agent

Examples:
  # Dry-run review of your repo
  $(basename "$0") --repo myorg/myrepo

  # Review a specific PR (live, posts comment)
  $(basename "$0") --pr https://github.com/myorg/myrepo/pull/42 --live

  # Install the GitHub Actions workflow into your repo
  cd /path/to/your/repo && $(basename "$0") --install-workflow
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)    REPO="$2"; shift 2 ;;
        --pr)      PR_URL="$2"; shift 2 ;;
        --live)    DRY_RUN="false"; shift ;;
        --watch)   MODE="watch"; DRY_RUN="false"; shift ;;
        --install-workflow) INSTALL_WORKFLOW=true; shift ;;
        --help|-h) usage ;;
        *)         echo "Unknown option: $1"; usage ;;
    esac
done

# Check prerequisites
check_prereqs() {
    local missing=()
    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v gh >/dev/null 2>&1     || missing+=("gh (GitHub CLI)")

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Missing prerequisites: ${missing[*]}"
        echo "Install them and try again."
        exit 1
    fi

    # Check GitHub auth
    if ! gh auth status >/dev/null 2>&1; then
        echo "GitHub CLI not authenticated. Run: gh auth login"
        exit 1
    fi
}

# Install GitHub Actions workflow
install_workflow() {
    local target_dir=".github/workflows"
    local source="$SCRIPT_DIR/../../.github/workflows/pr-debate.yml"

    if [[ ! -d ".git" ]]; then
        echo "Error: not in a git repository. cd to your repo root first."
        exit 1
    fi

    mkdir -p "$target_dir"
    cp "$source" "$target_dir/aragora-pr-review.yml"

    echo "Installed: $target_dir/aragora-pr-review.yml"
    echo ""
    echo "Next steps:"
    echo "  1. Add ANTHROPIC_API_KEY to your repo secrets (Settings > Secrets)"
    echo "  2. Optionally add OPENAI_API_KEY and OPENROUTER_API_KEY"
    echo "  3. git add $target_dir/aragora-pr-review.yml && git commit -m 'Add Aragora PR reviewer'"
    echo "  4. Push and open a PR to see it in action"
}

main() {
    check_prereqs

    if $INSTALL_WORKFLOW; then
        install_workflow
        exit 0
    fi

    echo "=== Aragora PR Reviewer ==="
    echo "Repo:     $REPO"
    echo "Mode:     $MODE"
    echo "Dry run:  $DRY_RUN"
    [[ -n "$PR_URL" ]] && echo "PR:       $PR_URL"
    echo ""

    # Get GitHub token from gh CLI if not set
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
    fi

    # Export for docker compose
    export GITHUB_REPO="$REPO"
    export GITHUB_TOKEN
    export AGENT_MODE="$MODE"
    export DRY_RUN
    export PR_URL

    # Determine profile
    local profile="default"
    if [[ "$DRY_RUN" == "false" ]]; then
        profile="live"
    fi

    echo "Starting reviewer container..."
    docker compose -f "$COMPOSE_FILE" --profile "$profile" up --build

    echo ""
    echo "Review complete. Audit log at: docker volume inspect aragora-pr-reviewer_reviewer-data"
}

main "$@"
