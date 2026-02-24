#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/open_pr.sh [--base <branch>] [--draft] [-- <extra gh pr create args>]

Examples:
  scripts/open_pr.sh
  scripts/open_pr.sh --draft
  scripts/open_pr.sh --base main -- --label governance --reviewer octocat

Behavior:
  - Fails on main/master branch.
  - Fails when working tree is dirty.
  - Pushes current branch to origin.
  - Creates PR with gh using --fill (unless one already exists).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) is required." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Error: gh is not authenticated. Run: gh auth login" >&2
  exit 1
fi

base_branch="${BASE_BRANCH:-main}"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --base requires a value" >&2
        exit 2
      fi
      base_branch="$1"
      ;;
    --draft)
      extra_args+=("--draft")
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        extra_args+=("$1")
        shift
      done
      break
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$branch" == "main" || "$branch" == "master" ]]; then
  echo "Error: refusing to open PR from '$branch'. Create/use a feature branch." >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Error: working tree is not clean. Commit/stash before opening PR." >&2
  exit 1
fi

if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "Error: untracked files present. Commit/stash/clean before opening PR." >&2
  exit 1
fi

if git rev-parse --verify --quiet "origin/$branch" >/dev/null; then
  git push origin "$branch"
else
  git push -u origin "$branch"
fi

existing_url="$(gh pr list --head "$branch" --base "$base_branch" --json url --jq '.[0].url // ""')"
if [[ -n "$existing_url" ]]; then
  echo "PR already exists: $existing_url"
  exit 0
fi

gh pr create --base "$base_branch" --head "$branch" --fill "${extra_args[@]}"
