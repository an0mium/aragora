#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/start_feature_branch.sh <type> <slug> [--base <ref>]

Examples:
  scripts/start_feature_branch.sh feat oracle-stream-guard
  scripts/start_feature_branch.sh fix oauth-redirect-check --base origin/main

Notes:
  - Requires a clean working tree.
  - Creates branch as: <type>/<normalized-slug>
  - Defaults base ref to origin/main (override with --base).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

branch_type="$1"
slug_raw="$2"
shift 2

base_ref="${BASE_REF:-origin/main}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --base requires a value" >&2
        exit 2
      fi
      base_ref="$1"
      ;;
    *)
      echo "Error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Error: working tree is not clean. Commit/stash changes first." >&2
  exit 1
fi

if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "Error: untracked files present. Commit/stash/clean first." >&2
  exit 1
fi

slug="$(echo "$slug_raw" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9._/-]+#-#g; s#-+#-#g; s#(^-|-$)##g')"
if [[ -z "$slug" ]]; then
  echo "Error: slug '$slug_raw' is invalid after normalization." >&2
  exit 1
fi

branch="${branch_type}/${slug}"

if git show-ref --verify --quiet "refs/heads/$branch"; then
  echo "Error: local branch '$branch' already exists." >&2
  exit 1
fi

if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
  echo "Error: remote branch 'origin/$branch' already exists." >&2
  exit 1
fi

git fetch origin main --quiet || true
git switch -c "$branch" "$base_ref"

echo "Created feature branch: $branch"
echo "Base ref: $base_ref"
