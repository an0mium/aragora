#!/usr/bin/env bash
# Thin wrapper over shared Python worktree maintainer lifecycle.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HAS_REPO_ARG=false

for arg in "$@"; do
    case "$arg" in
        --repo|--repo=*)
            HAS_REPO_ARG=true
            break
            ;;
    esac
done

export ARAGORA_REPO_ROOT="${REPO_ROOT}"
if [[ "${HAS_REPO_ARG}" == true ]]; then
    exec python3 -m aragora.worktree.maintainer "$@"
fi
exec python3 -m aragora.worktree.maintainer --repo "${REPO_ROOT}" "$@"
