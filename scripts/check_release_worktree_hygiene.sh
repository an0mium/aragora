#!/usr/bin/env bash
set -euo pipefail

# Validate release-ops worktree hygiene:
# - no uncommitted changes
# - synchronized with origin/main (no ahead/behind drift)
# - running from main branch or detached HEAD

BASE_BRANCH="${1:-main}"
UPSTREAM="origin/${BASE_BRANCH}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[hygiene] not inside a git repository"
  exit 2
fi

if ! git rev-parse --verify "${UPSTREAM}" >/dev/null 2>&1; then
  echo "[hygiene] upstream branch not found: ${UPSTREAM}"
  exit 2
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [ -z "${CURRENT_BRANCH}" ]; then
  CURRENT_BRANCH="(detached)"
fi

STATUS_PORCELAIN="$(git status --porcelain)"
read -r BEHIND AHEAD < <(git rev-list --left-right --count "${UPSTREAM}...HEAD")

echo "[hygiene] branch=${CURRENT_BRANCH}"
echo "[hygiene] upstream=${UPSTREAM}"
echo "[hygiene] ahead=${AHEAD} behind=${BEHIND}"

FAIL=0

if [ -n "${STATUS_PORCELAIN}" ]; then
  echo "[hygiene] FAIL: working tree is dirty"
  FAIL=1
fi

if [ "${AHEAD}" -ne 0 ] || [ "${BEHIND}" -ne 0 ]; then
  echo "[hygiene] FAIL: branch is not synchronized with ${UPSTREAM}"
  FAIL=1
fi

if [ "${CURRENT_BRANCH}" != "${BASE_BRANCH}" ] && [ "${CURRENT_BRANCH}" != "(detached)" ]; then
  echo "[hygiene] FAIL: run release ops from '${BASE_BRANCH}' or detached HEAD"
  FAIL=1
fi

if [ "${FAIL}" -ne 0 ]; then
  echo "[hygiene] suggested reset flow:"
  echo "  git fetch origin --prune"
  echo "  git worktree add /tmp/aragora-release-\$(date +%s) origin/${BASE_BRANCH}"
  exit 1
fi

echo "[hygiene] OK"
