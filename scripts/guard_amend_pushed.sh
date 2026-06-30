#!/usr/bin/env bash
# guard_amend_pushed.sh — refuse to amend a commit that is already published.
#
# v13 lane P72. Implements R19 ("never --amend a pushed commit"). Run this
# before `git commit --amend` (or `git rebase -i` that would rewrite HEAD)
# to confirm the current HEAD has not yet been pushed to <remote>/<branch>.
#
# Exit codes:
#   0  HEAD is local-only (or remote branch absent) — amend is safe.
#   1  AMEND-BLOCKED: HEAD SHA equals remote branch tip — refuse amend.
#   2  Usage / invocation error (not in a git repo, bad flags, etc.).
#
# Usage:
#   bash scripts/guard_amend_pushed.sh [--remote NAME] [--branch NAME]
#
# Defaults:
#   --remote origin
#   --branch  current branch (`git rev-parse --abbrev-ref HEAD`)

set -euo pipefail

REMOTE="origin"
BRANCH=""

usage() {
    printf '%s\n' \
        "Usage: scripts/guard_amend_pushed.sh [--remote NAME] [--branch NAME]" \
        "" \
        'Refuses `git commit --amend` when the current HEAD is already published' \
        "on the remote tracking branch. Implements v13 rule R19." \
        "" \
        "Options:" \
        "  --remote NAME   remote name (default: origin)" \
        "  --branch NAME   branch name (default: current branch from HEAD)" \
        "  -h, --help      show this help and exit" \
        "" \
        "Exit codes:" \
        "  0  amend is safe (HEAD ahead of remote, or remote branch absent)" \
        "  1  AMEND-BLOCKED (HEAD == remote tip)" \
        "  2  usage / invocation error"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)
            if [[ $# -lt 2 ]]; then
                echo "guard_amend_pushed: --remote requires a value" >&2
                exit 2
            fi
            REMOTE="$2"
            shift 2
            ;;
        --branch)
            if [[ $# -lt 2 ]]; then
                echo "guard_amend_pushed: --branch requires a value" >&2
                exit 2
            fi
            BRANCH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "guard_amend_pushed: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "guard_amend_pushed: not inside a git repository" >&2
    exit 2
fi

local_head="$(git rev-parse HEAD 2>/dev/null || true)"
if [[ -z "${local_head}" ]]; then
    echo "guard_amend_pushed: unable to resolve HEAD" >&2
    exit 2
fi

if [[ -z "${BRANCH}" ]]; then
    BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ -z "${BRANCH}" || "${BRANCH}" == "HEAD" ]]; then
        echo "guard_amend_pushed: detached HEAD — pass --branch explicitly" >&2
        exit 2
    fi
fi

# git ls-remote prints "<sha>\t<ref>" lines for matching refs. We restrict the
# match to the exact branch via `refs/heads/<branch>` to avoid partial matches.
remote_line="$(git ls-remote "${REMOTE}" "refs/heads/${BRANCH}" 2>/dev/null || true)"
remote_sha="$(printf '%s' "${remote_line}" | awk 'NR==1 {print $1}')"

if [[ -z "${remote_sha}" ]]; then
    echo "guard_amend_pushed: ${REMOTE}/${BRANCH} not found remotely — amend is safe."
    exit 0
fi

if [[ "${local_head}" == "${remote_sha}" ]]; then
    echo "AMEND-BLOCKED: HEAD is already published on ${REMOTE}/${BRANCH}. Use a new commit instead." >&2
    echo "  local HEAD : ${local_head}" >&2
    echo "  remote tip : ${remote_sha}" >&2
    exit 1
fi

echo "guard_amend_pushed: HEAD ${local_head:0:12} is ahead of ${REMOTE}/${BRANCH} (${remote_sha:0:12}) — amend is safe."
exit 0
