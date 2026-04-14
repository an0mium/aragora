#!/usr/bin/env bash
# Shared mergeability preflight for local Codex automations and swarm/boss-loop PRs.

set -euo pipefail

BASE_REF="${1:-origin/main}"
HEAD_REF="${2:-HEAD}"

usage() {
    cat <<'EOF'
Usage: scripts/automation_pr_preflight.sh [BASE_REF] [HEAD_REF]

Checks an automation branch before it is pushed or turned into a PR.

Checks:
  - base and head refs exist
  - branch has a non-empty diff versus the base
  - diff has no whitespace errors
  - session/log/coordination artifacts are not committed
  - source changes without test changes are called out for operator review
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if ! git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null 2>&1; then
    echo "preflight: base ref not found: ${BASE_REF}" >&2
    exit 2
fi

if ! git rev-parse --verify "${HEAD_REF}^{commit}" >/dev/null 2>&1; then
    echo "preflight: head ref not found: ${HEAD_REF}" >&2
    exit 2
fi

changed_files="$(git diff --name-only "${BASE_REF}...${HEAD_REF}")"
if [[ -z "${changed_files}" ]]; then
    echo "preflight: no branch diff versus ${BASE_REF}" >&2
    exit 1
fi

echo "preflight: checking whitespace"
git diff --check "${BASE_REF}...${HEAD_REF}"

forbidden_regex='(^|/)(\.codex_session_active|\.claude-session-active|\.nomic-session-active|\.codex_session_meta\.json|\.swarm_worker_stdout\.log|\.swarm_worker_stderr\.log|\.swarm_worker_status\.json|\.swarm_repair_journal\.json|\.operator_state\.json|\.operator_snapshot\.json)$|(^|/)(\.aragora_events|\.pytest_cache|__pycache__)/'
forbidden_files="$(printf '%s\n' "${changed_files}" | grep -E "${forbidden_regex}" || true)"
if [[ -n "${forbidden_files}" ]]; then
    echo "preflight: automation/session artifacts must not be committed:" >&2
    printf '%s\n' "${forbidden_files}" >&2
    exit 1
fi

source_changes="$(printf '%s\n' "${changed_files}" | grep -E '(^aragora/|^scripts/|^\.github/|^tests/).*\.(py|sh|ya?ml|toml|json|ts|tsx|js|jsx)$' || true)"
test_changes="$(printf '%s\n' "${changed_files}" | grep -E '(^tests/|__tests__/|\.test\.|\.spec\.)' || true)"
docs_only_changes="$(printf '%s\n' "${changed_files}" | grep -Ev '(^docs/|^docs-site/|\.md$)' || true)"

if [[ -n "${source_changes}" && -z "${test_changes}" ]]; then
    echo "preflight: source/config changes found without test changes." >&2
    echo "preflight: run the relevant validation command and include it in the PR body." >&2
    echo "preflight: changed source/config paths:" >&2
    printf '%s\n' "${source_changes}" >&2
fi

if [[ -z "${docs_only_changes}" ]]; then
    echo "preflight: docs-only diff detected"
fi

echo "preflight: changed files"
printf '%s\n' "${changed_files}" | sed 's/^/  - /'
echo "preflight: ok"
