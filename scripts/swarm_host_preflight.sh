#!/usr/bin/env bash
# Validate a host machine before installing or starting Aragora autonomy services.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATH_DEFAULT="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"
WORKER_MODEL="${WORKER_MODEL:-claude}"
REVIEW_MODEL="${REVIEW_MODEL:-codex}"
CLAUDE_PROFILES_RAW="${CLAUDE_RUNNER_PROFILES:-${ARAGORA_CLAUDE_PROFILE:-}}"

export PATH="${PATH_DEFAULT}:${PATH:-}"

usage() {
    cat <<'EOF'
Usage: ./scripts/swarm_host_preflight.sh

Checks:
  - repo and virtualenv presence
  - write access to .aragora/ and .worktrees/
  - gh auth status
  - aragora validate-env --json
  - Claude profile live verification when worker/reviewer model uses Claude
  - codex CLI presence when worker/reviewer model uses Codex
EOF
}

trim_text() {
    printf '%s' "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

collect_profiles() {
    local raw="$1"
    local -n target_ref=$2
    IFS=',' read -r -a pieces <<< "${raw}"
    for piece in "${pieces[@]}"; do
        local trimmed
        trimmed="$(trim_text "${piece}")"
        if [[ -n "${trimmed}" ]]; then
            target_ref+=("${trimmed}")
        fi
    done
}

needs_claude() {
    [[ "${WORKER_MODEL,,}" == *claude* || "${REVIEW_MODEL,,}" == *claude* ]]
}

needs_codex() {
    [[ "${WORKER_MODEL,,}" == *codex* || "${REVIEW_MODEL,,}" == *codex* ]]
}

require_provider_keys() {
    local missing=0

    if needs_claude; then
        if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
            echo "anthropic: NOT CONFIGURED" >&2
            missing=1
        else
            echo "anthropic: ok"
        fi
    fi

    if needs_codex; then
        if [[ -z "${OPENAI_API_KEY:-}" ]]; then
            echo "openai: NOT CONFIGURED" >&2
            missing=1
        else
            echo "openai: ok"
        fi
    fi

    return "${missing}"
}

run_worker_preflight() {
    python3 -m aragora.cli.main swarm preflight --worker-model "${WORKER_MODEL}"
}

main() {
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        usage
        exit 0
    fi

    cd "${REPO_ROOT}"

    echo "=== Aragora host preflight ==="
    echo "repo_root=${REPO_ROOT}"

    [[ -d .venv ]] || { echo "VENV: NOT FOUND" >&2; exit 1; }
    [[ -f "${VENV_ACTIVATE}" ]] || { echo "VENV ACTIVATE: NOT FOUND (${VENV_ACTIVATE})" >&2; exit 1; }
    source "${VENV_ACTIVATE}"
    command -v python3 >/dev/null || { echo "python3: NOT FOUND AFTER VENV ACTIVATE" >&2; exit 1; }

    mkdir -p .aragora .worktrees
    touch .aragora/.host_preflight_write_test
    touch .worktrees/.host_preflight_write_test
    rm -f .aragora/.host_preflight_write_test .worktrees/.host_preflight_write_test
    echo "write_access=ok"

    echo "--- gh auth ---"
    gh auth status

    echo ""
    echo "--- environment validation ---"
    python3 -m aragora.cli.main validate-env --json > /tmp/aragora-validate-env.json
    cat /tmp/aragora-validate-env.json

    if needs_codex; then
        echo ""
        echo "--- codex CLI ---"
        command -v codex >/dev/null || { echo "codex: NOT FOUND" >&2; exit 1; }
        codex --version
    fi

    if needs_claude; then
        echo ""
        echo "--- claude auth ---"
        command -v claude >/dev/null || { echo "claude: NOT FOUND" >&2; exit 1; }
        profiles=()
        collect_profiles "${CLAUDE_PROFILES_RAW}" profiles
        if [[ ${#profiles[@]} -eq 0 ]]; then
            echo "claude profiles: none specified; skipping live profile verify"
        else
            bash scripts/claude_profiles_bootstrap.sh verify "${profiles[@]}"
        fi
    fi

    echo ""
    echo "--- api provider check (worker-specific) ---"
    require_provider_keys

    if [[ "${ARAGORA_PREFLIGHT_SKIP_WORKER:-0}" != "1" ]]; then
        echo ""
        echo "--- worker preflight (read/write/commit/push/pr) ---"
        run_worker_preflight
    fi
    echo ""
    echo "preflight=ok"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
