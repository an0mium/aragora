#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POST_LOOP_ISSUE_REFILL="${ARAGORA_POST_LOOP_ISSUE_REFILL:-1}"
POST_LOOP_MAX_ISSUES="${ARAGORA_POST_LOOP_MAX_ISSUES:-20}"
POST_LOOP_DRY_RUN="${ARAGORA_POST_LOOP_DRY_RUN:-0}"
POST_LOOP_LABEL="${ARAGORA_POST_LOOP_LABEL:-}"
boss_repo=""
boss_label=""

resolve_python_bin() {
    local candidates=()
    local candidate=""
    local python_cmd=""

    if [[ -n "${ARAGORA_PYTHON:-}" ]]; then
        if [[ -x "${ARAGORA_PYTHON}" ]]; then
            printf '%s\n' "${ARAGORA_PYTHON}"
            return 0
        fi
        echo "ARAGORA_PYTHON is set but not executable: ${ARAGORA_PYTHON}" >&2
    fi
    if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
        candidates+=("${REPO_ROOT}/.venv/bin/python3")
    fi
    if python_cmd="$(command -v python3 2>/dev/null)"; then
        candidates+=("${python_cmd}")
    fi
    if python_cmd="$(command -v python 2>/dev/null)"; then
        candidates+=("${python_cmd}")
    fi
    for candidate in "${candidates[@]}"; do
        if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
            continue
        fi
        if "${candidate}" -c 'import pydantic' >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    if command -v pyenv >/dev/null 2>&1; then
        candidate="$(pyenv which python3 2>/dev/null || true)"
        if [[ -n "${candidate}" && -x "${candidate}" ]] && "${candidate}" -c 'import pydantic' >/dev/null 2>&1; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    echo "No usable python interpreter with pydantic found for boss-loop runtime." >&2
    return 1
}

args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --boss-repo)
            if ((i + 1 < ${#args[@]})); then
                boss_repo="${args[$((i + 1))]}"
            fi
            ;;
        --label)
            if [[ -z "${boss_label}" ]] && ((i + 1 < ${#args[@]})); then
                boss_label="${args[$((i + 1))]}"
            fi
            ;;
    esac
done

boss_repo="${boss_repo:-synaptent/aragora}"
boss_label="${POST_LOOP_LABEL:-${boss_label:-boss-ready}}"
PYTHON_BIN="$(resolve_python_bin)"

cd "${REPO_ROOT}"

echo "Starting boss-loop cycle for ${boss_repo} (label=${boss_label})..."
echo "Using Python interpreter: ${PYTHON_BIN}"
set +e
"${PYTHON_BIN}" -u -m aragora.cli.main swarm boss-loop "${args[@]}"
boss_status=$?
set -e
echo "Boss loop exited with status ${boss_status}."

if [[ "${POST_LOOP_ISSUE_REFILL}" != "1" ]]; then
    echo "Post-loop issue refill disabled."
    exit "${boss_status}"
fi

if [[ "${boss_status}" -ne 0 ]]; then
    echo "Skipping post-loop issue refill because boss loop exited non-zero." >&2
    exit "${boss_status}"
fi

refill_cmd=(
    "${PYTHON_BIN}"
    scripts/generate_boss_issues.py
    --repo
    "${boss_repo}"
    --max-issues
    "${POST_LOOP_MAX_ISSUES}"
    --label
    "${boss_label}"
)
if [[ "${POST_LOOP_DRY_RUN}" == "1" ]]; then
    refill_cmd+=(--dry-run)
fi

echo "Running post-loop issue refill: ${refill_cmd[*]}"
"${refill_cmd[@]}"
