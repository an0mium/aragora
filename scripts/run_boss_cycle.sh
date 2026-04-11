#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POST_LOOP_ISSUE_REFILL="${ARAGORA_POST_LOOP_ISSUE_REFILL:-1}"
POST_LOOP_MAX_ISSUES="${ARAGORA_POST_LOOP_MAX_ISSUES:-20}"
POST_LOOP_DRY_RUN="${ARAGORA_POST_LOOP_DRY_RUN:-0}"
POST_LOOP_LABEL="${ARAGORA_POST_LOOP_LABEL:-}"
boss_repo=""
boss_label=""

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

cd "${REPO_ROOT}"

echo "Starting boss-loop cycle for ${boss_repo} (label=${boss_label})..."
set +e
python3 -u -m aragora.cli.main swarm boss-loop "${args[@]}"
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
    python3
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
