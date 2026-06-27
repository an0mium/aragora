#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POST_LOOP_ISSUE_REFILL="${ARAGORA_POST_LOOP_ISSUE_REFILL:-1}"
POST_LOOP_MAX_ISSUES="${ARAGORA_POST_LOOP_MAX_ISSUES:-20}"
POST_LOOP_DRY_RUN="${ARAGORA_POST_LOOP_DRY_RUN:-0}"
POST_LOOP_LABEL="${ARAGORA_POST_LOOP_LABEL:-}"
boss_repo=""
boss_label=""

# shellcheck source=scripts/aragora_runtime.sh
source "${REPO_ROOT}/scripts/aragora_runtime.sh"

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
PYTHON_BIN="$(resolve_aragora_python 'import pydantic; import aragora.cli.commands.swarm' 'boss-loop' 'boss-loop runtime')"

cd "${REPO_ROOT}"

echo "Starting boss-loop cycle for ${boss_repo} (label=${boss_label})..."
echo "Using Python interpreter: ${PYTHON_BIN}"
set +e
"${PYTHON_BIN}" -u -m aragora.cli.main swarm boss-loop "${args[@]}"
boss_status=$?
set -e
echo "Boss loop exited with status ${boss_status}."

# Opt-in merge-gate liveness reconciler (resilience doc A1; Sprint 3 goal 3(i)).
# Re-runs stale-but-satisfiable aragora-merge-quorum checks. Best-effort: never
# fails the cycle. Default off until the operator flips ARAGORA_QUORUM_RECONCILER=1.
if [[ "${ARAGORA_QUORUM_RECONCILER:-0}" == "1" ]]; then
    echo "Running quorum-rerun reconciler (apply mode)..."
    "${PYTHON_BIN}" scripts/quorum_rerun_reconciler.py --repo "${boss_repo}" --apply \
        || echo "Quorum reconciler reported failures (non-fatal for the cycle)." >&2
fi

# Opt-in drain step: when the backlog is over the open-PR cap, DRAIN the queue
# (merge fully-green PRs via the settle gate, close empty ones, PLAN repairs)
# instead of idling/manufacturing more work. Default OFF until the operator sets
# ARAGORA_DRAIN_ENABLED=1, so the cycle is unchanged otherwise. Dry-run (plan
# only) unless ARAGORA_DRAIN_APPLY=1. Repair is never auto-dispatched here (no
# --enable-repair-dispatch flag), so it only prints the bounded repair plan.
# Best-effort: a drain hiccup never fails the cycle.
if [[ "${boss_status}" -eq 0 && "${ARAGORA_DRAIN_ENABLED:-0}" == "1" ]]; then
    set +e
    "${PYTHON_BIN}" scripts/backlog_gate.py --quiet >/dev/null 2>&1
    gate_mode=$?
    set -e
    if [[ "${gate_mode}" -eq 3 ]]; then
        drain_cmd=(
            "${PYTHON_BIN}" scripts/boss_drain_pass.py
            --repo "${boss_repo}"
            --off-limits-prefix structex/
            --off-limits-prefix claude/
            --max-repairs "${ARAGORA_DRAIN_MAX_REPAIRS:-2}"
        )
        [[ "${ARAGORA_DRAIN_APPLY:-0}" == "1" ]] && drain_cmd+=(--apply)
        echo "Backlog over cap (shepherd) -> draining: ${drain_cmd[*]}"
        "${drain_cmd[@]}" || echo "Drain pass reported failures (non-fatal for the cycle)." >&2
    else
        echo "Backlog under cap (generate) -> skipping drain."
    fi
fi

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
    --substrate-cap
    "${ARAGORA_SUBSTRATE_CAP:-0.3}"
    --closure-floor
    "${ARAGORA_CLOSURE_FLOOR:-0.25}"
)
if [[ "${POST_LOOP_DRY_RUN}" == "1" ]]; then
    refill_cmd+=(--dry-run)
fi

echo "Running post-loop issue refill: ${refill_cmd[*]}"
"${refill_cmd[@]}"
