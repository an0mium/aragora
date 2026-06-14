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

# Opt-in GitHub App auth for this cycle's shell-level `gh` calls
# (ARAGORA_GH_APP_AUTH=1, default off; mirrors run_merge_arbiter.sh and
# run_codex_automation_publisher.sh). The App installation token carries its
# OWN GraphQL budget, separate from the operator PAT whose budget is
# chronically exhausted by the poller fleet. Mint it ONCE here and share it
# across the read-heavy aux steps below (quorum reconciler + post-loop issue
# refill), both of which shell out via raw `subprocess.run(["gh", ...])` and
# therefore inherit the operator PAT rather than self-minting via
# aragora.swarm.github_app_auth like the boss-loop's own Python reads do.
# The substrate-cap scan in generate_boss_issues.py (gh pr list + per-PR
# files) is the heaviest GraphQL read draining the PAT each cycle.
#
# Each consumer applies the token inside its OWN subshell so it does not leak
# into the boss-loop Python process invoked above (which self-manages App auth
# per call and must keep the PAT for write ops). Fail-safe: gh_app_env.py
# prints nothing when App config is absent or the mint fails, so each block
# degrades to the existing gh auth byte-for-byte. The token value is never
# echoed. ARAGORA_GITHUB_AUTH_SOURCE tags the token so write ops can drop it.
BOSS_GH_APP_TOKEN=""
if [[ "${ARAGORA_GH_APP_AUTH:-0}" == "1" ]]; then
    BOSS_GH_APP_TOKEN="$("${PYTHON_BIN}" scripts/gh_app_env.py --print-token --quiet 2>/dev/null || true)"
    if [[ -n "${BOSS_GH_APP_TOKEN}" ]]; then
        echo "Boss cycle gh auth: GitHub App installation token (per-pass mint)."
    fi
fi

# Export the shared per-pass App token into the calling (sub)shell's gh env.
# No-op when the token is empty (App auth off or App config absent), so the
# default gh auth path is preserved byte-for-byte.
apply_boss_gh_app_token() {
    if [[ -n "${BOSS_GH_APP_TOKEN}" ]]; then
        export GH_TOKEN="${BOSS_GH_APP_TOKEN}"
        export GITHUB_TOKEN="${BOSS_GH_APP_TOKEN}"
        export ARAGORA_GITHUB_AUTH_SOURCE="github_app_installation"
    fi
}

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
    # Subshell so the shared per-pass App token (when minted) backs the
    # reconciler's gh check-run reads + `gh run rerun` calls without leaking
    # into the boss-loop Python process. Degrades to existing gh auth when the
    # token is empty.
    (
        apply_boss_gh_app_token
        exec "${PYTHON_BIN}" scripts/quorum_rerun_reconciler.py --repo "${boss_repo}" --apply
    ) || echo "Quorum reconciler reported failures (non-fatal for the cycle)." >&2
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
# Subshell so the shared per-pass App token (when minted) backs the heaviest
# read consumer in this cycle — generate_boss_issues.py's substrate-cap scan
# (gh pr list + per-PR files). Degrades to existing gh auth when the token is
# empty, preserving the default path byte-for-byte. The subshell propagates
# the refill's exit status to the (set -e) parent exactly as the bare call did.
(
    apply_boss_gh_app_token
    exec "${refill_cmd[@]}"
)
