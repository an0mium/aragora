#!/usr/bin/env bash
# Resolve the Python interpreter at runtime, then run the swarm merge-arbiter.
# Used as the launchd entrypoint so a moved/removed .venv degrades gracefully
# instead of breaking the service with a stale interpreter path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/aragora_runtime.sh
source "${REPO_ROOT}/scripts/aragora_runtime.sh"

PYTHON_BIN="$(resolve_aragora_python 'import pydantic; import aragora.cli.commands.swarm' 'merge-arbiter' 'merge-arbiter runtime')"

cd "${REPO_ROOT}"

# Opt-in funnel autopilot: draft-to-ready promotion (throughput lever 0).
# Runs BEFORE the auto-evidence step so freshly-promoted PRs get evidenced the
# same pass. Best-effort: a triage failure must never abort the arbiter.
# Default off until the operator flips ARAGORA_READY_TRIAGE=1 (mirrors the
# ARAGORA_AUTO_EVIDENCE pattern below). Honors the wrapper's cwd convention
# (cd REPO_ROOT above), so pr_ready_triage resolves the repo from gh defaults
# exactly like auto_evidence_cycle does.
if [[ "${ARAGORA_READY_TRIAGE:-0}" == "1" ]]; then
    echo "Running bounded draft-to-ready triage (apply mode)..."
    "${PYTHON_BIN}" scripts/pr_ready_triage.py --apply \
        || echo "Ready-triage reported failures (non-fatal for the arbiter)." >&2
fi

# Opt-in backlog backpressure refresh (throughput lever 2). When enabled,
# refreshes .aragora/backpressure.json each pass so writer lanes can consult
# the latest generate/shepherd signal. The gate is read-only except for its
# own signal file (no --apply flag exists; it writes the signal by default).
# Best-effort: never blocks the arbiter. Default off until the operator flips
# ARAGORA_BACKLOG_GATE=1.
if [[ "${ARAGORA_BACKLOG_GATE:-0}" == "1" ]]; then
    echo "Refreshing backlog backpressure signal..."
    "${PYTHON_BIN}" scripts/backlog_gate.py --quiet \
        || echo "Backlog gate reported failures (non-fatal for the arbiter)." >&2
fi

# Opt-in bounded auto-evidence cycle (throughput lever 1, run-20260610 c07b).
# Produces lint-validated two-family model-review evidence for ready Tier 0-2
# PRs missing counted evidence, then re-runs stale quorum checks. Best-effort:
# never blocks the arbiter. Default off until the operator flips
# ARAGORA_AUTO_EVIDENCE=1 (mirrors the ARAGORA_QUORUM_RECONCILER pattern in
# run_boss_cycle.sh).
if [[ "${ARAGORA_AUTO_EVIDENCE:-0}" == "1" ]]; then
    echo "Running bounded auto-evidence cycle (apply mode)..."
    # Opt-in GitHub App auth for the auto-evidence pass's shell-level `gh`
    # calls (ARAGORA_GH_APP_AUTH=1, default off; run-20260610 lane Z).
    # Minted per pass inside a subshell — installation tokens expire after
    # one hour, so the long-running arbiter exec'd below must NOT inherit a
    # one-shot token (it refreshes its own per call via
    # aragora.swarm.github_app_auth). Fail-safe: gh_app_env.py prints nothing
    # when App config is absent, so the pass degrades to existing gh auth.
    # The token value is never echoed. ARAGORA_GITHUB_AUTH_SOURCE tags the
    # token so write ops can drop it (narrow App scopes).
    (
        if [[ "${ARAGORA_GH_APP_AUTH:-0}" == "1" ]]; then
            tok="$("${PYTHON_BIN}" scripts/gh_app_env.py --print-token --quiet 2>/dev/null || true)"
            if [[ -n "${tok}" ]]; then
                export GH_TOKEN="${tok}"
                export GITHUB_TOKEN="${tok}"
                export ARAGORA_GITHUB_AUTH_SOURCE="github_app_installation"
                echo "Auto-evidence gh auth: GitHub App installation token (per-pass mint)."
            fi
            unset tok
        fi
        # --max-scan 40 mitigates scan starvation (harmless even after the
        # #8316 fix lands; it only widens the per-pass probe window).
        exec "${PYTHON_BIN}" scripts/auto_evidence_cycle.py --apply --max-scan 40
    ) || echo "Auto-evidence cycle reported failures (non-fatal for the arbiter)." >&2
fi

echo "Starting swarm merge-arbiter..."
echo "Using Python interpreter: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -u -m aragora.cli.main swarm merge-arbiter "$@"
