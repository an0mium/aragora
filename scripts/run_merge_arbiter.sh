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

# Opt-in GitHub App auth for this pass's shell-level `gh` calls
# (ARAGORA_GH_APP_AUTH=1, default off; run-20260610 lane Z). The App
# installation token carries its OWN GraphQL budget, separate from the
# operator's default PAT whose budget is chronically exhausted by the poller
# fleet. Mint it ONCE here and share it across the ready-triage, backlog-gate,
# and auto-evidence blocks below (#8355 follow-up): all three run in the same
# pass, so a single mint == the per-pass token and avoids redundant
# gh_app_env.py calls. Previously only the auto-evidence subshell minted a
# token, so pr_ready_triage (gh pr ready = GraphQL-only mutation) and
# backlog_gate ran on the exhausted PAT and the funnel starved.
#
# Each consumer exports the token inside its own subshell so the long-running
# arbiter exec'd at the bottom does NOT inherit this one-shot token
# (installation tokens expire after ~1h; the arbiter refreshes its own per
# call via aragora.swarm.github_app_auth). Fail-safe: gh_app_env.py prints
# nothing when App config is absent, so each block degrades to the existing gh
# auth. The token value is never echoed. ARAGORA_GITHUB_AUTH_SOURCE tags the
# token so write ops can drop it (narrow App scopes).
ARBITER_GH_APP_TOKEN=""
if [[ "${ARAGORA_GH_APP_AUTH:-0}" == "1" ]]; then
    ARBITER_GH_APP_TOKEN="$("${PYTHON_BIN}" scripts/gh_app_env.py --print-token --quiet 2>/dev/null || true)"
    if [[ -n "${ARBITER_GH_APP_TOKEN}" ]]; then
        echo "Arbiter pass gh auth: GitHub App installation token (per-pass mint)."
    fi
fi

# Export the shared per-pass App token into the calling (sub)shell's gh env.
# No-op when the token is empty (App auth off or App config absent), so the
# default gh auth path is preserved byte-for-byte.
apply_arbiter_gh_app_token() {
    if [[ -n "${ARBITER_GH_APP_TOKEN}" ]]; then
        export GH_TOKEN="${ARBITER_GH_APP_TOKEN}"
        export GITHUB_TOKEN="${ARBITER_GH_APP_TOKEN}"
        export ARAGORA_GITHUB_AUTH_SOURCE="github_app_installation"
    fi
}

# Opt-in funnel autopilot: draft-to-ready promotion (throughput lever 0).
# Runs BEFORE the auto-evidence step so freshly-promoted PRs get evidenced the
# same pass. Best-effort: a triage failure must never abort the arbiter.
# Default off until the operator flips ARAGORA_READY_TRIAGE=1 (mirrors the
# ARAGORA_AUTO_EVIDENCE pattern below). Honors the wrapper's cwd convention
# (cd REPO_ROOT above), so pr_ready_triage resolves the repo from gh defaults
# exactly like auto_evidence_cycle does.
if [[ "${ARAGORA_READY_TRIAGE:-0}" == "1" ]]; then
    echo "Running bounded draft-to-ready triage (apply mode)..."
    # Subshell so the App token (when minted) reaches `gh pr ready` — a
    # GraphQL-only mutation — without leaking into the long-running arbiter.
    (
        apply_arbiter_gh_app_token
        exec "${PYTHON_BIN}" scripts/pr_ready_triage.py --apply
    ) || echo "Ready-triage reported failures (non-fatal for the arbiter)." >&2
fi

# Opt-in backlog backpressure refresh (throughput lever 2). When enabled,
# refreshes .aragora/backpressure.json each pass so writer lanes can consult
# the latest generate/shepherd signal. The gate is read-only except for its
# own signal file (no --apply flag exists; it writes the signal by default).
# Best-effort: never blocks the arbiter. Default off until the operator flips
# ARAGORA_BACKLOG_GATE=1.
if [[ "${ARAGORA_BACKLOG_GATE:-0}" == "1" ]]; then
    echo "Refreshing backlog backpressure signal..."
    # Subshell so the App token (when minted) backs backlog_gate's gh probes
    # without leaking into the long-running arbiter.
    (
        apply_arbiter_gh_app_token
        exec "${PYTHON_BIN}" scripts/backlog_gate.py --quiet
    ) || echo "Backlog gate reported failures (non-fatal for the arbiter)." >&2
fi

# Legacy direct auto-evidence cycle (throughput lever 1, run-20260610 c07b).
# This path posts evidence via auto_evidence_cycle.py --apply without the
# §Conductor prepared-artifact replay proof, so ARAGORA_AUTO_EVIDENCE=1 is no
# longer sufficient by itself. The wrapper enforces the explicit
# ARAGORA_ALLOW_LEGACY_AUTO_EVIDENCE_APPLY=1 override before invoking the legacy
# command; without it the evidence step is skipped entirely, no legacy evidence
# collection/posting runs in that pass, and the merge-arbiter still starts.
if [[ "${ARAGORA_AUTO_EVIDENCE:-0}" == "1" ]]; then
    if [[ "${ARAGORA_ALLOW_LEGACY_AUTO_EVIDENCE_APPLY:-0}" != "1" ]]; then
        echo "Skipping ARAGORA_AUTO_EVIDENCE=1: legacy direct apply requires ARAGORA_ALLOW_LEGACY_AUTO_EVIDENCE_APPLY=1 under §Conductor." >&2
        echo "No legacy evidence collection or posting will run in this pass; merge-quorum throughput may drop until exact-head Conductor replay or the explicit override is used." >&2
        echo "The merge-arbiter will still start; set the override only for an explicit operator exception." >&2
    else
        echo "Running bounded auto-evidence cycle (legacy direct apply override mode)..."
        # Reuses the shared per-pass App token minted above (same
        # ARAGORA_GH_APP_AUTH gate). Applied inside this subshell so the
        # long-running arbiter exec'd below does NOT inherit the one-shot token
        # (installation tokens expire after ~1h; the arbiter refreshes its own per
        # call via aragora.swarm.github_app_auth). Fail-safe: when the token is
        # empty (App auth off or config absent) this degrades to existing gh auth.
        (
            apply_arbiter_gh_app_token
            # --max-scan 40 mitigates scan starvation (harmless even after the
            # #8316 fix lands; it only widens the per-pass probe window).
            exec "${PYTHON_BIN}" scripts/auto_evidence_cycle.py --apply --max-scan 40
        ) || echo "Auto-evidence cycle reported failures (non-fatal for the arbiter)." >&2
    fi
fi

echo "Starting swarm merge-arbiter..."
echo "Using Python interpreter: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -u -m aragora.cli.main swarm merge-arbiter "$@"
