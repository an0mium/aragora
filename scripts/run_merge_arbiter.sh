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

# Opt-in bounded auto-evidence cycle (throughput lever 1, run-20260610 c07b).
# Produces lint-validated two-family model-review evidence for ready Tier 0-2
# PRs missing counted evidence, then re-runs stale quorum checks. Best-effort:
# never blocks the arbiter. Default off until the operator flips
# ARAGORA_AUTO_EVIDENCE=1 (mirrors the ARAGORA_QUORUM_RECONCILER pattern in
# run_boss_cycle.sh).
if [[ "${ARAGORA_AUTO_EVIDENCE:-0}" == "1" ]]; then
    echo "Running bounded auto-evidence cycle (apply mode)..."
    "${PYTHON_BIN}" scripts/auto_evidence_cycle.py --apply \
        || echo "Auto-evidence cycle reported failures (non-fatal for the arbiter)." >&2
fi

echo "Starting swarm merge-arbiter..."
echo "Using Python interpreter: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -u -m aragora.cli.main swarm merge-arbiter "$@"
