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
echo "Starting swarm merge-arbiter..."
echo "Using Python interpreter: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -u -m aragora.cli.main swarm merge-arbiter "$@"
