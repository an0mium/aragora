#!/usr/bin/env bash
# Resolve the Python interpreter at runtime, then run the PR watch daemon.
# Used as the systemd ExecStart entrypoint so the unit no longer hardcodes an
# absolute interpreter path; an operator can still pin ARAGORA_PYTHON via the
# unit's EnvironmentFile.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/aragora_runtime.sh
source "${REPO_ROOT}/scripts/aragora_runtime.sh"

PYTHON_BIN="$(resolve_aragora_python 'import pydantic' 'pr-watch' 'pr-watch daemon')"

cd "${REPO_ROOT}"
echo "Starting Aragora PR watch daemon..."
echo "Using Python interpreter: ${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m aragora.compat.openclaw.pr_watch_daemon "$@"
