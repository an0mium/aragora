#!/usr/bin/env bash
# Run one Codex insights digest cycle. Designed for periodic invocation via
# launchd or cron.
#
# Reads ~/.codex/ via the aragora codex inspector (read-only), emits a
# SHA-256-bound JSON receipt to .aragora/codex_insights/, and best-effort
# ingests it into the Aragora Knowledge Mound via `aragora km store`.
#
# Exits 0 on success; non-zero on aragora CLI failure. Designed to be safe
# under launchd KeepAlive — never blocks indefinitely.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=scripts/aragora_runtime.sh
source "${REPO_ROOT}/scripts/aragora_runtime.sh"

SINCE="${ARAGORA_CODEX_INSIGHTS_SINCE:-1h}"
INGEST_KM="${ARAGORA_CODEX_INSIGHTS_INGEST_KM:-1}"
RECEIPT_DIR="${ARAGORA_CODEX_INSIGHTS_RECEIPT_DIR:-${REPO_ROOT}/.aragora/codex_insights}"

mkdir -p "${RECEIPT_DIR}"

if ! ARAGORA_PYTHON="$(resolve_aragora_python 'import pydantic' 'codex-insights' 'codex insights digest')"; then
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') ERROR: no usable python3 found" >&2
    exit 2
fi

DIGEST_ARGS=(codex insights digest "--since" "${SINCE}" "--emit-receipt" "--receipt-dir" "${RECEIPT_DIR}")
if [[ "${INGEST_KM}" == "1" || "${INGEST_KM,,}" == "true" ]]; then
    DIGEST_ARGS+=("--ingest-km")
fi

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') START aragora codex insights digest (since=${SINCE}, receipt_dir=${RECEIPT_DIR})"
if ! "${ARAGORA_PYTHON}" -m aragora.cli.main "${DIGEST_ARGS[@]}"; then
    rc=$?
    echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') ERROR digest exited rc=${rc}" >&2
    exit "${rc}"
fi
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') OK digest cycle complete"
