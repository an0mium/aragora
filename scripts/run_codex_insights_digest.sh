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

SINCE="${ARAGORA_CODEX_INSIGHTS_SINCE:-1h}"
INGEST_KM="${ARAGORA_CODEX_INSIGHTS_INGEST_KM:-1}"
RECEIPT_DIR="${ARAGORA_CODEX_INSIGHTS_RECEIPT_DIR:-${REPO_ROOT}/.aragora/codex_insights}"
ARAGORA_PYTHON="${ARAGORA_PYTHON:-}"

mkdir -p "${RECEIPT_DIR}"

if [[ -z "${ARAGORA_PYTHON}" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
        ARAGORA_PYTHON="${REPO_ROOT}/.venv/bin/python3"
    elif command -v python3 >/dev/null 2>&1; then
        ARAGORA_PYTHON="$(command -v python3)"
    else
        echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') ERROR: no python3 found" >&2
        exit 2
    fi
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
