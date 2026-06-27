#!/usr/bin/env bash
# Periodically refresh the recurring proof-surface status docs that
# probe_proof_surface_freshness.py reports stale-when->7d.
#
# Currently covers:
#   - TW-02: B0_BENCHMARK_TRUTH_STATUS.md
#     publisher : build_benchmark_truth_artifact.py
#     renderer  : render_benchmark_truth_status.py
#   - TW-03: TW03_RESCUE_PRODUCTIZATION_STATUS.md
#     publisher : publish_rescue_productization_report.py
#     promotion : .aragora/rescue_productization/latest.json
#                 -> docs/status/generated/rescue_productization/latest.json
#     renderer  : render_rescue_productization_status.py
#
# Why this script exists: the publisher writes to .aragora/<surface>/
# (operator-local, untracked), but the renderer reads from
# docs/status/generated/<surface>/ (tracked). The "promote" step had
# rotted between April-May 2026, causing TW03 to drift 31+ days stale
# (codex's disk-pressure report 2026-05-19, flagged in the H01 receipt
# refresh).
#
# Usage:
#   scripts/refresh_proof_surfaces.sh             # all surfaces (default)
#   scripts/refresh_proof_surfaces.sh --surface tw03
#   scripts/refresh_proof_surfaces.sh --check     # report-only, no writes
#   scripts/refresh_proof_surfaces.sh --commit    # auto-stage + commit (no push)
#
# Designed to be safe under cron/launchd: produces an idempotent commit
# only when the underlying data actually changed (no churn on stable
# zero-event ledgers).

set -euo pipefail

SURFACE="all"
CHECK_ONLY=0
AUTO_COMMIT=0

usage() {
    cat <<'EOF'
refresh_proof_surfaces.sh — periodic refresher for recurring proof-surface docs

  --surface {all,b0,tw03}   Which surface(s) to refresh. Default: all.
  --check                   Report-only; don't run publishers or write files.
  --commit                  After refresh, stage + commit if files changed.
                            Does NOT push; operator/cron handles push.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --surface)  SURFACE="$2"; shift 2 ;;
        --check)    CHECK_ONLY=1; shift ;;
        --commit)   AUTO_COMMIT=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "Unknown flag: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$SURFACE" in
    all|b0|tw03) ;;
    *) echo "error: --surface must be one of all|b0|tw03" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "refresh_proof_surfaces.sh: surface=$SURFACE check=$CHECK_ONLY commit=$AUTO_COMMIT repo=$REPO_ROOT"

# Pre-state: capture current freshness
echo ""
echo "--- pre-state freshness ---"
python3 scripts/probe_proof_surface_freshness.py --pretty 2>/dev/null \
    || echo "(freshness probe reports stale surface(s); will refresh below)"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo ""
    echo "refresh_proof_surfaces.sh: --check requested; exiting without writes"
    exit 0
fi

REFRESHED_FILES=()

refresh_tw03() {
    echo ""
    echo "--- TW-03: publish + promote + render ---"
    # 1. Publish fresh report JSON to operator-local untracked path
    python3 scripts/publish_rescue_productization_report.py --json >/dev/null
    # 2. Promote the freshest JSON to the tracked path
    local src_dir="$REPO_ROOT/.aragora/rescue_productization"
    local dst_dir="$REPO_ROOT/docs/status/generated/rescue_productization"
    mkdir -p "$dst_dir"
    cp "$src_dir/latest.json" "$dst_dir/latest.json"
    # Also copy the latest timestamped variant (newest only)
    local newest_ts
    newest_ts="$(ls -t "$src_dir"/rescue-productization-*.json 2>/dev/null | head -1)"
    if [[ -n "$newest_ts" ]]; then
        cp "$newest_ts" "$dst_dir/$(basename "$newest_ts")"
        REFRESHED_FILES+=("$dst_dir/$(basename "$newest_ts")")
    fi
    # 3. Re-render markdown
    python3 scripts/render_rescue_productization_status.py >/dev/null
    REFRESHED_FILES+=(
        "$dst_dir/latest.json"
        "$REPO_ROOT/docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md"
    )
    echo "TW-03: published + promoted + rendered"
}

refresh_b0() {
    echo ""
    echo "--- TW-02 / B0: render-only (publishers in scripts/measure_b0_scorecard.py + build_benchmark_truth_artifact.py) ---"
    # B0 has separate publishers that aren't currently scheduled. This
    # script's contract is to re-render the markdown from whatever
    # truth-artifact + scorecard data is already on disk; the publishers
    # themselves are owned by the wedge-first roadmap track.
    python3 scripts/render_benchmark_truth_status.py >/dev/null
    REFRESHED_FILES+=("$REPO_ROOT/docs/status/B0_BENCHMARK_TRUTH_STATUS.md")
    echo "B0: re-rendered (publisher runs are external)"
}

case "$SURFACE" in
    all)  refresh_tw03; refresh_b0 ;;
    tw03) refresh_tw03 ;;
    b0)   refresh_b0 ;;
esac

# Post-state freshness check (now expected to pass)
echo ""
echo "--- post-state freshness ---"
python3 scripts/probe_proof_surface_freshness.py --pretty

if [[ "$AUTO_COMMIT" -eq 1 ]]; then
    if [[ ${#REFRESHED_FILES[@]} -eq 0 ]]; then
        echo "refresh_proof_surfaces.sh: nothing to commit"
        exit 0
    fi
    # Determine if any of the refreshed files actually changed in git
    files_changed=0
    for f in "${REFRESHED_FILES[@]}"; do
        if [[ ! -e "$f" ]]; then
            continue
        fi
        if ! git diff --quiet -- "$f" 2>/dev/null; then
            files_changed=1
            break
        fi
        if ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
            files_changed=1
            break
        fi
    done
    if [[ "$files_changed" -eq 0 ]]; then
        echo "refresh_proof_surfaces.sh: no diff vs git; skipping commit (idempotent on stable data)"
        exit 0
    fi
    echo ""
    echo "--- staging refreshed files for commit ---"
    for f in "${REFRESHED_FILES[@]}"; do
        if [[ -e "$f" ]]; then
            git add "$f"
        fi
    done
    git commit -m "docs(status): refresh proof surfaces (surface=$SURFACE)

Periodic refresh via scripts/refresh_proof_surfaces.sh. No semantic
content change unless the underlying data on disk advanced.

Co-Authored-By: scripts/refresh_proof_surfaces.sh <noreply@anthropic.com>" \
        || { echo "commit failed (likely no changes)"; exit 0; }
    echo "committed; operator/cron handles push"
fi

exit 0
