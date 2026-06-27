# shellcheck shell=bash
# Shared Aragora runtime helpers. Source this file; do not execute it.
#
#   source "$(dirname "$0")/aragora_runtime.sh"
#   PYTHON_BIN="$(resolve_aragora_python 'import pydantic' 'aragora' 'aragora runtime')"
#
# resolve_aragora_python resolves a usable interpreter AT RUNTIME so a moved or
# removed .venv degrades gracefully instead of breaking a long-running launchd/
# systemd service. Installers must call this at launch time (via a wrapper),
# never bake an absolute interpreter path into the generated unit.

# Determine the repository root. Honors ARAGORA_REPO_ROOT, otherwise derives it
# from this file's location (scripts/.. == repo root) regardless of the caller.
aragora_repo_root() {
    if [[ -n "${ARAGORA_REPO_ROOT:-}" && -d "${ARAGORA_REPO_ROOT}" ]]; then
        printf '%s\n' "${ARAGORA_REPO_ROOT}"
        return 0
    fi
    (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
}

# Return 0 if interpreter $1 is executable and can run probe $2 from dir $3.
# An empty probe accepts any executable interpreter (no import check).
_aragora_python_ok() {
    local interp="$1"
    local probe="$2"
    local root="$3"
    [[ -n "${interp}" && -x "${interp}" ]] || return 1
    (cd "${root}" && "${interp}" -c "${probe}" >/dev/null 2>&1)
}

# resolve_aragora_python [probe] [import_label] [runtime_label]
#
# Echoes the path of the first usable interpreter; returns 1 with a diagnostic
# if none qualify. Resolution order:
#   1. $ARAGORA_PYTHON   2. <repo>/.venv/bin/python3   3. python3
#   4. python            5. pyenv which python3
resolve_aragora_python() {
    local probe="${1-import pydantic}"
    local import_label="${2:-aragora}"
    local runtime_label="${3:-aragora runtime}"
    local repo_root
    local candidate=""
    local python_cmd=""
    local candidates=()
    repo_root="$(aragora_repo_root)"

    if [[ -n "${ARAGORA_PYTHON:-}" ]]; then
        if _aragora_python_ok "${ARAGORA_PYTHON}" "${probe}" "${repo_root}"; then
            printf '%s\n' "${ARAGORA_PYTHON}"
            return 0
        fi
        echo "Skipping ARAGORA_PYTHON without usable ${import_label} imports: ${ARAGORA_PYTHON}" >&2
    fi

    if [[ -x "${repo_root}/.venv/bin/python3" ]]; then
        candidates+=("${repo_root}/.venv/bin/python3")
    fi
    if python_cmd="$(command -v python3 2>/dev/null)"; then
        candidates+=("${python_cmd}")
    fi
    if python_cmd="$(command -v python 2>/dev/null)"; then
        candidates+=("${python_cmd}")
    fi
    if command -v pyenv >/dev/null 2>&1; then
        python_cmd="$(pyenv which python3 2>/dev/null || true)"
        if [[ -n "${python_cmd}" ]]; then
            candidates+=("${python_cmd}")
        fi
    fi

    if [[ ${#candidates[@]} -gt 0 ]]; then
        for candidate in "${candidates[@]}"; do
            if _aragora_python_ok "${candidate}" "${probe}" "${repo_root}"; then
                printf '%s\n' "${candidate}"
                return 0
            fi
            echo "Skipping Python candidate without usable ${import_label} imports: ${candidate}" >&2
        done
    fi

    echo "No usable python interpreter with pydantic and ${import_label} imports found for ${runtime_label}." >&2
    echo "  Tried: \$ARAGORA_PYTHON, ${repo_root}/.venv/bin/python3, python3, python, pyenv." >&2
    echo "  Set ARAGORA_PYTHON to a working interpreter or install deps (e.g. pip install -e .)." >&2
    return 1
}
