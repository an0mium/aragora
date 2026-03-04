#!/usr/bin/env bash
set -euo pipefail

EXTRAS=""
PROJECT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extras)
      EXTRAS="${2:-}"
      shift 2
      ;;
    --project-dir)
      PROJECT_DIR="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

has_project_markers() {
  local dir="$1"
  [[ -f "${dir}/pyproject.toml" || -f "${dir}/setup.py" ]]
}

resolve_project_root() {
  local start="$1"
  [[ -n "$start" ]] || return 1
  local dir
  dir="$(cd "$start" 2>/dev/null && pwd -P)" || return 1
  while [[ "$dir" != "/" ]]; do
    if has_project_markers "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  if has_project_markers "/"; then
    printf '/\n'
    return 0
  fi
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_HINT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

declare -a CANDIDATES=()
if [[ -n "$PROJECT_DIR" ]]; then
  CANDIDATES+=("$PROJECT_DIR")
fi
CANDIDATES+=("$PWD")
if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
  CANDIDATES+=("${GITHUB_WORKSPACE}")
fi
CANDIDATES+=("$REPO_HINT")

PROJECT_ROOT=""
for candidate in "${CANDIDATES[@]}"; do
  if root="$(resolve_project_root "$candidate" 2>/dev/null)"; then
    PROJECT_ROOT="$root"
    break
  fi
done

if [[ -z "$PROJECT_ROOT" ]]; then
  echo "::error::Could not find pyproject.toml/setup.py for editable install." >&2
  echo "PWD=$PWD" >&2
  echo "GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-}" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
echo "[ci-install] project_root=$PROJECT_ROOT extras=${EXTRAS:-none}"

if [[ -n "$EXTRAS" ]]; then
  python -m pip install -e ".[${EXTRAS}]"
else
  python -m pip install -e .
fi
