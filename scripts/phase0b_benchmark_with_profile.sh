#!/usr/bin/env bash
# Run the Phase 0B benchmark tooling through an explicit Claude profile.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  phase0b_benchmark_with_profile.sh [--profile PROFILE] [--show-auth] <benchmark-args...>
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_TOOL="${SCRIPT_DIR}/claude_profile.sh"
BENCHMARK_TOOL="${SCRIPT_DIR}/phase0b_role_benchmark.py"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="${CLAUDE_BENCHMARK_PROFILE:-max-12}"
SHOW_AUTH=0

resolve_python() {
  if [[ -n "${ARAGORA_BENCHMARK_PYTHON:-}" ]]; then
    printf '%s\n' "${ARAGORA_BENCHMARK_PYTHON}"
    return 0
  fi

  if [[ -f "${PROJECT_ROOT}/.python-version" ]]; then
    local version
    version="$(tr -d '[:space:]' < "${PROJECT_ROOT}/.python-version")"
    local pyenv_python="${HOME}/.pyenv/versions/${version}/bin/python"
    if [[ -x "${pyenv_python}" ]]; then
      printf '%s\n' "${pyenv_python}"
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  die "could not resolve a Python interpreter; set ARAGORA_BENCHMARK_PYTHON"
}

PYTHON_BIN="$(resolve_python)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      PROFILE="$2"
      shift 2
      ;;
    --show-auth)
      SHOW_AUTH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -gt 0 ]] || {
  usage
  exit 1
}

STATUS_JSON="$("${PROFILE_TOOL}" status "${PROFILE}" 2>/dev/null || true)"
if [[ -z "${STATUS_JSON}" ]] || ! grep -q '"loggedIn": true' <<<"${STATUS_JSON}"; then
  die "Claude profile ${PROFILE} is not logged in. Run: ${PROFILE_TOOL} login ${PROFILE}"
fi

if [[ "${SHOW_AUTH}" -eq 1 ]]; then
  echo "Claude profile: ${PROFILE}"
  echo "${STATUS_JSON}"
fi

exec "${PROFILE_TOOL}" exec "${PROFILE}" -- env ARAGORA_SKIP_SECRETS="${ARAGORA_SKIP_SECRETS:-1}" "${PYTHON_BIN}" "${BENCHMARK_TOOL}" "$@"
