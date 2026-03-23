#!/usr/bin/env bash
# Explicit Codex CLI profile selector for local Aragora runs.
#
# This helper isolates Codex auth state via CODEX_HOME so local fixed-cost
# Codex subscriptions can coexist with Aragora's own secure-store/API-key paths.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  codex_profile.sh home <profile-name-or-home>
  codex_profile.sh status <profile-name-or-home>
  codex_profile.sh login <profile-name-or-home> [extra codex login args...]
  codex_profile.sh logout <profile-name-or-home>
  codex_profile.sh exec <profile-name-or-home> -- <command...>

Examples:
  scripts/codex_profile.sh status pro-01
  scripts/codex_profile.sh login pro-02 --device-auth
  scripts/codex_profile.sh exec pro-01 -- codex
  scripts/codex_profile.sh exec pro-01 -- \
    python3 -m aragora.cli.main quickstart --demo --no-browser

Conventions:
  Bare profile names resolve under ~/.aragora-codex by default.
  <profile-home> is a directory used as CODEX_HOME for the command it runs.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

MODE="$1"
PROFILE_HOME="$2"
shift 2

require_command codex

ORIGINAL_HOME="$HOME"
PROFILE_ROOT="${CODEX_PROFILE_ROOT:-${ORIGINAL_HOME}/.aragora-codex}"

resolve_profile_home() {
  local raw_path="$1"

  case "$raw_path" in
    "~") raw_path="${HOME}" ;;
    "~"/*) raw_path="${HOME}${raw_path#"~"}" ;;
    /*) ;;
    ./*|../*)
      raw_path="$(cd "$(dirname "$raw_path")" && pwd)/$(basename "$raw_path")"
      ;;
    *)
      raw_path="${PROFILE_ROOT}/${raw_path}"
      ;;
  esac

  local raw_parent
  raw_parent="$(dirname "$raw_path")"
  mkdir -p "$raw_parent"
  raw_path="$(cd "$raw_parent" && pwd)/$(basename "$raw_path")"
  mkdir -p "$raw_path"
  printf '%s\n' "$raw_path"
}

PROFILE_HOME="$(resolve_profile_home "$PROFILE_HOME")"

run_with_profile() {
  CODEX_HOME="${PROFILE_HOME}" \
  PATH="${PATH}" \
  env \
    -u ANTHROPIC_API_KEY \
    -u OPENAI_API_KEY \
    "$@"
}

case "$MODE" in
  home)
    if [[ $# -ne 0 ]]; then
      die "home does not take extra arguments"
    fi
    printf '%s\n' "$PROFILE_HOME"
    ;;
  status)
    if [[ $# -ne 0 ]]; then
      die "status does not take extra arguments"
    fi
    run_with_profile codex login status
    ;;
  login)
    echo "Using CODEX_HOME: ${PROFILE_HOME}"
    run_with_profile codex login "$@"
    ;;
  logout)
    if [[ $# -ne 0 ]]; then
      die "logout does not take extra arguments"
    fi
    run_with_profile codex logout
    ;;
  exec)
    if [[ $# -lt 2 || "$1" != "--" ]]; then
      die "exec requires '-- <command...>'"
    fi
    shift
    echo "Using CODEX_HOME: ${PROFILE_HOME}"
    echo "Command: $*"
    run_with_profile "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
