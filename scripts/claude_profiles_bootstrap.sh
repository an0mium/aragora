#!/usr/bin/env bash
# Sequential Claude profile login/status/verify helper.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  claude_profiles_bootstrap.sh login [--force] [profile-name...]
  claude_profiles_bootstrap.sh status [profile-name...]
  claude_profiles_bootstrap.sh verify [profile-name...]

  verify  - live-probe each profile to detect expired tokens
            (status can say loggedIn:true even when the token is expired)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_TOOL="${SCRIPT_DIR}/claude_profile.sh"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift

FORCE_LOGIN=0
if [[ "$MODE" == "login" && $# -gt 0 && "$1" == "--force" ]]; then
  FORCE_LOGIN=1
  shift
fi

default_profiles=(
  max-01
  max-02
  max-03
  max-04
  max-05
  max-06
  max-07
  max-08
  max-09
  max-10
  max-11
  max-12
  max-13
)

if [[ $# -gt 0 ]]; then
  profiles=("$@")
else
  profiles=("${default_profiles[@]}")
fi

already_logged_in() {
  local profile="$1"
  local status_output

  if ! status_output="$("${PROFILE_TOOL}" status "$profile" 2>/dev/null)"; then
    return 1
  fi
  if ! grep -q '"loggedIn": true' <<<"$status_output"; then
    return 1
  fi

  # Status says logged in, but the token might be expired.
  # Run a live probe to verify the token actually works.
  if ! timeout 15 "${PROFILE_TOOL}" exec "$profile" -- claude -p "ok" </dev/null >/dev/null 2>&1; then
    echo "  Token expired (status says logged in but live probe failed)"
    return 1
  fi
  return 0
}

verify_profile() {
  local profile="$1"

  printf "  %-10s " "$profile"

  local status_output
  if ! status_output="$("${PROFILE_TOOL}" status "$profile" 2>/dev/null)"; then
    echo "NOT CONFIGURED"
    return 1
  fi
  if ! grep -q '"loggedIn": true' <<<"$status_output"; then
    echo "NOT LOGGED IN"
    return 1
  fi

  # Live probe with /dev/null stdin and 15s timeout
  local probe_output
  if probe_output="$(timeout 15 "${PROFILE_TOOL}" exec "$profile" -- claude -p "ok" </dev/null 2>&1)"; then
    local email
    email="$(grep -o '"email": "[^"]*"' <<<"$status_output" | head -1 | sed 's/"email": "//;s/"//')"
    echo "OK  ($email)"
    return 0
  else
    local email
    email="$(grep -o '"email": "[^"]*"' <<<"$status_output" | head -1 | sed 's/"email": "//;s/"//')"
    local err_line
    err_line="$(echo "$probe_output" | grep -i -m1 'expired\|401\|error\|failed' || echo "$probe_output" | tail -1)"
    echo "EXPIRED  ($email)"
    return 1
  fi
}

print_status() {
  local profile="$1"
  echo
  echo "=== ${profile} ==="
  "${PROFILE_TOOL}" status "$profile" || true
}

profile_home() {
  "${PROFILE_TOOL}" home "$1"
}

launch_login_with_profile() {
  local profile_home="$1"
  HOME="$profile_home" \
  XDG_CONFIG_HOME="${profile_home}/.config" \
  CLAUDE_CONFIG_DIR="${profile_home}/.claude" \
  PATH="${PATH}" \
  env \
    -u ANTHROPIC_API_KEY \
    -u CLAUDECODE \
    -u CLAUDE_CODE_ENTRYPOINT \
    ARAGORA_OPENROUTER_FALLBACK_ENABLED=false \
    OPENROUTER_API_KEY= \
    claude auth login
}

extract_auth_url() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"https://claude\.ai/oauth/authorize\?[^\s]+", text)
print(matches[-1] if matches else "")
PY
}

extract_state_from_url() {
  local auth_url="$1"
  python3 - "$auth_url" <<'PY'
import sys
import urllib.parse

parsed = urllib.parse.urlparse(sys.argv[1])
query = urllib.parse.parse_qs(parsed.query)
print(query.get("state", [""])[0])
PY
}

wait_for_auth_url() {
  local pid="$1"
  local log_file="$2"
  local attempts="${3:-100}"
  local auth_url=""

  for ((i = 0; i < attempts; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    auth_url="$(extract_auth_url "$log_file")"
    if [[ -n "$auth_url" ]]; then
      printf '%s\n' "$auth_url"
      return 0
    fi
    sleep 0.2
  done

  printf '%s\n' ""
  return 1
}

wait_for_callback_port() {
  local pid="$1"
  local attempts="${2:-100}"
  local port=""

  for ((i = 0; i < attempts; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    port="$(lsof -a -p "$pid" -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $9}' | sed -E 's/.*:([0-9]+)$/\1/' | head -n1)"
    if [[ -n "$port" ]]; then
      printf '%s\n' "$port"
      return 0
    fi
    sleep 0.2
  done

  printf '%s\n' ""
  return 1
}

post_callback() {
  local port="$1"
  local code="$2"
  local state="$3"
  local encoded_code
  local encoded_state
  encoded_code="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$code")"
  encoded_state="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$state")"
  curl -g -fsS "http://[::1]:${port}/callback?code=${encoded_code}&state=${encoded_state}" >/dev/null
}

get_profile_email() {
  local profile="$1"
  local status_output
  status_output="$("${PROFILE_TOOL}" status "$profile" 2>/dev/null)" || true
  grep -o '"email": "[^"]*"' <<<"$status_output" | head -1 | sed 's/"email": "//;s/"//'
}

login_profile_manual_code() {
  local profile="$1"
  local profile_home_path
  local auth_log
  local pid=""
  local auth_url=""
  local callback_port=""
  local state=""
  local auth_result=""
  local code=""
  local pasted_state=""

  profile_home_path="$(profile_home "$profile")"
  mkdir -p "${profile_home_path}/.claude" "${profile_home_path}/.config"
  auth_log="$(mktemp -t "claude-auth-${profile}.XXXXXX")"

  # Show expected account so the user knows which Google account to pick
  local expected_email
  expected_email="$(get_profile_email "$profile")"
  if [[ -n "$expected_email" ]]; then
    echo "  Account: $expected_email"
  fi

  # Logout first so the login flow gets a clean OAuth redirect
  # (otherwise claude auth login may silently reuse the expired session)
  "${PROFILE_TOOL}" logout "$profile" >/dev/null 2>&1 || true

  cleanup_login() {
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$auth_log"
  }
  trap cleanup_login RETURN

  launch_login_with_profile "$profile_home_path" >"$auth_log" 2>&1 &
  pid=$!

  auth_url="$(wait_for_auth_url "$pid" "$auth_log" || true)"
  callback_port="$(wait_for_callback_port "$pid" || true)"

  if [[ -z "$auth_url" || -z "$callback_port" ]]; then
    echo "Failed to initialize Claude login flow for ${profile}."
    cat "$auth_log"
    return 1
  fi

  state="$(extract_state_from_url "$auth_url")"

  echo "  Profile home: ${profile_home_path}"
  if [[ -n "$expected_email" ]]; then
    echo
    echo "  >>> Log in as: $expected_email <<<"
  fi
  echo
  echo "  If the browser didn't open, visit:"
  echo "  ${auth_url}"
  echo
  echo "  After Google OAuth completes, paste the returned code or code#state:"

  while true; do
    if already_logged_in "$profile"; then
      wait "$pid" || true
      echo "Completed login flow."
      "${PROFILE_TOOL}" status "$profile"
      return 0
    fi

    if read -r -t 1 auth_result; then
      break
    fi
  done

  code="${auth_result%%#*}"
  pasted_state=""
  if [[ "$auth_result" == *"#"* ]]; then
    pasted_state="${auth_result#*#}"
  fi
  if [[ -n "$pasted_state" ]]; then
    state="$pasted_state"
  fi

  if [[ -z "$code" || -z "$state" ]]; then
    echo "Missing code or state; cannot complete callback."
    cat "$auth_log"
    return 1
  fi

  if ! post_callback "$callback_port" "$code" "$state"; then
    echo "Failed to post callback for ${profile}."
    cat "$auth_log"
    return 1
  fi

  wait "$pid" || true
  echo "Completed login flow."
  "${PROFILE_TOOL}" status "$profile"
}

case "$MODE" in
  status)
    for profile in "${profiles[@]}"; do
      print_status "$profile"
    done
    ;;
  verify)
    echo "Live-probing ${#profiles[@]} profiles..."
    echo
    pass=0
    fail=0
    for profile in "${profiles[@]}"; do
      if verify_profile "$profile"; then
        ((pass++)) || true
      else
        ((fail++)) || true
      fi
    done
    echo
    echo "${pass} passed, ${fail} failed"
    if [[ $fail -gt 0 ]]; then
      echo "Run: $0 login [profile...] to fix expired tokens"
      exit 1
    fi
    ;;
  login)
    for profile in "${profiles[@]}"; do
      echo
      echo "=== ${profile} ==="
      if [[ "$FORCE_LOGIN" -ne 1 ]] && already_logged_in "$profile"; then
        echo "Already logged in and verified; skipping. ($(get_profile_email "$profile"))"
        continue
      fi
      login_profile_manual_code "$profile"
    done
    ;;
  *)
    usage
    exit 1
    ;;
esac
