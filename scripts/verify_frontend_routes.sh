#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-https://aragora.ai}"
shift || true

AUTH_HEADER="${VERIFY_FRONTEND_AUTH_HEADER:-}"
ANNOTATION_LEVEL="${VERIFY_FRONTEND_ANNOTATION_LEVEL:-error}"
SOFT_FAIL="${VERIFY_FRONTEND_SOFT_FAIL:-0}"

case "$ANNOTATION_LEVEL" in
  error|warning|notice) ;;
  *)
    echo "Invalid VERIFY_FRONTEND_ANNOTATION_LEVEL: $ANNOTATION_LEVEL (expected error|warning|notice)"
    exit 2
    ;;
esac

if [ "$#" -gt 0 ]; then
  ROUTES=("$@")
else
  ROUTES=(
    "/"
    "/oracle/"
    "/debates/"
    "/about/"
    "/pricing/"
    "/settings/"
  )
fi

NOT_FOUND_PATTERN='404 - PAGE NOT FOUND|The requested route does not exist in the Aragora network'
ERRORS=0

echo "Verifying frontend routes at ${BASE_URL}"
if [ -n "$AUTH_HEADER" ]; then
  echo "Using configured auth header for route verification"
fi

for route in "${ROUTES[@]}"; do
  url="${BASE_URL%/}${route}"
  tmp_file="$(mktemp)"
  cleaned_file="$(mktemp)"
  curl_args=(-sS -L --connect-timeout 10 --max-time 40 -o "${tmp_file}" -w "%{http_code}")
  if [ -n "$AUTH_HEADER" ]; then
    curl_args+=(-H "$AUTH_HEADER")
  fi
  status="$(curl "${curl_args[@]}" "${url}" || echo "000")"

  # Next.js embeds not-found markup inside script payloads on successful pages.
  # Remove script blocks so we only inspect rendered HTML content.
  perl -0777 -pe 's#<script\b[^>]*>.*?</script>##gsi' "${tmp_file}" > "${cleaned_file}"

  if [ "${status}" != "200" ]; then
    echo "::${ANNOTATION_LEVEL}::Route check failed for ${url} (status ${status})"
    ERRORS=1
  elif grep -Eq "${NOT_FOUND_PATTERN}" "${cleaned_file}"; then
    echo "::${ANNOTATION_LEVEL}::Route check failed for ${url} (rendered not-found content)"
    ERRORS=1
  else
    echo "OK ${url}"
  fi

  rm -f "${tmp_file}"
  rm -f "${cleaned_file}"
done

if [ "${ERRORS}" -ne 0 ]; then
  if [ "$SOFT_FAIL" = "1" ]; then
    echo "Route verification encountered failures in soft-fail mode"
    exit 0
  fi
  exit 1
fi
