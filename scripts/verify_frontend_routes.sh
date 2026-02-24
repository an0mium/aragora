#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-https://aragora.pages.dev}"
shift || true

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

for route in "${ROUTES[@]}"; do
  url="${BASE_URL%/}${route}"
  tmp_file="$(mktemp)"
  cleaned_file="$(mktemp)"
  status="$(curl -sS -L --connect-timeout 10 --max-time 40 -o "${tmp_file}" -w "%{http_code}" "${url}" || echo "000")"

  # Next.js embeds not-found markup inside script payloads on successful pages.
  # Remove script blocks so we only inspect rendered HTML content.
  perl -0777 -pe 's#<script\b[^>]*>.*?</script>##gsi' "${tmp_file}" > "${cleaned_file}"

  if [ "${status}" != "200" ]; then
    echo "::error::Route check failed for ${url} (status ${status})"
    ERRORS=1
  elif grep -Eq "${NOT_FOUND_PATTERN}" "${cleaned_file}"; then
    echo "::error::Route check failed for ${url} (rendered not-found content)"
    ERRORS=1
  else
    echo "OK ${url}"
  fi

  rm -f "${tmp_file}"
  rm -f "${cleaned_file}"
done

if [ "${ERRORS}" -ne 0 ]; then
  exit 1
fi
