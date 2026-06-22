#!/usr/bin/env bash
# Onboarding smoke (HEALTH-5 #8262, P2 docs): prove a new contributor can go
# from a fresh checkout to a working CLI. From a throwaway /tmp virtualenv,
# install the package from the local checkout and exercise the documented
# zero-key aragora CLI surface (`--help` then the offline `demo`). Exits 0 only
# when the whole path works end to end. Run from anywhere inside a checkout.
set -euo pipefail

# AWS neutralization (library/environment.md): importing aragora.* can otherwise
# block on a botocore MFA getpass prompt (EOFError) in non-interactive runs.
export AWS_CONFIG_FILE=/dev/null
export AWS_SHARED_CREDENTIALS_FILE=/dev/null
export AWS_EC2_METADATA_DISABLED=true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_VENV="/tmp/aragora-onboarding-smoke-$$"
SMOKE_WORKDIR="/tmp/aragora-onboarding-smoke-work-$$"

cleanup() {
    rm -rf "$SMOKE_VENV" "$SMOKE_WORKDIR"
}
trap cleanup EXIT

rm -rf "$SMOKE_VENV" "$SMOKE_WORKDIR"
mkdir -p "$SMOKE_WORKDIR"

# 1. Fresh virtualenv under /tmp (never inside the repo checkout).
python3 -m venv /tmp/aragora-onboarding-smoke-$$
VENV_PY="$SMOKE_VENV/bin/python"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel --quiet

# 2. Install from the local checkout (the documented `pip install` path).
cd "$REPO_ROOT"
"$VENV_PY" -m pip install -e ".[test]" --quiet
# Runtime deps the CLI imports at startup that the pre-P3 root distribution does
# not yet declare in [project.dependencies]; P3 packaging folds these in.
"$VENV_PY" -m pip install --quiet httpx aiohttp websockets

# 3. Exercise the documented CLI surface from a scratch dir (keeps repo clean).
cd "$SMOKE_WORKDIR"
echo "== aragora --help =="
"$VENV_PY" -m aragora.cli.main --help >/dev/null
echo "== aragora demo (zero-key offline debate) =="
"$VENV_PY" -m aragora.cli.main demo

echo "onboarding smoke: OK (checkout -> venv -> install -> CLI demo)"
