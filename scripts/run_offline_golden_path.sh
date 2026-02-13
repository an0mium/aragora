#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[offline-golden-path] Installing dev/test dependencies"
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"

echo "[offline-golden-path] Running offline behavior tests"
python -m pytest tests/cli/test_offline_golden_path.py -v --timeout=120 --tb=short

echo "[offline-golden-path] Running offline demo CLI smoke"
ARAGORA_OFFLINE=1 aragora ask "Offline mode smoke" --demo --rounds 1

echo "[offline-golden-path] PASS"
