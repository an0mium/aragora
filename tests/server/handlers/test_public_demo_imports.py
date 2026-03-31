from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_BIN = shutil.which("python3") or shutil.which("python") or sys.executable


def _probe_import(module_name: str, env_value: str | None) -> str | None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(REPO_ROOT)
    )
    script = f"""
import importlib
import json
import os

value = {env_value!r}
if value is None:
    os.environ.pop("ARAGORA_USE_SECRETS_MANAGER", None)
else:
    os.environ["ARAGORA_USE_SECRETS_MANAGER"] = value

importlib.import_module({module_name!r})
print(json.dumps({{"value": os.environ.get("ARAGORA_USE_SECRETS_MANAGER")}}))
"""
    result = subprocess.run(
        [PYTHON_BIN, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)["value"]


def test_playground_import_defaults_secrets_manager_off(monkeypatch):
    monkeypatch.delenv("ARAGORA_USE_SECRETS_MANAGER", raising=False)
    assert _probe_import("aragora.server.handlers.playground", None) == "false"


def test_public_viewer_import_defaults_secrets_manager_off(monkeypatch):
    monkeypatch.delenv("ARAGORA_USE_SECRETS_MANAGER", raising=False)
    assert _probe_import("aragora.server.handlers.debates.public_viewer", None) == "false"


def test_playground_import_preserves_explicit_secrets_manager_opt_in(monkeypatch):
    monkeypatch.setenv("ARAGORA_USE_SECRETS_MANAGER", "true")
    assert _probe_import("aragora.server.handlers.playground", "true") == "true"
