from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import scripts.export_openapi as export_openapi


def test_export_openapi_prefers_repo_checkout(monkeypatch) -> None:
    repo_root = str(Path(export_openapi.__file__).resolve().parents[1])
    pruned_path = [entry for entry in sys.path if entry != repo_root]

    monkeypatch.setattr(sys, "path", pruned_path)
    monkeypatch.delenv("ARAGORA_USE_SECRETS_MANAGER", raising=False)

    reloaded = importlib.reload(export_openapi)

    assert repo_root in sys.path
    assert os.environ["ARAGORA_USE_SECRETS_MANAGER"] == "false"
    assert str(reloaded.PROJECT_ROOT) == repo_root
