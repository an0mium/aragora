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


def test_export_openapi_writes_primary_and_generated_artifacts(monkeypatch, tmp_path: Path) -> None:
    schema = {
        "openapi": "3.1.0",
        "info": {"title": "Aragora", "version": "test"},
        "paths": {},
    }

    monkeypatch.setattr(export_openapi, "generate_openapi_schema", lambda: schema)

    export_openapi.main(["--output-dir", str(tmp_path)])

    for name in export_openapi.ARTIFACT_NAMES:
        path = tmp_path / name
        assert path.exists(), f"Missing exported artifact: {path}"
        if name == export_openapi.GENERATED_YAML_NAME:
            assert path.read_text()
        else:
            assert path.read_text().startswith("{\n")
