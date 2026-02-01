from __future__ import annotations

import importlib
from pathlib import Path


def _reload_for_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    import aragora.config.legacy as legacy

    legacy = importlib.reload(legacy)
    import aragora.storage.schema as schema

    schema = importlib.reload(schema)
    return legacy, schema


def test_resolve_db_path_uses_data_dir(tmp_path, monkeypatch):
    legacy, _schema = _reload_for_data_dir(tmp_path, monkeypatch)
    resolved = Path(legacy.resolve_db_path("example.db"))
    assert resolved.parent == tmp_path
    assert resolved.name == "example.db"


def test_database_manager_resolves_relative_paths(tmp_path, monkeypatch):
    _legacy, schema = _reload_for_data_dir(tmp_path, monkeypatch)
    schema.DatabaseManager._instances.clear()

    manager = schema.DatabaseManager.get_instance("manager_test.db")
    assert str(tmp_path) in manager.db_path
