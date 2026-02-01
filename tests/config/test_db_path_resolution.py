from __future__ import annotations

import importlib
import subprocess
import sys
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


def test_get_nomic_dir_respects_aragora_data_dir(tmp_path, monkeypatch):
    """get_nomic_dir() should return ARAGORA_DATA_DIR when set."""
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    import aragora.persistence.db_config as db_config

    db_config = importlib.reload(db_config)
    assert db_config.get_nomic_dir() == tmp_path


def test_get_nomic_dir_falls_back_to_nomic_dir(tmp_path, monkeypatch):
    """get_nomic_dir() should fall back to ARAGORA_NOMIC_DIR."""
    monkeypatch.delenv("ARAGORA_DATA_DIR", raising=False)
    monkeypatch.setenv("ARAGORA_NOMIC_DIR", str(tmp_path))
    import aragora.persistence.db_config as db_config

    db_config = importlib.reload(db_config)
    assert db_config.get_nomic_dir() == tmp_path


def test_get_nomic_dir_default_is_nomic(monkeypatch):
    """get_nomic_dir() should default to .nomic when no env var is set."""
    monkeypatch.delenv("ARAGORA_DATA_DIR", raising=False)
    monkeypatch.delenv("ARAGORA_NOMIC_DIR", raising=False)
    import aragora.persistence.db_config as db_config

    db_config = importlib.reload(db_config)
    assert db_config.get_nomic_dir() == Path(".nomic")


def test_resolve_db_path_absolute_passthrough():
    """Absolute paths should be returned as-is."""
    import aragora.config.legacy as legacy

    result = legacy.resolve_db_path("/absolute/path/to/db.sqlite")
    assert result == "/absolute/path/to/db.sqlite"


def test_resolve_db_path_memory_passthrough():
    """SQLite :memory: should be preserved."""
    import aragora.config.legacy as legacy

    assert legacy.resolve_db_path(":memory:") == ":memory:"


def test_resolve_db_path_file_uri_passthrough():
    """SQLite file: URIs should be preserved."""
    import aragora.config.legacy as legacy

    assert legacy.resolve_db_path("file:test?mode=memory").startswith("file:")


def test_guard_repo_clean_scan_paths():
    """guard_repo_clean.py --scan-paths should pass on the codebase."""
    result = subprocess.run(
        [sys.executable, "scripts/guard_repo_clean.py", "--scan-paths"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"guard_repo_clean.py --scan-paths failed:\n{result.stdout}\n{result.stderr}"
    )


def test_guard_repo_clean_no_tracked_artifacts():
    """guard_repo_clean.py should pass (no tracked .db files)."""
    result = subprocess.run(
        [sys.executable, "scripts/guard_repo_clean.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"guard_repo_clean.py failed:\n{result.stdout}\n{result.stderr}"
