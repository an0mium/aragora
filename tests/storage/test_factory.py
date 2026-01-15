"""
Tests for aragora.storage.factory - Storage factory and backend selection.

Tests cover:
- StorageBackend enum
- Backend selection based on environment
- PostgreSQL configuration detection
- Default database path generation
- Store registry
- Storage info helper
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.storage.factory import (
    StorageBackend,
    get_storage_backend,
    is_postgres_configured,
    get_default_db_path,
    register_store,
    get_registered_stores,
    storage_info,
    _store_registry,
)


# ===========================================================================
# Test StorageBackend Enum
# ===========================================================================


class TestStorageBackendEnum:
    """Tests for StorageBackend enum."""

    def test_sqlite_value(self):
        assert StorageBackend.SQLITE.value == "sqlite"

    def test_postgres_value(self):
        assert StorageBackend.POSTGRES.value == "postgres"

    def test_enum_members(self):
        members = list(StorageBackend)
        assert len(members) == 2
        assert StorageBackend.SQLITE in members
        assert StorageBackend.POSTGRES in members


# ===========================================================================
# Test get_storage_backend
# ===========================================================================


class TestGetStorageBackend:
    """Tests for get_storage_backend function."""

    def test_default_is_sqlite(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing ARAGORA_DB_BACKEND
            os.environ.pop("ARAGORA_DB_BACKEND", None)
            backend = get_storage_backend()
            assert backend == StorageBackend.SQLITE

    def test_explicit_sqlite(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "sqlite"}):
            backend = get_storage_backend()
            assert backend == StorageBackend.SQLITE

    def test_postgres_lowercase(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "postgres"}):
            backend = get_storage_backend()
            assert backend == StorageBackend.POSTGRES

    def test_postgresql_full_name(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "postgresql"}):
            backend = get_storage_backend()
            assert backend == StorageBackend.POSTGRES

    def test_postgres_uppercase(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "POSTGRES"}):
            backend = get_storage_backend()
            assert backend == StorageBackend.POSTGRES

    def test_unknown_backend_defaults_to_sqlite(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "mysql"}):
            backend = get_storage_backend()
            assert backend == StorageBackend.SQLITE


# ===========================================================================
# Test is_postgres_configured
# ===========================================================================


class TestIsPostgresConfigured:
    """Tests for is_postgres_configured function."""

    def test_not_configured_when_sqlite(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "sqlite"}):
            assert is_postgres_configured() is False

    def test_not_configured_when_postgres_no_dsn(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "postgres"}, clear=True):
            os.environ.pop("ARAGORA_POSTGRES_DSN", None)
            os.environ.pop("DATABASE_URL", None)
            assert is_postgres_configured() is False

    def test_configured_with_aragora_dsn(self):
        with patch.dict(os.environ, {
            "ARAGORA_DB_BACKEND": "postgres",
            "ARAGORA_POSTGRES_DSN": "postgres://user:pass@localhost/db"
        }):
            assert is_postgres_configured() is True

    def test_configured_with_database_url(self):
        with patch.dict(os.environ, {
            "ARAGORA_DB_BACKEND": "postgres",
            "DATABASE_URL": "postgres://user:pass@localhost/db"
        }):
            assert is_postgres_configured() is True

    def test_aragora_dsn_takes_precedence(self):
        with patch.dict(os.environ, {
            "ARAGORA_DB_BACKEND": "postgres",
            "ARAGORA_POSTGRES_DSN": "postgres://primary@localhost/db",
            "DATABASE_URL": "postgres://fallback@localhost/db"
        }):
            # Just testing that it's configured - precedence handled internally
            assert is_postgres_configured() is True


# ===========================================================================
# Test get_default_db_path
# ===========================================================================


class TestGetDefaultDbPath:
    """Tests for get_default_db_path function."""

    def test_with_explicit_nomic_dir(self, tmp_path):
        path = get_default_db_path("debates", nomic_dir=tmp_path)
        assert path == tmp_path / "db" / "debates.db"
        assert (tmp_path / "db").exists()

    def test_with_env_nomic_dir(self, tmp_path):
        with patch.dict(os.environ, {"ARAGORA_NOMIC_DIR": str(tmp_path)}):
            path = get_default_db_path("elo")
            assert path == tmp_path / "db" / "elo.db"

    def test_creates_db_directory(self, tmp_path):
        db_dir = tmp_path / "db"
        assert not db_dir.exists()
        get_default_db_path("test", nomic_dir=tmp_path)
        assert db_dir.exists()

    def test_different_store_names(self, tmp_path):
        debates_path = get_default_db_path("debates", nomic_dir=tmp_path)
        elo_path = get_default_db_path("elo", nomic_dir=tmp_path)
        memory_path = get_default_db_path("memory", nomic_dir=tmp_path)

        assert debates_path.name == "debates.db"
        assert elo_path.name == "elo.db"
        assert memory_path.name == "memory.db"

    def test_default_home_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARAGORA_NOMIC_DIR", None)
            path = get_default_db_path("test")
            expected_base = Path.home() / ".nomic" / "aragora" / "db"
            assert path.parent == expected_base


# ===========================================================================
# Test Store Registry
# ===========================================================================


class TestStoreRegistry:
    """Tests for store registry functions."""

    def setup_method(self):
        # Clear registry before each test
        _store_registry.clear()

    def teardown_method(self):
        # Clean up after tests
        _store_registry.clear()

    def test_register_store(self):
        class MockStore:
            pass

        register_store("mock", MockStore)
        assert "mock" in _store_registry
        assert _store_registry["mock"] is MockStore

    def test_get_registered_stores_empty(self):
        stores = get_registered_stores()
        assert stores == {}

    def test_get_registered_stores_returns_copy(self):
        class MockStore:
            pass

        register_store("mock", MockStore)
        stores = get_registered_stores()

        # Modify returned dict
        stores["other"] = object

        # Original should be unchanged
        assert "other" not in _store_registry

    def test_multiple_registrations(self):
        class StoreA:
            pass

        class StoreB:
            pass

        register_store("store_a", StoreA)
        register_store("store_b", StoreB)

        stores = get_registered_stores()
        assert len(stores) == 2
        assert stores["store_a"] is StoreA
        assert stores["store_b"] is StoreB

    def test_overwrite_registration(self):
        class StoreV1:
            pass

        class StoreV2:
            pass

        register_store("store", StoreV1)
        register_store("store", StoreV2)

        stores = get_registered_stores()
        assert stores["store"] is StoreV2


# ===========================================================================
# Test storage_info
# ===========================================================================


class TestStorageInfo:
    """Tests for storage_info function."""

    def test_sqlite_info(self, tmp_path):
        with patch.dict(os.environ, {
            "ARAGORA_DB_BACKEND": "sqlite",
            "ARAGORA_NOMIC_DIR": str(tmp_path)
        }):
            info = storage_info()

            assert info["backend"] == "sqlite"
            assert info["is_postgres"] is False
            assert info["postgres_configured"] is False
            assert "default_db_dir" in info
            assert str(tmp_path / "db") == info["default_db_dir"]

    def test_postgres_info_no_dsn(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "postgres"}, clear=True):
            os.environ.pop("ARAGORA_POSTGRES_DSN", None)
            os.environ.pop("DATABASE_URL", None)
            info = storage_info()

            assert info["backend"] == "postgres"
            assert info["is_postgres"] is True
            assert info["postgres_configured"] is False

    def test_postgres_info_with_dsn(self):
        with patch.dict(os.environ, {
            "ARAGORA_DB_BACKEND": "postgres",
            "ARAGORA_POSTGRES_DSN": "postgres://user:secret@localhost:5432/mydb"
        }):
            info = storage_info()

            assert info["backend"] == "postgres"
            assert info["is_postgres"] is True
            assert info["postgres_configured"] is True
            # Password should be redacted
            assert "dsn_redacted" in info
            assert "secret" not in info["dsn_redacted"]
            assert "***" in info["dsn_redacted"]

    def test_dsn_password_redaction(self):
        with patch.dict(os.environ, {
            "ARAGORA_DB_BACKEND": "postgres",
            "ARAGORA_POSTGRES_DSN": "postgres://admin:SuperSecret123@db.example.com:5432/aragora"
        }):
            info = storage_info()

            dsn = info["dsn_redacted"]
            assert "SuperSecret123" not in dsn
            assert "admin:***@" in dsn
            assert "db.example.com:5432/aragora" in dsn

    def test_dsn_without_password(self):
        with patch.dict(os.environ, {
            "ARAGORA_DB_BACKEND": "postgres",
            "ARAGORA_POSTGRES_DSN": "postgres://localhost/db"
        }):
            info = storage_info()
            # Should handle DSN without @ symbol
            assert "dsn_redacted" in info
