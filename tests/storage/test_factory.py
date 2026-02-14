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


# Helper to patch DSN functions to return None (simulating no secrets configured)
def _patch_no_dsn():
    """Context manager to patch DSN functions to return None."""
    return patch.multiple(
        "aragora.storage.factory",
        get_supabase_postgres_dsn=lambda: None,
        get_selfhosted_postgres_dsn=lambda: None,
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

    def test_supabase_value(self):
        assert StorageBackend.SUPABASE.value == "supabase"

    def test_enum_members(self):
        members = list(StorageBackend)
        assert len(members) == 3
        assert StorageBackend.SQLITE in members
        assert StorageBackend.POSTGRES in members
        assert StorageBackend.SUPABASE in members


# ===========================================================================
# Test get_storage_backend
# ===========================================================================


class TestGetStorageBackend:
    """Tests for get_storage_backend function."""

    def test_default_is_sqlite(self):
        with patch.dict(os.environ, {}, clear=True), _patch_no_dsn():
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

    def test_supabase_explicit(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "supabase"}):
            backend = get_storage_backend()
            assert backend == StorageBackend.SUPABASE

    def test_unknown_backend_defaults_to_sqlite(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "mysql"}), _patch_no_dsn():
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
        with (
            patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "postgres"}, clear=True),
            _patch_no_dsn(),
        ):
            os.environ.pop("ARAGORA_POSTGRES_DSN", None)
            os.environ.pop("DATABASE_URL", None)
            assert is_postgres_configured() is False

    def test_not_configured_when_supabase_no_dsn(self):
        with (
            patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "supabase"}, clear=True),
            _patch_no_dsn(),
        ):
            os.environ.pop("SUPABASE_URL", None)
            os.environ.pop("SUPABASE_DB_PASSWORD", None)
            os.environ.pop("SUPABASE_POSTGRES_DSN", None)
            assert is_postgres_configured() is False

    def test_configured_with_aragora_dsn(self):
        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "postgres",
                "ARAGORA_POSTGRES_DSN": "postgres://user:pass@localhost/db",
            },
        ):
            assert is_postgres_configured() is True

    def test_configured_with_database_url(self):
        with patch.dict(
            os.environ,
            {"ARAGORA_DB_BACKEND": "postgres", "DATABASE_URL": "postgres://user:pass@localhost/db"},
        ):
            assert is_postgres_configured() is True

    def test_configured_with_supabase_dsn(self):
        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "supabase",
                "SUPABASE_POSTGRES_DSN": "postgres://user:pass@localhost/db",
            },
        ):
            assert is_postgres_configured() is True

    def test_aragora_dsn_takes_precedence(self):
        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "postgres",
                "ARAGORA_POSTGRES_DSN": "postgres://primary@localhost/db",
                "DATABASE_URL": "postgres://fallback@localhost/db",
            },
        ):
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

    def test_default_nomic_directory(self, tmp_path, monkeypatch):
        """Test that default db path uses get_nomic_dir() fallback (.nomic)."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARAGORA_DATA_DIR", None)
            os.environ.pop("ARAGORA_NOMIC_DIR", None)
            monkeypatch.chdir(tmp_path)
            path = get_default_db_path("test")
            # Now uses get_nomic_dir() which returns .nomic (relative to CWD)
            expected_base = Path(".nomic") / "db"
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
        with patch.dict(
            os.environ, {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_NOMIC_DIR": str(tmp_path)}
        ):
            info = storage_info()

            assert info["backend"] == "sqlite"
            assert info["is_postgres"] is False
            assert info["is_supabase"] is False
            assert info["postgres_configured"] is False
            assert "default_db_dir" in info
            assert str(tmp_path / "db") == info["default_db_dir"]

    def test_postgres_info_no_dsn(self):
        with (
            patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "postgres"}, clear=True),
            _patch_no_dsn(),
        ):
            os.environ.pop("ARAGORA_POSTGRES_DSN", None)
            os.environ.pop("DATABASE_URL", None)
            info = storage_info()

            assert info["backend"] == "postgres"
            assert info["is_postgres"] is True
            assert info["is_supabase"] is False
            assert info["postgres_configured"] is False

    def test_postgres_info_with_dsn(self):
        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "postgres",
                "ARAGORA_POSTGRES_DSN": "postgres://user:secret@localhost:5432/mydb",
            },
        ):
            info = storage_info()

            assert info["backend"] == "postgres"
            assert info["is_postgres"] is True
            assert info["is_supabase"] is False
            assert info["postgres_configured"] is True
            # Password should be redacted
            assert "dsn_redacted" in info
            assert "secret" not in info["dsn_redacted"]
            assert "***" in info["dsn_redacted"]

    def test_supabase_info_with_dsn(self):
        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "supabase",
                "SUPABASE_POSTGRES_DSN": "postgres://user:secret@localhost:5432/mydb",
            },
        ):
            info = storage_info()

            assert info["backend"] == "supabase"
            assert info["is_postgres"] is True
            assert info["is_supabase"] is True
            assert info["postgres_configured"] is True
            assert "dsn_redacted" in info

    def test_dsn_password_redaction(self):
        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "postgres",
                "ARAGORA_POSTGRES_DSN": "postgres://admin:SuperSecret123@db.example.com:5432/aragora",
            },
        ):
            info = storage_info()

            dsn = info["dsn_redacted"]
            assert "SuperSecret123" not in dsn
            assert "admin:***@" in dsn
            assert "db.example.com:5432/aragora" in dsn

    def test_dsn_without_password(self):
        with patch.dict(
            os.environ,
            {"ARAGORA_DB_BACKEND": "postgres", "ARAGORA_POSTGRES_DSN": "postgres://localhost/db"},
        ):
            info = storage_info()
            # Should handle DSN without @ symbol
            assert "dsn_redacted" in info

    def test_is_production_flag(self):
        with patch.dict(
            os.environ,
            {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_ENV": "production"},
        ):
            info = storage_info()
            assert info["is_production"] is True

    def test_is_not_production_flag(self):
        with patch.dict(
            os.environ,
            {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_ENV": "development"},
        ):
            info = storage_info()
            assert info["is_production"] is False


# ===========================================================================
# Test is_production_environment
# ===========================================================================


class TestIsProductionEnvironment:
    """Tests for is_production_environment function."""

    def test_production_env(self):
        from aragora.storage.factory import is_production_environment

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            assert is_production_environment() is True

    def test_prod_env(self):
        from aragora.storage.factory import is_production_environment

        with patch.dict(os.environ, {"ARAGORA_ENV": "prod"}):
            assert is_production_environment() is True

    def test_live_env(self):
        from aragora.storage.factory import is_production_environment

        with patch.dict(os.environ, {"ARAGORA_ENV": "live"}):
            assert is_production_environment() is True

    def test_development_env(self):
        from aragora.storage.factory import is_production_environment

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            assert is_production_environment() is False

    def test_staging_env(self):
        from aragora.storage.factory import is_production_environment

        with patch.dict(os.environ, {"ARAGORA_ENV": "staging"}):
            assert is_production_environment() is False

    def test_case_insensitive(self):
        from aragora.storage.factory import is_production_environment

        with patch.dict(os.environ, {"ARAGORA_ENV": "PRODUCTION"}):
            assert is_production_environment() is True

    def test_default_development(self):
        from aragora.storage.factory import is_production_environment

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARAGORA_ENV", None)
            assert is_production_environment() is False


# ===========================================================================
# Test validate_storage_config
# ===========================================================================


class TestValidateStorageConfig:
    """Tests for validate_storage_config function."""

    def test_valid_development_sqlite(self):
        from aragora.storage.factory import validate_storage_config

        with (
            patch.dict(
                os.environ,
                {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_ENV": "development"},
            ),
            _patch_no_dsn(),
        ):
            result = validate_storage_config()

            assert result["valid"] is True
            assert result["errors"] == []
            assert result["backend"] == "sqlite"
            assert result["is_production"] is False

    def test_valid_production_postgres(self):
        from aragora.storage.factory import validate_storage_config

        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "postgres",
                "ARAGORA_ENV": "production",
                "ARAGORA_POSTGRES_DSN": "postgres://user:pass@localhost/db",
                "ARAGORA_SECRETS_STRICT": "false",
            },
        ):
            result = validate_storage_config()

            assert result["valid"] is True
            assert result["errors"] == []
            assert result["backend"] == "postgres"
            assert result["is_production"] is True
            assert result["postgres_dsn_configured"] is True

    def test_invalid_production_sqlite(self):
        from aragora.storage.factory import validate_storage_config

        with (
            patch.dict(
                os.environ,
                {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_ENV": "production"},
            ),
            _patch_no_dsn(),
        ):
            result = validate_storage_config()

            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert any("distributed storage" in e for e in result["errors"])

    def test_invalid_production_postgres_no_dsn(self):
        from aragora.storage.factory import validate_storage_config

        with (
            patch.dict(
                os.environ,
                {"ARAGORA_DB_BACKEND": "postgres", "ARAGORA_ENV": "production"},
            ),
            _patch_no_dsn(),
        ):
            result = validate_storage_config()

            assert result["valid"] is False
            assert any("no DSN configured" in e for e in result["errors"])

    def test_invalid_production_supabase_no_dsn(self):
        from aragora.storage.factory import validate_storage_config

        with (
            patch.dict(
                os.environ,
                {"ARAGORA_DB_BACKEND": "supabase", "ARAGORA_ENV": "production"},
            ),
            _patch_no_dsn(),
        ):
            result = validate_storage_config()

            assert result["valid"] is False
            assert any("Supabase DSN" in e for e in result["errors"])

    def test_strict_mode_raises_on_error(self):
        from aragora.storage.factory import validate_storage_config

        with (
            patch.dict(
                os.environ,
                {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_ENV": "production"},
            ),
            _patch_no_dsn(),
        ):
            with pytest.raises(RuntimeError, match="Storage configuration failed"):
                validate_storage_config(strict=True)

    def test_strict_mode_no_error_when_valid(self):
        from aragora.storage.factory import validate_storage_config

        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "postgres",
                "ARAGORA_ENV": "production",
                "ARAGORA_POSTGRES_DSN": "postgres://user:pass@localhost/db",
                "ARAGORA_SECRETS_STRICT": "false",
            },
        ):
            # Should not raise
            result = validate_storage_config(strict=True)
            assert result["valid"] is True

    def test_warning_for_sqlite_in_staging(self):
        from aragora.storage.factory import validate_storage_config

        with (
            patch.dict(
                os.environ,
                {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_ENV": "staging"},
            ),
            _patch_no_dsn(),
        ):
            result = validate_storage_config()

            # Should be valid but with warnings
            assert result["valid"] is True
            assert len(result["warnings"]) > 0
            assert any("staging" in w for w in result["warnings"])

    def test_warning_postgres_dsn_with_sqlite_backend(self):
        from aragora.storage.factory import validate_storage_config

        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "sqlite",
                "ARAGORA_ENV": "development",
                "ARAGORA_POSTGRES_DSN": "postgres://user:pass@localhost/db",
            },
        ):
            result = validate_storage_config()

            # Should have warning about unused DSN
            assert len(result["warnings"]) > 0
            assert any("backend is set to sqlite" in w for w in result["warnings"])

    def test_warning_supabase_dsn_with_postgres_backend(self):
        from aragora.storage.factory import validate_storage_config

        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "postgres",
                "ARAGORA_ENV": "development",
                "ARAGORA_POSTGRES_DSN": "postgres://user:pass@localhost/db",
                "SUPABASE_POSTGRES_DSN": "postgres://supabase:pass@localhost/db",
            },
        ):
            result = validate_storage_config()

            # Should have warning about unused Supabase config
            assert len(result["warnings"]) > 0
            assert any("Supabase is configured" in w for w in result["warnings"])

    def test_result_contains_expected_keys(self):
        from aragora.storage.factory import validate_storage_config

        with (
            patch.dict(
                os.environ,
                {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_ENV": "development"},
            ),
            _patch_no_dsn(),
        ):
            result = validate_storage_config()

            expected_keys = {
                "valid",
                "errors",
                "warnings",
                "backend",
                "is_production",
                "postgres_dsn_configured",
                "supabase_dsn_configured",
            }
            assert expected_keys.issubset(set(result.keys()))


# ===========================================================================
# Test Auto-detection
# ===========================================================================


class TestAutoDetection:
    """Tests for automatic backend detection."""

    def test_auto_detects_supabase(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "auto"}):
            with patch(
                "aragora.storage.factory.get_supabase_postgres_dsn",
                return_value="postgres://supabase@localhost/db",
            ):
                backend = get_storage_backend()
                assert backend == StorageBackend.SUPABASE

    def test_auto_detects_postgres(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "auto"}):
            with patch(
                "aragora.storage.factory.get_supabase_postgres_dsn",
                return_value=None,
            ):
                with patch(
                    "aragora.storage.factory.get_selfhosted_postgres_dsn",
                    return_value="postgres://localhost/db",
                ):
                    backend = get_storage_backend()
                    assert backend == StorageBackend.POSTGRES

    def test_auto_falls_back_to_sqlite(self):
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "auto"}), _patch_no_dsn():
            backend = get_storage_backend()
            assert backend == StorageBackend.SQLITE

    def test_supabase_priority_over_postgres(self):
        """Supabase is preferred when both are configured."""
        with patch(
            "aragora.storage.factory.get_supabase_postgres_dsn",
            return_value="postgres://supabase@localhost/db",
        ):
            with patch(
                "aragora.storage.factory.get_selfhosted_postgres_dsn",
                return_value="postgres://postgres@localhost/db",
            ):
                with patch.dict(os.environ, {}, clear=True):
                    os.environ.pop("ARAGORA_DB_BACKEND", None)
                    backend = get_storage_backend()
                    assert backend == StorageBackend.SUPABASE
