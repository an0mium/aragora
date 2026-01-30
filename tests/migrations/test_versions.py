"""
Tests for aragora.migrations.versions package and migration discovery.

Covers:
- Migration module discovery via pkgutil
- Migration loading and registration
- Individual migration version files
- Migration ordering
- _load_migrations helper function

Run with:
    python -m pytest tests/migrations/test_versions.py -v --noconftest --timeout=30
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestVersionsPackageImport:
    """Verify the versions package can be imported."""

    def test_import_versions_package(self):
        import aragora.migrations.versions

        assert aragora.migrations.versions is not None

    def test_versions_package_has_init(self):
        from aragora.migrations import versions

        # Should have __file__ pointing to __init__.py
        assert versions.__file__ is not None
        assert "__init__" in versions.__file__


# ---------------------------------------------------------------------------
# Individual Migration Version Tests
# ---------------------------------------------------------------------------


class TestInitialSchemaMigration:
    """Tests for the initial schema migration."""

    def test_import_initial_schema(self):
        from aragora.migrations.versions import v20240101000000_initial_schema as mod

        assert hasattr(mod, "migration")

    def test_initial_schema_has_required_fields(self):
        from aragora.migrations.versions.v20240101000000_initial_schema import (
            migration,
        )

        assert migration.version == 20240101000000
        assert migration.name == "Initial schema"
        assert migration.up_sql is not None
        assert migration.down_sql is not None

    def test_initial_schema_creates_gauntlet_table(self):
        from aragora.migrations.versions.v20240101000000_initial_schema import (
            migration,
        )

        # Verify the SQL contains expected table creation
        assert "gauntlet_results" in migration.up_sql
        assert "CREATE TABLE" in migration.up_sql

    def test_initial_schema_down_drops_table(self):
        from aragora.migrations.versions.v20240101000000_initial_schema import (
            migration,
        )

        # Verify rollback drops the table
        assert "DROP TABLE" in migration.down_sql
        assert "gauntlet_results" in migration.down_sql


class TestKnowledgeMoundVisibilityMigration:
    """Tests for the Knowledge Mound visibility migration."""

    def test_import_km_visibility(self):
        from aragora.migrations.versions import (
            v20260119000000_knowledge_mound_visibility as mod,
        )

        assert hasattr(mod, "migration")
        assert hasattr(mod, "up_fn")
        assert hasattr(mod, "down_fn")

    def test_km_visibility_has_required_fields(self):
        from aragora.migrations.versions.v20260119000000_knowledge_mound_visibility import (
            migration,
        )

        assert migration.version == 20260119000000
        assert migration.name == "Knowledge Mound visibility and access grants"
        assert migration.up_fn is not None
        assert migration.down_fn is not None

    def test_km_visibility_up_fn_callable(self):
        from aragora.migrations.versions.v20260119000000_knowledge_mound_visibility import (
            up_fn,
        )

        assert callable(up_fn)

    def test_km_visibility_down_fn_callable(self):
        from aragora.migrations.versions.v20260119000000_knowledge_mound_visibility import (
            down_fn,
        )

        assert callable(down_fn)

    def test_km_visibility_helper_functions(self):
        from aragora.migrations.versions.v20260119000000_knowledge_mound_visibility import (
            _column_exists,
            _table_exists,
        )

        assert callable(_column_exists)
        assert callable(_table_exists)


class TestChannelGovernanceStoresMigration:
    """Tests for the channel governance stores migration."""

    def test_import_channel_governance(self):
        try:
            from aragora.migrations.versions import (
                v20260120000000_channel_governance_stores as mod,
            )

            assert hasattr(mod, "migration")
        except ImportError:
            pytest.skip("Channel governance migration not available")

    def test_channel_governance_version(self):
        try:
            from aragora.migrations.versions.v20260120000000_channel_governance_stores import (
                migration,
            )

            assert migration.version == 20260120000000
        except ImportError:
            pytest.skip("Channel governance migration not available")


class TestMarketplaceWebhooksBatchMigration:
    """Tests for the marketplace webhooks batch migration."""

    def test_import_marketplace_webhooks(self):
        try:
            from aragora.migrations.versions import (
                v20260120100000_marketplace_webhooks_batch as mod,
            )

            assert hasattr(mod, "migration")
        except ImportError:
            pytest.skip("Marketplace webhooks migration not available")

    def test_marketplace_webhooks_version(self):
        try:
            from aragora.migrations.versions.v20260120100000_marketplace_webhooks_batch import (
                migration,
            )

            assert migration.version == 20260120100000
        except ImportError:
            pytest.skip("Marketplace webhooks migration not available")


# ---------------------------------------------------------------------------
# Migration Discovery Tests
# ---------------------------------------------------------------------------


class TestMigrationDiscovery:
    """Tests for migration discovery functionality."""

    def test_discover_all_version_modules(self):
        """Verify all migration version modules can be discovered."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent

        discovered = []
        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                discovered.append(name)

        # Should discover at least the known migrations
        assert len(discovered) >= 1  # At least initial_schema
        assert any("20240101000000" in name for name in discovered)

    def test_all_discovered_modules_have_migration(self):
        """Verify all discovered modules export a migration object."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                module = importlib.import_module(f"aragora.migrations.versions.{name}")
                assert hasattr(module, "migration"), f"Module {name} missing 'migration'"


class TestLoadMigrations:
    """Tests for the _load_migrations helper function."""

    def test_load_migrations_registers_all(self):
        """Test that _load_migrations registers all discovered migrations."""
        from aragora.migrations.runner import MigrationRunner, _load_migrations

        # Create a runner with in-memory backend
        from tests.migrations.test_runner import InMemorySQLiteBackend

        backend = InMemorySQLiteBackend()
        runner = MigrationRunner(backend=backend)

        # Clear any existing migrations
        runner._migrations = []

        # Load migrations
        _load_migrations(runner)

        # Should have loaded at least one migration
        assert len(runner._migrations) >= 1

        backend.close()

    def test_load_migrations_sorted_by_version(self):
        """Test that loaded migrations are sorted by version."""
        from aragora.migrations.runner import MigrationRunner, _load_migrations

        from tests.migrations.test_runner import InMemorySQLiteBackend

        backend = InMemorySQLiteBackend()
        runner = MigrationRunner(backend=backend)
        runner._migrations = []

        _load_migrations(runner)

        versions = [m.version for m in runner._migrations]
        assert versions == sorted(versions), "Migrations should be sorted by version"

        backend.close()

    def test_load_migrations_handles_missing_package(self):
        """Test that _load_migrations handles missing versions package gracefully."""
        from aragora.migrations.runner import MigrationRunner, _load_migrations
        import sys

        from tests.migrations.test_runner import InMemorySQLiteBackend

        backend = InMemorySQLiteBackend()
        runner = MigrationRunner(backend=backend)

        # Temporarily remove the versions module from sys.modules
        saved_module = sys.modules.get("aragora.migrations.versions")
        try:
            # Remove the module to simulate it not being available
            sys.modules["aragora.migrations.versions"] = None

            # Mock the import to raise ImportError
            with patch(
                "aragora.migrations.runner.importlib.import_module",
                side_effect=ImportError("No module"),
            ):
                # Should not raise, just log debug message
                _load_migrations(runner)
        finally:
            # Restore the module
            if saved_module is not None:
                sys.modules["aragora.migrations.versions"] = saved_module
            else:
                sys.modules.pop("aragora.migrations.versions", None)

        backend.close()


# ---------------------------------------------------------------------------
# Migration Ordering Tests
# ---------------------------------------------------------------------------


class TestMigrationOrdering:
    """Tests for migration version ordering."""

    def test_versions_are_timestamps(self):
        """Verify migration versions follow timestamp format."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                # Extract version number from filename
                version_str = name.split("_")[0][1:]  # Remove 'v' prefix
                assert len(version_str) == 14, f"Version {version_str} should be 14 digits"
                assert version_str.isdigit(), f"Version {version_str} should be numeric"

    def test_versions_are_unique(self):
        """Verify all migration versions are unique."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent
        seen_versions = set()

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                version_str = name.split("_")[0][1:]
                assert version_str not in seen_versions, f"Duplicate version: {version_str}"
                seen_versions.add(version_str)

    def test_versions_chronological(self):
        """Verify versions are in chronological order by name."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent
        version_numbers = []

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                version_str = name.split("_")[0][1:]
                version_numbers.append(int(version_str))

        # When sorted alphabetically, they should also be numerically sorted
        assert version_numbers == sorted(version_numbers)


# ---------------------------------------------------------------------------
# Integration Tests with Runner
# ---------------------------------------------------------------------------


class TestMigrationIntegration:
    """Integration tests for migrations with the runner."""

    def test_get_migration_runner_loads_migrations(self):
        """Test that get_migration_runner loads all migrations."""
        from aragora.migrations.runner import reset_runner

        # Reset to ensure clean state
        reset_runner()

        from aragora.migrations.runner import get_migration_runner

        import tempfile
        import os

        # Create temp directory for test database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            runner = get_migration_runner(db_path=db_path)

            # Should have loaded migrations
            assert len(runner._migrations) >= 1

            # Clean up
            runner.close()
            reset_runner()

    def test_migrations_can_be_applied(self):
        """Test that loaded migrations can be applied."""
        from aragora.migrations.runner import MigrationRunner, _load_migrations

        from tests.migrations.test_runner import InMemorySQLiteBackend

        backend = InMemorySQLiteBackend()
        runner = MigrationRunner(backend=backend)
        _load_migrations(runner)

        # Apply just the first migration
        if runner._migrations:
            first_migration = runner._migrations[0]
            applied = runner.upgrade(target_version=first_migration.version)
            assert len(applied) >= 1

        backend.close()


# ---------------------------------------------------------------------------
# Migration Content Validation Tests
# ---------------------------------------------------------------------------


class TestMigrationContentValidation:
    """Tests for migration content validation."""

    def test_all_migrations_have_version(self):
        """All migrations must have a version number."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                module = importlib.import_module(f"aragora.migrations.versions.{name}")
                migration = module.migration
                assert migration.version is not None
                assert isinstance(migration.version, int)

    def test_all_migrations_have_name(self):
        """All migrations must have a descriptive name."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                module = importlib.import_module(f"aragora.migrations.versions.{name}")
                migration = module.migration
                assert migration.name is not None
                assert len(migration.name) > 0

    def test_all_migrations_have_up(self):
        """All migrations must have an up operation (SQL or function)."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                module = importlib.import_module(f"aragora.migrations.versions.{name}")
                migration = module.migration
                assert migration.up_sql is not None or migration.up_fn is not None, (
                    f"Migration {name} must have up_sql or up_fn"
                )

    def test_version_matches_filename(self):
        """Migration version should match the version in filename."""
        from aragora.migrations import versions

        versions_path = Path(versions.__file__).parent

        for _, name, _ in pkgutil.iter_modules([str(versions_path)]):
            if name.startswith("v"):
                # Extract version from filename
                filename_version = int(name.split("_")[0][1:])

                # Get version from migration object
                module = importlib.import_module(f"aragora.migrations.versions.{name}")
                migration = module.migration

                assert migration.version == filename_version, (
                    f"Migration {name}: filename version {filename_version} "
                    f"!= migration version {migration.version}"
                )
