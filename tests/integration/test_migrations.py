"""
Migration Configuration Tests.

Tests that verify the database migration infrastructure is properly set up.
These tests don't require PostgreSQL to run - they just verify file existence.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestMigrationConfiguration:
    """Tests for migration configuration files."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    def test_alembic_config_exists(self, project_root: Path):
        """Test that Alembic configuration is properly set up."""
        alembic_ini = project_root / "alembic.ini"
        migrations_dir = project_root / "migrations"
        env_py = migrations_dir / "env.py"
        versions_dir = migrations_dir / "versions"

        assert alembic_ini.exists(), "alembic.ini should exist"
        assert migrations_dir.exists(), "migrations directory should exist"
        assert env_py.exists(), "migrations/env.py should exist"
        assert versions_dir.exists(), "migrations/versions directory should exist"

    def test_alembic_ini_has_correct_config(self, project_root: Path):
        """Test that alembic.ini has correct configuration."""
        alembic_ini = project_root / "alembic.ini"
        content = alembic_ini.read_text()

        assert "script_location = migrations" in content
        assert "sqlalchemy.url" in content

    def test_initial_schema_exists(self, project_root: Path):
        """Test that initial schema SQL file exists."""
        schema_file = project_root / "migrations" / "sql" / "001_initial_schema.sql"

        assert schema_file.exists(), "Initial schema SQL file should exist"
        content = schema_file.read_text()
        # Verify it contains expected tables
        assert "CREATE TABLE" in content
        assert "webhook_configs" in content or "users" in content

    def test_initial_schema_has_required_tables(self, project_root: Path):
        """Test that initial schema contains all required tables."""
        schema_file = project_root / "migrations" / "sql" / "001_initial_schema.sql"
        content = schema_file.read_text().lower()

        required_tables = [
            "webhook_configs",
            "integrations",
            "gmail_tokens",
            "finding_workflows",
            "gauntlet_runs",
            "job_queue",
            "governance_artifacts",
            "marketplace_items",
            "approval_requests",
            "token_blacklist",
            "users",
        ]

        for table in required_tables:
            assert table in content, f"Schema should contain {table} table"

    def test_init_postgres_script_exists(self, project_root: Path):
        """Test that database initialization script exists and has expected options."""
        init_script = project_root / "scripts" / "init_postgres_db.py"

        assert init_script.exists(), "init_postgres_db.py should exist"
        content = init_script.read_text()
        # Verify it supports Alembic
        assert "--alembic" in content, "Script should support --alembic flag"
        assert "--verify" in content, "Script should support --verify flag"
        assert "--dsn" in content, "Script should support --dsn flag"

    def test_migration_env_uses_asyncpg(self, project_root: Path):
        """Test that migration env.py uses asyncpg for async support."""
        env_py = project_root / "migrations" / "env.py"
        content = env_py.read_text()

        assert "asyncpg" in content or "async" in content, (
            "Migration env should support async operations"
        )

    def test_migration_version_exists(self, project_root: Path):
        """Test that at least one migration version exists."""
        versions_dir = project_root / "migrations" / "versions"
        versions = list(versions_dir.glob("*.py"))

        assert len(versions) >= 1, "At least one migration version should exist"
        # Check first version is the initial schema
        version_names = [v.name for v in versions]
        assert any("001" in name or "initial" in name.lower() for name in version_names), (
            "Initial migration should exist"
        )
