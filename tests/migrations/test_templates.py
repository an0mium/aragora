"""
Tests for aragora.migrations.templates module.

Covers:
- Template generation functions
- Column and index definitions
- Migration file content generation
- Template validation

Run with:
    python -m pytest tests/migrations/test_templates.py -v --noconftest --timeout=30
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Import Smoke Tests
# ---------------------------------------------------------------------------


class TestTemplateImports:
    """Verify the templates module and its public API can be imported."""

    def test_import_templates_module(self):
        import aragora.migrations.templates as mod

        assert hasattr(mod, "generate_add_column_migration")
        assert hasattr(mod, "generate_add_index_migration")
        assert hasattr(mod, "generate_add_table_migration")
        assert hasattr(mod, "generate_data_migration_template")
        assert hasattr(mod, "generate_constraint_migration")
        assert hasattr(mod, "create_migration_file")

    def test_import_from_package_init(self):
        from aragora.migrations import (
            ColumnType,
            ColumnDefinition,
            IndexDefinition,
            TableDefinition,
            generate_add_column_migration,
            generate_add_index_migration,
            generate_add_table_migration,
            create_migration_file,
        )

        assert ColumnType is not None
        assert callable(generate_add_column_migration)
        assert callable(create_migration_file)


# ---------------------------------------------------------------------------
# ColumnType Tests
# ---------------------------------------------------------------------------


class TestColumnType:
    """Tests for ColumnType enumeration."""

    def test_column_types_exist(self):
        from aragora.migrations.templates import ColumnType

        assert ColumnType.TEXT is not None
        assert ColumnType.INTEGER is not None
        assert ColumnType.BOOLEAN is not None
        assert ColumnType.TIMESTAMP is not None
        assert ColumnType.JSONB is not None

    def test_column_type_postgres_sqlite_mapping(self):
        from aragora.migrations.templates import ColumnType

        # Text types
        assert ColumnType.TEXT.postgres == "TEXT"
        assert ColumnType.TEXT.sqlite == "TEXT"

        # Boolean (SQLite doesn't have native boolean)
        assert ColumnType.BOOLEAN.postgres == "BOOLEAN"
        assert ColumnType.BOOLEAN.sqlite == "INTEGER"

        # JSONB (SQLite uses TEXT)
        assert ColumnType.JSONB.postgres == "JSONB"
        assert ColumnType.JSONB.sqlite == "TEXT"


# ---------------------------------------------------------------------------
# ColumnDefinition Tests
# ---------------------------------------------------------------------------


class TestColumnDefinition:
    """Tests for ColumnDefinition dataclass."""

    def test_simple_column(self):
        from aragora.migrations.templates import ColumnDefinition

        col = ColumnDefinition(name="email", data_type="TEXT")
        sql = col.to_sql()

        assert '"email" TEXT' in sql

    def test_column_with_default(self):
        from aragora.migrations.templates import ColumnDefinition

        col = ColumnDefinition(name="status", data_type="TEXT", default="'pending'")
        sql = col.to_sql()

        assert '"status" TEXT' in sql
        assert "DEFAULT 'pending'" in sql

    def test_not_null_column(self):
        from aragora.migrations.templates import ColumnDefinition

        col = ColumnDefinition(name="required", data_type="TEXT", nullable=False)
        sql = col.to_sql()

        assert "NOT NULL" in sql

    def test_primary_key_column(self):
        from aragora.migrations.templates import ColumnDefinition

        col = ColumnDefinition(name="id", data_type="TEXT", primary_key=True)
        sql = col.to_sql()

        assert "PRIMARY KEY" in sql
        assert "NOT NULL" not in sql  # PK implies NOT NULL

    def test_unique_column(self):
        from aragora.migrations.templates import ColumnDefinition

        col = ColumnDefinition(name="email", data_type="TEXT", unique=True)
        sql = col.to_sql()

        assert "UNIQUE" in sql

    def test_column_with_check(self):
        from aragora.migrations.templates import ColumnDefinition

        col = ColumnDefinition(
            name="rating", data_type="INTEGER", check="rating >= 1 AND rating <= 5"
        )
        sql = col.to_sql()

        assert "CHECK (rating >= 1 AND rating <= 5)" in sql

    def test_column_with_references(self):
        from aragora.migrations.templates import ColumnDefinition

        col = ColumnDefinition(name="user_id", data_type="TEXT", references="users(id)")
        sql = col.to_sql()

        assert "REFERENCES users(id)" in sql

    def test_column_with_column_type_enum(self):
        from aragora.migrations.templates import ColumnDefinition, ColumnType

        col = ColumnDefinition(name="is_active", data_type=ColumnType.BOOLEAN)

        pg_sql = col.to_sql(is_postgres=True)
        sqlite_sql = col.to_sql(is_postgres=False)

        assert "BOOLEAN" in pg_sql
        assert "INTEGER" in sqlite_sql


# ---------------------------------------------------------------------------
# IndexDefinition Tests
# ---------------------------------------------------------------------------


class TestIndexDefinition:
    """Tests for IndexDefinition dataclass."""

    def test_simple_index(self):
        from aragora.migrations.templates import IndexDefinition

        idx = IndexDefinition(name="idx_users_email", table="users", columns=["email"])

        create_sql = idx.to_create_sql()
        assert "CREATE INDEX" in create_sql
        assert '"idx_users_email"' in create_sql
        assert '"users"' in create_sql
        assert '"email"' in create_sql

    def test_unique_index(self):
        from aragora.migrations.templates import IndexDefinition

        idx = IndexDefinition(
            name="idx_users_email_unique",
            table="users",
            columns=["email"],
            unique=True,
        )

        create_sql = idx.to_create_sql()
        assert "UNIQUE INDEX" in create_sql

    def test_composite_index(self):
        from aragora.migrations.templates import IndexDefinition

        idx = IndexDefinition(
            name="idx_users_status_created",
            table="users",
            columns=["status", "created_at"],
        )

        create_sql = idx.to_create_sql()
        assert '"status"' in create_sql
        assert '"created_at"' in create_sql

    def test_partial_index(self):
        from aragora.migrations.templates import IndexDefinition

        idx = IndexDefinition(
            name="idx_users_active",
            table="users",
            columns=["email"],
            where="is_active = TRUE",
        )

        create_sql = idx.to_create_sql()
        assert "WHERE is_active = TRUE" in create_sql

    def test_concurrently_postgres(self):
        from aragora.migrations.templates import IndexDefinition

        idx = IndexDefinition(name="idx_test", table="test", columns=["col"], concurrently=True)

        pg_sql = idx.to_create_sql(is_postgres=True)
        sqlite_sql = idx.to_create_sql(is_postgres=False)

        assert "CONCURRENTLY" in pg_sql
        assert "CONCURRENTLY" not in sqlite_sql

    def test_drop_index(self):
        from aragora.migrations.templates import IndexDefinition

        idx = IndexDefinition(name="idx_test", table="test", columns=["col"])

        drop_sql = idx.to_drop_sql()
        assert "DROP INDEX" in drop_sql
        assert '"idx_test"' in drop_sql


# ---------------------------------------------------------------------------
# Template Generation Tests
# ---------------------------------------------------------------------------


class TestGenerateAddColumnMigration:
    """Tests for generate_add_column_migration function."""

    def test_basic_add_column(self):
        from aragora.migrations.templates import generate_add_column_migration

        content = generate_add_column_migration(
            version=20260201120000,
            name="Add email to users",
            table="users",
            column="email",
            data_type="TEXT",
        )

        assert "version=20260201120000" in content
        assert "Add email to users" in content
        assert '"users"' in content
        assert '"email"' in content
        assert "TEXT" in content
        assert "up_fn" in content
        assert "down_fn" in content

    def test_add_column_with_default(self):
        from aragora.migrations.templates import generate_add_column_migration

        content = generate_add_column_migration(
            version=20260201120000,
            name="Add status to orders",
            table="orders",
            column="status",
            data_type="TEXT",
            default="'pending'",
        )

        assert "'pending'" in content

    def test_add_column_with_index(self):
        from aragora.migrations.templates import generate_add_column_migration

        content = generate_add_column_migration(
            version=20260201120000,
            name="Add email to users",
            table="users",
            column="email",
            data_type="TEXT",
            create_index=True,
        )

        assert "safe_create_index" in content
        assert "idx_users_email" in content

    def test_add_column_custom_index_name(self):
        from aragora.migrations.templates import generate_add_column_migration

        content = generate_add_column_migration(
            version=20260201120000,
            name="Add email to users",
            table="users",
            column="email",
            data_type="TEXT",
            create_index=True,
            index_name="my_custom_index",
        )

        assert "my_custom_index" in content


class TestGenerateAddIndexMigration:
    """Tests for generate_add_index_migration function."""

    def test_basic_add_index(self):
        from aragora.migrations.templates import generate_add_index_migration

        content = generate_add_index_migration(
            version=20260201120000,
            name="Add index on users email",
            table="users",
            columns=["email"],
        )

        assert "version=20260201120000" in content
        assert "idx_users_email" in content
        assert "safe_create_index" in content
        assert "safe_drop_index" in content

    def test_unique_index(self):
        from aragora.migrations.templates import generate_add_index_migration

        content = generate_add_index_migration(
            version=20260201120000,
            name="Add unique index",
            table="users",
            columns=["email"],
            unique=True,
        )

        assert "unique=True" in content

    def test_composite_index(self):
        from aragora.migrations.templates import generate_add_index_migration

        content = generate_add_index_migration(
            version=20260201120000,
            name="Add composite index",
            table="orders",
            columns=["user_id", "created_at"],
        )

        assert "user_id" in content
        assert "created_at" in content
        assert "idx_orders_user_id_created_at" in content


class TestGenerateAddTableMigration:
    """Tests for generate_add_table_migration function."""

    def test_basic_add_table(self):
        from aragora.migrations.templates import generate_add_table_migration

        content = generate_add_table_migration(
            version=20260201120000,
            name="Create users table",
            table="users",
            columns=[
                {"name": "id", "type": "TEXT", "primary_key": True},
                {"name": "email", "type": "TEXT", "nullable": False},
            ],
        )

        assert "version=20260201120000" in content
        assert '"users"' in content
        assert '"id" TEXT PRIMARY KEY' in content
        assert '"email" TEXT NOT NULL' in content
        assert "DROP TABLE" in content

    def test_add_table_with_indexes(self):
        from aragora.migrations.templates import generate_add_table_migration

        content = generate_add_table_migration(
            version=20260201120000,
            name="Create orders table",
            table="orders",
            columns=[
                {"name": "id", "type": "TEXT", "primary_key": True},
                {"name": "user_id", "type": "TEXT"},
            ],
            indexes=[
                {"columns": ["user_id"]},
            ],
        )

        assert "idx_orders_user_id" in content

    def test_add_table_with_audit_columns(self):
        from aragora.migrations.templates import generate_add_table_migration

        content = generate_add_table_migration(
            version=20260201120000,
            name="Create items table",
            table="items",
            columns=[{"name": "id", "type": "TEXT", "primary_key": True}],
            include_audit_columns=True,
        )

        assert "created_at" in content
        assert "updated_at" in content


class TestGenerateDataMigrationTemplate:
    """Tests for generate_data_migration_template function."""

    def test_basic_data_migration(self):
        from aragora.migrations.templates import generate_data_migration_template

        content = generate_data_migration_template(
            version=20260201120000,
            name="Backfill user emails",
            description="Populate email column from legacy data",
        )

        assert "version=20260201120000" in content
        assert "Backfill user emails" in content
        assert "Populate email column" in content
        assert "BATCH_SIZE" in content
        assert "backfill_column" in content


class TestGenerateConstraintMigration:
    """Tests for generate_constraint_migration function."""

    def test_check_constraint(self):
        from aragora.migrations.templates import generate_constraint_migration

        content = generate_constraint_migration(
            version=20260201120000,
            name="Add rating check constraint",
            table="reviews",
            constraint_type="CHECK",
            constraint_name="chk_rating_range",
            definition="rating >= 1 AND rating <= 5",
        )

        assert "version=20260201120000" in content
        assert "chk_rating_range" in content
        assert "CHECK" in content
        assert "rating >= 1 AND rating <= 5" in content

    def test_unique_constraint(self):
        from aragora.migrations.templates import generate_constraint_migration

        content = generate_constraint_migration(
            version=20260201120000,
            name="Add unique email constraint",
            table="users",
            constraint_type="UNIQUE",
            constraint_name="uq_users_email",
            definition="email",
        )

        assert "UNIQUE" in content
        assert "uq_users_email" in content


# ---------------------------------------------------------------------------
# create_migration_file Tests
# ---------------------------------------------------------------------------


class TestCreateMigrationFile:
    """Tests for create_migration_file function."""

    def test_basic_template(self):
        from aragora.migrations.templates import create_migration_file

        filename, content = create_migration_file(name="Test migration")

        assert filename.startswith("v")
        assert filename.endswith(".py")
        assert "test_migration" in filename
        assert "Migration" in content
        assert "up_sql" in content
        assert "down_sql" in content

    def test_add_column_template(self):
        from aragora.migrations.templates import create_migration_file

        filename, content = create_migration_file(
            name="Add email to users",
            template_type="add_column",
            table="users",
            column="email",
            data_type="TEXT",
        )

        assert "add_email_to_users" in filename
        assert "safe_add_nullable_column" in content

    def test_add_index_template(self):
        from aragora.migrations.templates import create_migration_file

        filename, content = create_migration_file(
            name="Add index on email",
            template_type="add_index",
            table="users",
            columns=["email"],
        )

        assert "safe_create_index" in content
        assert "safe_drop_index" in content

    def test_add_table_template(self):
        from aragora.migrations.templates import create_migration_file

        filename, content = create_migration_file(
            name="Create products table",
            template_type="add_table",
            table="products",
            columns=[
                {"name": "id", "type": "TEXT", "primary_key": True},
                {"name": "name", "type": "TEXT", "nullable": False},
            ],
        )

        assert "CREATE TABLE" in content
        assert "DROP TABLE" in content

    def test_data_migration_template(self):
        from aragora.migrations.templates import create_migration_file

        filename, content = create_migration_file(
            name="Backfill data",
            template_type="data_migration",
            description="Backfill column values",
        )

        assert "BATCH_SIZE" in content
        assert "backfill_column" in content

    def test_constraint_template(self):
        from aragora.migrations.templates import create_migration_file

        filename, content = create_migration_file(
            name="Add check constraint",
            template_type="constraint",
            table="reviews",
            constraint_type="CHECK",
            constraint_name="chk_rating",
            definition="rating > 0",
        )

        assert "CHECK" in content
        assert "chk_rating" in content

    def test_filename_sanitization(self):
        from aragora.migrations.templates import create_migration_file

        filename, _ = create_migration_file(name="Add email/verified to users!!")

        # Should remove special characters
        assert "/" not in filename
        assert "!" not in filename
        assert "add_emailverified_to_users" in filename


# ---------------------------------------------------------------------------
# Content Validation Tests
# ---------------------------------------------------------------------------


class TestMigrationContentValidity:
    """Test that generated migration content is valid Python."""

    def test_add_column_is_valid_python(self):
        from aragora.migrations.templates import generate_add_column_migration

        content = generate_add_column_migration(
            version=20260201120000,
            name="Test",
            table="t",
            column="c",
            data_type="TEXT",
        )

        # Should compile without syntax errors
        compile(content, "<string>", "exec")

    def test_add_index_is_valid_python(self):
        from aragora.migrations.templates import generate_add_index_migration

        content = generate_add_index_migration(
            version=20260201120000,
            name="Test",
            table="t",
            columns=["c"],
        )

        compile(content, "<string>", "exec")

    def test_add_table_is_valid_python(self):
        from aragora.migrations.templates import generate_add_table_migration

        content = generate_add_table_migration(
            version=20260201120000,
            name="Test",
            table="t",
            columns=[{"name": "id", "type": "TEXT", "primary_key": True}],
        )

        compile(content, "<string>", "exec")

    def test_data_migration_is_valid_python(self):
        from aragora.migrations.templates import generate_data_migration_template

        content = generate_data_migration_template(
            version=20260201120000,
            name="Test",
            description="Test",
        )

        compile(content, "<string>", "exec")

    def test_constraint_is_valid_python(self):
        from aragora.migrations.templates import generate_constraint_migration

        content = generate_constraint_migration(
            version=20260201120000,
            name="Test",
            table="t",
            constraint_type="CHECK",
            constraint_name="c",
            definition="x > 0",
        )

        compile(content, "<string>", "exec")
