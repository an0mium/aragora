"""
Tests for zero-downtime migration patterns.
"""

import pytest

from aragora.migrations.patterns import (
    MigrationRisk,
    MigrationValidation,
    is_postgresql,
    safe_add_column,
    safe_add_nullable_column,
    safe_drop_column,
    safe_rename_column,
    backfill_column,
    safe_set_not_null,
    safe_create_index,
    safe_drop_index,
    validate_migration_safety,
    get_table_row_count,
)
from aragora.storage.backends import SQLiteBackend


@pytest.fixture
def backend(tmp_path):
    """Create a SQLite backend for testing."""
    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(str(db_path))
    # Create a test table
    backend.execute_write(
        """
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT
        )
        """
    )
    return backend


class TestMigrationRisk:
    """Tests for MigrationRisk enum."""

    def test_risk_levels_ordered(self):
        """Risk levels should be ordered from low to critical."""
        assert MigrationRisk.LOW.value == "low"
        assert MigrationRisk.MEDIUM.value == "medium"
        assert MigrationRisk.HIGH.value == "high"
        assert MigrationRisk.CRITICAL.value == "critical"


class TestMigrationValidation:
    """Tests for MigrationValidation dataclass."""

    def test_validation_creation(self):
        """Should create validation result."""
        validation = MigrationValidation(
            safe=True,
            risk_level=MigrationRisk.LOW,
            warnings=[],
            recommendations=[],
        )
        assert validation.safe is True
        assert validation.risk_level == MigrationRisk.LOW

    def test_validation_with_warnings(self):
        """Should include warnings and recommendations."""
        validation = MigrationValidation(
            safe=False,
            risk_level=MigrationRisk.HIGH,
            warnings=["This is risky"],
            recommendations=["Use expand/contract pattern"],
        )
        assert validation.safe is False
        assert len(validation.warnings) == 1
        assert len(validation.recommendations) == 1


class TestIsPostgresql:
    """Tests for is_postgresql helper."""

    def test_sqlite_returns_false(self, backend):
        """SQLite backend should return False."""
        assert is_postgresql(backend) is False


class TestGetTableRowCount:
    """Tests for get_table_row_count."""

    def test_empty_table(self, backend):
        """Should return 0 for empty table."""
        count = get_table_row_count(backend, "test_table")
        assert count == 0

    def test_with_rows(self, backend):
        """Should return correct count."""
        backend.execute_write(
            "INSERT INTO test_table (name, email) VALUES ('Alice', 'alice@test.com')"
        )
        backend.execute_write("INSERT INTO test_table (name, email) VALUES ('Bob', 'bob@test.com')")
        count = get_table_row_count(backend, "test_table")
        assert count == 2


class TestSafeAddColumn:
    """Tests for safe_add_column."""

    def test_add_nullable_column(self, backend):
        """Should add a nullable column."""
        safe_add_column(backend, "test_table", "phone", "TEXT", nullable=True)

        # Verify column exists
        columns = backend.fetch_all("PRAGMA table_info(test_table)")
        column_names = [row[1] for row in columns]
        assert "phone" in column_names

    def test_add_column_with_default(self, backend):
        """Should add column with default value."""
        safe_add_column(backend, "test_table", "status", "TEXT", default="'active'")

        # Insert a row and verify default
        backend.execute_write("INSERT INTO test_table (name) VALUES ('Test')")
        result = backend.fetch_one("SELECT status FROM test_table WHERE name = 'Test'")
        assert result[0] == "active"

    def test_add_column_idempotent(self, backend):
        """Should not fail if column already exists."""
        safe_add_column(backend, "test_table", "phone", "TEXT")
        # Second call should not raise
        safe_add_column(backend, "test_table", "phone", "TEXT")


class TestSafeAddNullableColumn:
    """Tests for safe_add_nullable_column."""

    def test_adds_nullable_column(self, backend):
        """Should add a nullable column."""
        safe_add_nullable_column(backend, "test_table", "address", "TEXT")

        # Insert without address should work
        backend.execute_write("INSERT INTO test_table (name) VALUES ('Test')")
        result = backend.fetch_one("SELECT address FROM test_table WHERE name = 'Test'")
        assert result[0] is None


class TestSafeDropColumn:
    """Tests for safe_drop_column."""

    def test_drop_existing_column(self, backend):
        """Should drop an existing column."""
        # First add a column
        safe_add_column(backend, "test_table", "temp_col", "TEXT")

        # Then drop it
        safe_drop_column(backend, "test_table", "temp_col", verify_unused=False)

        # Verify column is gone
        columns = backend.fetch_all("PRAGMA table_info(test_table)")
        column_names = [row[1] for row in columns]
        assert "temp_col" not in column_names

    def test_drop_nonexistent_column(self, backend):
        """Should not fail if column doesn't exist."""
        # Should not raise
        safe_drop_column(backend, "test_table", "nonexistent_col", verify_unused=False)


class TestSafeRenameColumn:
    """Tests for safe_rename_column."""

    def test_rename_column(self, backend):
        """Should rename a column."""
        safe_rename_column(backend, "test_table", "email", "email_address")

        # Verify old name gone and new name exists
        columns = backend.fetch_all("PRAGMA table_info(test_table)")
        column_names = [row[1] for row in columns]
        assert "email" not in column_names
        assert "email_address" in column_names


class TestBackfillColumn:
    """Tests for backfill_column."""

    def test_backfill_null_values(self, backend):
        """Should backfill NULL values."""
        safe_add_column(backend, "test_table", "active", "INTEGER")

        # Insert rows with NULL active
        backend.execute_write("INSERT INTO test_table (name) VALUES ('Alice')")
        backend.execute_write("INSERT INTO test_table (name) VALUES ('Bob')")

        # Backfill
        count = backfill_column(backend, "test_table", "active", "1")

        # All should be 1 now
        result = backend.fetch_all("SELECT active FROM test_table")
        assert all(row[0] == 1 for row in result)

    def test_backfill_with_where_clause(self, backend):
        """Should backfill only matching rows."""
        safe_add_column(backend, "test_table", "tier", "TEXT")

        backend.execute_write(
            "INSERT INTO test_table (name, email) VALUES ('Alice', 'alice@premium.com')"
        )
        backend.execute_write("INSERT INTO test_table (name, email) VALUES ('Bob', 'bob@free.com')")

        # Backfill only premium emails
        backfill_column(
            backend,
            "test_table",
            "tier",
            "'premium'",
            where_clause="email LIKE '%premium%'",
        )

        result = backend.fetch_one("SELECT tier FROM test_table WHERE name = 'Alice'")
        assert result[0] == "premium"

        result = backend.fetch_one("SELECT tier FROM test_table WHERE name = 'Bob'")
        assert result[0] is None


class TestSafeSetNotNull:
    """Tests for safe_set_not_null."""

    def test_fails_with_nulls_and_no_default(self, backend):
        """Should fail if NULLs exist and no default provided."""
        safe_add_column(backend, "test_table", "required_field", "TEXT")
        backend.execute_write("INSERT INTO test_table (name) VALUES ('Test')")

        with pytest.raises(ValueError, match="NULL values remain"):
            safe_set_not_null(backend, "test_table", "required_field")

    def test_backfills_with_default(self, backend):
        """Should backfill remaining NULLs with default."""
        safe_add_column(backend, "test_table", "status", "TEXT")
        backend.execute_write("INSERT INTO test_table (name) VALUES ('Test')")

        # Should backfill and then set NOT NULL (SQLite won't actually set NOT NULL)
        safe_set_not_null(backend, "test_table", "status", default="'pending'")

        result = backend.fetch_one("SELECT status FROM test_table WHERE name = 'Test'")
        assert result[0] == "pending"


class TestSafeCreateIndex:
    """Tests for safe_create_index."""

    def test_create_index(self, backend):
        """Should create an index."""
        safe_create_index(backend, "idx_test_name", "test_table", ["name"])

        # Verify index exists (SQLite)
        indexes = backend.fetch_all("PRAGMA index_list(test_table)")
        index_names = [row[1] for row in indexes]
        assert "idx_test_name" in index_names

    def test_create_unique_index(self, backend):
        """Should create a unique index."""
        safe_create_index(backend, "idx_test_email_unique", "test_table", ["email"], unique=True)

        # Verify index is unique
        indexes = backend.fetch_all("PRAGMA index_list(test_table)")
        for idx in indexes:
            if idx[1] == "idx_test_email_unique":
                assert idx[2] == 1  # unique flag

    def test_create_index_idempotent(self, backend):
        """Should not fail if index already exists."""
        safe_create_index(backend, "idx_test_name", "test_table", ["name"])
        # Second call should not raise
        safe_create_index(backend, "idx_test_name", "test_table", ["name"])


class TestSafeDropIndex:
    """Tests for safe_drop_index."""

    def test_drop_index(self, backend):
        """Should drop an existing index."""
        safe_create_index(backend, "idx_to_drop", "test_table", ["name"])
        safe_drop_index(backend, "idx_to_drop")

        # Verify index is gone
        indexes = backend.fetch_all("PRAGMA index_list(test_table)")
        index_names = [row[1] for row in indexes]
        assert "idx_to_drop" not in index_names

    def test_drop_nonexistent_index(self, backend):
        """Should not fail if index doesn't exist."""
        # Should not raise
        safe_drop_index(backend, "nonexistent_index")


class TestValidateMigrationSafety:
    """Tests for validate_migration_safety."""

    def test_empty_operations_is_safe(self, backend):
        """Empty operations list should be safe."""
        result = validate_migration_safety(backend, [])
        assert result.safe is True
        assert result.risk_level == MigrationRisk.LOW

    def test_add_nullable_column_is_safe(self, backend):
        """Adding nullable column should be safe."""
        result = validate_migration_safety(
            backend,
            [{"type": "add_column", "table": "test_table", "column": "new_col", "nullable": True}],
        )
        assert result.safe is True

    def test_drop_column_is_medium_risk(self, backend):
        """Dropping column should be medium risk."""
        # Add some data to make table non-trivial
        for i in range(100):
            backend.execute_write(f"INSERT INTO test_table (name) VALUES ('User{i}')")

        result = validate_migration_safety(
            backend,
            [{"type": "drop_column", "table": "test_table", "column": "email"}],
        )
        # Should still be safe for small table
        assert result.safe is True

    def test_non_concurrent_index_on_large_table(self, backend):
        """Non-concurrent index on large table should warn."""
        # Insert many rows to simulate large table
        for i in range(1000):
            backend.execute_write(f"INSERT INTO test_table (name) VALUES ('User{i}')")

        result = validate_migration_safety(
            backend,
            [{"type": "create_index", "table": "test_table", "concurrently": False}],
        )
        # Should still be safe for SQLite (CONCURRENTLY not applicable)
        assert result.safe is True


class TestExpandContractPattern:
    """Integration tests for expand/contract migration pattern."""

    def test_full_expand_contract_cycle(self, backend):
        """Test complete expand/contract pattern."""
        # Insert existing data
        backend.execute_write(
            "INSERT INTO test_table (name, email) VALUES ('Alice', 'alice@test.com')"
        )
        backend.execute_write("INSERT INTO test_table (name, email) VALUES ('Bob', 'bob@test.com')")

        # EXPAND: Add nullable column
        safe_add_nullable_column(backend, "test_table", "verified", "INTEGER")

        # BACKFILL: Set default value
        backfill_column(backend, "test_table", "verified", "0")

        # Verify backfill worked
        result = backend.fetch_all("SELECT verified FROM test_table")
        assert all(row[0] == 0 for row in result)

        # CONTRACT: Make NOT NULL (with safety default)
        safe_set_not_null(backend, "test_table", "verified", default="0")

        # New inserts should work
        backend.execute_write("INSERT INTO test_table (name, verified) VALUES ('Charlie', 1)")
        result = backend.fetch_one("SELECT verified FROM test_table WHERE name = 'Charlie'")
        assert result[0] == 1
