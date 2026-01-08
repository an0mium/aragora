"""
Tests for persistence repositories.

Tests the BaseRepository class and SQL injection protection utilities.
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from aragora.persistence.repositories.base import (
    BaseRepository,
    EntityNotFoundError,
    RepositoryError,
    _validate_sql_identifier,
    _validate_where_clause,
)


# =============================================================================
# SQL Injection Protection Tests
# =============================================================================


class TestSQLIdentifierValidation:
    """Tests for _validate_sql_identifier function."""

    def test_valid_simple_identifier(self):
        """Should accept simple alphanumeric identifiers."""
        assert _validate_sql_identifier("users") == "users"
        assert _validate_sql_identifier("debate_log") == "debate_log"
        assert _validate_sql_identifier("Table1") == "Table1"

    def test_valid_identifier_starting_with_underscore(self):
        """Should accept identifiers starting with underscore."""
        assert _validate_sql_identifier("_private") == "_private"
        assert _validate_sql_identifier("_table_name") == "_table_name"

    def test_rejects_empty_identifier(self):
        """Should reject empty identifiers."""
        with pytest.raises(ValueError, match="Empty"):
            _validate_sql_identifier("")

    def test_rejects_identifier_with_spaces(self):
        """Should reject identifiers containing spaces."""
        with pytest.raises(ValueError, match="Invalid"):
            _validate_sql_identifier("table name")

    def test_rejects_identifier_with_special_chars(self):
        """Should reject identifiers with special characters."""
        with pytest.raises(ValueError, match="Invalid"):
            _validate_sql_identifier("table; DROP TABLE users")

        with pytest.raises(ValueError, match="Invalid"):
            _validate_sql_identifier("table--comment")

    def test_rejects_identifier_starting_with_number(self):
        """Should reject identifiers starting with number."""
        with pytest.raises(ValueError, match="Invalid"):
            _validate_sql_identifier("1table")

    def test_rejects_sql_keywords(self):
        """Should reject SQL keywords as identifiers."""
        with pytest.raises(ValueError, match="SQL keyword"):
            _validate_sql_identifier("DROP")

        with pytest.raises(ValueError, match="SQL keyword"):
            _validate_sql_identifier("SELECT")

        with pytest.raises(ValueError, match="SQL keyword"):
            _validate_sql_identifier("UNION")

    def test_rejects_overly_long_identifier(self):
        """Should reject identifiers exceeding max length."""
        long_name = "a" * 200
        with pytest.raises(ValueError, match="too long"):
            _validate_sql_identifier(long_name)

    def test_custom_context_in_error_message(self):
        """Should include custom context in error messages."""
        with pytest.raises(ValueError, match="column name"):
            _validate_sql_identifier("bad column!", "column name")


class TestWhereClauseValidation:
    """Tests for _validate_where_clause function."""

    def test_valid_simple_where(self):
        """Should accept simple WHERE clauses."""
        assert _validate_where_clause("id = ?") == "id = ?"
        assert _validate_where_clause("name = ? AND status = ?") == "name = ? AND status = ?"

    def test_valid_comparison_operators(self):
        """Should accept comparison operators."""
        assert _validate_where_clause("count > ?") == "count > ?"
        assert _validate_where_clause("created_at >= ?") == "created_at >= ?"

    def test_empty_where_returns_empty(self):
        """Should return empty string for empty input."""
        assert _validate_where_clause("") == ""
        assert _validate_where_clause(None) is None

    def test_rejects_drop_injection(self):
        """Should reject DROP TABLE injection."""
        with pytest.raises(ValueError, match="forbidden pattern"):
            _validate_where_clause("1=1; DROP TABLE users")

    def test_rejects_delete_injection(self):
        """Should reject DELETE injection."""
        with pytest.raises(ValueError, match="forbidden pattern"):
            _validate_where_clause("1=1; DELETE FROM users")

    def test_rejects_union_injection(self):
        """Should reject UNION injection."""
        with pytest.raises(ValueError, match="forbidden pattern"):
            _validate_where_clause("id = 1 UNION SELECT * FROM passwords")

    def test_rejects_comment_injection(self):
        """Should reject SQL comment injection."""
        with pytest.raises(ValueError, match="forbidden pattern"):
            _validate_where_clause("id = 1 -- comment")

        with pytest.raises(ValueError, match="forbidden pattern"):
            _validate_where_clause("id = 1 /* block comment */")


# =============================================================================
# Test Repository Implementation
# =============================================================================


class SampleEntity:
    """Simple test entity."""

    def __init__(self, id: str, name: str, value: int = 0):
        self.id = id
        self.name = name
        self.value = value


class SampleRepository(BaseRepository[SampleEntity]):
    """Concrete repository implementation for testing."""

    def __init__(self, db_path: str):
        super().__init__(db_path)

    @property
    def _table_name(self) -> str:
        return "test_entities"

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def _to_entity(self, row: sqlite3.Row) -> SampleEntity:
        return SampleEntity(
            id=row["id"],
            name=row["name"],
            value=row["value"],
        )

    def _from_entity(self, entity: SampleEntity) -> Dict[str, Any]:
        return {
            "id": entity.id,
            "name": entity.name,
            "value": entity.value,
        }


# =============================================================================
# BaseRepository Tests
# =============================================================================


class TestBaseRepositoryCRUD:
    """Tests for BaseRepository CRUD operations."""

    @pytest.fixture
    def repo(self, tmp_path):
        """Create a test repository with temp database."""
        db_path = tmp_path / "test.db"
        return SampleRepository(str(db_path))

    def test_save_creates_new_entity(self, repo):
        """Should create new entity with save."""
        entity = SampleEntity(id="test-1", name="Test One", value=100)

        entity_id = repo.save(entity)

        assert entity_id == "test-1"
        retrieved = repo.get("test-1")
        assert retrieved is not None
        assert retrieved.name == "Test One"
        assert retrieved.value == 100

    def test_save_updates_existing_entity(self, repo):
        """Should update existing entity with save."""
        entity = SampleEntity(id="test-1", name="Original", value=50)
        repo.save(entity)

        entity.name = "Updated"
        entity.value = 75
        repo.save(entity)

        retrieved = repo.get("test-1")
        assert retrieved.name == "Updated"
        assert retrieved.value == 75

    def test_get_returns_none_for_missing(self, repo):
        """Should return None for non-existent entity."""
        result = repo.get("non-existent")
        assert result is None

    def test_get_or_raise_throws_for_missing(self, repo):
        """Should raise EntityNotFoundError for missing entity."""
        with pytest.raises(EntityNotFoundError) as exc_info:
            repo.get_or_raise("missing-id")

        assert "missing-id" in str(exc_info.value)
        assert exc_info.value.entity_id == "missing-id"

    def test_delete_removes_entity(self, repo):
        """Should remove entity on delete."""
        entity = SampleEntity(id="test-1", name="Delete Me")
        repo.save(entity)

        result = repo.delete("test-1")

        assert result is True
        assert repo.get("test-1") is None

    def test_delete_returns_false_for_missing(self, repo):
        """Should return False when deleting non-existent entity."""
        result = repo.delete("non-existent")
        assert result is False

    def test_list_all_returns_entities(self, repo):
        """Should return all entities."""
        for i in range(5):
            repo.save(SampleEntity(id=f"test-{i}", name=f"Entity {i}"))

        entities = repo.list_all()

        assert len(entities) == 5

    def test_list_all_with_limit(self, repo):
        """Should respect limit parameter."""
        for i in range(10):
            repo.save(SampleEntity(id=f"test-{i}", name=f"Entity {i}"))

        entities = repo.list_all(limit=5)

        assert len(entities) == 5

    def test_list_all_with_offset(self, repo):
        """Should respect offset parameter."""
        for i in range(10):
            repo.save(SampleEntity(id=f"test-{i}", name=f"Entity {i}"))

        entities = repo.list_all(limit=5, offset=5)

        assert len(entities) == 5

    def test_count_returns_total(self, repo):
        """Should return total entity count."""
        assert repo.count() == 0

        for i in range(3):
            repo.save(SampleEntity(id=f"test-{i}", name=f"Entity {i}"))

        assert repo.count() == 3


class TestBaseRepositoryTransactions:
    """Tests for BaseRepository transaction handling."""

    @pytest.fixture
    def repo(self, tmp_path):
        """Create a test repository with temp database."""
        db_path = tmp_path / "test.db"
        return SampleRepository(str(db_path))

    def test_transaction_commits_on_success(self, repo):
        """Should commit transaction on success."""
        with repo._transaction() as conn:
            conn.execute(
                "INSERT INTO test_entities (id, name, value) VALUES (?, ?, ?)",
                ("tx-1", "Transaction Test", 42),
            )

        # Should be persisted
        entity = repo.get("tx-1")
        assert entity is not None
        assert entity.name == "Transaction Test"

    def test_transaction_rollbacks_on_error(self, repo):
        """Should rollback transaction on exception."""
        try:
            with repo._transaction() as conn:
                conn.execute(
                    "INSERT INTO test_entities (id, name, value) VALUES (?, ?, ?)",
                    ("tx-rollback", "Will Rollback", 99),
                )
                # Force an error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Should NOT be persisted
        entity = repo.get("tx-rollback")
        assert entity is None


class TestBaseRepositoryConnectionModes:
    """Tests for BaseRepository connection modes."""

    @pytest.fixture
    def repo(self, tmp_path):
        """Create a test repository with temp database."""
        db_path = tmp_path / "test.db"
        return SampleRepository(str(db_path))

    def test_readonly_connection_allows_reads(self, repo):
        """Should allow reads in readonly mode."""
        repo.save(SampleEntity(id="read-test", name="Read Only Test"))

        with repo._connection(readonly=True) as conn:
            cursor = conn.execute(
                "SELECT * FROM test_entities WHERE id = ?", ("read-test",)
            )
            row = cursor.fetchone()

        assert row["name"] == "Read Only Test"

    def test_readonly_connection_rejects_writes(self, repo):
        """Should reject writes in readonly mode."""
        with pytest.raises(sqlite3.OperationalError):
            with repo._connection(readonly=True) as conn:
                conn.execute(
                    "INSERT INTO test_entities (id, name) VALUES (?, ?)",
                    ("write-test", "Should Fail"),
                )


class TestBaseRepositoryHelpers:
    """Tests for BaseRepository helper methods."""

    @pytest.fixture
    def repo(self, tmp_path):
        """Create a test repository with temp database."""
        db_path = tmp_path / "test.db"
        return SampleRepository(str(db_path))

    def test_count_with_where_clause(self, repo):
        """Should count with WHERE clause."""
        repo.save(SampleEntity(id="a", name="Alpha", value=10))
        repo.save(SampleEntity(id="b", name="Beta", value=20))
        repo.save(SampleEntity(id="c", name="Gamma", value=10))

        count = repo._count("test_entities", "value = ?", (10,))

        assert count == 2

    def test_exists_returns_true_when_found(self, repo):
        """Should return True when entity exists."""
        repo.save(SampleEntity(id="exists-test", name="Exists"))

        result = repo._exists("test_entities", "id = ?", ("exists-test",))

        assert result is True

    def test_exists_returns_false_when_missing(self, repo):
        """Should return False when entity doesn't exist."""
        result = repo._exists("test_entities", "id = ?", ("missing",))

        assert result is False

    def test_execute_many_returns_rowcount(self, repo):
        """Should return rowcount from executemany."""
        params = [
            (f"batch-{i}", f"Batch {i}", i * 10)
            for i in range(5)
        ]

        rowcount = repo._execute_many(
            "INSERT INTO test_entities (id, name, value) VALUES (?, ?, ?)",
            params,
        )

        assert rowcount == 5
        assert repo.count() == 5


class TestEntityNotFoundError:
    """Tests for EntityNotFoundError exception."""

    def test_includes_entity_type_and_id(self):
        """Should include entity type and ID in message."""
        error = EntityNotFoundError("Debate", "debate-123")

        assert "Debate" in str(error)
        assert "debate-123" in str(error)
        assert error.entity_type == "Debate"
        assert error.entity_id == "debate-123"

    def test_is_repository_error(self):
        """Should be a subclass of RepositoryError."""
        error = EntityNotFoundError("Entity", "id")

        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)
