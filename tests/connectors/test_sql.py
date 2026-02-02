"""
Tests for SQL connector.

Tests cover:
- SQLConnector initialization and configuration
- Database type detection
- Query validation and security
- Evidence conversion
- Search and fetch operations
- Connection lifecycle
- SQLQueryResult dataclass
"""

import hashlib
import os
import pytest
import re
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.sql import SQLConnector, SQLQueryResult
from aragora.reasoning.provenance import SourceType


class TestSQLQueryResult:
    """Tests for SQLQueryResult dataclass."""

    def test_basic_initialization(self):
        """Should initialize with all fields."""
        result = SQLQueryResult(
            rows=[{"id": 1, "name": "test"}],
            column_names=["id", "name"],
            row_count=1,
            query_time_ms=5.5,
            database_type="postgresql",
        )

        assert result.rows == [{"id": 1, "name": "test"}]
        assert result.column_names == ["id", "name"]
        assert result.row_count == 1
        assert result.query_time_ms == 5.5
        assert result.database_type == "postgresql"

    def test_empty_result(self):
        """Should handle empty results."""
        result = SQLQueryResult(
            rows=[],
            column_names=["id"],
            row_count=0,
            query_time_ms=1.0,
            database_type="sqlite",
        )

        assert result.row_count == 0
        assert len(result.rows) == 0


class TestSQLConnectorInitialization:
    """Tests for SQLConnector initialization."""

    def test_default_initialization(self):
        """Should initialize with defaults."""
        connector = SQLConnector()

        assert connector._connection_string is None
        assert connector._read_only is True
        assert connector._query_timeout == 30.0
        assert connector._connection is None

    def test_with_connection_string(self):
        """Should accept connection string."""
        connector = SQLConnector(connection_string="postgresql://user:pass@localhost:5432/db")

        assert connector._connection_string == "postgresql://user:pass@localhost:5432/db"

    def test_from_environment_variable(self):
        """Should read connection string from environment."""
        with patch.dict(os.environ, {"ARAGORA_SQL_CONNECTION": "sqlite:///test.db"}):
            connector = SQLConnector()
            assert connector._connection_string == "sqlite:///test.db"

    def test_custom_settings(self):
        """Should accept custom settings."""
        connector = SQLConnector(
            connection_string="mysql://localhost/db",
            default_confidence=0.8,
            read_only=False,
            query_timeout=60.0,
        )

        assert connector.default_confidence == 0.8
        assert connector._read_only is False
        assert connector._query_timeout == 60.0

    def test_explicit_database_type(self):
        """Should accept explicit database type."""
        connector = SQLConnector(
            connection_string="custom://host/db",
            database_type="postgresql",
        )

        assert connector._database_type == "postgresql"


class TestDatabaseTypeDetection:
    """Tests for database type detection."""

    def test_detect_postgresql(self):
        """Should detect PostgreSQL connection strings."""
        for prefix in ["postgresql://", "postgres://"]:
            connector = SQLConnector(connection_string=f"{prefix}user:pass@host/db")
            assert connector._database_type == "postgresql"

    def test_detect_mysql(self):
        """Should detect MySQL connection strings."""
        connector = SQLConnector(connection_string="mysql://user:pass@host/db")
        assert connector._database_type == "mysql"

    def test_detect_sqlite(self):
        """Should detect SQLite connection strings."""
        connector = SQLConnector(connection_string="sqlite:///path/to/db.sqlite")
        assert connector._database_type == "sqlite"

    def test_unknown_database_type(self):
        """Should return None for unknown types."""
        connector = SQLConnector(connection_string="oracle://host/db")
        assert connector._database_type is None

    def test_no_connection_string(self):
        """Should return None when no connection string."""
        connector = SQLConnector()
        assert connector._database_type is None


class TestSourceProperties:
    """Tests for source type and name properties."""

    def test_source_type(self):
        """Should return DATABASE source type."""
        connector = SQLConnector()
        assert connector.source_type == SourceType.DATABASE

    def test_name_with_database_type(self):
        """Should include database type in name."""
        connector = SQLConnector(connection_string="postgresql://host/db")
        assert connector.name == "SQL (postgresql)"

    def test_name_unknown_type(self):
        """Should show unknown when type not detected."""
        connector = SQLConnector()
        assert connector.name == "SQL (unknown)"


class TestIsAvailable:
    """Tests for is_available property."""

    def test_not_available_without_connection(self):
        """Should not be available without connection string."""
        connector = SQLConnector()
        assert connector.is_available is False

    def test_sqlite_always_available(self):
        """SQLite should always be available (built-in)."""
        connector = SQLConnector(connection_string="sqlite:///test.db")
        assert connector.is_available is True

    def test_postgresql_availability(self):
        """PostgreSQL availability depends on driver."""
        connector = SQLConnector(connection_string="postgresql://host/db")

        # Result depends on whether asyncpg or psycopg2 is installed
        # Just verify it returns a boolean without error
        assert isinstance(connector.is_available, bool)

    def test_mysql_availability(self):
        """MySQL availability depends on aiomysql."""
        connector = SQLConnector(connection_string="mysql://host/db")
        assert isinstance(connector.is_available, bool)


class TestQueryValidation:
    """Tests for query validation and security."""

    def test_allows_select_queries(self):
        """Should allow SELECT queries."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        # Should not raise
        connector._validate_query("SELECT * FROM users")
        connector._validate_query("SELECT id, name FROM users WHERE active = 1")

    def test_allows_with_queries(self):
        """Should allow WITH (CTE) queries."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        connector._validate_query(
            "WITH active_users AS (SELECT * FROM users WHERE active = 1) SELECT * FROM active_users"
        )

    def test_blocks_drop_statements(self):
        """Should block DROP statements."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("DROP TABLE users")

    def test_blocks_delete_statements(self):
        """Should block DELETE statements."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("DELETE FROM users WHERE id = 1")

    def test_blocks_truncate_statements(self):
        """Should block TRUNCATE statements."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("TRUNCATE TABLE users")

    def test_blocks_alter_statements(self):
        """Should block ALTER statements."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("ALTER TABLE users ADD COLUMN email VARCHAR")

    def test_blocks_create_statements(self):
        """Should block CREATE statements."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("CREATE TABLE test (id INT)")

    def test_blocks_insert_statements(self):
        """Should block INSERT statements."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("INSERT INTO users VALUES (1, 'test')")

    def test_blocks_update_statements(self):
        """Should block UPDATE statements."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("UPDATE users SET name = 'test'")

    def test_blocks_sql_comments(self):
        """Should block SQL comments (potential injection)."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("SELECT * FROM users -- WHERE admin = 1")

    def test_blocks_multiple_statements(self):
        """Should block multiple statements."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("SELECT * FROM users; DROP TABLE users")

    def test_case_insensitive_blocking(self):
        """Should block regardless of case."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError):
            connector._validate_query("drop TABLE users")

        with pytest.raises(ValueError):
            connector._validate_query("DELETE from users")

    def test_allows_dangerous_when_not_readonly(self):
        """Should allow dangerous queries when read_only=False."""
        connector = SQLConnector(
            connection_string="sqlite:///test.db",
            read_only=False,
        )

        # Should not raise when not in read-only mode
        connector._validate_query("DELETE FROM users WHERE id = 1")

    def test_non_select_blocked_in_readonly(self):
        """Should block non-SELECT queries in read-only mode."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        with pytest.raises(ValueError, match="Only SELECT"):
            connector._validate_query("SHOW TABLES")


class TestRowToEvidence:
    """Tests for converting database rows to Evidence."""

    @pytest.fixture
    def connector(self):
        """Create connector for testing."""
        return SQLConnector(connection_string="sqlite:///test.db")

    def test_basic_conversion(self, connector):
        """Should convert row to Evidence."""
        row = {
            "id": 123,
            "title": "Test Article",
            "content": "This is test content.",
            "created_at": "2024-01-15T10:30:00",
            "author": "Test Author",
        }

        evidence = connector._row_to_evidence(row)

        assert evidence.id == "sql:123"
        assert evidence.title == "Test Article"
        assert evidence.content == "This is test content."
        assert evidence.author == "Test Author"
        assert evidence.source_type == SourceType.DATABASE

    def test_custom_column_names(self, connector):
        """Should use custom column mappings."""
        row = {
            "doc_id": "abc",
            "headline": "News Title",
            "body": "Article body text",
            "published": "2024-01-01",
            "writer": "Reporter",
        }

        evidence = connector._row_to_evidence(
            row,
            id_column="doc_id",
            title_column="headline",
            content_column="body",
            created_column="published",
            author_column="writer",
        )

        assert evidence.id == "sql:abc"
        assert evidence.title == "News Title"
        assert evidence.content == "Article body text"
        assert evidence.author == "Reporter"

    def test_missing_content_column(self, connector):
        """Should concatenate columns when content column missing."""
        row = {
            "id": 1,
            "field1": "Value 1",
            "field2": "Value 2",
        }

        evidence = connector._row_to_evidence(row, content_column="missing")

        assert "field1: Value 1" in evidence.content
        assert "field2: Value 2" in evidence.content

    def test_truncates_long_content(self, connector):
        """Should truncate very long content."""
        row = {
            "id": 1,
            "content": "x" * 20000,
        }

        evidence = connector._row_to_evidence(row)

        assert len(evidence.content) <= connector.MAX_CONTENT_LENGTH + 3  # +3 for "..."
        assert evidence.content.endswith("...")

    def test_generates_id_when_missing(self, connector):
        """Should generate hash ID when id column missing."""
        row = {
            "content": "Test content",
        }

        evidence = connector._row_to_evidence(row, id_column="missing_id")

        assert evidence.id.startswith("sql:")
        assert len(evidence.id) > 4  # "sql:" + hash

    def test_datetime_conversion(self, connector):
        """Should convert datetime objects."""
        row = {
            "id": 1,
            "content": "Test",
            "created_at": datetime(2024, 1, 15, 10, 30, 0),
        }

        evidence = connector._row_to_evidence(row)

        assert "2024-01-15" in evidence.created_at

    def test_freshness_calculation(self, connector):
        """Should calculate freshness score."""
        recent = {
            "id": 1,
            "content": "Recent",
            "created_at": datetime.now().isoformat(),
        }

        old = {
            "id": 2,
            "content": "Old",
            "created_at": (datetime.now() - timedelta(days=365)).isoformat(),
        }

        recent_evidence = connector._row_to_evidence(recent)
        old_evidence = connector._row_to_evidence(old)

        assert recent_evidence.freshness > old_evidence.freshness

    def test_default_authority(self, connector):
        """Should set default authority score."""
        row = {"id": 1, "content": "Test"}
        evidence = connector._row_to_evidence(row)

        assert evidence.authority == 0.6

    def test_metadata_includes_database_type(self, connector):
        """Should include database type in metadata."""
        row = {"id": 1, "content": "Test"}
        evidence = connector._row_to_evidence(row)

        assert "database_type" in evidence.metadata
        assert evidence.metadata["database_type"] == "sqlite"

    def test_metadata_includes_row_data(self, connector):
        """Should include row data in metadata (truncated)."""
        row = {"id": 1, "content": "Test", "extra": "x" * 500}
        evidence = connector._row_to_evidence(row)

        assert "row_data" in evidence.metadata
        # Values should be truncated to 200 chars
        assert len(evidence.metadata["row_data"]["extra"]) <= 200


class TestExecuteQuery:
    """Tests for execute_query method."""

    @pytest.fixture
    def sqlite_connector(self, tmp_path):
        """Create SQLite connector with temp database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@test.com')")
        conn.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@test.com')")
        conn.commit()
        conn.close()

        return SQLConnector(connection_string=f"sqlite:///{db_path}")

    @pytest.mark.asyncio
    async def test_basic_query(self, sqlite_connector):
        """Should execute basic SELECT query."""
        result = await sqlite_connector.execute_query("SELECT * FROM users")

        assert result.row_count == 2
        assert len(result.rows) == 2
        assert result.column_names == ["id", "name", "email"]
        assert result.database_type == "sqlite"

    @pytest.mark.asyncio
    async def test_parameterized_query(self, sqlite_connector):
        """Should execute parameterized query."""
        result = await sqlite_connector.execute_query(
            "SELECT * FROM users WHERE id = ?",
            params=[1],
        )

        assert result.row_count == 1
        assert result.rows[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_query_time_measured(self, sqlite_connector):
        """Should measure query execution time."""
        result = await sqlite_connector.execute_query("SELECT * FROM users")

        assert result.query_time_ms >= 0

    @pytest.mark.asyncio
    async def test_validates_query(self, sqlite_connector):
        """Should validate query before execution."""
        with pytest.raises(ValueError):
            await sqlite_connector.execute_query("DROP TABLE users")

    @pytest.mark.asyncio
    async def test_no_connection_string_error(self):
        """Should error when no connection string."""
        connector = SQLConnector()

        with pytest.raises(ValueError, match="No connection string"):
            await connector.execute_query("SELECT 1")


class TestSearch:
    """Tests for search method."""

    @pytest.fixture
    def sqlite_connector(self, tmp_path):
        """Create SQLite connector with test data."""
        db_path = tmp_path / "articles.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE articles "
            "(id INTEGER, title TEXT, content TEXT, created_at TEXT, author TEXT)"
        )
        conn.execute(
            "INSERT INTO articles VALUES "
            "(1, 'AI Safety', 'Content about AI safety...', '2024-01-15', 'Researcher')"
        )
        conn.execute(
            "INSERT INTO articles VALUES "
            "(2, 'ML Models', 'Content about ML models...', '2024-01-10', 'Engineer')"
        )
        conn.commit()
        conn.close()

        return SQLConnector(connection_string=f"sqlite:///{db_path}")

    @pytest.mark.asyncio
    async def test_basic_search(self, sqlite_connector):
        """Should return Evidence objects from search."""
        evidence = await sqlite_connector.search(
            "SELECT * FROM articles WHERE title LIKE ?",
            params=["%AI%"],
        )

        assert len(evidence) == 1
        assert evidence[0].title == "AI Safety"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, sqlite_connector):
        """Should respect limit parameter."""
        evidence = await sqlite_connector.search(
            "SELECT * FROM articles",
            limit=1,
        )

        assert len(evidence) == 1

    @pytest.mark.asyncio
    async def test_search_adds_limit_clause(self, sqlite_connector):
        """Should add LIMIT clause if not present."""
        evidence = await sqlite_connector.search(
            "SELECT * FROM articles",
            limit=5,
        )

        # Both rows should be returned (less than limit)
        assert len(evidence) == 2

    @pytest.mark.asyncio
    async def test_search_custom_columns(self, sqlite_connector):
        """Should use custom column mappings."""
        evidence = await sqlite_connector.search(
            "SELECT id, title, content, created_at, author FROM articles",
            content_column="content",
            title_column="title",
            id_column="id",
            created_column="created_at",
            author_column="author",
        )

        assert len(evidence) > 0
        assert evidence[0].author is not None


class TestFetch:
    """Tests for fetch method."""

    @pytest.mark.asyncio
    async def test_fetch_not_supported(self):
        """fetch() should log warning and return None."""
        connector = SQLConnector(connection_string="sqlite:///test.db")

        result = await connector.fetch("sql:123")

        assert result is None


class TestConnectionLifecycle:
    """Tests for connection lifecycle management."""

    @pytest.fixture
    def sqlite_connector(self, tmp_path):
        """Create SQLite connector."""
        db_path = tmp_path / "test.db"
        sqlite3.connect(str(db_path)).close()
        return SQLConnector(connection_string=f"sqlite:///{db_path}")

    @pytest.mark.asyncio
    async def test_context_manager(self, sqlite_connector):
        """Should work as async context manager."""
        async with sqlite_connector as conn:
            assert conn is sqlite_connector

        assert sqlite_connector._connection is None

    @pytest.mark.asyncio
    async def test_close_clears_connection(self, sqlite_connector):
        """Should clear connection on close."""
        # Create connection
        await sqlite_connector._get_connection()
        assert sqlite_connector._connection is not None

        # Close
        await sqlite_connector.close()
        assert sqlite_connector._connection is None

    @pytest.mark.asyncio
    async def test_close_when_no_connection(self, sqlite_connector):
        """Should handle close when no connection exists."""
        await sqlite_connector.close()  # Should not raise


class TestGetConnection:
    """Tests for _get_connection method."""

    @pytest.mark.asyncio
    async def test_creates_sqlite_connection(self, tmp_path):
        """Should create SQLite connection."""
        db_path = tmp_path / "test.db"
        sqlite3.connect(str(db_path)).close()

        connector = SQLConnector(connection_string=f"sqlite:///{db_path}")
        conn = await connector._get_connection()

        assert conn is not None
        await connector.close()

    @pytest.mark.asyncio
    async def test_reuses_existing_connection(self, tmp_path):
        """Should reuse existing connection."""
        db_path = tmp_path / "test.db"
        sqlite3.connect(str(db_path)).close()

        connector = SQLConnector(connection_string=f"sqlite:///{db_path}")

        conn1 = await connector._get_connection()
        conn2 = await connector._get_connection()

        assert conn1 is conn2
        await connector.close()

    @pytest.mark.asyncio
    async def test_memory_sqlite(self):
        """Should handle in-memory SQLite."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        conn = await connector._get_connection()

        assert conn is not None
        await connector.close()

    @pytest.mark.asyncio
    async def test_unsupported_database_type(self):
        """Should raise for unsupported database type."""
        connector = SQLConnector(
            connection_string="oracle://host/db",
            database_type="oracle",
        )

        with pytest.raises(ValueError, match="Unsupported database type"):
            await connector._get_connection()


class TestBlockedPatterns:
    """Tests for BLOCKED_PATTERNS constant."""

    def test_blocked_patterns_defined(self):
        """Should have blocked patterns defined."""
        assert len(SQLConnector.BLOCKED_PATTERNS) > 0

    def test_patterns_are_valid_regex(self):
        """All patterns should be valid regex."""
        for pattern in SQLConnector.BLOCKED_PATTERNS:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern '{pattern}': {e}")


class TestMaxContentLength:
    """Tests for MAX_CONTENT_LENGTH constant."""

    def test_max_content_length_defined(self):
        """Should have max content length defined."""
        assert SQLConnector.MAX_CONTENT_LENGTH > 0

    def test_max_content_length_reasonable(self):
        """Max content length should be reasonable (1K-100K)."""
        assert 1000 <= SQLConnector.MAX_CONTENT_LENGTH <= 100000
