"""
Tests for SQL Database Connector.

Tests cover:
- Query validation (security patterns)
- Database type detection
- Evidence conversion from rows
- SQLite in-memory database queries
- Read-only mode enforcement
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from aragora.connectors.sql import SQLConnector, SQLQueryResult
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


class TestSQLQueryResult:
    """Tests for SQLQueryResult dataclass."""

    def test_query_result_creation(self):
        """SQLQueryResult should store query results."""
        result = SQLQueryResult(
            rows=[{"id": 1, "name": "test"}],
            column_names=["id", "name"],
            row_count=1,
            query_time_ms=5.2,
            database_type="sqlite",
        )
        assert result.row_count == 1
        assert result.database_type == "sqlite"
        assert result.query_time_ms == 5.2
        assert result.rows[0]["name"] == "test"


class TestSQLConnectorInit:
    """Tests for SQLConnector initialization."""

    def test_init_with_connection_string(self):
        """SQLConnector should accept connection string."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
        )
        assert connector._database_type == "sqlite"
        assert connector._read_only is True

    def test_init_detects_postgresql(self):
        """Should detect PostgreSQL from connection string."""
        connector = SQLConnector(connection_string="postgresql://user:pass@localhost:5432/db")
        assert connector._database_type == "postgresql"

    def test_init_detects_mysql(self):
        """Should detect MySQL from connection string."""
        connector = SQLConnector(connection_string="mysql://user:pass@localhost:3306/db")
        assert connector._database_type == "mysql"

    def test_init_detects_sqlite(self):
        """Should detect SQLite from connection string."""
        connector = SQLConnector(connection_string="sqlite:///path/to/db.sqlite")
        assert connector._database_type == "sqlite"

    def test_source_type_is_database(self):
        """source_type should be DATABASE."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        assert connector.source_type == SourceType.DATABASE

    def test_name_includes_database_type(self):
        """name should include database type."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        assert "sqlite" in connector.name.lower()


class TestQueryValidation:
    """Tests for SQL query validation."""

    def test_allows_select_query(self):
        """Should allow SELECT queries in read-only mode."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=True,
        )
        # Should not raise
        connector._validate_query("SELECT * FROM users WHERE id = ?")

    def test_allows_with_query(self):
        """Should allow WITH (CTE) queries in read-only mode."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=True,
        )
        # Should not raise
        connector._validate_query("WITH cte AS (SELECT * FROM users) SELECT * FROM cte")

    def test_blocks_drop_query(self):
        """Should block DROP queries in read-only mode."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=True,
        )
        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("DROP TABLE users")

    def test_blocks_delete_query(self):
        """Should block DELETE queries in read-only mode."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=True,
        )
        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("DELETE FROM users WHERE id = 1")

    def test_blocks_insert_query(self):
        """Should block INSERT queries in read-only mode."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=True,
        )
        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("INSERT INTO users VALUES (1, 'test')")

    def test_blocks_update_query(self):
        """Should block UPDATE queries in read-only mode."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=True,
        )
        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("UPDATE users SET name = 'test' WHERE id = 1")

    def test_blocks_sql_comments(self):
        """Should block SQL comments (potential injection)."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=True,
        )
        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("SELECT * FROM users -- WHERE admin = 1")

    def test_blocks_multiple_statements(self):
        """Should block multiple statements (potential injection)."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=True,
        )
        with pytest.raises(ValueError, match="blocked pattern"):
            connector._validate_query("SELECT 1; DROP TABLE users")

    def test_allows_modification_when_not_read_only(self):
        """Should allow modifications when read_only=False."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=False,
        )
        # Should not raise
        connector._validate_query("INSERT INTO users VALUES (1, 'test')")


class TestRowToEvidence:
    """Tests for converting database rows to Evidence."""

    def test_converts_row_with_content_column(self):
        """Should convert row with content column to Evidence."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        row = {
            "id": 123,
            "title": "Test Article",
            "content": "This is the article content.",
            "author": "John Doe",
            "created_at": "2026-01-01T00:00:00Z",
        }
        evidence = connector._row_to_evidence(row)

        assert evidence.id == "sql:123"
        assert evidence.source_type == SourceType.DATABASE
        assert evidence.content == "This is the article content."
        assert evidence.title == "Test Article"
        assert evidence.author == "John Doe"

    def test_concatenates_columns_when_no_content(self):
        """Should concatenate columns when content column missing."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        row = {
            "id": 456,
            "name": "Test Item",
            "description": "A description",
        }
        evidence = connector._row_to_evidence(row)

        assert "name: Test Item" in evidence.content
        assert "description: A description" in evidence.content

    def test_truncates_long_content(self):
        """Should truncate content exceeding MAX_CONTENT_LENGTH."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        long_content = "x" * 15000
        row = {
            "id": 789,
            "content": long_content,
        }
        evidence = connector._row_to_evidence(row)

        assert len(evidence.content) <= SQLConnector.MAX_CONTENT_LENGTH + 3
        assert evidence.content.endswith("...")

    def test_generates_id_when_missing(self):
        """Should generate ID from row hash when id column missing."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        row = {
            "content": "Some content",
            "other": "data",
        }
        evidence = connector._row_to_evidence(row)

        assert evidence.id.startswith("sql:")
        assert len(evidence.id) > 5

    def test_custom_column_mappings(self):
        """Should support custom column name mappings."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        row = {
            "article_id": 999,
            "article_title": "Custom Title",
            "body": "Custom content",
            "written_by": "Author Name",
        }
        evidence = connector._row_to_evidence(
            row,
            content_column="body",
            title_column="article_title",
            id_column="article_id",
            author_column="written_by",
        )

        assert evidence.id == "sql:999"
        assert evidence.title == "Custom Title"
        assert evidence.content == "Custom content"
        assert evidence.author == "Author Name"


class TestSQLiteQueries:
    """Tests for SQLite in-memory database queries."""

    @pytest.mark.asyncio
    async def test_execute_query_on_sqlite(self):
        """Should execute query on SQLite database."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=False,  # Need to create table
        )

        # Create table and insert data
        conn = await connector._get_connection()
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice')")
        conn.execute("INSERT INTO test VALUES (2, 'Bob')")
        conn.commit()

        # Now query in read-only
        connector._read_only = True
        result = await connector.execute_query("SELECT * FROM test WHERE id = ?", params=(1,))

        assert result.row_count == 1
        assert result.rows[0]["name"] == "Alice"
        assert result.database_type == "sqlite"
        assert result.query_time_ms > 0

        await connector.close()

    @pytest.mark.asyncio
    async def test_search_returns_evidence_list(self):
        """Should return list of Evidence from search."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=False,
        )

        # Setup
        conn = await connector._get_connection()
        conn.execute(
            """
            CREATE TABLE articles (
                id INTEGER,
                title TEXT,
                content TEXT,
                author TEXT
            )
        """
        )
        conn.execute("INSERT INTO articles VALUES (1, 'Article 1', 'Content one', 'Alice')")
        conn.execute("INSERT INTO articles VALUES (2, 'Article 2', 'Content two', 'Bob')")
        conn.commit()

        connector._read_only = True
        results = await connector.search(
            query="SELECT * FROM articles WHERE id = ?",
            params=(1,),
            limit=10,
        )

        assert len(results) == 1
        assert isinstance(results[0], Evidence)
        assert results[0].title == "Article 1"
        assert results[0].content == "Content one"

        await connector.close()

    @pytest.mark.asyncio
    async def test_adds_limit_clause_if_missing(self):
        """Should add LIMIT clause if not present in query."""
        connector = SQLConnector(
            connection_string="sqlite://:memory:",
            read_only=False,
        )

        conn = await connector._get_connection()
        conn.execute("CREATE TABLE items (id INTEGER)")
        for i in range(20):
            conn.execute(f"INSERT INTO items VALUES ({i})")
        conn.commit()

        connector._read_only = True
        results = await connector.search(
            query="SELECT * FROM items",
            limit=5,
        )

        assert len(results) == 5

        await connector.close()


class TestAvailabilityCheck:
    """Tests for connector availability checks."""

    def test_not_available_without_connection_string(self):
        """Should not be available without connection string."""
        connector = SQLConnector(connection_string=None)
        assert connector.is_available is False

    def test_sqlite_always_available(self):
        """SQLite should always be available (built-in)."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        assert connector.is_available is True


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_closes_connection(self):
        """Context manager should close connection on exit."""
        async with SQLConnector(connection_string="sqlite://:memory:") as connector:
            # Connection created on first use
            await connector._get_connection()
            assert connector._connection is not None

        # Connection should be closed after exit
        assert connector._connection is None


class TestFetch:
    """Tests for fetch method."""

    @pytest.mark.asyncio
    async def test_fetch_returns_none_without_table(self):
        """fetch() should return None (requires table context)."""
        connector = SQLConnector(connection_string="sqlite://:memory:")
        result = await connector.fetch("sql:123")
        assert result is None
