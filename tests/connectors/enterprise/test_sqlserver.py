"""
Tests for SQL Server Enterprise Connector.

Tests the SQL Server integration including:
- Incremental sync using transaction timestamps or CDC columns
- CDC (Change Data Capture) for real-time change detection
- Change Tracking for lightweight change detection
- Table/view selection with schema support
- Connection pooling for performance
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.base import SyncState, SyncStatus
from aragora.connectors.enterprise.database.sqlserver import (
    DEFAULT_TIMESTAMP_COLUMNS,
    SQLServerConnector,
    _validate_sql_identifier,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_credentials():
    """Mock credential provider with SQL Server credentials."""
    from tests.connectors.enterprise.conftest import MockCredentialProvider

    return MockCredentialProvider(
        {
            "SQLSERVER_USER": "test_user",
            "SQLSERVER_PASSWORD": "test_password",
        }
    )


@pytest.fixture
def sqlserver_connector(mock_credentials, tmp_path):
    """Create a SQL Server connector for testing."""
    return SQLServerConnector(
        host="localhost",
        port=1433,
        database="testdb",
        schema="dbo",
        tables=["users", "orders"],
        credentials=mock_credentials,
        state_dir=tmp_path / "sync_state",
    )


@pytest.fixture
def cdc_connector(mock_credentials, tmp_path):
    """Create a SQL Server connector with CDC enabled."""
    return SQLServerConnector(
        host="localhost",
        port=1433,
        database="testdb",
        schema="dbo",
        tables=["users"],
        use_cdc=True,
        credentials=mock_credentials,
        state_dir=tmp_path / "sync_state",
    )


@pytest.fixture
def ct_connector(mock_credentials, tmp_path):
    """Create a SQL Server connector with Change Tracking enabled."""
    return SQLServerConnector(
        host="localhost",
        port=1433,
        database="testdb",
        schema="dbo",
        tables=["users"],
        use_change_tracking=True,
        credentials=mock_credentials,
        state_dir=tmp_path / "sync_state",
    )


@pytest.fixture
def sample_users_rows():
    """Sample user table rows."""
    return [
        {
            "id": 1,
            "username": "alice",
            "email": "alice@example.com",
            "created_at": datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 16, 14, 30, tzinfo=timezone.utc),
        },
        {
            "id": 2,
            "username": "bob",
            "email": "bob@example.com",
            "created_at": datetime(2024, 1, 14, 9, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 17, 11, 0, tzinfo=timezone.utc),
        },
    ]


@pytest.fixture
def sample_columns():
    """Sample column information."""
    return [
        {"column_name": "id", "data_type": "int", "is_nullable": "NO"},
        {"column_name": "username", "data_type": "nvarchar", "is_nullable": "NO"},
        {"column_name": "email", "data_type": "nvarchar", "is_nullable": "YES"},
        {"column_name": "created_at", "data_type": "datetime2", "is_nullable": "YES"},
        {"column_name": "updated_at", "data_type": "datetime2", "is_nullable": "YES"},
    ]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSQLServerConnectorInit:
    """Test SQLServerConnector initialization."""

    def test_init_with_defaults(self, mock_credentials, tmp_path):
        """Test initialization with default options."""
        connector = SQLServerConnector(
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.host == "localhost"
        assert connector.port == 1433
        assert connector.database == "master"
        assert connector.schema == "dbo"
        assert connector.tables == []
        assert connector.pool_size == 5
        assert connector.use_cdc is False
        assert connector.use_change_tracking is False

    def test_init_with_custom_options(self, mock_credentials, tmp_path):
        """Test initialization with custom options."""
        connector = SQLServerConnector(
            host="db.example.com",
            port=1434,
            database="production",
            schema="myschema",
            tables=["table1", "table2"],
            timestamp_column="modified_date",
            primary_key_column="pk_id",
            content_columns=["title", "body"],
            use_cdc=True,
            poll_interval_seconds=10,
            pool_size=10,
            credentials=mock_credentials,
            state_dir=tmp_path,
        )

        assert connector.host == "db.example.com"
        assert connector.port == 1434
        assert connector.database == "production"
        assert connector.schema == "myschema"
        assert connector.tables == ["table1", "table2"]
        assert connector.timestamp_column == "modified_date"
        assert connector.primary_key_column == "pk_id"
        assert connector.content_columns == ["title", "body"]
        assert connector.use_cdc is True
        assert connector.poll_interval_seconds == 10
        assert connector.pool_size == 10

    def test_connector_id_format(self, sqlserver_connector):
        """Test connector ID is properly formatted."""
        assert sqlserver_connector.connector_id == "sqlserver_localhost_testdb_dbo"

    def test_source_type(self, sqlserver_connector):
        """Test source type property."""
        from aragora.reasoning.provenance import SourceType

        assert sqlserver_connector.source_type == SourceType.DATABASE

    def test_name_property(self, sqlserver_connector):
        """Test name property."""
        assert sqlserver_connector.name == "SQL Server (testdb.dbo)"


# =============================================================================
# Row to Content Tests
# =============================================================================


class TestRowToContent:
    """Test row to content conversion."""

    def test_row_to_content_basic(self, sqlserver_connector, sample_users_rows):
        """Test basic row to content conversion."""
        content = sqlserver_connector._row_to_content(sample_users_rows[0])

        assert "id: 1" in content
        assert "username: alice" in content
        assert "email: alice@example.com" in content

    def test_row_to_content_with_columns_filter(self, sqlserver_connector, sample_users_rows):
        """Test row to content with column filter."""
        content = sqlserver_connector._row_to_content(
            sample_users_rows[0],
            columns=["username", "email"],
        )

        assert "username: alice" in content
        assert "email: alice@example.com" in content
        assert "id:" not in content

    def test_row_to_content_handles_none_values(self, sqlserver_connector):
        """Test row to content handles None values."""
        row = {"id": 1, "name": "Test", "email": None}
        content = sqlserver_connector._row_to_content(row)

        assert "id: 1" in content
        assert "name: Test" in content

    def test_row_to_content_handles_json_values(self, sqlserver_connector):
        """Test row to content handles JSON/dict values."""
        row = {"id": 1, "metadata": {"key": "value"}}
        content = sqlserver_connector._row_to_content(row)

        assert "id: 1" in content
        assert "metadata:" in content

    def test_row_to_content_handles_datetime(self, sqlserver_connector):
        """Test row to content handles datetime values."""
        row = {"id": 1, "created_at": datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)}
        content = sqlserver_connector._row_to_content(row)

        assert "created_at:" in content
        assert "2024-01-15" in content


# =============================================================================
# Timestamp Column Tests
# =============================================================================


class TestTimestampColumn:
    """Test timestamp column detection."""

    def test_find_explicit_timestamp_column(self, mock_credentials, tmp_path):
        """Test finding explicitly configured timestamp column."""
        connector = SQLServerConnector(
            timestamp_column="modified_date",
            credentials=mock_credentials,
            state_dir=tmp_path,
        )

        columns = [
            {"column_name": "id", "data_type": "int"},
            {"column_name": "updated_at", "data_type": "datetime2"},
            {"column_name": "modified_date", "data_type": "datetime2"},
        ]

        result = connector._find_timestamp_column(columns)
        assert result == "modified_date"

    def test_find_timestamp_column_auto(self, sqlserver_connector):
        """Test auto-detecting timestamp column."""
        columns = [
            {"column_name": "id", "data_type": "int"},
            {"column_name": "name", "data_type": "nvarchar"},
            {"column_name": "updated_at", "data_type": "datetime2"},
        ]

        result = sqlserver_connector._find_timestamp_column(columns)
        assert result == "updated_at"

    def test_find_timestamp_column_none(self, sqlserver_connector):
        """Test when no timestamp column exists."""
        columns = [
            {"column_name": "id", "data_type": "int"},
            {"column_name": "name", "data_type": "nvarchar"},
        ]

        result = sqlserver_connector._find_timestamp_column(columns)
        assert result is None


# =============================================================================
# Connection Pool Tests
# =============================================================================


class TestConnectionPool:
    """Test connection pool management."""

    @pytest.mark.asyncio
    async def test_get_pool_creates_pool(self, sqlserver_connector):
        """Test pool is created on first access."""
        import sys

        mock_pool = MagicMock()
        mock_aioodbc = MagicMock()
        mock_aioodbc.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict(sys.modules, {"aioodbc": mock_aioodbc}):
            pool = await sqlserver_connector._get_pool()

            assert pool == mock_pool
            assert sqlserver_connector._pool == mock_pool

    @pytest.mark.asyncio
    async def test_get_pool_reuses_existing(self, sqlserver_connector):
        """Test existing pool is reused."""
        existing_pool = MagicMock()
        sqlserver_connector._pool = existing_pool

        pool = await sqlserver_connector._get_pool()

        assert pool == existing_pool


# =============================================================================
# Table Discovery Tests
# =============================================================================


class TestTableDiscovery:
    """Test table discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_tables(self, sqlserver_connector):
        """Test discovering tables in schema."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[("users",), ("orders",), ("products",)])
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        # Setup pool.acquire() as async context manager
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        sqlserver_connector._pool = mock_pool

        tables = await sqlserver_connector._discover_tables()

        assert len(tables) == 3
        assert "users" in tables
        assert "orders" in tables


# =============================================================================
# CDC Manager Tests
# =============================================================================


class TestCDCManager:
    """Test CDC manager functionality."""

    def test_cdc_manager_created_lazily(self, sqlserver_connector):
        """Test CDC manager is created lazily."""
        assert sqlserver_connector._cdc_manager is None
        manager = sqlserver_connector.cdc_manager
        assert manager is not None

    def test_add_change_handler(self, sqlserver_connector):
        """Test adding change handlers."""
        mock_handler = MagicMock()
        sqlserver_connector.add_change_handler(mock_handler)

        assert mock_handler in sqlserver_connector._change_handlers


# =============================================================================
# CDC Polling Tests
# =============================================================================


class TestCDCPolling:
    """Test CDC polling functionality."""

    @pytest.mark.asyncio
    async def test_start_cdc_polling_disabled(self, sqlserver_connector):
        """Test CDC doesn't start when disabled."""
        sqlserver_connector.use_cdc = False
        await sqlserver_connector.start_cdc_polling()
        assert sqlserver_connector._cdc_task is None

    @pytest.mark.asyncio
    async def test_stop_cdc_polling(self, cdc_connector):
        """Test stopping CDC polling."""
        import asyncio

        async def dummy_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy_task())
        cdc_connector._cdc_task = task

        await cdc_connector.stop_cdc_polling()

        assert task.cancelled()
        assert cdc_connector._cdc_task is None


# =============================================================================
# Change Tracking Polling Tests
# =============================================================================


class TestChangeTrackingPolling:
    """Test Change Tracking polling functionality."""

    @pytest.mark.asyncio
    async def test_start_change_tracking_polling_disabled(self, sqlserver_connector):
        """Test Change Tracking doesn't start when disabled."""
        sqlserver_connector.use_change_tracking = False
        await sqlserver_connector.start_change_tracking_polling()
        # CT uses same _cdc_task as CDC
        assert sqlserver_connector._cdc_task is None

    @pytest.mark.asyncio
    async def test_stop_ct_via_stop_cdc_polling(self, ct_connector):
        """Test stopping CT via stop_cdc_polling (shared method)."""
        import asyncio

        async def dummy_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy_task())
        ct_connector._cdc_task = task

        # CT uses stop_cdc_polling
        await ct_connector.stop_cdc_polling()

        assert task.cancelled()
        assert ct_connector._cdc_task is None


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, sqlserver_connector):
        """Test health check returns healthy status."""
        mock_cursor = AsyncMock()
        mock_cursor.execute = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(1,))
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        # Setup pool.acquire() as async context manager
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        sqlserver_connector._pool = mock_pool

        health = await sqlserver_connector.health_check()

        assert health["healthy"] is True
        assert health["database"] == "testdb"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, sqlserver_connector):
        """Test health check returns unhealthy on error."""
        with patch.object(
            sqlserver_connector, "_get_pool", side_effect=Exception("Connection failed")
        ):
            health = await sqlserver_connector.health_check()

        assert health["healthy"] is False
        assert "error" in health


# =============================================================================
# Close Tests
# =============================================================================


class TestClose:
    """Test connection cleanup."""

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, sqlserver_connector):
        """Test close method cleans up resources."""
        mock_pool = MagicMock()
        mock_pool.close = MagicMock()
        mock_pool.wait_closed = AsyncMock()
        sqlserver_connector._pool = mock_pool

        await sqlserver_connector.close()

        mock_pool.close.assert_called_once()
        assert sqlserver_connector._pool is None


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test module constants."""

    def test_default_timestamp_columns(self):
        """Test default timestamp column names."""
        assert "updated_at" in DEFAULT_TIMESTAMP_COLUMNS
        assert "modified_at" in DEFAULT_TIMESTAMP_COLUMNS


# =============================================================================
# SQL Identifier Validation Tests
# =============================================================================


class TestSQLIdentifierValidation:
    """Test SQL identifier validation for injection prevention."""

    def test_valid_identifiers(self):
        """Test valid SQL identifiers are accepted."""
        assert _validate_sql_identifier("users", "table") == "users"
        assert _validate_sql_identifier("dbo", "schema") == "dbo"
        assert _validate_sql_identifier("updated_at", "column") == "updated_at"
        assert _validate_sql_identifier("Table1", "table") == "Table1"
        assert _validate_sql_identifier("_internal", "table") == "_internal"
        assert _validate_sql_identifier("user-data", "table") == "user-data"

    def test_empty_identifier_rejected(self):
        """Test empty identifiers are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_sql_identifier("", "table")

    def test_sql_injection_patterns_rejected(self):
        """Test SQL injection patterns are rejected."""
        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("users; DROP TABLE users;--", "table")

        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("users'", "table")

        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("users OR 1=1", "table")

        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("users/**/", "table")

    def test_special_characters_rejected(self):
        """Test special characters are rejected."""
        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("table;name", "table")

        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("table.name", "table")

        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("table()", "table")

    def test_identifier_starting_with_number_rejected(self):
        """Test identifiers starting with numbers are rejected."""
        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("1users", "table")

        with pytest.raises(ValueError, match="Invalid SQL"):
            _validate_sql_identifier("123", "table")

    def test_max_length_128_for_sqlserver(self):
        """Test SQL Server max identifier length is 128."""
        # 128 chars should be valid
        assert _validate_sql_identifier("a" * 128, "table") == "a" * 128

        # 129 chars should fail
        with pytest.raises(ValueError, match="too long"):
            _validate_sql_identifier("a" * 129, "table")
