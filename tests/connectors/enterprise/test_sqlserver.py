"""
Tests for SQL Server Enterprise Connector.

Tests the SQL Server integration including:
- Incremental sync using transaction timestamps or CDC columns
- CDC (Change Data Capture) for real-time change detection
- Change Tracking for lightweight change detection
- Table/view selection with schema support
- Connection pooling for performance
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib

import pytest

from aragora.connectors.enterprise.base import SyncState, SyncStatus
from aragora.connectors.enterprise.database.sqlserver import (
    SQLServerConnector,
    DEFAULT_TIMESTAMP_COLUMNS,
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
        enable_cdc=True,
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
        enable_change_tracking=True,
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
def sample_orders_rows():
    """Sample order table rows."""
    return [
        {
            "id": 101,
            "user_id": 1,
            "total": 99.99,
            "status": "completed",
            "created_at": datetime(2024, 1, 16, 10, 0, tzinfo=timezone.utc),
        },
        {
            "id": 102,
            "user_id": 2,
            "total": 149.50,
            "status": "pending",
            "created_at": datetime(2024, 1, 17, 9, 0, tzinfo=timezone.utc),
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


@pytest.fixture
def sample_cdc_changes():
    """Sample CDC change records."""
    return [
        {
            "__$operation": 2,  # Insert
            "__$start_lsn": b"\x00\x00\x00\x00\x00\x00\x00\x01",
            "id": 1,
            "username": "alice",
            "email": "alice@example.com",
        },
        {
            "__$operation": 4,  # Update (after image)
            "__$start_lsn": b"\x00\x00\x00\x00\x00\x00\x00\x02",
            "id": 1,
            "username": "alice_updated",
            "email": "alice@example.com",
        },
        {
            "__$operation": 1,  # Delete
            "__$start_lsn": b"\x00\x00\x00\x00\x00\x00\x00\x03",
            "id": 2,
            "username": "bob",
            "email": "bob@example.com",
        },
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
        assert connector.enable_cdc is False
        assert connector.enable_change_tracking is False

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
            enable_cdc=True,
            cdc_poll_interval=5.0,
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
        assert connector.enable_cdc is True
        assert connector.cdc_poll_interval == 5.0
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
        # None values should not appear in content
        lines = content.split("\n")
        assert not any("email:" in line for line in lines if "None" in line)

    def test_row_to_content_handles_json_values(self, sqlserver_connector):
        """Test row to content handles JSON/dict values."""
        row = {"id": 1, "metadata": {"key": "value", "nested": {"a": 1}}}
        content = sqlserver_connector._row_to_content(row)

        assert "id: 1" in content
        assert "metadata:" in content
        assert '"key"' in content or "key" in content

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
        assert result == "modified_date"  # Uses explicit config

    def test_find_timestamp_column_auto(self, sqlserver_connector):
        """Test auto-detecting timestamp column."""
        columns = [
            {"column_name": "id", "data_type": "int"},
            {"column_name": "name", "data_type": "nvarchar"},
            {"column_name": "updated_at", "data_type": "datetime2"},
        ]

        result = sqlserver_connector._find_timestamp_column(columns)
        assert result == "updated_at"

    def test_find_timestamp_column_modified_at(self, sqlserver_connector):
        """Test detecting modified_at column."""
        columns = [
            {"column_name": "id", "data_type": "int"},
            {"column_name": "modified_at", "data_type": "datetime2"},
        ]

        result = sqlserver_connector._find_timestamp_column(columns)
        assert result == "modified_at"

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

        # Inject mock aioodbc module
        with patch.dict(sys.modules, {"aioodbc": mock_aioodbc}):
            pool = await sqlserver_connector._get_pool()

            assert pool == mock_pool
            assert sqlserver_connector._pool == mock_pool
            mock_aioodbc.create_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pool_reuses_existing(self, sqlserver_connector):
        """Test existing pool is reused."""
        existing_pool = MagicMock()
        sqlserver_connector._pool = existing_pool

        pool = await sqlserver_connector._get_pool()

        assert pool == existing_pool

    @pytest.mark.asyncio
    async def test_get_pool_aioodbc_not_installed(self, sqlserver_connector):
        """Test error when aioodbc not installed."""
        import sys

        # Remove aioodbc from modules and ensure import fails
        with patch.dict(sys.modules, {"aioodbc": None}):
            with pytest.raises(ImportError):
                await sqlserver_connector._get_pool()


# =============================================================================
# Table Discovery Tests
# =============================================================================


class TestTableDiscovery:
    """Test table discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_tables(self, sqlserver_connector):
        """Test discovering tables in schema."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("users",),
                ("orders",),
                ("products",),
            ]
        )
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        sqlserver_connector._pool = mock_pool

        tables = await sqlserver_connector._discover_tables()

        assert len(tables) == 3
        assert "users" in tables
        assert "orders" in tables
        assert "products" in tables

    @pytest.mark.asyncio
    async def test_get_table_columns(self, sqlserver_connector, sample_columns):
        """Test getting column information."""
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                (col["column_name"], col["data_type"], col["is_nullable"]) for col in sample_columns
            ]
        )
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=None)

        mock_conn = MagicMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        sqlserver_connector._pool = mock_pool

        columns = await sqlserver_connector._get_table_columns("users")

        assert len(columns) == 5
        assert columns[0]["column_name"] == "id"
        assert columns[1]["column_name"] == "username"


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
        assert sqlserver_connector._cdc_manager is manager

    def test_add_change_handler(self, sqlserver_connector):
        """Test adding change handlers."""
        mock_handler = MagicMock()
        sqlserver_connector.add_change_handler(mock_handler)

        assert mock_handler in sqlserver_connector._change_handlers
        # CDC manager should be reset
        assert sqlserver_connector._cdc_manager is None


# =============================================================================
# CDC Operation Mapping Tests
# =============================================================================


class TestCDCOperationMapping:
    """Test CDC operation code mapping."""

    def test_map_cdc_operation_insert(self, cdc_connector):
        """Test mapping CDC insert operation."""
        from aragora.connectors.enterprise.database.cdc import ChangeOperation

        op = cdc_connector._map_cdc_operation(2)
        assert op == ChangeOperation.INSERT

    def test_map_cdc_operation_update(self, cdc_connector):
        """Test mapping CDC update operation."""
        from aragora.connectors.enterprise.database.cdc import ChangeOperation

        op = cdc_connector._map_cdc_operation(4)
        assert op == ChangeOperation.UPDATE

    def test_map_cdc_operation_delete(self, cdc_connector):
        """Test mapping CDC delete operation."""
        from aragora.connectors.enterprise.database.cdc import ChangeOperation

        op = cdc_connector._map_cdc_operation(1)
        assert op == ChangeOperation.DELETE

    def test_map_cdc_operation_unknown(self, cdc_connector):
        """Test mapping unknown operation returns None."""
        op = cdc_connector._map_cdc_operation(99)
        assert op is None


# =============================================================================
# Change Tracking Operation Mapping Tests
# =============================================================================


class TestChangeTrackingOperationMapping:
    """Test Change Tracking operation mapping."""

    def test_map_ct_operation_insert(self, ct_connector):
        """Test mapping CT insert operation."""
        from aragora.connectors.enterprise.database.cdc import ChangeOperation

        op = ct_connector._map_ct_operation("I")
        assert op == ChangeOperation.INSERT

    def test_map_ct_operation_update(self, ct_connector):
        """Test mapping CT update operation."""
        from aragora.connectors.enterprise.database.cdc import ChangeOperation

        op = ct_connector._map_ct_operation("U")
        assert op == ChangeOperation.UPDATE

    def test_map_ct_operation_delete(self, ct_connector):
        """Test mapping CT delete operation."""
        from aragora.connectors.enterprise.database.cdc import ChangeOperation

        op = ct_connector._map_ct_operation("D")
        assert op == ChangeOperation.DELETE

    def test_map_ct_operation_unknown(self, ct_connector):
        """Test mapping unknown CT operation returns None."""
        op = ct_connector._map_ct_operation("X")
        assert op is None


# =============================================================================
# CDC Polling Tests
# =============================================================================


class TestCDCPolling:
    """Test CDC polling functionality."""

    @pytest.mark.asyncio
    async def test_start_cdc_disabled(self, sqlserver_connector):
        """Test CDC doesn't start when disabled."""
        sqlserver_connector.enable_cdc = False
        await sqlserver_connector.start_cdc()
        assert sqlserver_connector._cdc_task is None

    @pytest.mark.asyncio
    async def test_stop_cdc(self, cdc_connector):
        """Test stopping CDC polling."""
        import asyncio

        # Create a mock task
        async def dummy_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy_task())
        cdc_connector._cdc_task = task

        await cdc_connector.stop_cdc()

        assert task.cancelled()
        assert cdc_connector._cdc_task is None


# =============================================================================
# Change Tracking Polling Tests
# =============================================================================


class TestChangeTrackingPolling:
    """Test Change Tracking polling functionality."""

    @pytest.mark.asyncio
    async def test_start_change_tracking_disabled(self, sqlserver_connector):
        """Test Change Tracking doesn't start when disabled."""
        sqlserver_connector.enable_change_tracking = False
        await sqlserver_connector.start_change_tracking()
        assert sqlserver_connector._ct_task is None

    @pytest.mark.asyncio
    async def test_stop_change_tracking(self, ct_connector):
        """Test stopping Change Tracking polling."""
        import asyncio

        # Create a mock task
        async def dummy_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy_task())
        ct_connector._ct_task = task

        await ct_connector.stop_change_tracking()

        assert task.cancelled()
        assert ct_connector._ct_task is None


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
        mock_pool.acquire = MagicMock(return_value=mock_conn)
        sqlserver_connector._pool = mock_pool

        health = await sqlserver_connector.health_check()

        assert health["healthy"] is True
        assert health["database"] == "testdb"
        assert health["host"] == "localhost"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, sqlserver_connector):
        """Test health check returns unhealthy on error."""
        sqlserver_connector._pool = None

        # Mock _get_pool to raise an exception
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
        mock_pool.close = AsyncMock()
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
        assert "last_modified" in DEFAULT_TIMESTAMP_COLUMNS
        assert "timestamp" in DEFAULT_TIMESTAMP_COLUMNS
