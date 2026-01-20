"""
Tests for Snowflake enterprise connector.

Tests cover:
- SnowflakeConnector initialization and configuration
- Connection parameter building
- Query execution (mocked)
- Table discovery and column inspection
- Domain inference
- Row to content conversion
- Time travel queries
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Connector Initialization Tests
# ============================================================================


class TestSnowflakeConnectorInit:
    """Tests for SnowflakeConnector initialization."""

    def test_basic_initialization(self):
        """Test basic connector initialization."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org-account",
            warehouse="COMPUTE_WH",
            database="ANALYTICS",
            schema="PUBLIC",
        )

        assert connector.account == "org-account"
        assert connector.warehouse == "COMPUTE_WH"
        assert connector.database == "ANALYTICS"
        assert connector.schema == "PUBLIC"

    def test_initialization_with_options(self):
        """Test initialization with all options."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org-account",
            warehouse="COMPUTE_WH",
            database="ANALYTICS",
            schema="STAGING",
            role="ANALYST",
            tables=["CUSTOMERS", "ORDERS"],
            timestamp_column="UPDATED_AT",
            primary_key_column="CUSTOMER_ID",
            content_columns=["NAME", "EMAIL"],
            use_change_tracking=True,
            pool_size=5,
        )

        assert connector.role == "ANALYST"
        assert connector.tables == ["CUSTOMERS", "ORDERS"]
        assert connector.timestamp_column == "UPDATED_AT"
        assert connector.primary_key_column == "CUSTOMER_ID"
        assert connector.content_columns == ["NAME", "EMAIL"]
        assert connector.use_change_tracking is True
        assert connector.pool_size == 5

    def test_connector_id_generation(self):
        """Test connector ID is generated correctly."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="myorg-myaccount",
            warehouse="WH",
            database="DB",
            schema="SCHEMA",
        )

        assert "snowflake_myorg-myaccount_DB_SCHEMA" in connector.connector_id

    def test_source_type(self):
        """Test source type is DATABASE."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector
        from aragora.reasoning.provenance import SourceType

        connector = SnowflakeConnector(
            account="org-account",
            warehouse="WH",
            database="DB",
        )

        assert connector.source_type == SourceType.DATABASE

    def test_name_property(self):
        """Test name property includes database and schema."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org-account",
            warehouse="WH",
            database="ANALYTICS",
            schema="PROD",
        )

        assert "ANALYTICS.PROD" in connector.name


# ============================================================================
# Connection Parameter Tests
# ============================================================================


class TestSnowflakeConnectionParams:
    """Tests for connection parameter building."""

    def test_basic_params_with_password(self):
        """Test connection params with password auth."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org-account",
            warehouse="COMPUTE_WH",
            database="DB",
            schema="PUBLIC",
            role="ANALYST",
        )

        with patch.dict(
            "os.environ",
            {
                "SNOWFLAKE_USER": "testuser",
                "SNOWFLAKE_PASSWORD": "testpass",
            },
        ):
            params = connector._get_connection_params()

        assert params["account"] == "org-account"
        assert params["warehouse"] == "COMPUTE_WH"
        assert params["database"] == "DB"
        assert params["schema"] == "PUBLIC"
        assert params["role"] == "ANALYST"
        assert params["user"] == "testuser"
        assert params["password"] == "testpass"

    def test_params_with_authenticator(self):
        """Test connection params with external authenticator."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org-account",
            warehouse="WH",
            database="DB",
        )

        with patch.dict(
            "os.environ",
            {
                "SNOWFLAKE_USER": "testuser",
                "SNOWFLAKE_AUTHENTICATOR": "externalbrowser",
            },
        ):
            params = connector._get_connection_params()

        assert params["authenticator"] == "externalbrowser"
        assert "password" not in params

    def test_missing_user_raises_error(self):
        """Test error when SNOWFLAKE_USER is not set."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org-account",
            warehouse="WH",
            database="DB",
        )

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="SNOWFLAKE_USER"):
                connector._get_connection_params()

    def test_missing_credentials_raises_error(self):
        """Test error when no auth method is configured."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org-account",
            warehouse="WH",
            database="DB",
        )

        with patch.dict(
            "os.environ",
            {
                "SNOWFLAKE_USER": "testuser",
                # No password, key, or authenticator
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="credentials not configured"):
                connector._get_connection_params()


# ============================================================================
# Domain Inference Tests
# ============================================================================


class TestSnowflakeDomainInference:
    """Tests for domain inference from table names."""

    def test_user_tables(self):
        """Test user-related tables get correct domain."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        assert connector._infer_domain("USERS") == "operational/users"
        assert connector._infer_domain("USER_ACCOUNTS") == "operational/users"
        assert connector._infer_domain("CUSTOMER_PROFILES") == "operational/users"

    def test_financial_tables(self):
        """Test financial tables get correct domain."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        assert connector._infer_domain("ORDERS") == "financial/transactions"
        assert connector._infer_domain("PAYMENTS") == "financial/transactions"
        assert connector._infer_domain("BILLING_HISTORY") == "financial/transactions"

    def test_product_tables(self):
        """Test product tables get correct domain."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        assert connector._infer_domain("PRODUCTS") == "operational/products"
        assert connector._infer_domain("INVENTORY") == "operational/products"
        assert connector._infer_domain("CATALOG_ITEMS") == "operational/products"

    def test_analytics_tables(self):
        """Test analytics tables get correct domain."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        assert connector._infer_domain("METRICS") == "analytics/metrics"
        assert connector._infer_domain("DAILY_STATS") == "analytics/metrics"
        assert connector._infer_domain("DIM_DATE") == "analytics/warehouse"
        assert connector._infer_domain("FACT_SALES") == "analytics/warehouse"

    def test_log_tables(self):
        """Test log tables get correct domain."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        assert connector._infer_domain("AUDIT_LOGS") == "operational/logs"
        assert connector._infer_domain("EVENT_LOG") == "operational/logs"

    def test_unknown_tables(self):
        """Test unknown tables get default domain."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        assert connector._infer_domain("RANDOM_TABLE") == "general/database"
        assert connector._infer_domain("XYZ") == "general/database"


# ============================================================================
# Row to Content Tests
# ============================================================================


class TestSnowflakeRowToContent:
    """Tests for converting rows to text content."""

    def test_simple_row(self):
        """Test converting simple row to content."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        row = {
            "ID": 123,
            "NAME": "John Doe",
            "EMAIL": "john@example.com",
        }

        content = connector._row_to_content(row)

        assert "ID: 123" in content
        assert "NAME: John Doe" in content
        assert "EMAIL: john@example.com" in content

    def test_row_with_datetime(self):
        """Test converting row with datetime values."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        now = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        row = {
            "ID": 1,
            "CREATED_AT": now,
        }

        content = connector._row_to_content(row)

        assert "2024-01-15" in content

    def test_row_with_dict_value(self):
        """Test converting row with nested dict."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        row = {
            "ID": 1,
            "METADATA": {"key": "value"},
        }

        content = connector._row_to_content(row)

        assert '"key"' in content or "key" in content

    def test_row_with_column_filter(self):
        """Test converting row with column filter."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        row = {
            "ID": 1,
            "NAME": "Test",
            "SECRET": "hidden",
        }

        content = connector._row_to_content(row, columns=["ID", "NAME"])

        assert "ID: 1" in content
        assert "NAME: Test" in content
        assert "SECRET" not in content

    def test_row_with_none_values(self):
        """Test row with None values are skipped."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        row = {
            "ID": 1,
            "NAME": None,
            "EMAIL": "test@example.com",
        }

        content = connector._row_to_content(row)

        assert "ID: 1" in content
        assert "EMAIL: test@example.com" in content
        assert "NAME:" not in content


# ============================================================================
# Timestamp Column Detection Tests
# ============================================================================


class TestSnowflakeTimestampColumn:
    """Tests for timestamp column detection."""

    def test_explicit_timestamp_column(self):
        """Test explicit timestamp column is used."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org",
            warehouse="WH",
            database="DB",
            timestamp_column="MODIFIED_TS",
        )

        columns = [
            {"COLUMN_NAME": "ID", "DATA_TYPE": "NUMBER"},
            {"COLUMN_NAME": "UPDATED_AT", "DATA_TYPE": "TIMESTAMP_NTZ"},
        ]

        result = connector._find_timestamp_column(columns)
        assert result == "MODIFIED_TS"

    def test_auto_detect_updated_at(self):
        """Test auto-detection of updated_at column."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        columns = [
            {"COLUMN_NAME": "ID", "DATA_TYPE": "NUMBER"},
            {"COLUMN_NAME": "UPDATED_AT", "DATA_TYPE": "TIMESTAMP_NTZ"},
        ]

        result = connector._find_timestamp_column(columns)
        assert result == "UPDATED_AT"

    def test_auto_detect_modified_at(self):
        """Test auto-detection of modified_at column."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        columns = [
            {"COLUMN_NAME": "ID", "DATA_TYPE": "NUMBER"},
            {"COLUMN_NAME": "MODIFIED_AT", "DATA_TYPE": "TIMESTAMP_NTZ"},
        ]

        result = connector._find_timestamp_column(columns)
        assert result == "MODIFIED_AT"

    def test_no_timestamp_column_found(self):
        """Test when no timestamp column is found."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        columns = [
            {"COLUMN_NAME": "ID", "DATA_TYPE": "NUMBER"},
            {"COLUMN_NAME": "NAME", "DATA_TYPE": "VARCHAR"},
        ]

        result = connector._find_timestamp_column(columns)
        assert result is None


# ============================================================================
# Query Execution Tests (Mocked)
# ============================================================================


class TestSnowflakeQueryExecution:
    """Tests for query execution."""

    @pytest.mark.asyncio
    async def test_async_query(self):
        """Test async query execution via thread pool."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        # Mock _execute_query
        connector._execute_query = MagicMock(
            return_value=[
                {"ID": 1, "NAME": "Test"},
            ]
        )

        result = await connector._async_query("SELECT * FROM TEST")

        assert len(result) == 1
        assert result[0]["NAME"] == "Test"


# ============================================================================
# Table Discovery Tests (Mocked)
# ============================================================================


class TestSnowflakeTableDiscovery:
    """Tests for table discovery."""

    @pytest.mark.asyncio
    async def test_discover_tables(self):
        """Test table discovery query."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org", warehouse="WH", database="DB", schema="PUBLIC"
        )

        # Mock _async_query
        connector._async_query = AsyncMock(
            return_value=[
                {"TABLE_NAME": "CUSTOMERS"},
                {"TABLE_NAME": "ORDERS"},
            ]
        )

        tables = await connector._discover_tables()

        assert "CUSTOMERS" in tables
        assert "ORDERS" in tables

    @pytest.mark.asyncio
    async def test_get_table_columns(self):
        """Test column discovery query."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org", warehouse="WH", database="DB", schema="PUBLIC"
        )

        # Mock _async_query
        connector._async_query = AsyncMock(
            return_value=[
                {"COLUMN_NAME": "ID", "DATA_TYPE": "NUMBER", "IS_NULLABLE": "NO"},
                {"COLUMN_NAME": "NAME", "DATA_TYPE": "VARCHAR", "IS_NULLABLE": "YES"},
            ]
        )

        columns = await connector._get_table_columns("CUSTOMERS")

        assert len(columns) == 2
        assert columns[0]["COLUMN_NAME"] == "ID"
        assert columns[1]["DATA_TYPE"] == "VARCHAR"


# ============================================================================
# Time Travel Tests (Mocked)
# ============================================================================


class TestSnowflakeTimeTravel:
    """Tests for time travel queries."""

    @pytest.mark.asyncio
    async def test_time_travel_query(self):
        """Test time travel query builds correct SQL."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org", warehouse="WH", database="DB", schema="PUBLIC"
        )

        captured_queries = []

        async def mock_query(query, params=None):
            captured_queries.append(query)
            return [{"ID": 1}]

        connector._async_query = mock_query

        timestamp = datetime(2024, 1, 15, 10, 0, 0)
        await connector.time_travel_query("CUSTOMERS", timestamp, limit=50)

        assert len(captured_queries) == 1
        assert "AT(TIMESTAMP =>" in captured_queries[0]
        assert "2024-01-15 10:00:00" in captured_queries[0]
        assert "LIMIT 50" in captured_queries[0]


# ============================================================================
# Table Stats Tests (Mocked)
# ============================================================================


class TestSnowflakeTableStats:
    """Tests for table statistics."""

    @pytest.mark.asyncio
    async def test_get_table_stats(self):
        """Test table stats query."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org", warehouse="WH", database="DB", schema="PUBLIC"
        )

        # Mock _async_query
        connector._async_query = AsyncMock(
            return_value=[
                {"row_count": 1000, "max_id": 1050},
            ]
        )

        stats = await connector.get_table_stats("CUSTOMERS")

        assert stats["row_count"] == 1000
        assert stats["max_id"] == 1050

    @pytest.mark.asyncio
    async def test_get_table_stats_error(self):
        """Test table stats error handling."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(
            account="org", warehouse="WH", database="DB", schema="PUBLIC"
        )

        # Mock _async_query to raise error
        connector._async_query = AsyncMock(side_effect=Exception("Query failed"))

        stats = await connector.get_table_stats("NONEXISTENT")

        assert stats["row_count"] == 0
        assert stats["max_id"] is None


# ============================================================================
# Webhook Tests
# ============================================================================


class TestSnowflakeWebhooks:
    """Tests for webhook handling."""

    @pytest.mark.asyncio
    async def test_webhook_with_table_operation(self):
        """Test webhook with table and operation triggers sync."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        # Mock sync
        connector.sync = AsyncMock()

        payload = {
            "table": "CUSTOMERS",
            "operation": "INSERT",
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_webhook_without_table(self):
        """Test webhook without table returns False."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        payload = {"status": "ok"}

        result = await connector.handle_webhook(payload)
        assert result is False


# ============================================================================
# Fetch Tests
# ============================================================================


class TestSnowflakeFetch:
    """Tests for fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_invalid_id(self):
        """Test fetch with invalid evidence ID."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        result = await connector.fetch("invalid-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_wrong_account(self):
        """Test fetch with wrong account in ID."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        result = await connector.fetch("sf:other-org:DB:TABLE:hash123")
        assert result is None


# ============================================================================
# Close/Cleanup Tests
# ============================================================================


class TestSnowflakeCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing connector releases resources."""
        from aragora.connectors.enterprise.database.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(account="org", warehouse="WH", database="DB")

        # Mock connection
        mock_conn = MagicMock()
        connector._connection = mock_conn

        await connector.close()

        mock_conn.close.assert_called_once()
        assert connector._connection is None


# ============================================================================
# Module Export Tests
# ============================================================================


class TestSnowflakeExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ includes expected classes."""
        from aragora.connectors.enterprise.database import snowflake

        assert "SnowflakeConnector" in snowflake.__all__
