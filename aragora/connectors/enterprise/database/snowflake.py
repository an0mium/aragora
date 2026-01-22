"""
Snowflake Enterprise Connector.

Features:
- Incremental sync using timestamps or CHANGES clause
- Multi-warehouse, database, and schema support
- Table/view selection with role-based access
- Connection pooling for performance
- Time travel support for historical data
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

# Default columns to use for change tracking
DEFAULT_TIMESTAMP_COLUMNS = [
    "updated_at",
    "modified_at",
    "last_modified",
    "_updated_at",
    "metadata$action",
]


class SnowflakeConnector(EnterpriseConnector):
    """
    Snowflake connector for enterprise data sync.

    Supports:
    - Incremental sync using timestamp columns or CHANGES clause
    - Multi-database and schema access
    - Warehouse selection
    - Role-based access control
    - Time travel queries

    Authentication:
    - Username/password
    - Key pair authentication
    - OAuth (via externalbrowser)

    Usage:
        connector = SnowflakeConnector(
            account="org-account",
            warehouse="COMPUTE_WH",
            database="ANALYTICS",
            schema="PUBLIC",
            tables=["CUSTOMERS", "ORDERS"],
        )
        result = await connector.sync()
    """

    def __init__(
        self,
        account: str,
        warehouse: str,
        database: str,
        schema: str = "PUBLIC",
        role: Optional[str] = None,
        tables: Optional[List[str]] = None,
        timestamp_column: Optional[str] = None,
        primary_key_column: str = "ID",
        content_columns: Optional[List[str]] = None,
        use_change_tracking: bool = False,
        pool_size: int = 3,
        **kwargs,
    ):
        """
        Initialize Snowflake connector.

        Args:
            account: Snowflake account identifier (org-account)
            warehouse: Compute warehouse to use
            database: Database name
            schema: Schema name (default: PUBLIC)
            role: Role to use (None = default role)
            tables: Specific tables to sync (None = all tables)
            timestamp_column: Column to use for incremental sync
            primary_key_column: Primary key column name
            content_columns: Columns to include in content (None = all)
            use_change_tracking: Use Snowflake Change Tracking if available
            pool_size: Connection pool size
        """
        connector_id = f"snowflake_{account}_{database}_{schema}"
        super().__init__(connector_id=connector_id, **kwargs)

        self.account = account
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.role = role
        self.tables = tables or []
        self.timestamp_column = timestamp_column
        self.primary_key_column = primary_key_column
        self.content_columns = content_columns
        self.use_change_tracking = use_change_tracking
        self.pool_size = pool_size

        self._connection = None
        self._executor = ThreadPoolExecutor(max_workers=pool_size)

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def name(self) -> str:
        return f"Snowflake ({self.database}.{self.schema})"

    def _get_connection_params(self) -> Dict[str, Any]:
        """Build connection parameters."""
        import os

        params = {
            "account": self.account,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
        }

        if self.role:
            params["role"] = self.role

        # Get credentials from environment
        user = os.environ.get("SNOWFLAKE_USER")
        password = os.environ.get("SNOWFLAKE_PASSWORD")
        private_key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
        authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR")

        if not user:
            raise ValueError("SNOWFLAKE_USER environment variable not set")

        params["user"] = user

        if authenticator:
            params["authenticator"] = authenticator
        elif private_key_path:
            # Key pair authentication
            with open(private_key_path, "rb") as key_file:
                from cryptography.hazmat.backends import default_backend
                from cryptography.hazmat.primitives import serialization

                p_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "").encode()
                    or None,
                    backend=default_backend(),
                )
                params["private_key"] = p_key  # type: ignore[assignment]
        elif password:
            params["password"] = password
        else:
            raise ValueError(
                "Snowflake credentials not configured. Set SNOWFLAKE_PASSWORD, "
                "SNOWFLAKE_PRIVATE_KEY_PATH, or SNOWFLAKE_AUTHENTICATOR"
            )

        return params

    def _get_connection(self):
        """Get or create Snowflake connection."""
        if self._connection is not None:
            return self._connection

        try:
            import snowflake.connector

            params = self._get_connection_params()
            self._connection = snowflake.connector.connect(**params)
            return self._connection

        except ImportError:
            logger.error(
                "snowflake-connector-python not installed. Run: pip install snowflake-connector-python"
            )
            raise

    def _execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a query synchronously (for thread pool)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            return [dict(zip(columns, row)) for row in rows]

        finally:
            cursor.close()

    async def _async_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query asynchronously via thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._execute_query,
            query,
            params,
        )

    async def _discover_tables(self) -> List[str]:
        """Discover tables in the schema."""
        query = """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = %s
            AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """
        rows = await self._async_query(query, (self.schema,))
        return [row["TABLE_NAME"] for row in rows]

    async def _get_table_columns(self, table: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        query = """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
        """
        return await self._async_query(query, (self.schema, table))

    async def _check_change_tracking(self, table: str) -> bool:
        """Check if change tracking is enabled for a table."""
        if not self.use_change_tracking:
            return False

        try:
            query = f"""
                SELECT SYSTEM$STREAM_HAS_DATA('{self.database}.{self.schema}.{table}_CHANGES') as has_data
            """
            await self._async_query(query)
            return True
        except (ValueError, RuntimeError, OSError) as e:
            logger.debug(f"Change tracking check failed for {table}: {e}")
            return False

    def _find_timestamp_column(self, columns: List[Dict[str, Any]]) -> Optional[str]:
        """Find a suitable timestamp column for incremental sync."""
        if self.timestamp_column:
            return self.timestamp_column

        column_names = {col["COLUMN_NAME"].lower() for col in columns}
        for candidate in DEFAULT_TIMESTAMP_COLUMNS:
            if candidate.lower() in column_names:
                # Return actual column name (case-sensitive)
                for col in columns:
                    if col["COLUMN_NAME"].lower() == candidate.lower():
                        return col["COLUMN_NAME"]
        return None

    def _row_to_content(self, row: Dict[str, Any], columns: Optional[List[str]] = None) -> str:
        """Convert a row to text content for indexing."""
        if columns:
            filtered = {k: v for k, v in row.items() if k in columns}
        else:
            filtered = row

        # Convert to readable format
        parts = []
        for key, value in filtered.items():
            if value is not None:
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, (dict, list)):
                    value = json.dumps(value, default=str)
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    def _infer_domain(self, table: str) -> str:
        """Infer domain from table name."""
        table_lower = table.lower()

        if any(t in table_lower for t in ["user", "account", "profile", "customer"]):
            return "operational/users"
        elif any(
            t in table_lower for t in ["order", "invoice", "payment", "transaction", "billing"]
        ):
            return "financial/transactions"
        elif any(t in table_lower for t in ["product", "inventory", "catalog", "item"]):
            return "operational/products"
        elif any(t in table_lower for t in ["log", "audit", "event", "activity"]):
            return "operational/logs"
        elif any(t in table_lower for t in ["config", "setting", "preference", "parameter"]):
            return "technical/configuration"
        elif any(t in table_lower for t in ["metric", "analytics", "stat", "kpi"]):
            return "analytics/metrics"
        elif any(t in table_lower for t in ["dim_", "fact_", "stage_"]):
            return "analytics/warehouse"

        return "general/database"

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield items to sync from Snowflake tables.

        Uses timestamp columns or change tracking for incremental sync.
        """
        # Get tables to sync
        tables = self.tables or await self._discover_tables()
        state.items_total = len(tables)

        for table in tables:
            try:
                columns = await self._get_table_columns(table)
                ts_column = self._find_timestamp_column(columns)
                has_change_tracking = await self._check_change_tracking(table)

                # Determine sync method
                if has_change_tracking:
                    # Use change tracking stream
                    query = f"""
                        SELECT *
                        FROM {self.database}.{self.schema}.{table}_CHANGES
                        LIMIT {batch_size}
                    """
                    rows = await self._async_query(query)

                elif ts_column and state.last_item_timestamp:
                    # Incremental sync using timestamp
                    query = f"""
                        SELECT *
                        FROM {self.database}.{self.schema}.{table}
                        WHERE "{ts_column}" > %s
                        ORDER BY "{ts_column}" ASC
                        LIMIT {batch_size}
                    """
                    rows = await self._async_query(query, (state.last_item_timestamp,))

                else:
                    # Full sync with cursor-based pagination
                    if state.cursor and state.cursor.startswith(f"{table}:"):
                        last_id = state.cursor.split(":", 1)[1]
                        query = f"""
                            SELECT *
                            FROM {self.database}.{self.schema}.{table}
                            WHERE "{self.primary_key_column}" > %s
                            ORDER BY "{self.primary_key_column}" ASC
                            LIMIT {batch_size}
                        """
                        rows = await self._async_query(query, (last_id,))
                    else:
                        query = f"""
                            SELECT *
                            FROM {self.database}.{self.schema}.{table}
                            ORDER BY "{self.primary_key_column}" ASC
                            LIMIT {batch_size}
                        """
                        rows = await self._async_query(query)

                for row in rows:
                    pk_value = row.get(self.primary_key_column, "")

                    # Generate content
                    content = self._row_to_content(row, self.content_columns)

                    # Get timestamp if available
                    updated_at = datetime.now(timezone.utc)
                    if ts_column and row.get(ts_column):
                        ts_value = row[ts_column]
                        if isinstance(ts_value, datetime):
                            updated_at = (
                                ts_value.replace(tzinfo=timezone.utc)
                                if ts_value.tzinfo is None
                                else ts_value
                            )

                    # Create sync item
                    item_id = f"sf:{self.account}:{self.database}:{table}:{hashlib.sha256(str(pk_value).encode()).hexdigest()[:12]}"

                    yield SyncItem(
                        id=item_id,
                        content=content[:100000],
                        source_type="database",
                        source_id=f"snowflake://{self.account}/{self.database}/{self.schema}/{table}/{pk_value}",
                        title=f"{table} #{pk_value}",
                        url=f"snowflake://{self.account}/{self.database}/{self.schema}/{table}?id={pk_value}",
                        updated_at=updated_at,
                        domain=self._infer_domain(table),
                        confidence=0.85,
                        metadata={
                            "account": self.account,
                            "warehouse": self.warehouse,
                            "database": self.database,
                            "schema": self.schema,
                            "table": table,
                            "primary_key": str(pk_value),
                            "columns": list(row.keys()),
                        },
                    )

                    # Update cursor
                    state.cursor = f"{table}:{pk_value}"

                    # Update timestamp for incremental
                    if ts_column and updated_at:
                        state.last_item_timestamp = updated_at

            except Exception as e:
                logger.warning(f"Failed to sync table {table}: {e}")
                state.errors.append(f"{table}: {str(e)}")
                continue

    async def search(
        self,
        query: str,
        limit: int = 10,
        table: Optional[str] = None,
        **kwargs,
    ) -> list:
        """
        Search across indexed tables.

        Note: Full-text search in Snowflake requires SEARCH OPTIMIZATION or
        Cortex functions. This falls back to ILIKE for basic search.
        """
        results = []

        tables = [table] if table else (self.tables or await self._discover_tables())

        for tbl in tables[:5]:  # Limit to first 5 tables
            try:
                columns = await self._get_table_columns(tbl)
                text_columns = [
                    c["COLUMN_NAME"]
                    for c in columns
                    if c["DATA_TYPE"] in ("TEXT", "VARCHAR", "STRING", "CHAR")
                ]

                if not text_columns:
                    continue

                # Build ILIKE conditions
                conditions = " OR ".join([f'"{col}" ILIKE %s' for col in text_columns[:3]])
                search_pattern = f"%{query}%"

                search_query = f"""
                    SELECT *
                    FROM {self.database}.{self.schema}.{tbl}
                    WHERE {conditions}
                    LIMIT {limit}
                """

                params = tuple([search_pattern] * min(len(text_columns), 3))
                rows = await self._async_query(search_query, params)

                for row in rows:
                    results.append(
                        {
                            "table": tbl,
                            "data": row,
                            "rank": 0.5,
                        }
                    )

            except Exception as e:
                logger.debug(f"Search failed on {tbl}: {e}")
                continue

        return sorted(results, key=lambda x: x.get("rank", 0), reverse=True)[:limit]  # type: ignore[arg-type,return-value]

    async def fetch(self, evidence_id: str) -> Optional[Any]:
        """Fetch a specific row by evidence ID."""
        if not evidence_id.startswith("sf:"):
            return None

        parts = evidence_id.split(":")
        if len(parts) < 5:
            return None

        account, database, _table, _pk_hash = parts[1], parts[2], parts[3], parts[4]

        if account != self.account or database != self.database:
            return None

        # We can't reverse the hash, so this is limited
        logger.debug(f"[{self.name}] Fetch not implemented for hash-based IDs")
        return None

    async def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a custom query.

        This is useful for ad-hoc queries or integration testing.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of result rows as dictionaries
        """
        return await self._async_query(query, params)

    async def get_table_stats(self, table: str) -> Dict[str, Any]:
        """Get statistics for a table."""
        query = f"""
            SELECT
                COUNT(*) as row_count,
                MAX("{self.primary_key_column}") as max_id
            FROM {self.database}.{self.schema}.{table}
        """

        try:
            rows = await self._async_query(query)
            if rows:
                return rows[0]
        except Exception as e:
            logger.warning(f"Failed to get stats for {table}: {e}")

        return {"row_count": 0, "max_id": None}

    async def time_travel_query(
        self,
        table: str,
        timestamp: datetime,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query table data as it existed at a specific timestamp.

        Uses Snowflake Time Travel feature.

        Args:
            table: Table name
            timestamp: Point in time to query
            limit: Maximum rows to return

        Returns:
            Rows as they existed at the specified timestamp
        """
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
            SELECT *
            FROM {self.database}.{self.schema}.{table}
            AT(TIMESTAMP => '{ts_str}'::TIMESTAMP_LTZ)
            LIMIT {limit}
        """

        return await self._async_query(query)

    async def close(self):
        """Close connection and executor."""
        if self._connection:
            self._connection.close()
            self._connection = None

        self._executor.shutdown(wait=False)

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """
        Handle webhook for Snowflake notifications.

        Snowflake can send notifications via:
        - Snowpipe (data loading)
        - Task notifications
        - External functions

        Args:
            payload: Webhook payload

        Returns:
            True if handled successfully
        """
        table = payload.get("table")
        operation = payload.get("operation")

        if table and operation:
            logger.info(f"[{self.name}] Webhook: {operation} on {table}")
            asyncio.create_task(self.sync(max_items=10))
            return True

        return False


__all__ = ["SnowflakeConnector"]
