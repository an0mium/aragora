"""
SQL Server Enterprise Connector.

Features:
- Incremental sync using transaction timestamps or custom columns
- Change Data Capture (CDC) for real-time change detection
- Change Tracking for lightweight change detection
- Table/view selection with schema support
- Connection pooling for performance
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.connectors.enterprise.database.cdc import (
    CDCSourceType,
    CDCStreamManager,
    ChangeEvent,
    ChangeEventHandler,
    ChangeOperation,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

# Default columns to use for change tracking
DEFAULT_TIMESTAMP_COLUMNS = ["updated_at", "modified_at", "last_modified", "timestamp"]


class SQLServerConnector(EnterpriseConnector):
    """
    SQL Server connector for enterprise data sync.

    Supports:
    - Incremental sync using timestamp columns
    - Real-time updates via SQL Server CDC (Change Data Capture)
    - Change Tracking for simpler scenarios
    - Schema-qualified table access
    - Connection pooling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1433,
        database: str = "master",
        schema: str = "dbo",
        tables: Optional[List[str]] = None,
        timestamp_column: Optional[str] = None,
        primary_key_column: str = "id",
        content_columns: Optional[List[str]] = None,
        use_cdc: bool = False,  # Use SQL Server CDC
        use_change_tracking: bool = False,  # Use Change Tracking
        poll_interval_seconds: int = 5,
        pool_size: int = 5,
        **kwargs,
    ):
        connector_id = f"sqlserver_{host}_{database}_{schema}"
        super().__init__(connector_id=connector_id, **kwargs)

        self.host = host
        self.port = port
        self.database = database
        self.schema = schema
        self.tables = tables or []
        self.timestamp_column = timestamp_column
        self.primary_key_column = primary_key_column
        self.content_columns = content_columns
        self.use_cdc = use_cdc
        self.use_change_tracking = use_change_tracking
        self.poll_interval_seconds = poll_interval_seconds
        self.pool_size = pool_size

        self._pool = None
        self._cdc_task = None
        self._last_lsn: Optional[bytes] = None  # Last processed LSN for CDC

        # CDC support
        self._cdc_manager: Optional[CDCStreamManager] = None
        self._change_handlers: List[ChangeEventHandler] = []

    @property
    def cdc_manager(self) -> CDCStreamManager:
        """Get or create the CDC stream manager."""
        if self._cdc_manager is None:
            from aragora.connectors.enterprise.database.cdc import CompositeHandler

            handler = CompositeHandler(self._change_handlers)
            self._cdc_manager = CDCStreamManager(
                connector_id=self.connector_id,
                source_type=CDCSourceType.SQLSERVER,
                handler=handler,
            )
        return self._cdc_manager

    def add_change_handler(self, handler: ChangeEventHandler) -> None:
        """Add a handler for change events."""
        self._change_handlers.append(handler)
        self._cdc_manager = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def name(self) -> str:
        return f"SQL Server ({self.database}.{self.schema})"

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is not None:
            return self._pool

        try:
            import aioodbc

            # Get credentials
            username = await self.credentials.get_credential("SQLSERVER_USER") or "sa"
            password = await self.credentials.get_credential("SQLSERVER_PASSWORD") or ""

            # Build connection string
            dsn = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.host},{self.port};"
                f"DATABASE={self.database};"
                f"UID={username};"
                f"PWD={password}"
            )

            self._pool = await aioodbc.create_pool(
                dsn=dsn,
                minsize=1,
                maxsize=self.pool_size,
            )
            return self._pool

        except ImportError:
            logger.error("aioodbc not installed. Run: pip install aioodbc")
            raise

    async def _discover_tables(self) -> List[str]:
        """Discover tables in the schema."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT TABLE_NAME
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = ?
                    AND TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_NAME
                    """,
                    self.schema,
                )
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

    async def _get_table_columns(self, table: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                    """,
                    self.schema,
                    table,
                )
                rows = await cursor.fetchall()
                return [
                    {
                        "column_name": row[0],
                        "data_type": row[1],
                        "is_nullable": row[2],
                    }
                    for row in rows
                ]

    def _find_timestamp_column(self, columns: List[Dict[str, Any]]) -> Optional[str]:
        """Find a suitable timestamp column for incremental sync."""
        if self.timestamp_column:
            return self.timestamp_column

        column_names = {col["column_name"].lower() for col in columns}
        for candidate in DEFAULT_TIMESTAMP_COLUMNS:
            if candidate in column_names:
                return candidate
        return None

    def _row_to_content(self, row: Dict[str, Any], columns: Optional[List[str]] = None) -> str:
        """Convert a row to text content for indexing."""
        if columns:
            filtered = {k: v for k, v in row.items() if k in columns}
        else:
            filtered = row

        parts = []
        for key, value in filtered.items():
            if value is not None:
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, (dict, list)):
                    value = json.dumps(value, default=str)
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    async def sync_items(self, state: Optional[SyncState] = None) -> AsyncIterator[SyncItem]:
        """
        Sync items from SQL Server tables.

        Supports incremental sync using timestamp columns.
        """
        pool = await self._get_pool()

        tables = self.tables or await self._discover_tables()
        last_sync = state.last_sync if state else None

        for table in tables:
            columns = await self._get_table_columns(table)
            ts_column = self._find_timestamp_column(columns)

            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    if ts_column and last_sync:
                        query = f"""
                            SELECT * FROM [{self.schema}].[{table}]
                            WHERE [{ts_column}] > ?
                            ORDER BY [{ts_column}]
                        """
                        await cursor.execute(query, last_sync)
                    else:
                        query = f"SELECT * FROM [{self.schema}].[{table}]"
                        await cursor.execute(query)

                    # Get column names
                    col_names = [desc[0] for desc in cursor.description]

                    async for row in cursor:
                        row_dict = dict(zip(col_names, row))
                        pk_value = row_dict.get(self.primary_key_column)
                        content = self._row_to_content(row_dict, self.content_columns)

                        item_id = hashlib.sha256(
                            f"{self.database}.{self.schema}.{table}.{pk_value}".encode()
                        ).hexdigest()[:16]

                        yield SyncItem(
                            id=item_id,
                            content=content,
                            metadata={
                                "source": "sqlserver",
                                "database": self.database,
                                "schema": self.schema,
                                "table": table,
                                "primary_key": pk_value,
                                "row_data": row_dict,
                            },
                            timestamp=row_dict.get(ts_column) if ts_column else None,
                        )

    async def _check_cdc_enabled(self, table: str) -> bool:
        """Check if CDC is enabled for a table."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM cdc.change_tables
                    WHERE source_object_id = OBJECT_ID(?)
                    """,
                    f"{self.schema}.{table}",
                )
                row = await cursor.fetchone()
                return row[0] > 0 if row else False

    async def start_cdc_polling(self) -> None:
        """
        Start CDC polling for change detection.

        Requires CDC to be enabled on the database and tables:
        EXEC sys.sp_cdc_enable_db;
        EXEC sys.sp_cdc_enable_table @source_schema = 'dbo', @source_name = 'mytable', ...;
        """
        if not self.use_cdc:
            logger.warning("CDC not enabled for this connector")
            return

        logger.info(f"[SQL Server CDC] Starting CDC polling for {self.database}")
        self._cdc_task = asyncio.create_task(self._poll_cdc_changes())

    async def _poll_cdc_changes(self) -> None:
        """Poll CDC tables for changes."""
        pool = await self._get_pool()
        tables = self.tables or await self._discover_tables()

        # Filter to CDC-enabled tables
        cdc_tables = []
        for table in tables:
            if await self._check_cdc_enabled(table):
                cdc_tables.append(table)
            else:
                logger.debug(f"[SQL Server CDC] Table {table} not CDC-enabled, skipping")

        if not cdc_tables:
            logger.warning("[SQL Server CDC] No CDC-enabled tables found")
            return

        try:
            while True:
                for table in cdc_tables:
                    await self._process_table_cdc_changes(pool, table)

                await asyncio.sleep(self.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.info("[SQL Server CDC] Polling cancelled")
        except Exception as e:
            logger.error(f"[SQL Server CDC] Polling error: {e}")
            raise

    async def _process_table_cdc_changes(self, pool, table: str) -> None:
        """Process CDC changes for a single table."""
        # Get the CDC capture instance name
        capture_instance = f"{self.schema}_{table}"

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get LSN range
                await cursor.execute("SELECT sys.fn_cdc_get_min_lsn(?)", capture_instance)
                min_lsn_row = await cursor.fetchone()
                min_lsn = min_lsn_row[0] if min_lsn_row else None

                await cursor.execute("SELECT sys.fn_cdc_get_max_lsn()")
                max_lsn_row = await cursor.fetchone()
                max_lsn = max_lsn_row[0] if max_lsn_row else None

                if not min_lsn or not max_lsn:
                    return

                # Use last processed LSN or min LSN
                from_lsn = self._last_lsn if self._last_lsn else min_lsn

                # Query CDC changes
                cdc_query = f"""
                    SELECT *
                    FROM cdc.fn_cdc_get_all_changes_{capture_instance}(?, ?, 'all')
                    ORDER BY __$start_lsn
                """

                try:
                    await cursor.execute(cdc_query, from_lsn, max_lsn)
                except Exception as e:
                    logger.debug(f"[SQL Server CDC] No changes or error for {table}: {e}")
                    return

                col_names = [desc[0] for desc in cursor.description]

                async for row in cursor:
                    row_dict = dict(zip(col_names, row))

                    # Map CDC operation codes
                    operation_code = row_dict.get("__$operation")
                    if operation_code == 1:
                        operation = ChangeOperation.DELETE
                    elif operation_code == 2:
                        operation = ChangeOperation.INSERT
                    elif operation_code in (3, 4):  # Before/after update
                        operation = ChangeOperation.UPDATE
                    else:
                        continue

                    # Remove CDC metadata columns from data
                    data = {k: v for k, v in row_dict.items() if not k.startswith("__$")}

                    event = ChangeEvent(
                        id="",
                        source_type=CDCSourceType.SQLSERVER,
                        connector_id=self.connector_id,
                        operation=operation,
                        timestamp=datetime.now(timezone.utc),
                        database=self.database,
                        schema=self.schema,
                        table=table,
                        data=data if operation != ChangeOperation.DELETE else None,
                        old_data=data if operation == ChangeOperation.DELETE else None,
                        primary_key={"id": data.get("id")},
                        resume_token=row_dict.get("__$start_lsn", b"").hex(),
                    )

                    await self.cdc_manager.process_event(event)

                # Update last processed LSN
                self._last_lsn = max_lsn

    async def start_change_tracking_polling(self) -> None:
        """
        Start Change Tracking polling for change detection.

        Lighter weight than CDC, requires:
        ALTER DATABASE [db] SET CHANGE_TRACKING = ON;
        ALTER TABLE [table] ENABLE CHANGE_TRACKING;
        """
        if not self.use_change_tracking:
            logger.warning("Change Tracking not enabled for this connector")
            return

        logger.info(f"[SQL Server CT] Starting Change Tracking polling for {self.database}")
        self._cdc_task = asyncio.create_task(self._poll_change_tracking())

    async def _poll_change_tracking(self) -> None:
        """Poll Change Tracking for changes."""
        pool = await self._get_pool()
        tables = self.tables or await self._discover_tables()

        try:
            while True:
                for table in tables:
                    await self._process_table_ct_changes(pool, table)

                await asyncio.sleep(self.poll_interval_seconds)

        except asyncio.CancelledError:
            logger.info("[SQL Server CT] Polling cancelled")
        except Exception as e:
            logger.error(f"[SQL Server CT] Polling error: {e}")
            raise

    async def _process_table_ct_changes(self, pool, table: str) -> None:
        """Process Change Tracking changes for a single table."""
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Get minimum valid version
                await cursor.execute(
                    "SELECT CHANGE_TRACKING_MIN_VALID_VERSION(OBJECT_ID(?))",
                    f"{self.schema}.{table}",
                )
                min_version_row = await cursor.fetchone()
                min_version = min_version_row[0] if min_version_row else None

                if min_version is None:
                    logger.debug(f"[SQL Server CT] Change Tracking not enabled for {table}")
                    return

                # Get current version
                await cursor.execute("SELECT CHANGE_TRACKING_CURRENT_VERSION()")
                current_version_row = await cursor.fetchone()
                current_version = current_version_row[0] if current_version_row else None

                if not current_version:
                    return

                # Query changes since last version
                last_version = getattr(self, f"_ct_version_{table}", min_version)

                ct_query = f"""
                    SELECT ct.SYS_CHANGE_OPERATION, ct.{self.primary_key_column}, t.*
                    FROM CHANGETABLE(CHANGES [{self.schema}].[{table}], ?) AS ct
                    LEFT JOIN [{self.schema}].[{table}] t
                        ON ct.{self.primary_key_column} = t.{self.primary_key_column}
                """

                try:
                    await cursor.execute(ct_query, last_version)
                except Exception as e:
                    logger.debug(f"[SQL Server CT] No changes or error for {table}: {e}")
                    return

                col_names = [desc[0] for desc in cursor.description]

                async for row in cursor:
                    row_dict = dict(zip(col_names, row))

                    # Map Change Tracking operation codes
                    operation_code = row_dict.get("SYS_CHANGE_OPERATION")
                    if operation_code == "D":
                        operation = ChangeOperation.DELETE
                    elif operation_code == "I":
                        operation = ChangeOperation.INSERT
                    elif operation_code == "U":
                        operation = ChangeOperation.UPDATE
                    else:
                        continue

                    # Remove CT metadata
                    data = {k: v for k, v in row_dict.items() if not k.startswith("SYS_CHANGE_")}

                    event = ChangeEvent(
                        id="",
                        source_type=CDCSourceType.SQLSERVER,
                        connector_id=self.connector_id,
                        operation=operation,
                        timestamp=datetime.now(timezone.utc),
                        database=self.database,
                        schema=self.schema,
                        table=table,
                        data=data if operation != ChangeOperation.DELETE else None,
                        primary_key={
                            self.primary_key_column: row_dict.get(self.primary_key_column)
                        },
                    )

                    await self.cdc_manager.process_event(event)

                # Update last processed version
                setattr(self, f"_ct_version_{table}", current_version)

    async def stop_cdc_polling(self) -> None:
        """Stop CDC/Change Tracking polling."""
        if self._cdc_task:
            self._cdc_task.cancel()
            try:
                await self._cdc_task
            except asyncio.CancelledError:
                pass

        logger.info(f"[SQL Server CDC/CT] Stopped polling for {self.database}")

    async def close(self) -> None:
        """Close connections and cleanup resources."""
        await self.stop_cdc_polling()

        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def health_check(self) -> Dict[str, Any]:
        """Check SQL Server connection health."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    await cursor.fetchone()

            return {
                "healthy": True,
                "database": self.database,
                "schema": self.schema,
                "host": self.host,
                "cdc_enabled": self.use_cdc,
                "change_tracking_enabled": self.use_change_tracking,
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "database": self.database,
            }
