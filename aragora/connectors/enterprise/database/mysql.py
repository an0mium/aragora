"""
MySQL Enterprise Connector.

Features:
- Incremental sync using transaction timestamps or custom columns
- Binary log (binlog) CDC for real-time change detection
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

# Optional dependency for async MySQL
try:
    import aiomysql
except ImportError:
    aiomysql = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default columns to use for change tracking
DEFAULT_TIMESTAMP_COLUMNS = ["updated_at", "modified_at", "last_modified", "timestamp"]


class MySQLConnector(EnterpriseConnector):
    """
    MySQL connector for enterprise data sync.

    Supports:
    - Incremental sync using timestamp columns
    - Real-time updates via binlog CDC (requires mysql-replication)
    - Schema-qualified table access
    - Connection pooling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "mysql",
        tables: Optional[List[str]] = None,
        timestamp_column: Optional[str] = None,
        primary_key_column: str = "id",
        content_columns: Optional[List[str]] = None,
        enable_binlog_cdc: bool = False,
        server_id: int = 100,  # For binlog replication
        pool_size: int = 5,
        **kwargs,
    ):
        connector_id = f"mysql_{host}_{database}"
        super().__init__(connector_id=connector_id, **kwargs)

        self.host = host
        self.port = port
        self.database = database
        self.tables = tables or []
        self.timestamp_column = timestamp_column
        self.primary_key_column = primary_key_column
        self.content_columns = content_columns
        self.enable_binlog_cdc = enable_binlog_cdc
        self.server_id = server_id
        self.pool_size = pool_size

        self._pool = None
        self._binlog_stream = None
        self._cdc_task = None

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
                source_type=CDCSourceType.MYSQL,
                handler=handler,
            )
        return self._cdc_manager

    def add_change_handler(self, handler: ChangeEventHandler) -> None:
        """Add a handler for change events."""
        self._change_handlers.append(handler)
        # Reset CDC manager to pick up new handler
        self._cdc_manager = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def name(self) -> str:
        return f"MySQL ({self.database})"

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is not None:
            return self._pool

        try:
            import aiomysql

            # Get credentials
            username = await self.credentials.get_credential("MYSQL_USER") or "root"
            password = await self.credentials.get_credential("MYSQL_PASSWORD") or ""

            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                db=self.database,
                user=username,
                password=password,
                minsize=1,
                maxsize=self.pool_size,
                autocommit=True,
            )
            return self._pool

        except ImportError:
            logger.error("aiomysql not installed. Run: pip install aiomysql")
            raise

    async def _discover_tables(self) -> List[str]:
        """Discover tables in the database."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """,
                    (self.database,),
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
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (self.database, table),
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
        Sync items from MySQL tables.

        Supports incremental sync using timestamp columns.
        """
        pool = await self._get_pool()

        tables = self.tables or await self._discover_tables()
        last_sync = state.last_sync if state else None

        for table in tables:
            columns = await self._get_table_columns(table)
            ts_column = self._find_timestamp_column(columns)

            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    if ts_column and last_sync:
                        query = f"""
                            SELECT * FROM `{table}`
                            WHERE `{ts_column}` > %s
                            ORDER BY `{ts_column}`
                        """
                        await cursor.execute(query, (last_sync,))
                    else:
                        query = f"SELECT * FROM `{table}`"
                        await cursor.execute(query)

                    async for row in cursor:
                        row_dict = dict(row)
                        pk_value = row_dict.get(self.primary_key_column)
                        content = self._row_to_content(row_dict, self.content_columns)

                        item_id = hashlib.sha256(
                            f"{self.database}.{table}.{pk_value}".encode()
                        ).hexdigest()[:16]

                        yield SyncItem(
                            id=item_id,
                            content=content,
                            metadata={
                                "source": "mysql",
                                "database": self.database,
                                "table": table,
                                "primary_key": pk_value,
                                "row_data": row_dict,
                            },
                            timestamp=row_dict.get(ts_column) if ts_column else None,
                        )

    async def start_binlog_cdc(self) -> None:
        """
        Start CDC using MySQL binary log.

        Requires mysql-replication package and binlog enabled on server.
        """
        if not self.enable_binlog_cdc:
            logger.warning("Binlog CDC not enabled for this connector")
            return

        try:
            from pymysqlreplication import BinLogStreamReader
            from pymysqlreplication.row_event import (
                DeleteRowsEvent,
                UpdateRowsEvent,
                WriteRowsEvent,
            )

            username = await self.credentials.get_credential("MYSQL_USER") or "root"
            password = await self.credentials.get_credential("MYSQL_PASSWORD") or ""

            mysql_settings = {
                "host": self.host,
                "port": self.port,
                "user": username,
                "passwd": password,
            }

            self._binlog_stream = BinLogStreamReader(
                connection_settings=mysql_settings,
                server_id=self.server_id,
                blocking=True,
                only_events=[WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent],
                only_schemas=[self.database] if self.database else None,
                only_tables=self.tables if self.tables else None,
                resume_stream=True,
            )

            logger.info(f"[MySQL CDC] Started binlog stream for {self.database}")

            self._cdc_task = asyncio.create_task(self._process_binlog_events())

        except ImportError:
            logger.error("mysql-replication not installed. Run: pip install mysql-replication")
            raise

    async def _process_binlog_events(self) -> None:
        """Process binlog events and emit ChangeEvents."""
        from pymysqlreplication.row_event import (
            DeleteRowsEvent,
            UpdateRowsEvent,
            WriteRowsEvent,
        )

        try:
            for binlog_event in self._binlog_stream:
                # Map binlog event to ChangeOperation
                if isinstance(binlog_event, WriteRowsEvent):
                    operation = ChangeOperation.INSERT
                elif isinstance(binlog_event, UpdateRowsEvent):
                    operation = ChangeOperation.UPDATE
                elif isinstance(binlog_event, DeleteRowsEvent):
                    operation = ChangeOperation.DELETE
                else:
                    continue

                for row in binlog_event.rows:
                    # Extract data based on event type
                    if operation == ChangeOperation.UPDATE:
                        data = row.get("after_values", {})
                        old_data = row.get("before_values", {})
                    elif operation == ChangeOperation.DELETE:
                        data = None
                        old_data = row.get("values", {})
                    else:  # INSERT
                        data = row.get("values", {})
                        old_data = None

                    # Create ChangeEvent
                    event = ChangeEvent(
                        id="",
                        source_type=CDCSourceType.MYSQL,
                        connector_id=self.connector_id,
                        operation=operation,
                        timestamp=datetime.now(timezone.utc),
                        database=binlog_event.schema,
                        table=binlog_event.table,
                        data=data,
                        old_data=old_data,
                        primary_key={"id": data.get("id") if data else old_data.get("id")},
                    )

                    # Process through CDC manager
                    await self.cdc_manager.process_event(event)

        except Exception as e:
            logger.error(f"[MySQL CDC] Binlog processing error: {e}")
            raise
        finally:
            if self._binlog_stream:
                self._binlog_stream.close()

    async def stop_binlog_cdc(self) -> None:
        """Stop binlog CDC stream."""
        if self._cdc_task:
            self._cdc_task.cancel()
            try:
                await self._cdc_task
            except asyncio.CancelledError:
                pass

        if self._binlog_stream:
            self._binlog_stream.close()
            self._binlog_stream = None

        logger.info(f"[MySQL CDC] Stopped binlog stream for {self.database}")

    async def close(self) -> None:
        """Close connections and cleanup resources."""
        await self.stop_binlog_cdc()

        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def health_check(self) -> Dict[str, Any]:
        """Check MySQL connection health."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    await cursor.fetchone()

            return {
                "healthy": True,
                "database": self.database,
                "host": self.host,
                "binlog_cdc_enabled": self.enable_binlog_cdc,
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "database": self.database,
            }
