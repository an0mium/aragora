"""
PostgreSQL Enterprise Connector.

Features:
- Incremental sync using transaction timestamps or custom columns
- LISTEN/NOTIFY for real-time change detection
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
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

# Default columns to use for change tracking
DEFAULT_TIMESTAMP_COLUMNS = ["updated_at", "modified_at", "last_modified", "timestamp"]


class PostgreSQLConnector(EnterpriseConnector):
    """
    PostgreSQL connector for enterprise data sync.

    Supports:
    - Incremental sync using timestamp columns
    - Real-time updates via LISTEN/NOTIFY
    - Schema-qualified table access
    - Connection pooling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        schema: str = "public",
        tables: Optional[List[str]] = None,
        timestamp_column: Optional[str] = None,
        primary_key_column: str = "id",
        content_columns: Optional[List[str]] = None,
        notify_channel: Optional[str] = None,
        pool_size: int = 5,
        **kwargs,
    ):
        connector_id = f"postgres_{host}_{database}_{schema}"
        super().__init__(connector_id=connector_id, **kwargs)

        self.host = host
        self.port = port
        self.database = database
        self.schema = schema
        self.tables = tables or []
        self.timestamp_column = timestamp_column
        self.primary_key_column = primary_key_column
        self.content_columns = content_columns
        self.notify_channel = notify_channel
        self.pool_size = pool_size

        self._pool = None
        self._listener_task = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DATABASE

    @property
    def name(self) -> str:
        return f"PostgreSQL ({self.database}.{self.schema})"

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is not None:
            return self._pool

        try:
            import asyncpg

            # Get credentials
            username = await self.credentials.get_credential("POSTGRES_USER") or "postgres"
            password = await self.credentials.get_credential("POSTGRES_PASSWORD") or ""

            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=username,
                password=password,
                min_size=1,
                max_size=self.pool_size,
            )
            return self._pool

        except ImportError:
            logger.error("asyncpg not installed. Run: pip install asyncpg")
            raise

    async def _discover_tables(self) -> List[str]:
        """Discover tables in the schema."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = $1
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """,
                self.schema,
            )
            return [row["table_name"] for row in rows]

    async def _get_table_columns(self, table: str) -> List[Dict[str, Any]]:
        """Get column information for a table."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2
                ORDER BY ordinal_position
                """,
                self.schema,
                table,
            )
            return [dict(row) for row in rows]

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

        if any(t in table_lower for t in ["user", "account", "profile", "auth"]):
            return "operational/users"
        elif any(t in table_lower for t in ["order", "invoice", "payment", "transaction"]):
            return "financial/transactions"
        elif any(t in table_lower for t in ["product", "inventory", "catalog"]):
            return "operational/products"
        elif any(t in table_lower for t in ["log", "audit", "event"]):
            return "operational/logs"
        elif any(t in table_lower for t in ["config", "setting", "preference"]):
            return "technical/configuration"
        elif any(t in table_lower for t in ["document", "file", "attachment"]):
            return "general/documents"

        return "general/database"

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield items to sync from PostgreSQL tables.

        Uses timestamp columns for incremental sync when available.
        """
        pool = await self._get_pool()

        # Get tables to sync
        tables = self.tables or await self._discover_tables()
        state.items_total = len(tables)

        for table in tables:
            try:
                columns = await self._get_table_columns(table)
                ts_column = self._find_timestamp_column(columns)

                async with pool.acquire() as conn:
                    # Build query
                    qualified_table = f'"{self.schema}"."{table}"'

                    if ts_column and state.last_item_timestamp:
                        # Incremental sync
                        query = f"""
                            SELECT * FROM {qualified_table}
                            WHERE "{ts_column}" > $1
                            ORDER BY "{ts_column}" ASC
                            LIMIT $2
                        """
                        rows = await conn.fetch(query, state.last_item_timestamp, batch_size)
                    else:
                        # Full sync with cursor-based pagination
                        if state.cursor and state.cursor.startswith(f"{table}:"):
                            last_id = state.cursor.split(":", 1)[1]
                            query = f"""
                                SELECT * FROM {qualified_table}
                                WHERE "{self.primary_key_column}" > $1
                                ORDER BY "{self.primary_key_column}" ASC
                                LIMIT $2
                            """
                            rows = await conn.fetch(query, last_id, batch_size)
                        else:
                            query = f"""
                                SELECT * FROM {qualified_table}
                                ORDER BY "{self.primary_key_column}" ASC
                                LIMIT $1
                            """
                            rows = await conn.fetch(query, batch_size)

                    for row in rows:
                        row_dict = dict(row)
                        pk_value = row_dict.get(self.primary_key_column, "")

                        # Generate content
                        content = self._row_to_content(row_dict, self.content_columns)

                        # Get timestamp if available
                        updated_at = datetime.now(timezone.utc)
                        if ts_column and row_dict.get(ts_column):
                            ts_value = row_dict[ts_column]
                            if isinstance(ts_value, datetime):
                                updated_at = (
                                    ts_value.replace(tzinfo=timezone.utc)
                                    if ts_value.tzinfo is None
                                    else ts_value
                                )

                        # Create sync item
                        item_id = f"pg:{self.database}:{table}:{hashlib.sha256(str(pk_value).encode()).hexdigest()[:12]}"

                        yield SyncItem(
                            id=item_id,
                            content=content[:100000],
                            source_type="database",
                            source_id=f"postgresql://{self.host}:{self.port}/{self.database}/{self.schema}/{table}/{pk_value}",
                            title=f"{table} #{pk_value}",
                            url=f"postgresql://{self.host}/{self.database}/{table}?id={pk_value}",
                            updated_at=updated_at,
                            domain=self._infer_domain(table),
                            confidence=0.85,
                            metadata={
                                "database": self.database,
                                "schema": self.schema,
                                "table": table,
                                "primary_key": str(pk_value),
                                "columns": list(row_dict.keys()),
                            },
                        )

                        # Update cursor
                        state.cursor = f"{table}:{pk_value}"

            except Exception as e:
                logger.warning(f"Failed to sync table {table}: {e}")
                state.errors.append(f"{table}: {str(e)}")
                continue

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list:
        """
        Search across indexed tables using full-text search.

        Requires tables to have tsvector columns for best results.
        """
        pool = await self._get_pool()
        results = []

        tables = self.tables or await self._discover_tables()

        async with pool.acquire() as conn:
            for table in tables[:5]:  # Limit to first 5 tables
                try:
                    qualified_table = f'"{self.schema}"."{table}"'

                    # Try full-text search if available
                    fts_query = f"""
                        SELECT *, ts_rank(to_tsvector('english', coalesce(content::text, '')), plainto_tsquery('english', $1)) as rank
                        FROM {qualified_table}
                        WHERE to_tsvector('english', coalesce(content::text, '')) @@ plainto_tsquery('english', $1)
                        ORDER BY rank DESC
                        LIMIT $2
                    """

                    try:
                        rows = await conn.fetch(fts_query, query, limit)
                        for row in rows:
                            results.append(
                                {
                                    "table": table,
                                    "data": dict(row),
                                    "rank": row.get("rank", 0),
                                }
                            )
                    except Exception as e:
                        # Fallback to ILIKE search (FTS may not be configured)
                        logger.debug(f"FTS query failed on {table}, falling back to ILIKE: {e}")
                        columns = await self._get_table_columns(table)
                        text_columns = [
                            c["column_name"]
                            for c in columns
                            if "char" in c["data_type"] or "text" in c["data_type"]
                        ]

                        if text_columns:
                            conditions = " OR ".join(
                                [f'"{col}"::text ILIKE $1' for col in text_columns[:3]]
                            )
                            fallback_query = f"""
                                SELECT * FROM {qualified_table}
                                WHERE {conditions}
                                LIMIT $2
                            """
                            rows = await conn.fetch(fallback_query, f"%{query}%", limit)
                            for row in rows:
                                results.append(
                                    {
                                        "table": table,
                                        "data": dict(row),
                                        "rank": 0.5,
                                    }
                                )

                except Exception as e:
                    logger.debug(f"Search failed on {table}: {e}")
                    continue

        return sorted(results, key=lambda x: x.get("rank", 0), reverse=True)[:limit]

    async def fetch(self, evidence_id: str):
        """Fetch a specific row by evidence ID."""
        if not evidence_id.startswith("pg:"):
            return None

        parts = evidence_id.split(":")
        if len(parts) < 4:
            return None

        database, _table, _pk_hash = parts[1], parts[2], parts[3]

        if database != self.database:
            return None

        # We can't reverse the hash, so this is limited
        logger.debug(f"[{self.name}] Fetch not implemented for hash-based IDs")
        return None

    async def start_listener(self):
        """Start LISTEN/NOTIFY listener for real-time updates."""
        if not self.notify_channel:
            return

        pool = await self._get_pool()

        async def listener_loop():
            async with pool.acquire() as conn:
                await conn.add_listener(self.notify_channel, self._handle_notification)
                logger.info(f"[{self.name}] Listening on channel: {self.notify_channel}")

                # Keep connection alive
                while True:
                    await asyncio.sleep(60)

        self._listener_task = asyncio.create_task(listener_loop())

    async def _handle_notification(self, connection, pid, channel, payload):
        """Handle NOTIFY message."""
        try:
            data = json.loads(payload) if payload else {}
            table = data.get("table")
            operation = data.get("operation")

            logger.info(f"[{self.name}] Notification: {operation} on {table}")

            # Trigger incremental sync
            asyncio.create_task(self.sync(max_items=10))

        except Exception as e:
            logger.warning(f"[{self.name}] Notification handler error: {e}")

    async def stop_listener(self):
        """Stop the LISTEN/NOTIFY listener."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                logger.debug("[%s] Listener task cancelled during stop", self.name)
            self._listener_task = None

    async def close(self):
        """Close connection pool."""
        await self.stop_listener()
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle webhook for database changes (e.g., from triggers)."""
        table = payload.get("table")
        operation = payload.get("operation")

        if table and operation:
            logger.info(f"[{self.name}] Webhook: {operation} on {table}")
            asyncio.create_task(self.sync(max_items=10))
            return True

        return False
