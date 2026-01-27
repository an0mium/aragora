"""
Protocol Message Store for Aragora Debates.

Provides SQLite-backed persistence and querying for protocol messages.
Enables audit trails, debugging, and debate replay functionality.

Inspired by gastown's beads storage pattern for persistent work state.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional

from .messages import ProtocolMessage, ProtocolMessageType

logger = logging.getLogger(__name__)

# Module-level singleton
_protocol_store: Optional["ProtocolMessageStore"] = None
_store_lock = threading.Lock()


def get_protocol_store(db_path: Optional[str] = None) -> "ProtocolMessageStore":
    """Get or create the global protocol message store."""
    global _protocol_store
    with _store_lock:
        if _protocol_store is None:
            _protocol_store = ProtocolMessageStore(db_path)
        return _protocol_store


@dataclass
class QueryFilters:
    """Filters for querying protocol messages."""

    debate_id: Optional[str] = None
    agent_id: Optional[str] = None
    message_type: Optional[ProtocolMessageType] = None
    message_types: Optional[List[ProtocolMessageType]] = None
    round_number: Optional[int] = None
    min_round: Optional[int] = None
    max_round: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    correlation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    limit: int = 1000
    offset: int = 0
    order_by: str = "timestamp"
    order_desc: bool = False


class ProtocolMessageStore:
    """
    SQLite-backed store for protocol messages.

    Features:
    - ACID-compliant persistence
    - Indexed queries by debate, agent, type, round
    - Time-range queries for audit trails
    - JSONL export for replay
    - Thread-safe operations
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the protocol message store.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self._local = threading.local()
        self._init_schema()
        logger.info(f"ProtocolMessageStore initialized: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS protocol_messages (
                    message_id TEXT PRIMARY KEY,
                    message_type TEXT NOT NULL,
                    debate_id TEXT NOT NULL,
                    agent_id TEXT,
                    round_number INTEGER,
                    timestamp TEXT NOT NULL,
                    correlation_id TEXT,
                    parent_message_id TEXT,
                    payload TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_protocol_debate
                ON protocol_messages(debate_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_protocol_agent
                ON protocol_messages(agent_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_protocol_type
                ON protocol_messages(message_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_protocol_round
                ON protocol_messages(debate_id, round_number)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_protocol_timestamp
                ON protocol_messages(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_protocol_correlation
                ON protocol_messages(correlation_id)
            """)

    async def record(self, message: ProtocolMessage) -> str:
        """
        Record a protocol message.

        Args:
            message: The protocol message to record.

        Returns:
            The message_id of the recorded message.
        """
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO protocol_messages (
                    message_id, message_type, debate_id, agent_id, round_number,
                    timestamp, correlation_id, parent_message_id, payload, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message.message_id,
                    message.message_type.value,
                    message.debate_id,
                    message.agent_id,
                    message.round_number,
                    message.timestamp.isoformat(),
                    message.correlation_id,
                    message.parent_message_id,
                    (
                        json.dumps(message.payload.to_dict())
                        if message.payload and hasattr(message.payload, "to_dict")
                        else json.dumps(message.payload)
                        if message.payload
                        else None
                    ),
                    json.dumps(message.metadata) if message.metadata else None,
                ),
            )

        logger.debug(
            f"Recorded protocol message: {message.message_type.value} "
            f"for debate {message.debate_id[:8]}..."
        )
        return message.message_id

    def record_sync(self, message: ProtocolMessage) -> str:
        """Synchronous version of record for non-async contexts."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO protocol_messages (
                    message_id, message_type, debate_id, agent_id, round_number,
                    timestamp, correlation_id, parent_message_id, payload, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    message.message_id,
                    message.message_type.value,
                    message.debate_id,
                    message.agent_id,
                    message.round_number,
                    message.timestamp.isoformat(),
                    message.correlation_id,
                    message.parent_message_id,
                    (
                        json.dumps(message.payload.to_dict())
                        if message.payload and hasattr(message.payload, "to_dict")
                        else json.dumps(message.payload)
                        if message.payload
                        else None
                    ),
                    json.dumps(message.metadata) if message.metadata else None,
                ),
            )
        return message.message_id

    async def query(self, filters: Optional[QueryFilters] = None) -> List[ProtocolMessage]:
        """
        Query protocol messages with filters.

        Args:
            filters: Query filters. If None, returns all messages.

        Returns:
            List of matching protocol messages.
        """
        filters = filters or QueryFilters()
        conditions: List[str] = []
        params: List[Any] = []

        if filters.debate_id:
            conditions.append("debate_id = ?")
            params.append(filters.debate_id)

        if filters.agent_id:
            conditions.append("agent_id = ?")
            params.append(filters.agent_id)

        if filters.message_type:
            conditions.append("message_type = ?")
            params.append(filters.message_type.value)

        if filters.message_types:
            placeholders = ",".join("?" for _ in filters.message_types)
            conditions.append(f"message_type IN ({placeholders})")
            params.extend(mt.value for mt in filters.message_types)

        if filters.round_number is not None:
            conditions.append("round_number = ?")
            params.append(filters.round_number)

        if filters.min_round is not None:
            conditions.append("round_number >= ?")
            params.append(filters.min_round)

        if filters.max_round is not None:
            conditions.append("round_number <= ?")
            params.append(filters.max_round)

        if filters.start_time:
            conditions.append("timestamp >= ?")
            params.append(filters.start_time.isoformat())

        if filters.end_time:
            conditions.append("timestamp <= ?")
            params.append(filters.end_time.isoformat())

        if filters.correlation_id:
            conditions.append("correlation_id = ?")
            params.append(filters.correlation_id)

        if filters.parent_message_id:
            conditions.append("parent_message_id = ?")
            params.append(filters.parent_message_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order_direction = "DESC" if filters.order_desc else "ASC"

        query = f"""
            SELECT * FROM protocol_messages
            WHERE {where_clause}
            ORDER BY {filters.order_by} {order_direction}
            LIMIT ? OFFSET ?
        """
        params.extend([filters.limit, filters.offset])

        with self._cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_message(row) for row in rows]

    def query_sync(self, filters: Optional[QueryFilters] = None) -> List[ProtocolMessage]:
        """Synchronous version of query."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, run synchronously
                return self._query_sync_impl(filters)
            return loop.run_until_complete(self.query(filters))
        except RuntimeError:
            return self._query_sync_impl(filters)

    def _query_sync_impl(self, filters: Optional[QueryFilters] = None) -> List[ProtocolMessage]:
        """Internal synchronous query implementation."""
        filters = filters or QueryFilters()
        conditions: List[str] = []
        params: List[Any] = []

        if filters.debate_id:
            conditions.append("debate_id = ?")
            params.append(filters.debate_id)

        if filters.agent_id:
            conditions.append("agent_id = ?")
            params.append(filters.agent_id)

        if filters.message_type:
            conditions.append("message_type = ?")
            params.append(filters.message_type.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order_direction = "DESC" if filters.order_desc else "ASC"

        query = f"""
            SELECT * FROM protocol_messages
            WHERE {where_clause}
            ORDER BY {filters.order_by} {order_direction}
            LIMIT ? OFFSET ?
        """
        params.extend([filters.limit, filters.offset])

        with self._cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_message(row) for row in rows]

    async def get(self, message_id: str) -> Optional[ProtocolMessage]:
        """Get a single message by ID."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM protocol_messages WHERE message_id = ?",
                (message_id,),
            )
            row = cursor.fetchone()

        if row:
            return self._row_to_message(row)
        return None

    async def get_debate_timeline(
        self, debate_id: str, include_types: Optional[List[ProtocolMessageType]] = None
    ) -> List[ProtocolMessage]:
        """
        Get full timeline of messages for a debate.

        Args:
            debate_id: The debate ID.
            include_types: Optional filter for specific message types.

        Returns:
            List of messages in chronological order.
        """
        filters = QueryFilters(
            debate_id=debate_id,
            message_types=include_types,
            order_by="timestamp",
            order_desc=False,
        )
        return await self.query(filters)

    async def get_round_messages(self, debate_id: str, round_number: int) -> List[ProtocolMessage]:
        """Get all messages for a specific round."""
        filters = QueryFilters(
            debate_id=debate_id,
            round_number=round_number,
            order_by="timestamp",
        )
        return await self.query(filters)

    async def get_agent_messages(self, debate_id: str, agent_id: str) -> List[ProtocolMessage]:
        """Get all messages from a specific agent in a debate."""
        filters = QueryFilters(
            debate_id=debate_id,
            agent_id=agent_id,
            order_by="timestamp",
        )
        return await self.query(filters)

    async def count(self, filters: Optional[QueryFilters] = None) -> int:
        """Count messages matching filters."""
        filters = filters or QueryFilters()
        conditions: List[str] = []
        params: List[Any] = []

        if filters.debate_id:
            conditions.append("debate_id = ?")
            params.append(filters.debate_id)

        if filters.message_type:
            conditions.append("message_type = ?")
            params.append(filters.message_type.value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) FROM protocol_messages WHERE {where_clause}",
                params,
            )
            return cursor.fetchone()[0]

    async def export_jsonl(self, debate_id: str, output_path: str) -> int:
        """
        Export debate messages to JSONL file for replay.

        Args:
            debate_id: The debate to export.
            output_path: Path to output file.

        Returns:
            Number of messages exported.
        """
        messages = await self.get_debate_timeline(debate_id)

        with open(output_path, "w") as f:
            for msg in messages:
                f.write(msg.to_json() + "\n")

        logger.info(f"Exported {len(messages)} messages to {output_path}")
        return len(messages)

    async def delete_debate(self, debate_id: str) -> int:
        """
        Delete all messages for a debate.

        Args:
            debate_id: The debate to delete.

        Returns:
            Number of messages deleted.
        """
        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM protocol_messages WHERE debate_id = ?",
                (debate_id,),
            )
            count = cursor.rowcount

        logger.info(f"Deleted {count} messages for debate {debate_id[:8]}...")
        return count

    async def cleanup_old(self, days: int = 30) -> int:
        """
        Delete messages older than specified days.

        Args:
            days: Number of days to retain.

        Returns:
            Number of messages deleted.
        """
        cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff.replace(day=cutoff.day - days)

        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM protocol_messages WHERE timestamp < ?",
                (cutoff.isoformat(),),
            )
            count = cursor.rowcount

        logger.info(f"Cleaned up {count} messages older than {days} days")
        return count

    def _row_to_message(self, row: sqlite3.Row) -> ProtocolMessage:
        """Convert database row to ProtocolMessage."""
        payload = None
        if row["payload"]:
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                payload = row["payload"]

        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        timestamp = row["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        return ProtocolMessage(
            message_id=row["message_id"],
            message_type=ProtocolMessageType(row["message_type"]),
            debate_id=row["debate_id"],
            agent_id=row["agent_id"],
            round_number=row["round_number"],
            timestamp=timestamp,
            correlation_id=row["correlation_id"],
            parent_message_id=row["parent_message_id"],
            payload=payload,
            metadata=metadata,
        )

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
