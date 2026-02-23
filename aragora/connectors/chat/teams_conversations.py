"""
Teams Conversation Reference Storage.

Stores conversation references from Teams for proactive messaging.
When a debate starts from Teams, we save the conversation reference
so we can send results back to the same channel/chat later.

Usage:
    from aragora.connectors.chat.teams_conversations import (
        TeamsConversationStore,
        TeamsConversationReference,
    )

    store = TeamsConversationStore()

    # Save reference when debate starts
    ref = TeamsConversationReference.from_activity(activity)
    await store.save_reference(debate_id, ref)

    # Later, retrieve to send proactive message
    ref = await store.get_reference(debate_id)
    if ref:
        await teams_connector.send_proactive_message(ref, message)

Schema:
    CREATE TABLE teams_conversations (
        debate_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        service_url TEXT NOT NULL,
        channel_id TEXT,
        tenant_id TEXT NOT NULL,
        bot_id TEXT NOT NULL,
        activity_id TEXT,
        user_id TEXT,
        message_id TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        metadata TEXT
    );
"""

from __future__ import annotations

import contextvars
import json
import logging
import sqlite3
import threading
import time
from types import SimpleNamespace
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TeamsConversationReference:
    """
    Reference to a Teams conversation for proactive messaging.

    Contains all information needed to send a proactive message back
    to a Teams conversation (channel, chat, or 1:1).
    """

    conversation_id: str
    service_url: str
    tenant_id: str
    bot_id: str
    channel_id: str | None = None
    activity_id: str | None = None  # For replying to specific message
    user_id: str | None = None  # For 1:1 conversations
    message_id: str | None = None  # Thread/reply reference
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "service_url": self.service_url,
            "channel_id": self.channel_id,
            "tenant_id": self.tenant_id,
            "bot_id": self.bot_id,
            "activity_id": self.activity_id,
            "user_id": self.user_id,
            "message_id": self.message_id,
            "metadata": self.metadata,
        }

    def to_bot_framework_reference(self) -> dict[str, Any]:
        """
        Convert to Bot Framework conversation reference format.

        This format is used by the Bot Framework SDK for proactive messaging.
        """
        reference: dict[str, Any] = {
            "conversation": {
                "id": self.conversation_id,
                "tenantId": self.tenant_id,
            },
            "serviceUrl": self.service_url,
            "channelId": "msteams",
            "bot": {
                "id": self.bot_id,
            },
        }

        if self.channel_id:
            reference["conversation"]["conversationType"] = "channel"
            reference["channelId"] = self.channel_id
        else:
            reference["conversation"]["conversationType"] = (
                "personal" if self.user_id else "groupChat"
            )

        if self.user_id:
            reference["user"] = {"id": self.user_id}

        if self.activity_id:
            reference["activityId"] = self.activity_id

        return reference

    @classmethod
    def from_activity(cls, activity: dict[str, Any]) -> TeamsConversationReference:
        """
        Create a conversation reference from a Bot Framework activity.

        Args:
            activity: Bot Framework activity dictionary

        Returns:
            TeamsConversationReference
        """
        conversation = activity.get("conversation", {})
        channel_data = activity.get("channelData", {})
        recipient = activity.get("recipient", {})

        # Get tenant ID from various possible locations
        tenant_id = conversation.get("tenantId") or channel_data.get("tenant", {}).get("id") or ""

        # Get channel ID if it's a channel message
        channel_id = channel_data.get("channel", {}).get("id")

        # Get bot ID from recipient (the bot is the recipient of incoming messages)
        bot_id = recipient.get("id", "")

        return cls(
            conversation_id=conversation.get("id", ""),
            service_url=activity.get("serviceUrl", ""),
            channel_id=channel_id,
            tenant_id=tenant_id,
            bot_id=bot_id,
            activity_id=activity.get("id"),
            user_id=activity.get("from", {}).get("id"),
            message_id=activity.get("replyToId"),
            metadata={
                "conversation_name": conversation.get("name"),
                "channel_name": channel_data.get("channel", {}).get("name"),
                "team_id": channel_data.get("team", {}).get("id"),
                "team_name": channel_data.get("team", {}).get("name"),
            },
        )


@dataclass
class StoredConversation:
    """A stored conversation reference with timestamps."""

    debate_id: str
    reference: TeamsConversationReference
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "debate_id": self.debate_id,
            "reference": self.reference.to_dict(),
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "updated_at": self.updated_at,
        }


class TeamsConversationStore:
    """
    Storage for Teams conversation references.

    Persists conversation references so we can send proactive messages
    back to Teams after asynchronous operations (like debates) complete.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS teams_conversations (
        debate_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        service_url TEXT NOT NULL,
        channel_id TEXT,
        tenant_id TEXT NOT NULL,
        bot_id TEXT NOT NULL,
        activity_id TEXT,
        user_id TEXT,
        message_id TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        metadata TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_teams_conv_tenant
        ON teams_conversations(tenant_id);

    CREATE INDEX IF NOT EXISTS idx_teams_conv_created
        ON teams_conversations(created_at DESC);
    """

    _conn_var: contextvars.ContextVar[sqlite3.Connection | None] = contextvars.ContextVar(
        "teams_conversation_conn", default=None
    )

    def __init__(self, db_path: str | None = None):
        """Initialize the conversation store.

        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            from aragora.persistence.db_config import get_nomic_dir

            data_dir = get_nomic_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "teams_conversations.db")

        self._db_path = db_path
        self._connections: list[sqlite3.Connection] = []
        self._init_lock = threading.Lock()
        self._initialized = False
        # Use a per-instance context var to avoid cross-test/db contamination.
        self._conn_var = contextvars.ContextVar(f"teams_conversation_conn_{id(self)}", default=None)
        # Backwards-compatible local storage for tests and legacy callers.
        self._local = SimpleNamespace(connection=None)

    def _get_connection(self) -> sqlite3.Connection:
        """Get context-local database connection."""
        local_conn = getattr(self._local, "connection", None)
        if local_conn is not None:
            return local_conn

        conn = self._conn_var.get()
        if conn is None:
            db_dir = Path(self._db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            self._conn_var.set(conn)
            self._connections.append(conn)
            self._local.connection = conn

        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure database schema exists."""
        with self._init_lock:
            if not self._initialized:
                conn.executescript(self.SCHEMA)
                conn.commit()
                self._initialized = True

    async def save_reference(
        self,
        debate_id: str,
        reference: TeamsConversationReference,
    ) -> bool:
        """
        Save a conversation reference for a debate.

        Args:
            debate_id: The debate ID to associate with this conversation
            reference: The conversation reference

        Returns:
            True if saved successfully
        """
        conn = self._get_connection()
        now = time.time()

        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO teams_conversations
                (debate_id, conversation_id, service_url, channel_id, tenant_id,
                 bot_id, activity_id, user_id, message_id, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    debate_id,
                    reference.conversation_id,
                    reference.service_url,
                    reference.channel_id,
                    reference.tenant_id,
                    reference.bot_id,
                    reference.activity_id,
                    reference.user_id,
                    reference.message_id,
                    now,
                    now,
                    json.dumps(reference.metadata),
                ),
            )
            conn.commit()
            logger.debug("Saved conversation reference for debate: %s", debate_id)
            return True

        except (sqlite3.Error, Exception) as e:
            logger.error("Failed to save conversation reference: %s", e)
            return False

    async def get_reference(
        self,
        debate_id: str,
    ) -> TeamsConversationReference | None:
        """
        Get the conversation reference for a debate.

        Args:
            debate_id: The debate ID

        Returns:
            TeamsConversationReference if found
        """
        conn = self._get_connection()

        try:
            cursor = conn.execute(
                "SELECT * FROM teams_conversations WHERE debate_id = ?",
                (debate_id,),
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_reference(row)
            return None

        except (sqlite3.Error, Exception) as e:
            logger.error("Failed to get conversation reference: %s", e)
            return None

    async def delete_reference(self, debate_id: str) -> bool:
        """
        Delete a conversation reference.

        Args:
            debate_id: The debate ID

        Returns:
            True if deleted
        """
        conn = self._get_connection()

        try:
            result = conn.execute(
                "DELETE FROM teams_conversations WHERE debate_id = ?",
                (debate_id,),
            )
            conn.commit()
            return result.rowcount > 0

        except (sqlite3.Error, Exception) as e:
            logger.error("Failed to delete conversation reference: %s", e)
            return False

    async def get_by_tenant(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[StoredConversation]:
        """
        Get all conversation references for a tenant.

        Args:
            tenant_id: Azure AD tenant ID
            limit: Maximum results

        Returns:
            List of StoredConversation
        """
        conn = self._get_connection()

        try:
            cursor = conn.execute(
                """
                SELECT * FROM teams_conversations
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (tenant_id, limit),
            )

            results = []
            for row in cursor.fetchall():
                ref = self._row_to_reference(row)
                results.append(
                    StoredConversation(
                        debate_id=row["debate_id"],
                        reference=ref,
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )
            return results

        except (sqlite3.Error, Exception) as e:
            logger.error("Failed to get conversations by tenant: %s", e)
            return []

    async def cleanup_old(self, max_age_days: int = 30) -> int:
        """
        Clean up old conversation references.

        Args:
            max_age_days: Delete references older than this

        Returns:
            Number of references deleted
        """
        conn = self._get_connection()
        cutoff = time.time() - (max_age_days * 86400)

        try:
            result = conn.execute(
                "DELETE FROM teams_conversations WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            count = result.rowcount
            if count > 0:
                logger.info("Cleaned up %s old conversation references", count)
            return count

        except (sqlite3.Error, Exception) as e:
            logger.error("Failed to cleanup old references: %s", e)
            return 0

    def _row_to_reference(self, row: sqlite3.Row) -> TeamsConversationReference:
        """Convert database row to TeamsConversationReference."""
        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError as e:
                logger.debug("Failed to parse JSON data: %s", e)

        return TeamsConversationReference(
            conversation_id=row["conversation_id"],
            service_url=row["service_url"],
            channel_id=row["channel_id"],
            tenant_id=row["tenant_id"],
            bot_id=row["bot_id"],
            activity_id=row["activity_id"],
            user_id=row["user_id"],
            message_id=row["message_id"],
            metadata=metadata,
        )


# Singleton instance
_store: TeamsConversationStore | None = None


def get_teams_conversation_store() -> TeamsConversationStore:
    """Get or create the Teams conversation store singleton."""
    global _store
    if _store is None:
        _store = TeamsConversationStore()
    return _store
