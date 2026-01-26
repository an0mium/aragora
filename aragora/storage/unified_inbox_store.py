"""
Unified Inbox Store.

Provides persistent storage for unified inbox accounts, messages, and triage results.
Backends:
- SQLiteUnifiedInboxStore: Default single-instance persistence
- PostgresUnifiedInboxStore: Multi-instance production persistence
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = "unified_inbox.db"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


class UnifiedInboxStoreBackend(ABC):
    """Abstract base for unified inbox storage backends."""

    @abstractmethod
    async def save_account(self, tenant_id: str, account: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_account(self, tenant_id: str, account_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def list_accounts(self, tenant_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_account(self, tenant_id: str, account_id: str) -> bool:
        pass

    @abstractmethod
    async def update_account_fields(
        self, tenant_id: str, account_id: str, updates: Dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    async def increment_account_counts(
        self,
        tenant_id: str,
        account_id: str,
        total_delta: int = 0,
        unread_delta: int = 0,
        sync_error_delta: int = 0,
    ) -> None:
        pass

    @abstractmethod
    async def save_message(self, tenant_id: str, message: Dict[str, Any]) -> Tuple[str, bool]:
        pass

    @abstractmethod
    async def get_message(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def list_messages(
        self,
        tenant_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        priority_tier: Optional[str] = None,
        account_id: Optional[str] = None,
        unread_only: bool = False,
        search: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        pass

    @abstractmethod
    async def delete_message(self, tenant_id: str, message_id: str) -> bool:
        pass

    @abstractmethod
    async def update_message_flags(
        self,
        tenant_id: str,
        message_id: str,
        is_read: Optional[bool] = None,
        is_starred: Optional[bool] = None,
    ) -> bool:
        pass

    @abstractmethod
    async def update_message_triage(
        self,
        tenant_id: str,
        message_id: str,
        triage_action: Optional[str],
        triage_rationale: Optional[str],
    ) -> None:
        pass

    @abstractmethod
    async def save_triage_result(self, tenant_id: str, triage: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_triage_result(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        pass


class InMemoryUnifiedInboxStore(UnifiedInboxStoreBackend):
    """In-memory unified inbox store for testing."""

    def __init__(self) -> None:
        self._accounts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._messages: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._triage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._message_index: Dict[Tuple[str, str, str], str] = {}
        self._lock = threading.Lock()

    async def save_account(self, tenant_id: str, account: Dict[str, Any]) -> None:
        with self._lock:
            self._accounts.setdefault(tenant_id, {})[account["id"]] = dict(account)

    async def get_account(self, tenant_id: str, account_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._accounts.get(tenant_id, {}).get(account_id)

    async def list_accounts(self, tenant_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._accounts.get(tenant_id, {}).values())

    async def delete_account(self, tenant_id: str, account_id: str) -> bool:
        with self._lock:
            accounts = self._accounts.get(tenant_id, {})
            if account_id not in accounts:
                return False
            del accounts[account_id]
            messages = self._messages.get(tenant_id, {})
            for message_id, message in list(messages.items()):
                if message.get("account_id") == account_id:
                    del messages[message_id]
                    self._triage.get(tenant_id, {}).pop(message_id, None)
            self._message_index = {
                key: val for key, val in self._message_index.items() if key[0] != tenant_id
            }
            return True

    async def update_account_fields(
        self, tenant_id: str, account_id: str, updates: Dict[str, Any]
    ) -> None:
        with self._lock:
            account = self._accounts.get(tenant_id, {}).get(account_id)
            if account:
                account.update(updates)

    async def increment_account_counts(
        self,
        tenant_id: str,
        account_id: str,
        total_delta: int = 0,
        unread_delta: int = 0,
        sync_error_delta: int = 0,
    ) -> None:
        with self._lock:
            account = self._accounts.get(tenant_id, {}).get(account_id)
            if not account:
                return
            account["total_messages"] = max(0, int(account.get("total_messages", 0)) + total_delta)
            account["unread_count"] = max(0, int(account.get("unread_count", 0)) + unread_delta)
            account["sync_errors"] = max(0, int(account.get("sync_errors", 0)) + sync_error_delta)

    async def save_message(self, tenant_id: str, message: Dict[str, Any]) -> Tuple[str, bool]:
        with self._lock:
            key = (tenant_id, message["account_id"], message["external_id"])
            messages = self._messages.setdefault(tenant_id, {})
            existing_id = self._message_index.get(key)
            if existing_id and existing_id in messages:
                existing = messages[existing_id]
                old_read = bool(existing.get("is_read"))
                new_read = bool(message.get("is_read"))
                messages[existing_id] = dict(message, id=existing_id)
                if old_read != new_read:
                    delta = -1 if new_read else 1
                    await self.increment_account_counts(
                        tenant_id, message["account_id"], unread_delta=delta
                    )
                return existing_id, False

            message_id = message["id"]
            messages[message_id] = dict(message)
            self._message_index[key] = message_id
            await self.increment_account_counts(
                tenant_id,
                message["account_id"],
                total_delta=1,
                unread_delta=0 if message.get("is_read") else 1,
            )
            return message_id, True

    async def get_message(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._messages.get(tenant_id, {}).get(message_id)

    async def list_messages(
        self,
        tenant_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        priority_tier: Optional[str] = None,
        account_id: Optional[str] = None,
        unread_only: bool = False,
        search: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            messages = list(self._messages.get(tenant_id, {}).values())
        if priority_tier:
            messages = [m for m in messages if m.get("priority_tier") == priority_tier]
        if account_id:
            messages = [m for m in messages if m.get("account_id") == account_id]
        if unread_only:
            messages = [m for m in messages if not m.get("is_read")]
        if search:
            q = search.lower()
            messages = [
                m
                for m in messages
                if q in (m.get("subject") or "").lower()
                or q in (m.get("sender_email") or "").lower()
                or q in (m.get("snippet") or "").lower()
            ]
        messages.sort(
            key=lambda m: (-(m.get("priority_score") or 0.0), m.get("received_at") or ""),
            reverse=False,
        )
        total = len(messages)
        if limit is not None:
            messages = messages[offset : offset + limit]
        return messages, total

    async def delete_message(self, tenant_id: str, message_id: str) -> bool:
        with self._lock:
            messages = self._messages.get(tenant_id, {})
            message = messages.pop(message_id, None)
            if not message:
                return False
            self._triage.get(tenant_id, {}).pop(message_id, None)
            if (
                tenant_id,
                message.get("account_id"),
                message.get("external_id"),
            ) in self._message_index:
                self._message_index.pop(
                    (tenant_id, message.get("account_id"), message.get("external_id")), None
                )
        unread_delta = -1 if not message.get("is_read") else 0
        await self.increment_account_counts(
            tenant_id, message["account_id"], total_delta=-1, unread_delta=unread_delta
        )
        return True

    async def update_message_flags(
        self,
        tenant_id: str,
        message_id: str,
        is_read: Optional[bool] = None,
        is_starred: Optional[bool] = None,
    ) -> bool:
        with self._lock:
            message = self._messages.get(tenant_id, {}).get(message_id)
            if not message:
                return False
            old_read = bool(message.get("is_read"))
            if is_read is not None:
                message["is_read"] = is_read
            if is_starred is not None:
                message["is_starred"] = is_starred
            new_read = bool(message.get("is_read"))
        if old_read != new_read:
            delta = -1 if new_read else 1
            await self.increment_account_counts(
                tenant_id, message["account_id"], unread_delta=delta
            )
        return True

    async def update_message_triage(
        self,
        tenant_id: str,
        message_id: str,
        triage_action: Optional[str],
        triage_rationale: Optional[str],
    ) -> None:
        with self._lock:
            message = self._messages.get(tenant_id, {}).get(message_id)
            if not message:
                return
            message["triage_action"] = triage_action
            message["triage_rationale"] = triage_rationale

    async def save_triage_result(self, tenant_id: str, triage: Dict[str, Any]) -> None:
        with self._lock:
            self._triage.setdefault(tenant_id, {})[triage["message_id"]] = dict(triage)

    async def get_triage_result(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._triage.get(tenant_id, {}).get(message_id)


class SQLiteUnifiedInboxStore(UnifiedInboxStoreBackend):
    """SQLite-backed unified inbox store."""

    def __init__(self, db_path: Path | str):
        # SECURITY: Check production guards for SQLite usage
        try:
            from aragora.storage.production_guards import (
                require_distributed_store,
                StorageMode,
            )

            require_distributed_store(
                "unified_inbox_store",
                StorageMode.SQLITE,
                "Unified inbox store using SQLite - use PostgreSQL for multi-instance deployments",
            )
        except ImportError:
            pass  # Guards not available, allow SQLite

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"SQLiteUnifiedInboxStore initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return cast(sqlite3.Connection, self._local.conn)

    def _init_schema(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS unified_inbox_accounts (
                tenant_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                email_address TEXT NOT NULL,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL,
                connected_at TEXT,
                last_sync TEXT,
                total_messages INTEGER DEFAULT 0,
                unread_count INTEGER DEFAULT 0,
                sync_errors INTEGER DEFAULT 0,
                metadata_json TEXT,
                PRIMARY KEY (tenant_id, account_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS unified_inbox_messages (
                tenant_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                external_id TEXT NOT NULL,
                subject TEXT,
                sender_email TEXT,
                sender_name TEXT,
                recipients_json TEXT,
                cc_json TEXT,
                received_at TEXT,
                snippet TEXT,
                body_preview TEXT,
                is_read INTEGER DEFAULT 0,
                is_starred INTEGER DEFAULT 0,
                has_attachments INTEGER DEFAULT 0,
                labels_json TEXT,
                thread_id TEXT,
                priority_score REAL,
                priority_tier TEXT,
                priority_reasons_json TEXT,
                triage_action TEXT,
                triage_rationale TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (tenant_id, message_id)
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unified_inbox_messages_external
            ON unified_inbox_messages(tenant_id, account_id, external_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS unified_inbox_triage_results (
                tenant_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                recommended_action TEXT,
                confidence REAL,
                rationale TEXT,
                suggested_response TEXT,
                delegate_to TEXT,
                schedule_for TEXT,
                agents_json TEXT,
                debate_summary TEXT,
                created_at TEXT,
                PRIMARY KEY (tenant_id, message_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_unified_inbox_accounts_tenant ON unified_inbox_accounts(tenant_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_unified_inbox_messages_tenant ON unified_inbox_messages(tenant_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_unified_inbox_messages_account ON unified_inbox_messages(account_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_unified_inbox_messages_priority ON unified_inbox_messages(priority_tier)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_unified_inbox_messages_received ON unified_inbox_messages(received_at)"
        )
        conn.commit()
        conn.close()

    async def save_account(self, tenant_id: str, account: Dict[str, Any]) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO unified_inbox_accounts
            (tenant_id, account_id, provider, email_address, display_name, status,
             connected_at, last_sync, total_messages, unread_count, sync_errors, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                tenant_id,
                account["id"],
                account["provider"],
                account.get("email_address", ""),
                account.get("display_name", ""),
                account.get("status", "pending"),
                _format_dt(account.get("connected_at")),
                _format_dt(account.get("last_sync")),
                int(account.get("total_messages", 0)),
                int(account.get("unread_count", 0)),
                int(account.get("sync_errors", 0)),
                json.dumps(account.get("metadata") or {}),
            ),
        )
        conn.commit()

    async def get_account(self, tenant_id: str, account_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT tenant_id, account_id, provider, email_address, display_name, status,
                   connected_at, last_sync, total_messages, unread_count, sync_errors, metadata_json
            FROM unified_inbox_accounts
            WHERE tenant_id = ? AND account_id = ?
        """,
            (tenant_id, account_id),
        ).fetchone()
        if not row:
            return None
        return self._row_to_account(row)

    async def list_accounts(self, tenant_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT tenant_id, account_id, provider, email_address, display_name, status,
                   connected_at, last_sync, total_messages, unread_count, sync_errors, metadata_json
            FROM unified_inbox_accounts
            WHERE tenant_id = ?
            ORDER BY connected_at DESC
        """,
            (tenant_id,),
        ).fetchall()
        return [self._row_to_account(row) for row in rows]

    async def delete_account(self, tenant_id: str, account_id: str) -> bool:
        conn = self._get_conn()
        message_ids = [
            row[0]
            for row in conn.execute(
                "SELECT message_id FROM unified_inbox_messages WHERE tenant_id = ? AND account_id = ?",
                (tenant_id, account_id),
            ).fetchall()
        ]
        if message_ids:
            conn.executemany(
                "DELETE FROM unified_inbox_triage_results WHERE tenant_id = ? AND message_id = ?",
                [(tenant_id, message_id) for message_id in message_ids],
            )
        conn.execute(
            "DELETE FROM unified_inbox_messages WHERE tenant_id = ? AND account_id = ?",
            (tenant_id, account_id),
        )
        cursor = conn.execute(
            "DELETE FROM unified_inbox_accounts WHERE tenant_id = ? AND account_id = ?",
            (tenant_id, account_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    async def update_account_fields(
        self, tenant_id: str, account_id: str, updates: Dict[str, Any]
    ) -> None:
        if not updates:
            return
        fields = []
        params: List[Any] = []
        for key, value in updates.items():
            if key in ("connected_at", "last_sync"):
                value = _format_dt(value)
            if key == "metadata":
                value = json.dumps(value or {})
                key = "metadata_json"
            fields.append(f"{key} = ?")
            params.append(value)
        params.extend([tenant_id, account_id])
        sql = f"""
            UPDATE unified_inbox_accounts
            SET {", ".join(fields)}
            WHERE tenant_id = ? AND account_id = ?
        """
        conn = self._get_conn()
        conn.execute(sql, tuple(params))
        conn.commit()

    async def increment_account_counts(
        self,
        tenant_id: str,
        account_id: str,
        total_delta: int = 0,
        unread_delta: int = 0,
        sync_error_delta: int = 0,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE unified_inbox_accounts
            SET total_messages = MAX(0, total_messages + ?),
                unread_count = MAX(0, unread_count + ?),
                sync_errors = MAX(0, sync_errors + ?)
            WHERE tenant_id = ? AND account_id = ?
        """,
            (total_delta, unread_delta, sync_error_delta, tenant_id, account_id),
        )
        conn.commit()

    async def save_message(self, tenant_id: str, message: Dict[str, Any]) -> Tuple[str, bool]:
        conn = self._get_conn()
        existing = conn.execute(
            """
            SELECT message_id, is_read FROM unified_inbox_messages
            WHERE tenant_id = ? AND account_id = ? AND external_id = ?
        """,
            (tenant_id, message["account_id"], message["external_id"]),
        ).fetchone()

        now = _format_dt(message.get("received_at") or _utc_now())
        created_at = _format_dt(message.get("created_at") or _utc_now())
        updated_at = _format_dt(_utc_now())
        is_read = 1 if message.get("is_read") else 0
        is_starred = 1 if message.get("is_starred") else 0
        has_attachments = 1 if message.get("has_attachments") else 0

        if existing:
            message_id = existing["message_id"]
            old_read = bool(existing["is_read"])
            conn.execute(
                """
                UPDATE unified_inbox_messages
                SET subject = ?, sender_email = ?, sender_name = ?, recipients_json = ?,
                    cc_json = ?, received_at = ?, snippet = ?, body_preview = ?, is_read = ?,
                    is_starred = ?, has_attachments = ?, labels_json = ?, thread_id = ?,
                    priority_score = ?, priority_tier = ?, priority_reasons_json = ?,
                    triage_action = ?, triage_rationale = ?, updated_at = ?
                WHERE tenant_id = ? AND message_id = ?
            """,
                (
                    message.get("subject"),
                    message.get("sender_email"),
                    message.get("sender_name"),
                    json.dumps(message.get("recipients") or []),
                    json.dumps(message.get("cc") or []),
                    _format_dt(message.get("received_at") or now),  # type: ignore[arg-type]
                    message.get("snippet"),
                    message.get("body_preview"),
                    is_read,
                    is_starred,
                    has_attachments,
                    json.dumps(message.get("labels") or []),
                    message.get("thread_id"),
                    float(message.get("priority_score") or 0.0),
                    message.get("priority_tier"),
                    json.dumps(message.get("priority_reasons") or []),
                    message.get("triage_action"),
                    message.get("triage_rationale"),
                    updated_at,
                    tenant_id,
                    message_id,
                ),
            )
            if old_read != bool(is_read):
                delta = -1 if is_read else 1
                await self.increment_account_counts(
                    tenant_id, message["account_id"], unread_delta=delta
                )
            conn.commit()
            return message_id, False

        message_id = message["id"]
        conn.execute(
            """
            INSERT INTO unified_inbox_messages
            (tenant_id, message_id, account_id, provider, external_id, subject, sender_email,
             sender_name, recipients_json, cc_json, received_at, snippet, body_preview,
             is_read, is_starred, has_attachments, labels_json, thread_id, priority_score,
             priority_tier, priority_reasons_json, triage_action, triage_rationale,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                tenant_id,
                message_id,
                message["account_id"],
                message["provider"],
                message["external_id"],
                message.get("subject"),
                message.get("sender_email"),
                message.get("sender_name"),
                json.dumps(message.get("recipients") or []),
                json.dumps(message.get("cc") or []),
                _format_dt(message.get("received_at") or now),  # type: ignore[arg-type]
                message.get("snippet"),
                message.get("body_preview"),
                is_read,
                is_starred,
                has_attachments,
                json.dumps(message.get("labels") or []),
                message.get("thread_id"),
                float(message.get("priority_score") or 0.0),
                message.get("priority_tier"),
                json.dumps(message.get("priority_reasons") or []),
                message.get("triage_action"),
                message.get("triage_rationale"),
                created_at,
                updated_at,
            ),
        )
        conn.commit()
        await self.increment_account_counts(
            tenant_id,
            message["account_id"],
            total_delta=1,
            unread_delta=0 if is_read else 1,
        )
        return message_id, True

    async def get_message(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT tenant_id, message_id, account_id, provider, external_id, subject,
                   sender_email, sender_name, recipients_json, cc_json, received_at,
                   snippet, body_preview, is_read, is_starred, has_attachments, labels_json,
                   thread_id, priority_score, priority_tier, priority_reasons_json,
                   triage_action, triage_rationale
            FROM unified_inbox_messages
            WHERE tenant_id = ? AND message_id = ?
        """,
            (tenant_id, message_id),
        ).fetchone()
        if not row:
            return None
        return self._row_to_message(row)

    async def list_messages(
        self,
        tenant_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        priority_tier: Optional[str] = None,
        account_id: Optional[str] = None,
        unread_only: bool = False,
        search: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        clauses = ["tenant_id = ?"]
        params: List[Any] = [tenant_id]
        if priority_tier:
            clauses.append("priority_tier = ?")
            params.append(priority_tier)
        if account_id:
            clauses.append("account_id = ?")
            params.append(account_id)
        if unread_only:
            clauses.append("is_read = 0")
        if search:
            clauses.append(
                "(LOWER(subject) LIKE ? OR LOWER(sender_email) LIKE ? OR LOWER(snippet) LIKE ?)"
            )
            search_value = f"%{search.lower()}%"
            params.extend([search_value, search_value, search_value])

        where_clause = " AND ".join(clauses)
        conn = self._get_conn()
        total_row = conn.execute(
            f"SELECT COUNT(*) AS total FROM unified_inbox_messages WHERE {where_clause}",
            tuple(params),
        ).fetchone()
        total = int(total_row[0]) if total_row else 0

        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        rows = conn.execute(
            f"""
            SELECT tenant_id, message_id, account_id, provider, external_id, subject,
                   sender_email, sender_name, recipients_json, cc_json, received_at,
                   snippet, body_preview, is_read, is_starred, has_attachments, labels_json,
                   thread_id, priority_score, priority_tier, priority_reasons_json,
                   triage_action, triage_rationale
            FROM unified_inbox_messages
            WHERE {where_clause}
            ORDER BY priority_score DESC, received_at DESC
            {limit_clause}
        """,
            tuple(params),
        ).fetchall()
        return [self._row_to_message(row) for row in rows], total

    async def delete_message(self, tenant_id: str, message_id: str) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT account_id, is_read
            FROM unified_inbox_messages
            WHERE tenant_id = ? AND message_id = ?
        """,
            (tenant_id, message_id),
        ).fetchone()
        if not row:
            return False
        account_id = row["account_id"]
        is_read = bool(row["is_read"])
        conn.execute(
            "DELETE FROM unified_inbox_triage_results WHERE tenant_id = ? AND message_id = ?",
            (tenant_id, message_id),
        )
        cursor = conn.execute(
            "DELETE FROM unified_inbox_messages WHERE tenant_id = ? AND message_id = ?",
            (tenant_id, message_id),
        )
        conn.commit()
        if cursor.rowcount > 0:
            await self.increment_account_counts(
                tenant_id,
                account_id,
                total_delta=-1,
                unread_delta=0 if is_read else -1,
            )
            return True
        return False

    async def update_message_flags(
        self,
        tenant_id: str,
        message_id: str,
        is_read: Optional[bool] = None,
        is_starred: Optional[bool] = None,
    ) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT account_id, is_read
            FROM unified_inbox_messages
            WHERE tenant_id = ? AND message_id = ?
        """,
            (tenant_id, message_id),
        ).fetchone()
        if not row:
            return False
        account_id = row["account_id"]
        old_read = bool(row["is_read"])
        updates = []
        params: List[Any] = []
        if is_read is not None:
            updates.append("is_read = ?")
            params.append(1 if is_read else 0)
        if is_starred is not None:
            updates.append("is_starred = ?")
            params.append(1 if is_starred else 0)
        if not updates:
            return True
        updates.append("updated_at = ?")
        params.append(_format_dt(_utc_now()))
        params.extend([tenant_id, message_id])
        conn.execute(
            f"""
            UPDATE unified_inbox_messages
            SET {", ".join(updates)}
            WHERE tenant_id = ? AND message_id = ?
        """,
            tuple(params),
        )
        conn.commit()
        new_read = is_read if is_read is not None else old_read
        if old_read != new_read:
            delta = -1 if new_read else 1
            await self.increment_account_counts(tenant_id, account_id, unread_delta=delta)
        return True

    async def update_message_triage(
        self,
        tenant_id: str,
        message_id: str,
        triage_action: Optional[str],
        triage_rationale: Optional[str],
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE unified_inbox_messages
            SET triage_action = ?, triage_rationale = ?, updated_at = ?
            WHERE tenant_id = ? AND message_id = ?
        """,
            (triage_action, triage_rationale, _format_dt(_utc_now()), tenant_id, message_id),
        )
        conn.commit()

    async def save_triage_result(self, tenant_id: str, triage: Dict[str, Any]) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO unified_inbox_triage_results
            (tenant_id, message_id, recommended_action, confidence, rationale,
             suggested_response, delegate_to, schedule_for, agents_json, debate_summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                tenant_id,
                triage["message_id"],
                triage.get("recommended_action"),
                triage.get("confidence"),
                triage.get("rationale"),
                triage.get("suggested_response"),
                triage.get("delegate_to"),
                _format_dt(triage.get("schedule_for")),
                json.dumps(triage.get("agents_involved") or []),
                triage.get("debate_summary"),
                _format_dt(triage.get("created_at") or _utc_now()),
            ),
        )
        conn.commit()

    async def get_triage_result(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT tenant_id, message_id, recommended_action, confidence, rationale,
                   suggested_response, delegate_to, schedule_for, agents_json,
                   debate_summary, created_at
            FROM unified_inbox_triage_results
            WHERE tenant_id = ? AND message_id = ?
        """,
            (tenant_id, message_id),
        ).fetchone()
        if not row:
            return None
        return self._row_to_triage(row)

    def _row_to_account(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["account_id"],
            "provider": row["provider"],
            "email_address": row["email_address"],
            "display_name": row["display_name"],
            "status": row["status"],
            "connected_at": _parse_dt(row["connected_at"]),
            "last_sync": _parse_dt(row["last_sync"]),
            "total_messages": int(row["total_messages"] or 0),
            "unread_count": int(row["unread_count"] or 0),
            "sync_errors": int(row["sync_errors"] or 0),
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def _row_to_message(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["message_id"],
            "account_id": row["account_id"],
            "provider": row["provider"],
            "external_id": row["external_id"],
            "subject": row["subject"] or "",
            "sender_email": row["sender_email"] or "",
            "sender_name": row["sender_name"] or "",
            "recipients": _json_loads(row["recipients_json"], []),
            "cc": _json_loads(row["cc_json"], []),
            "received_at": _parse_dt(row["received_at"]) or _utc_now(),
            "snippet": row["snippet"] or "",
            "body_preview": row["body_preview"] or "",
            "is_read": bool(row["is_read"]),
            "is_starred": bool(row["is_starred"]),
            "has_attachments": bool(row["has_attachments"]),
            "labels": _json_loads(row["labels_json"], []),
            "thread_id": row["thread_id"],
            "priority_score": float(row["priority_score"] or 0.0),
            "priority_tier": row["priority_tier"] or "medium",
            "priority_reasons": _json_loads(row["priority_reasons_json"], []),
            "triage_action": row["triage_action"],
            "triage_rationale": row["triage_rationale"],
        }

    def _row_to_triage(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "message_id": row["message_id"],
            "recommended_action": row["recommended_action"],
            "confidence": float(row["confidence"] or 0.0),
            "rationale": row["rationale"],
            "suggested_response": row["suggested_response"],
            "delegate_to": row["delegate_to"],
            "schedule_for": _parse_dt(row["schedule_for"]),
            "agents_involved": _json_loads(row["agents_json"], []),
            "debate_summary": row["debate_summary"],
            "created_at": _parse_dt(row["created_at"]),
        }


class PostgresUnifiedInboxStore(UnifiedInboxStoreBackend):
    """PostgreSQL-backed unified inbox store."""

    SCHEMA_NAME = "unified_inbox"
    SCHEMA_VERSION = 1
    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS unified_inbox_accounts (
            tenant_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            email_address TEXT NOT NULL,
            display_name TEXT NOT NULL,
            status TEXT NOT NULL,
            connected_at TIMESTAMPTZ,
            last_sync TIMESTAMPTZ,
            total_messages INTEGER DEFAULT 0,
            unread_count INTEGER DEFAULT 0,
            sync_errors INTEGER DEFAULT 0,
            metadata_json TEXT,
            PRIMARY KEY (tenant_id, account_id)
        );

        CREATE TABLE IF NOT EXISTS unified_inbox_messages (
            tenant_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            external_id TEXT NOT NULL,
            subject TEXT,
            sender_email TEXT,
            sender_name TEXT,
            recipients_json TEXT,
            cc_json TEXT,
            received_at TIMESTAMPTZ,
            snippet TEXT,
            body_preview TEXT,
            is_read BOOLEAN DEFAULT FALSE,
            is_starred BOOLEAN DEFAULT FALSE,
            has_attachments BOOLEAN DEFAULT FALSE,
            labels_json TEXT,
            thread_id TEXT,
            priority_score DOUBLE PRECISION,
            priority_tier TEXT,
            priority_reasons_json TEXT,
            triage_action TEXT,
            triage_rationale TEXT,
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ,
            PRIMARY KEY (tenant_id, message_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unified_inbox_messages_external
            ON unified_inbox_messages(tenant_id, account_id, external_id);

        CREATE TABLE IF NOT EXISTS unified_inbox_triage_results (
            tenant_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            recommended_action TEXT,
            confidence DOUBLE PRECISION,
            rationale TEXT,
            suggested_response TEXT,
            delegate_to TEXT,
            schedule_for TIMESTAMPTZ,
            agents_json TEXT,
            debate_summary TEXT,
            created_at TIMESTAMPTZ,
            PRIMARY KEY (tenant_id, message_id)
        );

        CREATE INDEX IF NOT EXISTS idx_unified_inbox_accounts_tenant
            ON unified_inbox_accounts(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_unified_inbox_messages_tenant
            ON unified_inbox_messages(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_unified_inbox_messages_account
            ON unified_inbox_messages(account_id);
        CREATE INDEX IF NOT EXISTS idx_unified_inbox_messages_priority
            ON unified_inbox_messages(priority_tier);
        CREATE INDEX IF NOT EXISTS idx_unified_inbox_messages_received
            ON unified_inbox_messages(received_at);
    """

    def __init__(self, pool: "Pool"):
        from aragora.storage.postgres_store import PostgresStore

        class _Store(PostgresStore):
            SCHEMA_NAME = PostgresUnifiedInboxStore.SCHEMA_NAME
            SCHEMA_VERSION = PostgresUnifiedInboxStore.SCHEMA_VERSION
            INITIAL_SCHEMA = PostgresUnifiedInboxStore.INITIAL_SCHEMA

        self._store = _Store(pool)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._store.initialize()
        self._initialized = True
        logger.debug("PostgresUnifiedInboxStore schema initialized")

    async def save_account(self, tenant_id: str, account: Dict[str, Any]) -> None:
        await self._store.execute(
            """
            INSERT INTO unified_inbox_accounts
            (tenant_id, account_id, provider, email_address, display_name, status,
             connected_at, last_sync, total_messages, unread_count, sync_errors, metadata_json)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (tenant_id, account_id) DO UPDATE SET
                provider = EXCLUDED.provider,
                email_address = EXCLUDED.email_address,
                display_name = EXCLUDED.display_name,
                status = EXCLUDED.status,
                connected_at = EXCLUDED.connected_at,
                last_sync = EXCLUDED.last_sync,
                total_messages = EXCLUDED.total_messages,
                unread_count = EXCLUDED.unread_count,
                sync_errors = EXCLUDED.sync_errors,
                metadata_json = EXCLUDED.metadata_json
        """,
            tenant_id,
            account["id"],
            account["provider"],
            account.get("email_address", ""),
            account.get("display_name", ""),
            account.get("status", "pending"),
            account.get("connected_at"),
            account.get("last_sync"),
            int(account.get("total_messages", 0)),
            int(account.get("unread_count", 0)),
            int(account.get("sync_errors", 0)),
            json.dumps(account.get("metadata") or {}),
        )

    async def get_account(self, tenant_id: str, account_id: str) -> Optional[Dict[str, Any]]:
        row = await self._store.fetch_one(
            """
            SELECT tenant_id, account_id, provider, email_address, display_name, status,
                   connected_at, last_sync, total_messages, unread_count, sync_errors, metadata_json
            FROM unified_inbox_accounts
            WHERE tenant_id = $1 AND account_id = $2
        """,
            tenant_id,
            account_id,
        )
        if not row:
            return None
        return self._row_to_account(row)

    async def list_accounts(self, tenant_id: str) -> List[Dict[str, Any]]:
        rows = await self._store.fetch_all(
            """
            SELECT tenant_id, account_id, provider, email_address, display_name, status,
                   connected_at, last_sync, total_messages, unread_count, sync_errors, metadata_json
            FROM unified_inbox_accounts
            WHERE tenant_id = $1
            ORDER BY connected_at DESC
        """,
            tenant_id,
        )
        return [self._row_to_account(row) for row in rows]

    async def delete_account(self, tenant_id: str, account_id: str) -> bool:
        rows = await self._store.fetch_all(
            """
            SELECT message_id FROM unified_inbox_messages
            WHERE tenant_id = $1 AND account_id = $2
        """,
            tenant_id,
            account_id,
        )
        if rows:
            await self._store.executemany(
                """
                DELETE FROM unified_inbox_triage_results
                WHERE tenant_id = $1 AND message_id = $2
            """,
                [(tenant_id, row["message_id"]) for row in rows],
            )
        await self._store.execute(
            """
            DELETE FROM unified_inbox_messages
            WHERE tenant_id = $1 AND account_id = $2
        """,
            tenant_id,
            account_id,
        )
        result = await self._store.execute(
            """
            DELETE FROM unified_inbox_accounts
            WHERE tenant_id = $1 AND account_id = $2
        """,
            tenant_id,
            account_id,
        )
        return result != "DELETE 0"

    async def update_account_fields(
        self, tenant_id: str, account_id: str, updates: Dict[str, Any]
    ) -> None:
        if not updates:
            return
        fields = []
        values: List[Any] = []
        idx = 1
        for key, value in updates.items():
            if key == "metadata":
                key = "metadata_json"
                value = json.dumps(value or {})
            fields.append(f"{key} = ${idx}")
            values.append(value)
            idx += 1
        values.extend([tenant_id, account_id])
        sql = f"""
            UPDATE unified_inbox_accounts
            SET {", ".join(fields)}
            WHERE tenant_id = ${idx} AND account_id = ${idx + 1}
        """
        await self._store.execute(sql, *values)

    async def increment_account_counts(
        self,
        tenant_id: str,
        account_id: str,
        total_delta: int = 0,
        unread_delta: int = 0,
        sync_error_delta: int = 0,
    ) -> None:
        await self._store.execute(
            """
            UPDATE unified_inbox_accounts
            SET total_messages = GREATEST(0, total_messages + $1),
                unread_count = GREATEST(0, unread_count + $2),
                sync_errors = GREATEST(0, sync_errors + $3)
            WHERE tenant_id = $4 AND account_id = $5
        """,
            total_delta,
            unread_delta,
            sync_error_delta,
            tenant_id,
            account_id,
        )

    async def save_message(self, tenant_id: str, message: Dict[str, Any]) -> Tuple[str, bool]:
        row = await self._store.fetch_one(
            """
            SELECT message_id, is_read
            FROM unified_inbox_messages
            WHERE tenant_id = $1 AND account_id = $2 AND external_id = $3
        """,
            tenant_id,
            message["account_id"],
            message["external_id"],
        )
        is_read = bool(message.get("is_read"))
        if row:
            message_id = row["message_id"]
            old_read = bool(row["is_read"])
            await self._store.execute(
                """
                UPDATE unified_inbox_messages
                SET subject = $1, sender_email = $2, sender_name = $3, recipients_json = $4,
                    cc_json = $5, received_at = $6, snippet = $7, body_preview = $8,
                    is_read = $9, is_starred = $10, has_attachments = $11, labels_json = $12,
                    thread_id = $13, priority_score = $14, priority_tier = $15,
                    priority_reasons_json = $16, triage_action = $17, triage_rationale = $18,
                    updated_at = $19
                WHERE tenant_id = $20 AND message_id = $21
            """,
                message.get("subject"),
                message.get("sender_email"),
                message.get("sender_name"),
                json.dumps(message.get("recipients") or []),
                json.dumps(message.get("cc") or []),
                message.get("received_at") or _utc_now(),
                message.get("snippet"),
                message.get("body_preview"),
                is_read,
                bool(message.get("is_starred")),
                bool(message.get("has_attachments")),
                json.dumps(message.get("labels") or []),
                message.get("thread_id"),
                float(message.get("priority_score") or 0.0),
                message.get("priority_tier"),
                json.dumps(message.get("priority_reasons") or []),
                message.get("triage_action"),
                message.get("triage_rationale"),
                _utc_now(),
                tenant_id,
                message_id,
            )
            if old_read != is_read:
                delta = -1 if is_read else 1
                await self.increment_account_counts(
                    tenant_id, message["account_id"], unread_delta=delta
                )
            return message_id, False

        message_id = message["id"]
        await self._store.execute(
            """
            INSERT INTO unified_inbox_messages
            (tenant_id, message_id, account_id, provider, external_id, subject, sender_email,
             sender_name, recipients_json, cc_json, received_at, snippet, body_preview, is_read,
             is_starred, has_attachments, labels_json, thread_id, priority_score, priority_tier,
             priority_reasons_json, triage_action, triage_rationale, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                    $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
        """,
            tenant_id,
            message_id,
            message["account_id"],
            message["provider"],
            message["external_id"],
            message.get("subject"),
            message.get("sender_email"),
            message.get("sender_name"),
            json.dumps(message.get("recipients") or []),
            json.dumps(message.get("cc") or []),
            message.get("received_at") or _utc_now(),
            message.get("snippet"),
            message.get("body_preview"),
            is_read,
            bool(message.get("is_starred")),
            bool(message.get("has_attachments")),
            json.dumps(message.get("labels") or []),
            message.get("thread_id"),
            float(message.get("priority_score") or 0.0),
            message.get("priority_tier"),
            json.dumps(message.get("priority_reasons") or []),
            message.get("triage_action"),
            message.get("triage_rationale"),
            _utc_now(),
            _utc_now(),
        )
        await self.increment_account_counts(
            tenant_id,
            message["account_id"],
            total_delta=1,
            unread_delta=0 if is_read else 1,
        )
        return message_id, True

    async def get_message(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        row = await self._store.fetch_one(
            """
            SELECT tenant_id, message_id, account_id, provider, external_id, subject,
                   sender_email, sender_name, recipients_json, cc_json, received_at,
                   snippet, body_preview, is_read, is_starred, has_attachments, labels_json,
                   thread_id, priority_score, priority_tier, priority_reasons_json,
                   triage_action, triage_rationale
            FROM unified_inbox_messages
            WHERE tenant_id = $1 AND message_id = $2
        """,
            tenant_id,
            message_id,
        )
        if not row:
            return None
        return self._row_to_message(row)

    async def list_messages(
        self,
        tenant_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        priority_tier: Optional[str] = None,
        account_id: Optional[str] = None,
        unread_only: bool = False,
        search: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        clauses = ["tenant_id = $1"]
        params: List[Any] = [tenant_id]
        idx = 2
        if priority_tier:
            clauses.append(f"priority_tier = ${idx}")
            params.append(priority_tier)
            idx += 1
        if account_id:
            clauses.append(f"account_id = ${idx}")
            params.append(account_id)
            idx += 1
        if unread_only:
            clauses.append("is_read = FALSE")
        if search:
            clauses.append(
                f"(LOWER(subject) LIKE ${idx} OR LOWER(sender_email) LIKE ${idx} OR LOWER(snippet) LIKE ${idx})"
            )
            params.append(f"%{search.lower()}%")
            idx += 1

        where_clause = " AND ".join(clauses)
        total_row = await self._store.fetch_one(
            f"SELECT COUNT(*) AS total FROM unified_inbox_messages WHERE {where_clause}",
            *params,
        )
        total = int(total_row["total"]) if total_row else 0

        limit_clause = ""
        if limit is not None:
            limit_clause = f" LIMIT {limit} OFFSET {offset}"

        rows = await self._store.fetch_all(
            f"""
            SELECT tenant_id, message_id, account_id, provider, external_id, subject,
                   sender_email, sender_name, recipients_json, cc_json, received_at,
                   snippet, body_preview, is_read, is_starred, has_attachments, labels_json,
                   thread_id, priority_score, priority_tier, priority_reasons_json,
                   triage_action, triage_rationale
            FROM unified_inbox_messages
            WHERE {where_clause}
            ORDER BY priority_score DESC, received_at DESC
            {limit_clause}
        """,
            *params,
        )
        return [self._row_to_message(row) for row in rows], total

    async def delete_message(self, tenant_id: str, message_id: str) -> bool:
        row = await self._store.fetch_one(
            """
            SELECT account_id, is_read
            FROM unified_inbox_messages
            WHERE tenant_id = $1 AND message_id = $2
        """,
            tenant_id,
            message_id,
        )
        if not row:
            return False
        account_id = row["account_id"]
        is_read = bool(row["is_read"])
        await self._store.execute(
            """
            DELETE FROM unified_inbox_triage_results
            WHERE tenant_id = $1 AND message_id = $2
        """,
            tenant_id,
            message_id,
        )
        result = await self._store.execute(
            """
            DELETE FROM unified_inbox_messages
            WHERE tenant_id = $1 AND message_id = $2
        """,
            tenant_id,
            message_id,
        )
        if result != "DELETE 0":
            await self.increment_account_counts(
                tenant_id,
                account_id,
                total_delta=-1,
                unread_delta=0 if is_read else -1,
            )
            return True
        return False

    async def update_message_flags(
        self,
        tenant_id: str,
        message_id: str,
        is_read: Optional[bool] = None,
        is_starred: Optional[bool] = None,
    ) -> bool:
        row = await self._store.fetch_one(
            """
            SELECT account_id, is_read
            FROM unified_inbox_messages
            WHERE tenant_id = $1 AND message_id = $2
        """,
            tenant_id,
            message_id,
        )
        if not row:
            return False
        account_id = row["account_id"]
        old_read = bool(row["is_read"])
        updates = []
        params: List[Any] = []
        idx = 1
        if is_read is not None:
            updates.append(f"is_read = ${idx}")
            params.append(is_read)
            idx += 1
        if is_starred is not None:
            updates.append(f"is_starred = ${idx}")
            params.append(is_starred)
            idx += 1
        if not updates:
            return True
        updates.append(f"updated_at = ${idx}")
        params.append(_utc_now())
        params.extend([tenant_id, message_id])
        await self._store.execute(
            f"""
            UPDATE unified_inbox_messages
            SET {", ".join(updates)}
            WHERE tenant_id = ${idx + 1} AND message_id = ${idx + 2}
        """,
            *params,
        )
        new_read = is_read if is_read is not None else old_read
        if old_read != new_read:
            delta = -1 if new_read else 1
            await self.increment_account_counts(tenant_id, account_id, unread_delta=delta)
        return True

    async def update_message_triage(
        self,
        tenant_id: str,
        message_id: str,
        triage_action: Optional[str],
        triage_rationale: Optional[str],
    ) -> None:
        await self._store.execute(
            """
            UPDATE unified_inbox_messages
            SET triage_action = $1, triage_rationale = $2, updated_at = $3
            WHERE tenant_id = $4 AND message_id = $5
        """,
            triage_action,
            triage_rationale,
            _utc_now(),
            tenant_id,
            message_id,
        )

    async def save_triage_result(self, tenant_id: str, triage: Dict[str, Any]) -> None:
        await self._store.execute(
            """
            INSERT INTO unified_inbox_triage_results
            (tenant_id, message_id, recommended_action, confidence, rationale,
             suggested_response, delegate_to, schedule_for, agents_json, debate_summary, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (tenant_id, message_id) DO UPDATE SET
                recommended_action = EXCLUDED.recommended_action,
                confidence = EXCLUDED.confidence,
                rationale = EXCLUDED.rationale,
                suggested_response = EXCLUDED.suggested_response,
                delegate_to = EXCLUDED.delegate_to,
                schedule_for = EXCLUDED.schedule_for,
                agents_json = EXCLUDED.agents_json,
                debate_summary = EXCLUDED.debate_summary,
                created_at = EXCLUDED.created_at
        """,
            tenant_id,
            triage["message_id"],
            triage.get("recommended_action"),
            triage.get("confidence"),
            triage.get("rationale"),
            triage.get("suggested_response"),
            triage.get("delegate_to"),
            triage.get("schedule_for"),
            json.dumps(triage.get("agents_involved") or []),
            triage.get("debate_summary"),
            triage.get("created_at") or _utc_now(),
        )

    async def get_triage_result(self, tenant_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        row = await self._store.fetch_one(
            """
            SELECT tenant_id, message_id, recommended_action, confidence, rationale,
                   suggested_response, delegate_to, schedule_for, agents_json,
                   debate_summary, created_at
            FROM unified_inbox_triage_results
            WHERE tenant_id = $1 AND message_id = $2
        """,
            tenant_id,
            message_id,
        )
        if not row:
            return None
        return self._row_to_triage(row)

    def _row_to_account(self, row: Any) -> Dict[str, Any]:
        return {
            "id": row["account_id"],
            "provider": row["provider"],
            "email_address": row["email_address"],
            "display_name": row["display_name"],
            "status": row["status"],
            "connected_at": row["connected_at"],
            "last_sync": row["last_sync"],
            "total_messages": int(row["total_messages"] or 0),
            "unread_count": int(row["unread_count"] or 0),
            "sync_errors": int(row["sync_errors"] or 0),
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def _row_to_message(self, row: Any) -> Dict[str, Any]:
        return {
            "id": row["message_id"],
            "account_id": row["account_id"],
            "provider": row["provider"],
            "external_id": row["external_id"],
            "subject": row["subject"] or "",
            "sender_email": row["sender_email"] or "",
            "sender_name": row["sender_name"] or "",
            "recipients": _json_loads(row["recipients_json"], []),
            "cc": _json_loads(row["cc_json"], []),
            "received_at": row["received_at"] or _utc_now(),
            "snippet": row["snippet"] or "",
            "body_preview": row["body_preview"] or "",
            "is_read": bool(row["is_read"]),
            "is_starred": bool(row["is_starred"]),
            "has_attachments": bool(row["has_attachments"]),
            "labels": _json_loads(row["labels_json"], []),
            "thread_id": row["thread_id"],
            "priority_score": float(row["priority_score"] or 0.0),
            "priority_tier": row["priority_tier"] or "medium",
            "priority_reasons": _json_loads(row["priority_reasons_json"], []),
            "triage_action": row["triage_action"],
            "triage_rationale": row["triage_rationale"],
        }

    def _row_to_triage(self, row: Any) -> Dict[str, Any]:
        return {
            "message_id": row["message_id"],
            "recommended_action": row["recommended_action"],
            "confidence": float(row["confidence"] or 0.0),
            "rationale": row["rationale"],
            "suggested_response": row["suggested_response"],
            "delegate_to": row["delegate_to"],
            "schedule_for": row["schedule_for"],
            "agents_involved": _json_loads(row["agents_json"], []),
            "debate_summary": row["debate_summary"],
            "created_at": row["created_at"],
        }


_unified_inbox_store: Optional[UnifiedInboxStoreBackend] = None
_store_lock = threading.Lock()


def get_unified_inbox_store() -> UnifiedInboxStoreBackend:
    """
    Get the unified inbox store.

    Backend selection (in preference order):
    1. Supabase PostgreSQL (if SUPABASE_URL + SUPABASE_DB_PASSWORD configured)
    2. Self-hosted PostgreSQL (if DATABASE_URL or ARAGORA_POSTGRES_DSN configured)
    3. SQLite (fallback, with production warning)

    Override via:
    - ARAGORA_INBOX_STORE_BACKEND: "memory", "sqlite", "postgres", or "supabase"
    - ARAGORA_DB_BACKEND: Global override
    """
    global _unified_inbox_store

    if _unified_inbox_store is not None:
        return _unified_inbox_store

    with _store_lock:
        if _unified_inbox_store is not None:
            return _unified_inbox_store

        from aragora.storage.connection_factory import create_persistent_store

        _unified_inbox_store = create_persistent_store(
            store_name="inbox",
            sqlite_class=SQLiteUnifiedInboxStore,
            postgres_class=PostgresUnifiedInboxStore,
            db_filename=DEFAULT_DB_NAME,
            memory_class=InMemoryUnifiedInboxStore,
        )

        return _unified_inbox_store


def set_unified_inbox_store(store: UnifiedInboxStoreBackend) -> None:
    """Set a custom unified inbox store (testing or customization)."""
    global _unified_inbox_store
    _unified_inbox_store = store


def reset_unified_inbox_store() -> None:
    """Reset the unified inbox store singleton (testing)."""
    global _unified_inbox_store
    _unified_inbox_store = None


__all__ = [
    "UnifiedInboxStoreBackend",
    "InMemoryUnifiedInboxStore",
    "SQLiteUnifiedInboxStore",
    "PostgresUnifiedInboxStore",
    "get_unified_inbox_store",
    "set_unified_inbox_store",
    "reset_unified_inbox_store",
]
