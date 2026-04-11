"""
Durable queue and processor for consequential blockchain actions.

Request-serving paths enqueue actions here instead of signing and sending
transactions inline. Processing requires an explicit chain-write approval.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aragora.security.capability_gate import (
    Capability,
    authorize_capability_dispatch,
)
from aragora.storage.backends import DatabaseBackend, get_database_backend

logger = logging.getLogger(__name__)

UTC = timezone.utc


class ChainActionType(str, Enum):
    REGISTER_AGENT = "register_agent"


class ChainActionStatus(str, Enum):
    QUEUED = "queued"
    SUBMITTED = "submitted"
    MINED = "mined"
    CONFIRMED = "confirmed"
    FAILED = "failed"


@dataclass(slots=True)
class ChainActionRecord:
    action_id: str
    action_type: ChainActionType
    requested_by: str
    approval_id: str = ""
    receipt_id: str = ""
    status: ChainActionStatus = ChainActionStatus.QUEUED
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _parse_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class ChainActionStore:
    def __init__(self, backend: DatabaseBackend | None = None) -> None:
        self._backend = backend or get_database_backend()
        self._init_db()

    def _init_db(self) -> None:
        self._backend.execute_write(
            """
            CREATE TABLE IF NOT EXISTS blockchain_chain_actions (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                requested_by TEXT NOT NULL,
                approval_id TEXT DEFAULT '',
                receipt_id TEXT DEFAULT '',
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT DEFAULT '{}',
                error TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._backend.execute_write(
            """
            CREATE INDEX IF NOT EXISTS idx_blockchain_chain_actions_status
            ON blockchain_chain_actions(status, action_type)
            """
        )

    def create_action(
        self,
        *,
        action_type: ChainActionType,
        requested_by: str,
        approval_id: str = "",
        receipt_id: str = "",
        payload: dict[str, Any],
    ) -> ChainActionRecord:
        now = datetime.now(UTC).isoformat()
        record = ChainActionRecord(
            action_id=f"chain-{uuid.uuid4().hex[:12]}",
            action_type=action_type,
            requested_by=requested_by,
            approval_id=approval_id,
            receipt_id=receipt_id,
            status=ChainActionStatus.QUEUED,
            payload=dict(payload),
            created_at=now,
            updated_at=now,
        )
        self._backend.execute_write(
            """
            INSERT INTO blockchain_chain_actions (
                action_id, action_type, requested_by, approval_id, receipt_id,
                status, payload_json, result_json, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.action_id,
                record.action_type.value,
                record.requested_by,
                record.approval_id,
                record.receipt_id,
                record.status.value,
                json.dumps(record.payload, sort_keys=True, default=str),
                json.dumps(record.result, sort_keys=True, default=str),
                record.error,
                record.created_at,
                record.updated_at,
            ),
        )
        return record

    def get_action(self, action_id: str) -> ChainActionRecord | None:
        row = self._backend.fetch_one(
            """
            SELECT action_id, action_type, requested_by, approval_id, receipt_id,
                   status, payload_json, result_json, error, created_at, updated_at
            FROM blockchain_chain_actions
            WHERE action_id = ?
            """,
            (action_id,),
        )
        if row is None:
            return None
        return ChainActionRecord(
            action_id=str(row[0]),
            action_type=ChainActionType(str(row[1])),
            requested_by=str(row[2]),
            approval_id=str(row[3] or ""),
            receipt_id=str(row[4] or ""),
            status=ChainActionStatus(str(row[5])),
            payload=_parse_json(row[6]),
            result=_parse_json(row[7]),
            error=str(row[8] or ""),
            created_at=str(row[9]),
            updated_at=str(row[10]),
        )

    def update_action(
        self,
        action_id: str,
        *,
        status: ChainActionStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        approval_id: str | None = None,
    ) -> ChainActionRecord | None:
        current = self.get_action(action_id)
        if current is None:
            return None
        merged_result = dict(current.result)
        if result:
            merged_result.update(result)
        self._backend.execute_write(
            """
            UPDATE blockchain_chain_actions
            SET status = ?, result_json = ?, error = ?, approval_id = ?, updated_at = ?
            WHERE action_id = ?
            """,
            (
                status.value,
                json.dumps(merged_result, sort_keys=True, default=str),
                error if error is not None else current.error,
                approval_id if approval_id is not None else current.approval_id,
                datetime.now(UTC).isoformat(),
                action_id,
            ),
        )
        return self.get_action(action_id)

    def list_pending(self) -> list[ChainActionRecord]:
        rows = self._backend.fetch_all(
            """
            SELECT action_id
            FROM blockchain_chain_actions
            WHERE status IN (?, ?, ?)
            ORDER BY created_at ASC
            """,
            (
                ChainActionStatus.QUEUED.value,
                ChainActionStatus.SUBMITTED.value,
                ChainActionStatus.MINED.value,
            ),
        )
        return [record for row in rows if (record := self.get_action(str(row[0])))]


_store: ChainActionStore | None = None


def get_chain_action_store() -> ChainActionStore:
    global _store
    if _store is None:
        _store = ChainActionStore()
    return _store


def enqueue_register_agent_action(
    *,
    agent_uri: str,
    metadata: dict[str, Any] | None = None,
    requested_by: str = "",
    approval_id: str = "",
    receipt_id: str = "",
) -> ChainActionRecord:
    payload = {
        "agent_uri": agent_uri,
        "metadata": dict(metadata or {}),
    }
    return get_chain_action_store().create_action(
        action_type=ChainActionType.REGISTER_AGENT,
        requested_by=requested_by or "system",
        approval_id=approval_id,
        receipt_id=receipt_id,
        payload=payload,
    )


def process_chain_action(action_id: str, *, approval_id: str = "") -> ChainActionRecord:
    record = get_chain_action_store().get_action(action_id)
    if record is None:
        raise KeyError(f"Unknown chain action: {action_id}")
    if record.action_type != ChainActionType.REGISTER_AGENT:
        raise ValueError(f"Unsupported chain action type: {record.action_type.value}")

    payload = dict(record.payload)
    effective_approval_id = str(approval_id or record.approval_id or "").strip()
    capability_action = authorize_capability_dispatch(
        capability=Capability.CHAIN_WRITE,
        actor_id=record.requested_by or "system",
        target_resource=f"erc8004:{record.action_type.value}",
        input_payload=payload,
        approval_id=effective_approval_id,
        receipt_id=record.receipt_id,
        metadata={"chain_action_id": record.action_id},
    )
    get_chain_action_store().update_action(
        action_id,
        status=ChainActionStatus.SUBMITTED,
        approval_id=effective_approval_id,
        result={"capability_action_id": capability_action.action_id},
    )

    try:
        from aragora.blockchain.contracts.identity import IdentityRegistryContract
        from aragora.blockchain.models import MetadataEntry
        from aragora.blockchain.provider import Web3Provider
        from aragora.blockchain.wallet import WalletSigner
    except ImportError as exc:
        get_chain_action_store().update_action(
            action_id,
            status=ChainActionStatus.FAILED,
            error=f"Blockchain dependencies unavailable: {exc}",
        )
        raise

    provider = Web3Provider.from_env()
    signer = WalletSigner.from_env()
    contract = IdentityRegistryContract(provider)
    metadata_entries = [
        MetadataEntry(key=str(key), value=str(value).encode("utf-8"))
        for key, value in dict(payload.get("metadata") or {}).items()
    ]
    try:
        token_id = contract.register_agent(
            str(payload.get("agent_uri") or ""), signer, metadata_entries
        )
        get_chain_action_store().update_action(
            action_id,
            status=ChainActionStatus.MINED,
            result={"token_id": token_id, "owner": signer.address},
        )
        confirmed = get_chain_action_store().update_action(
            action_id,
            status=ChainActionStatus.CONFIRMED,
            result={"token_id": token_id, "owner": signer.address},
        )
        return confirmed or record
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
        logger.error("Chain action %s failed: %s", action_id, exc)
        get_chain_action_store().update_action(
            action_id,
            status=ChainActionStatus.FAILED,
            error=str(exc),
        )
        raise


__all__ = [
    "ChainActionRecord",
    "ChainActionStatus",
    "ChainActionStore",
    "ChainActionType",
    "enqueue_register_agent_action",
    "get_chain_action_store",
    "process_chain_action",
]
