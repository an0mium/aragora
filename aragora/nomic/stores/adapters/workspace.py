"""
Workspace <-> Nomic mapping helpers.

Centralizes status translations and metadata encoding for workspace
beads/convoys backed by the canonical Nomic stores.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

from aragora.nomic.beads import Bead as NomicBead
from aragora.nomic.beads import BeadStatus as NomicBeadStatus
from aragora.nomic.beads import BeadType as NomicBeadType
from aragora.nomic.convoys import Convoy as NomicConvoy
from aragora.nomic.convoys import ConvoyStatus as NomicConvoyStatus

WorkspaceBeadT = TypeVar("WorkspaceBeadT")
WorkspaceConvoyT = TypeVar("WorkspaceConvoyT")


_WORKSPACE_BEAD_TO_NOMIC = {
    "pending": NomicBeadStatus.PENDING,
    "assigned": NomicBeadStatus.CLAIMED,
    "running": NomicBeadStatus.RUNNING,
    "done": NomicBeadStatus.COMPLETED,
    "failed": NomicBeadStatus.FAILED,
    "skipped": NomicBeadStatus.CANCELLED,
}

_NOMIC_TO_WORKSPACE_BEAD = {
    NomicBeadStatus.PENDING: "pending",
    NomicBeadStatus.CLAIMED: "assigned",
    NomicBeadStatus.RUNNING: "running",
    NomicBeadStatus.COMPLETED: "done",
    NomicBeadStatus.FAILED: "failed",
    NomicBeadStatus.CANCELLED: "skipped",
    NomicBeadStatus.BLOCKED: "pending",
}

_WORKSPACE_CONVOY_TO_NOMIC = {
    "created": NomicConvoyStatus.PENDING,
    "assigning": NomicConvoyStatus.ACTIVE,
    "executing": NomicConvoyStatus.ACTIVE,
    "merging": NomicConvoyStatus.ACTIVE,
    "done": NomicConvoyStatus.COMPLETED,
    "failed": NomicConvoyStatus.FAILED,
    "cancelled": NomicConvoyStatus.CANCELLED,
}

_NOMIC_TO_WORKSPACE_CONVOY = {
    NomicConvoyStatus.PENDING: "created",
    NomicConvoyStatus.ACTIVE: "executing",
    NomicConvoyStatus.COMPLETED: "done",
    NomicConvoyStatus.FAILED: "failed",
    NomicConvoyStatus.CANCELLED: "cancelled",
    NomicConvoyStatus.PARTIAL: "executing",
}


def workspace_bead_status_to_nomic(status: Any) -> NomicBeadStatus:
    value = status.value if hasattr(status, "value") else str(status)
    return _WORKSPACE_BEAD_TO_NOMIC.get(value, NomicBeadStatus.PENDING)


def resolve_workspace_bead_status(
    nomic_status: NomicBeadStatus,
    metadata: dict[str, Any],
    status_cls: type,
) -> Any:
    stored = metadata.get("workspace_status")
    if isinstance(stored, str):
        try:
            return status_cls(stored)
        except (ValueError, KeyError) as e:
            logger.debug("Invalid workspace bead status '%s': %s", stored, e)
    return status_cls(_NOMIC_TO_WORKSPACE_BEAD.get(nomic_status, "pending"))


def workspace_bead_metadata(bead: Any) -> dict[str, Any]:
    return {
        "workspace_id": bead.workspace_id,
        "convoy_id": bead.convoy_id,
        "payload": bead.payload,
        "result": bead.result,
        "git_ref": bead.git_ref,
        "started_at": bead.started_at,
        "completed_at": bead.completed_at,
        "workspace_status": bead.status.value,
        "metadata": bead.metadata,
    }


def workspace_bead_to_nomic(bead: Any) -> NomicBead:
    created_at = datetime.fromtimestamp(bead.created_at, tz=timezone.utc)
    updated_at = datetime.fromtimestamp(bead.updated_at, tz=timezone.utc)
    completed_at = (
        datetime.fromtimestamp(bead.completed_at, tz=timezone.utc) if bead.completed_at else None
    )
    return NomicBead(
        id=bead.bead_id,
        bead_type=NomicBeadType.TASK,
        status=workspace_bead_status_to_nomic(bead.status),
        title=bead.title,
        description=bead.description,
        created_at=created_at,
        updated_at=updated_at,
        claimed_by=bead.assigned_agent,
        claimed_at=None,
        completed_at=completed_at,
        parent_id=None,
        dependencies=list(bead.depends_on),
        metadata=workspace_bead_metadata(bead),
    )


def nomic_bead_to_workspace(
    bead: NomicBead,
    *,
    bead_cls: type[WorkspaceBeadT],
    status_cls: type,
) -> WorkspaceBeadT:
    metadata = bead.metadata or {}
    status = resolve_workspace_bead_status(bead.status, metadata, status_cls)
    # TypeVar bound cls: mypy cannot verify constructor args for generic type parameter
    return cast(Any, bead_cls)(
        bead_id=bead.id,
        convoy_id=metadata.get("convoy_id", ""),
        workspace_id=metadata.get("workspace_id", ""),
        title=bead.title,
        description=bead.description,
        status=status,
        assigned_agent=bead.claimed_by,
        payload=metadata.get("payload", {}),
        result=metadata.get("result"),
        error=bead.error_message,
        created_at=bead.created_at.timestamp(),
        updated_at=bead.updated_at.timestamp(),
        started_at=metadata.get("started_at"),
        completed_at=metadata.get("completed_at"),
        git_ref=metadata.get("git_ref"),
        depends_on=list(bead.dependencies),
        metadata=metadata.get("metadata", {}),
    )


def workspace_convoy_status_to_nomic(status: Any) -> NomicConvoyStatus:
    value = status.value if hasattr(status, "value") else str(status)
    return _WORKSPACE_CONVOY_TO_NOMIC.get(value, NomicConvoyStatus.PENDING)


def resolve_workspace_convoy_status(
    nomic_status: NomicConvoyStatus,
    metadata: dict[str, Any],
    status_cls: type,
) -> Any:
    stored = metadata.get("workspace_status")
    if isinstance(stored, str):
        try:
            return status_cls(stored)
        except (ValueError, KeyError) as e:
            logger.debug("Invalid workspace convoy status '%s': %s", stored, e)
    return status_cls(_NOMIC_TO_WORKSPACE_CONVOY.get(nomic_status, "created"))


def workspace_convoy_metadata(
    *,
    workspace_id: str,
    rig_id: str,
    name: str,
    description: str,
    status: Any,
    merge_result: dict[str, Any] | None = None,
    error: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(extra_metadata or {})
    metadata.update(
        {
            "workspace_id": workspace_id,
            "rig_id": rig_id,
            "workspace_name": name,
            "workspace_description": description,
            "workspace_status": status.value if hasattr(status, "value") else str(status),
        }
    )
    if merge_result is not None:
        metadata["merge_result"] = merge_result
    if error:
        metadata["error"] = error
    return metadata


def nomic_convoy_to_workspace(
    convoy: NomicConvoy,
    *,
    convoy_cls: type[WorkspaceConvoyT],
    status_cls: type,
) -> WorkspaceConvoyT:
    metadata = convoy.metadata or {}
    status = resolve_workspace_convoy_status(convoy.status, metadata, status_cls)
    # TypeVar bound cls: mypy cannot verify constructor args for generic type parameter
    return cast(Any, convoy_cls)(
        convoy_id=convoy.id,
        workspace_id=metadata.get("workspace_id", ""),
        rig_id=metadata.get("rig_id", ""),
        name=metadata.get("workspace_name", convoy.title),
        description=metadata.get("workspace_description", convoy.description),
        status=status,
        bead_ids=list(convoy.bead_ids),
        assigned_agents=list(convoy.assigned_to),
        created_at=convoy.created_at.timestamp(),
        updated_at=convoy.updated_at.timestamp(),
        started_at=convoy.started_at.timestamp() if convoy.started_at else None,
        completed_at=convoy.completed_at.timestamp() if convoy.completed_at else None,
        merge_result=metadata.get("merge_result"),
        error=convoy.error_message or metadata.get("error"),
        metadata=metadata if isinstance(metadata, dict) else {},
    )
