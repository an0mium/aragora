"""Bidirectional Obsidian sync with conflict detection and resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SyncDirection(str, Enum):
    FORWARD = "vault_to_km"
    REVERSE = "km_to_vault"


class ConflictResolution(str, Enum):
    PREFER_VAULT_CONTENT = "prefer_vault_content"
    PREFER_KM_METADATA = "prefer_km_metadata"
    MERGE = "merge"


@dataclass
class SyncRecord:
    """Tracks sync state for a single note."""

    note_path: str
    vault_modified: datetime | None = None
    km_modified: datetime | None = None
    last_synced: datetime | None = None


@dataclass
class SyncConflict:
    """Detected conflict between vault and KM."""

    note_path: str
    vault_modified: datetime
    km_modified: datetime
    resolution: ConflictResolution | None = None


class BidirectionalSyncManager:
    """Manages bidirectional sync between Obsidian vault and Knowledge Mound.

    Strategy: user edits to note content take precedence.
    KM validation results are appended as frontmatter (never overwrite content).
    Conflicts are logged to decision receipt for audit.
    """

    def __init__(
        self,
        connector: Any,
        adapter: Any,
        resolution_strategy: ConflictResolution = ConflictResolution.PREFER_VAULT_CONTENT,
    ) -> None:
        self._connector = connector
        self._adapter = adapter
        self._resolution_strategy = resolution_strategy
        self._sync_records: dict[str, SyncRecord] = {}
        self._conflicts: list[SyncConflict] = []

    def detect_conflict(self, record: SyncRecord) -> SyncConflict | None:
        """Detect if both vault and KM changed since last sync."""
        if record.last_synced is None:
            return None

        vault_changed = (
            record.vault_modified is not None and record.vault_modified > record.last_synced
        )
        km_changed = record.km_modified is not None and record.km_modified > record.last_synced

        if vault_changed and km_changed:
            conflict = SyncConflict(
                note_path=record.note_path,
                vault_modified=record.vault_modified,
                km_modified=record.km_modified,
            )
            self._conflicts.append(conflict)
            return conflict

        return None

    def resolve_conflict(self, conflict: SyncConflict) -> ConflictResolution:
        """Apply resolution strategy to a conflict."""
        conflict.resolution = self._resolution_strategy
        logger.info(
            "Resolved conflict for %s: %s",
            conflict.note_path,
            self._resolution_strategy.value,
        )
        return self._resolution_strategy

    async def sync_forward(self, **kwargs: Any) -> Any:
        """Sync vault -> Knowledge Mound."""
        return await self._adapter.sync_to_km(**kwargs)

    async def sync_reverse(self, **kwargs: Any) -> Any:
        """Sync Knowledge Mound -> vault (frontmatter only)."""
        return await self._adapter.sync_from_km(**kwargs)

    async def sync_bidirectional(self, **kwargs: Any) -> dict[str, Any]:
        """Run full bidirectional sync with conflict detection."""
        forward_result = await self.sync_forward(**kwargs)
        reverse_result = await self.sync_reverse(**kwargs)
        return {
            "forward": forward_result,
            "reverse": reverse_result,
            "conflicts": [
                {
                    "path": c.note_path,
                    "resolution": c.resolution.value if c.resolution else None,
                }
                for c in self._conflicts
            ],
        }
