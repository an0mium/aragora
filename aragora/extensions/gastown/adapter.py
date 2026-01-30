"""
Gastown compatibility adapter for canonical Nomic APIs.

This is a thin facade that keeps Gastown semantics while delegating
convoy persistence and status tracking to the canonical Nomic layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .convoy import ConvoyTracker
from .models import Convoy, ConvoyStatus


class GastownConvoyAdapter:
    """Adapter that bridges Gastown convoys to the canonical Nomic layer."""

    def __init__(
        self,
        tracker: ConvoyTracker | None = None,
        storage_path: str | Path | None = None,
    ) -> None:
        # ConvoyTracker already forwards to Nomic when storage is enabled.
        self._tracker = tracker or ConvoyTracker(
            storage_path=storage_path,
            use_nomic_store=True,
        )

    async def create_convoy(
        self,
        *,
        rig_id: str,
        title: str,
        description: str = "",
        issue_ref: str | None = None,
        parent_convoy: str | None = None,
        priority: int = 0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Convoy:
        return await self._tracker.create_convoy(
            rig_id=rig_id,
            title=title,
            description=description,
            issue_ref=issue_ref,
            parent_convoy=parent_convoy,
            priority=priority,
            tags=tags,
            metadata=metadata,
        )

    async def get_convoy(self, convoy_id: str) -> Convoy | None:
        return await self._tracker.get_convoy(convoy_id)

    async def list_convoys(
        self,
        *,
        rig_id: str | None = None,
        status: ConvoyStatus | None = None,
        agent_id: str | None = None,
    ) -> list[Convoy]:
        return await self._tracker.list_convoys(
            rig_id=rig_id,
            status=status,
            agent_id=agent_id,
        )
