"""
Control Plane Policy History.

Tracks policy version history for auditing and rollback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aragora.observability import get_logger

from .types import ControlPlanePolicy

logger = get_logger(__name__)


@dataclass
class PolicyVersion:
    """A snapshot of a policy at a specific version."""

    policy_id: str
    version: int
    policy_data: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    change_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "policy_data": self.policy_data,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "change_description": self.change_description,
        }


class PolicyHistory:
    """Tracks policy version history for auditing and rollback.

    Maintains a history of policy versions, enabling:
    - Viewing historical versions of any policy
    - Rolling back to previous versions
    - Audit trail of who changed what and when
    """

    def __init__(self, max_versions_per_policy: int = 50):
        """Initialize the policy history tracker.

        Args:
            max_versions_per_policy: Maximum versions to retain per policy
        """
        self._history: dict[str, list[PolicyVersion]] = {}
        self._max_versions = max_versions_per_policy
        self._lock = asyncio.Lock()

    async def record_version(
        self,
        policy: ControlPlanePolicy,
        change_description: str = "",
        changed_by: str | None = None,
    ) -> PolicyVersion:
        """Record a new version of a policy.

        Args:
            policy: The policy to record
            change_description: Description of what changed
            changed_by: User who made the change

        Returns:
            The recorded PolicyVersion
        """
        async with self._lock:
            policy_id = policy.id

            if policy_id not in self._history:
                self._history[policy_id] = []

            version = PolicyVersion(
                policy_id=policy_id,
                version=policy.version,
                policy_data=policy.to_dict(),
                created_by=changed_by,
                change_description=change_description,
            )

            self._history[policy_id].append(version)

            # Prune old versions
            if len(self._history[policy_id]) > self._max_versions:
                self._history[policy_id] = self._history[policy_id][-self._max_versions :]

            logger.info(
                "Policy version recorded: %s v%s by %s",
                policy.name,
                policy.version,
                changed_by or "system",
            )

            return version

    async def get_history(
        self,
        policy_id: str,
        limit: int = 10,
    ) -> list[PolicyVersion]:
        """Get version history for a policy.

        Args:
            policy_id: The policy ID
            limit: Maximum versions to return

        Returns:
            List of PolicyVersions, newest first
        """
        async with self._lock:
            versions = self._history.get(policy_id, [])
            return list(reversed(versions[-limit:]))

    async def get_version(
        self,
        policy_id: str,
        version: int,
    ) -> PolicyVersion | None:
        """Get a specific version of a policy.

        Args:
            policy_id: The policy ID
            version: The version number

        Returns:
            PolicyVersion if found, None otherwise
        """
        async with self._lock:
            versions = self._history.get(policy_id, [])
            for v in versions:
                if v.version == version:
                    return v
            return None

    async def rollback_to_version(
        self,
        policy_id: str,
        version: int,
        rolled_back_by: str | None = None,
    ) -> ControlPlanePolicy | None:
        """Restore a policy to a previous version.

        Args:
            policy_id: The policy ID
            version: The version to restore
            rolled_back_by: User performing the rollback

        Returns:
            New ControlPlanePolicy instance with restored data, or None if not found
        """
        target_version = await self.get_version(policy_id, version)
        if not target_version:
            logger.warning("Policy version not found: %s v%s", policy_id, version)
            return None

        # Get current version number
        history = self._history.get(policy_id, [])
        current_version = max((v.version for v in history), default=0)

        # Create new policy from historical data
        policy_data = target_version.policy_data.copy()
        policy_data["version"] = current_version + 1
        policy_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        policy_data["updated_by"] = rolled_back_by
        policy_data["previous_version_id"] = f"{policy_id}_v{current_version}"
        policy_data["metadata"]["rolled_back_from_version"] = version

        restored_policy = ControlPlanePolicy.from_dict(policy_data)

        # Record the rollback as a new version
        await self.record_version(
            restored_policy,
            change_description=f"Rollback to version {version}",
            changed_by=rolled_back_by,
        )

        logger.info(
            "Policy rolled back: %s from v%s to v%s (now v%s) by %s",
            policy_id,
            current_version,
            version,
            restored_policy.version,
            rolled_back_by or "system",
        )

        return restored_policy

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about policy history."""
        total_versions = sum(len(v) for v in self._history.values())
        return {
            "tracked_policies": len(self._history),
            "total_versions": total_versions,
            "max_versions_per_policy": self._max_versions,
            "policies": {policy_id: len(versions) for policy_id, versions in self._history.items()},
        }


# Global policy history instance
_policy_history: PolicyHistory | None = None


def get_policy_history() -> PolicyHistory:
    """Get the global policy history instance."""
    global _policy_history
    if _policy_history is None:
        _policy_history = PolicyHistory()
    return _policy_history
