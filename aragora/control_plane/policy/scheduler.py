"""
Control Plane Continuous Policy Sync Scheduler.

Background scheduler for continuous policy synchronization.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from aragora.observability import get_logger

from .cache import RedisPolicyCache
from .conflicts import PolicyConflict, PolicyConflictDetector
from .manager import ControlPlanePolicyManager

logger = get_logger(__name__)


class PolicySyncScheduler:
    """
    Background scheduler for continuous policy synchronization.

    Periodically syncs policies from the compliance store and control plane
    policy store, ensuring the in-memory policy manager stays up-to-date
    with persistent storage.

    Also runs conflict detection after each sync and can optionally
    invalidate the policy cache when changes are detected.

    Usage:
        manager = ControlPlanePolicyManager()
        cache = RedisPolicyCache(...)
        scheduler = PolicySyncScheduler(
            policy_manager=manager,
            policy_cache=cache,
            sync_interval_seconds=60,
        )

        # Start background sync
        await scheduler.start()

        # ... application runs ...

        # Stop on shutdown
        await scheduler.stop()
    """

    def __init__(
        self,
        policy_manager: ControlPlanePolicyManager,
        sync_interval_seconds: float = 60.0,
        policy_cache: Optional[RedisPolicyCache] = None,
        conflict_callback: Optional[Callable[[List[PolicyConflict]], None]] = None,
        sync_from_compliance_store: bool = True,
        sync_from_control_plane_store: bool = True,
        workspace_id: Optional[str] = None,
    ):
        """
        Initialize the sync scheduler.

        Args:
            policy_manager: The policy manager to sync
            sync_interval_seconds: Interval between sync operations
            policy_cache: Optional cache to invalidate on policy changes
            conflict_callback: Callback invoked when conflicts are detected
            sync_from_compliance_store: Whether to sync from compliance store
            sync_from_control_plane_store: Whether to sync from control plane store
            workspace_id: Optional workspace filter for sync
        """
        self._policy_manager = policy_manager
        self._sync_interval = sync_interval_seconds
        self._policy_cache = policy_cache
        self._conflict_callback = conflict_callback
        self._sync_from_compliance = sync_from_compliance_store
        self._sync_from_cp_store = sync_from_control_plane_store
        self._workspace_id = workspace_id

        self._conflict_detector = PolicyConflictDetector()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_sync: Optional[datetime] = None
        self._last_policy_hash: Optional[str] = None
        self._sync_count = 0
        self._error_count = 0
        self._detected_conflicts: List[PolicyConflict] = []

    async def start(self) -> None:
        """Start the background sync task."""
        if self._running:
            logger.warning("Policy sync scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info(
            "policy_sync_scheduler_started",
            interval_seconds=self._sync_interval,
            workspace_id=self._workspace_id,
        )

    async def stop(self) -> None:
        """Stop the background sync task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("policy_sync_scheduler_stopped")

    async def sync_now(self) -> Dict[str, Any]:
        """
        Trigger an immediate sync.

        Returns:
            Sync result with counts and any detected conflicts
        """
        return await self._do_sync()

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        while self._running:
            try:
                await self._do_sync()
            except asyncio.CancelledError:
                # Don't swallow cancellation - allow graceful shutdown
                raise
            except Exception as e:
                self._error_count += 1
                logger.error("policy_sync_error", error=str(e))

            await asyncio.sleep(self._sync_interval)

    async def _do_sync(self) -> Dict[str, Any]:
        """Perform a single sync operation."""
        self._sync_count += 1
        synced_compliance = 0
        synced_cp_store = 0
        changes_detected = False

        # Sync from compliance store
        if self._sync_from_compliance:
            try:
                synced_compliance = self._policy_manager.sync_from_compliance_store(
                    workspace_id=self._workspace_id,
                    replace=False,  # Don't replace, merge
                )
            except (ImportError, OSError, ConnectionError, TimeoutError) as e:
                logger.warning("compliance_store_sync_failed", error=str(e))

        # Sync from control plane policy store
        if self._sync_from_cp_store:
            try:
                synced_cp_store = self._policy_manager.sync_from_store(
                    workspace=self._workspace_id,
                )
            except (ImportError, OSError, ConnectionError, TimeoutError) as e:
                logger.warning("control_plane_store_sync_failed", error=str(e))

        # Calculate hash after sync (includes manually added policies)
        current_hash = self._compute_policy_hash()

        # Detect changes by comparing to PREVIOUS sync's hash
        # This catches both: changes from store sync AND manual policy additions
        if self._last_policy_hash is not None:
            changes_detected = self._last_policy_hash != current_hash
        else:
            # First sync - no previous hash to compare
            changes_detected = False

        # Invalidate cache if changes detected
        if changes_detected and self._policy_cache:
            await self._policy_cache.invalidate_all()

        # Run conflict detection
        policies = self._policy_manager.list_policies(enabled_only=True)
        self._detected_conflicts = self._conflict_detector.detect_conflicts(policies)

        if self._detected_conflicts:
            logger.warning(
                "policy_conflicts_detected",
                count=len(self._detected_conflicts),
                conflicts=[c.to_dict() for c in self._detected_conflicts[:5]],  # Log first 5
            )

            if self._conflict_callback:
                try:
                    self._conflict_callback(self._detected_conflicts)
                except Exception as e:  # User callback - any exception possible
                    logger.warning(
                        "conflict_callback_error",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        self._last_sync = datetime.now(timezone.utc)
        self._last_policy_hash = current_hash

        result = {
            "synced_from_compliance": synced_compliance,
            "synced_from_cp_store": synced_cp_store,
            "changes_detected": changes_detected,
            "conflicts_detected": len(self._detected_conflicts),
            "total_policies": len(policies),
            "sync_time": self._last_sync.isoformat(),
        }

        logger.info("policy_sync_completed", **result)
        return result

    def _compute_policy_hash(self) -> str:
        """Compute a hash of current policy state for change detection."""
        policies = self._policy_manager.list_policies(enabled_only=False)
        policy_data = sorted([p.to_dict() for p in policies], key=lambda p: p["id"])
        data_str = json.dumps(policy_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        return {
            "running": self._running,
            "sync_interval_seconds": self._sync_interval,
            "sync_count": self._sync_count,
            "error_count": self._error_count,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "last_policy_hash": self._last_policy_hash,
            "detected_conflicts": len(self._detected_conflicts),
            "workspace_id": self._workspace_id,
        }

    def get_conflicts(self) -> List[PolicyConflict]:
        """Get currently detected conflicts."""
        return self._detected_conflicts.copy()

    @property
    def policy_version(self) -> Optional[str]:
        """Get the current policy version hash (for cache keys)."""
        return self._last_policy_hash
