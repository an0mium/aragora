"""
Refinery - Merge queue for convoy work integration.

The Refinery manages the orderly merging of completed convoy work back
into the main branch. This is a key Gastown concept for coordinating
parallel agent work.

Key responsibilities:
- Queue completed convoys for merge
- Validate merge readiness (all beads done, tests pass)
- Handle merge conflicts with automatic rebase or escalation
- Track merge history and rollback support

Usage:
    from aragora.workspace import Refinery

    refinery = Refinery()
    await refinery.queue_for_merge("convoy-123")
    merged = await refinery.process_queue()
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MergeStatus(Enum):
    """Status of a merge request."""

    QUEUED = "queued"
    VALIDATING = "validating"
    REBASING = "rebasing"
    MERGING = "merging"
    MERGED = "merged"
    CONFLICT = "conflict"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ConflictResolution(Enum):
    """How to handle merge conflicts."""

    AUTO_REBASE = "auto_rebase"
    MANUAL = "manual"
    ABORT = "abort"
    THEIRS = "theirs"
    OURS = "ours"


@dataclass
class MergeRequest:
    """A request to merge convoy work into the target branch."""

    convoy_id: str
    rig_id: str
    source_branch: str
    target_branch: str = "main"
    request_id: str = ""
    status: MergeStatus = MergeStatus.QUEUED
    priority: int = 0  # Higher = more urgent
    queued_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    conflict_files: list[str] = field(default_factory=list)
    merge_commit: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"merge-{uuid.uuid4().hex[:8]}"
        if not self.queued_at:
            self.queued_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "convoy_id": self.convoy_id,
            "rig_id": self.rig_id,
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "request_id": self.request_id,
            "status": self.status.value,
            "priority": self.priority,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "conflict_files": self.conflict_files,
            "merge_commit": self.merge_commit,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeRequest:
        """Deserialize from dictionary."""
        return cls(
            convoy_id=data["convoy_id"],
            rig_id=data["rig_id"],
            source_branch=data["source_branch"],
            target_branch=data.get("target_branch", "main"),
            request_id=data.get("request_id", ""),
            status=MergeStatus(data.get("status", "queued")),
            priority=data.get("priority", 0),
            queued_at=data.get("queued_at", 0.0),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            conflict_files=data.get("conflict_files", []),
            merge_commit=data.get("merge_commit"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RefineryConfig:
    """Configuration for the Refinery."""

    max_concurrent_merges: int = 1  # Usually 1 to avoid conflicts
    auto_rebase: bool = True
    require_tests: bool = True
    require_review: bool = False
    conflict_resolution: ConflictResolution = ConflictResolution.AUTO_REBASE
    retry_on_conflict: int = 3
    merge_timeout_seconds: float = 300.0
    rollback_on_test_failure: bool = True


class Refinery:
    """
    Merge queue manager for convoy work integration.

    The Refinery ensures orderly merging of parallel agent work,
    handling conflicts and maintaining codebase integrity.
    """

    def __init__(self, config: RefineryConfig | None = None):
        """Initialize the Refinery."""
        self.config = config or RefineryConfig()
        self._queue: list[MergeRequest] = []
        self._active: dict[str, MergeRequest] = {}
        self._history: list[MergeRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False

    async def queue_for_merge(
        self,
        convoy_id: str,
        rig_id: str,
        source_branch: str,
        target_branch: str = "main",
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> MergeRequest:
        """
        Queue a convoy for merge.

        Args:
            convoy_id: ID of the completed convoy
            rig_id: ID of the rig that produced the work
            source_branch: Branch containing the convoy work
            target_branch: Branch to merge into (default: main)
            priority: Higher priority merges first
            metadata: Optional metadata

        Returns:
            The created merge request
        """
        request = MergeRequest(
            convoy_id=convoy_id,
            rig_id=rig_id,
            source_branch=source_branch,
            target_branch=target_branch,
            priority=priority,
            metadata=metadata or {},
        )

        async with self._lock:
            self._queue.append(request)
            # Keep queue sorted by priority (highest first), then by queued time
            self._queue.sort(key=lambda r: (-r.priority, r.queued_at))

        return request

    async def get_request(self, request_id: str) -> MergeRequest | None:
        """Get a merge request by ID."""
        async with self._lock:
            # Check queue
            for req in self._queue:
                if req.request_id == request_id:
                    return req
            # Check active
            if request_id in self._active:
                return self._active[request_id]
            # Check history
            for req in self._history:
                if req.request_id == request_id:
                    return req
        return None

    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a queued merge request.

        Returns True if cancelled, False if not found or already processing.
        """
        async with self._lock:
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    self._queue.pop(i)
                    return True
        return False

    async def get_queue(self) -> list[MergeRequest]:
        """Get all queued merge requests."""
        async with self._lock:
            return list(self._queue)

    async def get_active(self) -> list[MergeRequest]:
        """Get currently processing merge requests."""
        async with self._lock:
            return list(self._active.values())

    async def get_history(
        self, limit: int = 100, status: MergeStatus | None = None
    ) -> list[MergeRequest]:
        """Get merge history."""
        async with self._lock:
            history = self._history
            if status:
                history = [r for r in history if r.status == status]
            return history[-limit:]

    async def process_next(self) -> MergeRequest | None:
        """
        Process the next merge request in the queue.

        Returns the merge request if one was processed, None if queue is empty.
        """
        async with self._lock:
            if not self._queue:
                return None
            if len(self._active) >= self.config.max_concurrent_merges:
                return None

            request = self._queue.pop(0)
            request.status = MergeStatus.VALIDATING
            request.started_at = time.time()
            self._active[request.request_id] = request

        try:
            # Validation phase
            if self.config.require_tests:
                valid = await self._run_tests(request)
                if not valid:
                    await self._fail_request(request, "Tests failed")
                    return request

            if self.config.require_review:
                approved = await self._check_approval(request)
                if not approved:
                    await self._fail_request(request, "Review not approved")
                    return request

            # Rebase phase
            request.status = MergeStatus.REBASING
            rebase_success = await self._rebase(request)

            if not rebase_success:
                if self.config.conflict_resolution == ConflictResolution.MANUAL:
                    request.status = MergeStatus.CONFLICT
                    return request
                elif self.config.conflict_resolution == ConflictResolution.ABORT:
                    await self._fail_request(request, "Merge conflict - aborting")
                    return request

            # Merge phase
            request.status = MergeStatus.MERGING
            merge_commit = await self._do_merge(request)

            if merge_commit:
                request.status = MergeStatus.MERGED
                request.merge_commit = merge_commit
                request.completed_at = time.time()
            else:
                await self._fail_request(request, "Merge failed")

        except Exception as e:
            await self._fail_request(request, str(e))

        finally:
            async with self._lock:
                if request.request_id in self._active:
                    del self._active[request.request_id]
                self._history.append(request)

        return request

    async def process_queue(self) -> list[MergeRequest]:
        """
        Process all queued merge requests.

        Returns list of processed requests.
        """
        processed = []
        self._processing = True

        try:
            while True:
                result = await self.process_next()
                if result is None:
                    break
                processed.append(result)
        finally:
            self._processing = False

        return processed

    async def rollback(self, request_id: str) -> bool:
        """
        Rollback a merged request.

        Returns True if rollback succeeded.
        """
        request = await self.get_request(request_id)
        if not request or request.status != MergeStatus.MERGED:
            return False

        if not request.merge_commit:
            return False

        # Perform git revert (stub - actual implementation would use subprocess)
        success = await self._do_rollback(request)

        if success:
            request.status = MergeStatus.ROLLED_BACK
            return True

        return False

    async def get_stats(self) -> dict[str, Any]:
        """Get refinery statistics."""
        async with self._lock:
            return {
                "queued": len(self._queue),
                "active": len(self._active),
                "total_merged": sum(1 for r in self._history if r.status == MergeStatus.MERGED),
                "total_failed": sum(1 for r in self._history if r.status == MergeStatus.FAILED),
                "total_conflicts": sum(
                    1 for r in self._history if r.status == MergeStatus.CONFLICT
                ),
                "total_rolled_back": sum(
                    1 for r in self._history if r.status == MergeStatus.ROLLED_BACK
                ),
                "processing": self._processing,
            }

    # Internal methods (stubs for actual git operations)

    async def _run_tests(self, request: MergeRequest) -> bool:
        """Run tests on the source branch. Stub implementation."""
        # In real implementation: run pytest, cargo test, etc.
        return True

    async def _check_approval(self, request: MergeRequest) -> bool:
        """Check if the merge has been approved. Stub implementation."""
        # In real implementation: check PR approval status
        return True

    async def _rebase(self, request: MergeRequest) -> bool:
        """
        Rebase source branch onto target. Stub implementation.

        Returns True if rebase succeeded, False if conflicts.
        """
        # In real implementation: git rebase
        return True

    async def _do_merge(self, request: MergeRequest) -> str | None:
        """
        Perform the actual merge. Stub implementation.

        Returns merge commit SHA if successful.
        """
        # In real implementation: git merge --no-ff
        return f"commit-{uuid.uuid4().hex[:7]}"

    async def _do_rollback(self, request: MergeRequest) -> bool:
        """Rollback a merge. Stub implementation."""
        # In real implementation: git revert
        return True

    async def _fail_request(self, request: MergeRequest, error: str):
        """Mark a request as failed."""
        request.status = MergeStatus.FAILED
        request.error = error
        request.completed_at = time.time()
