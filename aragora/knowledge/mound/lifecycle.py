"""
Knowledge Lifecycle Management for Knowledge Mound.

Provides comprehensive lifecycle management:
- Retention policies (how long to keep knowledge)
- Archival management (move to cold storage)
- Cleanup of expired/stale knowledge
- Version history pruning
- Lifecycle reporting and analytics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from uuid import uuid4

from aragora.observability import get_logger

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound
    from aragora.knowledge.mound.staleness import StalenessDetector

logger = get_logger(__name__)


class LifecycleStage(str, Enum):
    """Stages in the knowledge lifecycle."""

    ACTIVE = "active"  # In use, frequently accessed
    WARM = "warm"  # Less frequently accessed
    COLD = "cold"  # Rarely accessed, archived
    EXPIRED = "expired"  # Past retention, pending deletion
    DELETED = "deleted"  # Marked for deletion


class RetentionAction(str, Enum):
    """Actions for retention policy."""

    KEEP = "keep"  # Keep the knowledge
    ARCHIVE = "archive"  # Move to cold storage
    DELETE = "delete"  # Delete permanently
    REVALIDATE = "revalidate"  # Trigger revalidation


@dataclass
class RetentionPolicy:
    """Defines retention rules for knowledge."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Target (what this policy applies to)
    workspace_ids: List[str] = field(default_factory=list)  # Empty = all
    knowledge_types: List[str] = field(default_factory=list)  # Empty = all
    tiers: List[str] = field(default_factory=list)  # fast, medium, slow, glacial

    # Retention periods
    active_period: timedelta = timedelta(days=30)  # How long to keep active
    warm_period: timedelta = timedelta(days=90)  # How long in warm storage
    cold_period: timedelta = timedelta(days=365)  # How long in cold storage
    max_age: Optional[timedelta] = None  # Maximum total age (delete after)

    # Access-based retention
    min_access_count_for_keep: int = 0  # Minimum accesses to avoid archival
    last_access_threshold: timedelta = timedelta(days=30)  # Archive if not accessed

    # Quality-based retention
    min_confidence_for_keep: float = 0.0  # Archive if confidence drops below
    min_validation_score: float = 0.0  # Archive if validation fails

    # Version history
    max_versions_to_keep: int = 10  # Maximum versions per knowledge item
    version_retention_days: int = 90  # How long to keep old versions

    # Priority and status
    priority: int = 0
    enabled: bool = True

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        workspace_id: Optional[str] = None,
        knowledge_type: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> bool:
        """Check if this policy applies to the given knowledge."""
        if not self.enabled:
            return False

        if self.workspace_ids and workspace_id not in self.workspace_ids:
            return False

        if self.knowledge_types and knowledge_type not in self.knowledge_types:
            return False

        if self.tiers and tier not in self.tiers:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "workspace_ids": self.workspace_ids,
            "knowledge_types": self.knowledge_types,
            "tiers": self.tiers,
            "active_period_days": self.active_period.days,
            "warm_period_days": self.warm_period.days,
            "cold_period_days": self.cold_period.days,
            "max_age_days": self.max_age.days if self.max_age else None,
            "min_access_count_for_keep": self.min_access_count_for_keep,
            "min_confidence_for_keep": self.min_confidence_for_keep,
            "max_versions_to_keep": self.max_versions_to_keep,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class LifecycleTransition:
    """Record of a lifecycle stage transition."""

    id: str = field(default_factory=lambda: str(uuid4()))
    knowledge_id: str = ""
    from_stage: LifecycleStage = LifecycleStage.ACTIVE
    to_stage: LifecycleStage = LifecycleStage.WARM
    reason: str = ""
    policy_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "knowledge_id": self.knowledge_id,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class LifecycleReport:
    """Report on knowledge lifecycle status."""

    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_knowledge_items: int = 0

    # Stage distribution
    active_count: int = 0
    warm_count: int = 0
    cold_count: int = 0
    expired_count: int = 0

    # Age distribution
    items_by_age_bucket: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    avg_confidence: float = 0.0
    stale_count: int = 0
    needs_revalidation_count: int = 0

    # Space usage
    total_size_bytes: int = 0
    archived_size_bytes: int = 0

    # Recent transitions
    recent_archives: int = 0
    recent_deletions: int = 0
    recent_revalidations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_knowledge_items": self.total_knowledge_items,
            "stage_distribution": {
                "active": self.active_count,
                "warm": self.warm_count,
                "cold": self.cold_count,
                "expired": self.expired_count,
            },
            "items_by_age_bucket": self.items_by_age_bucket,
            "quality_metrics": {
                "avg_confidence": self.avg_confidence,
                "stale_count": self.stale_count,
                "needs_revalidation_count": self.needs_revalidation_count,
            },
            "space_usage": {
                "total_size_bytes": self.total_size_bytes,
                "archived_size_bytes": self.archived_size_bytes,
            },
            "recent_activity": {
                "archives": self.recent_archives,
                "deletions": self.recent_deletions,
                "revalidations": self.recent_revalidations,
            },
        }


class LifecycleManager:
    """
    Manages the lifecycle of knowledge in the Knowledge Mound.

    Features:
    - Retention policy enforcement
    - Automatic archival based on age/access patterns
    - Cleanup of expired knowledge
    - Version history management
    - Lifecycle reporting
    """

    def __init__(
        self,
        mound: Optional["KnowledgeMound"] = None,
        staleness_detector: Optional["StalenessDetector"] = None,
    ):
        """
        Initialize lifecycle manager.

        Args:
            mound: Knowledge Mound instance
            staleness_detector: Staleness detector for stale knowledge
        """
        self._mound = mound
        self._staleness_detector = staleness_detector

        # Retention policies
        self._policies: Dict[str, RetentionPolicy] = {}

        # Lifecycle state (knowledge_id -> stage)
        self._stages: Dict[str, LifecycleStage] = {}

        # Transition history (limited to recent)
        self._transitions: List[LifecycleTransition] = []
        self._max_transitions = 1000

        # Access tracking (knowledge_id -> (last_access, access_count))
        self._access_log: Dict[str, Dict[str, Any]] = {}

        # Callbacks
        self._archive_callbacks: List[Callable[[str, str], None]] = []
        self._delete_callbacks: List[Callable[[str, str], None]] = []

        logger.info("LifecycleManager initialized")

    def set_mound(self, mound: "KnowledgeMound") -> None:
        """Set the Knowledge Mound instance."""
        self._mound = mound

    def add_policy(self, policy: RetentionPolicy) -> None:
        """Add a retention policy."""
        self._policies[policy.id] = policy
        logger.info(
            "retention_policy_added",
            policy_id=policy.id,
            policy_name=policy.name,
        )

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a retention policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            logger.info("retention_policy_removed", policy_id=policy_id)
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(self, enabled_only: bool = True) -> List[RetentionPolicy]:
        """List all retention policies."""
        policies = []
        for policy in self._policies.values():
            if enabled_only and not policy.enabled:
                continue
            policies.append(policy)
        return sorted(policies, key=lambda p: p.priority, reverse=True)

    def record_access(self, knowledge_id: str) -> None:
        """Record an access to knowledge for lifecycle tracking."""
        now = datetime.now(timezone.utc)

        if knowledge_id not in self._access_log:
            self._access_log[knowledge_id] = {
                "first_access": now,
                "last_access": now,
                "access_count": 0,
            }

        self._access_log[knowledge_id]["last_access"] = now
        self._access_log[knowledge_id]["access_count"] += 1

    def get_stage(self, knowledge_id: str) -> LifecycleStage:
        """Get the current lifecycle stage of knowledge."""
        return self._stages.get(knowledge_id, LifecycleStage.ACTIVE)

    def set_stage(
        self,
        knowledge_id: str,
        stage: LifecycleStage,
        reason: str = "",
        policy_id: Optional[str] = None,
    ) -> LifecycleTransition:
        """Set the lifecycle stage of knowledge."""
        old_stage = self._stages.get(knowledge_id, LifecycleStage.ACTIVE)

        transition = LifecycleTransition(
            knowledge_id=knowledge_id,
            from_stage=old_stage,
            to_stage=stage,
            reason=reason,
            policy_id=policy_id,
        )

        self._stages[knowledge_id] = stage
        self._transitions.append(transition)

        # Keep transitions bounded
        if len(self._transitions) > self._max_transitions:
            self._transitions = self._transitions[-self._max_transitions :]

        logger.info(
            "lifecycle_transition",
            knowledge_id=knowledge_id,
            from_stage=old_stage.value,
            to_stage=stage.value,
            reason=reason,
        )

        # Trigger callbacks
        if stage == LifecycleStage.COLD:
            for callback in self._archive_callbacks:
                try:
                    callback(knowledge_id, reason)
                except Exception as e:
                    logger.error(f"Archive callback error: {e}")
        elif stage == LifecycleStage.DELETED:
            for callback in self._delete_callbacks:
                try:
                    callback(knowledge_id, reason)
                except Exception as e:
                    logger.error(f"Delete callback error: {e}")

        return transition

    def evaluate_retention(
        self,
        knowledge_id: str,
        knowledge_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        tier: Optional[str] = None,
        created_at: Optional[datetime] = None,
        confidence: float = 1.0,
    ) -> RetentionAction:
        """
        Evaluate what retention action should be taken for knowledge.

        Args:
            knowledge_id: ID of the knowledge
            knowledge_type: Type of knowledge
            workspace_id: Workspace ID
            tier: Memory tier
            created_at: When the knowledge was created
            confidence: Current confidence score

        Returns:
            Recommended retention action
        """
        now = datetime.now(timezone.utc)

        # Get applicable policies
        applicable_policies = []
        for policy in self._policies.values():
            if policy.matches(
                workspace_id=workspace_id,
                knowledge_type=knowledge_type,
                tier=tier,
            ):
                applicable_policies.append(policy)

        if not applicable_policies:
            return RetentionAction.KEEP

        # Use highest priority policy
        policy = max(applicable_policies, key=lambda p: p.priority)

        # Check age
        age = now - created_at if created_at else timedelta(0)

        # Check max age
        if policy.max_age and age > policy.max_age:
            return RetentionAction.DELETE

        # Check confidence threshold
        if confidence < policy.min_confidence_for_keep:
            return RetentionAction.REVALIDATE

        # Check access patterns
        access_info = self._access_log.get(knowledge_id, {})
        last_access = access_info.get("last_access")
        access_count = access_info.get("access_count", 0)

        if last_access:
            time_since_access = now - last_access
            if time_since_access > policy.last_access_threshold:
                if access_count < policy.min_access_count_for_keep:
                    return RetentionAction.ARCHIVE

        # Check age-based transitions
        if age > policy.active_period + policy.warm_period + policy.cold_period:
            return RetentionAction.DELETE
        elif age > policy.active_period + policy.warm_period:
            return RetentionAction.ARCHIVE
        elif age > policy.active_period:
            # Move to warm (just log, don't archive yet)
            pass

        return RetentionAction.KEEP

    async def run_cleanup(
        self,
        workspace_id: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Run lifecycle cleanup based on retention policies.

        Args:
            workspace_id: Optional workspace filter
            dry_run: If True, don't actually delete/archive

        Returns:
            Cleanup report
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "evaluated": 0,
            "archived": 0,
            "deleted": 0,
            "revalidated": 0,
            "kept": 0,
            "errors": [],
        }

        # Get all knowledge items to evaluate
        # In a real implementation, this would query the mound
        knowledge_ids = list(self._stages.keys())

        for knowledge_id in knowledge_ids:
            report["evaluated"] += 1

            try:
                # Get current stage
                current_stage = self.get_stage(knowledge_id)

                # Skip already deleted
                if current_stage == LifecycleStage.DELETED:
                    continue

                # Evaluate retention (would need knowledge metadata in real impl)
                action = self.evaluate_retention(
                    knowledge_id=knowledge_id,
                    workspace_id=workspace_id,
                )

                if action == RetentionAction.DELETE:
                    report["deleted"] += 1
                    if not dry_run:
                        self.set_stage(
                            knowledge_id,
                            LifecycleStage.DELETED,
                            reason="Retention policy - max age exceeded",
                        )

                elif action == RetentionAction.ARCHIVE:
                    report["archived"] += 1
                    if not dry_run:
                        self.set_stage(
                            knowledge_id,
                            LifecycleStage.COLD,
                            reason="Retention policy - not accessed",
                        )

                elif action == RetentionAction.REVALIDATE:
                    report["revalidated"] += 1
                    # Would trigger revalidation in real implementation

                else:
                    report["kept"] += 1

            except Exception as e:
                report["errors"].append(
                    {
                        "knowledge_id": knowledge_id,
                        "error": str(e),
                    }
                )

        logger.info(
            "lifecycle_cleanup_completed",
            dry_run=dry_run,
            evaluated=report["evaluated"],
            archived=report["archived"],
            deleted=report["deleted"],
        )

        return report

    async def prune_versions(
        self,
        knowledge_id: str,
        max_versions: Optional[int] = None,
        older_than: Optional[timedelta] = None,
    ) -> int:
        """
        Prune old versions of knowledge.

        Args:
            knowledge_id: Knowledge to prune versions for
            max_versions: Maximum versions to keep (uses policy default if None)
            older_than: Delete versions older than this (uses policy default if None)

        Returns:
            Number of versions pruned
        """
        # Get applicable policy for defaults
        policy = None
        for p in self._policies.values():
            if p.enabled:
                policy = p
                break

        if max_versions is None:
            max_versions = policy.max_versions_to_keep if policy else 10

        if older_than is None:
            older_than = timedelta(days=policy.version_retention_days if policy else 90)

        # In a real implementation, this would query and delete versions
        # For now, just log
        logger.info(
            "version_prune_requested",
            knowledge_id=knowledge_id,
            max_versions=max_versions,
            older_than_days=older_than.days,
        )

        return 0

    def generate_report(
        self,
        workspace_id: Optional[str] = None,
    ) -> LifecycleReport:
        """
        Generate a lifecycle status report.

        Args:
            workspace_id: Optional workspace filter

        Returns:
            Lifecycle report
        """
        report = LifecycleReport()

        # Count by stage
        for knowledge_id, stage in self._stages.items():
            report.total_knowledge_items += 1

            if stage == LifecycleStage.ACTIVE:
                report.active_count += 1
            elif stage == LifecycleStage.WARM:
                report.warm_count += 1
            elif stage == LifecycleStage.COLD:
                report.cold_count += 1
            elif stage == LifecycleStage.EXPIRED:
                report.expired_count += 1

        # Count recent transitions
        now = datetime.now(timezone.utc)
        recent_cutoff = now - timedelta(days=7)

        for transition in self._transitions:
            if transition.timestamp >= recent_cutoff:
                if transition.to_stage == LifecycleStage.COLD:
                    report.recent_archives += 1
                elif transition.to_stage == LifecycleStage.DELETED:
                    report.recent_deletions += 1

        return report

    def get_transition_history(
        self,
        knowledge_id: Optional[str] = None,
        stage: Optional[LifecycleStage] = None,
        limit: int = 100,
    ) -> List[LifecycleTransition]:
        """
        Get transition history.

        Args:
            knowledge_id: Filter by knowledge ID
            stage: Filter by target stage
            limit: Maximum transitions to return

        Returns:
            List of transitions
        """
        result = []

        for transition in reversed(self._transitions):
            if knowledge_id and transition.knowledge_id != knowledge_id:
                continue
            if stage and transition.to_stage != stage:
                continue

            result.append(transition)

            if len(result) >= limit:
                break

        return result

    def add_archive_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for archive events."""
        self._archive_callbacks.append(callback)

    def add_delete_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for delete events."""
        self._delete_callbacks.append(callback)


# Factory function
def create_lifecycle_manager(
    mound: Optional["KnowledgeMound"] = None,
    staleness_detector: Optional["StalenessDetector"] = None,
) -> LifecycleManager:
    """Create a LifecycleManager instance."""
    return LifecycleManager(
        mound=mound,
        staleness_detector=staleness_detector,
    )
