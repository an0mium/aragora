"""
Cost Attribution System for granular cost tracking.

Provides cost attribution at multiple levels:
- Per-user cost tracking
- Per-task cost attribution
- Per-debate cost breakdown
- Workspace and organization rollups
- Chargeback/showback reporting
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from aragora.observability import get_logger

if TYPE_CHECKING:
    from aragora.billing.cost_tracker import CostTracker

logger = get_logger(__name__)


class AttributionLevel(str, Enum):
    """Levels of cost attribution."""

    USER = "user"
    TASK = "task"
    DEBATE = "debate"
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"
    TEAM = "team"
    PROJECT = "project"


class AllocationMethod(str, Enum):
    """Methods for allocating shared costs."""

    DIRECT = "direct"  # Attribute directly to entity that incurred cost
    PROPORTIONAL = "proportional"  # Distribute based on usage proportion
    EQUAL = "equal"  # Split equally among entities
    WEIGHTED = "weighted"  # Use custom weights


@dataclass
class CostAllocation:
    """A cost allocation to an entity."""

    entity_id: str
    entity_type: AttributionLevel
    cost_usd: Decimal
    tokens_in: int = 0
    tokens_out: int = 0
    api_calls: int = 0
    allocation_method: AllocationMethod = AllocationMethod.DIRECT
    allocation_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionEntry:
    """A single cost attribution entry."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Source of cost
    source_type: str = ""  # "api_call", "debate_round", "task_execution"
    source_id: str = ""

    # Cost details
    cost_usd: Decimal = Decimal("0")
    tokens_in: int = 0
    tokens_out: int = 0

    # Provider info
    provider: str = ""
    model: str = ""
    agent_id: str = ""
    agent_name: str = ""

    # Attribution targets
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    debate_id: Optional[str] = None
    workspace_id: Optional[str] = None
    org_id: Optional[str] = None
    team_id: Optional[str] = None
    project_id: Optional[str] = None

    # Additional context
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type,
            "source_id": self.source_id,
            "cost_usd": str(self.cost_usd),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "provider": self.provider,
            "model": self.model,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "debate_id": self.debate_id,
            "workspace_id": self.workspace_id,
            "org_id": self.org_id,
            "team_id": self.team_id,
            "project_id": self.project_id,
            "operation": self.operation,
            "metadata": self.metadata,
        }


@dataclass
class AttributionSummary:
    """Summary of costs for an entity."""

    entity_id: str
    entity_type: AttributionLevel
    period_start: datetime
    period_end: datetime

    # Totals
    total_cost_usd: Decimal = Decimal("0")
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_api_calls: int = 0

    # Breakdowns
    cost_by_model: Dict[str, Decimal] = field(default_factory=dict)
    cost_by_provider: Dict[str, Decimal] = field(default_factory=dict)
    cost_by_agent: Dict[str, Decimal] = field(default_factory=dict)
    cost_by_operation: Dict[str, Decimal] = field(default_factory=dict)

    # Time series
    daily_costs: List[Dict[str, Any]] = field(default_factory=list)

    # Derived metrics
    avg_cost_per_call: Decimal = Decimal("0")
    avg_tokens_per_call: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_cost_usd": str(self.total_cost_usd),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_api_calls": self.total_api_calls,
            "cost_by_model": {k: str(v) for k, v in self.cost_by_model.items()},
            "cost_by_provider": {k: str(v) for k, v in self.cost_by_provider.items()},
            "cost_by_agent": {k: str(v) for k, v in self.cost_by_agent.items()},
            "cost_by_operation": {k: str(v) for k, v in self.cost_by_operation.items()},
            "daily_costs": self.daily_costs,
            "avg_cost_per_call": str(self.avg_cost_per_call),
            "avg_tokens_per_call": self.avg_tokens_per_call,
        }


@dataclass
class ChargebackReport:
    """Chargeback report for billing purposes."""

    id: str = field(default_factory=lambda: str(uuid4()))
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Scope
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # Totals
    total_cost_usd: Decimal = Decimal("0")

    # Allocations by entity
    allocations_by_user: Dict[str, CostAllocation] = field(default_factory=dict)
    allocations_by_team: Dict[str, CostAllocation] = field(default_factory=dict)
    allocations_by_project: Dict[str, CostAllocation] = field(default_factory=dict)

    # Shared costs
    shared_costs_usd: Decimal = Decimal("0")
    shared_cost_allocation_method: AllocationMethod = AllocationMethod.PROPORTIONAL

    # Metadata
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "total_cost_usd": str(self.total_cost_usd),
            "allocations_by_user": {
                k: {
                    "entity_id": v.entity_id,
                    "cost_usd": str(v.cost_usd),
                    "tokens_in": v.tokens_in,
                    "tokens_out": v.tokens_out,
                    "api_calls": v.api_calls,
                }
                for k, v in self.allocations_by_user.items()
            },
            "allocations_by_team": {
                k: {
                    "entity_id": v.entity_id,
                    "cost_usd": str(v.cost_usd),
                }
                for k, v in self.allocations_by_team.items()
            },
            "allocations_by_project": {
                k: {
                    "entity_id": v.entity_id,
                    "cost_usd": str(v.cost_usd),
                }
                for k, v in self.allocations_by_project.items()
            },
            "shared_costs_usd": str(self.shared_costs_usd),
            "shared_cost_allocation_method": self.shared_cost_allocation_method.value,
            "notes": self.notes,
        }


class CostAttributor:
    """
    Tracks and attributes costs at multiple levels.

    Features:
    - Per-user cost tracking
    - Per-task cost attribution
    - Hierarchical rollups (user -> team -> workspace -> org)
    - Chargeback/showback reporting
    - Cost allocation rules for shared resources
    """

    def __init__(
        self,
        cost_tracker: Optional["CostTracker"] = None,
        max_entries: int = 10000,
    ):
        """
        Initialize cost attributor.

        Args:
            cost_tracker: Optional CostTracker for integration
            max_entries: Maximum entries to keep in memory
        """
        self._cost_tracker = cost_tracker
        self._max_entries = max_entries

        # Attribution entries (most recent)
        self._entries: List[AttributionEntry] = []

        # Aggregated stats by entity
        self._user_costs: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_cost": Decimal("0"),
                "tokens_in": 0,
                "tokens_out": 0,
                "api_calls": 0,
                "by_model": defaultdict(lambda: Decimal("0")),
                "by_agent": defaultdict(lambda: Decimal("0")),
            }
        )
        self._task_costs: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_cost": Decimal("0"),
                "tokens_in": 0,
                "tokens_out": 0,
                "api_calls": 0,
                "user_id": None,
                "workspace_id": None,
            }
        )
        self._team_costs: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_cost": Decimal("0"),
                "members": set(),
            }
        )
        self._project_costs: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_cost": Decimal("0"),
                "tasks": set(),
            }
        )

        # User-to-team mapping
        self._user_teams: Dict[str, str] = {}

        # Task-to-project mapping
        self._task_projects: Dict[str, str] = {}

        logger.info("CostAttributor initialized", max_entries=max_entries)

    def set_cost_tracker(self, cost_tracker: "CostTracker") -> None:
        """Set cost tracker for integration."""
        self._cost_tracker = cost_tracker

    def set_user_team(self, user_id: str, team_id: str) -> None:
        """Set the team for a user (for rollup calculations)."""
        self._user_teams[user_id] = team_id
        self._team_costs[team_id]["members"].add(user_id)

    def set_task_project(self, task_id: str, project_id: str) -> None:
        """Set the project for a task (for rollup calculations)."""
        self._task_projects[task_id] = project_id
        self._project_costs[project_id]["tasks"].add(task_id)

    def record_cost(
        self,
        cost_usd: Decimal,
        tokens_in: int = 0,
        tokens_out: int = 0,
        provider: str = "",
        model: str = "",
        agent_id: str = "",
        agent_name: str = "",
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        debate_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        org_id: Optional[str] = None,
        operation: str = "",
        source_type: str = "api_call",
        source_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttributionEntry:
        """
        Record a cost and attribute it to relevant entities.

        Args:
            cost_usd: Cost in USD
            tokens_in: Input tokens
            tokens_out: Output tokens
            provider: Provider name
            model: Model name
            agent_id: Agent ID
            agent_name: Agent name
            user_id: User who initiated the action
            task_id: Task ID
            debate_id: Debate ID
            workspace_id: Workspace ID
            org_id: Organization ID
            operation: Operation type
            source_type: Source type
            source_id: Source ID
            metadata: Additional metadata

        Returns:
            The created attribution entry
        """
        entry = AttributionEntry(
            source_type=source_type,
            source_id=source_id or str(uuid4()),
            cost_usd=cost_usd,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            provider=provider,
            model=model,
            agent_id=agent_id,
            agent_name=agent_name,
            user_id=user_id,
            task_id=task_id,
            debate_id=debate_id,
            workspace_id=workspace_id,
            org_id=org_id,
            operation=operation,
            metadata=metadata or {},
        )

        # Add team/project from mappings
        if user_id and user_id in self._user_teams:
            entry.team_id = self._user_teams[user_id]
        if task_id and task_id in self._task_projects:
            entry.project_id = self._task_projects[task_id]

        # Store entry
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

        # Update aggregated stats
        self._update_aggregates(entry)

        logger.debug(
            "cost_attributed",
            cost_usd=str(cost_usd),
            user_id=user_id,
            task_id=task_id,
            debate_id=debate_id,
        )

        return entry

    def _update_aggregates(self, entry: AttributionEntry) -> None:
        """Update aggregated stats from a new entry."""
        # User costs
        if entry.user_id:
            stats = self._user_costs[entry.user_id]
            stats["total_cost"] += entry.cost_usd
            stats["tokens_in"] += entry.tokens_in
            stats["tokens_out"] += entry.tokens_out
            stats["api_calls"] += 1
            if entry.model:
                stats["by_model"][entry.model] += entry.cost_usd
            if entry.agent_name:
                stats["by_agent"][entry.agent_name] += entry.cost_usd

        # Task costs
        if entry.task_id:
            stats = self._task_costs[entry.task_id]
            stats["total_cost"] += entry.cost_usd
            stats["tokens_in"] += entry.tokens_in
            stats["tokens_out"] += entry.tokens_out
            stats["api_calls"] += 1
            if entry.user_id:
                stats["user_id"] = entry.user_id
            if entry.workspace_id:
                stats["workspace_id"] = entry.workspace_id

        # Team costs (rollup from user)
        if entry.team_id:
            self._team_costs[entry.team_id]["total_cost"] += entry.cost_usd

        # Project costs (rollup from task)
        if entry.project_id:
            self._project_costs[entry.project_id]["total_cost"] += entry.cost_usd

    def get_user_summary(
        self,
        user_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> AttributionSummary:
        """
        Get cost summary for a user.

        Args:
            user_id: User ID
            period_start: Start of period (defaults to 30 days ago)
            period_end: End of period (defaults to now)

        Returns:
            Attribution summary for the user
        """
        if period_end is None:
            period_end = datetime.now(timezone.utc)
        if period_start is None:
            period_start = period_end - timedelta(days=30)

        # Get from aggregates for current period
        stats = self._user_costs.get(user_id, {})

        summary = AttributionSummary(
            entity_id=user_id,
            entity_type=AttributionLevel.USER,
            period_start=period_start,
            period_end=period_end,
            total_cost_usd=stats.get("total_cost", Decimal("0")),
            total_tokens_in=stats.get("tokens_in", 0),
            total_tokens_out=stats.get("tokens_out", 0),
            total_api_calls=stats.get("api_calls", 0),
            cost_by_model=dict(stats.get("by_model", {})),
            cost_by_agent=dict(stats.get("by_agent", {})),
        )

        # Calculate averages
        if summary.total_api_calls > 0:
            summary.avg_cost_per_call = summary.total_cost_usd / summary.total_api_calls
            summary.avg_tokens_per_call = (
                summary.total_tokens_in + summary.total_tokens_out
            ) / summary.total_api_calls

        # Calculate daily costs from entries
        daily_totals: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        for entry in self._entries:
            if entry.user_id == user_id:
                if period_start <= entry.timestamp <= period_end:
                    day = entry.timestamp.strftime("%Y-%m-%d")
                    daily_totals[day] += entry.cost_usd

        summary.daily_costs = [
            {"date": day, "cost_usd": str(cost)} for day, cost in sorted(daily_totals.items())
        ]

        return summary

    def get_task_summary(self, task_id: str) -> Dict[str, Any]:
        """
        Get cost summary for a task.

        Args:
            task_id: Task ID

        Returns:
            Task cost summary
        """
        stats = self._task_costs.get(task_id, {})

        return {
            "task_id": task_id,
            "total_cost_usd": str(stats.get("total_cost", Decimal("0"))),
            "tokens_in": stats.get("tokens_in", 0),
            "tokens_out": stats.get("tokens_out", 0),
            "api_calls": stats.get("api_calls", 0),
            "user_id": stats.get("user_id"),
            "workspace_id": stats.get("workspace_id"),
            "project_id": self._task_projects.get(task_id),
        }

    def get_team_summary(
        self,
        team_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get cost summary for a team.

        Args:
            team_id: Team ID
            period_start: Start of period
            period_end: End of period

        Returns:
            Team cost summary with member breakdown
        """
        stats = self._team_costs.get(team_id, {})
        members = stats.get("members", set())

        # Get costs per member
        member_costs = {}
        for user_id in members:
            user_stats = self._user_costs.get(user_id, {})
            member_costs[user_id] = {
                "cost_usd": str(user_stats.get("total_cost", Decimal("0"))),
                "api_calls": user_stats.get("api_calls", 0),
            }

        return {
            "team_id": team_id,
            "total_cost_usd": str(stats.get("total_cost", Decimal("0"))),
            "member_count": len(members),
            "member_costs": member_costs,
        }

    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """
        Get cost summary for a project.

        Args:
            project_id: Project ID

        Returns:
            Project cost summary with task breakdown
        """
        stats = self._project_costs.get(project_id, {})
        tasks = stats.get("tasks", set())

        # Get costs per task
        task_costs = {}
        for task_id in tasks:
            task_stats = self._task_costs.get(task_id, {})
            task_costs[task_id] = str(task_stats.get("total_cost", Decimal("0")))

        return {
            "project_id": project_id,
            "total_cost_usd": str(stats.get("total_cost", Decimal("0"))),
            "task_count": len(tasks),
            "task_costs": task_costs,
        }

    def generate_chargeback_report(
        self,
        workspace_id: Optional[str] = None,
        org_id: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        allocation_method: AllocationMethod = AllocationMethod.DIRECT,
    ) -> ChargebackReport:
        """
        Generate a chargeback report for billing.

        Args:
            workspace_id: Filter by workspace
            org_id: Filter by organization
            period_start: Report period start
            period_end: Report period end
            allocation_method: How to allocate shared costs

        Returns:
            Chargeback report
        """
        if period_end is None:
            period_end = datetime.now(timezone.utc)
        if period_start is None:
            period_start = period_end - timedelta(days=30)

        report = ChargebackReport(
            period_start=period_start,
            period_end=period_end,
            workspace_id=workspace_id,
            org_id=org_id,
            shared_cost_allocation_method=allocation_method,
        )

        # Filter entries by period and scope
        relevant_entries = []
        for entry in self._entries:
            if period_start <= entry.timestamp <= period_end:
                if workspace_id and entry.workspace_id != workspace_id:
                    continue
                if org_id and entry.org_id != org_id:
                    continue
                relevant_entries.append(entry)

        # Calculate allocations
        user_allocations: Dict[str, CostAllocation] = {}
        team_allocations: Dict[str, CostAllocation] = {}
        project_allocations: Dict[str, CostAllocation] = {}
        shared_cost = Decimal("0")

        for entry in relevant_entries:
            report.total_cost_usd += entry.cost_usd

            if entry.user_id:
                if entry.user_id not in user_allocations:
                    user_allocations[entry.user_id] = CostAllocation(
                        entity_id=entry.user_id,
                        entity_type=AttributionLevel.USER,
                        cost_usd=Decimal("0"),
                    )
                alloc = user_allocations[entry.user_id]
                alloc.cost_usd += entry.cost_usd
                alloc.tokens_in += entry.tokens_in
                alloc.tokens_out += entry.tokens_out
                alloc.api_calls += 1

                # Roll up to team
                if entry.team_id:
                    if entry.team_id not in team_allocations:
                        team_allocations[entry.team_id] = CostAllocation(
                            entity_id=entry.team_id,
                            entity_type=AttributionLevel.TEAM,
                            cost_usd=Decimal("0"),
                        )
                    team_allocations[entry.team_id].cost_usd += entry.cost_usd
            else:
                # No user attribution - count as shared
                shared_cost += entry.cost_usd

            # Project allocations
            if entry.project_id:
                if entry.project_id not in project_allocations:
                    project_allocations[entry.project_id] = CostAllocation(
                        entity_id=entry.project_id,
                        entity_type=AttributionLevel.PROJECT,
                        cost_usd=Decimal("0"),
                    )
                project_allocations[entry.project_id].cost_usd += entry.cost_usd

        # Handle shared costs based on allocation method
        if shared_cost > 0 and user_allocations:
            report.shared_costs_usd = shared_cost

            if allocation_method == AllocationMethod.PROPORTIONAL:
                # Distribute proportionally based on direct usage
                total_direct = sum(a.cost_usd for a in user_allocations.values())
                if total_direct > 0:
                    for user_id, alloc in user_allocations.items():
                        proportion = alloc.cost_usd / total_direct
                        alloc.cost_usd += shared_cost * proportion
                        report.notes.append(
                            f"User {user_id} allocated "
                            f"{float(proportion * 100):.1f}% of shared costs"
                        )

            elif allocation_method == AllocationMethod.EQUAL:
                # Split equally
                per_user = shared_cost / len(user_allocations)
                for alloc in user_allocations.values():
                    alloc.cost_usd += per_user

        report.allocations_by_user = user_allocations
        report.allocations_by_team = team_allocations
        report.allocations_by_project = project_allocations

        logger.info(
            "chargeback_report_generated",
            report_id=report.id,
            total_cost_usd=str(report.total_cost_usd),
            user_count=len(user_allocations),
            team_count=len(team_allocations),
        )

        return report

    def get_top_users_by_cost(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top users by cost.

        Args:
            workspace_id: Optional workspace filter
            limit: Number of users to return

        Returns:
            List of users sorted by cost
        """
        user_costs = []

        for user_id, stats in self._user_costs.items():
            # Filter by workspace if specified
            if workspace_id:
                has_workspace = any(
                    e.workspace_id == workspace_id for e in self._entries if e.user_id == user_id
                )
                if not has_workspace:
                    continue

            user_costs.append(
                {
                    "user_id": user_id,
                    "total_cost_usd": str(stats["total_cost"]),
                    "api_calls": stats["api_calls"],
                    "tokens_in": stats["tokens_in"],
                    "tokens_out": stats["tokens_out"],
                }
            )

        # Sort by cost descending
        user_costs.sort(key=lambda x: Decimal(x["total_cost_usd"]), reverse=True)

        return user_costs[:limit]

    def get_cost_trends(
        self,
        entity_type: AttributionLevel,
        entity_id: str,
        granularity: str = "daily",
        period_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get cost trends over time for an entity.

        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            granularity: "daily" or "weekly"
            period_days: Number of days to look back

        Returns:
            List of cost data points over time
        """
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=period_days)

        # Aggregate by time bucket
        buckets: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))

        for entry in self._entries:
            if period_start <= entry.timestamp <= period_end:
                # Check if entry matches entity
                matches = False
                if entity_type == AttributionLevel.USER and entry.user_id == entity_id:
                    matches = True
                elif entity_type == AttributionLevel.TASK and entry.task_id == entity_id:
                    matches = True
                elif entity_type == AttributionLevel.WORKSPACE and entry.workspace_id == entity_id:
                    matches = True
                elif entity_type == AttributionLevel.TEAM and entry.team_id == entity_id:
                    matches = True

                if matches:
                    if granularity == "weekly":
                        # ISO week
                        bucket = entry.timestamp.strftime("%Y-W%W")
                    else:
                        bucket = entry.timestamp.strftime("%Y-%m-%d")
                    buckets[bucket] += entry.cost_usd

        return [
            {"period": period, "cost_usd": str(cost)} for period, cost in sorted(buckets.items())
        ]


# Factory function for easy instantiation
def create_cost_attributor(
    cost_tracker: Optional["CostTracker"] = None,
) -> CostAttributor:
    """
    Create a CostAttributor instance.

    Args:
        cost_tracker: Optional CostTracker for integration

    Returns:
        Configured CostAttributor
    """
    return CostAttributor(cost_tracker=cost_tracker)
