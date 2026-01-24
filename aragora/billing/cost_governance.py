"""
Cost Governance Policy Engine.

Provides governance policies for AI model costs:
- Model usage restrictions (which models can be used)
- Spending limits per user/team/workspace
- Cost caps per operation type
- Approval workflows for expensive operations
- Audit logging of cost policy decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from uuid import uuid4

from aragora.observability import get_logger

if TYPE_CHECKING:
    from aragora.billing.cost_tracker import CostTracker
    from aragora.billing.cost_attribution import CostAttributor

logger = get_logger(__name__)


class CostPolicyType(str, Enum):
    """Types of cost governance policies."""

    MODEL_RESTRICTION = "model_restriction"  # Control which models can be used
    SPENDING_LIMIT = "spending_limit"  # Set spending caps
    OPERATION_CAP = "operation_cap"  # Cap cost per operation type
    APPROVAL_REQUIRED = "approval_required"  # Require approval for expensive ops
    TIME_RESTRICTION = "time_restriction"  # Time-based restrictions


class CostPolicyScope(str, Enum):
    """Scope of a cost policy."""

    GLOBAL = "global"  # Applies to entire organization
    WORKSPACE = "workspace"  # Applies to specific workspaces
    TEAM = "team"  # Applies to specific teams
    USER = "user"  # Applies to specific users
    PROJECT = "project"  # Applies to specific projects


class CostPolicyEnforcement(str, Enum):
    """Enforcement level for cost policies."""

    HARD = "hard"  # Block operations that violate policy
    SOFT = "soft"  # Allow but log violations
    WARN = "warn"  # Warn but allow
    AUDIT = "audit"  # Only log, no enforcement


class CostPolicyAction(str, Enum):
    """Actions when a policy is violated."""

    ALLOW = "allow"
    DENY = "deny"
    DOWNGRADE = "downgrade"  # Use cheaper model
    QUEUE = "queue"  # Queue for approval
    THROTTLE = "throttle"  # Reduce priority


@dataclass
class ModelRestriction:
    """Restriction on model usage."""

    model_pattern: str  # Model name or pattern (e.g., "claude-opus*")
    allowed: bool = True
    max_requests_per_hour: Optional[int] = None
    max_tokens_per_request: Optional[int] = None
    allowed_operations: List[str] = field(default_factory=list)  # Empty = all
    fallback_model: Optional[str] = None  # Use this if denied

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_pattern": self.model_pattern,
            "allowed": self.allowed,
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_tokens_per_request": self.max_tokens_per_request,
            "allowed_operations": self.allowed_operations,
            "fallback_model": self.fallback_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRestriction":
        """Create from dictionary."""
        return cls(
            model_pattern=data["model_pattern"],
            allowed=data.get("allowed", True),
            max_requests_per_hour=data.get("max_requests_per_hour"),
            max_tokens_per_request=data.get("max_tokens_per_request"),
            allowed_operations=data.get("allowed_operations", []),
            fallback_model=data.get("fallback_model"),
        )


@dataclass
class SpendingLimit:
    """Spending limit configuration."""

    daily_limit_usd: Optional[Decimal] = None
    weekly_limit_usd: Optional[Decimal] = None
    monthly_limit_usd: Optional[Decimal] = None
    per_operation_limit_usd: Optional[Decimal] = None
    alert_threshold_percent: float = 80.0
    hard_limit: bool = True  # If true, block when limit reached

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "daily_limit_usd": str(self.daily_limit_usd) if self.daily_limit_usd else None,
            "weekly_limit_usd": str(self.weekly_limit_usd) if self.weekly_limit_usd else None,
            "monthly_limit_usd": str(self.monthly_limit_usd) if self.monthly_limit_usd else None,
            "per_operation_limit_usd": str(self.per_operation_limit_usd)
            if self.per_operation_limit_usd
            else None,
            "alert_threshold_percent": self.alert_threshold_percent,
            "hard_limit": self.hard_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendingLimit":
        """Create from dictionary."""
        return cls(
            daily_limit_usd=Decimal(data["daily_limit_usd"])
            if data.get("daily_limit_usd")
            else None,
            weekly_limit_usd=Decimal(data["weekly_limit_usd"])
            if data.get("weekly_limit_usd")
            else None,
            monthly_limit_usd=Decimal(data["monthly_limit_usd"])
            if data.get("monthly_limit_usd")
            else None,
            per_operation_limit_usd=Decimal(data["per_operation_limit_usd"])
            if data.get("per_operation_limit_usd")
            else None,
            alert_threshold_percent=data.get("alert_threshold_percent", 80.0),
            hard_limit=data.get("hard_limit", True),
        )


@dataclass
class TimeRestriction:
    """Time-based usage restriction."""

    allowed_hours_start: int = 0  # 0-23
    allowed_hours_end: int = 24  # 0-24
    allowed_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    timezone: str = "UTC"

    def is_allowed_now(self) -> bool:
        """Check if current time is within allowed window."""
        now = datetime.now(timezone.utc)
        day_of_week = now.weekday()
        hour = now.hour

        if day_of_week not in self.allowed_days:
            return False
        if not (self.allowed_hours_start <= hour < self.allowed_hours_end):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "allowed_hours_start": self.allowed_hours_start,
            "allowed_hours_end": self.allowed_hours_end,
            "allowed_days": self.allowed_days,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeRestriction":
        """Create from dictionary."""
        return cls(
            allowed_hours_start=data.get("allowed_hours_start", 0),
            allowed_hours_end=data.get("allowed_hours_end", 24),
            allowed_days=data.get("allowed_days", [0, 1, 2, 3, 4]),
            timezone=data.get("timezone", "UTC"),
        )


@dataclass
class CostGovernancePolicy:
    """A cost governance policy."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""

    # Policy type and scope
    policy_type: CostPolicyType = CostPolicyType.SPENDING_LIMIT
    scope: CostPolicyScope = CostPolicyScope.GLOBAL
    enforcement: CostPolicyEnforcement = CostPolicyEnforcement.HARD

    # Target entities (based on scope)
    target_workspaces: List[str] = field(default_factory=list)
    target_teams: List[str] = field(default_factory=list)
    target_users: List[str] = field(default_factory=list)
    target_projects: List[str] = field(default_factory=list)

    # Policy rules
    model_restrictions: List[ModelRestriction] = field(default_factory=list)
    spending_limit: Optional[SpendingLimit] = None
    time_restriction: Optional[TimeRestriction] = None

    # Approval configuration (for APPROVAL_REQUIRED type)
    approval_threshold_usd: Optional[Decimal] = None
    approvers: List[str] = field(default_factory=list)
    auto_approve_under_usd: Optional[Decimal] = None

    # Priority and status
    priority: int = 0  # Higher = evaluated first
    enabled: bool = True

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        workspace_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> bool:
        """Check if this policy applies to the given context."""
        if not self.enabled:
            return False

        # Global scope matches everything
        if self.scope == CostPolicyScope.GLOBAL:
            return True

        # Check specific scopes
        if self.scope == CostPolicyScope.WORKSPACE:
            if not self.target_workspaces:
                return True  # Empty = all workspaces
            return workspace_id in self.target_workspaces if workspace_id else False

        if self.scope == CostPolicyScope.TEAM:
            if not self.target_teams:
                return True
            return team_id in self.target_teams if team_id else False

        if self.scope == CostPolicyScope.USER:
            if not self.target_users:
                return True
            return user_id in self.target_users if user_id else False

        if self.scope == CostPolicyScope.PROJECT:
            if not self.target_projects:
                return True
            return project_id in self.target_projects if project_id else False

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type.value,
            "scope": self.scope.value,
            "enforcement": self.enforcement.value,
            "target_workspaces": self.target_workspaces,
            "target_teams": self.target_teams,
            "target_users": self.target_users,
            "target_projects": self.target_projects,
            "model_restrictions": [r.to_dict() for r in self.model_restrictions],
            "spending_limit": self.spending_limit.to_dict() if self.spending_limit else None,
            "time_restriction": self.time_restriction.to_dict() if self.time_restriction else None,
            "approval_threshold_usd": str(self.approval_threshold_usd)
            if self.approval_threshold_usd
            else None,
            "approvers": self.approvers,
            "auto_approve_under_usd": str(self.auto_approve_under_usd)
            if self.auto_approve_under_usd
            else None,
            "priority": self.priority,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostGovernancePolicy":
        """Create from dictionary."""
        model_restrictions = [
            ModelRestriction.from_dict(r) for r in data.get("model_restrictions", [])
        ]

        spending_limit = None
        if data.get("spending_limit"):
            spending_limit = SpendingLimit.from_dict(data["spending_limit"])

        time_restriction = None
        if data.get("time_restriction"):
            time_restriction = TimeRestriction.from_dict(data["time_restriction"])

        return cls(
            id=data.get("id", str(uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            policy_type=CostPolicyType(data.get("policy_type", "spending_limit")),
            scope=CostPolicyScope(data.get("scope", "global")),
            enforcement=CostPolicyEnforcement(data.get("enforcement", "hard")),
            target_workspaces=data.get("target_workspaces", []),
            target_teams=data.get("target_teams", []),
            target_users=data.get("target_users", []),
            target_projects=data.get("target_projects", []),
            model_restrictions=model_restrictions,
            spending_limit=spending_limit,
            time_restriction=time_restriction,
            approval_threshold_usd=Decimal(data["approval_threshold_usd"])
            if data.get("approval_threshold_usd")
            else None,
            approvers=data.get("approvers", []),
            auto_approve_under_usd=Decimal(data["auto_approve_under_usd"])
            if data.get("auto_approve_under_usd")
            else None,
            priority=data.get("priority", 0),
            enabled=data.get("enabled", True),
            created_by=data.get("created_by"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PolicyEvaluationContext:
    """Context for evaluating a cost policy."""

    # Entity identifiers
    workspace_id: Optional[str] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None

    # Operation details
    operation: str = ""
    model: str = ""
    estimated_cost_usd: Decimal = Decimal("0")
    estimated_tokens: int = 0

    # Current spend
    current_daily_spend: Decimal = Decimal("0")
    current_weekly_spend: Decimal = Decimal("0")
    current_monthly_spend: Decimal = Decimal("0")

    # Request metadata
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PolicyViolation:
    """A cost policy violation."""

    policy_id: str
    policy_name: str
    violation_type: str
    message: str
    severity: CostPolicyEnforcement
    action: CostPolicyAction
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "violation_type": self.violation_type,
            "message": self.message,
            "severity": self.severity.value,
            "action": self.action.value,
            "details": self.details,
        }


@dataclass
class PolicyEvaluationResult:
    """Result of evaluating cost policies."""

    allowed: bool = True
    action: CostPolicyAction = CostPolicyAction.ALLOW
    violations: List[PolicyViolation] = field(default_factory=list)
    suggested_model: Optional[str] = None  # If downgrade suggested
    requires_approval: bool = False
    approvers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "violations": [v.to_dict() for v in self.violations],
            "suggested_model": self.suggested_model,
            "requires_approval": self.requires_approval,
            "approvers": self.approvers,
            "warnings": self.warnings,
        }


class CostGovernanceEngine:
    """
    Engine for evaluating and enforcing cost governance policies.

    Features:
    - Model usage restrictions
    - Spending limit enforcement
    - Approval workflows
    - Policy conflict detection
    - Audit logging
    """

    def __init__(
        self,
        cost_tracker: Optional["CostTracker"] = None,
        cost_attributor: Optional["CostAttributor"] = None,
    ):
        """
        Initialize cost governance engine.

        Args:
            cost_tracker: CostTracker for current spend data
            cost_attributor: CostAttributor for attribution data
        """
        self._cost_tracker = cost_tracker
        self._cost_attributor = cost_attributor
        self._policies: Dict[str, CostGovernancePolicy] = {}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Audit callbacks
        self._audit_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Model request counters (for rate limiting)
        self._model_requests: Dict[str, List[datetime]] = {}
        self._request_window = timedelta(hours=1)

        logger.info("CostGovernanceEngine initialized")

    def add_policy(self, policy: CostGovernancePolicy) -> None:
        """Add a cost governance policy."""
        self._policies[policy.id] = policy
        logger.info(
            "cost_policy_added",
            policy_id=policy.id,
            policy_name=policy.name,
            policy_type=policy.policy_type.value,
        )

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a cost governance policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
            logger.info("cost_policy_removed", policy_id=policy_id)
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[CostGovernancePolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(
        self,
        policy_type: Optional[CostPolicyType] = None,
        scope: Optional[CostPolicyScope] = None,
        enabled_only: bool = True,
    ) -> List[CostGovernancePolicy]:
        """List all policies matching criteria."""
        policies = []
        for policy in self._policies.values():
            if enabled_only and not policy.enabled:
                continue
            if policy_type and policy.policy_type != policy_type:
                continue
            if scope and policy.scope != scope:
                continue
            policies.append(policy)

        # Sort by priority (higher first)
        return sorted(policies, key=lambda p: p.priority, reverse=True)

    def evaluate(self, context: PolicyEvaluationContext) -> PolicyEvaluationResult:
        """
        Evaluate all applicable policies for a given context.

        Args:
            context: Evaluation context

        Returns:
            Policy evaluation result
        """
        result = PolicyEvaluationResult()

        # Get applicable policies sorted by priority
        applicable_policies = []
        for policy in self._policies.values():
            if policy.matches(
                workspace_id=context.workspace_id,
                team_id=context.team_id,
                user_id=context.user_id,
                project_id=context.project_id,
            ):
                applicable_policies.append(policy)

        applicable_policies.sort(key=lambda p: p.priority, reverse=True)

        # Evaluate each policy
        for policy in applicable_policies:
            self._evaluate_policy(policy, context, result)

            # If hard enforcement and denied, stop evaluation
            if (
                policy.enforcement == CostPolicyEnforcement.HARD
                and result.action == CostPolicyAction.DENY
            ):
                break

        # Determine final action
        if result.violations:
            # Get most severe action from violations
            actions = [v.action for v in result.violations]
            if CostPolicyAction.DENY in actions:
                result.action = CostPolicyAction.DENY
                result.allowed = False
            elif CostPolicyAction.QUEUE in actions:
                result.action = CostPolicyAction.QUEUE
                result.requires_approval = True
            elif CostPolicyAction.DOWNGRADE in actions:
                result.action = CostPolicyAction.DOWNGRADE
            elif CostPolicyAction.THROTTLE in actions:
                result.action = CostPolicyAction.THROTTLE

        # Log evaluation
        self._audit_evaluation(context, result)

        return result

    def _evaluate_policy(
        self,
        policy: CostGovernancePolicy,
        context: PolicyEvaluationContext,
        result: PolicyEvaluationResult,
    ) -> None:
        """Evaluate a single policy."""
        if policy.policy_type == CostPolicyType.MODEL_RESTRICTION:
            self._evaluate_model_restriction(policy, context, result)
        elif policy.policy_type == CostPolicyType.SPENDING_LIMIT:
            self._evaluate_spending_limit(policy, context, result)
        elif policy.policy_type == CostPolicyType.APPROVAL_REQUIRED:
            self._evaluate_approval_required(policy, context, result)
        elif policy.policy_type == CostPolicyType.TIME_RESTRICTION:
            self._evaluate_time_restriction(policy, context, result)

    def _evaluate_model_restriction(
        self,
        policy: CostGovernancePolicy,
        context: PolicyEvaluationContext,
        result: PolicyEvaluationResult,
    ) -> None:
        """Evaluate model restriction policy."""
        for restriction in policy.model_restrictions:
            if not self._model_matches_pattern(context.model, restriction.model_pattern):
                continue

            # Check if model is allowed
            if not restriction.allowed:
                action = self._get_action_for_enforcement(policy.enforcement)
                result.violations.append(
                    PolicyViolation(
                        policy_id=policy.id,
                        policy_name=policy.name,
                        violation_type="model_not_allowed",
                        message=f"Model '{context.model}' is not allowed by policy",
                        severity=policy.enforcement,
                        action=action,
                        details={"model": context.model},
                    )
                )
                if restriction.fallback_model:
                    result.suggested_model = restriction.fallback_model
                continue

            # Check operation restrictions
            if restriction.allowed_operations:
                if context.operation not in restriction.allowed_operations:
                    action = self._get_action_for_enforcement(policy.enforcement)
                    result.violations.append(
                        PolicyViolation(
                            policy_id=policy.id,
                            policy_name=policy.name,
                            violation_type="operation_not_allowed",
                            message=f"Operation '{context.operation}' not allowed for model '{context.model}'",
                            severity=policy.enforcement,
                            action=action,
                            details={
                                "model": context.model,
                                "operation": context.operation,
                                "allowed_operations": restriction.allowed_operations,
                            },
                        )
                    )

            # Check rate limit
            if restriction.max_requests_per_hour:
                request_count = self._count_recent_requests(context.model)
                if request_count >= restriction.max_requests_per_hour:
                    result.violations.append(
                        PolicyViolation(
                            policy_id=policy.id,
                            policy_name=policy.name,
                            violation_type="rate_limit_exceeded",
                            message=f"Rate limit exceeded for model '{context.model}'",
                            severity=policy.enforcement,
                            action=CostPolicyAction.THROTTLE,
                            details={
                                "model": context.model,
                                "current_requests": request_count,
                                "limit": restriction.max_requests_per_hour,
                            },
                        )
                    )

            # Check token limit
            if restriction.max_tokens_per_request:
                if context.estimated_tokens > restriction.max_tokens_per_request:
                    result.violations.append(
                        PolicyViolation(
                            policy_id=policy.id,
                            policy_name=policy.name,
                            violation_type="token_limit_exceeded",
                            message=f"Token limit exceeded for model '{context.model}'",
                            severity=policy.enforcement,
                            action=self._get_action_for_enforcement(policy.enforcement),
                            details={
                                "model": context.model,
                                "estimated_tokens": context.estimated_tokens,
                                "limit": restriction.max_tokens_per_request,
                            },
                        )
                    )

    def _evaluate_spending_limit(
        self,
        policy: CostGovernancePolicy,
        context: PolicyEvaluationContext,
        result: PolicyEvaluationResult,
    ) -> None:
        """Evaluate spending limit policy."""
        if not policy.spending_limit:
            return

        limit = policy.spending_limit

        # Check daily limit
        if limit.daily_limit_usd:
            if context.current_daily_spend >= limit.daily_limit_usd:
                action = CostPolicyAction.DENY if limit.hard_limit else CostPolicyAction.ALLOW
                result.violations.append(
                    PolicyViolation(
                        policy_id=policy.id,
                        policy_name=policy.name,
                        violation_type="daily_limit_exceeded",
                        message=f"Daily spending limit of ${limit.daily_limit_usd} exceeded",
                        severity=policy.enforcement,
                        action=action,
                        details={
                            "current_spend": str(context.current_daily_spend),
                            "limit": str(limit.daily_limit_usd),
                        },
                    )
                )
            elif (
                context.current_daily_spend / limit.daily_limit_usd * 100
            ) >= limit.alert_threshold_percent:
                result.warnings.append(
                    f"Approaching daily limit: {context.current_daily_spend}/{limit.daily_limit_usd}"
                )

        # Check weekly limit
        if limit.weekly_limit_usd:
            if context.current_weekly_spend >= limit.weekly_limit_usd:
                action = CostPolicyAction.DENY if limit.hard_limit else CostPolicyAction.ALLOW
                result.violations.append(
                    PolicyViolation(
                        policy_id=policy.id,
                        policy_name=policy.name,
                        violation_type="weekly_limit_exceeded",
                        message=f"Weekly spending limit of ${limit.weekly_limit_usd} exceeded",
                        severity=policy.enforcement,
                        action=action,
                        details={
                            "current_spend": str(context.current_weekly_spend),
                            "limit": str(limit.weekly_limit_usd),
                        },
                    )
                )

        # Check monthly limit
        if limit.monthly_limit_usd:
            if context.current_monthly_spend >= limit.monthly_limit_usd:
                action = CostPolicyAction.DENY if limit.hard_limit else CostPolicyAction.ALLOW
                result.violations.append(
                    PolicyViolation(
                        policy_id=policy.id,
                        policy_name=policy.name,
                        violation_type="monthly_limit_exceeded",
                        message=f"Monthly spending limit of ${limit.monthly_limit_usd} exceeded",
                        severity=policy.enforcement,
                        action=action,
                        details={
                            "current_spend": str(context.current_monthly_spend),
                            "limit": str(limit.monthly_limit_usd),
                        },
                    )
                )

        # Check per-operation limit
        if limit.per_operation_limit_usd:
            if context.estimated_cost_usd > limit.per_operation_limit_usd:
                result.violations.append(
                    PolicyViolation(
                        policy_id=policy.id,
                        policy_name=policy.name,
                        violation_type="operation_cost_exceeded",
                        message=f"Operation cost ${context.estimated_cost_usd} exceeds limit ${limit.per_operation_limit_usd}",
                        severity=policy.enforcement,
                        action=CostPolicyAction.QUEUE,
                        details={
                            "estimated_cost": str(context.estimated_cost_usd),
                            "limit": str(limit.per_operation_limit_usd),
                        },
                    )
                )

    def _evaluate_approval_required(
        self,
        policy: CostGovernancePolicy,
        context: PolicyEvaluationContext,
        result: PolicyEvaluationResult,
    ) -> None:
        """Evaluate approval required policy."""
        if policy.approval_threshold_usd:
            # Auto-approve if under threshold
            if (
                policy.auto_approve_under_usd
                and context.estimated_cost_usd <= policy.auto_approve_under_usd
            ):
                return

            # Check if exceeds approval threshold
            if context.estimated_cost_usd >= policy.approval_threshold_usd:
                result.violations.append(
                    PolicyViolation(
                        policy_id=policy.id,
                        policy_name=policy.name,
                        violation_type="approval_required",
                        message=f"Operation cost ${context.estimated_cost_usd} requires approval",
                        severity=policy.enforcement,
                        action=CostPolicyAction.QUEUE,
                        details={
                            "estimated_cost": str(context.estimated_cost_usd),
                            "threshold": str(policy.approval_threshold_usd),
                        },
                    )
                )
                result.requires_approval = True
                result.approvers = policy.approvers.copy()

    def _evaluate_time_restriction(
        self,
        policy: CostGovernancePolicy,
        context: PolicyEvaluationContext,
        result: PolicyEvaluationResult,
    ) -> None:
        """Evaluate time restriction policy."""
        if policy.time_restriction and not policy.time_restriction.is_allowed_now():
            result.violations.append(
                PolicyViolation(
                    policy_id=policy.id,
                    policy_name=policy.name,
                    violation_type="time_restriction",
                    message="Operation not allowed at this time",
                    severity=policy.enforcement,
                    action=self._get_action_for_enforcement(policy.enforcement),
                    details={
                        "allowed_hours": f"{policy.time_restriction.allowed_hours_start}-{policy.time_restriction.allowed_hours_end}",
                        "allowed_days": policy.time_restriction.allowed_days,
                    },
                )
            )

    def _model_matches_pattern(self, model: str, pattern: str) -> bool:
        """Check if a model name matches a pattern."""
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            return model.startswith(prefix)
        return model == pattern

    def _count_recent_requests(self, model: str) -> int:
        """Count recent requests for a model."""
        now = datetime.now(timezone.utc)
        cutoff = now - self._request_window

        if model not in self._model_requests:
            return 0

        # Clean old requests
        self._model_requests[model] = [t for t in self._model_requests[model] if t >= cutoff]

        return len(self._model_requests[model])

    def record_model_request(self, model: str) -> None:
        """Record a model request for rate limiting."""
        if model not in self._model_requests:
            self._model_requests[model] = []
        self._model_requests[model].append(datetime.now(timezone.utc))

    def _get_action_for_enforcement(self, enforcement: CostPolicyEnforcement) -> CostPolicyAction:
        """Get the appropriate action for an enforcement level."""
        if enforcement == CostPolicyEnforcement.HARD:
            return CostPolicyAction.DENY
        elif enforcement == CostPolicyEnforcement.SOFT:
            return CostPolicyAction.ALLOW
        elif enforcement == CostPolicyEnforcement.WARN:
            return CostPolicyAction.ALLOW
        return CostPolicyAction.ALLOW

    def _audit_evaluation(
        self, context: PolicyEvaluationContext, result: PolicyEvaluationResult
    ) -> None:
        """Log policy evaluation for audit."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": {
                "workspace_id": context.workspace_id,
                "user_id": context.user_id,
                "operation": context.operation,
                "model": context.model,
                "estimated_cost_usd": str(context.estimated_cost_usd),
            },
            "result": result.to_dict(),
        }

        logger.info(
            "cost_policy_evaluation",
            allowed=result.allowed,
            action=result.action.value,
            violation_count=len(result.violations),
        )

        for callback in self._audit_callbacks:
            try:
                callback(audit_entry)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")

    def add_audit_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add an audit callback."""
        self._audit_callbacks.append(callback)

    def request_approval(
        self,
        context: PolicyEvaluationContext,
        policy_id: str,
        requestor_id: str,
    ) -> str:
        """Request approval for an operation."""
        request_id = str(uuid4())

        self._pending_approvals[request_id] = {
            "request_id": request_id,
            "context": context,
            "policy_id": policy_id,
            "requestor_id": requestor_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "approved_by": None,
            "approved_at": None,
        }

        logger.info(
            "approval_requested",
            request_id=request_id,
            policy_id=policy_id,
            requestor_id=requestor_id,
        )

        return request_id

    def approve_request(self, request_id: str, approver_id: str) -> bool:
        """Approve a pending request."""
        if request_id not in self._pending_approvals:
            return False

        approval = self._pending_approvals[request_id]

        # Verify approver is authorized
        policy = self.get_policy(approval["policy_id"])
        if policy and approver_id not in policy.approvers:
            logger.warning(
                "unauthorized_approver",
                request_id=request_id,
                approver_id=approver_id,
            )
            return False

        approval["status"] = "approved"
        approval["approved_by"] = approver_id
        approval["approved_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "request_approved",
            request_id=request_id,
            approver_id=approver_id,
        )

        return True

    def deny_request(self, request_id: str, denier_id: str, reason: str = "") -> bool:
        """Deny a pending request."""
        if request_id not in self._pending_approvals:
            return False

        approval = self._pending_approvals[request_id]
        approval["status"] = "denied"
        approval["denied_by"] = denier_id
        approval["denied_at"] = datetime.now(timezone.utc).isoformat()
        approval["denial_reason"] = reason

        logger.info(
            "request_denied",
            request_id=request_id,
            denier_id=denier_id,
        )

        return True

    def get_pending_approvals(self, approver_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending approval requests."""
        pending = []
        for request in self._pending_approvals.values():
            if request["status"] != "pending":
                continue

            if approver_id:
                policy = self.get_policy(request["policy_id"])
                if policy and approver_id not in policy.approvers:
                    continue

            pending.append(request)

        return pending


# Factory function
def create_cost_governance_engine(
    cost_tracker: Optional["CostTracker"] = None,
    cost_attributor: Optional["CostAttributor"] = None,
) -> CostGovernanceEngine:
    """Create a CostGovernanceEngine instance."""
    return CostGovernanceEngine(
        cost_tracker=cost_tracker,
        cost_attributor=cost_attributor,
    )
