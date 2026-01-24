"""
Cost Optimization Recommendations.

Dataclasses and enums for representing cost optimization recommendations
including model downgrades, caching opportunities, and batching strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class RecommendationType(str, Enum):
    """Types of cost optimization recommendations."""

    MODEL_DOWNGRADE = "model_downgrade"  # Use cheaper model for simple tasks
    CACHING = "caching"  # Enable/improve caching
    BATCHING = "batching"  # Batch similar requests
    RATE_LIMITING = "rate_limiting"  # Reduce request frequency
    PROMPT_OPTIMIZATION = "prompt_optimization"  # Shorten prompts
    PROVIDER_SWITCH = "provider_switch"  # Use cheaper provider
    TIME_SHIFTING = "time_shifting"  # Shift load to off-peak
    QUOTA_ADJUSTMENT = "quota_adjustment"  # Adjust usage quotas


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""

    CRITICAL = "critical"  # Act immediately, major savings
    HIGH = "high"  # Should implement soon
    MEDIUM = "medium"  # Good to have
    LOW = "low"  # Minor optimization


class RecommendationStatus(str, Enum):
    """Status of a recommendation."""

    PENDING = "pending"  # Not yet acted upon
    APPLIED = "applied"  # User applied the recommendation
    DISMISSED = "dismissed"  # User dismissed
    EXPIRED = "expired"  # No longer applicable
    PARTIAL = "partial"  # Partially applied


@dataclass
class ModelAlternative:
    """An alternative model that could be used."""

    provider: str
    model: str
    cost_per_1k_input: Decimal
    cost_per_1k_output: Decimal
    quality_score: float  # 0-1, relative quality vs current
    latency_multiplier: float  # 1.0 = same, 2.0 = 2x slower
    suitable_for: List[str] = field(default_factory=list)  # Task types


@dataclass
class CachingOpportunity:
    """Details about a caching opportunity."""

    pattern: str  # e.g., "system_prompt", "repeated_query"
    estimated_hit_rate: float  # 0-1
    unique_queries: int
    repeat_count: int
    cache_strategy: str  # "exact", "semantic", "prefix"


@dataclass
class BatchingOpportunity:
    """Details about a batching opportunity."""

    operation_type: str
    current_batch_size: int
    optimal_batch_size: int
    requests_per_hour: int
    latency_impact_ms: float


@dataclass
class ImplementationStep:
    """A step to implement a recommendation."""

    order: int
    description: str
    code_snippet: Optional[str] = None
    config_change: Optional[Dict[str, Any]] = None
    estimated_effort: str = "low"  # low, medium, high


@dataclass
class OptimizationRecommendation:
    """
    A cost optimization recommendation.

    Represents an actionable suggestion to reduce AI model costs
    while maintaining acceptable quality and performance.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    type: RecommendationType = RecommendationType.MODEL_DOWNGRADE
    priority: RecommendationPriority = RecommendationPriority.MEDIUM
    status: RecommendationStatus = RecommendationStatus.PENDING

    # Cost impact
    current_cost_usd: Decimal = Decimal("0")
    projected_cost_usd: Decimal = Decimal("0")
    estimated_savings_usd: Decimal = Decimal("0")
    savings_percentage: float = 0.0
    confidence_score: float = 0.0  # 0-1, how confident in the estimate

    # Context
    workspace_id: str = ""
    org_id: Optional[str] = None
    affected_agents: List[str] = field(default_factory=list)
    affected_operations: List[str] = field(default_factory=list)

    # Human-readable
    title: str = ""
    description: str = ""
    rationale: str = ""  # Why this recommendation

    # Type-specific details
    model_alternative: Optional[ModelAlternative] = None
    caching_opportunity: Optional[CachingOpportunity] = None
    batching_opportunity: Optional[BatchingOpportunity] = None

    # Implementation
    implementation_steps: List[ImplementationStep] = field(default_factory=list)
    auto_apply_available: bool = False  # Can be applied automatically
    requires_approval: bool = True  # Needs human approval

    # Quality/risk assessment
    quality_impact: str = ""  # Description of quality tradeoff
    quality_impact_score: float = 0.0  # 0 = no impact, 1 = major degradation
    risk_level: str = "low"  # low, medium, high

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None

    # Tracking
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        if self.current_cost_usd > 0 and self.projected_cost_usd >= 0:
            self.estimated_savings_usd = self.current_cost_usd - self.projected_cost_usd
            self.savings_percentage = float(
                (self.estimated_savings_usd / self.current_cost_usd) * 100
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "current_cost_usd": str(self.current_cost_usd),
            "projected_cost_usd": str(self.projected_cost_usd),
            "estimated_savings_usd": str(self.estimated_savings_usd),
            "savings_percentage": round(self.savings_percentage, 1),
            "confidence_score": round(self.confidence_score, 2),
            "workspace_id": self.workspace_id,
            "org_id": self.org_id,
            "affected_agents": self.affected_agents,
            "affected_operations": self.affected_operations,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "model_alternative": (
                {
                    "provider": self.model_alternative.provider,
                    "model": self.model_alternative.model,
                    "cost_per_1k_input": str(self.model_alternative.cost_per_1k_input),
                    "cost_per_1k_output": str(self.model_alternative.cost_per_1k_output),
                    "quality_score": self.model_alternative.quality_score,
                    "latency_multiplier": self.model_alternative.latency_multiplier,
                }
                if self.model_alternative
                else None
            ),
            "caching_opportunity": (
                {
                    "pattern": self.caching_opportunity.pattern,
                    "estimated_hit_rate": self.caching_opportunity.estimated_hit_rate,
                    "unique_queries": self.caching_opportunity.unique_queries,
                    "repeat_count": self.caching_opportunity.repeat_count,
                    "cache_strategy": self.caching_opportunity.cache_strategy,
                }
                if self.caching_opportunity
                else None
            ),
            "batching_opportunity": (
                {
                    "operation_type": self.batching_opportunity.operation_type,
                    "current_batch_size": self.batching_opportunity.current_batch_size,
                    "optimal_batch_size": self.batching_opportunity.optimal_batch_size,
                    "requests_per_hour": self.batching_opportunity.requests_per_hour,
                    "latency_impact_ms": self.batching_opportunity.latency_impact_ms,
                }
                if self.batching_opportunity
                else None
            ),
            "implementation_steps": [
                {
                    "order": step.order,
                    "description": step.description,
                    "code_snippet": step.code_snippet,
                    "estimated_effort": step.estimated_effort,
                }
                for step in self.implementation_steps
            ],
            "auto_apply_available": self.auto_apply_available,
            "requires_approval": self.requires_approval,
            "quality_impact": self.quality_impact,
            "quality_impact_score": self.quality_impact_score,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "applied_by": self.applied_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OptimizationRecommendation:
        """Create from dictionary."""
        rec = cls(
            id=data.get("id", str(uuid4())),
            type=RecommendationType(data.get("type", "model_downgrade")),
            priority=RecommendationPriority(data.get("priority", "medium")),
            status=RecommendationStatus(data.get("status", "pending")),
            current_cost_usd=Decimal(data.get("current_cost_usd", "0")),
            projected_cost_usd=Decimal(data.get("projected_cost_usd", "0")),
            confidence_score=data.get("confidence_score", 0.0),
            workspace_id=data.get("workspace_id", ""),
            org_id=data.get("org_id"),
            affected_agents=data.get("affected_agents", []),
            affected_operations=data.get("affected_operations", []),
            title=data.get("title", ""),
            description=data.get("description", ""),
            rationale=data.get("rationale", ""),
            quality_impact=data.get("quality_impact", ""),
            quality_impact_score=data.get("quality_impact_score", 0.0),
            risk_level=data.get("risk_level", "low"),
            auto_apply_available=data.get("auto_apply_available", False),
            requires_approval=data.get("requires_approval", True),
            metadata=data.get("metadata", {}),
        )
        return rec

    def apply(self, user_id: str) -> None:
        """Mark recommendation as applied."""
        self.status = RecommendationStatus.APPLIED
        self.applied_at = datetime.now(timezone.utc)
        self.applied_by = user_id

    def dismiss(self) -> None:
        """Mark recommendation as dismissed."""
        self.status = RecommendationStatus.DISMISSED


@dataclass
class RecommendationSummary:
    """Summary of recommendations for a workspace."""

    workspace_id: str
    total_recommendations: int = 0
    pending_count: int = 0
    applied_count: int = 0
    dismissed_count: int = 0

    # Aggregate savings
    total_potential_savings: Decimal = Decimal("0")
    realized_savings: Decimal = Decimal("0")

    # By priority
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # By type
    by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workspace_id": self.workspace_id,
            "total_recommendations": self.total_recommendations,
            "pending_count": self.pending_count,
            "applied_count": self.applied_count,
            "dismissed_count": self.dismissed_count,
            "total_potential_savings_usd": str(self.total_potential_savings),
            "realized_savings_usd": str(self.realized_savings),
            "by_priority": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "by_type": self.by_type,
        }


__all__ = [
    "RecommendationType",
    "RecommendationPriority",
    "RecommendationStatus",
    "ModelAlternative",
    "CachingOpportunity",
    "BatchingOpportunity",
    "ImplementationStep",
    "OptimizationRecommendation",
    "RecommendationSummary",
]
