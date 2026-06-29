"""Routing configuration helpers.

Centralizes Gateway routing criteria configuration so unified routing,
DecisionRoutingMiddleware, and programmatic consumers use consistent defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


_DEF_RISK_LEVELS = {"high", "critical"}
_DEF_COMPLIANCE_FLAGS = {"pii", "financial", "hipaa", "gdpr"}
_DEF_DEBATE_KEYWORDS = {
    "consensus",
    "debate",
    "discuss",
    "decide",
    "vote",
    "approve",
}
_DEF_EXECUTE_KEYWORDS = {"execute", "run", "perform", "do", "just do it"}


@dataclass(frozen=True)
class GatewayRoutingConfig:
    """Configuration for Gateway routing criteria."""

    financial_threshold: float = 10000.0
    risk_levels: set[str] = None  # type: ignore[assignment]
    compliance_flags: set[str] = None  # type: ignore[assignment]
    stakeholder_threshold: int = 3
    require_debate_keywords: set[str] = None  # type: ignore[assignment]
    require_execute_keywords: set[str] = None  # type: ignore[assignment]
    time_sensitive_threshold_seconds: int = 60
    confidence_threshold: float = 0.85
    cache_ttl_seconds: int = 300  # 5-minute cache for provider routing decisions

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_levels", self.risk_levels or set(_DEF_RISK_LEVELS))
        object.__setattr__(
            self, "compliance_flags", self.compliance_flags or set(_DEF_COMPLIANCE_FLAGS)
        )
        object.__setattr__(
            self,
            "require_debate_keywords",
            self.require_debate_keywords or set(_DEF_DEBATE_KEYWORDS),
        )
        object.__setattr__(
            self,
            "require_execute_keywords",
            self.require_execute_keywords or set(_DEF_EXECUTE_KEYWORDS),
        )


@dataclass(frozen=True)
class TaskRoutingContext:
    """Per-unit-of-work routing inputs for the hybrid orchestrator.

    Carries the signals the Pareto model-router and mode-selector both need so a
    single unit of work can be (a) assigned an execution pattern and (b) routed
    to a cost/quality/latency-optimal model. Pure data -- no behavior, no I/O.

    Fields:
      - ``complexity_score``: 0-10, typically from ``DomainDetector`` complexity
        estimation; higher means more reasoning-heavy.
      - ``budget_usd``: optional per-unit spend cap; ``None`` means uncapped.
      - ``latency_ms_sla``: optional soft latency target; ``None`` means none.
      - ``quality_floor``: 0-1 minimum acceptable quality for candidate models.
      - ``diversity_preference``: 0-1 weight favoring heterogeneous model sets
        (matters for multi-agent/team patterns; ignored for single-model work).
    """

    task_id: str
    domain: str
    complexity_score: float = 0.0
    budget_usd: float | None = None
    latency_ms_sla: float | None = None
    quality_floor: float = 0.0
    diversity_preference: float = 0.0

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not 0.0 <= self.complexity_score <= 10.0:
            raise ValueError("complexity_score must be in [0, 10]")
        if not 0.0 <= self.quality_floor <= 1.0:
            raise ValueError("quality_floor must be in [0, 1]")
        if not 0.0 <= self.diversity_preference <= 1.0:
            raise ValueError("diversity_preference must be in [0, 1]")
        if self.budget_usd is not None and self.budget_usd < 0:
            raise ValueError("budget_usd must be non-negative or None")
        if self.latency_ms_sla is not None and self.latency_ms_sla <= 0:
            raise ValueError("latency_ms_sla must be positive or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "complexity_score": self.complexity_score,
            "budget_usd": self.budget_usd,
            "latency_ms_sla": self.latency_ms_sla,
            "quality_floor": self.quality_floor,
            "diversity_preference": self.diversity_preference,
        }


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {v.strip().lower() for v in value.split(",") if v.strip()}


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_gateway_routing_config() -> GatewayRoutingConfig:
    """Load routing criteria from environment variables.

    Environment variables:
      - ARAGORA_ROUTING_FINANCIAL_THRESHOLD
      - ARAGORA_ROUTING_RISK_LEVELS (comma-separated)
      - ARAGORA_ROUTING_COMPLIANCE_FLAGS (comma-separated)
      - ARAGORA_ROUTING_STAKEHOLDER_THRESHOLD
      - ARAGORA_ROUTING_REQUIRE_DEBATE_KEYWORDS (comma-separated)
      - ARAGORA_ROUTING_REQUIRE_EXECUTE_KEYWORDS (comma-separated)
      - ARAGORA_ROUTING_TIME_SENSITIVE_SECONDS
      - ARAGORA_ROUTING_CONFIDENCE_THRESHOLD
      - ARAGORA_ROUTING_CACHE_TTL_SECONDS (default 300 = 5 minutes)
    """

    financial_threshold = _parse_float(
        os.getenv("ARAGORA_ROUTING_FINANCIAL_THRESHOLD"),
        10000.0,
    )
    risk_levels = _parse_csv(os.getenv("ARAGORA_ROUTING_RISK_LEVELS"))
    compliance_flags = _parse_csv(os.getenv("ARAGORA_ROUTING_COMPLIANCE_FLAGS"))
    stakeholder_threshold = _parse_int(
        os.getenv("ARAGORA_ROUTING_STAKEHOLDER_THRESHOLD"),
        3,
    )
    require_debate_keywords = _parse_csv(os.getenv("ARAGORA_ROUTING_REQUIRE_DEBATE_KEYWORDS"))
    require_execute_keywords = _parse_csv(os.getenv("ARAGORA_ROUTING_REQUIRE_EXECUTE_KEYWORDS"))
    time_sensitive_threshold_seconds = _parse_int(
        os.getenv("ARAGORA_ROUTING_TIME_SENSITIVE_SECONDS"),
        60,
    )
    confidence_threshold = _parse_float(
        os.getenv("ARAGORA_ROUTING_CONFIDENCE_THRESHOLD"),
        0.85,
    )
    cache_ttl_seconds = _parse_int(
        os.getenv("ARAGORA_ROUTING_CACHE_TTL_SECONDS"),
        300,
    )

    # Pass parsed sets directly: __post_init__ applies defaults for an empty set
    # exactly as it does for None (``self.x or set(_DEF)``), so this is
    # behavior-preserving and keeps the arg type ``set[str]`` (not ``set[str] | None``).
    return GatewayRoutingConfig(
        financial_threshold=financial_threshold,
        risk_levels=risk_levels,
        compliance_flags=compliance_flags,
        stakeholder_threshold=stakeholder_threshold,
        require_debate_keywords=require_debate_keywords,
        require_execute_keywords=require_execute_keywords,
        time_sensitive_threshold_seconds=time_sensitive_threshold_seconds,
        confidence_threshold=confidence_threshold,
        cache_ttl_seconds=cache_ttl_seconds,
    )
