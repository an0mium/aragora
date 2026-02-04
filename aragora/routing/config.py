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

    return GatewayRoutingConfig(
        financial_threshold=financial_threshold,
        risk_levels=risk_levels or None,
        compliance_flags=compliance_flags or None,
        stakeholder_threshold=stakeholder_threshold,
        require_debate_keywords=require_debate_keywords or None,
        require_execute_keywords=require_execute_keywords or None,
        time_sensitive_threshold_seconds=time_sensitive_threshold_seconds,
        confidence_threshold=confidence_threshold,
    )
