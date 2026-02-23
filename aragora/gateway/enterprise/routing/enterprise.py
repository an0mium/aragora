"""
Enterprise Tenant Routing Handler.

Provides specialized routing logic for enterprise tenants with advanced features:
- SLA-based endpoint selection
- Dedicated resource pools
- Priority routing
- Custom compliance requirements
- Advanced load balancing

Usage:
    from aragora.gateway.enterprise.routing.enterprise import (
        EnterpriseTenantHandler,
        EnterpriseRoutingConfig,
    )

    handler = EnterpriseTenantHandler()
    config = EnterpriseRoutingConfig(
        tenant_id="acme-corp",
        sla_tier="platinum",
        dedicated_endpoints=True,
    )
    endpoint = await handler.select_endpoint(config, available_endpoints)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


from .quotas import TenantQuotas

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class SLATier(str, Enum):
    """SLA tier levels for enterprise tenants."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    CUSTOM = "custom"


class ComplianceRequirement(str, Enum):
    """Compliance requirements for enterprise tenants."""

    SOC2 = "soc2"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    FedRAMP = "fedramp"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EnterpriseRoutingConfig:
    """
    Routing configuration for enterprise tenants.

    Attributes:
        tenant_id: Unique tenant identifier.
        sla_tier: SLA tier level for this tenant.
        dedicated_endpoints: Whether tenant has dedicated endpoints.
        compliance_requirements: Set of compliance requirements.
        priority_weight: Priority weight for routing (higher = more priority).
        max_latency_ms: Maximum acceptable latency in milliseconds.
        failover_enabled: Whether automatic failover is enabled.
        geographic_affinity: Preferred geographic region for routing.
        custom_headers: Custom headers to include in requests.
        metadata: Additional configuration metadata.
    """

    tenant_id: str
    sla_tier: SLATier = SLATier.SILVER
    dedicated_endpoints: bool = False
    compliance_requirements: set[ComplianceRequirement] = field(default_factory=set)
    priority_weight: int = 100
    max_latency_ms: float = 1000.0
    failover_enabled: bool = True
    geographic_affinity: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_quota_multiplier(self) -> float:
        """Get quota multiplier based on SLA tier."""
        multipliers = {
            SLATier.BRONZE: 1.0,
            SLATier.SILVER: 2.0,
            SLATier.GOLD: 5.0,
            SLATier.PLATINUM: 10.0,
            SLATier.CUSTOM: self.metadata.get("quota_multiplier", 1.0),
        }
        return multipliers.get(self.sla_tier, 1.0)


@dataclass
class EnterpriseEndpoint:
    """
    Endpoint configuration for enterprise routing.

    Attributes:
        url: Base URL of the endpoint.
        region: Geographic region of the endpoint.
        compliance_certifications: Compliance certifications this endpoint has.
        is_dedicated: Whether this is a dedicated endpoint.
        priority: Priority for failover (lower = higher priority).
        weight: Weight for load balancing.
        max_connections: Maximum concurrent connections.
        health_check_interval: Health check interval in seconds.
    """

    url: str
    region: str = "us-east-1"
    compliance_certifications: set[ComplianceRequirement] = field(default_factory=set)
    is_dedicated: bool = False
    priority: int = 1
    weight: int = 100
    max_connections: int = 100
    health_check_interval: float = 30.0


@dataclass
class EnterpriseRoutingDecision:
    """
    Routing decision for enterprise tenants.

    Attributes:
        selected_endpoint: The selected endpoint URL.
        sla_tier: SLA tier used for selection.
        compliance_satisfied: Whether compliance requirements are met.
        latency_estimate_ms: Estimated latency in milliseconds.
        decision_reason: Human-readable reason for the decision.
        fallback_used: Whether a fallback was used.
        metadata: Additional decision metadata.
    """

    selected_endpoint: str
    sla_tier: SLATier
    compliance_satisfied: bool = True
    latency_estimate_ms: float = 0.0
    decision_reason: str = ""
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Enterprise Tenant Handler
# =============================================================================


class EnterpriseTenantHandler:
    """
    Handler for enterprise tenant routing decisions.

    Implements advanced routing logic for enterprise tenants with
    SLA-based selection, compliance filtering, and geographic affinity.
    """

    def __init__(self) -> None:
        """Initialize enterprise tenant handler."""
        self._sla_latency_targets = {
            SLATier.BRONZE: 2000.0,
            SLATier.SILVER: 1000.0,
            SLATier.GOLD: 500.0,
            SLATier.PLATINUM: 200.0,
            SLATier.CUSTOM: 100.0,
        }

    def get_enhanced_quotas(
        self,
        base_quotas: TenantQuotas,
        config: EnterpriseRoutingConfig,
    ) -> TenantQuotas:
        """
        Get enhanced quotas for enterprise tenant based on SLA tier.

        Args:
            base_quotas: Base quota configuration.
            config: Enterprise routing configuration.

        Returns:
            Enhanced TenantQuotas with SLA multipliers applied.
        """
        multiplier = config.get_quota_multiplier()

        return TenantQuotas(
            requests_per_minute=int(base_quotas.requests_per_minute * multiplier),
            requests_per_hour=int(base_quotas.requests_per_hour * multiplier),
            requests_per_day=int(base_quotas.requests_per_day * multiplier),
            concurrent_requests=int(base_quotas.concurrent_requests * multiplier),
            bandwidth_bytes_per_minute=int(base_quotas.bandwidth_bytes_per_minute * multiplier),
            warn_threshold=base_quotas.warn_threshold,
        )

    def filter_compliant_endpoints(
        self,
        endpoints: list[EnterpriseEndpoint],
        requirements: set[ComplianceRequirement],
    ) -> list[EnterpriseEndpoint]:
        """
        Filter endpoints that meet compliance requirements.

        Args:
            endpoints: List of available endpoints.
            requirements: Required compliance certifications.

        Returns:
            List of endpoints meeting all compliance requirements.
        """
        if not requirements:
            return endpoints

        compliant = []
        for endpoint in endpoints:
            if requirements.issubset(endpoint.compliance_certifications):
                compliant.append(endpoint)

        return compliant

    def filter_by_region(
        self,
        endpoints: list[EnterpriseEndpoint],
        preferred_region: str | None,
    ) -> list[EnterpriseEndpoint]:
        """
        Sort endpoints by geographic affinity.

        Args:
            endpoints: List of available endpoints.
            preferred_region: Preferred geographic region.

        Returns:
            Endpoints sorted by regional preference.
        """
        if not preferred_region:
            return endpoints

        # Prioritize endpoints in the preferred region
        in_region = [e for e in endpoints if e.region == preferred_region]
        out_of_region = [e for e in endpoints if e.region != preferred_region]

        return in_region + out_of_region

    def select_by_priority(
        self,
        endpoints: list[EnterpriseEndpoint],
    ) -> EnterpriseEndpoint | None:
        """
        Select endpoint based on priority.

        Args:
            endpoints: List of available endpoints.

        Returns:
            Highest priority endpoint or None.
        """
        if not endpoints:
            return None

        # Sort by priority (lower = higher priority), then by weight (higher = preferred)
        sorted_endpoints = sorted(endpoints, key=lambda e: (e.priority, -e.weight))
        return sorted_endpoints[0]

    def select_dedicated_endpoint(
        self,
        endpoints: list[EnterpriseEndpoint],
        tenant_id: str,
    ) -> EnterpriseEndpoint | None:
        """
        Select a dedicated endpoint for the tenant.

        Args:
            endpoints: List of available endpoints.
            tenant_id: Tenant identifier.

        Returns:
            Dedicated endpoint for the tenant or None.
        """
        dedicated = [e for e in endpoints if e.is_dedicated]
        if dedicated:
            return dedicated[0]
        return None

    async def select_endpoint(
        self,
        config: EnterpriseRoutingConfig,
        endpoints: list[EnterpriseEndpoint],
        latencies: dict[str, float] | None = None,
    ) -> EnterpriseRoutingDecision:
        """
        Select the best endpoint for an enterprise tenant.

        Args:
            config: Enterprise routing configuration.
            endpoints: List of available endpoints.
            latencies: Optional dictionary of endpoint URL to latency in ms.

        Returns:
            EnterpriseRoutingDecision with the selected endpoint.
        """
        if not endpoints:
            return EnterpriseRoutingDecision(
                selected_endpoint="",
                sla_tier=config.sla_tier,
                compliance_satisfied=False,
                decision_reason="No endpoints available",
            )

        # Check for dedicated endpoints first
        if config.dedicated_endpoints:
            dedicated = self.select_dedicated_endpoint(endpoints, config.tenant_id)
            if dedicated:
                return EnterpriseRoutingDecision(
                    selected_endpoint=dedicated.url,
                    sla_tier=config.sla_tier,
                    compliance_satisfied=True,
                    decision_reason="Dedicated endpoint selected",
                    metadata={"dedicated": True},
                )

        # Filter by compliance requirements
        compliant = self.filter_compliant_endpoints(endpoints, config.compliance_requirements)
        compliance_satisfied = len(compliant) > 0

        if not compliant:
            logger.warning(
                "No compliant endpoints for tenant %s, requirements: %s",
                config.tenant_id,
                config.compliance_requirements,
            )
            # Fall back to all endpoints if no compliant ones
            compliant = endpoints
            compliance_satisfied = False

        # Filter by geographic affinity
        regional = self.filter_by_region(compliant, config.geographic_affinity)

        # Filter by latency if available
        target_latency = self._sla_latency_targets.get(config.sla_tier, 1000.0)
        if config.max_latency_ms > 0:
            target_latency = min(target_latency, config.max_latency_ms)

        if latencies:
            low_latency = [e for e in regional if latencies.get(e.url, 0) <= target_latency]
            if low_latency:
                regional = low_latency

        # Select by priority
        selected = self.select_by_priority(regional)

        if selected:
            latency_estimate = latencies.get(selected.url, 0.0) if latencies else 0.0
            return EnterpriseRoutingDecision(
                selected_endpoint=selected.url,
                sla_tier=config.sla_tier,
                compliance_satisfied=compliance_satisfied,
                latency_estimate_ms=latency_estimate,
                decision_reason=self._build_decision_reason(config, selected, compliance_satisfied),
                metadata={
                    "region": selected.region,
                    "priority": selected.priority,
                },
            )

        # Fallback - use first available endpoint
        fallback = endpoints[0]
        return EnterpriseRoutingDecision(
            selected_endpoint=fallback.url,
            sla_tier=config.sla_tier,
            compliance_satisfied=False,
            decision_reason="Fallback endpoint selected",
            fallback_used=True,
        )

    def _build_decision_reason(
        self,
        config: EnterpriseRoutingConfig,
        endpoint: EnterpriseEndpoint,
        compliance_satisfied: bool,
    ) -> str:
        """Build a human-readable decision reason."""
        reasons = []

        if config.dedicated_endpoints and endpoint.is_dedicated:
            reasons.append("dedicated")
        if config.geographic_affinity and endpoint.region == config.geographic_affinity:
            reasons.append(f"region={endpoint.region}")
        if compliance_satisfied:
            reasons.append("compliant")
        reasons.append(f"priority={endpoint.priority}")
        reasons.append(f"sla={config.sla_tier.value}")

        return "Selected: " + ", ".join(reasons)

    def get_sla_latency_target(self, sla_tier: SLATier) -> float:
        """
        Get the latency target for an SLA tier.

        Args:
            sla_tier: The SLA tier.

        Returns:
            Target latency in milliseconds.
        """
        return self._sla_latency_targets.get(sla_tier, 1000.0)

    def validate_sla_compliance(
        self,
        config: EnterpriseRoutingConfig,
        actual_latency_ms: float,
    ) -> dict[str, Any]:
        """
        Validate that SLA latency requirements are being met.

        Args:
            config: Enterprise routing configuration.
            actual_latency_ms: Actual observed latency.

        Returns:
            Dictionary with SLA compliance status.
        """
        target = self.get_sla_latency_target(config.sla_tier)
        is_compliant = actual_latency_ms <= target

        return {
            "sla_tier": config.sla_tier.value,
            "target_latency_ms": target,
            "actual_latency_ms": actual_latency_ms,
            "is_compliant": is_compliant,
            "margin_ms": target - actual_latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


__all__ = [
    "SLATier",
    "ComplianceRequirement",
    "EnterpriseRoutingConfig",
    "EnterpriseEndpoint",
    "EnterpriseRoutingDecision",
    "EnterpriseTenantHandler",
]
