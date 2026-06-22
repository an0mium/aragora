"""A2A capability discovery layer — AGT-02 / #6063 sub-deliverable 3.

Machine-readable catalog of platform capabilities and per-agent summaries so
external agents can discover what Aragora offers without HTML parsing.

Gate: ``ARAGORA_A2A_DISCOVERY_ENABLED`` (default off).  Dataclasses are always
importable; :func:`platform_catalog` and :func:`agent_catalog` raise when off.
Out of scope: HTTP routes, pagination, reputation slices (AGT-05).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "discovery_enabled",
    "DiscoveryError",
    "PlatformCapability",
    "AgentCapabilitySummary",
    "platform_catalog",
    "agent_catalog",
]

_TRUTHY = {"1", "true", "yes", "on"}


def discovery_enabled() -> bool:
    return os.environ.get("ARAGORA_A2A_DISCOVERY_ENABLED", "").lower().strip() in _TRUTHY


def _require_enabled() -> None:
    if not discovery_enabled():
        raise DiscoveryError("Set ARAGORA_A2A_DISCOVERY_ENABLED=1 to enable discovery.")


class DiscoveryError(RuntimeError):
    """Raised when a discovery function is called while the feature gate is off."""


@dataclass(frozen=True)
class PlatformCapability:
    """Versioned, JSON-serializable capability offered by the Aragora platform.

    ``flag_required`` is the env-var for production activation; empty means always-on.
    """

    capability_id: str
    name: str
    description: str
    schema_version: str
    category: str
    flag_required: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "description": self.description,
            "schema_version": self.schema_version,
            "category": self.category,
            "flag_required": self.flag_required,
        }


@dataclass
class AgentCapabilitySummary:
    """Per-agent capability view from the A2A registry.

    ``reputation_stub`` is a placeholder until AGT-05 wires live ERC-8004 scores.
    """

    agent_id: str
    declared_capabilities: list[str] = field(default_factory=list)
    reputation_stub: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "declared_capabilities": list(self.declared_capabilities),
            "reputation_stub": dict(self.reputation_stub),
        }


_PLATFORM_CAPABILITIES: list[PlatformCapability] = [
    PlatformCapability(
        capability_id="aragora.agent.capabilities",
        name="Capability Discovery",
        description="Query platform or per-agent capability catalogs in machine-readable JSON.",
        schema_version="1.0",
        category="registry",
        flag_required="ARAGORA_A2A_DISCOVERY_ENABLED",
    ),
    PlatformCapability(
        capability_id="aragora.agent.register",
        name="Agent Registration",
        description="Register an external agent identity and receive a signed identity receipt.",
        schema_version="1.0",
        category="registry",
    ),
    PlatformCapability(
        capability_id="aragora.claim.verify",
        name="Verify Executable Claim",
        description="Run a verification pass against a named executable claim manifest.",
        schema_version="1.0",
        category="claim",
        flag_required="ARAGORA_EPISTEMIC_CLAIMS_ENABLED",
    ),
    PlatformCapability(
        capability_id="aragora.debate.run",
        name="Run Debate",
        description="Submit a question to Aragora's debate engine and receive a signed receipt.",
        schema_version="1.0",
        category="debate",
    ),
    PlatformCapability(
        capability_id="aragora.knowledge.query",
        name="Knowledge Mound Query",
        description="Semantic search over the Knowledge Mound with provenance-tracked results.",
        schema_version="1.0",
        category="knowledge",
    ),
    PlatformCapability(
        capability_id="aragora.market.predict",
        name="Synthetic Market Prediction",
        description="Record an agent position in a synthetic GitHub prediction market.",
        schema_version="1.0",
        category="market",
        flag_required="ARAGORA_SYNTHETIC_MARKETS_ENABLED",
    ),
    PlatformCapability(
        capability_id="aragora.receipt.fetch",
        name="Fetch Decision Receipt",
        description="Retrieve a signed AgentReceipt for a completed debate or claim.",
        schema_version="1.0",
        category="receipt",
    ),
]


def platform_catalog(
    *, category: str | None = None, require_enabled: bool = True
) -> list[PlatformCapability]:
    """Return platform capabilities sorted by ``capability_id``.

    ``category`` filters to one group; ``None`` returns all.
    ``require_enabled=False`` bypasses the gate for schema-only introspection.
    """
    if require_enabled:
        _require_enabled()
    caps = (
        _PLATFORM_CAPABILITIES
        if category is None
        else [c for c in _PLATFORM_CAPABILITIES if c.category == category]
    )
    return sorted(caps, key=lambda c: c.capability_id)


def agent_catalog(
    agent_id: str, *, server: Any = None, require_enabled: bool = True
) -> AgentCapabilitySummary:
    """Return the capability summary for a registered agent.

    Returns an empty summary when the agent is unknown or ``server`` is ``None``.
    """
    if require_enabled:
        _require_enabled()
    if server is None:
        return AgentCapabilitySummary(agent_id=agent_id)
    card = server.get_agent(agent_id)
    if card is None:
        return AgentCapabilitySummary(agent_id=agent_id)
    raw = getattr(card, "capabilities", None) or []
    return AgentCapabilitySummary(
        agent_id=agent_id,
        declared_capabilities=[str(getattr(c, "value", c)) for c in raw]
        if isinstance(raw, list)
        else [],
    )
