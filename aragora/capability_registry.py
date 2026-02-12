"""
Capability Registry for Aragora.

Maps features across CLI, API (HTTP endpoints), and SDK layers.
Use this module to:
- Discover what features exist across each surface
- Check coverage gaps between layers
- Generate documentation for available capabilities

Usage:
    from aragora.capability_registry import CapabilityRegistry, CoverageLevel

    registry = CapabilityRegistry()

    # Find features with full coverage
    full = registry.get_by_coverage(CoverageLevel.FULL)

    # Get coverage report
    report = registry.coverage_report()

    # Find gaps
    gaps = registry.get_gaps()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CoverageLevel(str, Enum):
    """Coverage level across CLI, API, and SDK."""

    FULL = "full"  # CLI + API + SDK
    PARTIAL = "partial"  # 2 of 3 layers
    MINIMAL = "minimal"  # Only 1 layer
    NONE = "none"  # No implementation


@dataclass
class Capability:
    """A feature/capability that can be exposed via CLI, API, or SDK."""

    name: str
    domain: str
    description: str
    cli_commands: list[str] = field(default_factory=list)
    api_endpoints: list[str] = field(default_factory=list)
    sdk_namespaces: list[str] = field(default_factory=list)
    sdk_methods: list[str] = field(default_factory=list)
    status: str = "stable"  # stable, beta, deprecated

    @property
    def coverage_level(self) -> CoverageLevel:
        """Calculate coverage level based on implementations."""
        layers = sum(
            [
                len(self.cli_commands) > 0,
                len(self.api_endpoints) > 0,
                len(self.sdk_namespaces) > 0 or len(self.sdk_methods) > 0,
            ]
        )
        if layers == 3:
            return CoverageLevel.FULL
        elif layers == 2:
            return CoverageLevel.PARTIAL
        elif layers == 1:
            return CoverageLevel.MINIMAL
        return CoverageLevel.NONE

    @property
    def has_cli(self) -> bool:
        return len(self.cli_commands) > 0

    @property
    def has_api(self) -> bool:
        return len(self.api_endpoints) > 0

    @property
    def has_sdk(self) -> bool:
        return len(self.sdk_namespaces) > 0 or len(self.sdk_methods) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "cli_commands": self.cli_commands,
            "api_endpoints": self.api_endpoints,
            "sdk_namespaces": self.sdk_namespaces,
            "sdk_methods": self.sdk_methods,
            "status": self.status,
            "coverage_level": self.coverage_level.value,
            "has_cli": self.has_cli,
            "has_api": self.has_api,
            "has_sdk": self.has_sdk,
        }


# =============================================================================
# Capability Definitions
# =============================================================================

# Domain: Debates (Full coverage)
DEBATES_CAPABILITY = Capability(
    name="debates",
    domain="core",
    description="Multi-agent debates with consensus detection",
    cli_commands=["gt debate", "gt decide"],
    api_endpoints=[
        "POST /debates",
        "GET /debates",
        "GET /debates/{id}",
        "POST /debates/{id}/vote",
        "POST /debates/{id}/suggest",
        "GET /debates/{id}/export",
    ],
    sdk_namespaces=["debates"],
    sdk_methods=[
        "debates.create",
        "debates.get",
        "debates.list",
        "debates.vote",
        "debates.export",
    ],
)

# Domain: Agents (Full coverage)
AGENTS_CAPABILITY = Capability(
    name="agents",
    domain="core",
    description="AI agent management and configuration",
    cli_commands=["gt status", "gt stats agents"],
    api_endpoints=[
        "GET /agents",
        "GET /agents/{name}",
        "GET /agents/{name}/stats",
        "POST /agents/{name}/configure",
    ],
    sdk_namespaces=["agents"],
    sdk_methods=["agents.list", "agents.get", "agents.configure"],
)

# Domain: Memory (Full coverage)
MEMORY_CAPABILITY = Capability(
    name="memory",
    domain="core",
    description="Multi-tier memory system (fast/medium/slow/glacial)",
    cli_commands=["gt stats memory"],
    api_endpoints=[
        "POST /memory/store",
        "GET /memory/query",
        "POST /memory/promote",
        "GET /memory/tiers",
    ],
    sdk_namespaces=["memory"],
    sdk_methods=["memory.store", "memory.query", "memory.promote"],
)

# Domain: Ranking/ELO (Full coverage)
RANKING_CAPABILITY = Capability(
    name="ranking",
    domain="core",
    description="Agent ELO rankings and leaderboards",
    cli_commands=["gt stats elo", "gt stats leaderboard"],
    api_endpoints=[
        "GET /ranking/leaderboard",
        "GET /ranking/{agent}",
        "GET /ranking/history/{agent}",
    ],
    sdk_namespaces=["ranking", "leaderboard"],
    sdk_methods=["ranking.leaderboard", "ranking.get_rating", "ranking.history"],
)

# Domain: Analytics (Full coverage)
ANALYTICS_CAPABILITY = Capability(
    name="analytics",
    domain="analytics",
    description="Debate analytics and insights",
    cli_commands=["gt stats"],
    api_endpoints=[
        "GET /analytics/debates",
        "GET /analytics/agents",
        "GET /analytics/trends",
        "GET /analytics/dashboard",
    ],
    sdk_namespaces=["analytics", "dashboard"],
    sdk_methods=["analytics.debates", "analytics.agents", "analytics.trends"],
)

# Domain: Knowledge (Partial - missing CLI)
KNOWLEDGE_CAPABILITY = Capability(
    name="knowledge",
    domain="knowledge",
    description="Knowledge Mound and cross-debate learning",
    cli_commands=[],  # Gap: no CLI exposure
    api_endpoints=[
        "POST /knowledge/store",
        "GET /knowledge/search",
        "GET /knowledge/mound/stats",
    ],
    sdk_namespaces=["knowledge", "knowledge_chat"],
    sdk_methods=["knowledge.store", "knowledge.search", "knowledge.chat"],
)

# Domain: Nomic Loop (Partial - missing SDK)
NOMIC_CAPABILITY = Capability(
    name="nomic",
    domain="self-improvement",
    description="Autonomous self-improvement loop",
    cli_commands=["python scripts/nomic_loop.py", "python scripts/self_develop.py"],
    api_endpoints=[
        "POST /nomic/cycle",
        "GET /nomic/status",
        "GET /nomic/history",
    ],
    sdk_namespaces=["nomic"],
    sdk_methods=["nomic.start_cycle", "nomic.status"],
    status="beta",
)

# Domain: Workflow (Partial - missing CLI)
WORKFLOW_CAPABILITY = Capability(
    name="workflow",
    domain="automation",
    description="DAG-based workflow automation",
    cli_commands=[],  # Gap: no CLI exposure
    api_endpoints=[
        "POST /workflows",
        "GET /workflows",
        "GET /workflows/{id}",
        "POST /workflows/{id}/run",
    ],
    sdk_namespaces=["workflows"],
    sdk_methods=["workflows.create", "workflows.list", "workflows.run"],
)

# Domain: Graph Debate (Minimal - only API)
GRAPH_DEBATE_CAPABILITY = Capability(
    name="graph_debate",
    domain="advanced-debate",
    description="Multi-perspective graph-based debates",
    cli_commands=[],  # Gap: no CLI
    api_endpoints=["POST /debates/graph", "GET /debates/graph/{id}"],
    sdk_namespaces=[],  # Gap: no SDK
    sdk_methods=[],
)

# Domain: Matrix Debate (Minimal - only API)
MATRIX_DEBATE_CAPABILITY = Capability(
    name="matrix_debate",
    domain="advanced-debate",
    description="Cross-dimensional matrix debates",
    cli_commands=[],  # Gap: no CLI
    api_endpoints=["POST /debates/matrix", "GET /debates/matrix/{id}"],
    sdk_namespaces=[],  # Gap: no SDK
    sdk_methods=[],
)

# Domain: Gauntlet (Full coverage)
GAUNTLET_CAPABILITY = Capability(
    name="gauntlet",
    domain="verification",
    description="Adversarial verification framework",
    cli_commands=["gt testfix"],
    api_endpoints=[
        "POST /gauntlet/run",
        "GET /gauntlet/findings",
        "GET /gauntlet/receipts",
    ],
    sdk_namespaces=["gauntlet"],
    sdk_methods=["gauntlet.run", "gauntlet.findings", "gauntlet.receipts"],
)

# Domain: Marketplace (Partial - CLI needs unification)
MARKETPLACE_CAPABILITY = Capability(
    name="marketplace",
    domain="ecosystem",
    description="Skill and plugin marketplace",
    cli_commands=["gt tools"],  # Needs: marketplace list/install/publish
    api_endpoints=[
        "GET /marketplace/skills",
        "POST /marketplace/skills/{id}/install",
        "POST /marketplace/skills/publish",
    ],
    sdk_namespaces=["marketplace", "skills"],
    sdk_methods=["marketplace.list", "marketplace.install", "marketplace.publish"],
)

# Domain: Billing (Full coverage)
BILLING_CAPABILITY = Capability(
    name="billing",
    domain="enterprise",
    description="Usage tracking and billing",
    cli_commands=["gt stats costs"],
    api_endpoints=[
        "GET /billing/usage",
        "GET /billing/invoices",
        "GET /billing/forecast",
    ],
    sdk_namespaces=["billing", "cost_management", "budgets"],
    sdk_methods=["billing.usage", "billing.invoices", "billing.forecast"],
)

# Domain: RBAC (Full coverage)
RBAC_CAPABILITY = Capability(
    name="rbac",
    domain="enterprise",
    description="Role-based access control",
    cli_commands=[],  # Could add: gt admin rbac
    api_endpoints=[
        "GET /rbac/roles",
        "POST /rbac/roles",
        "GET /rbac/permissions",
        "POST /rbac/assignments",
    ],
    sdk_namespaces=["rbac"],
    sdk_methods=["rbac.roles", "rbac.permissions", "rbac.assign"],
)

# Domain: Compliance (Partial)
COMPLIANCE_CAPABILITY = Capability(
    name="compliance",
    domain="enterprise",
    description="SOC 2, GDPR, HIPAA compliance",
    cli_commands=[],  # Gap: no CLI
    api_endpoints=[
        "GET /compliance/status",
        "GET /compliance/reports",
        "POST /compliance/scan",
    ],
    sdk_namespaces=["compliance"],
    sdk_methods=["compliance.status", "compliance.reports", "compliance.scan"],
)

# Domain: Pulse (Partial - missing CLI)
PULSE_CAPABILITY = Capability(
    name="pulse",
    domain="intelligence",
    description="Trending topics from HackerNews, Reddit, Twitter",
    cli_commands=[],  # Gap: no CLI
    api_endpoints=[
        "GET /pulse/trending",
        "GET /pulse/sources",
        "POST /pulse/ingest",
    ],
    sdk_namespaces=["pulse"],
    sdk_methods=["pulse.trending", "pulse.sources"],
)

# Domain: Explainability (Full coverage)
EXPLAINABILITY_CAPABILITY = Capability(
    name="explainability",
    domain="transparency",
    description="Decision explanations and factor analysis",
    cli_commands=["gt decide --explain"],
    api_endpoints=[
        "GET /explainability/{debate_id}",
        "GET /explainability/{debate_id}/factors",
        "GET /explainability/{debate_id}/counterfactuals",
    ],
    sdk_namespaces=["explainability"],
    sdk_methods=[
        "explainability.explain",
        "explainability.factors",
        "explainability.counterfactuals",
    ],
)

# Domain: Backups (Partial - missing CLI)
BACKUPS_CAPABILITY = Capability(
    name="backups",
    domain="operations",
    description="Disaster recovery and backups",
    cli_commands=[],  # Gap: no CLI (could add: gt admin backup)
    api_endpoints=[
        "POST /backups",
        "GET /backups",
        "POST /backups/{id}/restore",
    ],
    sdk_namespaces=["backups"],
    sdk_methods=["backups.create", "backups.list", "backups.restore"],
)

# Domain: Calibration (Full coverage)
CALIBRATION_CAPABILITY = Capability(
    name="calibration",
    domain="tracking",
    description="Agent prediction calibration tracking",
    cli_commands=["gt stats calibration"],
    api_endpoints=[
        "GET /calibration/{agent}",
        "GET /calibration/leaderboard",
    ],
    sdk_namespaces=["calibration"],
    sdk_methods=["calibration.get", "calibration.leaderboard"],
)

# Domain: Consensus Memory (Partial)
CONSENSUS_MEMORY_CAPABILITY = Capability(
    name="consensus_memory",
    domain="knowledge",
    description="Historical consensus outcomes",
    cli_commands=[],  # Gap
    api_endpoints=[
        "GET /consensus/history",
        "GET /consensus/{topic}",
    ],
    sdk_namespaces=["consensus"],
    sdk_methods=["consensus.history", "consensus.get"],
)

# Domain: Health/Monitoring (Full coverage)
HEALTH_CAPABILITY = Capability(
    name="health",
    domain="operations",
    description="System health and monitoring",
    cli_commands=["gt status"],
    api_endpoints=[
        "GET /health",
        "GET /health/detailed",
        "GET /metrics",
    ],
    sdk_namespaces=["health", "monitoring", "metrics"],
    sdk_methods=["health.check", "health.detailed", "metrics.get"],
)

# Domain: SME (Full coverage)
SME_CAPABILITY = Capability(
    name="sme",
    domain="verticals",
    description="SME business features (AP/AR, invoicing)",
    cli_commands=[],  # Could add SME-specific commands
    api_endpoints=[
        "POST /sme/invoices/process",
        "GET /sme/cash-flow",
        "POST /sme/expenses/categorize",
    ],
    sdk_namespaces=["sme", "invoice_processing", "ar_automation", "expenses"],
    sdk_methods=["sme.process_invoice", "sme.cash_flow", "sme.categorize_expenses"],
)

# Domain: Bots (Full coverage)
BOTS_CAPABILITY = Capability(
    name="bots",
    domain="integrations",
    description="Slack, Teams, Telegram, WhatsApp bots",
    cli_commands=["gt server"],  # Server includes bot endpoints
    api_endpoints=[
        "POST /bots/slack/webhook",
        "POST /bots/teams/webhook",
        "POST /bots/telegram/webhook",
        "POST /bots/whatsapp/webhook",
    ],
    sdk_namespaces=["bots", "chat"],
    sdk_methods=["bots.send", "bots.configure", "chat.send"],
)


# =============================================================================
# Capability Registry
# =============================================================================


class CapabilityRegistry:
    """Registry of all Aragora capabilities across CLI, API, and SDK."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._load_default_capabilities()

    def _load_default_capabilities(self) -> None:
        """Load all default capability definitions."""
        defaults = [
            DEBATES_CAPABILITY,
            AGENTS_CAPABILITY,
            MEMORY_CAPABILITY,
            RANKING_CAPABILITY,
            ANALYTICS_CAPABILITY,
            KNOWLEDGE_CAPABILITY,
            NOMIC_CAPABILITY,
            WORKFLOW_CAPABILITY,
            GRAPH_DEBATE_CAPABILITY,
            MATRIX_DEBATE_CAPABILITY,
            GAUNTLET_CAPABILITY,
            MARKETPLACE_CAPABILITY,
            BILLING_CAPABILITY,
            RBAC_CAPABILITY,
            COMPLIANCE_CAPABILITY,
            PULSE_CAPABILITY,
            EXPLAINABILITY_CAPABILITY,
            BACKUPS_CAPABILITY,
            CALIBRATION_CAPABILITY,
            CONSENSUS_MEMORY_CAPABILITY,
            HEALTH_CAPABILITY,
            SME_CAPABILITY,
            BOTS_CAPABILITY,
        ]
        for cap in defaults:
            self._capabilities[cap.name] = cap

    def register(self, capability: Capability) -> None:
        """Register a new capability."""
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> Capability | None:
        """Get a capability by name."""
        return self._capabilities.get(name)

    def all(self) -> list[Capability]:
        """Get all registered capabilities."""
        return list(self._capabilities.values())

    def get_by_domain(self, domain: str) -> list[Capability]:
        """Get capabilities by domain."""
        return [c for c in self._capabilities.values() if c.domain == domain]

    def get_by_coverage(self, level: CoverageLevel) -> list[Capability]:
        """Get capabilities by coverage level."""
        return [c for c in self._capabilities.values() if c.coverage_level == level]

    def get_gaps(self) -> dict[str, list[Capability]]:
        """Get capabilities with coverage gaps."""
        return {
            "missing_cli": [c for c in self._capabilities.values() if not c.has_cli],
            "missing_api": [c for c in self._capabilities.values() if not c.has_api],
            "missing_sdk": [c for c in self._capabilities.values() if not c.has_sdk],
        }

    def coverage_report(self) -> dict[str, Any]:
        """Generate a coverage report."""
        caps = list(self._capabilities.values())
        full = [c for c in caps if c.coverage_level == CoverageLevel.FULL]
        partial = [c for c in caps if c.coverage_level == CoverageLevel.PARTIAL]
        minimal = [c for c in caps if c.coverage_level == CoverageLevel.MINIMAL]

        return {
            "total_capabilities": len(caps),
            "full_coverage": len(full),
            "partial_coverage": len(partial),
            "minimal_coverage": len(minimal),
            "coverage_percentage": ((len(full) / len(caps) * 100) if caps else 0),
            "by_domain": self._group_by_domain(),
            "gaps": {
                "cli": [c.name for c in caps if not c.has_cli],
                "api": [c.name for c in caps if not c.has_api],
                "sdk": [c.name for c in caps if not c.has_sdk],
            },
            "full_coverage_list": [c.name for c in full],
            "needs_attention": [c.name for c in partial + minimal],
        }

    def _group_by_domain(self) -> dict[str, list[str]]:
        """Group capabilities by domain."""
        domains: dict[str, list[str]] = {}
        for cap in self._capabilities.values():
            if cap.domain not in domains:
                domains[cap.domain] = []
            domains[cap.domain].append(cap.name)
        return domains

    def to_markdown(self) -> str:
        """Generate markdown documentation of capabilities."""
        lines = [
            "# Aragora Capability Registry",
            "",
            "## Coverage Summary",
            "",
        ]

        report = self.coverage_report()
        lines.extend(
            [
                f"- **Total Capabilities**: {report['total_capabilities']}",
                f"- **Full Coverage (CLI+API+SDK)**: {report['full_coverage']}",
                f"- **Partial Coverage**: {report['partial_coverage']}",
                f"- **Minimal Coverage**: {report['minimal_coverage']}",
                "",
                "## Capabilities by Domain",
                "",
            ]
        )

        for domain in sorted(report["by_domain"].keys()):
            lines.append(f"### {domain.title()}")
            lines.append("")
            for cap_name in report["by_domain"][domain]:
                cap = self._capabilities[cap_name]
                status = f"[{cap.coverage_level.value}]"
                lines.append(f"- **{cap.name}** {status}: {cap.description}")
            lines.append("")

        lines.extend(
            [
                "## Gaps",
                "",
                "### Missing CLI",
                "",
            ]
        )
        for name in report["gaps"]["cli"]:
            lines.append(f"- {name}")

        lines.extend(
            [
                "",
                "### Missing SDK",
                "",
            ]
        )
        for name in report["gaps"]["sdk"]:
            lines.append(f"- {name}")

        return "\n".join(lines)


# Singleton instance
_registry: CapabilityRegistry | None = None


def get_capability_registry() -> CapabilityRegistry:
    """Get the singleton capability registry."""
    global _registry
    if _registry is None:
        _registry = CapabilityRegistry()
    return _registry


__all__ = [
    "Capability",
    "CapabilityRegistry",
    "CoverageLevel",
    "get_capability_registry",
]
