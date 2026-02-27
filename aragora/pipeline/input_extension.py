"""Input Extension Engine.

Extends user inputs with implications, constraints, prior art,
and considerations the user didn't explicitly state but would
want to know about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Implication:
    """Something implied by the user's request that they didn't state."""

    statement: str
    confidence: float = 0.0  # 0-1, how confident we are this is implied
    source: str = ""  # where this implication was derived from
    category: str = "general"  # technical, business, security, compliance


@dataclass
class Constraint:
    """A constraint the user would likely want if they knew about it."""

    description: str
    reason: str
    severity: str = "recommended"  # required, recommended, optional
    category: str = "general"


@dataclass
class PriorArt:
    """Existing solution or approach relevant to the user's request."""

    title: str
    description: str
    relevance: float = 0.0  # 0-1
    source_url: str = ""
    source_type: str = ""  # "internal" (KM), "obsidian", "web"


@dataclass
class ExtendedInput:
    """User's original input enriched with extensions."""

    original_prompt: str
    implications: list[Implication] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    prior_art: list[PriorArt] = field(default_factory=list)
    domain_context: str = ""
    risk_factors: list[str] = field(default_factory=list)

    @property
    def has_extensions(self) -> bool:
        return bool(self.implications or self.constraints or self.prior_art)

    @property
    def high_confidence_implications(self) -> list[Implication]:
        """Implications with confidence >= 0.7."""
        return [i for i in self.implications if i.confidence >= 0.7]

    @property
    def required_constraints(self) -> list[Constraint]:
        """Constraints marked as required."""
        return [c for c in self.constraints if c.severity == "required"]

    def to_context_block(self) -> str:
        """Format extensions as a context block for prompts."""
        parts: list[str] = []

        if self.implications:
            parts.append("## Implied Requirements")
            for imp in sorted(self.implications, key=lambda i: i.confidence, reverse=True):
                parts.append(
                    f"- [{imp.category}] {imp.statement} (confidence: {imp.confidence:.0%})"
                )

        if self.constraints:
            parts.append("\n## Recommended Constraints")
            for con in self.constraints:
                parts.append(f"- [{con.severity}] {con.description}: {con.reason}")

        if self.prior_art:
            parts.append("\n## Prior Art")
            for pa in sorted(self.prior_art, key=lambda p: p.relevance, reverse=True)[:5]:
                parts.append(f"- {pa.title}: {pa.description}")

        if self.risk_factors:
            parts.append("\n## Risk Factors")
            for risk in self.risk_factors:
                parts.append(f"- {risk}")

        return "\n".join(parts) if parts else ""


class InputExtensionEngine:
    """Extends user inputs with context they didn't provide.

    Uses research from KnowledgeMound and other sources to surface
    implications, constraints, and prior art that enrich the user's
    original prompt before debate and spec generation.
    """

    def __init__(
        self,
        knowledge_mound: Any | None = None,
        researcher: Any | None = None,
    ) -> None:
        self._km = knowledge_mound
        self._researcher = researcher

    async def extend(
        self,
        prompt: str,
        domain: str = "",
        research_context: Any | None = None,
    ) -> ExtendedInput:
        """Extend a user prompt with inferred context.

        Args:
            prompt: The user's original input.
            domain: Domain context (e.g., "technical", "business").
            research_context: Pre-gathered research from UnifiedResearcher.

        Returns:
            ExtendedInput with enrichments.
        """
        result = ExtendedInput(original_prompt=prompt, domain_context=domain)

        # Extract implications from research context
        if research_context is not None:
            result.implications.extend(self._extract_implications(prompt, research_context))

        # Query KM for prior art
        if self._km is not None:
            result.prior_art.extend(self._find_prior_art(prompt))

        # Infer constraints from domain
        if domain:
            result.constraints.extend(self._infer_domain_constraints(domain))

        # Detect risk factors
        result.risk_factors = self._detect_risks(prompt, domain)

        return result

    def _extract_implications(self, prompt: str, research_context: Any) -> list[Implication]:
        """Extract implications from research findings."""
        implications: list[Implication] = []

        if hasattr(research_context, "results"):
            for r in research_context.results:
                if hasattr(r, "content") and hasattr(r, "relevance"):
                    if r.relevance >= 0.6:
                        implications.append(
                            Implication(
                                statement=f"Based on prior work: {r.content[:200]}",
                                confidence=r.relevance,
                                source=getattr(r, "source", "research"),
                                category="technical",
                            )
                        )

        return implications[:5]  # Cap at 5 implications

    def _find_prior_art(self, prompt: str) -> list[PriorArt]:
        """Find relevant prior art from KnowledgeMound."""
        if self._km is None:
            return []

        try:
            results = self._km.query(prompt, limit=3)
            if not results:
                return []

            items = results if isinstance(results, list) else [results]
            return [
                PriorArt(
                    title=str(r.get("title", r.get("type", "Prior decision"))),
                    description=str(r.get("content", r.get("text", str(r))))[:300],
                    relevance=float(r.get("relevance", r.get("score", 0.5))),
                    source_type="internal",
                )
                for r in items
            ][:3]
        except Exception:
            logger.warning("Failed to query KM for prior art")
            return []

    def _infer_domain_constraints(self, domain: str) -> list[Constraint]:
        """Infer standard constraints for a domain."""
        domain_constraints: dict[str, list[Constraint]] = {
            "security": [
                Constraint(
                    description="Input validation on all user-facing endpoints",
                    reason="OWASP Top 10 requirement",
                    severity="required",
                    category="security",
                ),
                Constraint(
                    description="No secrets in source code or logs",
                    reason="Credential leak prevention",
                    severity="required",
                    category="security",
                ),
            ],
            "compliance": [
                Constraint(
                    description="Audit trail for all state changes",
                    reason="Regulatory compliance",
                    severity="required",
                    category="compliance",
                ),
            ],
            "technical": [
                Constraint(
                    description="Backward-compatible API changes",
                    reason="Avoid breaking existing consumers",
                    severity="recommended",
                    category="technical",
                ),
            ],
            "healthcare": [
                Constraint(
                    description="HIPAA-compliant data handling",
                    reason="Protected health information regulations",
                    severity="required",
                    category="compliance",
                ),
            ],
            "financial": [
                Constraint(
                    description="SOX-compliant audit trails",
                    reason="Financial reporting regulations",
                    severity="required",
                    category="compliance",
                ),
            ],
        }
        return domain_constraints.get(domain, [])

    def _detect_risks(self, prompt: str, domain: str) -> list[str]:
        """Detect risk factors from prompt and domain."""
        risks: list[str] = []
        prompt_lower = prompt.lower()

        risk_keywords = {
            "migration": "Data migration risks: potential data loss, downtime",
            "delete": "Destructive operation: ensure rollback capability",
            "auth": "Authentication changes: risk of lockout or bypass",
            "payment": "Payment system: financial risk, PCI compliance needed",
            "database": "Schema change: migration complexity, rollback plan needed",
            "api": "API change: breaking change risk for consumers",
            "deploy": "Deployment risk: rollback plan, canary deployment recommended",
        }

        for keyword, risk in risk_keywords.items():
            if keyword in prompt_lower:
                risks.append(risk)

        return risks
