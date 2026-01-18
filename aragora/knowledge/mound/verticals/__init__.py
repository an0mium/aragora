"""
Vertical-specific knowledge modules for the Knowledge Mound.

Provides domain-specific fact extraction, validation, and pattern detection
for different enterprise verticals (software, legal, healthcare, accounting, research).

Each vertical module implements:
- Domain-specific fact extraction from content
- Fact validation with domain expertise
- Pattern detection across facts
- Compliance checking against relevant regulations

Usage:
    from aragora.knowledge.mound.verticals import (
        VerticalRegistry,
        BaseVerticalKnowledge,
        VerticalFact,
        VerticalCapabilities,
    )

    # Get a vertical module
    software = VerticalRegistry.get("software")

    # Extract facts from code
    facts = await software.extract_facts(code_content, {"language": "python"})

    # Validate a fact
    is_valid, confidence = await software.validate_fact(fact)
"""

from aragora.knowledge.mound.verticals.base import (
    BaseVerticalKnowledge,
    ComplianceCheckResult,
    PatternMatch,
    VerticalCapabilities,
    VerticalFact,
)
from aragora.knowledge.mound.verticals.registry import VerticalRegistry

# Import specific verticals (auto-registered via registry)
from aragora.knowledge.mound.verticals.software import SoftwareKnowledge
from aragora.knowledge.mound.verticals.legal import LegalKnowledge
from aragora.knowledge.mound.verticals.healthcare import HealthcareKnowledge
from aragora.knowledge.mound.verticals.accounting import AccountingKnowledge
from aragora.knowledge.mound.verticals.research import ResearchKnowledge

__all__ = [
    # Base classes
    "BaseVerticalKnowledge",
    "ComplianceCheckResult",
    "PatternMatch",
    "VerticalCapabilities",
    "VerticalFact",
    "VerticalRegistry",
    # Vertical implementations
    "SoftwareKnowledge",
    "LegalKnowledge",
    "HealthcareKnowledge",
    "AccountingKnowledge",
    "ResearchKnowledge",
]
