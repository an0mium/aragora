"""
Base classes for vertical-specific knowledge modules.

Defines the abstract interface that all vertical knowledge modules must implement,
enabling the Knowledge Mound to work with domain-specific fact extraction,
validation, and pattern detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence
import logging

logger = logging.getLogger(__name__)


@dataclass
class VerticalCapabilities:
    """
    Declares what a vertical knowledge module can do.

    Used for capability discovery and appropriate module selection.
    """

    # Analysis modes
    supports_pattern_detection: bool = True
    supports_cross_reference: bool = False
    supports_compliance_check: bool = False

    # Requirements
    requires_llm: bool = False
    requires_vector_search: bool = True

    # Domain-specific
    pattern_categories: list[str] = field(default_factory=list)
    compliance_frameworks: list[str] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)


@dataclass
class VerticalFact:
    """
    A domain-specific fact with vertical context.

    Facts are the atomic units of knowledge in the Knowledge Mound.
    Each fact has domain-specific context and staleness tracking.
    """

    id: str
    vertical: str  # "software", "legal", "healthcare", etc.
    content: str
    category: str  # Domain-specific category
    confidence: float
    staleness_days: float = 0.0  # Days since last verification
    decay_rate: float = 0.1  # Confidence decay per day (vertical-specific)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    verified_at: datetime = field(default_factory=datetime.now)

    @property
    def adjusted_confidence(self) -> float:
        """Confidence adjusted for staleness."""
        decay = min(0.9, self.staleness_days * self.decay_rate)
        return max(0.0, self.confidence * (1.0 - decay))

    @property
    def needs_reverification(self) -> bool:
        """Check if fact is stale enough to need reverification."""
        return self.adjusted_confidence < 0.5 * self.confidence

    def refresh(self, new_confidence: Optional[float] = None) -> None:
        """Mark fact as freshly verified."""
        self.verified_at = datetime.now()
        self.staleness_days = 0.0
        if new_confidence is not None:
            self.confidence = new_confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "vertical": self.vertical,
            "content": self.content,
            "category": self.category,
            "confidence": self.confidence,
            "adjusted_confidence": self.adjusted_confidence,
            "staleness_days": self.staleness_days,
            "decay_rate": self.decay_rate,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "verified_at": self.verified_at.isoformat(),
        }


@dataclass
class PatternMatch:
    """A detected pattern across facts."""

    pattern_id: str
    pattern_name: str
    pattern_type: str
    description: str
    confidence: float
    supporting_facts: list[str]  # Fact IDs
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check."""

    rule_id: str
    rule_name: str
    framework: str
    passed: bool
    severity: str  # "critical", "high", "medium", "low", "info"
    findings: list[str]
    evidence: list[str]
    recommendations: list[str]
    confidence: float


class BaseVerticalKnowledge(ABC):
    """
    Abstract base class for vertical-specific knowledge modules.

    Each vertical (software, legal, healthcare, accounting, research)
    implements this interface to provide domain-specific:
    - Fact extraction from content
    - Fact validation
    - Pattern detection
    - Compliance checking

    Follows the pattern from aragora.audit.base_auditor for consistency.
    """

    @property
    @abstractmethod
    def vertical_id(self) -> str:
        """Unique identifier (e.g., 'software', 'legal')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of this vertical's knowledge domain."""
        ...

    @property
    def capabilities(self) -> VerticalCapabilities:
        """Declare capabilities of this vertical."""
        return VerticalCapabilities()

    @property
    def decay_rates(self) -> dict[str, float]:
        """
        Category-specific decay rates for fact staleness.

        Override to provide vertical-specific rates.
        Default: 0.02 per day (~50% confidence after 35 days).
        """
        return {"default": 0.02}

    def get_decay_rate(self, category: str) -> float:
        """Get decay rate for a category."""
        rates = self.decay_rates
        return rates.get(category, rates.get("default", 0.02))

    # -------------------------------------------------------------------------
    # Fact Extraction
    # -------------------------------------------------------------------------

    @abstractmethod
    async def extract_facts(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[VerticalFact]:
        """
        Extract domain-specific facts from content.

        Args:
            content: Text content to analyze
            metadata: Optional context (e.g., file type, source)

        Returns:
            List of extracted facts with confidence scores
        """
        ...

    # -------------------------------------------------------------------------
    # Fact Validation
    # -------------------------------------------------------------------------

    @abstractmethod
    async def validate_fact(
        self,
        fact: VerticalFact,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, float]:
        """
        Validate a fact, return (is_valid, new_confidence).

        Args:
            fact: Fact to validate
            context: Optional validation context

        Returns:
            Tuple of (is_valid, new_confidence)
        """
        ...

    # -------------------------------------------------------------------------
    # Pattern Detection (Optional)
    # -------------------------------------------------------------------------

    async def detect_patterns(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[PatternMatch]:
        """
        Detect patterns across facts.

        Override if vertical supports pattern detection.

        Args:
            facts: Facts to analyze for patterns

        Returns:
            List of detected patterns
        """
        return []

    # -------------------------------------------------------------------------
    # Compliance Checking (Optional)
    # -------------------------------------------------------------------------

    async def check_compliance(
        self,
        facts: Sequence[VerticalFact],
        framework: str,
    ) -> list[ComplianceCheckResult]:
        """
        Check compliance against a framework.

        Override if vertical supports compliance checking.

        Args:
            facts: Facts to check
            framework: Compliance framework (e.g., "OWASP", "HIPAA", "SOX")

        Returns:
            List of compliance check results
        """
        return []

    async def get_compliance_frameworks(self) -> list[str]:
        """Get supported compliance frameworks."""
        return self.capabilities.compliance_frameworks

    # -------------------------------------------------------------------------
    # Cross-Reference (Optional)
    # -------------------------------------------------------------------------

    async def cross_reference(
        self,
        fact: VerticalFact,
        other_facts: Sequence[VerticalFact],
    ) -> list[tuple[str, str, float]]:
        """
        Find related facts via cross-reference.

        Override if vertical supports cross-reference.

        Args:
            fact: Fact to find references for
            other_facts: Pool of facts to search

        Returns:
            List of (fact_id, relationship_type, confidence)
        """
        return []

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def create_fact(
        self,
        content: str,
        category: str,
        confidence: float = 0.5,
        provenance: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> VerticalFact:
        """Helper to create a fact with vertical-specific defaults."""
        import uuid

        return VerticalFact(
            id=f"{self.vertical_id}_{uuid.uuid4().hex[:12]}",
            vertical=self.vertical_id,
            content=content,
            category=category,
            confidence=confidence,
            decay_rate=self.get_decay_rate(category),
            provenance=provenance or {},
            metadata=metadata or {},
        )
