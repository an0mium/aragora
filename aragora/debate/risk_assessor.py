"""
Domain risk assessment for debates.

Identifies potential risks in debate domains and topics:
- Safety-sensitive topics (medical, legal, financial advice)
- Speculative domains with high uncertainty
- Topics requiring specialized expertise
- Areas with potential for harmful misinformation
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# Type alias for pattern configuration
PatternConfig = dict[str, Any]

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels for domain assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Result of domain risk analysis."""

    level: RiskLevel
    domain: str
    category: str
    description: str
    mitigations: list[str] = field(default_factory=list)
    confidence: float = 0.5


# Domain patterns that indicate elevated risk
RISK_PATTERNS = {
    "medical": {
        "keywords": [
            "diagnosis",
            "treatment",
            "medication",
            "symptom",
            "disease",
            "health",
            "doctor",
            "medical",
        ],
        "level": RiskLevel.HIGH,
        "category": "health_advice",
        "description": "Medical topics require professional expertise. AI-generated advice may be inaccurate.",
        "mitigations": [
            "Consult healthcare professional",
            "Include medical disclaimer",
            "Cite authoritative sources",
        ],
    },
    "legal": {
        "keywords": [
            "lawsuit",
            "contract",
            "liability",
            "court",
            "attorney",
            "legal",
            "law",
            "compliance",
        ],
        "level": RiskLevel.HIGH,
        "category": "legal_advice",
        "description": "Legal advice requires jurisdiction-specific expertise and professional qualification.",
        "mitigations": [
            "Consult licensed attorney",
            "Include legal disclaimer",
            "Specify jurisdiction limitations",
        ],
    },
    "financial": {
        "keywords": [
            "investment",
            "trading",
            "stock",
            "crypto",
            "financial",
            "money",
            "tax",
            "portfolio",
        ],
        "level": RiskLevel.MEDIUM,
        "category": "financial_advice",
        "description": "Financial recommendations can cause monetary harm if followed without due diligence.",
        "mitigations": [
            "Consult financial advisor",
            "Include investment disclaimer",
            "Note market volatility",
        ],
    },
    "safety": {
        "keywords": [
            "dangerous",
            "explosive",
            "weapon",
            "harm",
            "attack",
            "vulnerability",
            "exploit",
        ],
        "level": RiskLevel.CRITICAL,
        "category": "safety_concern",
        "description": "Topic may involve physical safety or security risks.",
        "mitigations": [
            "Review content carefully",
            "Limit detailed instructions",
            "Report concerning content",
        ],
    },
    "speculative": {
        "keywords": ["predict", "future", "forecast", "will happen", "guaranteed", "certain"],
        "level": RiskLevel.MEDIUM,
        "category": "speculation",
        "description": "Speculative claims about uncertain outcomes may be presented with false confidence.",
        "mitigations": [
            "Express uncertainty explicitly",
            "Provide probability ranges",
            "Acknowledge limitations",
        ],
    },
}


class RiskAssessor:
    """
    Assesses domain-specific risks for debate topics.

    Usage:
        assessor = RiskAssessor()
        risks = assessor.assess_topic("Should I stop taking my medication?")
        for risk in risks:
            print(f"{risk.level.value}: {risk.description}")
    """

    def __init__(self, custom_patterns: Optional[dict] = None):
        """
        Initialize risk assessor.

        Args:
            custom_patterns: Additional risk patterns to check
        """
        self.patterns: dict[str, PatternConfig] = {**RISK_PATTERNS}
        if custom_patterns:
            self.patterns.update(custom_patterns)

    def assess_topic(self, topic: str, domain: Optional[str] = None) -> list[RiskAssessment]:
        """
        Assess risks for a debate topic.

        Args:
            topic: The debate topic/question
            domain: Optional pre-classified domain

        Returns:
            List of identified risks, sorted by severity
        """
        risks = []
        topic_lower = topic.lower()

        for pattern_name, pattern_config in self.patterns.items():
            keywords = pattern_config.get("keywords", [])
            matches = sum(1 for kw in keywords if kw in topic_lower)

            if matches > 0:
                # Calculate confidence based on keyword matches
                confidence = min(0.9, 0.3 + (matches * 0.15))

                risks.append(
                    RiskAssessment(
                        level=pattern_config["level"],
                        domain=domain or pattern_name,
                        category=pattern_config["category"],
                        description=pattern_config["description"],
                        mitigations=pattern_config.get("mitigations", []),
                        confidence=confidence,
                    )
                )

        # Sort by severity (critical first)
        severity_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 3,
        }
        risks.sort(key=lambda r: (severity_order.get(r.level, 99), -r.confidence))

        return risks

    def get_highest_risk(
        self, topic: str, domain: Optional[str] = None
    ) -> Optional[RiskAssessment]:
        """Get the highest severity risk for a topic."""
        risks = self.assess_topic(topic, domain)
        return risks[0] if risks else None

    def to_event_data(self, assessment: RiskAssessment, debate_id: str = "") -> dict:
        """Convert assessment to event data for WebSocket emission."""
        return {
            "level": assessment.level.value,
            "domain": assessment.domain,
            "category": assessment.category,
            "description": assessment.description,
            "mitigations": assessment.mitigations,
            "confidence": assessment.confidence,
            "debate_id": debate_id,
        }


# Module-level instance for convenience
_default_assessor: Optional[RiskAssessor] = None


def get_risk_assessor() -> RiskAssessor:
    """Get or create the default RiskAssessor instance."""
    global _default_assessor
    if _default_assessor is None:
        _default_assessor = RiskAssessor()
    return _default_assessor


def assess_debate_risk(topic: str, domain: Optional[str] = None) -> list[RiskAssessment]:
    """Convenience function to assess debate risks."""
    return get_risk_assessor().assess_topic(topic, domain)
