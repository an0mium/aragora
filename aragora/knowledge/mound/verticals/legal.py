"""
Legal vertical knowledge module.

Provides domain-specific fact extraction, validation, and pattern detection
for legal documents including:
- Contract clauses and obligations
- Risk identification
- Regulatory compliance (GDPR, HIPAA, SOX, etc.)
- Citation analysis
- Term definitions
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from aragora.knowledge.mound.verticals.base import (
    BaseVerticalKnowledge,
    ComplianceCheckResult,
    PatternMatch,
    VerticalCapabilities,
    VerticalFact,
)

logger = logging.getLogger(__name__)


@dataclass
class ClausePattern:
    """Pattern for detecting contract clauses."""

    name: str
    pattern: str
    category: str  # obligation, right, condition, termination, etc.
    risk_level: str  # high, medium, low
    description: str
    flags: int = re.IGNORECASE | re.MULTILINE


@dataclass
class CompliancePattern:
    """Pattern for detecting compliance-related terms."""

    name: str
    pattern: str
    framework: str  # GDPR, HIPAA, SOX, etc.
    description: str


class LegalKnowledge(BaseVerticalKnowledge):
    """
    Legal vertical knowledge module.

    Specializes in:
    - Contract clause extraction
    - Obligation identification
    - Risk assessment
    - Regulatory compliance detection
    - Citation and reference analysis
    """

    # Contract clause patterns
    CLAUSE_PATTERNS = [
        ClausePattern(
            name="Termination Clause",
            pattern=r'(?:terminat(?:e|ion)|cancel(?:lation)?|end(?:ing)?)\s+(?:of\s+)?(?:this\s+)?(?:agreement|contract)',
            category="termination",
            risk_level="high",
            description="Contract termination provisions",
        ),
        ClausePattern(
            name="Indemnification",
            pattern=r'(?:indemnif(?:y|ication)|hold\s+harmless)',
            category="liability",
            risk_level="high",
            description="Indemnification and hold harmless clauses",
        ),
        ClausePattern(
            name="Limitation of Liability",
            pattern=r'(?:limit(?:ation)?\s+of\s+liability|consequential\s+damages|indirect\s+damages)',
            category="liability",
            risk_level="high",
            description="Liability limitation provisions",
        ),
        ClausePattern(
            name="Force Majeure",
            pattern=r'(?:force\s+majeure|act\s+of\s+god|unforeseeable\s+circumstances)',
            category="exception",
            risk_level="medium",
            description="Force majeure and unforeseeable events",
        ),
        ClausePattern(
            name="Confidentiality",
            pattern=r'(?:confidential(?:ity)?|non-disclosure|proprietary\s+information)',
            category="obligation",
            risk_level="medium",
            description="Confidentiality and NDA provisions",
        ),
        ClausePattern(
            name="Assignment",
            pattern=r'(?:assign(?:ment)?|transfer(?:ability)?)\s+(?:of\s+)?(?:rights|obligations)',
            category="right",
            risk_level="medium",
            description="Assignment and transfer provisions",
        ),
        ClausePattern(
            name="Warranty",
            pattern=r'(?:warrant(?:y|ies)|guarantee|representation)',
            category="obligation",
            risk_level="medium",
            description="Warranty and representation clauses",
        ),
        ClausePattern(
            name="Governing Law",
            pattern=r'(?:govern(?:ing|ed)\s+(?:by\s+)?(?:the\s+)?law(?:s)?|jurisdiction|venue)',
            category="procedural",
            risk_level="low",
            description="Choice of law and jurisdiction",
        ),
        ClausePattern(
            name="Payment Terms",
            pattern=r'(?:payment\s+(?:term|due|schedule)|net\s+\d+|within\s+\d+\s+days)',
            category="obligation",
            risk_level="medium",
            description="Payment terms and schedules",
        ),
        ClausePattern(
            name="Intellectual Property",
            pattern=r'(?:intellectual\s+property|patent|copyright|trademark|trade\s+secret)',
            category="right",
            risk_level="high",
            description="IP rights and ownership",
        ),
        ClausePattern(
            name="Non-Compete",
            pattern=r'(?:non-?compet(?:e|ition)|restrictive\s+covenant)',
            category="restriction",
            risk_level="high",
            description="Non-compete and restrictive covenants",
        ),
        ClausePattern(
            name="Arbitration",
            pattern=r'(?:arbitrat(?:e|ion)|dispute\s+resolution|mediat(?:e|ion))',
            category="procedural",
            risk_level="medium",
            description="Alternative dispute resolution",
        ),
    ]

    # Compliance patterns
    COMPLIANCE_PATTERNS = [
        CompliancePattern(
            name="GDPR - Data Subject Rights",
            pattern=r'(?:right\s+to\s+(?:access|erasure|rectification|portability)|data\s+subject\s+rights)',
            framework="GDPR",
            description="GDPR data subject rights",
        ),
        CompliancePattern(
            name="GDPR - Consent",
            pattern=r'(?:explicit\s+consent|consent\s+(?:to|for)\s+processing|withdraw\s+consent)',
            framework="GDPR",
            description="GDPR consent requirements",
        ),
        CompliancePattern(
            name="GDPR - Data Protection Officer",
            pattern=r'(?:data\s+protection\s+officer|DPO)',
            framework="GDPR",
            description="GDPR DPO requirement",
        ),
        CompliancePattern(
            name="HIPAA - PHI",
            pattern=r'(?:protected\s+health\s+information|PHI|medical\s+record)',
            framework="HIPAA",
            description="HIPAA protected health information",
        ),
        CompliancePattern(
            name="HIPAA - Authorization",
            pattern=r'(?:HIPAA\s+authorization|patient\s+authorization|disclosure\s+authorization)',
            framework="HIPAA",
            description="HIPAA authorization requirements",
        ),
        CompliancePattern(
            name="SOX - Internal Controls",
            pattern=r'(?:internal\s+control(?:s)?|financial\s+reporting|audit\s+committee)',
            framework="SOX",
            description="SOX internal control requirements",
        ),
        CompliancePattern(
            name="SOX - Certification",
            pattern=r'(?:CEO\s+certification|CFO\s+certification|management\s+certification)',
            framework="SOX",
            description="SOX certification requirements",
        ),
        CompliancePattern(
            name="PCI-DSS - Card Data",
            pattern=r'(?:cardholder\s+data|payment\s+card|PCI|credit\s+card\s+(?:number|information))',
            framework="PCI-DSS",
            description="PCI-DSS cardholder data",
        ),
    ]

    # Risk indicators
    RISK_PATTERNS = [
        (r'(?:unlimited\s+liability|uncapped)', "high", "Unlimited liability exposure"),
        (r'(?:automatic\s+renewal|evergreen)', "medium", "Automatic renewal clause"),
        (r'(?:sole\s+discretion|absolute\s+discretion)', "high", "Unilateral discretion"),
        (r'(?:irrevocable|perpetual\s+license)', "high", "Irrevocable rights"),
        (r'(?:waive(?:s|r)?|release)', "medium", "Rights waiver"),
        (r'(?:material\s+breach)', "medium", "Material breach provision"),
    ]

    @property
    def vertical_id(self) -> str:
        return "legal"

    @property
    def display_name(self) -> str:
        return "Legal & Contracts"

    @property
    def description(self) -> str:
        return "Contract analysis, clause extraction, obligation tracking, and regulatory compliance"

    @property
    def capabilities(self) -> VerticalCapabilities:
        return VerticalCapabilities(
            supports_pattern_detection=True,
            supports_cross_reference=True,
            supports_compliance_check=True,
            requires_llm=False,
            requires_vector_search=True,
            pattern_categories=[
                "clause",
                "obligation",
                "right",
                "risk",
                "compliance",
                "citation",
            ],
            compliance_frameworks=["GDPR", "HIPAA", "SOX", "PCI-DSS", "CCPA"],
            document_types=["contract", "agreement", "policy", "regulation", "terms"],
        )

    @property
    def decay_rates(self) -> dict[str, float]:
        """Legal-specific decay rates."""
        return {
            "clause": 0.005,  # Contract clauses are very stable
            "obligation": 0.01,  # Obligations don't change often
            "regulation": 0.03,  # Regulations update periodically
            "case_law": 0.02,  # Case law is stable unless overturned
            "risk": 0.02,  # Risk assessments may need updates
            "definition": 0.001,  # Legal definitions rarely change
            "default": 0.01,
        }

    # -------------------------------------------------------------------------
    # Fact Extraction
    # -------------------------------------------------------------------------

    async def extract_facts(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[VerticalFact]:
        """Extract legal facts from document content."""
        facts = []
        metadata = metadata or {}

        # Extract clause facts
        for clause in self.CLAUSE_PATTERNS:
            matches = re.findall(clause.pattern, content, clause.flags)
            if matches:
                facts.append(
                    self.create_fact(
                        content=f"Found {clause.name}: {clause.description}",
                        category="clause",
                        confidence=0.75,
                        provenance={
                            "pattern": clause.name,
                            "clause_type": clause.category,
                            "match_count": len(matches),
                        },
                        metadata={
                            "risk_level": clause.risk_level,
                            "clause_category": clause.category,
                            **metadata,
                        },
                    )
                )

        # Extract compliance facts
        for comp in self.COMPLIANCE_PATTERNS:
            if re.search(comp.pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"Compliance reference: {comp.name} - {comp.description}",
                        category="compliance",
                        confidence=0.8,
                        provenance={
                            "pattern": comp.name,
                            "framework": comp.framework,
                        },
                        metadata={
                            "framework": comp.framework,
                            **metadata,
                        },
                    )
                )

        # Extract risk indicators
        for pattern, risk_level, description in self.RISK_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"Risk indicator: {description}",
                        category="risk",
                        confidence=0.7,
                        metadata={
                            "risk_level": risk_level,
                            **metadata,
                        },
                    )
                )

        # Extract term definitions
        definition_pattern = r'["\']([^"\']+)["\']\s+(?:means?|refers?\s+to|shall\s+mean)'
        for match in re.finditer(definition_pattern, content, re.IGNORECASE):
            term = match.group(1)
            if len(term) < 100:  # Reasonable term length
                facts.append(
                    self.create_fact(
                        content=f"Defined term: '{term}'",
                        category="definition",
                        confidence=0.85,
                        metadata=metadata,
                    )
                )

        return facts

    # -------------------------------------------------------------------------
    # Fact Validation
    # -------------------------------------------------------------------------

    async def validate_fact(
        self,
        fact: VerticalFact,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, float]:
        """
        Validate a legal fact.

        Legal facts are generally stable; validation mainly checks
        if the underlying regulation or contract is still in effect.
        """
        if fact.category == "clause":
            # Clauses are stable unless contract is amended
            return True, min(0.95, fact.confidence * 1.02)

        if fact.category == "regulation":
            # Regulations may be updated
            new_confidence = max(0.5, fact.confidence * 0.98)
            return True, new_confidence

        if fact.category == "compliance":
            # Compliance requirements may change
            new_confidence = max(0.6, fact.confidence * 0.97)
            return True, new_confidence

        # Default: slight confidence boost
        return True, min(0.95, fact.confidence * 1.01)

    # -------------------------------------------------------------------------
    # Pattern Detection
    # -------------------------------------------------------------------------

    async def detect_patterns(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[PatternMatch]:
        """Detect patterns across legal facts."""
        patterns = []

        # Group facts by category
        by_category: dict[str, list[VerticalFact]] = {}
        for fact in facts:
            by_category.setdefault(fact.category, []).append(fact)

        # Pattern: Multiple high-risk clauses
        clause_facts = by_category.get("clause", [])
        high_risk_clauses = [
            f for f in clause_facts if f.metadata.get("risk_level") == "high"
        ]
        if len(high_risk_clauses) >= 2:
            patterns.append(
                PatternMatch(
                    pattern_id=f"high_risk_contract_{uuid.uuid4().hex[:8]}",
                    pattern_name="High-Risk Contract",
                    pattern_type="risk_concentration",
                    description=f"Contract contains {len(high_risk_clauses)} high-risk clauses",
                    confidence=0.85,
                    supporting_facts=[f.id for f in high_risk_clauses],
                    metadata={"risk_count": len(high_risk_clauses)},
                )
            )

        # Pattern: One-sided agreement
        obligation_facts = [
            f for f in clause_facts if f.metadata.get("clause_category") == "obligation"
        ]
        right_facts = [
            f for f in clause_facts if f.metadata.get("clause_category") == "right"
        ]
        if len(obligation_facts) > len(right_facts) * 2:
            patterns.append(
                PatternMatch(
                    pattern_id=f"one_sided_{uuid.uuid4().hex[:8]}",
                    pattern_name="One-Sided Agreement",
                    pattern_type="balance_issue",
                    description="Agreement appears heavily weighted with obligations over rights",
                    confidence=0.7,
                    supporting_facts=[f.id for f in obligation_facts],
                )
            )

        # Pattern: Multiple compliance frameworks
        compliance_facts = by_category.get("compliance", [])
        frameworks = set(f.metadata.get("framework") for f in compliance_facts)
        if len(frameworks) >= 2:
            patterns.append(
                PatternMatch(
                    pattern_id=f"multi_compliance_{uuid.uuid4().hex[:8]}",
                    pattern_name="Multi-Framework Compliance",
                    pattern_type="complexity",
                    description=f"Document references {len(frameworks)} compliance frameworks",
                    confidence=0.8,
                    supporting_facts=[f.id for f in compliance_facts],
                    metadata={"frameworks": list(frameworks)},
                )
            )

        return patterns

    # -------------------------------------------------------------------------
    # Compliance Checking
    # -------------------------------------------------------------------------

    async def check_compliance(
        self,
        facts: Sequence[VerticalFact],
        framework: str,
    ) -> list[ComplianceCheckResult]:
        """Check compliance against regulatory frameworks."""
        results = []

        if framework.upper() == "GDPR":
            results.extend(await self._check_gdpr_compliance(facts))
        elif framework.upper() == "HIPAA":
            results.extend(await self._check_hipaa_compliance(facts))
        elif framework.upper() == "SOX":
            results.extend(await self._check_sox_compliance(facts))
        elif framework.upper() == "PCI-DSS":
            results.extend(await self._check_pci_compliance(facts))

        return results

    async def _check_gdpr_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check GDPR compliance."""
        results = []
        gdpr_facts = [
            f for f in facts
            if f.category == "compliance" and f.metadata.get("framework") == "GDPR"
        ]

        # Check for required GDPR elements
        required_elements = {
            "consent": r'consent',
            "data_subject_rights": r'data\s+subject|right\s+to',
            "dpo": r'data\s+protection\s+officer|DPO',
        }

        for element, pattern in required_elements.items():
            found = any(re.search(pattern, f.content, re.IGNORECASE) for f in gdpr_facts)
            results.append(
                ComplianceCheckResult(
                    rule_id=f"gdpr_{element}",
                    rule_name=f"GDPR {element.replace('_', ' ').title()}",
                    framework="GDPR",
                    passed=found,
                    severity="high" if element == "consent" else "medium",
                    findings=[f"GDPR {element} {'found' if found else 'not found'}"],
                    evidence=[f.id for f in gdpr_facts if re.search(pattern, f.content, re.IGNORECASE)],
                    recommendations=[
                        f"Review and document {element} requirements" if not found else "Requirements appear to be addressed"
                    ],
                    confidence=0.75 if found else 0.6,
                )
            )

        return results

    async def _check_hipaa_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check HIPAA compliance."""
        results = []
        hipaa_facts = [
            f for f in facts
            if f.category == "compliance" and f.metadata.get("framework") == "HIPAA"
        ]

        if hipaa_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="hipaa_phi",
                    rule_name="HIPAA PHI Protection",
                    framework="HIPAA",
                    passed=True,
                    severity="high",
                    findings=["HIPAA provisions found - document addresses PHI"],
                    evidence=[f.id for f in hipaa_facts],
                    recommendations=["Ensure BAA is in place with all covered entities"],
                    confidence=0.8,
                )
            )

        return results

    async def _check_sox_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check SOX compliance."""
        results = []
        sox_facts = [
            f for f in facts
            if f.category == "compliance" and f.metadata.get("framework") == "SOX"
        ]

        if sox_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="sox_controls",
                    rule_name="SOX Internal Controls",
                    framework="SOX",
                    passed=True,
                    severity="high",
                    findings=["SOX-related provisions found"],
                    evidence=[f.id for f in sox_facts],
                    recommendations=["Verify internal control documentation is current"],
                    confidence=0.75,
                )
            )

        return results

    async def _check_pci_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check PCI-DSS compliance."""
        results = []
        pci_facts = [
            f for f in facts
            if f.category == "compliance" and f.metadata.get("framework") == "PCI-DSS"
        ]

        if pci_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="pci_card_data",
                    rule_name="PCI-DSS Cardholder Data",
                    framework="PCI-DSS",
                    passed=True,
                    severity="high",
                    findings=["PCI-DSS related provisions found"],
                    evidence=[f.id for f in pci_facts],
                    recommendations=["Ensure PCI-DSS Level compliance certification is current"],
                    confidence=0.8,
                )
            )

        return results

    # -------------------------------------------------------------------------
    # Cross-Reference
    # -------------------------------------------------------------------------

    async def cross_reference(
        self,
        fact: VerticalFact,
        other_facts: Sequence[VerticalFact],
    ) -> list[tuple[str, str, float]]:
        """Find related legal facts via cross-reference."""
        references = []

        # Find clauses that reference each other
        if fact.category == "clause":
            for other in other_facts:
                if other.id == fact.id:
                    continue

                # Check if same clause category (e.g., multiple liability clauses)
                if (
                    other.category == "clause" and
                    fact.metadata.get("clause_category") == other.metadata.get("clause_category")
                ):
                    references.append((other.id, "same_category", 0.7))

                # Check if related compliance framework
                if other.category == "compliance":
                    references.append((other.id, "compliance_link", 0.5))

        return references
