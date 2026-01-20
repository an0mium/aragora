"""
Research vertical knowledge module.

Provides domain-specific fact extraction, validation, and pattern detection
for research documents including:
- Academic citations and references
- Methodology analysis
- Statistical findings
- Hypothesis and claims
- Literature review patterns
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
class ResearchPattern:
    """Pattern for detecting research elements."""

    name: str
    pattern: str
    category: str  # methodology, finding, claim, citation, etc.
    confidence_weight: float  # How much to trust this type
    description: str


@dataclass
class StatisticalPattern:
    """Pattern for detecting statistical claims."""

    name: str
    pattern: str
    category: str  # p-value, confidence_interval, effect_size, etc.
    description: str


class ResearchKnowledge(BaseVerticalKnowledge):
    """
    Research vertical knowledge module.

    Specializes in:
    - Citation extraction and analysis
    - Methodology identification
    - Statistical claims verification
    - Hypothesis and findings extraction
    - Research integrity patterns
    """

    # Research structure patterns
    RESEARCH_PATTERNS = [
        ResearchPattern(
            name="Hypothesis",
            pattern=r"(?:hypothes(?:is|es|ize)|H[0-9]+:|we\s+(?:hypothesize|propose|predict))",
            category="hypothesis",
            confidence_weight=0.7,
            description="Research hypothesis or prediction",
        ),
        ResearchPattern(
            name="Methodology",
            pattern=r"(?:method(?:ology|s)?|procedure|protocol|experimental\s+design)",
            category="methodology",
            confidence_weight=0.8,
            description="Research methodology",
        ),
        ResearchPattern(
            name="Sample Size",
            pattern=r"(?:n\s*=\s*\d+|sample\s+(?:size|of)\s+\d+|participants?|subjects?)",
            category="methodology",
            confidence_weight=0.85,
            description="Sample size information",
        ),
        ResearchPattern(
            name="Results",
            pattern=r"(?:results?\s+(?:show|indicate|demonstrate|suggest)|findings?\s+(?:reveal|show))",
            category="finding",
            confidence_weight=0.75,
            description="Research results",
        ),
        ResearchPattern(
            name="Conclusion",
            pattern=r"(?:conclude|in\s+conclusion|we\s+find\s+that|our\s+(?:results|findings)\s+suggest)",
            category="conclusion",
            confidence_weight=0.7,
            description="Research conclusions",
        ),
        ResearchPattern(
            name="Limitation",
            pattern=r"(?:limitation|caveat|shortcoming|future\s+(?:work|research)|further\s+study)",
            category="limitation",
            confidence_weight=0.8,
            description="Study limitations",
        ),
        ResearchPattern(
            name="Literature Review",
            pattern=r"(?:previous\s+(?:work|studies?|research)|literature\s+review|prior\s+research)",
            category="literature",
            confidence_weight=0.75,
            description="Literature review",
        ),
        ResearchPattern(
            name="Data Source",
            pattern=r"(?:data(?:set|base)?|corpus|survey|questionnaire|interview)",
            category="data",
            confidence_weight=0.8,
            description="Data source",
        ),
    ]

    # Statistical patterns
    STATISTICAL_PATTERNS = [
        StatisticalPattern(
            name="P-Value",
            pattern=r"(?:p\s*[<>=]\s*[0-9.]+|p-?value|statistical(?:ly)?\s+significant)",
            category="p_value",
            description="Statistical significance",
        ),
        StatisticalPattern(
            name="Confidence Interval",
            pattern=r"(?:confidence\s+interval|CI\s*[:=]?\s*\[?[0-9.]+|95%\s+CI)",
            category="confidence_interval",
            description="Confidence interval",
        ),
        StatisticalPattern(
            name="Effect Size",
            pattern=r"(?:effect\s+size|Cohen\'?s?\s+d|eta\s+squared|r\s*=\s*[0-9.]+)",
            category="effect_size",
            description="Effect size measure",
        ),
        StatisticalPattern(
            name="Regression",
            pattern=r"(?:regression|R²|R-squared|beta\s+coefficient|linear\s+model)",
            category="regression",
            description="Regression analysis",
        ),
        StatisticalPattern(
            name="Correlation",
            pattern=r"(?:correlat(?:e|ion)|Pearson|Spearman|r\s*=\s*-?[0-9.]+)",
            category="correlation",
            description="Correlation analysis",
        ),
        StatisticalPattern(
            name="Statistical Test",
            pattern=r"(?:t-?test|ANOVA|chi-?square|Mann-?Whitney|Wilcoxon|Kruskal)",
            category="test",
            description="Statistical test",
        ),
        StatisticalPattern(
            name="Mean/SD",
            pattern=r"(?:mean\s*=|M\s*=|SD\s*=|standard\s+deviation|μ\s*=|σ\s*=)",
            category="descriptive",
            description="Descriptive statistics",
        ),
    ]

    # Citation patterns
    CITATION_PATTERNS = [
        (r"\((?:[A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?),?\s*\d{4}\)", "author_year"),
        (r"\[\d+\]", "numbered"),
        (r"(?:doi|DOI):\s*[0-9.]+/[^\s]+", "doi"),
        (r"(?:arXiv|arxiv):[0-9.]+", "arxiv"),
        (r"(?:PMID|pmid):\s*\d+", "pubmed"),
        (r"(?:ISBN|isbn)[-:\s]*[\d-X]+", "isbn"),
    ]

    # Research integrity patterns
    INTEGRITY_PATTERNS = [
        (r"(?:conflict\s+of\s+interest|COI|disclosure)", "coi", "Conflict of interest disclosure"),
        (
            r"(?:IRB|ethics\s+(?:committee|approval)|informed\s+consent)",
            "ethics",
            "Ethics approval",
        ),
        (r"(?:pre-?register|registered\s+report)", "preregistration", "Study preregistration"),
        (r"(?:replicat(?:e|ion)|reproduc(?:e|ibility))", "replication", "Replication study"),
        (r"(?:peer\s+review|double-?blind)", "peer_review", "Peer review status"),
    ]

    @property
    def vertical_id(self) -> str:
        return "research"

    @property
    def display_name(self) -> str:
        return "Research & Academia"

    @property
    def description(self) -> str:
        return (
            "Academic research, citations, methodology, statistical analysis, and literature review"
        )

    @property
    def capabilities(self) -> VerticalCapabilities:
        return VerticalCapabilities(
            supports_pattern_detection=True,
            supports_cross_reference=True,
            supports_compliance_check=True,
            requires_llm=False,
            requires_vector_search=True,
            pattern_categories=[
                "citation",
                "methodology",
                "finding",
                "hypothesis",
                "statistics",
                "integrity",
            ],
            compliance_frameworks=["IRB", "CONSORT", "PRISMA", "APA"],
            document_types=["paper", "thesis", "preprint", "review", "protocol", "grant"],
        )

    @property
    def decay_rates(self) -> dict[str, float]:
        """Research-specific decay rates."""
        return {
            "finding": 0.01,  # Research findings are stable
            "methodology": 0.005,  # Methods don't change
            "citation": 0.001,  # Citations are permanent
            "hypothesis": 0.005,  # Hypotheses are stable
            "statistics": 0.002,  # Stats don't change
            "claim": 0.02,  # Claims may be refuted
            "literature": 0.03,  # Literature reviews need updates
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
        """Extract research facts from academic content."""
        facts = []
        metadata = metadata or {}

        # Extract research structure facts
        for research in self.RESEARCH_PATTERNS:
            matches = re.findall(research.pattern, content, re.IGNORECASE)
            if matches:
                facts.append(
                    self.create_fact(
                        content=f"Research element: {research.name} - {research.description}",
                        category=research.category,
                        confidence=research.confidence_weight,
                        provenance={
                            "pattern": research.name,
                            "research_category": research.category,
                            "match_count": len(matches),
                        },
                        metadata=metadata,
                    )
                )

        # Extract statistical claims
        for stat in self.STATISTICAL_PATTERNS:
            if re.search(stat.pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"Statistical claim: {stat.name} - {stat.description}",
                        category="statistics",
                        confidence=0.8,
                        provenance={
                            "pattern": stat.name,
                            "stat_type": stat.category,
                        },
                        metadata={
                            "stat_category": stat.category,
                            **metadata,
                        },
                    )
                )

        # Extract citations
        citation_count = 0
        for pattern, citation_type in self.CITATION_PATTERNS:
            matches = re.findall(pattern, content)
            if matches:
                citation_count += len(matches)

        if citation_count > 0:
            facts.append(
                self.create_fact(
                    content=f"Citations detected: {citation_count} references",
                    category="citation",
                    confidence=0.9,
                    provenance={"citation_count": citation_count},
                    metadata=metadata,
                )
            )

        # Extract research integrity markers
        for pattern, integrity_type, description in self.INTEGRITY_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"Research integrity: {description}",
                        category="integrity",
                        confidence=0.85,
                        provenance={"integrity_type": integrity_type},
                        metadata={
                            "integrity_type": integrity_type,
                            **metadata,
                        },
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
        Validate a research fact.

        Research facts are generally stable once published, but may be
        superseded by newer research or retractions.
        """
        if fact.category == "citation":
            # Citations don't change
            return True, min(0.99, fact.confidence * 1.01)

        if fact.category == "finding":
            # Findings may be replicated or refuted
            return True, fact.confidence

        if fact.category == "statistics":
            # Statistical claims don't change but may be questioned
            return True, fact.confidence

        if fact.category == "methodology":
            # Methods are stable
            return True, min(0.95, fact.confidence * 1.02)

        return True, fact.confidence

    # -------------------------------------------------------------------------
    # Pattern Detection
    # -------------------------------------------------------------------------

    async def detect_patterns(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[PatternMatch]:
        """Detect patterns across research facts."""
        patterns = []

        # Group facts by category
        by_category: dict[str, list[VerticalFact]] = {}
        for fact in facts:
            by_category.setdefault(fact.category, []).append(fact)

        # Pattern: Well-structured research
        required_sections = {"methodology", "finding", "hypothesis"}
        found_sections = set(by_category.keys()) & required_sections
        if len(found_sections) >= 2:
            patterns.append(
                PatternMatch(
                    pattern_id=f"structured_research_{uuid.uuid4().hex[:8]}",
                    pattern_name="Well-Structured Research",
                    pattern_type="quality",
                    description=f"Document includes {len(found_sections)} key research sections",
                    confidence=0.8,
                    supporting_facts=[
                        f.id for cat in found_sections for f in by_category.get(cat, [])
                    ][:5],
                    metadata={"sections_found": list(found_sections)},
                )
            )

        # Pattern: Strong statistical evidence
        stat_facts = by_category.get("statistics", [])
        if len(stat_facts) >= 3:
            patterns.append(
                PatternMatch(
                    pattern_id=f"statistical_rigor_{uuid.uuid4().hex[:8]}",
                    pattern_name="Statistical Rigor",
                    pattern_type="quality",
                    description="Multiple statistical methods and claims present",
                    confidence=0.75,
                    supporting_facts=[f.id for f in stat_facts[:5]],
                )
            )

        # Pattern: Extensive citations
        citation_facts = by_category.get("citation", [])
        for cf in citation_facts:
            count = cf.provenance.get("citation_count", 0)
            if count >= 20:
                patterns.append(
                    PatternMatch(
                        pattern_id=f"extensive_citations_{uuid.uuid4().hex[:8]}",
                        pattern_name="Extensive Literature Review",
                        pattern_type="coverage",
                        description=f"Document cites {count}+ references",
                        confidence=0.85,
                        supporting_facts=[cf.id],
                    )
                )
                break

        # Pattern: Research integrity markers
        integrity_facts = by_category.get("integrity", [])
        if len(integrity_facts) >= 2:
            patterns.append(
                PatternMatch(
                    pattern_id=f"research_integrity_{uuid.uuid4().hex[:8]}",
                    pattern_name="Research Integrity",
                    pattern_type="quality",
                    description="Multiple research integrity markers present",
                    confidence=0.9,
                    supporting_facts=[f.id for f in integrity_facts],
                )
            )

        # Pattern: Study limitations acknowledged
        limitation_facts = [f for f in facts if f.category == "limitation"]
        if limitation_facts:
            patterns.append(
                PatternMatch(
                    pattern_id=f"limitations_acknowledged_{uuid.uuid4().hex[:8]}",
                    pattern_name="Limitations Acknowledged",
                    pattern_type="transparency",
                    description="Study limitations are explicitly discussed",
                    confidence=0.85,
                    supporting_facts=[f.id for f in limitation_facts],
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
        """Check compliance against research frameworks."""
        results = []

        if framework.upper() == "IRB":
            results.extend(await self._check_irb_compliance(facts))
        elif framework.upper() == "CONSORT":
            results.extend(await self._check_consort_compliance(facts))
        elif framework.upper() == "PRISMA":
            results.extend(await self._check_prisma_compliance(facts))
        elif framework.upper() == "APA":
            results.extend(await self._check_apa_compliance(facts))

        return results

    async def _check_irb_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check IRB/ethics compliance."""
        results = []

        integrity_facts = [f for f in facts if f.category == "integrity"]
        ethics_facts = [
            f for f in integrity_facts if f.provenance.get("integrity_type") == "ethics"
        ]

        if ethics_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="irb_approval",
                    rule_name="IRB/Ethics Approval",
                    framework="IRB",
                    passed=True,
                    severity="high",
                    findings=["Ethics approval mentioned"],
                    evidence=[f.id for f in ethics_facts],
                    recommendations=["Verify IRB number is included"],
                    confidence=0.8,
                )
            )
        else:
            results.append(
                ComplianceCheckResult(
                    rule_id="irb_approval",
                    rule_name="IRB/Ethics Approval",
                    framework="IRB",
                    passed=False,
                    severity="high",
                    findings=["No ethics approval statement found"],
                    evidence=[],
                    recommendations=[
                        "Include IRB approval number",
                        "Add ethics committee approval statement",
                        "Document informed consent procedures",
                    ],
                    confidence=0.6,
                )
            )

        return results

    async def _check_consort_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check CONSORT compliance for clinical trials."""
        results = []

        # Check for required CONSORT elements
        consort_elements = {
            "methodology": "Methods section",
            "finding": "Results section",
            "statistics": "Statistical analysis",
        }

        found = []
        missing = []
        evidence = []

        for category, name in consort_elements.items():
            cat_facts = [f for f in facts if f.category == category]
            if cat_facts:
                found.append(name)
                evidence.extend(f.id for f in cat_facts[:2])
            else:
                missing.append(name)

        results.append(
            ComplianceCheckResult(
                rule_id="consort_elements",
                rule_name="CONSORT Required Elements",
                framework="CONSORT",
                passed=len(missing) == 0,
                severity="medium" if missing else "low",
                findings=[
                    f"Found: {', '.join(found)}" if found else "No CONSORT elements found",
                    f"Missing: {', '.join(missing)}" if missing else "",
                ],
                evidence=evidence,
                recommendations=[f"Include {m}" for m in missing] if missing else [],
                confidence=0.7,
            )
        )

        return results

    async def _check_prisma_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check PRISMA compliance for systematic reviews."""
        results = []

        citation_facts = [f for f in facts if f.category == "citation"]
        literature_facts = [f for f in facts if f.category == "literature"]

        if citation_facts or literature_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="prisma_search",
                    rule_name="PRISMA Search Strategy",
                    framework="PRISMA",
                    passed=True,
                    severity="medium",
                    findings=["Literature search evidence found"],
                    evidence=[f.id for f in (citation_facts + literature_facts)[:3]],
                    recommendations=["Verify PRISMA flow diagram is included"],
                    confidence=0.65,
                )
            )

        return results

    async def _check_apa_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check APA style compliance."""
        results = []

        citation_facts = [f for f in facts if f.category == "citation"]

        if citation_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="apa_citations",
                    rule_name="APA Citation Format",
                    framework="APA",
                    passed=True,
                    severity="low",
                    findings=["Citations present - verify APA format"],
                    evidence=[f.id for f in citation_facts[:2]],
                    recommendations=["Review citations for APA 7th edition format"],
                    confidence=0.5,  # Can't verify format without deeper analysis
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
        """Find related research facts via cross-reference."""
        references = []

        # Link hypotheses with findings
        if fact.category == "hypothesis":
            for other in other_facts:
                if other.id == fact.id:
                    continue
                if other.category == "finding":
                    references.append((other.id, "tested_by", 0.7))
                if other.category == "statistics":
                    references.append((other.id, "analyzed_by", 0.6))

        # Link methodology with results
        if fact.category == "methodology":
            for other in other_facts:
                if other.category == "finding":
                    references.append((other.id, "produces", 0.65))

        # Link findings with citations
        if fact.category == "finding":
            for other in other_facts:
                if other.category == "citation":
                    references.append((other.id, "referenced_in", 0.5))

        return references
