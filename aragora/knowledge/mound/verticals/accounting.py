"""
Accounting vertical knowledge module.

Provides domain-specific fact extraction, validation, and pattern detection
for financial and accounting documents including:
- Financial statements analysis
- SOX compliance
- GAAP/IFRS standards
- Audit findings
- Tax regulations
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
class FinancialPattern:
    """Pattern for detecting financial terms."""

    name: str
    pattern: str
    category: str  # asset, liability, revenue, expense, etc.
    risk_level: str  # high, medium, low
    description: str


@dataclass
class AuditPattern:
    """Pattern for detecting audit-related terms."""

    name: str
    pattern: str
    category: str  # finding, control, test, etc.
    severity: str
    description: str


class AccountingKnowledge(BaseVerticalKnowledge):
    """
    Accounting vertical knowledge module.

    Specializes in:
    - Financial statement analysis
    - SOX compliance verification
    - GAAP/IFRS standards application
    - Internal control assessment
    - Tax compliance
    """

    # Financial statement patterns
    FINANCIAL_PATTERNS = [
        FinancialPattern(
            name="Revenue Recognition",
            pattern=r"(?:revenue\s+recogni(?:tion|zed?)|sales\s+revenue|earned\s+revenue)",
            category="revenue",
            risk_level="high",
            description="Revenue recognition assessment",
        ),
        FinancialPattern(
            name="Accounts Receivable",
            pattern=r"(?:accounts?\s+receivable|A/R|trade\s+receivables|allowance\s+for\s+doubtful)",
            category="asset",
            risk_level="medium",
            description="Accounts receivable",
        ),
        FinancialPattern(
            name="Inventory",
            pattern=r"(?:inventory|FIFO|LIFO|weighted\s+average|inventory\s+reserve)",
            category="asset",
            risk_level="medium",
            description="Inventory valuation",
        ),
        FinancialPattern(
            name="Fixed Assets",
            pattern=r"(?:fixed\s+asset|PP&E|property\s+plant\s+equipment|depreciation|amortization)",
            category="asset",
            risk_level="medium",
            description="Fixed asset accounting",
        ),
        FinancialPattern(
            name="Accounts Payable",
            pattern=r"(?:accounts?\s+payable|A/P|trade\s+payables)",
            category="liability",
            risk_level="low",
            description="Accounts payable",
        ),
        FinancialPattern(
            name="Debt",
            pattern=r"(?:long-?term\s+debt|notes?\s+payable|bond|debenture|credit\s+facility)",
            category="liability",
            risk_level="high",
            description="Debt instruments",
        ),
        FinancialPattern(
            name="Lease",
            pattern=r"(?:lease|right-?of-?use|ROU|ASC\s+842|IFRS\s+16)",
            category="liability",
            risk_level="high",
            description="Lease accounting",
        ),
        FinancialPattern(
            name="Goodwill",
            pattern=r"(?:goodwill|impairment\s+(?:test|loss)|intangible\s+asset)",
            category="asset",
            risk_level="high",
            description="Goodwill and intangibles",
        ),
        FinancialPattern(
            name="Equity",
            pattern=r"(?:stockholder(?:s)?\'?\s+equity|share\s+capital|retained\s+earnings|treasury\s+stock)",
            category="equity",
            risk_level="medium",
            description="Equity accounts",
        ),
        FinancialPattern(
            name="Contingency",
            pattern=r"(?:contingent\s+liability|contingency|lawsuit|litigation|warranty)",
            category="liability",
            risk_level="high",
            description="Contingent liabilities",
        ),
    ]

    # Audit-related patterns
    AUDIT_PATTERNS = [
        AuditPattern(
            name="Material Weakness",
            pattern=r"(?:material\s+weakness|MW|significant\s+deficiency|SD)",
            category="finding",
            severity="high",
            description="Internal control weakness",
        ),
        AuditPattern(
            name="Audit Opinion",
            pattern=r"(?:unqualified\s+opinion|qualified\s+opinion|adverse\s+opinion|disclaimer)",
            category="opinion",
            severity="high",
            description="Auditor opinion",
        ),
        AuditPattern(
            name="Substantive Test",
            pattern=r"(?:substantive\s+(?:test|procedure)|analytical\s+review|confirmation)",
            category="test",
            severity="medium",
            description="Substantive audit procedures",
        ),
        AuditPattern(
            name="Control Test",
            pattern=r"(?:test\s+of\s+control|control\s+(?:test|procedure)|walkthrough)",
            category="test",
            severity="medium",
            description="Control testing",
        ),
        AuditPattern(
            name="Misstatement",
            pattern=r"(?:misstatement|error|irregularity|adjustment)",
            category="finding",
            severity="high",
            description="Audit adjustment or misstatement",
        ),
        AuditPattern(
            name="Related Party",
            pattern=r"(?:related\s+party|affiliate|intercompany)",
            category="risk",
            severity="high",
            description="Related party transactions",
        ),
    ]

    # Compliance patterns
    COMPLIANCE_PATTERNS = [
        (r"(?:SOX|Sarbanes-?Oxley|section\s+404)", "SOX", "Sarbanes-Oxley compliance"),
        (r"(?:GAAP|US\s+GAAP|generally\s+accepted)", "GAAP", "US GAAP standards"),
        (
            r"(?:IFRS|international\s+financial)",
            "IFRS",
            "International Financial Reporting Standards",
        ),
        (r"(?:PCAOB)", "PCAOB", "PCAOB standards"),
        (r"(?:SEC|securities\s+(?:and\s+)?exchange)", "SEC", "SEC regulations"),
        (r"(?:FASB|ASC\s+\d+)", "FASB", "FASB Accounting Standards"),
    ]

    # Risk indicators
    RISK_PATTERNS = [
        (r"(?:going\s+concern)", "high", "Going concern uncertainty"),
        (r"(?:restatement|restated)", "high", "Financial restatement"),
        (r"(?:fraud|fraudulent)", "critical", "Fraud indicator"),
        (r"(?:management\s+override)", "high", "Management override of controls"),
        (r"(?:off-?balance\s+sheet)", "high", "Off-balance sheet arrangements"),
        (r"(?:non-?compliance)", "medium", "Non-compliance with regulations"),
    ]

    @property
    def vertical_id(self) -> str:
        return "accounting"

    @property
    def display_name(self) -> str:
        return "Accounting & Finance"

    @property
    def description(self) -> str:
        return "Financial analysis, audit findings, SOX compliance, and accounting standards"

    @property
    def capabilities(self) -> VerticalCapabilities:
        return VerticalCapabilities(
            supports_pattern_detection=True,
            supports_cross_reference=True,
            supports_compliance_check=True,
            requires_llm=False,
            requires_vector_search=True,
            pattern_categories=[
                "financial",
                "audit",
                "compliance",
                "risk",
                "control",
                "tax",
            ],
            compliance_frameworks=["SOX", "GAAP", "IFRS", "SEC", "PCAOB"],
            document_types=["financial_statement", "audit_report", "10K", "10Q", "proxy", "memo"],
        )

    @property
    def decay_rates(self) -> dict[str, float]:
        """Accounting-specific decay rates."""
        return {
            "financial": 0.03,  # Financial data changes quarterly
            "audit": 0.02,  # Audit findings are stable until addressed
            "control": 0.02,  # Control assessments are stable
            "standard": 0.01,  # Accounting standards change slowly
            "tax": 0.04,  # Tax rules change annually
            "risk": 0.03,  # Risk assessments need periodic updates
            "default": 0.02,
        }

    # -------------------------------------------------------------------------
    # Fact Extraction
    # -------------------------------------------------------------------------

    async def extract_facts(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[VerticalFact]:
        """Extract accounting facts from financial documents."""
        facts = []
        metadata = metadata or {}

        # Extract financial facts
        for fin in self.FINANCIAL_PATTERNS:
            matches = re.findall(fin.pattern, content, re.IGNORECASE)
            if matches:
                facts.append(
                    self.create_fact(
                        content=f"Financial item: {fin.name} - {fin.description}",
                        category="financial",
                        confidence=0.75,
                        provenance={
                            "pattern": fin.name,
                            "financial_category": fin.category,
                            "match_count": len(matches),
                        },
                        metadata={
                            "risk_level": fin.risk_level,
                            "account_type": fin.category,
                            **metadata,
                        },
                    )
                )

        # Extract audit facts
        for audit in self.AUDIT_PATTERNS:
            if re.search(audit.pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"Audit finding: {audit.name} - {audit.description}",
                        category="audit",
                        confidence=0.8,
                        provenance={
                            "pattern": audit.name,
                            "audit_category": audit.category,
                        },
                        metadata={
                            "severity": audit.severity,
                            "audit_type": audit.category,
                            **metadata,
                        },
                    )
                )

        # Extract compliance references
        for pattern, framework, description in self.COMPLIANCE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"Compliance reference: {framework} - {description}",
                        category="compliance",
                        confidence=0.85,
                        provenance={"framework": framework},
                        metadata={
                            "framework": framework,
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
        Validate an accounting fact.

        Financial facts may require validation against source documents
        and may become stale after reporting periods.
        """
        if fact.category == "financial":
            # Financial data becomes stale after quarter end
            new_confidence = max(0.5, fact.confidence * 0.97)
            return True, new_confidence

        if fact.category == "audit":
            # Audit findings remain relevant until remediated
            return True, min(0.95, fact.confidence * 1.01)

        if fact.category == "risk":
            # Risk assessments need periodic updates
            new_confidence = max(0.4, fact.confidence * 0.95)
            return True, new_confidence

        return True, fact.confidence

    # -------------------------------------------------------------------------
    # Pattern Detection
    # -------------------------------------------------------------------------

    async def detect_patterns(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[PatternMatch]:
        """Detect patterns across accounting facts."""
        patterns = []

        # Group facts by category
        by_category: dict[str, list[VerticalFact]] = {}
        for fact in facts:
            by_category.setdefault(fact.category, []).append(fact)

        # Pattern: Multiple high-risk areas
        financial_facts = by_category.get("financial", [])
        high_risk_items = [f for f in financial_facts if f.metadata.get("risk_level") == "high"]
        if len(high_risk_items) >= 3:
            patterns.append(
                PatternMatch(
                    pattern_id=f"high_risk_areas_{uuid.uuid4().hex[:8]}",
                    pattern_name="Multiple High-Risk Areas",
                    pattern_type="audit_risk",
                    description=f"Document covers {len(high_risk_items)} high-risk accounting areas",
                    confidence=0.8,
                    supporting_facts=[f.id for f in high_risk_items],
                    metadata={"risk_count": len(high_risk_items)},
                )
            )

        # Pattern: Audit findings present
        audit_facts = by_category.get("audit", [])
        high_severity = [f for f in audit_facts if f.metadata.get("severity") == "high"]
        if high_severity:
            patterns.append(
                PatternMatch(
                    pattern_id=f"audit_issues_{uuid.uuid4().hex[:8]}",
                    pattern_name="Significant Audit Findings",
                    pattern_type="control_risk",
                    description="High-severity audit findings detected",
                    confidence=0.85,
                    supporting_facts=[f.id for f in high_severity],
                )
            )

        # Pattern: Complex regulatory environment
        compliance_facts = by_category.get("compliance", [])
        frameworks = set(f.metadata.get("framework") for f in compliance_facts)
        if len(frameworks) >= 3:
            patterns.append(
                PatternMatch(
                    pattern_id=f"multi_framework_{uuid.uuid4().hex[:8]}",
                    pattern_name="Complex Regulatory Environment",
                    pattern_type="compliance_complexity",
                    description=f"Subject to {len(frameworks)} regulatory frameworks",
                    confidence=0.75,
                    supporting_facts=[f.id for f in compliance_facts],
                    metadata={"frameworks": list(frameworks)},
                )
            )

        # Pattern: Going concern or fraud risk
        risk_facts = by_category.get("risk", [])
        critical_risks = [
            f for f in risk_facts if f.metadata.get("risk_level") in ("critical", "high")
        ]
        if critical_risks:
            patterns.append(
                PatternMatch(
                    pattern_id=f"critical_risk_{uuid.uuid4().hex[:8]}",
                    pattern_name="Critical Risk Indicators",
                    pattern_type="enterprise_risk",
                    description="Critical or high-risk indicators detected",
                    confidence=0.9,
                    supporting_facts=[f.id for f in critical_risks],
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
        """Check compliance against accounting frameworks."""
        results = []

        if framework.upper() == "SOX":
            results.extend(await self._check_sox_compliance(facts))
        elif framework.upper() in ("GAAP", "US GAAP"):
            results.extend(await self._check_gaap_compliance(facts))
        elif framework.upper() == "SEC":
            results.extend(await self._check_sec_compliance(facts))

        return results

    async def _check_sox_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check SOX compliance."""
        results = []

        # Check for material weaknesses
        audit_facts = [f for f in facts if f.category == "audit"]
        mw_facts = [
            f
            for f in audit_facts
            if "weakness" in f.content.lower() or "deficiency" in f.content.lower()
        ]

        if mw_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="sox_404",
                    rule_name="SOX Section 404 - Internal Controls",
                    framework="SOX",
                    passed=False,
                    severity="high",
                    findings=["Material weakness or significant deficiency identified"],
                    evidence=[f.id for f in mw_facts],
                    recommendations=[
                        "Document remediation plan",
                        "Implement compensating controls",
                        "Disclose in management report",
                    ],
                    confidence=0.85,
                )
            )

        # Check for related party transactions
        related_party = [
            f
            for f in facts
            if "related party" in f.content.lower() or "intercompany" in f.content.lower()
        ]
        if related_party:
            results.append(
                ComplianceCheckResult(
                    rule_id="sox_related_party",
                    rule_name="SOX Related Party Transactions",
                    framework="SOX",
                    passed=True,  # Needs review
                    severity="medium",
                    findings=["Related party transactions detected"],
                    evidence=[f.id for f in related_party],
                    recommendations=[
                        "Ensure arm's length pricing",
                        "Verify audit committee approval",
                        "Document business purpose",
                    ],
                    confidence=0.75,
                )
            )

        return results

    async def _check_gaap_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check GAAP compliance."""
        results = []

        # Check for revenue recognition
        revenue_facts = [f for f in facts if "revenue" in f.content.lower()]
        if revenue_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="gaap_asc606",
                    rule_name="ASC 606 Revenue Recognition",
                    framework="GAAP",
                    passed=True,  # Needs review
                    severity="high",
                    findings=["Revenue recognition identified - verify ASC 606 compliance"],
                    evidence=[f.id for f in revenue_facts],
                    recommendations=[
                        "Verify five-step model application",
                        "Document performance obligations",
                        "Confirm transaction price allocation",
                    ],
                    confidence=0.7,
                )
            )

        # Check for lease accounting
        lease_facts = [
            f for f in facts if "lease" in f.content.lower() or "rou" in f.content.lower()
        ]
        if lease_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="gaap_asc842",
                    rule_name="ASC 842 Lease Accounting",
                    framework="GAAP",
                    passed=True,
                    severity="high",
                    findings=["Lease accounting identified"],
                    evidence=[f.id for f in lease_facts],
                    recommendations=[
                        "Verify lease classification",
                        "Confirm ROU asset calculation",
                        "Review variable lease payments",
                    ],
                    confidence=0.75,
                )
            )

        return results

    async def _check_sec_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check SEC compliance."""
        results = []

        # Check for risk indicators in SEC filings
        risk_facts = [f for f in facts if f.category == "risk"]
        if risk_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="sec_risk_factors",
                    rule_name="SEC Risk Factor Disclosure",
                    framework="SEC",
                    passed=True,
                    severity="medium",
                    findings=["Risk factors identified for disclosure"],
                    evidence=[f.id for f in risk_facts],
                    recommendations=[
                        "Ensure material risks are disclosed",
                        "Update risk factors for current period",
                    ],
                    confidence=0.7,
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
        """Find related accounting facts via cross-reference."""
        references = []

        # Link financial items with audit findings
        if fact.category == "financial":
            for other in other_facts:
                if other.id == fact.id:
                    continue

                # Same account type
                if other.category == "financial" and fact.metadata.get(
                    "account_type"
                ) == other.metadata.get("account_type"):
                    references.append((other.id, "same_category", 0.7))

                # Audit findings for financial items
                if other.category == "audit":
                    references.append((other.id, "audited_by", 0.5))

        # Link audit findings with compliance requirements
        if fact.category == "audit":
            for other in other_facts:
                if other.category == "compliance":
                    references.append((other.id, "compliance_requirement", 0.6))

        return references
