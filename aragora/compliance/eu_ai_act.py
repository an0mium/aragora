"""
EU AI Act Compliance Module.

Maps Aragora Decision Receipts to EU AI Act requirements for conformity
assessment. The EU AI Act (Regulation (EU) 2024/1689) takes effect
August 2, 2026 for Annex III high-risk systems.

Key articles mapped:
- Article 6:  Classification rules for high-risk AI systems
- Article 9:  Risk management system
- Article 13: Transparency and provision of information to deployers
- Article 14: Human oversight
- Article 50: Transparency obligations (formerly Art. 52 in drafts)

Annex III defines 8 categories of high-risk AI systems:
1. Biometrics
2. Critical infrastructure
3. Education and vocational training
4. Employment and worker management
5. Access to essential services
6. Law enforcement
7. Migration, asylum and border control
8. Administration of justice and democratic processes
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Risk classification per EU AI Act Article 6 + Annex III
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    """EU AI Act risk tiers."""

    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


@dataclass
class RiskClassification:
    """Result of classifying a use case under the EU AI Act."""

    risk_level: RiskLevel
    annex_iii_category: str | None = None
    annex_iii_number: int | None = None
    rationale: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    applicable_articles: list[str] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "annex_iii_category": self.annex_iii_category,
            "annex_iii_number": self.annex_iii_number,
            "rationale": self.rationale,
            "matched_keywords": self.matched_keywords,
            "applicable_articles": self.applicable_articles,
            "obligations": self.obligations,
        }


# Annex III categories with detection keywords
ANNEX_III_CATEGORIES: list[dict[str, Any]] = [
    {
        "number": 1,
        "name": "Biometrics",
        "description": "Remote biometric identification, biometric categorization, emotion recognition",
        "keywords": [
            "biometric", "facial recognition", "face detection", "emotion recognition",
            "fingerprint identification", "iris scan", "voice identification",
            "remote identification", "biometric categorization",
        ],
    },
    {
        "number": 2,
        "name": "Critical infrastructure",
        "description": "Safety components in critical digital infrastructure, road traffic, water, gas, heating, electricity",
        "keywords": [
            "critical infrastructure", "power grid", "water supply", "gas supply",
            "electricity", "road traffic", "traffic management", "digital infrastructure",
            "energy supply", "heating supply", "safety component",
        ],
    },
    {
        "number": 3,
        "name": "Education and vocational training",
        "description": "Determining access to education, assessing students, monitoring exams",
        "keywords": [
            "student assessment", "educational admission", "exam proctoring",
            "grading system", "academic evaluation", "learning assessment",
            "vocational training", "educational institution", "student scoring",
        ],
    },
    {
        "number": 4,
        "name": "Employment and worker management",
        "description": "Recruitment, CV screening, performance evaluation, task allocation, termination",
        "keywords": [
            "recruitment", "cv screening", "resume screening", "hiring decision",
            "job application", "performance evaluation", "employee monitoring",
            "worker management", "task allocation", "termination decision",
            "promotion decision", "workforce management",
        ],
    },
    {
        "number": 5,
        "name": "Access to essential services",
        "description": "Credit scoring, insurance risk, public benefit eligibility, emergency dispatch",
        "keywords": [
            "credit scoring", "credit score", "creditworthiness", "loan decision",
            "insurance risk", "health insurance", "benefit eligibility",
            "public assistance", "emergency dispatch", "emergency services",
            "essential services", "social benefit",
        ],
    },
    {
        "number": 6,
        "name": "Law enforcement",
        "description": "Crime analytics, evidence reliability, recidivism risk, profiling",
        "keywords": [
            "law enforcement", "crime prediction", "predictive policing",
            "evidence evaluation", "recidivism", "criminal profiling",
            "crime analytics", "suspect identification", "polygraph",
        ],
    },
    {
        "number": 7,
        "name": "Migration, asylum and border control",
        "description": "Security risk assessment, visa/asylum application processing, border surveillance",
        "keywords": [
            "migration", "asylum", "border control", "visa application",
            "immigration", "border surveillance", "refugee", "deportation",
            "travel document", "security risk assessment",
        ],
    },
    {
        "number": 8,
        "name": "Administration of justice and democratic processes",
        "description": "Judicial research, legal interpretation, election influence",
        "keywords": [
            "judicial", "court decision", "legal interpretation", "sentencing",
            "justice system", "election", "voting", "democratic process",
            "legal reasoning", "case law analysis",
        ],
    },
]

# Patterns for unacceptable-risk AI (Article 5 prohibitions)
UNACCEPTABLE_KEYWORDS: list[str] = [
    "social scoring", "social credit", "subliminal manipulation",
    "exploit vulnerability", "real-time remote biometric identification",
    "emotion recognition in workplace", "emotion recognition in education",
    "untargeted scraping of facial images",
    "cognitive behavioral manipulation",
]


class RiskClassifier:
    """
    Classify AI use cases by EU AI Act risk level.

    Implements Article 6 classification rules and Annex III category matching.
    """

    def classify(self, description: str) -> RiskClassification:
        """
        Classify a use case description by EU AI Act risk level.

        Args:
            description: Free-text description of the AI use case.

        Returns:
            RiskClassification with level, category, and obligations.
        """
        description_lower = description.lower()

        # Check unacceptable risk first (Article 5)
        unacceptable_matches = [
            kw for kw in UNACCEPTABLE_KEYWORDS
            if kw in description_lower
        ]
        if unacceptable_matches:
            return RiskClassification(
                risk_level=RiskLevel.UNACCEPTABLE,
                rationale="Use case matches prohibited AI practices under Article 5.",
                matched_keywords=unacceptable_matches,
                applicable_articles=["Article 5"],
                obligations=["This AI practice is prohibited under the EU AI Act."],
            )

        # Check high-risk (Annex III categories)
        for category in ANNEX_III_CATEGORIES:
            matches = [
                kw for kw in category["keywords"]
                if kw in description_lower
            ]
            if matches:
                return RiskClassification(
                    risk_level=RiskLevel.HIGH,
                    annex_iii_category=category["name"],
                    annex_iii_number=category["number"],
                    rationale=(
                        f"Use case falls under Annex III category {category['number']}: "
                        f"{category['name']}. {category['description']}."
                    ),
                    matched_keywords=matches,
                    applicable_articles=[
                        "Article 6 (Classification)",
                        "Article 9 (Risk management)",
                        "Article 13 (Transparency)",
                        "Article 14 (Human oversight)",
                        "Article 15 (Accuracy, robustness, cybersecurity)",
                    ],
                    obligations=_high_risk_obligations(category["name"]),
                )

        # Check limited-risk (Article 50 transparency obligations)
        limited_keywords = [
            "chatbot", "generated content", "deepfake", "synthetic media",
            "ai-generated", "virtual assistant", "conversational ai",
        ]
        limited_matches = [kw for kw in limited_keywords if kw in description_lower]
        if limited_matches:
            return RiskClassification(
                risk_level=RiskLevel.LIMITED,
                rationale="Use case involves AI systems with transparency obligations under Article 50.",
                matched_keywords=limited_matches,
                applicable_articles=["Article 50 (Transparency obligations)"],
                obligations=[
                    "Inform users they are interacting with an AI system.",
                    "Label AI-generated content as artificially generated or manipulated.",
                    "Maintain technical documentation.",
                ],
            )

        # Default: minimal risk
        return RiskClassification(
            risk_level=RiskLevel.MINIMAL,
            rationale="Use case does not match high-risk or limited-risk categories. Minimal obligations apply.",
            applicable_articles=[],
            obligations=[
                "Voluntary adoption of codes of conduct encouraged (Article 95).",
            ],
        )

    def classify_receipt(self, receipt_dict: dict[str, Any]) -> RiskClassification:
        """
        Classify a DecisionReceipt's underlying use case.

        Uses the input_summary and verdict_reasoning fields to infer the domain.
        """
        text = " ".join([
            receipt_dict.get("input_summary", ""),
            receipt_dict.get("verdict_reasoning", ""),
        ])
        return self.classify(text)


def _high_risk_obligations(category_name: str) -> list[str]:
    """Return obligations for a high-risk AI system."""
    base = [
        "Establish and maintain a risk management system (Art. 9).",
        "Use high-quality training, validation, and testing data (Art. 10).",
        "Maintain technical documentation (Art. 11).",
        "Implement automatic logging of events (Art. 12).",
        "Ensure transparency and provide instructions for deployers (Art. 13).",
        "Design for effective human oversight (Art. 14).",
        "Achieve appropriate accuracy, robustness, and cybersecurity (Art. 15).",
        "Register in the EU database before placing on market (Art. 49).",
        "Undergo conformity assessment (Art. 43).",
    ]
    return base


# ---------------------------------------------------------------------------
# Conformity assessment mapping: Receipt -> EU AI Act articles
# ---------------------------------------------------------------------------

@dataclass
class ArticleMapping:
    """Maps a receipt field to an EU AI Act article requirement."""

    article: str
    article_title: str
    requirement: str
    receipt_field: str
    status: str  # "satisfied", "partial", "not_satisfied", "not_applicable"
    evidence: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "article": self.article,
            "article_title": self.article_title,
            "requirement": self.requirement,
            "receipt_field": self.receipt_field,
            "status": self.status,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }


@dataclass
class ConformityReport:
    """EU AI Act conformity assessment report generated from a DecisionReceipt."""

    report_id: str
    receipt_id: str
    generated_at: str
    risk_classification: RiskClassification
    article_mappings: list[ArticleMapping]
    overall_status: str  # "conformant", "partial", "non_conformant"
    summary: str
    recommendations: list[str]
    integrity_hash: str = ""

    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        content = json.dumps(
            {
                "report_id": self.report_id,
                "receipt_id": self.receipt_id,
                "overall_status": self.overall_status,
                "risk_level": self.risk_classification.risk_level.value,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "receipt_id": self.receipt_id,
            "generated_at": self.generated_at,
            "risk_classification": self.risk_classification.to_dict(),
            "article_mappings": [m.to_dict() for m in self.article_mappings],
            "overall_status": self.overall_status,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "integrity_hash": self.integrity_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Export as human-readable markdown."""
        lines = [
            f"# EU AI Act Conformity Report",
            "",
            f"**Report ID:** {self.report_id}",
            f"**Receipt ID:** {self.receipt_id}",
            f"**Generated:** {self.generated_at}",
            f"**Integrity Hash:** `{self.integrity_hash[:16]}...`",
            "",
            "---",
            "",
            "## Risk Classification",
            "",
            f"**Risk Level:** {self.risk_classification.risk_level.value.upper()}",
        ]

        if self.risk_classification.annex_iii_category:
            lines.append(
                f"**Annex III Category:** {self.risk_classification.annex_iii_number}. "
                f"{self.risk_classification.annex_iii_category}"
            )
        lines.append(f"**Rationale:** {self.risk_classification.rationale}")
        lines.append("")

        if self.risk_classification.obligations:
            lines.append("### Obligations")
            lines.append("")
            for obligation in self.risk_classification.obligations:
                lines.append(f"- {obligation}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Article Compliance Assessment",
            "",
            f"**Overall Status:** {self.overall_status.upper()}",
            "",
            "| Article | Requirement | Status | Evidence |",
            "|---------|-------------|--------|----------|",
        ])

        for m in self.article_mappings:
            status_indicator = {
                "satisfied": "PASS",
                "partial": "PARTIAL",
                "not_satisfied": "FAIL",
                "not_applicable": "N/A",
            }.get(m.status, m.status)
            evidence_short = m.evidence[:60] + "..." if len(m.evidence) > 60 else m.evidence
            lines.append(
                f"| {m.article} | {m.requirement[:50]} | {status_indicator} | {evidence_short} |"
            )

        lines.append("")

        if self.recommendations:
            lines.extend([
                "---",
                "",
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Summary",
            "",
            self.summary,
            "",
        ])

        return "\n".join(lines)


class ConformityReportGenerator:
    """
    Generate EU AI Act conformity assessment reports from DecisionReceipts.

    Maps receipt fields to article requirements:
    - Article 9  (Risk management):  receipt.risk_summary, confidence, robustness_score
    - Article 13 (Transparency):     provenance_chain, consensus_proof (agent participation)
    - Article 14 (Human oversight):  config_used for human approval indicators
    - Article 12 (Record-keeping):   provenance_chain completeness
    - Article 15 (Accuracy):         confidence, robustness_score
    """

    def __init__(self, classifier: RiskClassifier | None = None):
        self._classifier = classifier or RiskClassifier()

    def generate(self, receipt_dict: dict[str, Any]) -> ConformityReport:
        """
        Generate a conformity report from a receipt dictionary.

        Args:
            receipt_dict: Output of DecisionReceipt.to_dict().

        Returns:
            ConformityReport with article mappings and recommendations.
        """
        report_id = f"EUAIA-{str(uuid.uuid4())[:8]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        receipt_id = receipt_dict.get("receipt_id", "unknown")

        # Classify the underlying use case
        classification = self._classifier.classify_receipt(receipt_dict)

        # Map receipt fields to article requirements
        mappings = self._map_articles(receipt_dict, classification)

        # Determine overall status
        statuses = [m.status for m in mappings if m.status != "not_applicable"]
        if all(s == "satisfied" for s in statuses):
            overall = "conformant"
        elif any(s == "not_satisfied" for s in statuses):
            overall = "non_conformant"
        else:
            overall = "partial"

        # Build recommendations
        recommendations = []
        for m in mappings:
            if m.recommendation and m.status != "satisfied":
                recommendations.append(m.recommendation)

        # Build summary
        satisfied_count = sum(1 for m in mappings if m.status == "satisfied")
        total_applicable = sum(1 for m in mappings if m.status != "not_applicable")
        summary = (
            f"Conformity assessment for receipt {receipt_id} against the EU AI Act. "
            f"Risk level: {classification.risk_level.value}. "
            f"{satisfied_count}/{total_applicable} applicable article requirements satisfied."
        )

        return ConformityReport(
            report_id=report_id,
            receipt_id=receipt_id,
            generated_at=timestamp,
            risk_classification=classification,
            article_mappings=mappings,
            overall_status=overall,
            summary=summary,
            recommendations=recommendations,
        )

    def _map_articles(
        self,
        receipt: dict[str, Any],
        classification: RiskClassification,
    ) -> list[ArticleMapping]:
        """Map receipt fields to EU AI Act article requirements."""
        mappings: list[ArticleMapping] = []

        # --- Article 9: Risk Management ---
        risk_summary = receipt.get("risk_summary", {})
        confidence = receipt.get("confidence", 0.0)
        robustness = receipt.get("robustness_score", 0.0)

        risk_total = risk_summary.get("total", 0)
        risk_critical = risk_summary.get("critical", 0)

        if risk_total > 0 or confidence > 0:
            risk_status = "satisfied" if risk_critical == 0 and confidence >= 0.5 else "partial"
            if risk_critical > 0:
                risk_status = "not_satisfied"
            mappings.append(ArticleMapping(
                article="Article 9",
                article_title="Risk management system",
                requirement="Identify and analyze known and reasonably foreseeable risks",
                receipt_field="risk_summary, confidence",
                status=risk_status,
                evidence=(
                    f"Risk assessment performed: {risk_total} risks identified "
                    f"({risk_critical} critical). Confidence: {confidence:.1%}."
                ),
                recommendation=(
                    "Address critical risks before deployment."
                    if risk_critical > 0 else ""
                ),
            ))
        else:
            mappings.append(ArticleMapping(
                article="Article 9",
                article_title="Risk management system",
                requirement="Identify and analyze known and reasonably foreseeable risks",
                receipt_field="risk_summary",
                status="not_satisfied",
                evidence="No risk assessment data found in receipt.",
                recommendation="Conduct a risk assessment and record findings in the receipt.",
            ))

        # --- Article 12: Record-keeping (automatic logging) ---
        provenance = receipt.get("provenance_chain", [])
        log_status = "satisfied" if len(provenance) >= 2 else "partial"
        if not provenance:
            log_status = "not_satisfied"
        mappings.append(ArticleMapping(
            article="Article 12",
            article_title="Record-keeping",
            requirement="Automatic logging of events with traceability",
            receipt_field="provenance_chain",
            status=log_status,
            evidence=f"Provenance chain contains {len(provenance)} events.",
            recommendation=(
                "Ensure all decision events are logged in the provenance chain."
                if log_status != "satisfied" else ""
            ),
        ))

        # --- Article 13: Transparency ---
        consensus = receipt.get("consensus_proof") or {}
        supporting = consensus.get("supporting_agents", [])
        dissenting = consensus.get("dissenting_agents", [])
        all_agents = list(set(supporting + dissenting))
        dissenting_views = receipt.get("dissenting_views", [])
        verdict_reasoning = receipt.get("verdict_reasoning", "")

        transparency_satisfied = bool(all_agents) and bool(verdict_reasoning)
        mappings.append(ArticleMapping(
            article="Article 13",
            article_title="Transparency and provision of information to deployers",
            requirement="Identify participating agents, their arguments, and decision rationale",
            receipt_field="consensus_proof, verdict_reasoning, dissenting_views",
            status="satisfied" if transparency_satisfied else "partial",
            evidence=(
                f"{len(all_agents)} agents participated. "
                f"Verdict reasoning: {verdict_reasoning[:100]}{'...' if len(verdict_reasoning) > 100 else ''}. "
                f"{len(dissenting_views)} dissenting view(s) recorded."
            ),
            recommendation=(
                "Include agent identities and reasoning in all receipts."
                if not transparency_satisfied else ""
            ),
        ))

        # --- Article 14: Human oversight ---
        config = receipt.get("config_used", {})
        has_human_oversight = _detect_human_oversight(config, receipt)
        mappings.append(ArticleMapping(
            article="Article 14",
            article_title="Human oversight",
            requirement="Enable human oversight, including ability to override or halt",
            receipt_field="config_used",
            status="satisfied" if has_human_oversight else "partial",
            evidence=(
                "Human approval/override mechanism detected in receipt configuration."
                if has_human_oversight
                else "No explicit human oversight mechanism found in receipt."
            ),
            recommendation=(
                ""
                if has_human_oversight
                else "Integrate human-in-the-loop approval before critical decisions are finalized."
            ),
        ))

        # --- Article 15: Accuracy, robustness, cybersecurity ---
        integrity_valid = bool(receipt.get("artifact_hash"))
        has_signature = bool(receipt.get("signature"))
        acc_status = "satisfied"
        if robustness < 0.5:
            acc_status = "partial"
        if robustness < 0.2:
            acc_status = "not_satisfied"

        mappings.append(ArticleMapping(
            article="Article 15",
            article_title="Accuracy, robustness and cybersecurity",
            requirement="Appropriate levels of accuracy and robustness; resilience to attacks",
            receipt_field="robustness_score, artifact_hash, signature",
            status=acc_status,
            evidence=(
                f"Robustness score: {robustness:.1%}. "
                f"Integrity hash: {'present' if integrity_valid else 'missing'}. "
                f"Cryptographic signature: {'present' if has_signature else 'absent'}."
            ),
            recommendation=(
                "Improve robustness score and add cryptographic signing."
                if acc_status != "satisfied" else ""
            ),
        ))

        return mappings


def _detect_human_oversight(config: dict[str, Any], receipt: dict[str, Any]) -> bool:
    """Detect whether human oversight was present in the decision process."""
    # Check for human-related config keys
    oversight_indicators = [
        "human_approval", "require_approval", "human_in_loop",
        "human_override", "approver", "approver_id", "approval_record",
    ]
    config_str = json.dumps(config).lower()
    for indicator in oversight_indicators:
        if indicator in config_str:
            return True

    # Check provenance chain for human events
    for event in receipt.get("provenance_chain", []):
        event_type = ""
        if isinstance(event, dict):
            event_type = event.get("event_type", "")
        elif hasattr(event, "event_type"):
            event_type = event.event_type
        if event_type in ("human_approval", "plan_approved", "human_override"):
            return True

    return False


__all__ = [
    "RiskLevel",
    "RiskClassification",
    "RiskClassifier",
    "ANNEX_III_CATEGORIES",
    "ArticleMapping",
    "ConformityReport",
    "ConformityReportGenerator",
]
