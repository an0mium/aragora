"""
Compliance Artifact Generator for Decision Receipts.

Generates framework-specific compliance artifacts from DecisionReceipts
for regulatory reporting. Supported frameworks:

- **EU AI Act** (Regulation 2024/1689): Risk classification, transparency
  requirements, human oversight documentation, data governance, and
  technical documentation summary.
- **SOC 2** (Trust Services Criteria): Security, Availability, Processing
  Integrity, Confidentiality, and Privacy mappings with control evidence.
- **HIPAA** (Health Insurance Portability and Accountability Act): PHI
  handling attestation, access control verification, audit trail reference,
  minimum necessary compliance, and breach notification readiness.

Usage::

    from aragora.compliance.artifact_generator import ReceiptComplianceGenerator

    generator = ReceiptComplianceGenerator()
    artifacts = generator.generate(receipt_dict, frameworks=["eu_ai_act", "soc2", "hipaa"])

    # Or generate individually
    eu_artifact = generator.generate_eu_ai_act(receipt_dict)
    soc2_artifact = generator.generate_soc2(receipt_dict)
    hipaa_artifact = generator.generate_hipaa(receipt_dict)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# EU AI Act Artifact
# ---------------------------------------------------------------------------


@dataclass
class TransparencyChecklist:
    """EU AI Act transparency requirements checklist."""

    agents_identified: bool = False
    decision_rationale_provided: bool = False
    dissenting_views_recorded: bool = False
    confidence_score_disclosed: bool = False
    ai_system_disclosed: bool = False
    limitations_documented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents_identified": self.agents_identified,
            "decision_rationale_provided": self.decision_rationale_provided,
            "dissenting_views_recorded": self.dissenting_views_recorded,
            "confidence_score_disclosed": self.confidence_score_disclosed,
            "ai_system_disclosed": self.ai_system_disclosed,
            "limitations_documented": self.limitations_documented,
        }

    @property
    def all_satisfied(self) -> bool:
        return all(
            [
                self.agents_identified,
                self.decision_rationale_provided,
                self.dissenting_views_recorded,
                self.confidence_score_disclosed,
                self.ai_system_disclosed,
                self.limitations_documented,
            ]
        )


@dataclass
class EUAIActArtifact:
    """EU AI Act compliance artifact generated from a DecisionReceipt.

    Maps receipt data to EU AI Act requirements including risk classification,
    transparency obligations, human oversight documentation, data governance,
    and technical documentation.

    Attributes:
        artifact_id: Unique identifier for this artifact.
        receipt_id: The DecisionReceipt this artifact was generated from.
        generated_at: ISO timestamp of generation.
        risk_classification: One of minimal, limited, high, unacceptable.
        risk_rationale: Explanation for the risk classification.
        transparency_checklist: Checklist of Article 13/50 requirements.
        human_oversight: Documentation of Article 14 human oversight.
        data_governance: Data governance statement per Article 10.
        technical_documentation: Summary per Annex IV.
        applicable_articles: List of applicable EU AI Act articles.
        integrity_hash: SHA-256 hash for tamper detection.
    """

    artifact_id: str
    receipt_id: str
    generated_at: str
    risk_classification: str  # "minimal", "limited", "high", "unacceptable"
    risk_rationale: str
    transparency_checklist: TransparencyChecklist
    human_oversight: dict[str, Any]
    data_governance: dict[str, Any]
    technical_documentation: dict[str, Any]
    applicable_articles: list[str] = field(default_factory=list)
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        content = json.dumps(
            {
                "artifact_id": self.artifact_id,
                "receipt_id": self.receipt_id,
                "risk_classification": self.risk_classification,
                "framework": "eu_ai_act",
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": "eu_ai_act",
            "regulation": "EU AI Act (Regulation 2024/1689)",
            "artifact_id": self.artifact_id,
            "receipt_id": self.receipt_id,
            "generated_at": self.generated_at,
            "risk_classification": self.risk_classification,
            "risk_rationale": self.risk_rationale,
            "transparency_checklist": self.transparency_checklist.to_dict(),
            "human_oversight": self.human_oversight,
            "data_governance": self.data_governance,
            "technical_documentation": self.technical_documentation,
            "applicable_articles": self.applicable_articles,
            "integrity_hash": self.integrity_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# SOC 2 Artifact
# ---------------------------------------------------------------------------


@dataclass
class TrustServiceCriteria:
    """SOC 2 Trust Service Criteria mapping.

    Each field maps to a TSC category with status and evidence.
    """

    security: dict[str, Any] = field(default_factory=dict)
    availability: dict[str, Any] = field(default_factory=dict)
    processing_integrity: dict[str, Any] = field(default_factory=dict)
    confidentiality: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "security": self.security,
            "availability": self.availability,
            "processing_integrity": self.processing_integrity,
            "confidentiality": self.confidentiality,
            "privacy": self.privacy,
        }


@dataclass
class SOC2Artifact:
    """SOC 2 compliance artifact generated from a DecisionReceipt.

    Maps receipt data to SOC 2 Trust Service Criteria with control
    evidence references, exception logging, and period of review.

    Attributes:
        artifact_id: Unique identifier for this artifact.
        receipt_id: The DecisionReceipt this artifact was generated from.
        generated_at: ISO timestamp of generation.
        trust_service_criteria: Mapping to the 5 TSC categories.
        control_evidence: References to controls exercised.
        exceptions: Any exceptions or deviations noted.
        period_of_review: Start and end of the review period.
        service_organization: Organization details.
        integrity_hash: SHA-256 hash for tamper detection.
    """

    artifact_id: str
    receipt_id: str
    generated_at: str
    trust_service_criteria: TrustServiceCriteria
    control_evidence: list[dict[str, Any]]
    exceptions: list[dict[str, Any]]
    period_of_review: dict[str, str]
    service_organization: dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        content = json.dumps(
            {
                "artifact_id": self.artifact_id,
                "receipt_id": self.receipt_id,
                "framework": "soc2",
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": "soc2",
            "standard": "SOC 2 Type II (AICPA Trust Services Criteria)",
            "artifact_id": self.artifact_id,
            "receipt_id": self.receipt_id,
            "generated_at": self.generated_at,
            "trust_service_criteria": self.trust_service_criteria.to_dict(),
            "control_evidence": self.control_evidence,
            "exceptions": self.exceptions,
            "period_of_review": self.period_of_review,
            "service_organization": self.service_organization,
            "integrity_hash": self.integrity_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# HIPAA Artifact
# ---------------------------------------------------------------------------


@dataclass
class HIPAAArtifact:
    """HIPAA compliance artifact generated from a DecisionReceipt.

    Maps receipt data to HIPAA Security and Privacy Rule requirements
    including PHI handling, access controls, audit trails, minimum
    necessary standard, and breach notification readiness.

    Attributes:
        artifact_id: Unique identifier for this artifact.
        receipt_id: The DecisionReceipt this artifact was generated from.
        generated_at: ISO timestamp of generation.
        phi_handling: PHI handling attestation details.
        access_control: Access control verification results.
        audit_trail: Audit trail reference and completeness.
        minimum_necessary: Minimum necessary standard compliance.
        breach_notification: Breach notification readiness assessment.
        safeguards: Administrative, physical, and technical safeguards.
        integrity_hash: SHA-256 hash for tamper detection.
    """

    artifact_id: str
    receipt_id: str
    generated_at: str
    phi_handling: dict[str, Any]
    access_control: dict[str, Any]
    audit_trail: dict[str, Any]
    minimum_necessary: dict[str, Any]
    breach_notification: dict[str, Any]
    safeguards: dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        content = json.dumps(
            {
                "artifact_id": self.artifact_id,
                "receipt_id": self.receipt_id,
                "framework": "hipaa",
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": "hipaa",
            "regulation": "HIPAA (Health Insurance Portability and Accountability Act)",
            "artifact_id": self.artifact_id,
            "receipt_id": self.receipt_id,
            "generated_at": self.generated_at,
            "phi_handling": self.phi_handling,
            "access_control": self.access_control,
            "audit_trail": self.audit_trail,
            "minimum_necessary": self.minimum_necessary,
            "breach_notification": self.breach_notification,
            "safeguards": self.safeguards,
            "integrity_hash": self.integrity_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Unified compliance artifact result
# ---------------------------------------------------------------------------


@dataclass
class ComplianceArtifactResult:
    """Result of generating compliance artifacts for a receipt.

    Contains one or more framework-specific artifacts along with
    metadata about the generation.
    """

    receipt_id: str
    generated_at: str
    frameworks_generated: list[str]
    eu_ai_act: EUAIActArtifact | None = None
    soc2: SOC2Artifact | None = None
    hipaa: HIPAAArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "receipt_id": self.receipt_id,
            "generated_at": self.generated_at,
            "frameworks_generated": self.frameworks_generated,
        }
        if self.eu_ai_act is not None:
            result["eu_ai_act"] = self.eu_ai_act.to_dict()
        if self.soc2 is not None:
            result["soc2"] = self.soc2.to_dict()
        if self.hipaa is not None:
            result["hipaa"] = self.hipaa.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


# Keywords that suggest PHI handling in input/reasoning
_PHI_KEYWORDS = [
    "patient",
    "medical",
    "health",
    "diagnosis",
    "treatment",
    "clinical",
    "phi",
    "hipaa",
    "healthcare",
    "medical record",
    "prescription",
    "lab result",
]

# Keywords that suggest high-risk EU AI Act categories
_EU_HIGH_RISK_KEYWORDS = [
    "biometric",
    "recruitment",
    "hiring",
    "credit scoring",
    "law enforcement",
    "judicial",
    "migration",
    "border",
    "critical infrastructure",
    "education",
    "employment",
    "worker",
]

_EU_UNACCEPTABLE_KEYWORDS = [
    "social scoring",
    "social credit",
    "subliminal manipulation",
    "real-time remote biometric identification",
]

_EU_LIMITED_KEYWORDS = [
    "chatbot",
    "generated content",
    "deepfake",
    "synthetic media",
    "ai-generated",
]


class ReceiptComplianceGenerator:
    """Generate compliance artifacts from DecisionReceipt dictionaries.

    This class bridges the DecisionReceipt system and regulatory compliance
    by extracting evidence from receipt fields and mapping them to
    framework-specific requirements.

    Supported frameworks:
        - ``eu_ai_act``: EU AI Act (Regulation 2024/1689)
        - ``soc2``: SOC 2 Type II (AICPA Trust Services Criteria)
        - ``hipaa``: HIPAA Security and Privacy Rule

    Example::

        generator = ReceiptComplianceGenerator()
        result = generator.generate(receipt.to_dict(), frameworks=["soc2", "hipaa"])
        print(result.soc2.to_json())
    """

    SUPPORTED_FRAMEWORKS = ("eu_ai_act", "soc2", "hipaa")

    def __init__(
        self,
        *,
        organization_name: str = "Aragora Inc.",
        system_name: str = "Aragora Decision Integrity Platform",
        system_version: str = "2.6.3",
    ) -> None:
        self._organization_name = organization_name
        self._system_name = system_name
        self._system_version = system_version

    # -- Public API ---------------------------------------------------------

    def generate(
        self,
        receipt: dict[str, Any],
        frameworks: list[str] | None = None,
    ) -> ComplianceArtifactResult:
        """Generate compliance artifacts for the specified frameworks.

        Args:
            receipt: Dictionary from ``DecisionReceipt.to_dict()``.
            frameworks: List of framework identifiers to generate. If ``None``,
                all supported frameworks are generated.

        Returns:
            ComplianceArtifactResult containing the requested artifacts.

        Raises:
            ValueError: If an unsupported framework is requested.
        """
        if frameworks is None:
            frameworks = list(self.SUPPORTED_FRAMEWORKS)

        unsupported = [f for f in frameworks if f not in self.SUPPORTED_FRAMEWORKS]
        if unsupported:
            raise ValueError(
                f"Unsupported framework(s): {unsupported}. "
                f"Supported: {list(self.SUPPORTED_FRAMEWORKS)}"
            )

        receipt_id = receipt.get("receipt_id", "")
        timestamp = datetime.now(timezone.utc).isoformat()

        eu_artifact = None
        soc2_artifact = None
        hipaa_artifact = None

        if "eu_ai_act" in frameworks:
            eu_artifact = self.generate_eu_ai_act(receipt)
        if "soc2" in frameworks:
            soc2_artifact = self.generate_soc2(receipt)
        if "hipaa" in frameworks:
            hipaa_artifact = self.generate_hipaa(receipt)

        return ComplianceArtifactResult(
            receipt_id=receipt_id,
            generated_at=timestamp,
            frameworks_generated=frameworks,
            eu_ai_act=eu_artifact,
            soc2=soc2_artifact,
            hipaa=hipaa_artifact,
        )

    def generate_eu_ai_act(self, receipt: dict[str, Any]) -> EUAIActArtifact:
        """Generate EU AI Act compliance artifact from a receipt.

        Performs risk classification based on the receipt's input_summary
        and verdict_reasoning, then maps receipt fields to transparency,
        human oversight, data governance, and technical documentation
        requirements.

        Args:
            receipt: Dictionary from ``DecisionReceipt.to_dict()``.

        Returns:
            EUAIActArtifact with all required fields populated.
        """
        receipt_id = receipt.get("receipt_id", "")
        timestamp = datetime.now(timezone.utc).isoformat()
        artifact_id = f"EUAIA-{uuid.uuid4().hex[:8]}"

        # Classify risk
        text = " ".join(
            [
                receipt.get("input_summary", ""),
                receipt.get("verdict_reasoning", ""),
            ]
        ).lower()

        risk_level, risk_rationale, applicable_articles = self._classify_eu_risk(text)

        # Build transparency checklist
        consensus = receipt.get("consensus_proof") or {}
        supporting = consensus.get("supporting_agents", [])
        dissenting_agents = consensus.get("dissenting_agents", [])
        all_agents = list(set(supporting + dissenting_agents))
        dissenting_views = receipt.get("dissenting_views", [])
        verdict_reasoning = receipt.get("verdict_reasoning", "")
        confidence = receipt.get("confidence", 0.0)

        transparency = TransparencyChecklist(
            agents_identified=len(all_agents) > 0,
            decision_rationale_provided=bool(verdict_reasoning),
            dissenting_views_recorded=len(dissenting_views) > 0 or len(dissenting_agents) > 0,
            confidence_score_disclosed=confidence > 0,
            ai_system_disclosed=True,  # Aragora always discloses it is an AI system
            limitations_documented=True,  # System docs always include limitations
        )

        # Human oversight
        config = receipt.get("config_used", {})
        has_human = self._detect_human_oversight(config, receipt)

        human_oversight = {
            "human_in_the_loop": has_human,
            "oversight_model": "HITL" if has_human else "HOTL",
            "override_capability": True,
            "intervention_capability": True,
            "evidence": (
                "Human approval mechanism detected in receipt configuration."
                if has_human
                else "Monitoring-based oversight; human can intervene on anomalies."
            ),
        }

        # Data governance
        input_hash = receipt.get("input_hash", "")
        artifact_hash = receipt.get("artifact_hash", "")
        has_signature = bool(receipt.get("signature"))

        data_governance = {
            "input_integrity_verified": bool(input_hash),
            "input_hash_algorithm": "SHA-256" if input_hash else "none",
            "output_integrity_verified": bool(artifact_hash),
            "cryptographic_signature": has_signature,
            "data_minimization": "Decision receipt contains summary, not raw input data.",
            "retention_policy": "Configurable per deployment; default 6 months for high-risk.",
        }

        # Technical documentation summary (Annex IV)
        provenance = receipt.get("provenance_chain", [])
        risk_summary = receipt.get("risk_summary", {})
        robustness = receipt.get("robustness_score", 0.0)

        technical_documentation = {
            "system_description": {
                "name": self._system_name,
                "version": self._system_version,
                "architecture": "Multi-agent adversarial debate with consensus",
            },
            "performance_metrics": {
                "confidence": confidence,
                "robustness_score": robustness,
                "consensus_reached": consensus.get("reached", False),
                "agents_participating": len(all_agents),
            },
            "risk_assessment": {
                "risks_identified": risk_summary.get("total", 0),
                "critical_risks": risk_summary.get("critical", 0),
                "high_risks": risk_summary.get("high", 0),
            },
            "logging": {
                "provenance_events": len(provenance),
                "audit_trail_complete": len(provenance) >= 2,
            },
        }

        return EUAIActArtifact(
            artifact_id=artifact_id,
            receipt_id=receipt_id,
            generated_at=timestamp,
            risk_classification=risk_level,
            risk_rationale=risk_rationale,
            transparency_checklist=transparency,
            human_oversight=human_oversight,
            data_governance=data_governance,
            technical_documentation=technical_documentation,
            applicable_articles=applicable_articles,
        )

    def generate_soc2(self, receipt: dict[str, Any]) -> SOC2Artifact:
        """Generate SOC 2 compliance artifact from a receipt.

        Maps receipt fields to SOC 2 Trust Service Criteria:
        - **Security** (CC): Access controls, cryptographic integrity
        - **Availability** (A): System resilience indicators
        - **Processing Integrity** (PI): Decision accuracy and consensus
        - **Confidentiality** (C): Data protection measures
        - **Privacy** (P): Data handling practices

        Args:
            receipt: Dictionary from ``DecisionReceipt.to_dict()``.

        Returns:
            SOC2Artifact with trust service criteria mappings.
        """
        receipt_id = receipt.get("receipt_id", "")
        timestamp_str = datetime.now(timezone.utc).isoformat()
        artifact_id = f"SOC2-{uuid.uuid4().hex[:8]}"

        consensus = receipt.get("consensus_proof") or {}
        confidence = receipt.get("confidence", 0.0)
        robustness = receipt.get("robustness_score", 0.0)
        provenance = receipt.get("provenance_chain", [])
        artifact_hash = receipt.get("artifact_hash", "")
        has_signature = bool(receipt.get("signature"))
        input_hash = receipt.get("input_hash", "")
        risk_summary = receipt.get("risk_summary", {})
        receipt.get("config_used", {})

        # Security (CC6, CC7, CC8)
        security = {
            "criteria": "CC6/CC7/CC8 - Logical and Physical Access, System Operations, Change Management",
            "status": "satisfied" if (artifact_hash and input_hash) else "partial",
            "controls": {
                "access_control": "Multi-agent RBAC with per-agent isolation",
                "cryptographic_integrity": {
                    "input_hash": bool(input_hash),
                    "artifact_hash": bool(artifact_hash),
                    "algorithm": "SHA-256",
                    "signature": has_signature,
                },
                "change_management": "Provenance chain records all state transitions",
            },
            "evidence_count": len(provenance),
        }

        # Availability (A1)
        agents_used = list(
            set(consensus.get("supporting_agents", []) + consensus.get("dissenting_agents", []))
        )
        availability = {
            "criteria": "A1 - Availability",
            "status": "satisfied" if len(agents_used) > 0 else "not_satisfied",
            "controls": {
                "multi_agent_redundancy": f"{len(agents_used)} agents participated",
                "circuit_breaker": "Per-agent failure isolation via AirlockProxy",
                "fallback_mechanism": "OpenRouter fallback on quota errors",
            },
        }

        # Processing Integrity (PI1)
        verdict = receipt.get("verdict", "")
        processing_integrity = {
            "criteria": "PI1 - Processing Integrity",
            "status": "satisfied" if confidence >= 0.5 and verdict else "partial",
            "controls": {
                "consensus_validation": {
                    "method": consensus.get("method", "unknown"),
                    "confidence": confidence,
                    "consensus_reached": consensus.get("reached", False),
                },
                "robustness_score": robustness,
                "verdict": verdict,
                "adversarial_validation": "Multi-agent challenge reduces single-point-of-failure",
            },
        }

        # Confidentiality (C1)
        confidentiality = {
            "criteria": "C1 - Confidentiality",
            "status": "satisfied" if input_hash else "partial",
            "controls": {
                "data_hashing": "Input content hashed with SHA-256",
                "summary_only": "Receipt contains summary, not raw sensitive data",
                "encryption_at_rest": "Configurable AES-256-GCM encryption",
                "encryption_in_transit": "TLS 1.2+ enforced",
            },
        }

        # Privacy (P1-P8)
        privacy = {
            "criteria": "P1-P8 - Privacy Criteria",
            "status": "satisfied",
            "controls": {
                "data_minimization": "Receipt stores decision summary, not raw input",
                "purpose_limitation": "Data used exclusively for decision audit trail",
                "retention": "Configurable retention policy with automatic deletion",
                "access_control": "RBAC-governed access to receipts",
            },
        }

        tsc = TrustServiceCriteria(
            security=security,
            availability=availability,
            processing_integrity=processing_integrity,
            confidentiality=confidentiality,
            privacy=privacy,
        )

        # Control evidence
        control_evidence: list[dict[str, Any]] = []

        if artifact_hash:
            control_evidence.append(
                {
                    "control": "CC6.1 - Integrity Controls",
                    "evidence_type": "artifact_hash",
                    "value": artifact_hash[:16] + "...",
                    "description": "SHA-256 content-addressable hash of receipt",
                }
            )

        if has_signature:
            control_evidence.append(
                {
                    "control": "CC6.1 - Cryptographic Signing",
                    "evidence_type": "digital_signature",
                    "description": "Receipt cryptographically signed for non-repudiation",
                }
            )

        if provenance:
            control_evidence.append(
                {
                    "control": "CC7.2 - Monitoring",
                    "evidence_type": "provenance_chain",
                    "event_count": len(provenance),
                    "description": "Complete provenance chain of decision events",
                }
            )

        if consensus.get("reached"):
            control_evidence.append(
                {
                    "control": "PI1.3 - Processing Accuracy",
                    "evidence_type": "consensus_proof",
                    "confidence": confidence,
                    "description": f"Multi-agent consensus with {confidence:.1%} confidence",
                }
            )

        # Exceptions
        exceptions: list[dict[str, Any]] = []
        risk_critical = risk_summary.get("critical", 0)
        risk_high = risk_summary.get("high", 0)

        if risk_critical > 0:
            exceptions.append(
                {
                    "severity": "critical",
                    "description": f"{risk_critical} critical risk(s) identified during validation",
                    "remediation": "Address critical risks before production use",
                }
            )

        if risk_high > 0:
            exceptions.append(
                {
                    "severity": "high",
                    "description": f"{risk_high} high-severity risk(s) identified",
                    "remediation": "Review and mitigate high risks per risk register",
                }
            )

        if not has_signature:
            exceptions.append(
                {
                    "severity": "medium",
                    "description": "Receipt not cryptographically signed",
                    "remediation": "Enable receipt signing for non-repudiation",
                }
            )

        # Period of review
        receipt_timestamp = receipt.get("timestamp", timestamp_str)
        period_of_review = {
            "start": receipt_timestamp,
            "end": timestamp_str,
            "type": "point-in-time",
        }

        return SOC2Artifact(
            artifact_id=artifact_id,
            receipt_id=receipt_id,
            generated_at=timestamp_str,
            trust_service_criteria=tsc,
            control_evidence=control_evidence,
            exceptions=exceptions,
            period_of_review=period_of_review,
            service_organization={
                "name": self._organization_name,
                "system": self._system_name,
                "version": self._system_version,
            },
        )

    def generate_hipaa(self, receipt: dict[str, Any]) -> HIPAAArtifact:
        """Generate HIPAA compliance artifact from a receipt.

        Maps receipt fields to HIPAA Security Rule (45 CFR 164.312)
        and Privacy Rule requirements:
        - PHI handling attestation
        - Access control verification (164.312(a))
        - Audit trail reference (164.312(b))
        - Minimum necessary standard (164.502(b))
        - Breach notification readiness (164.404-164.410)

        Args:
            receipt: Dictionary from ``DecisionReceipt.to_dict()``.

        Returns:
            HIPAAArtifact with all required fields populated.
        """
        receipt_id = receipt.get("receipt_id", "")
        timestamp_str = datetime.now(timezone.utc).isoformat()
        artifact_id = f"HIPAA-{uuid.uuid4().hex[:8]}"

        consensus = receipt.get("consensus_proof") or {}
        provenance = receipt.get("provenance_chain", [])
        input_hash = receipt.get("input_hash", "")
        artifact_hash = receipt.get("artifact_hash", "")
        has_signature = bool(receipt.get("signature"))
        config = receipt.get("config_used", {})
        input_summary = receipt.get("input_summary", "")

        # Detect if PHI may be involved
        phi_detected = any(kw in input_summary.lower() for kw in _PHI_KEYWORDS)

        # PHI Handling Attestation
        phi_handling = {
            "phi_detected_in_summary": phi_detected,
            "attestation": (
                "PHI-related content detected in decision input. "
                "Receipt stores summary only; raw PHI is not persisted in the receipt."
                if phi_detected
                else "No PHI indicators detected in decision input summary."
            ),
            "encryption_at_rest": "AES-256-GCM (configurable)",
            "encryption_in_transit": "TLS 1.2+",
            "de_identification": (
                "Input hashed with SHA-256; summary contains redacted reference only."
            ),
            "input_hash_present": bool(input_hash),
        }

        # Access Control Verification (164.312(a))
        agents = list(
            set(consensus.get("supporting_agents", []) + consensus.get("dissenting_agents", []))
        )
        has_rbac = bool(config.get("rbac_enabled") or config.get("require_approval"))

        access_control = {
            "regulation_ref": "45 CFR 164.312(a) - Access Control",
            "status": "satisfied" if (input_hash and agents) else "partial",
            "unique_user_identification": {
                "agents_identified": len(agents) > 0,
                "agent_names": agents,
                "description": "Each participating agent has a unique identifier",
            },
            "emergency_access_procedure": "Override mechanisms documented in system config",
            "automatic_logoff": "Session-based agent interaction with timeout",
            "encryption_and_decryption": {
                "input_hashed": bool(input_hash),
                "receipt_integrity": bool(artifact_hash),
                "signed": has_signature,
            },
            "rbac_enabled": has_rbac,
        }

        # Audit Trail Reference (164.312(b))
        audit_trail = {
            "regulation_ref": "45 CFR 164.312(b) - Audit Controls",
            "status": "satisfied" if len(provenance) >= 2 else "partial",
            "provenance_chain_length": len(provenance),
            "events_logged": [
                {
                    "event_type": (
                        entry.get("event_type", "unknown")
                        if isinstance(entry, dict)
                        else getattr(entry, "event_type", "unknown")
                    ),
                    "has_timestamp": bool(
                        entry.get("timestamp")
                        if isinstance(entry, dict)
                        else getattr(entry, "timestamp", "")
                    ),
                }
                for entry in provenance[:10]  # Cap at 10 for readability
            ],
            "integrity_mechanism": "SHA-256 hash chain",
            "tamper_evident": bool(artifact_hash),
            "cryptographically_signed": has_signature,
        }

        # Minimum Necessary Standard (164.502(b))
        minimum_necessary = {
            "regulation_ref": "45 CFR 164.502(b) - Minimum Necessary Standard",
            "status": "satisfied",
            "controls": {
                "data_minimization": (
                    "Receipt stores decision summary (max 500 chars), not full input data."
                ),
                "purpose_limitation": "Data accessed only for decision validation and audit",
                "access_scope": (
                    f"{len(agents)} agents accessed only the data required "
                    "for their validation role."
                ),
            },
        }

        # Breach Notification Readiness (164.404-410)
        breach_notification = {
            "regulation_ref": "45 CFR 164.404-164.410 - Breach Notification Rule",
            "readiness_status": "prepared",
            "capabilities": {
                "detection": {
                    "tamper_detection": bool(artifact_hash),
                    "integrity_verification": "Receipt integrity can be verified via artifact_hash",
                    "signature_verification": has_signature,
                },
                "assessment": {
                    "risk_assessment_data": bool(receipt.get("risk_summary")),
                    "phi_scope_determinable": True,
                    "affected_records_trackable": True,
                },
                "notification_timeline": {
                    "individual_notification": "Without unreasonable delay, no later than 60 days",
                    "hhs_notification": "Annually for <500 records; within 60 days for >=500",
                    "media_notification": "Required if >=500 residents of a state affected",
                },
            },
        }

        # Safeguards (Administrative, Physical, Technical)
        safeguards = {
            "administrative": {
                "risk_analysis": bool(receipt.get("risk_summary")),
                "workforce_training": "Agent-specific role-based access",
                "contingency_plan": "Multi-agent fallback with circuit breakers",
            },
            "physical": {
                "facility_access": "Cloud-hosted with provider physical security controls",
                "workstation_security": "Agent isolation via AirlockProxy",
            },
            "technical": {
                "access_control": "RBAC with unique agent identifiers",
                "audit_controls": f"{len(provenance)} provenance events logged",
                "integrity_controls": {
                    "input_hash": bool(input_hash),
                    "artifact_hash": bool(artifact_hash),
                    "digital_signature": has_signature,
                },
                "transmission_security": "TLS 1.2+ enforced",
            },
        }

        return HIPAAArtifact(
            artifact_id=artifact_id,
            receipt_id=receipt_id,
            generated_at=timestamp_str,
            phi_handling=phi_handling,
            access_control=access_control,
            audit_trail=audit_trail,
            minimum_necessary=minimum_necessary,
            breach_notification=breach_notification,
            safeguards=safeguards,
        )

    # -- Internal helpers ---------------------------------------------------

    def _classify_eu_risk(self, text: str) -> tuple[str, str, list[str]]:
        """Classify EU AI Act risk level from free text.

        Returns:
            Tuple of (risk_level, rationale, applicable_articles).
        """
        # Check unacceptable first
        for kw in _EU_UNACCEPTABLE_KEYWORDS:
            if kw in text:
                return (
                    "unacceptable",
                    f"Matches prohibited AI practice: '{kw}'. Banned under Article 5.",
                    ["Article 5"],
                )

        # Check high-risk
        for kw in _EU_HIGH_RISK_KEYWORDS:
            if kw in text:
                return (
                    "high",
                    f"Matches high-risk category keyword: '{kw}'. "
                    "Subject to Annex III obligations.",
                    [
                        "Article 6 (Classification)",
                        "Article 9 (Risk management)",
                        "Article 13 (Transparency)",
                        "Article 14 (Human oversight)",
                        "Article 15 (Accuracy, robustness, cybersecurity)",
                    ],
                )

        # Check limited-risk
        for kw in _EU_LIMITED_KEYWORDS:
            if kw in text:
                return (
                    "limited",
                    f"Matches limited-risk keyword: '{kw}'. "
                    "Subject to transparency obligations under Article 50.",
                    ["Article 50 (Transparency obligations)"],
                )

        # Default: minimal
        return (
            "minimal",
            "No high-risk or limited-risk indicators detected. Minimal obligations apply.",
            [],
        )

    def _detect_human_oversight(self, config: dict[str, Any], receipt: dict[str, Any]) -> bool:
        """Detect whether human oversight was present in the decision process."""
        oversight_indicators = [
            "human_approval",
            "require_approval",
            "human_in_loop",
            "human_override",
            "approver",
            "approver_id",
            "approval_record",
        ]
        config_str = json.dumps(config).lower()
        for indicator in oversight_indicators:
            if indicator in config_str:
                return True

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
    "TransparencyChecklist",
    "EUAIActArtifact",
    "TrustServiceCriteria",
    "SOC2Artifact",
    "HIPAAArtifact",
    "ComplianceArtifactResult",
    "ReceiptComplianceGenerator",
]
