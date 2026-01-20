"""
Healthcare vertical knowledge module.

Provides domain-specific fact extraction, validation, and pattern detection
for healthcare documents including:
- Clinical terminology and findings
- HIPAA compliance
- Drug interactions and contraindications
- Medical coding (ICD-10, CPT)
- Treatment protocols
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
class ClinicalPattern:
    """Pattern for detecting clinical terms."""

    name: str
    pattern: str
    category: str  # diagnosis, treatment, symptom, vital, etc.
    sensitivity: str  # high, medium, low (PHI sensitivity)
    description: str


@dataclass
class DrugPattern:
    """Pattern for detecting drug-related information."""

    name: str
    pattern: str
    category: str  # medication, dosage, interaction, contraindication
    description: str


class HealthcareKnowledge(BaseVerticalKnowledge):
    """
    Healthcare vertical knowledge module.

    Specializes in:
    - Clinical documentation analysis
    - HIPAA compliance verification
    - Drug and treatment extraction
    - Medical coding references
    - Patient safety patterns
    """

    # Clinical terminology patterns
    CLINICAL_PATTERNS = [
        ClinicalPattern(
            name="Diagnosis",
            pattern=r"(?:diagnos(?:is|ed|tic)|Dx|assessed\s+as|impression)",
            category="diagnosis",
            sensitivity="high",
            description="Clinical diagnosis or assessment",
        ),
        ClinicalPattern(
            name="Chief Complaint",
            pattern=r"(?:chief\s+complaint|presents?\s+with|c/o)",
            category="symptom",
            sensitivity="high",
            description="Patient's primary complaint",
        ),
        ClinicalPattern(
            name="Vital Signs",
            pattern=r"(?:vital\s+signs?|BP|blood\s+pressure|HR|heart\s+rate|RR|respiratory\s+rate|temp(?:erature)?)",
            category="vital",
            sensitivity="medium",
            description="Patient vital signs",
        ),
        ClinicalPattern(
            name="Physical Exam",
            pattern=r"(?:physical\s+exam(?:ination)?|PE|on\s+exam|noted\s+(?:to\s+be)?)",
            category="exam",
            sensitivity="medium",
            description="Physical examination findings",
        ),
        ClinicalPattern(
            name="Lab Results",
            pattern=r"(?:lab(?:oratory)?\s+(?:results?|values?)|CBC|BMP|CMP|HbA1c|lipid\s+panel)",
            category="laboratory",
            sensitivity="high",
            description="Laboratory test results",
        ),
        ClinicalPattern(
            name="Imaging",
            pattern=r"(?:imaging|X-?ray|CT\s+scan|MRI|ultrasound|radiograph)",
            category="imaging",
            sensitivity="high",
            description="Imaging studies and results",
        ),
        ClinicalPattern(
            name="Treatment Plan",
            pattern=r"(?:treatment\s+plan|plan(?:ned)?|recommend(?:ation)?s?|Rx)",
            category="treatment",
            sensitivity="medium",
            description="Treatment recommendations",
        ),
        ClinicalPattern(
            name="Allergy",
            pattern=r"(?:allerg(?:y|ies|ic)|NKDA|no\s+known\s+(?:drug\s+)?allergies)",
            category="allergy",
            sensitivity="high",
            description="Allergy information",
        ),
        ClinicalPattern(
            name="Medical History",
            pattern=r"(?:medical\s+history|PMH|past\s+(?:medical\s+)?history|h/o)",
            category="history",
            sensitivity="high",
            description="Patient medical history",
        ),
        ClinicalPattern(
            name="Procedure",
            pattern=r"(?:procedure|surgery|operation|biopsy|resection)",
            category="procedure",
            sensitivity="high",
            description="Medical procedures",
        ),
    ]

    # Drug-related patterns
    DRUG_PATTERNS = [
        DrugPattern(
            name="Medication",
            pattern=r"(?:medication|prescription|Rx|prescribed|taking)",
            category="medication",
            description="Medication reference",
        ),
        DrugPattern(
            name="Dosage",
            pattern=r"(?:\d+\s*(?:mg|mcg|g|ml|units?|tabs?|caps?)|(?:q\.?d\.?|b\.?i\.?d\.?|t\.?i\.?d\.?|q\.?i\.?d\.?|PRN))",
            category="dosage",
            description="Drug dosage information",
        ),
        DrugPattern(
            name="Drug Interaction",
            pattern=r"(?:interaction|contraindicated\s+with|do\s+not\s+(?:use|take)\s+with)",
            category="interaction",
            description="Drug interaction warning",
        ),
        DrugPattern(
            name="Side Effect",
            pattern=r"(?:side\s+effect|adverse\s+(?:event|reaction)|AE)",
            category="adverse",
            description="Adverse effects",
        ),
        DrugPattern(
            name="Contraindication",
            pattern=r"(?:contraindicated?|should\s+not\s+(?:be\s+)?(?:used|given)|avoid)",
            category="contraindication",
            description="Drug contraindication",
        ),
    ]

    # HIPAA-specific patterns
    HIPAA_PATTERNS = [
        (r"(?:patient\s+name|full\s+name)", "PHI - Patient Name"),
        (r"(?:date\s+of\s+birth|DOB|birthdate)", "PHI - Date of Birth"),
        (r"(?:social\s+security|SSN)", "PHI - Social Security Number"),
        (r"(?:medical\s+record\s+number|MRN)", "PHI - Medical Record Number"),
        (r"(?:address|street|city,?\s+state)", "PHI - Address"),
        (r"(?:phone|telephone|fax)", "PHI - Phone/Fax"),
        (r"(?:email\s+address)", "PHI - Email"),
        (r"(?:health\s+plan|insurance\s+(?:id|number))", "PHI - Health Plan ID"),
    ]

    # Medical coding patterns
    CODING_PATTERNS = [
        (r"(?:ICD-?10|ICD-?9)", "ICD Diagnosis Code"),
        (r"(?:CPT|procedure\s+code)", "CPT Procedure Code"),
        (r"(?:HCPCS)", "HCPCS Code"),
        (r"(?:NDC|drug\s+code)", "NDC Drug Code"),
        (r"(?:LOINC)", "LOINC Code"),
        (r"(?:SNOMED)", "SNOMED CT"),
    ]

    @property
    def vertical_id(self) -> str:
        return "healthcare"

    @property
    def display_name(self) -> str:
        return "Healthcare & Clinical"

    @property
    def description(self) -> str:
        return "Clinical documentation, HIPAA compliance, drug information, and medical coding"

    @property
    def capabilities(self) -> VerticalCapabilities:
        return VerticalCapabilities(
            supports_pattern_detection=True,
            supports_cross_reference=True,
            supports_compliance_check=True,
            requires_llm=False,
            requires_vector_search=True,
            pattern_categories=[
                "clinical",
                "diagnosis",
                "treatment",
                "medication",
                "phi",
                "coding",
            ],
            compliance_frameworks=["HIPAA", "HITECH", "FDA", "CMS"],
            document_types=[
                "clinical_note",
                "lab_report",
                "prescription",
                "discharge_summary",
                "protocol",
            ],
        )

    @property
    def decay_rates(self) -> dict[str, float]:
        """Healthcare-specific decay rates."""
        return {
            "clinical": 0.02,  # Clinical findings may need updates
            "diagnosis": 0.01,  # Diagnoses are fairly stable
            "treatment": 0.03,  # Treatment guidelines evolve
            "medication": 0.05,  # Drug information changes more frequently
            "protocol": 0.03,  # Protocols update periodically
            "coding": 0.02,  # Coding standards update annually
            "phi": 0.01,  # PHI facts are stable
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
        """Extract healthcare facts from clinical content."""
        facts = []
        metadata = metadata or {}

        # Extract clinical facts
        for clinical in self.CLINICAL_PATTERNS:
            matches = re.findall(clinical.pattern, content, re.IGNORECASE)
            if matches:
                facts.append(
                    self.create_fact(
                        content=f"Clinical finding: {clinical.name} - {clinical.description}",
                        category="clinical",
                        confidence=0.75,
                        provenance={
                            "pattern": clinical.name,
                            "clinical_category": clinical.category,
                            "match_count": len(matches),
                        },
                        metadata={
                            "sensitivity": clinical.sensitivity,
                            "clinical_type": clinical.category,
                            **metadata,
                        },
                    )
                )

        # Extract drug-related facts
        for drug in self.DRUG_PATTERNS:
            if re.search(drug.pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"Drug information: {drug.name} - {drug.description}",
                        category="medication",
                        confidence=0.7,
                        provenance={"pattern": drug.name},
                        metadata={
                            "drug_category": drug.category,
                            **metadata,
                        },
                    )
                )

        # Extract PHI indicators
        for pattern, phi_type in self.HIPAA_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"PHI detected: {phi_type}",
                        category="phi",
                        confidence=0.85,
                        provenance={"phi_type": phi_type},
                        metadata={
                            "sensitivity": "high",
                            "requires_protection": True,
                            **metadata,
                        },
                    )
                )

        # Extract medical coding references
        for pattern, code_type in self.CODING_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                facts.append(
                    self.create_fact(
                        content=f"Medical coding reference: {code_type}",
                        category="coding",
                        confidence=0.8,
                        provenance={"code_type": code_type},
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
        Validate a healthcare fact.

        Clinical facts require careful validation as they may affect
        patient care decisions.
        """
        if fact.category == "medication":
            # Drug information may become outdated
            new_confidence = max(0.5, fact.confidence * 0.95)
            return True, new_confidence

        if fact.category == "clinical":
            # Clinical findings are generally stable
            return True, min(0.9, fact.confidence * 1.01)

        if fact.category == "phi":
            # PHI identification should remain stable
            return True, min(0.95, fact.confidence * 1.02)

        if fact.category == "coding":
            # Coding standards update annually
            return True, fact.confidence * 0.99

        return True, fact.confidence

    # -------------------------------------------------------------------------
    # Pattern Detection
    # -------------------------------------------------------------------------

    async def detect_patterns(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[PatternMatch]:
        """Detect patterns across healthcare facts."""
        patterns = []

        # Group facts by category
        by_category: dict[str, list[VerticalFact]] = {}
        for fact in facts:
            by_category.setdefault(fact.category, []).append(fact)

        # Pattern: Multiple PHI types detected
        phi_facts = by_category.get("phi", [])
        if len(phi_facts) >= 3:
            patterns.append(
                PatternMatch(
                    pattern_id=f"phi_concentration_{uuid.uuid4().hex[:8]}",
                    pattern_name="PHI Concentration",
                    pattern_type="privacy_risk",
                    description=f"Document contains {len(phi_facts)} types of PHI - enhanced protection required",
                    confidence=0.9,
                    supporting_facts=[f.id for f in phi_facts],
                    metadata={"phi_count": len(phi_facts)},
                )
            )

        # Pattern: Drug interaction potential
        med_facts = by_category.get("medication", [])
        interaction_facts = [
            f for f in med_facts if f.metadata.get("drug_category") == "interaction"
        ]
        if interaction_facts:
            patterns.append(
                PatternMatch(
                    pattern_id=f"drug_interaction_{uuid.uuid4().hex[:8]}",
                    pattern_name="Drug Interaction Alert",
                    pattern_type="safety",
                    description="Potential drug interaction detected",
                    confidence=0.85,
                    supporting_facts=[f.id for f in interaction_facts],
                )
            )

        # Pattern: Complex clinical case
        clinical_facts = by_category.get("clinical", [])
        if len(clinical_facts) >= 5:
            patterns.append(
                PatternMatch(
                    pattern_id=f"complex_case_{uuid.uuid4().hex[:8]}",
                    pattern_name="Complex Clinical Case",
                    pattern_type="complexity",
                    description="Multiple clinical findings suggest complex case",
                    confidence=0.7,
                    supporting_facts=[f.id for f in clinical_facts[:5]],
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
        """Check compliance against healthcare frameworks."""
        results = []

        if framework.upper() == "HIPAA":
            results.extend(await self._check_hipaa_compliance(facts))
        elif framework.upper() == "HITECH":
            results.extend(await self._check_hitech_compliance(facts))

        return results

    async def _check_hipaa_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check HIPAA compliance."""
        results = []
        phi_facts = [f for f in facts if f.category == "phi"]

        if phi_facts:
            # Document contains PHI - check for protection measures
            results.append(
                ComplianceCheckResult(
                    rule_id="hipaa_phi_protection",
                    rule_name="HIPAA PHI Protection",
                    framework="HIPAA",
                    passed=False,  # Document contains PHI, needs protection review
                    severity="high",
                    findings=[f"Found {len(phi_facts)} PHI elements requiring protection"],
                    evidence=[f.id for f in phi_facts],
                    recommendations=[
                        "Ensure minimum necessary standard is applied",
                        "Verify access controls are in place",
                        "Confirm encryption at rest and in transit",
                    ],
                    confidence=0.85,
                )
            )

        # Check for all 18 HIPAA identifiers
        phi_types = {
            f.provenance.get("phi_type") for f in phi_facts if f.provenance.get("phi_type")
        }
        if len(phi_types) >= 5:
            results.append(
                ComplianceCheckResult(
                    rule_id="hipaa_deidentification",
                    rule_name="HIPAA De-identification Required",
                    framework="HIPAA",
                    passed=False,
                    severity="high",
                    findings=[
                        f"Document contains {len(phi_types)} PHI categories - de-identification may be required"
                    ],
                    evidence=[f.id for f in phi_facts],
                    recommendations=[
                        "Consider Safe Harbor de-identification method",
                        "Review expert determination requirements",
                    ],
                    confidence=0.8,
                )
            )

        return results

    async def _check_hitech_compliance(
        self,
        facts: Sequence[VerticalFact],
    ) -> list[ComplianceCheckResult]:
        """Check HITECH compliance."""
        results = []
        phi_facts = [f for f in facts if f.category == "phi"]

        if phi_facts:
            results.append(
                ComplianceCheckResult(
                    rule_id="hitech_breach_notification",
                    rule_name="HITECH Breach Notification",
                    framework="HITECH",
                    passed=True,  # Advisory
                    severity="medium",
                    findings=["PHI present - breach notification requirements apply"],
                    evidence=[f.id for f in phi_facts],
                    recommendations=[
                        "Ensure breach notification procedures are documented",
                        "Maintain audit logs for PHI access",
                    ],
                    confidence=0.75,
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
        """Find related healthcare facts via cross-reference."""
        references = []

        # Link medications with clinical findings
        if fact.category == "medication":
            for other in other_facts:
                if other.id == fact.id:
                    continue
                if other.category == "clinical":
                    references.append((other.id, "treatment_for", 0.6))

        # Link diagnoses with treatments
        if fact.category == "clinical" and fact.metadata.get("clinical_type") == "diagnosis":
            for other in other_facts:
                if other.id == fact.id:
                    continue
                if other.metadata.get("clinical_type") == "treatment":
                    references.append((other.id, "treated_by", 0.7))

        return references
