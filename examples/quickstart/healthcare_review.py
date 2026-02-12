#!/usr/bin/env python3
"""
Healthcare Clinical Decision Review with HIPAA-Compliant Receipt.

Demonstrates a clinical decision support workflow:
1. Patient data is ingested (de-identified for this demo)
2. Multiple models independently evaluate the treatment plan
3. Models debate -- challenging each other on drug interactions,
   contraindications, and evidence quality
4. A Decision Receipt is produced for the clinical record

This example uses mocked responses. In production, connect to FHIR/HL7
endpoints and use real LLM providers.

Usage:
    python examples/quickstart/healthcare_review.py
"""

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# --- Mock clinical data (de-identified) ---

CLINICAL_QUESTION = """
Patient: 68-year-old, history of Type 2 diabetes (A1C 7.8%), hypertension,
and chronic kidney disease (Stage 3a, eGFR 52).

Current medications: metformin 1000mg BID, lisinopril 20mg daily,
atorvastatin 40mg daily.

Presenting complaint: Persistent joint pain in both knees, affecting mobility.

Question: Evaluate the proposed addition of ibuprofen 400mg TID for
pain management. Consider drug interactions, renal implications,
and alternative approaches.
"""


@dataclass
class ModelPosition:
    """A single model's position in the debate."""

    model: str
    recommendation: str
    confidence: float
    key_concerns: list[str]
    evidence_cited: list[str]


@dataclass
class ClinicalReceipt:
    """HIPAA-compliant decision receipt for clinical decisions."""

    receipt_id: str
    timestamp: str
    clinical_question: str
    models_consulted: list[str]
    positions: list[ModelPosition]
    consensus_reached: bool
    consensus_recommendation: str
    dissent_points: list[str]
    confidence_score: float
    evidence_chain: list[str]
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash over receipt content."""
        content = json.dumps(
            {
                "receipt_id": self.receipt_id,
                "timestamp": self.timestamp,
                "clinical_question": self.clinical_question,
                "consensus_recommendation": self.consensus_recommendation,
                "dissent_points": self.dissent_points,
                "confidence_score": self.confidence_score,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()


def run_mock_clinical_debate() -> ClinicalReceipt:
    """Simulate a multi-model clinical decision debate."""

    positions = [
        ModelPosition(
            model="claude-medical",
            recommendation="AGAINST ibuprofen. Recommend acetaminophen or topical NSAID.",
            confidence=0.92,
            key_concerns=[
                "NSAIDs contraindicated with CKD Stage 3a (eGFR 52) -- risk of further renal decline",
                "Ibuprofen-lisinopril interaction reduces antihypertensive effect and compounds renal risk",
                "Metformin clearance already reduced by CKD; adding renal stressor increases lactic acidosis risk",
            ],
            evidence_cited=[
                "KDIGO 2024 Guidelines: Avoid NSAIDs in CKD Stage 3+",
                "FDA Drug Safety Communication: NSAIDs and CKD (2023)",
            ],
        ),
        ModelPosition(
            model="gpt-medical",
            recommendation="AGAINST ibuprofen. Recommend duloxetine or physical therapy.",
            confidence=0.88,
            key_concerns=[
                "NSAID use in CKD patients associated with 25-30% increased risk of AKI",
                "Triple whammy combination: NSAID + ACE inhibitor + diuretic (if added) is high-risk",
                "Age 68 increases GI bleeding risk with NSAID use",
            ],
            evidence_cited=[
                "Dreischulte et al. (2015): Triple whammy AKI risk",
                "American Geriatrics Society Beers Criteria (2023): NSAIDs potentially inappropriate in elderly",
            ],
        ),
        ModelPosition(
            model="gemini-medical",
            recommendation="AGAINST systemic ibuprofen. Consider short-course topical diclofenac.",
            confidence=0.79,
            key_concerns=[
                "Systemic NSAID clearly contraindicated given CKD and ACE inhibitor",
                "Topical NSAID has lower systemic exposure -- may be acceptable for limited duration",
                "Should obtain rheumatology consult to rule out inflammatory arthritis",
            ],
            evidence_cited=[
                "Derry et al. (2015): Topical NSAIDs for osteoarthritis, Cochrane review",
                "ACR/AF 2019 Guidelines: First-line OA treatment recommendations",
            ],
        ),
    ]

    # Build receipt
    receipt = ClinicalReceipt(
        receipt_id="CR-2026-0212-001",
        timestamp=datetime.now(timezone.utc).isoformat(),
        clinical_question=CLINICAL_QUESTION.strip(),
        models_consulted=[p.model for p in positions],
        positions=positions,
        consensus_reached=True,
        consensus_recommendation=(
            "UNANIMOUS: Do not prescribe systemic ibuprofen. "
            "The combination of CKD Stage 3a, concurrent ACE inhibitor (lisinopril), "
            "and age-related GI bleeding risk makes systemic NSAIDs contraindicated. "
            "Alternative approaches: acetaminophen (first-line), topical NSAID "
            "(limited duration, with renal monitoring), duloxetine, physical therapy, "
            "or rheumatology referral."
        ),
        dissent_points=[
            "Models disagree on topical NSAID safety: claude-medical recommends against "
            "all NSAIDs, gemini-medical considers topical acceptable for short course. "
            "Recommend physician judgment on topical option with renal function monitoring.",
        ],
        confidence_score=0.86,
        evidence_chain=[
            "KDIGO 2024 Guidelines",
            "FDA Drug Safety Communication (2023)",
            "AGS Beers Criteria (2023)",
            "Dreischulte et al. (2015) -- Triple whammy AKI risk",
            "Derry et al. (2015) -- Cochrane review on topical NSAIDs",
            "ACR/AF 2019 OA Guidelines",
        ],
    )
    receipt.content_hash = receipt.compute_hash()
    return receipt


def print_receipt(receipt: ClinicalReceipt):
    """Print a formatted clinical decision receipt."""

    print("=" * 60)
    print("CLINICAL DECISION RECEIPT")
    print("=" * 60)
    print(f"Receipt ID:  {receipt.receipt_id}")
    print(f"Timestamp:   {receipt.timestamp}")
    print(f"Hash:        {receipt.content_hash[:16]}...")
    print()

    print("CLINICAL QUESTION:")
    for line in receipt.clinical_question.split("\n"):
        if line.strip():
            print(f"  {line.strip()}")
    print()

    print(f"MODELS CONSULTED: {', '.join(receipt.models_consulted)}")
    print()

    print("INDIVIDUAL POSITIONS:")
    for p in receipt.positions:
        print(f"\n  {p.model} (confidence: {p.confidence:.0%}):")
        print(f"    Recommendation: {p.recommendation}")
        print("    Key concerns:")
        for c in p.key_concerns:
            print(f"      - {c}")
    print()

    print("CONSENSUS RECOMMENDATION:")
    print(f"  {receipt.consensus_recommendation}")
    print()

    if receipt.dissent_points:
        print("DISSENT POINTS (where models disagreed):")
        for d in receipt.dissent_points:
            print(f"  - {d}")
        print()

    print(f"CONFIDENCE SCORE: {receipt.confidence_score:.0%}")
    print()

    print("EVIDENCE CHAIN:")
    for i, e in enumerate(receipt.evidence_chain, 1):
        print(f"  {i}. {e}")
    print()

    print(f"INTEGRITY HASH: {receipt.content_hash}")
    print("=" * 60)


def main():
    print("Aragora Healthcare Clinical Decision Review (Demo)")
    print("-" * 50)
    print()

    receipt = run_mock_clinical_debate()
    print_receipt(receipt)

    print()
    print("This receipt is suitable for inclusion in the clinical record.")
    print("In production, connect FHIR/HL7 endpoints and use real LLM providers:")
    print()
    print("  from aragora import Arena, Environment, DebateProtocol")
    print("  protocol = DebateProtocol(rounds=3, consensus='unanimous',")
    print("                            weight_profile='healthcare_hipaa')")


if __name__ == "__main__":
    main()
