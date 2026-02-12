#!/usr/bin/env python3
"""
Healthcare Clinical Decision Review Example
=============================================

Demonstrates Aragora's healthcare vertical integration:
1. Construct a FHIR R4 Bundle with patient clinical data
2. Run an adversarial debate using the healthcare_hipaa profile
3. Generate a HIPAA-compliant receipt with PHI redaction

Time: ~3-5 minutes (requires API keys)
Requirements: At least one API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)

Usage:
    python examples/healthcare/clinical_decision.py
    python examples/healthcare/clinical_decision.py --demo  # Uses demo bundle
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path if running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora.cli.commands.healthcare import (
    DEMO_CLINICAL_QUESTION,
    DEMO_FHIR_BUNDLE,
    fhir_bundle_to_clinical_summary,
    run_healthcare_review,
    strip_phi_from_metadata,
)


def build_sample_fhir_bundle() -> dict:
    """Build a sample FHIR R4 Bundle for a diabetes patient.

    In production, this would come from an EHR system via the FHIR connector.
    Here we construct it manually to demonstrate the data flow.
    """
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "example-001",
                    "gender": "female",
                    "birthDate": "1965",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-a",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "73211009",
                                "display": "Diabetes mellitus (disorder)",
                            }
                        ],
                        "text": "Type 2 Diabetes Mellitus",
                    },
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "onsetDateTime": "2018-05-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-a",
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "4548-4",
                                "display": "Hemoglobin A1c",
                            }
                        ],
                        "text": "HbA1c",
                    },
                    "valueQuantity": {"value": 9.1, "unit": "%"},
                    "effectiveDateTime": "2026-01-15",
                    "interpretation": [
                        {"coding": [{"code": "H", "display": "High"}]}
                    ],
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-a",
                    "status": "active",
                    "intent": "order",
                    "medicationCodeableConcept": {
                        "text": "Metformin 1000mg twice daily",
                    },
                    "dosageInstruction": [
                        {"text": "1000mg twice daily with meals"}
                    ],
                }
            },
        ],
    }


async def main():
    """Run the clinical decision review example."""
    print("=" * 60)
    print("Aragora Healthcare Clinical Decision Review")
    print("=" * 60)
    print()

    # Step 1: Build FHIR data
    use_demo = "--demo" in sys.argv
    if use_demo:
        fhir_bundle = DEMO_FHIR_BUNDLE
        clinical_question = DEMO_CLINICAL_QUESTION
        print("[Step 1] Using demo FHIR bundle and clinical question")
    else:
        fhir_bundle = build_sample_fhir_bundle()
        clinical_question = (
            "Patient has Type 2 Diabetes with HbA1c of 9.1%, currently on "
            "Metformin 1000mg BID (maximum dose). Should we add a GLP-1 "
            "receptor agonist (semaglutide) or a SGLT2 inhibitor "
            "(empagliflozin)? Consider cardiovascular risk profile, "
            "renal function, and patient preference for injection vs oral."
        )
        print("[Step 1] Built custom FHIR bundle")

    # Step 2: Show the clinical context
    print()
    print("[Step 2] Clinical context extracted from FHIR:")
    print("-" * 40)
    summary = fhir_bundle_to_clinical_summary(fhir_bundle)
    print(summary)
    print()

    # Step 3: Show the clinical question
    print("[Step 3] Clinical question for debate:")
    print("-" * 40)
    print(clinical_question)
    print()

    # Step 4: Run the healthcare review
    print("[Step 4] Running adversarial clinical review...")
    print("  Profile: healthcare_hipaa")
    print("  Agents: anthropic-api (proposer), openai-api (critic), anthropic-api (synthesizer)")
    print()

    try:
        result = await run_healthcare_review(
            clinical_input=clinical_question,
            fhir_bundle=fhir_bundle,
            verbose=True,
        )
    except Exception as e:
        print(f"\nReview failed: {e}")
        print("Ensure you have at least one API key configured:")
        print("  export ANTHROPIC_API_KEY=...")
        print("  export OPENAI_API_KEY=...")
        sys.exit(1)

    # Step 5: Show the receipt
    receipt = result["receipt"]
    print()
    print("[Step 5] HIPAA-compliant receipt generated:")
    print("-" * 40)
    print(f"Receipt ID:       {receipt['receipt_id'][:12]}...")
    print(f"Profile:          {receipt['profile']}")
    print(f"HIPAA Compliant:  {receipt['compliance']['hipaa_compliant']}")
    print(f"PHI Redacted:     {receipt['compliance']['phi_redacted']}")
    print(f"Consensus:        {'Reached' if receipt['verdict']['consensus_reached'] else 'Not reached'}")
    print(f"Confidence:       {receipt['verdict']['confidence']:.1%}")
    print()

    if receipt["verdict"].get("final_answer"):
        print("Clinical recommendation (first 500 chars):")
        print(receipt["verdict"]["final_answer"][:500])
        print()

    print(f"Agents consulted: {receipt['audit_trail']['agents_consulted']}")
    print(f"Rounds:           {receipt['audit_trail']['rounds_completed']}")
    print(f"Dissenting views: {receipt['audit_trail']['dissenting_views_count']}")
    print(f"Artifact hash:    {receipt['integrity']['artifact_hash'][:32]}...")
    print()

    # Step 6: Demonstrate PHI stripping
    print("[Step 6] PHI stripping demonstration:")
    print("-" * 40)
    sample_metadata = {
        "patient_name": "Jane Doe",
        "mrn": "12345",
        "condition": "Type 2 Diabetes",
        "hba1c": 9.1,
        "email": "jane@example.com",
    }
    cleaned = strip_phi_from_metadata(sample_metadata)
    print(f"  Before: {json.dumps(sample_metadata, indent=4)}")
    print(f"  After:  {json.dumps(cleaned, indent=4)}")
    print()

    print("Done. Receipt is HIPAA-compliant and audit-ready.")


if __name__ == "__main__":
    asyncio.run(main())
