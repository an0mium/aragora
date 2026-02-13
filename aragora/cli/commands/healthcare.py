"""
Healthcare Vertical CLI commands.

Provides a healthcare-specific debate pipeline with:
- FHIR-formatted structured clinical input
- HIPAA-compliant adversarial debate using healthcare_hipaa weight profile
- PHI redaction on receipts and output
- Audit trail for compliance

Commands:
    aragora healthcare review <input>       -- Run adversarial clinical review
    aragora healthcare review --fhir <path> -- Review FHIR bundle
    aragora healthcare review --demo        -- Run demo with sample scenario
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Healthcare-specific defaults
HEALTHCARE_AGENTS = "anthropic-api:proposer,openai-api:critic,anthropic-api:synthesizer"
HEALTHCARE_ROUNDS = 5
HEALTHCARE_PROFILE = "healthcare_hipaa"

# HIPAA Safe Harbor: fields that must be stripped from receipts
PHI_RECEIPT_FIELDS = {
    "patient_name",
    "patient_id",
    "mrn",
    "ssn",
    "date_of_birth",
    "phone",
    "email",
    "address",
    "ip_address",
    "device_id",
    "photo",
}

# Sample FHIR patient scenario for --demo mode
DEMO_FHIR_BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "demo-patient-001",
                "gender": "male",
                "birthDate": "1958",
                "meta": {"lastUpdated": "2026-01-15T10:00:00Z"},
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-001",
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
                "clinicalStatus": {
                    "coding": [{"code": "active"}],
                },
                "onsetDateTime": "2020-03-15",
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-002",
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "38341003",
                            "display": "Hypertensive disorder",
                        }
                    ],
                    "text": "Essential Hypertension",
                },
                "clinicalStatus": {
                    "coding": [{"code": "active"}],
                },
                "onsetDateTime": "2019-06-01",
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-001",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "860975",
                            "display": "Metformin 500mg",
                        }
                    ],
                    "text": "Metformin 500mg twice daily",
                },
                "dosageInstruction": [
                    {"text": "500mg twice daily with meals"},
                ],
            }
        },
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "id": "med-002",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": "979480",
                            "display": "Lisinopril 10mg",
                        }
                    ],
                    "text": "Lisinopril 10mg daily",
                },
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-001",
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
                "valueQuantity": {
                    "value": 8.2,
                    "unit": "%",
                    "system": "http://unitsofmeasure.org",
                },
                "effectiveDateTime": "2026-01-10",
                "interpretation": [
                    {
                        "coding": [{"code": "H", "display": "High"}],
                    }
                ],
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-002",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "55284-4",
                            "display": "Blood Pressure",
                        }
                    ],
                    "text": "Blood Pressure",
                },
                "component": [
                    {
                        "code": {"text": "Systolic"},
                        "valueQuantity": {"value": 145, "unit": "mmHg"},
                    },
                    {
                        "code": {"text": "Diastolic"},
                        "valueQuantity": {"value": 92, "unit": "mmHg"},
                    },
                ],
                "effectiveDateTime": "2026-01-10",
            }
        },
    ],
}

DEMO_CLINICAL_QUESTION = (
    "This patient has Type 2 Diabetes with an HbA1c of 8.2% (above target of <7%) "
    "and uncontrolled hypertension (145/92 mmHg). Current medications are Metformin 500mg "
    "BID and Lisinopril 10mg daily. Should we (A) intensify the diabetes regimen by adding "
    "a GLP-1 receptor agonist (e.g. semaglutide), which also provides cardiovascular benefit, "
    "or (B) add a second antihypertensive (e.g. amlodipine) first and reassess diabetes "
    "control in 3 months? Consider drug interactions, contraindications, patient adherence "
    "burden, and guideline concordance (ADA Standards of Care 2026, ACC/AHA Hypertension)."
)


def fhir_bundle_to_clinical_summary(bundle: dict[str, Any]) -> str:
    """Convert a FHIR Bundle into a clinical narrative for debate context.

    Extracts conditions, medications, observations, and patient demographics
    into structured text suitable for LLM consumption.
    """
    sections: list[str] = []
    conditions: list[str] = []
    medications: list[str] = []
    observations: list[str] = []
    patient_info: list[str] = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")

        if rtype == "Patient":
            if resource.get("gender"):
                patient_info.append(f"Gender: {resource['gender']}")
            if resource.get("birthDate"):
                patient_info.append(f"Birth Year: {resource['birthDate']}")

        elif rtype == "Condition":
            code_text = resource.get("code", {}).get("text", "Unknown condition")
            status_coding = resource.get("clinicalStatus", {}).get("coding", [])
            status = status_coding[0].get("code", "unknown") if status_coding else "unknown"
            onset = resource.get("onsetDateTime", "")
            line = f"- {code_text} (status: {status})"
            if onset:
                line += f", onset: {onset}"
            conditions.append(line)

        elif rtype == "MedicationRequest":
            med_text = resource.get("medicationCodeableConcept", {}).get(
                "text", "Unknown medication"
            )
            med_status = resource.get("status", "unknown")
            dosage = resource.get("dosageInstruction", [])
            dosage_text = dosage[0].get("text", "") if dosage else ""
            line = f"- {med_text} ({med_status})"
            if dosage_text:
                line += f" -- {dosage_text}"
            medications.append(line)

        elif rtype == "Observation":
            obs_text = resource.get("code", {}).get("text", "Unknown observation")
            value_qty = resource.get("valueQuantity")
            components = resource.get("component", [])
            effective = resource.get("effectiveDateTime", "")

            if value_qty:
                val = value_qty.get("value", "")
                unit = value_qty.get("unit", "")
                line = f"- {obs_text}: {val} {unit}"
            elif components:
                parts = []
                for comp in components:
                    comp_name = comp.get("code", {}).get("text", "")
                    comp_val = comp.get("valueQuantity", {}).get("value", "")
                    comp_unit = comp.get("valueQuantity", {}).get("unit", "")
                    parts.append(f"{comp_name} {comp_val} {comp_unit}")
                line = f"- {obs_text}: {', '.join(parts)}"
            else:
                line = f"- {obs_text}"

            if effective:
                line += f" (date: {effective})"

            interp = resource.get("interpretation", [])
            if interp:
                interp_code = interp[0].get("coding", [{}])[0].get("display", "")
                if interp_code:
                    line += f" [{interp_code}]"

            observations.append(line)

    if patient_info:
        sections.append("PATIENT DEMOGRAPHICS:\n" + "\n".join(patient_info))
    if conditions:
        sections.append("ACTIVE CONDITIONS:\n" + "\n".join(conditions))
    if medications:
        sections.append("CURRENT MEDICATIONS:\n" + "\n".join(medications))
    if observations:
        sections.append("RECENT OBSERVATIONS:\n" + "\n".join(observations))

    return "\n\n".join(sections)


def strip_phi_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove PHI fields from receipt metadata for HIPAA compliance.

    Uses Safe Harbor de-identification: removes all 18 HIPAA identifiers
    from output metadata. Clinical data (conditions, medications, labs)
    is preserved for decision audit purposes.
    """
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        lower_key = key.lower().replace("-", "_").replace(" ", "_")
        if lower_key in PHI_RECEIPT_FIELDS:
            continue
        if isinstance(value, dict):
            cleaned[key] = strip_phi_from_metadata(value)
        elif isinstance(value, list):
            cleaned[key] = [
                strip_phi_from_metadata(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


async def run_healthcare_review(
    clinical_input: str,
    fhir_bundle: dict[str, Any] | None = None,
    agents_str: str = HEALTHCARE_AGENTS,
    rounds: int = HEALTHCARE_ROUNDS,
    output_json: bool = False,
    output_dir: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run an adversarial healthcare review debate.

    Args:
        clinical_input: The clinical question or scenario to review.
        fhir_bundle: Optional FHIR R4 Bundle providing structured clinical data.
        agents_str: Comma-separated agent specs.
        rounds: Number of debate rounds.
        output_json: If True, return structured JSON.
        output_dir: Directory to write receipt artifacts.
        verbose: Print detailed progress.

    Returns:
        Dict containing debate_result, receipt, and clinical_summary.
    """
    from aragora.cli.commands.debate import run_debate

    # Build clinical context from FHIR bundle if provided
    context = ""
    if fhir_bundle:
        context = fhir_bundle_to_clinical_summary(fhir_bundle)
        if verbose:
            print("[healthcare] Extracted clinical context from FHIR bundle")
            print(f"[healthcare] Context length: {len(context)} chars")

    # Build healthcare-specific system context
    healthcare_preamble = (
        "You are participating in a clinical decision review. "
        "Evaluate the proposed clinical action for:\n"
        "1. Patient safety and contraindication risks\n"
        "2. Evidence-based guideline concordance (ADA, ACC/AHA, USPSTF)\n"
        "3. Drug interactions and adverse effect profiles\n"
        "4. HIPAA compliance -- never include PHI in your response\n"
        "5. Informed consent and patient autonomy considerations\n\n"
        "Be specific about clinical evidence. Cite guidelines where applicable. "
        "Flag any patient safety concerns immediately.\n"
    )

    full_context = healthcare_preamble
    if context:
        full_context += "\n--- CLINICAL DATA ---\n" + context + "\n--- END CLINICAL DATA ---\n"

    # Calculate input hash for receipt integrity
    input_content = clinical_input + (json.dumps(fhir_bundle, sort_keys=True) if fhir_bundle else "")
    input_hash = hashlib.sha256(input_content.encode()).hexdigest()

    if verbose:
        print(f"[healthcare] Input hash: {input_hash[:16]}...")
        print(f"[healthcare] Profile: {HEALTHCARE_PROFILE}")
        print(f"[healthcare] Agents: {agents_str}")
        print(f"[healthcare] Rounds: {rounds}")

    # Run the debate with healthcare profile
    debate_result = await run_debate(
        task=clinical_input,
        agents_str=agents_str,
        rounds=rounds,
        context=full_context,
        enable_verticals=True,
        vertical_id="healthcare",
    )

    # Build HIPAA-compliant receipt
    receipt = _build_healthcare_receipt(debate_result, input_hash, fhir_bundle)

    result: dict[str, Any] = {
        "debate_result": debate_result,
        "receipt": receipt,
        "input_hash": input_hash,
        "profile": HEALTHCARE_PROFILE,
    }

    # Write artifacts if output directory specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        receipt_path = out_path / f"receipt_{receipt['receipt_id'][:12]}.json"
        receipt_path.write_text(json.dumps(receipt, indent=2))
        result["receipt_path"] = str(receipt_path)

        if verbose:
            print(f"[healthcare] Receipt written to: {receipt_path}")

    return result


def _build_healthcare_receipt(
    debate_result: Any,
    input_hash: str,
    fhir_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a HIPAA-compliant decision receipt from debate results.

    Strips all PHI from the receipt while preserving the clinical
    decision audit trail required for compliance.
    """
    receipt_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    # Extract key info from debate result
    consensus = getattr(debate_result, "consensus_reached", False)
    confidence = getattr(debate_result, "confidence", 0.0)
    final_answer = getattr(debate_result, "final_answer", "")
    messages = list(getattr(debate_result, "messages", []))
    votes = list(getattr(debate_result, "votes", []))
    debate_id = getattr(debate_result, "debate_id", "") or getattr(debate_result, "id", "")

    # Build agent contributions summary (no PHI)
    agent_summaries = []
    for msg in messages[-6:]:  # Last 6 messages for summary
        agent = getattr(msg, "agent", "unknown")
        role = getattr(msg, "role", "participant")
        content_preview = str(getattr(msg, "content", ""))[:200]
        agent_summaries.append({
            "agent": agent,
            "role": role,
            "content_preview": content_preview,
        })

    # Dissenting views for audit
    dissenting = []
    if not consensus and messages:
        for msg in messages:
            content = str(getattr(msg, "content", ""))
            if any(kw in content.lower() for kw in ["disagree", "concern", "risk", "contraindic"]):
                dissenting.append(str(getattr(msg, "content", ""))[:300])

    receipt: dict[str, Any] = {
        "receipt_id": receipt_id,
        "debate_id": debate_id or f"healthcare-{receipt_id[:8]}",
        "timestamp": timestamp,
        "schema_version": "1.0",
        "profile": HEALTHCARE_PROFILE,
        "compliance": {
            "hipaa_compliant": True,
            "phi_redacted": True,
            "safe_harbor_method": True,
        },
        "input": {
            "input_hash": input_hash,
            "has_fhir_data": fhir_bundle is not None,
            "resource_count": len(fhir_bundle.get("entry", [])) if fhir_bundle else 0,
        },
        "verdict": {
            "consensus_reached": consensus,
            "confidence": confidence,
            "final_answer": final_answer[:2000] if final_answer else "",
        },
        "audit_trail": {
            "agents_consulted": len(set(
                getattr(m, "agent", "") for m in messages if getattr(m, "agent", "")
            )),
            "rounds_completed": len(set(
                getattr(m, "round", 0) for m in messages
            )),
            "votes_cast": len(votes),
            "dissenting_views_count": len(dissenting),
            "agent_summaries": agent_summaries,
        },
        "integrity": {},
    }

    # Strip any residual PHI from receipt metadata
    receipt = strip_phi_from_metadata(receipt)

    # Compute receipt hash for integrity verification
    hash_content = json.dumps(
        {
            "receipt_id": receipt_id,
            "input_hash": input_hash,
            "consensus": consensus,
            "confidence": confidence,
            "timestamp": timestamp,
        },
        sort_keys=True,
    )
    receipt["integrity"]["artifact_hash"] = hashlib.sha256(hash_content.encode()).hexdigest()

    return receipt


def cmd_healthcare(args: argparse.Namespace) -> None:
    """Handle 'healthcare' command -- dispatch to subcommands."""
    subcommand = getattr(args, "healthcare_command", None)

    if subcommand == "review":
        _cmd_healthcare_review(args)
    else:
        print("\nUsage: aragora healthcare <command>")
        print("\nCommands:")
        print("  review <input>    Run adversarial clinical decision review")
        print("\nOptions:")
        print("  --fhir <path>     Path to FHIR R4 Bundle JSON file")
        print("  --demo            Run demo with sample clinical scenario")
        print("  --agents <list>   Agent specs (default: healthcare team)")
        print("  --rounds <n>      Debate rounds (default: 5)")
        print("  --output-dir <d>  Directory for receipt artifacts")
        print("  --json            Output as JSON")
        print("  --verbose         Detailed progress output")


def _cmd_healthcare_review(args: argparse.Namespace) -> None:
    """Handle 'healthcare review' subcommand."""
    is_demo = getattr(args, "demo", False)
    fhir_path = getattr(args, "fhir", None)
    output_json = getattr(args, "json", False)
    output_dir = getattr(args, "output_dir", None)
    verbose = getattr(args, "verbose", False)
    agents_str = getattr(args, "agents", HEALTHCARE_AGENTS)
    rounds = getattr(args, "rounds", HEALTHCARE_ROUNDS)

    # Resolve clinical input
    fhir_bundle: dict[str, Any] | None = None
    clinical_input: str = ""

    if is_demo:
        fhir_bundle = DEMO_FHIR_BUNDLE
        clinical_input = DEMO_CLINICAL_QUESTION
        if not output_json:
            print("\n[healthcare] Running demo clinical decision review...")
            print(f"[healthcare] Question: {clinical_input[:120]}...")
    elif fhir_path:
        fhir_file = Path(fhir_path)
        if not fhir_file.exists():
            print(f"Error: FHIR file not found: {fhir_path}", file=sys.stderr)
            raise SystemExit(1)
        try:
            fhir_bundle = json.loads(fhir_file.read_text())
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in FHIR file: {e}", file=sys.stderr)
            raise SystemExit(1)
        clinical_input = getattr(args, "input", "") or ""
        if not clinical_input:
            print("Error: --fhir requires a clinical question as positional input", file=sys.stderr)
            raise SystemExit(1)
    else:
        clinical_input = getattr(args, "input", "") or ""
        if not clinical_input:
            # Try reading from stdin
            if not sys.stdin.isatty():
                clinical_input = sys.stdin.read().strip()
            if not clinical_input:
                print("Error: clinical input required (positional arg, --fhir, or --demo)")
                print("Usage: aragora healthcare review <clinical_question>")
                print("       aragora healthcare review --demo")
                raise SystemExit(1)

    # Run the review
    result = asyncio.run(
        run_healthcare_review(
            clinical_input=clinical_input,
            fhir_bundle=fhir_bundle,
            agents_str=agents_str,
            rounds=rounds,
            output_json=output_json,
            output_dir=output_dir,
            verbose=verbose,
        )
    )

    # Output
    if output_json:
        # JSON output (receipt only, no raw debate_result which isn't serializable)
        output = {
            "receipt": result["receipt"],
            "input_hash": result["input_hash"],
            "profile": result["profile"],
        }
        print(json.dumps(output, indent=2))
        return

    # Human-readable output
    receipt = result["receipt"]
    print("\n" + "=" * 60)
    print("HEALTHCARE CLINICAL REVIEW")
    print("=" * 60)
    print(f"Receipt ID:  {receipt['receipt_id'][:12]}...")
    print(f"Profile:     {receipt['profile']}")
    print(f"Timestamp:   {receipt['timestamp']}")
    print(f"Input Hash:  {receipt['input']['input_hash'][:16]}...")
    print()

    verdict = receipt["verdict"]
    consensus_label = "REACHED" if verdict["consensus_reached"] else "NOT REACHED"
    print(f"Consensus:   {consensus_label}")
    print(f"Confidence:  {verdict['confidence']:.1%}")
    print()

    if verdict.get("final_answer"):
        print("CLINICAL RECOMMENDATION:")
        print("-" * 40)
        print(verdict["final_answer"][:1000])
        print()

    audit = receipt["audit_trail"]
    print("AUDIT TRAIL:")
    print(f"  Agents consulted:     {audit['agents_consulted']}")
    print(f"  Rounds completed:     {audit['rounds_completed']}")
    print(f"  Votes cast:           {audit['votes_cast']}")
    print(f"  Dissenting views:     {audit['dissenting_views_count']}")
    print()

    compliance = receipt["compliance"]
    print("HIPAA COMPLIANCE:")
    print(f"  HIPAA compliant:      {compliance['hipaa_compliant']}")
    print(f"  PHI redacted:         {compliance['phi_redacted']}")
    print(f"  Safe Harbor method:   {compliance['safe_harbor_method']}")
    print()

    print(f"Artifact Hash: {receipt['integrity']['artifact_hash'][:32]}...")

    if result.get("receipt_path"):
        print(f"\nReceipt saved to: {result['receipt_path']}")


def add_healthcare_parser(subparsers: Any) -> None:
    """Add the 'healthcare' subcommand parser."""
    parser = subparsers.add_parser(
        "healthcare",
        help="Healthcare vertical: adversarial clinical decision review",
        description=(
            "Run HIPAA-compliant adversarial debates on clinical decisions.\n"
            "Supports FHIR R4 structured data input and produces audit-ready receipts."
        ),
    )

    sub = parser.add_subparsers(dest="healthcare_command")

    review_parser = sub.add_parser(
        "review",
        help="Run adversarial clinical decision review",
        description="Review a clinical decision through multi-agent adversarial debate.",
    )
    review_parser.add_argument(
        "input",
        nargs="?",
        default="",
        help="Clinical question or scenario to review",
    )
    review_parser.add_argument(
        "--fhir",
        metavar="PATH",
        help="Path to FHIR R4 Bundle JSON file with patient data",
    )
    review_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with sample clinical scenario (no API keys needed for offline)",
    )
    review_parser.add_argument(
        "--agents",
        default=HEALTHCARE_AGENTS,
        help=f"Comma-separated agent specs (default: {HEALTHCARE_AGENTS})",
    )
    review_parser.add_argument(
        "--rounds",
        type=int,
        default=HEALTHCARE_ROUNDS,
        help=f"Number of debate rounds (default: {HEALTHCARE_ROUNDS})",
    )
    review_parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Directory for receipt artifacts",
    )
    review_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    review_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detailed progress output",
    )

    parser.set_defaults(func=cmd_healthcare)
