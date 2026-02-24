#!/usr/bin/env python3
"""EU AI Act Compliance Demo: Automated Conformity Assessment.

Demonstrates Aragora's EU AI Act compliance infrastructure:
1. Classify AI use cases by risk level (Art. 5/6 + Annex III)
2. Generate conformity reports from decision receipts
3. Assess article compliance (Art. 9, 12, 13, 14, 15)
4. Generate dedicated Art. 12/13/14 compliance artifacts
5. Export audit-ready artifact bundles (JSON + Markdown)
6. Show deployer log retention obligations (Art. 26)

The EU AI Act (Regulation (EU) 2024/1689) takes effect August 2, 2026
for Annex III high-risk systems. Aragora maps decision receipts to
article requirements automatically.

Usage:
    python examples/eu_ai_act_compliance.py --demo       # Full demo (default)
    python examples/eu_ai_act_compliance.py --classify    # Risk classification only
    python examples/eu_ai_act_compliance.py --report      # Generate conformity report
    python examples/eu_ai_act_compliance.py --artifacts   # Art. 12/13/14 artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any

# Import from Aragora compliance module
from aragora.compliance.eu_ai_act import (
    ComplianceArtifactGenerator,
    ConformityReportGenerator,
    RiskClassifier,
    RiskLevel,
)


# =============================================================================
# Sample Decision Receipts (simulating real Aragora output)
# =============================================================================


def _hr_recruitment_receipt() -> dict[str, Any]:
    """Receipt for a high-risk HR screening system (Annex III Cat. 4)."""
    return {
        "receipt_id": "RCP-HR-2026-0041",
        "input_summary": (
            "Should we deploy the AI-powered CV screening and recruitment "
            "decision system to automate initial candidate filtering for "
            "engineering roles?"
        ),
        "verdict": "approve_with_conditions",
        "verdict_reasoning": (
            "The recruitment screening system meets accuracy thresholds but "
            "requires additional bias auditing before deployment. The hiring "
            "decision model shows 94% agreement between AI recommendations and "
            "human recruiter decisions on the validation set. However, "
            "demographic parity gaps of 8% were detected in the gender dimension."
        ),
        "confidence": 0.78,
        "robustness_score": 0.72,
        "risk_summary": {
            "total": 5,
            "critical": 0,
            "high": 2,
            "medium": 2,
            "low": 1,
        },
        "consensus_proof": {
            "method": "weighted_majority",
            "threshold": 0.66,
            "supporting_agents": ["claude-analyst", "mistral-auditor", "gpt4-ethics"],
            "dissenting_agents": ["gemini-challenger"],
            "agreement_ratio": 0.75,
        },
        "dissenting_views": [
            {
                "agent": "gemini-challenger",
                "view": (
                    "Demographic parity gap of 8% exceeds the 5% threshold for "
                    "employment AI under the EU AI Act. Recommend delaying "
                    "deployment until bias mitigation reduces the gap below 3%."
                ),
            },
        ],
        "provenance_chain": [
            {
                "event_type": "debate_started",
                "timestamp": "2026-02-12T10:00:00Z",
                "actor": "system",
            },
            {
                "event_type": "proposal_submitted",
                "timestamp": "2026-02-12T10:01:00Z",
                "actor": "claude-analyst",
            },
            {
                "event_type": "critique_submitted",
                "timestamp": "2026-02-12T10:02:00Z",
                "actor": "gemini-challenger",
            },
            {
                "event_type": "revision_submitted",
                "timestamp": "2026-02-12T10:03:00Z",
                "actor": "mistral-auditor",
            },
            {
                "event_type": "vote_cast",
                "timestamp": "2026-02-12T10:04:00Z",
                "actor": "claude-analyst",
            },
            {
                "event_type": "vote_cast",
                "timestamp": "2026-02-12T10:04:01Z",
                "actor": "gpt4-ethics",
            },
            {
                "event_type": "vote_cast",
                "timestamp": "2026-02-12T10:04:02Z",
                "actor": "gemini-challenger",
            },
            {
                "event_type": "vote_cast",
                "timestamp": "2026-02-12T10:04:03Z",
                "actor": "mistral-auditor",
            },
            {
                "event_type": "human_approval",
                "timestamp": "2026-02-12T10:10:00Z",
                "actor": "hr-director@acme.com",
            },
            {
                "event_type": "receipt_generated",
                "timestamp": "2026-02-12T10:10:05Z",
                "actor": "system",
            },
        ],
        "config_used": {
            "protocol": "adversarial",
            "rounds": 2,
            "require_approval": True,
            "human_in_loop": True,
            "approver": "hr-director@acme.com",
        },
        "artifact_hash": hashlib.sha256(b"hr-recruitment-receipt-v1").hexdigest(),
        "signature": "ed25519:mock_signature_for_demo_purposes",
    }


def _credit_scoring_receipt() -> dict[str, Any]:
    """Receipt for a high-risk credit scoring system (Annex III Cat. 5)."""
    return {
        "receipt_id": "RCP-FIN-2026-0087",
        "input_summary": (
            "Evaluate the credit scoring model for consumer loan applications. "
            "The creditworthiness assessment uses income, employment history, "
            "and repayment behavior to determine loan eligibility."
        ),
        "verdict": "reject",
        "verdict_reasoning": (
            "The credit scoring model shows significant disparate impact on "
            "protected demographic groups. The loan decision algorithm achieves "
            "82% accuracy overall but drops to 71% for underrepresented groups. "
            "The model lacks sufficient explainability for individual credit "
            "decisions as required by the EU AI Act."
        ),
        "confidence": 0.65,
        "robustness_score": 0.45,
        "risk_summary": {
            "total": 7,
            "critical": 2,
            "high": 3,
            "medium": 1,
            "low": 1,
        },
        "consensus_proof": {
            "method": "unanimous",
            "threshold": 1.0,
            "supporting_agents": ["claude-analyst", "gpt4-ethics"],
            "dissenting_agents": ["mistral-auditor", "gemini-challenger"],
            "agreement_ratio": 0.50,
        },
        "dissenting_views": [
            {
                "agent": "mistral-auditor",
                "view": "Model accuracy is acceptable but fairness metrics need remediation.",
            },
            {
                "agent": "gemini-challenger",
                "view": "Reject is too conservative. A conditional approval with bias monitoring would suffice.",
            },
        ],
        "provenance_chain": [
            {
                "event_type": "debate_started",
                "timestamp": "2026-02-12T14:00:00Z",
                "actor": "system",
            },
            {
                "event_type": "proposal_submitted",
                "timestamp": "2026-02-12T14:02:00Z",
                "actor": "claude-analyst",
            },
            {
                "event_type": "receipt_generated",
                "timestamp": "2026-02-12T14:15:00Z",
                "actor": "system",
            },
        ],
        "config_used": {"protocol": "adversarial", "rounds": 3},
        "artifact_hash": hashlib.sha256(b"credit-scoring-receipt-v1").hexdigest(),
        "signature": "",
    }


def _chatbot_receipt() -> dict[str, Any]:
    """Receipt for a limited-risk customer service chatbot (Art. 50)."""
    return {
        "receipt_id": "RCP-SVC-2026-0203",
        "input_summary": (
            "Deploy a customer service chatbot for handling product inquiries "
            "and order status requests. The virtual assistant uses generated "
            "content to provide natural language responses."
        ),
        "verdict": "approve",
        "verdict_reasoning": (
            "The chatbot handles low-stakes informational queries only. No "
            "access to personal data beyond order IDs. Responses are clearly "
            "labeled as AI-generated per Article 50 transparency requirements."
        ),
        "confidence": 0.92,
        "robustness_score": 0.85,
        "risk_summary": {"total": 1, "critical": 0, "high": 0, "medium": 0, "low": 1},
        "consensus_proof": {
            "method": "weighted_majority",
            "threshold": 0.66,
            "supporting_agents": ["claude-analyst", "mistral-auditor", "gpt4-ethics"],
            "dissenting_agents": [],
            "agreement_ratio": 1.0,
        },
        "dissenting_views": [],
        "provenance_chain": [
            {
                "event_type": "debate_started",
                "timestamp": "2026-02-12T09:00:00Z",
                "actor": "system",
            },
            {
                "event_type": "proposal_submitted",
                "timestamp": "2026-02-12T09:01:00Z",
                "actor": "claude-analyst",
            },
            {
                "event_type": "vote_cast",
                "timestamp": "2026-02-12T09:05:00Z",
                "actor": "claude-analyst",
            },
            {
                "event_type": "vote_cast",
                "timestamp": "2026-02-12T09:05:01Z",
                "actor": "mistral-auditor",
            },
            {
                "event_type": "vote_cast",
                "timestamp": "2026-02-12T09:05:02Z",
                "actor": "gpt4-ethics",
            },
            {
                "event_type": "human_approval",
                "timestamp": "2026-02-12T09:10:00Z",
                "actor": "product-manager@acme.com",
            },
            {
                "event_type": "receipt_generated",
                "timestamp": "2026-02-12T09:10:05Z",
                "actor": "system",
            },
        ],
        "config_used": {
            "protocol": "quick_consensus",
            "rounds": 1,
            "human_approval": True,
        },
        "artifact_hash": hashlib.sha256(b"chatbot-receipt-v1").hexdigest(),
        "signature": "ed25519:mock_signature_chatbot",
    }


# =============================================================================
# Use Case Classification Scenarios
# =============================================================================

CLASSIFICATION_SCENARIOS = [
    {
        "name": "HR Resume Screening",
        "description": "AI-powered CV screening and recruitment decision system for filtering job applications",
        "expected_level": "HIGH",
        "expected_category": "Employment and worker management",
    },
    {
        "name": "Credit Scoring Engine",
        "description": "Creditworthiness assessment for consumer loan decisions using income and repayment data",
        "expected_level": "HIGH",
        "expected_category": "Access to essential services",
    },
    {
        "name": "Customer Service Chatbot",
        "description": "Virtual assistant chatbot providing generated content for product inquiries",
        "expected_level": "LIMITED",
        "expected_category": None,
    },
    {
        "name": "Social Credit System",
        "description": "Social scoring system that evaluates citizens based on their social behavior and assigns penalties",
        "expected_level": "UNACCEPTABLE",
        "expected_category": None,
    },
    {
        "name": "Weather Forecasting",
        "description": "Machine learning model for 7-day weather prediction using satellite imagery",
        "expected_level": "MINIMAL",
        "expected_category": None,
    },
    {
        "name": "Predictive Policing",
        "description": "Crime prediction system for law enforcement using historical crime analytics and suspect identification",
        "expected_level": "HIGH",
        "expected_category": "Law enforcement",
    },
    {
        "name": "Student Exam Proctoring",
        "description": "AI exam proctoring and student assessment system for educational institutions",
        "expected_level": "HIGH",
        "expected_category": "Education and vocational training",
    },
    {
        "name": "Border Surveillance",
        "description": "AI-powered border control and migration risk assessment for asylum applications",
        "expected_level": "HIGH",
        "expected_category": "Migration, asylum and border control",
    },
]


# =============================================================================
# Demo Runners
# =============================================================================


def _print_header(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def _print_section(title: str) -> None:
    print()
    print(f"--- {title} ---")
    print()


def run_classification_demo() -> None:
    """Demonstrate EU AI Act risk classification across use case types."""
    _print_header("EU AI Act Risk Classification (Article 6 + Annex III)")

    classifier = RiskClassifier()
    results: list[tuple[str, str, str, bool]] = []

    for scenario in CLASSIFICATION_SCENARIOS:
        result = classifier.classify(scenario["description"])
        level_str = result.risk_level.value.upper()
        expected = scenario["expected_level"]
        match = level_str == expected

        results.append((scenario["name"], level_str, expected, match))

        indicator = {
            "UNACCEPTABLE": "[PROHIBITED]",
            "HIGH": "[HIGH RISK] ",
            "LIMITED": "[LIMITED]   ",
            "MINIMAL": "[MINIMAL]   ",
        }.get(level_str, level_str)

        status = "PASS" if match else "FAIL"
        print(f"  {indicator}  {scenario['name']:<30s}  [{status}]")

        if result.annex_iii_category:
            print(
                f"              Annex III Cat. {result.annex_iii_number}: {result.annex_iii_category}"
            )
        if result.matched_keywords:
            print(f"              Keywords: {', '.join(result.matched_keywords[:3])}")
        if result.obligations:
            print(f"              Obligations: {len(result.obligations)} article requirements")
        print()

    passed = sum(1 for _, _, _, m in results if m)
    total = len(results)
    print(f"Classification accuracy: {passed}/{total} scenarios matched expected risk level")
    if passed == total:
        print("All classifications correct.")


def run_conformity_demo() -> None:
    """Demonstrate conformity report generation from decision receipts."""
    _print_header("EU AI Act Conformity Assessment Reports")

    generator = ConformityReportGenerator()
    receipts = [
        ("HR Recruitment Screening", _hr_recruitment_receipt()),
        ("Credit Scoring Engine", _credit_scoring_receipt()),
        ("Customer Service Chatbot", _chatbot_receipt()),
    ]

    for name, receipt in receipts:
        _print_section(f"Receipt: {name} ({receipt['receipt_id']})")

        report = generator.generate(receipt)

        # Summary
        print(f"  Risk Level:      {report.risk_classification.risk_level.value.upper()}")
        if report.risk_classification.annex_iii_category:
            print(
                f"  Annex III:       Cat. {report.risk_classification.annex_iii_number} "
                f"({report.risk_classification.annex_iii_category})"
            )
        print(f"  Overall Status:  {report.overall_status.upper()}")
        print(f"  Report ID:       {report.report_id}")
        print(f"  Integrity Hash:  {report.integrity_hash[:16]}...")
        print()

        # Article compliance table
        print(f"  {'Article':<12s} {'Requirement':<45s} {'Status':<10s}")
        print(f"  {'-' * 12} {'-' * 45} {'-' * 10}")
        for mapping in report.article_mappings:
            status_display = {
                "satisfied": "PASS",
                "partial": "PARTIAL",
                "not_satisfied": "FAIL",
                "not_applicable": "N/A",
            }.get(mapping.status, mapping.status)
            req_short = (
                mapping.requirement[:43] + ".."
                if len(mapping.requirement) > 45
                else mapping.requirement
            )
            print(f"  {mapping.article:<12s} {req_short:<45s} {status_display:<10s}")
        print()

        # Recommendations
        if report.recommendations:
            print(f"  Recommendations ({len(report.recommendations)}):")
            for rec in report.recommendations:
                wrapped = textwrap.fill(
                    rec, width=64, initial_indent="    - ", subsequent_indent="      "
                )
                print(wrapped)
            print()

        # Dissent tracking (unique to Aragora)
        dissenting = receipt.get("dissenting_views", [])
        if dissenting:
            print(f"  Dissenting Views ({len(dissenting)}):")
            for dv in dissenting:
                wrapped = textwrap.fill(
                    dv["view"],
                    width=64,
                    initial_indent=f"    [{dv['agent']}] ",
                    subsequent_indent="      ",
                )
                print(wrapped)
            print()


def run_artifact_export_demo() -> None:
    """Demonstrate export of audit-ready compliance artifacts."""
    _print_header("Compliance Artifact Export")

    generator = ConformityReportGenerator()
    receipt = _hr_recruitment_receipt()
    report = generator.generate(receipt)

    # JSON export
    _print_section("JSON Artifact (machine-readable)")
    json_output = report.to_json(indent=2)
    # Show first 40 lines
    lines = json_output.split("\n")
    for line in lines[:40]:
        print(f"  {line}")
    if len(lines) > 40:
        print(f"  ... ({len(lines) - 40} more lines)")
    print()
    print(f"  Total JSON size: {len(json_output)} bytes")

    # Markdown export
    _print_section("Markdown Artifact (human-readable)")
    md_output = report.to_markdown()
    md_lines = md_output.split("\n")
    for line in md_lines[:30]:
        print(f"  {line}")
    if len(md_lines) > 30:
        print(f"  ... ({len(md_lines) - 30} more lines)")
    print()
    print(f"  Total Markdown size: {len(md_output)} bytes")

    # Deployer obligations summary
    _print_section("Deployer Log Retention (Article 26)")
    print("  Per Article 26 of the EU AI Act, deployers of high-risk AI")
    print("  systems must retain automatically generated logs for a minimum")
    print("  of six months. Aragora's audit infrastructure satisfies this")
    print("  through:")
    print()
    print("    1. Immutable provenance chain in every decision receipt")
    print("    2. SHA-256 integrity hashing prevents log tampering")
    print("    3. Configurable retention policies (default: 180 days)")
    print("    4. Ed25519 cryptographic signatures for non-repudiation")
    print()
    chain_len = len(receipt.get("provenance_chain", []))
    print(f"  This receipt contains {chain_len} provenance events covering")
    print(f"  the full decision lifecycle from debate initiation through")
    print(f"  human approval to receipt generation.")


def run_artifact_generation_demo() -> None:
    """Demonstrate generation of dedicated Art. 12/13/14 compliance artifacts."""
    _print_header("EU AI Act Artifact Generation (Articles 12, 13, 14)")

    generator = ComplianceArtifactGenerator(
        provider_name="Aragora Inc.",
        provider_contact="compliance@aragora.ai",
        eu_representative="Aragora EU GmbH, Berlin, Germany",
    )

    receipt = _hr_recruitment_receipt()
    bundle = generator.generate(receipt)

    # Bundle overview
    print(f"  Bundle ID:         {bundle.bundle_id}")
    print(f"  Receipt ID:        {bundle.receipt_id}")
    print(f"  Risk Level:        {bundle.risk_classification.risk_level.value.upper()}")
    print(f"  Conformity Status: {bundle.conformity_report.overall_status.upper()}")
    print(f"  Integrity Hash:    {bundle.integrity_hash[:24]}...")

    # Article 12: Record-Keeping
    _print_section("Article 12: Record-Keeping")
    art12 = bundle.article_12
    print(f"  Events logged:        {len(art12.event_log)}")
    print(f"  Reference databases:  {len(art12.reference_databases)}")
    print(f"  Input hash:           {art12.input_record.get('input_hash', 'N/A')[:24]}...")
    print(f"  Retention:            {art12.retention_policy['minimum_months']} months minimum")
    print()

    print("  Event Log:")
    for evt in art12.event_log[:5]:
        print(f"    [{evt['event_id']}] {evt['event_type']:<24s} by {evt['actor']}")
    if len(art12.event_log) > 5:
        print(f"    ... and {len(art12.event_log) - 5} more events")
    print()

    tech = art12.technical_documentation
    sec1 = tech.get("annex_iv_sec1_general", {})
    print(f"  Annex IV Technical Documentation:")
    print(f"    System:  {sec1.get('system_name', 'N/A')} v{sec1.get('version', 'N/A')}")
    print(f"    Provider: {sec1.get('provider', 'N/A')}")

    # Article 13: Transparency
    _print_section("Article 13: Transparency")
    art13 = bundle.article_13
    print(f"  Provider:     {art13.provider_identity['name']}")
    print(f"  EU Rep:       {art13.provider_identity.get('eu_representative', 'N/A')}")
    print(f"  Known risks:  {len(art13.known_risks)}")
    for risk in art13.known_risks:
        print(f"    - {risk['risk']} ({risk['article_ref']})")
    print()

    interp = art13.output_interpretation
    print(f"  Output Interpretation:")
    print(f"    Confidence: {interp['confidence']:.0%} — {interp['confidence_interpretation']}")
    print(f"    Dissent: {interp['dissent_significance']}")
    print()

    human_ref = art13.human_oversight_reference
    print(
        f"  Human Oversight: {'Detected' if human_ref['human_approval_detected'] else 'Not detected'}"
    )

    # Article 14: Human Oversight
    _print_section("Article 14: Human Oversight")
    art14 = bundle.article_14
    om = art14.oversight_model
    print(f"  Model:              {om['primary']}")
    print(f"  Human approval:     {'Yes' if om['human_approval_detected'] else 'No'}")
    print()

    print(
        f"  14.4(a) Monitoring:     {len(art14.understanding_monitoring['monitoring_features'])} features"
    )
    print(
        f"  14.4(b) Bias safeguards: {len(art14.automation_bias_safeguards['mechanisms'])} mechanisms"
    )
    print(
        f"  14.4(c) Interpretation:  {len(art14.interpretation_features['explainability'])} features"
    )
    print(f"  14.4(d) Override:        {len(art14.override_capability['mechanisms'])} mechanisms")
    print(
        f"  14.4(e) Intervention:    {len(art14.intervention_capability['mechanisms'])} stop mechanisms"
    )

    # Export bundle
    _print_section("Artifact Bundle Export")
    json_output = bundle.to_json(indent=2)
    json_lines = json_output.split("\n")
    print(f"  Total JSON size: {len(json_output):,} bytes ({len(json_lines)} lines)")
    print()
    print("  JSON preview (first 20 lines):")
    for line in json_lines[:20]:
        print(f"    {line}")
    print(f"    ... ({len(json_lines) - 20} more lines)")

    print()
    print("  To generate artifacts for your own receipts:")
    print("    from aragora.compliance.eu_ai_act import ComplianceArtifactGenerator")
    print("    generator = ComplianceArtifactGenerator(provider_name='Your Org')")
    print("    bundle = generator.generate(receipt_dict)")
    print("    print(bundle.to_json())")


def run_full_demo() -> None:
    """Run the complete EU AI Act compliance demonstration."""
    print()
    print("  Aragora EU AI Act Compliance Engine")
    print("  Regulation (EU) 2024/1689 | Effective August 2, 2026")
    print("  -------------------------------------------------------")
    print("  Automated conformity assessment for AI decision receipts")
    print()

    run_classification_demo()
    run_conformity_demo()
    run_artifact_generation_demo()
    run_artifact_export_demo()

    _print_header("Demo Complete")
    print("  The EU AI Act compliance module maps Aragora decision receipts")
    print("  to regulatory requirements automatically. For high-risk systems,")
    print("  this includes:")
    print()
    print("    - Article 6:  Risk classification (Annex III category matching)")
    print("    - Article 9:  Risk management assessment")
    print("    - Article 12: Record-keeping and automatic logging (with artifacts)")
    print("    - Article 13: Transparency (deployer instructions, known risks)")
    print("    - Article 14: Human oversight (override, stop, bias safeguards)")
    print("    - Article 15: Accuracy, robustness, and cybersecurity")
    print("    - Article 26: Deployer log retention (6-month minimum)")
    print()
    print("  Artifacts are exportable as JSON (machine-readable) and Markdown")
    print("  (human-readable) for audit submission and regulatory review.")
    print()
    print("  See aragora/compliance/eu_ai_act.py for the full implementation.")
    print()


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aragora EU AI Act Compliance Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python examples/eu_ai_act_compliance.py --demo       Full demonstration
              python examples/eu_ai_act_compliance.py --classify    Risk classification only
              python examples/eu_ai_act_compliance.py --report      Conformity report only
              python examples/eu_ai_act_compliance.py --artifacts   Art. 12/13/14 artifacts
              python examples/eu_ai_act_compliance.py --export      Artifact export only
        """),
    )
    parser.add_argument("--demo", action="store_true", default=True, help="Run full demo (default)")
    parser.add_argument("--classify", action="store_true", help="Risk classification demo only")
    parser.add_argument("--report", action="store_true", help="Conformity report demo only")
    parser.add_argument(
        "--artifacts", action="store_true", help="Art. 12/13/14 artifact generation only"
    )
    parser.add_argument("--export", action="store_true", help="Artifact export demo only")

    args = parser.parse_args()

    if args.classify:
        run_classification_demo()
    elif args.report:
        run_conformity_demo()
    elif args.artifacts:
        run_artifact_generation_demo()
    elif args.export:
        run_artifact_export_demo()
    else:
        run_full_demo()

    return 0


if __name__ == "__main__":
    sys.exit(main())
