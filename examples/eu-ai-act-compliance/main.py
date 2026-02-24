#!/usr/bin/env python3
"""
EU AI Act Compliance Demo

Demonstrates how Aragora generates Article 12/13/14 compliance artifact bundles
from decision receipts. These artifacts are audit-ready documents for conformity
assessment under the EU AI Act (Regulation (EU) 2024/1689, effective Aug 2, 2026).

Usage:
    python examples/eu-ai-act-compliance/main.py
    python examples/eu-ai-act-compliance/main.py --use-case "credit scoring for loan applications"
    python examples/eu-ai-act-compliance/main.py --output bundle.json

No API keys required -- this demo uses synthetic data to show the artifact format.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

# Aragora imports
from aragora.compliance.eu_ai_act import (
    RiskClassifier,
    ComplianceArtifactGenerator,
)


def create_synthetic_receipt(use_case: str) -> dict:
    """Create a synthetic decision receipt for demo purposes."""
    return {
        "id": "receipt-demo-001",
        "task": use_case,
        "agents": [
            {"name": "claude", "provider": "anthropic", "role": "proposer"},
            {"name": "gpt-4", "provider": "openai", "role": "critic"},
            {"name": "gemini", "provider": "google", "role": "synthesizer"},
        ],
        "rounds": 3,
        "consensus": {
            "reached": True,
            "method": "majority",
            "confidence": 0.87,
        },
        "final_answer": (
            f"Multi-agent analysis of: {use_case}. "
            "Three independent models evaluated the proposal across security, "
            "fairness, and regulatory dimensions. Consensus reached with 87% "
            "confidence after 3 rounds of adversarial debate."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dissenting_opinions": [
            {
                "agent": "gpt-4",
                "position": "Additional fairness testing recommended for protected classes.",
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate EU AI Act compliance artifacts from a decision receipt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --use-case "resume screening for hiring decisions"
  python main.py --use-case "medical image triage" --output artifacts.json
        """,
    )
    parser.add_argument(
        "--use-case",
        default="credit scoring system for consumer loan applications",
        help="AI use case description for risk classification",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write full artifact bundle to JSON file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  EU AI Act Compliance Artifact Generator")
    print("  Aragora Decision Integrity Platform")
    print("=" * 60)

    # Step 1: Classify risk level
    print(f'\n1. Classifying risk level for: "{args.use_case}"')
    classifier = RiskClassifier()
    classification = classifier.classify(args.use_case)

    print(f"   Risk Level: {classification.risk_level.value.upper()}")
    if classification.annex_iii_category:
        print(
            f"   Annex III Category: {classification.annex_iii_number}. {classification.annex_iii_category}"
        )
    if classification.matched_keywords:
        print(f"   Matched Keywords: {', '.join(classification.matched_keywords)}")
    if classification.applicable_articles:
        print(f"   Applicable Articles: {', '.join(classification.applicable_articles)}")
    print(f"   Obligations: {len(classification.obligations)} requirements identified")

    # Step 2: Create synthetic receipt
    print("\n2. Generating synthetic decision receipt (3 agents, 3 rounds)...")
    receipt = create_synthetic_receipt(args.use_case)
    print(f"   Receipt ID: {receipt['id']}")
    print(f"   Agents: {', '.join(a['name'] for a in receipt['agents'])}")
    print(
        f"   Consensus: {'Reached' if receipt['consensus']['reached'] else 'Not reached'} "
        f"({receipt['consensus']['confidence']:.0%} confidence)"
    )

    # Step 3: Generate compliance artifacts
    print("\n3. Generating Article 12/13/14 compliance artifacts...")

    # Add use case context so the generator can classify risk internally
    receipt["task"] = args.use_case

    generator = ComplianceArtifactGenerator(
        provider_name="Demo Organization",
        provider_contact="compliance@demo.example.com",
        system_name=f"Aragora Decision Platform - {args.use_case}",
    )
    bundle = generator.generate(receipt)

    print(f"   Bundle ID: {bundle.bundle_id}")
    print(f"   Integrity Hash: {bundle.integrity_hash[:16]}...")

    # Article 12: Record-Keeping
    art12 = bundle.article_12
    print(f"\n   Article 12 (Record-Keeping):")
    print(f"     Event Log Entries: {len(art12.event_log)}")
    print(f"     Retention Policy: {art12.retention_policy}")

    # Article 13: Transparency
    art13 = bundle.article_13
    print(f"\n   Article 13 (Transparency):")
    print(f"     Provider: {art13.provider_identity}")
    print(f"     Known Risks: {len(art13.known_risks)}")
    print(f"     Output Interpretation: {'Provided' if art13.output_interpretation else 'N/A'}")

    # Article 14: Human Oversight
    art14 = bundle.article_14
    print(f"\n   Article 14 (Human Oversight):")
    print(f"     Oversight Model: {art14.oversight_model}")
    print(f"     Override Capability: {'Yes' if art14.override_capability else 'No'}")
    print(f"     Bias Safeguards: {len(art14.automation_bias_safeguards)}")

    # Step 4: Output
    if args.output:
        bundle_json = bundle.to_json(indent=2)
        with open(args.output, "w") as f:
            f.write(bundle_json)
        print(f"\n4. Full artifact bundle written to: {args.output}")
    else:
        print("\n4. Use --output bundle.json to save the full artifact bundle.")

    # Summary
    print("\n" + "=" * 60)
    if classification.risk_level.value == "high":
        print("  RESULT: HIGH-RISK AI SYSTEM")
        print("  This system requires conformity assessment under the EU AI Act.")
        print("  The generated artifacts satisfy Articles 12, 13, and 14.")
    elif classification.risk_level.value == "unacceptable":
        print("  RESULT: PROHIBITED AI PRACTICE")
        print("  This AI use case is banned under Article 5 of the EU AI Act.")
    elif classification.risk_level.value == "limited":
        print("  RESULT: LIMITED-RISK AI SYSTEM")
        print("  Transparency obligations apply (Article 50).")
    else:
        print("  RESULT: MINIMAL-RISK AI SYSTEM")
        print("  No specific obligations under the EU AI Act.")

    print(f"\n  EU AI Act effective date: August 2, 2026")
    print(f"  Artifacts generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
