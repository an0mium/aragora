"""
Compliance CLI commands.

Provides CLI access to EU AI Act compliance tooling:
- aragora compliance audit <receipt_file>  -- Generate conformity report
- aragora compliance classify <description> -- Classify use case by risk level
- aragora compliance eu-ai-act generate    -- Generate full artifact bundle
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def add_compliance_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the compliance subcommand and its sub-subcommands."""
    parser = subparsers.add_parser(
        "compliance",
        help="EU AI Act compliance tools",
        description=(
            "Generate conformity reports, classify AI use cases, and produce "
            "audit-ready compliance artifact bundles per the EU AI Act "
            "(Regulation 2024/1689). Enforcement deadline: August 2, 2026."
        ),
    )
    parser.set_defaults(func=cmd_compliance)
    sub = parser.add_subparsers(dest="compliance_command")

    # -- aragora compliance audit <receipt_file> --
    audit_p = sub.add_parser(
        "audit",
        help="Generate EU AI Act conformity report from a receipt JSON file",
    )
    audit_p.add_argument(
        "receipt_file",
        help="Path to a DecisionReceipt JSON file",
    )
    audit_p.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        dest="output_format",
        help="Output format (default: markdown)",
    )
    audit_p.add_argument(
        "--output", "-o",
        help="Write report to file instead of stdout",
    )

    # -- aragora compliance classify <description> --
    classify_p = sub.add_parser(
        "classify",
        help="Classify a use case by EU AI Act risk level",
    )
    classify_p.add_argument(
        "description",
        nargs="+",
        help="Free-text description of the AI use case",
    )

    # -- aragora compliance eu-ai-act generate --
    eu_p = sub.add_parser(
        "eu-ai-act",
        help="Generate EU AI Act compliance artifact bundles (Articles 12, 13, 14)",
        description=(
            "Generate complete EU AI Act compliance artifact bundles from a "
            "DecisionReceipt JSON file. Produces dedicated Article 12 "
            "(Record-Keeping), Article 13 (Transparency), and Article 14 "
            "(Human Oversight) artifacts with SHA-256 integrity hash."
        ),
    )
    eu_sub = eu_p.add_subparsers(dest="eu_ai_act_command")
    generate_p = eu_sub.add_parser(
        "generate",
        help="Generate a compliance artifact bundle from a receipt",
    )
    generate_p.add_argument(
        "receipt_file",
        nargs="?",
        help=(
            "Path to a DecisionReceipt JSON file. If omitted, generates a "
            "demonstration bundle with synthetic data."
        ),
    )
    generate_p.add_argument(
        "--output", "-o",
        default="./compliance-bundle",
        help="Output directory for the artifact bundle (default: ./compliance-bundle/)",
    )
    generate_p.add_argument(
        "--provider-name",
        default="",
        help="Organization name for the provider identity field",
    )
    generate_p.add_argument(
        "--provider-contact",
        default="",
        help="Contact email for compliance inquiries",
    )
    generate_p.add_argument(
        "--eu-representative",
        default="",
        help="EU authorized representative (required for non-EU providers)",
    )
    generate_p.add_argument(
        "--system-name",
        default="",
        help="Name of the AI system under assessment",
    )
    generate_p.add_argument(
        "--system-version",
        default="",
        help="Version of the AI system under assessment",
    )
    generate_p.add_argument(
        "--format",
        choices=["json", "all"],
        default="all",
        dest="output_format",
        help="Output format: 'json' for bundle JSON only, 'all' for JSON + per-article files (default: all)",
    )


def cmd_compliance(args: argparse.Namespace) -> None:
    """Dispatch compliance sub-commands."""
    command = getattr(args, "compliance_command", None)
    if command == "audit":
        _cmd_audit(args)
    elif command == "classify":
        _cmd_classify(args)
    elif command == "eu-ai-act":
        eu_command = getattr(args, "eu_ai_act_command", None)
        if eu_command == "generate":
            _cmd_eu_ai_act_generate(args)
        else:
            print("Usage: aragora compliance eu-ai-act generate [receipt_file] --output ./compliance-bundle/")
            print()
            print("Generate EU AI Act compliance artifact bundles (Articles 12, 13, 14).")
            print("Omit receipt_file to generate a demonstration bundle with synthetic data.")
            sys.exit(1)
    else:
        print("Usage: aragora compliance {audit,classify,eu-ai-act}")
        print()
        print("  audit        Generate EU AI Act conformity report from a receipt")
        print("  classify     Classify a use case by EU AI Act risk level")
        print("  eu-ai-act    Generate compliance artifact bundles (Articles 12/13/14)")
        sys.exit(1)


def _cmd_audit(args: argparse.Namespace) -> None:
    """Generate EU AI Act conformity report from a receipt file."""
    from aragora.compliance.eu_ai_act import ConformityReportGenerator

    # Load receipt JSON
    try:
        with open(args.receipt_file) as f:
            receipt_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.receipt_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    generator = ConformityReportGenerator()
    report = generator.generate(receipt_dict)

    if args.output_format == "json":
        output = report.to_json()
    else:
        output = report.to_markdown()

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


def _cmd_classify(args: argparse.Namespace) -> None:
    """Classify a use case by EU AI Act risk level."""
    from aragora.compliance.eu_ai_act import RiskClassifier

    description = " ".join(args.description)
    classifier = RiskClassifier()
    result = classifier.classify(description)

    # Color-coded output
    level_colors = {
        "unacceptable": "\033[91m",  # red
        "high": "\033[93m",          # yellow
        "limited": "\033[96m",       # cyan
        "minimal": "\033[92m",       # green
    }
    reset = "\033[0m"
    color = level_colors.get(result.risk_level.value, "")

    print(f"\nRisk Level: {color}{result.risk_level.value.upper()}{reset}")
    print(f"Rationale:  {result.rationale}")

    if result.annex_iii_category:
        print(f"Annex III:  {result.annex_iii_number}. {result.annex_iii_category}")

    if result.matched_keywords:
        print(f"Keywords:   {', '.join(result.matched_keywords)}")

    if result.applicable_articles:
        print("\nApplicable Articles:")
        for art in result.applicable_articles:
            print(f"  - {art}")

    if result.obligations:
        print("\nObligations:")
        for obl in result.obligations:
            print(f"  - {obl}")

    print()


def _cmd_eu_ai_act_generate(args: argparse.Namespace) -> None:
    """Generate EU AI Act compliance artifact bundle."""
    from aragora.compliance.eu_ai_act import (
        ComplianceArtifactGenerator,
        RiskClassifier,
    )

    # Load or create receipt
    if args.receipt_file:
        try:
            with open(args.receipt_file) as f:
                receipt_dict = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found: {args.receipt_file}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Generate demonstration bundle with synthetic receipt
        receipt_dict = _synthetic_receipt()
        print("No receipt file provided. Using synthetic demonstration data.")
        print()

    # Build generator with user-provided or default settings
    generator = ComplianceArtifactGenerator(
        provider_name=args.provider_name or "Your Organization",
        provider_contact=args.provider_contact or "compliance@your-org.example.com",
        eu_representative=args.eu_representative or "",
        system_name=args.system_name or "AI Decision System",
        system_version=args.system_version or "1.0.0",
    )

    bundle = generator.generate(receipt_dict)

    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Write the full bundle JSON
    bundle_path = os.path.join(output_dir, "compliance_bundle.json")
    with open(bundle_path, "w") as f:
        f.write(bundle.to_json(indent=2))

    if args.output_format == "all":
        # Write individual article artifacts
        art12_path = os.path.join(output_dir, "article_12_record_keeping.json")
        with open(art12_path, "w") as f:
            json.dump(bundle.article_12.to_dict(), f, indent=2)

        art13_path = os.path.join(output_dir, "article_13_transparency.json")
        with open(art13_path, "w") as f:
            json.dump(bundle.article_13.to_dict(), f, indent=2)

        art14_path = os.path.join(output_dir, "article_14_human_oversight.json")
        with open(art14_path, "w") as f:
            json.dump(bundle.article_14.to_dict(), f, indent=2)

        # Write conformity report as markdown
        report_md_path = os.path.join(output_dir, "conformity_report.md")
        with open(report_md_path, "w") as f:
            f.write(bundle.conformity_report.to_markdown())

        # Write conformity report as JSON
        report_json_path = os.path.join(output_dir, "conformity_report.json")
        with open(report_json_path, "w") as f:
            f.write(bundle.conformity_report.to_json(indent=2))

    # Print summary to stdout
    classification = bundle.risk_classification
    print("EU AI Act Compliance Artifact Bundle Generated")
    print(f"{'=' * 50}")
    print(f"Bundle ID:       {bundle.bundle_id}")
    print(f"Receipt ID:      {bundle.receipt_id}")
    print(f"Risk Level:      {classification.risk_level.value.upper()}")
    if classification.annex_iii_category:
        print(f"Annex III:       Cat. {classification.annex_iii_number} ({classification.annex_iii_category})")
    print(f"Conformity:      {bundle.conformity_report.overall_status.upper()}")
    print(f"Integrity Hash:  {bundle.integrity_hash}")
    print("Deadline:        August 2, 2026")
    print()

    # List generated files
    print(f"Generated files in {output_dir}/:")
    print("  compliance_bundle.json          Full artifact bundle")
    if args.output_format == "all":
        print("  article_12_record_keeping.json  Art. 12 event log, tech docs, retention policy")
        print("  article_13_transparency.json    Art. 13 provider identity, risks, interpretation")
        print("  article_14_human_oversight.json  Art. 14 oversight model, override, stop mechanisms")
        print("  conformity_report.md            Human-readable conformity assessment")
        print("  conformity_report.json          Machine-readable conformity assessment")
    print()

    # Article compliance summary
    print("Article Compliance Summary:")
    for mapping in bundle.conformity_report.article_mappings:
        status_display = {
            "satisfied": "PASS",
            "partial": "PARTIAL",
            "not_satisfied": "FAIL",
            "not_applicable": "N/A",
        }.get(mapping.status, mapping.status)
        print(f"  {mapping.article:<12s} {mapping.requirement[:50]:<52s} [{status_display}]")

    if bundle.conformity_report.recommendations:
        print()
        print("Recommendations:")
        for i, rec in enumerate(bundle.conformity_report.recommendations, 1):
            print(f"  {i}. {rec}")


def _synthetic_receipt() -> dict:
    """Generate a synthetic decision receipt for demonstration purposes."""
    import hashlib
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    return {
        "receipt_id": "DEMO-RCP-001",
        "gauntlet_id": "demo-gauntlet-001",
        "timestamp": now,
        "input_summary": (
            "Evaluate AI-powered recruitment and CV screening system for "
            "automated candidate filtering in hiring decisions."
        ),
        "input_hash": hashlib.sha256(b"demo-input").hexdigest(),
        "risk_summary": {
            "total": 4,
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 1,
        },
        "attacks_attempted": 8,
        "attacks_successful": 1,
        "probes_run": 12,
        "vulnerabilities_found": 2,
        "verdict": "CONDITIONAL",
        "confidence": 0.78,
        "robustness_score": 0.72,
        "verdict_reasoning": (
            "The recruitment screening system meets accuracy thresholds but "
            "requires additional bias auditing before full deployment. "
            "Demographic parity gaps detected in the gender dimension."
        ),
        "dissenting_views": [
            "Agent-Challenger: demographic parity gap exceeds 5% threshold for employment AI"
        ],
        "consensus_proof": {
            "reached": True,
            "confidence": 0.78,
            "supporting_agents": ["claude-analyst", "mistral-auditor", "gpt4-ethics"],
            "dissenting_agents": ["gemini-challenger"],
            "method": "weighted_majority",
            "agreement_ratio": 0.75,
            "evidence_hash": hashlib.sha256(b"demo-evidence").hexdigest(),
        },
        "provenance_chain": [
            {"event_type": "debate_started", "timestamp": now, "actor": "system"},
            {"event_type": "proposal_submitted", "timestamp": now, "actor": "claude-analyst"},
            {"event_type": "critique_submitted", "timestamp": now, "actor": "gemini-challenger"},
            {"event_type": "revision_submitted", "timestamp": now, "actor": "mistral-auditor"},
            {"event_type": "vote_cast", "timestamp": now, "actor": "claude-analyst"},
            {"event_type": "vote_cast", "timestamp": now, "actor": "gpt4-ethics"},
            {"event_type": "vote_cast", "timestamp": now, "actor": "gemini-challenger"},
            {"event_type": "vote_cast", "timestamp": now, "actor": "mistral-auditor"},
            {"event_type": "human_approval", "timestamp": now, "actor": "compliance-officer@org.example.com"},
            {"event_type": "receipt_generated", "timestamp": now, "actor": "system"},
        ],
        "schema_version": "1.0",
        "artifact_hash": hashlib.sha256(b"demo-artifact").hexdigest(),
        "signature": "ed25519:demo_signature",
        "config_used": {
            "protocol": "adversarial",
            "rounds": 3,
            "require_approval": True,
            "human_in_loop": True,
            "approver": "compliance-officer@org.example.com",
        },
    }
