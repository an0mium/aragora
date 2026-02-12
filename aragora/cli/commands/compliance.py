"""
Compliance CLI commands.

Provides CLI access to EU AI Act compliance tooling:
- aragora compliance audit <receipt_file>  -- Generate conformity report
- aragora compliance classify <description> -- Classify use case by risk level
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


def add_compliance_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the compliance subcommand and its sub-subcommands."""
    parser = subparsers.add_parser(
        "compliance",
        help="EU AI Act compliance tools",
        description="Generate conformity reports and classify AI use cases per the EU AI Act.",
    )
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


def cmd_compliance(args: argparse.Namespace) -> None:
    """Dispatch compliance sub-commands."""
    command = getattr(args, "compliance_command", None)
    if command == "audit":
        _cmd_audit(args)
    elif command == "classify":
        _cmd_classify(args)
    else:
        print("Usage: aragora compliance {audit,classify}")
        print("  audit    -- Generate EU AI Act conformity report from a receipt")
        print("  classify -- Classify a use case by EU AI Act risk level")
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
        print(f"\nApplicable Articles:")
        for art in result.applicable_articles:
            print(f"  - {art}")

    if result.obligations:
        print(f"\nObligations:")
        for obl in result.obligations:
            print(f"  - {obl}")

    print()
