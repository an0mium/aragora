"""
Gauntlet CLI - Standalone entry point.

Run adversarial stress-testing from the command line:
    python -m aragora.gauntlet spec.md --profile thorough
    gauntlet spec.md --input-type architecture --persona gdpr
"""

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="gauntlet",
        description="""
Gauntlet - Adversarial Stress-Testing CLI

Run comprehensive adversarial validation on documents, specifications,
and code. Produces Decision Receipts for audit and compliance.

Combines:
- Red-team attacks (logical fallacies, edge cases, security)
- Capability probing (hallucination, sycophancy, consistency)
- Deep audit (multi-round intensive analysis)
- Formal verification (Z3/Lean proofs where applicable)
- Risk assessment (domain-specific hazards)

Examples:
    gauntlet spec.md --input-type spec
    gauntlet architecture.md --profile thorough --output receipt.html
    gauntlet policy.yaml --persona gdpr --format json
    gauntlet code.py --profile code --verify
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Path to input file (spec, architecture, policy, code)",
    )
    parser.add_argument(
        "--input-type",
        "-t",
        choices=["spec", "architecture", "policy", "code", "strategy", "contract"],
        default="spec",
        help="Type of input document (default: spec)",
    )
    parser.add_argument(
        "--agents",
        "-a",
        default="anthropic-api,openai-api",
        help="Comma-separated agents for stress-testing (default: anthropic-api,openai-api)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        choices=[
            "default",
            "quick",
            "thorough",
            "code",
            "policy",
            "gdpr",
            "hipaa",
            "ai_act",
            "security",
            "sox",
        ],
        default="default",
        help="Pre-configured test profile (default: default)",
    )
    parser.add_argument(
        "--persona",
        choices=[
            "gdpr",
            "hipaa",
            "ai_act",
            "security",
            "soc2",
            "sox",
            "pci_dss",
            "nist_csf",
        ],
        help="Regulatory persona for compliance-focused testing",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        help="Number of deep audit rounds (overrides profile)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Maximum duration in seconds (overrides profile)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for Decision Receipt",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "md", "html"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable formal verification (Z3/Lean)",
    )
    parser.add_argument(
        "--no-redteam",
        action="store_true",
        help="Disable red-team attacks",
    )
    parser.add_argument(
        "--no-probing",
        action="store_true",
        help="Disable capability probing",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable deep audit",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (aragora gauntlet)",
    )

    return parser.parse_args()


def _save_receipt(receipt, output_path: Path, format_ext: str) -> Path:
    """Save decision receipt in the specified format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    format_handlers = {
        "json": (lambda r: r.to_json(), ".json"),
        "md": (lambda r: r.to_markdown(), ".md"),
    }

    handler, suffix = format_handlers.get(format_ext, (lambda r: r.to_html(), ".html"))
    output_file = output_path.with_suffix(suffix)
    output_file.write_text(handler(receipt))
    return output_file


def main() -> int:
    """Main entry point for Gauntlet CLI."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import after arg parsing for faster --help
    from aragora.agents.base import create_agent
    from aragora.agents.spec import AgentSpec
    from aragora.gauntlet import (
        AI_ACT_GAUNTLET,
        CODE_REVIEW_GAUNTLET,
        GDPR_GAUNTLET,
        HIPAA_GAUNTLET,
        POLICY_GAUNTLET,
        QUICK_GAUNTLET,
        SECURITY_GAUNTLET,
        SOX_GAUNTLET,
        THOROUGH_GAUNTLET,
        DecisionReceipt,
        GauntletOrchestrator,
        GauntletProgress,
        InputType,
        OrchestratorConfig,
        get_compliance_gauntlet,
    )

    if not args.quiet:
        pass

    # Load input content
    input_path = Path(args.input)
    if not input_path.exists():
        return 1

    input_content = input_path.read_text()
    if not args.quiet:
        pass

    # Determine input type
    input_type_map = {
        "spec": InputType.SPEC,
        "architecture": InputType.ARCHITECTURE,
        "policy": InputType.POLICY,
        "code": InputType.CODE,
        "strategy": InputType.STRATEGY,
        "contract": InputType.CONTRACT,
    }
    input_type = input_type_map.get(args.input_type, InputType.SPEC)
    if not args.quiet:
        pass

    # Parse and create agents
    specs = AgentSpec.parse_list(args.agents)
    agents = []
    failed_agents = []

    for i, spec in enumerate(specs):
        role = spec.role
        if role is None:
            if i == 0:
                role = "proposer"
            elif i == len(specs) - 1 and len(specs) > 1:
                role = "synthesizer"
            else:
                role = "critic"

        try:
            agent = create_agent(
                model_type=spec.provider,  # type: ignore[arg-type]
                name=f"{spec.provider}_{role}",
                role=role,
            )
            agents.append(agent)
        except Exception as e:
            failed_agents.append((spec.provider, str(e)))
            if not args.quiet:
                pass

    if not agents:
        for agent_type, error in failed_agents:
            pass
        return 1

    if not args.quiet:
        pass

    # Profile configuration
    profile_configs = {
        "quick": (QUICK_GAUNTLET, None),
        "thorough": (THOROUGH_GAUNTLET, None),
        "code": (CODE_REVIEW_GAUNTLET, None),
        "policy": (POLICY_GAUNTLET, None),
        "gdpr": (GDPR_GAUNTLET, "gdpr"),
        "hipaa": (HIPAA_GAUNTLET, "hipaa"),
        "ai_act": (AI_ACT_GAUNTLET, "ai_act"),
        "security": (SECURITY_GAUNTLET, "security"),
        "sox": (SOX_GAUNTLET, "sox"),
    }

    # Select config profile
    persona = args.persona
    if persona:
        if not args.quiet:
            pass
        if args.profile in ("quick", "thorough"):
            base_config, _ = profile_configs[args.profile]
        else:
            base_config = get_compliance_gauntlet(persona)
    elif args.profile in profile_configs:
        base_config, profile_persona = profile_configs[args.profile]
        persona = profile_persona
    else:
        base_config = OrchestratorConfig()

    # Build config
    config = OrchestratorConfig(
        input_type=input_type,
        input_content=input_content,
        input_path=input_path,
        severity_threshold=base_config.severity_threshold,
        risk_threshold=base_config.risk_threshold,
        max_duration_seconds=args.timeout or base_config.max_duration_seconds,
        deep_audit_rounds=args.rounds or base_config.deep_audit_rounds,
        enable_redteam=not args.no_redteam,
        enable_probing=not args.no_probing,
        enable_deep_audit=not args.no_audit,
        enable_verification=args.verify,
        persona=persona,
    )

    if not args.quiet:
        pass

    # Progress callback
    last_phase: list[Optional[str]] = [None]

    def on_progress(progress: GauntletProgress) -> None:
        """Display progress updates."""
        if args.quiet:
            return

        bar_width = 40
        filled = int(bar_width * progress.percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        line = f"\r[{bar}] {progress.percent:5.1f}% | {progress.phase}"
        if progress.findings_so_far > 0:
            line += f" | {progress.findings_so_far} findings"

        sys.stderr.write(line + " " * 10)
        sys.stderr.flush()

        if progress.phase != last_phase[0] and last_phase[0] is not None:
            sys.stderr.write("\n")
            sys.stderr.flush()
        last_phase[0] = progress.phase

        if progress.percent >= 100:
            sys.stderr.write("\n")
            sys.stderr.flush()

    # Run gauntlet
    orchestrator = GauntletOrchestrator(agents, on_progress=on_progress)
    result = asyncio.run(orchestrator.run(config))

    # Print summary

    # Generate and save receipt
    if args.output:
        output_path = Path(args.output)
        input_hash = hashlib.sha256(config.input_content.encode()).hexdigest()
        receipt = DecisionReceipt.from_mode_result(result, input_hash=input_hash)

        format_ext = args.format or output_path.suffix.lstrip(".")
        if format_ext not in ("json", "md", "html"):
            format_ext = "html"

        _save_receipt(receipt, output_path, format_ext)

    # Exit code based on verdict
    if result.verdict.value == "rejected":
        return 1
    elif result.verdict.value == "needs_review":
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
