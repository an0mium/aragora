"""
Gauntlet CLI command - adversarial stress-testing.

Extracted from main.py for modularity.
"""

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_agents(agents_str: str) -> list[tuple[str, str]]:
    """Parse agent string using unified AgentSpec.

    Supports both formats:
    - New pipe format: provider|model|persona|role (explicit role)
    - Legacy colon format: provider:persona (role defaults to 'proposer')

    Returns tuples of (provider, role). Note that the legacy colon format
    sets the persona, NOT the role. Use pipe format for explicit roles:
    - "claude:critic" -> ("claude", "proposer")  # 'critic' is persona
    - "claude|||critic" -> ("claude", "critic")  # 'critic' is role

    Args:
        agents_str: Comma-separated agent specs

    Returns:
        List of (provider, role) tuples
    """
    from aragora.agents.spec import AgentSpec

    specs = AgentSpec.parse_list(agents_str)
    return [(spec.provider, spec.role) for spec in specs]


# API key environment variable mapping for error messages
_API_KEY_ENV_VARS = {
    "anthropic-api": "ANTHROPIC_API_KEY",
    "openai-api": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "grok": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _format_agent_error(agent_type: str, error: str) -> str:
    """Format agent creation error with helpful guidance."""
    is_api_agent = "api" in agent_type.lower() or agent_type in _API_KEY_ENV_VARS
    if is_api_agent:
        env_var = _API_KEY_ENV_VARS.get(
            agent_type, f"{agent_type.upper().replace('-', '_')}_API_KEY"
        )
        return f"  - {agent_type}: {env_var} not set or invalid"
    return f"  - {agent_type}: {error}"


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


def cmd_gauntlet(args: argparse.Namespace) -> None:
    """Handle 'gauntlet' command - adversarial stress-testing."""
    from aragora.agents.base import create_agent
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

    print("\n" + "=" * 60)
    print("GAUNTLET - Adversarial Stress-Testing")
    print("=" * 60)

    # Load input content
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\nError: Input file not found: {input_path}")
        print("\nPlease check:")
        print("  - The file path is correct")
        print(f"  - The file exists: ls -la {input_path.parent}")
        print("\nUsage:")
        print("  aragora gauntlet path/to/spec.md --input-type spec")
        return

    input_content = input_path.read_text()
    print(f"\nInput: {input_path} ({len(input_content)} chars)")

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
    print(f"Type: {input_type.value}")

    # Create agents
    agent_specs = parse_agents(args.agents)
    agents = []
    failed_agents = []
    for i, (agent_type, role) in enumerate(agent_specs):
        role = role or f"agent_{i}"
        try:
            agent = create_agent(
                model_type=agent_type,  # type: ignore[arg-type]
                name=f"{agent_type}_{role}",
                role=role,
            )
            agents.append(agent)
        except Exception as e:
            failed_agents.append((agent_type, str(e)))
            print(f"Warning: Could not create agent {agent_type}: {e}")

    if not agents:
        print("\nError: No agents could be created.")
        print("\nFailed agents:")
        for agent_type, error in failed_agents:
            print(_format_agent_error(agent_type, error))
        print("\nTo fix:")
        print("  1. Set the required API key: export ANTHROPIC_API_KEY='your-key'")
        print("  2. Run 'aragora agents' to see available agents")
        print("  3. Run 'aragora doctor' to diagnose configuration issues")
        return

    print(f"Agents: {', '.join(a.name for a in agents)}")

    # Profile configuration: profile -> (config, persona)
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
    persona = getattr(args, "persona", None)
    if persona:
        print(f"Persona: {persona}")
        # Use persona-based compliance profile, but allow quick/thorough override
        if args.profile in ("quick", "thorough"):
            base_config, _ = profile_configs[args.profile]
        else:
            base_config = get_compliance_gauntlet(persona)
    elif args.profile in profile_configs:
        base_config, profile_persona = profile_configs[args.profile]
        persona = profile_persona  # Set persona from profile if defined
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

    print(f"Profile: {args.profile}")
    print(f"Max duration: {config.max_duration_seconds}s")
    print("\n" + "-" * 60)
    print("Running stress-test...")
    print("-" * 60 + "\n")

    # Progress callback for CLI display
    last_phase = [None]  # Use list for mutable closure

    def on_progress(progress: GauntletProgress) -> None:
        """Display progress updates in the CLI."""
        # Progress bar
        bar_width = 40
        filled = int(bar_width * progress.percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Clear line and print progress
        line = f"\r[{bar}] {progress.percent:5.1f}% | {progress.phase}"
        if progress.findings_so_far > 0:
            line += f" | {progress.findings_so_far} findings"

        # Print to stderr for live updates (stdout may be buffered)
        sys.stderr.write(line + " " * 10)  # Extra spaces to clear old text
        sys.stderr.flush()

        # Print phase change message on new line
        if progress.phase != last_phase[0] and last_phase[0] is not None:
            sys.stderr.write("\n")
            sys.stderr.flush()
        last_phase[0] = progress.phase

        # Print completion message
        if progress.percent >= 100:
            sys.stderr.write("\n")
            sys.stderr.flush()

    # Run gauntlet with progress callback
    orchestrator = GauntletOrchestrator(agents, on_progress=on_progress)
    result = asyncio.run(orchestrator.run(config))

    # Print summary
    print("\n" + result.summary())

    # Generate and save receipt
    if args.output:
        output_path = Path(args.output)
        input_hash = hashlib.sha256(config.input_content.encode()).hexdigest()
        receipt = DecisionReceipt.from_mode_result(result, input_hash=input_hash)

        # Determine format from extension or --format
        format_ext = args.format or output_path.suffix.lstrip(".")
        if format_ext not in ("json", "md", "html"):
            format_ext = "html"

        output_file = _save_receipt(receipt, output_path, format_ext)
        print(f"\nDecision Receipt saved: {output_file}")
        print(f"Artifact Hash: {receipt.artifact_hash[:16]}...")

    # Exit with non-zero if rejected
    if result.verdict.value == "rejected":
        print("\n[REJECTED] This input failed the stress-test.")
        sys.exit(1)
    elif result.verdict.value == "needs_review":
        print("\n[NEEDS REVIEW] This input requires human review.")
        sys.exit(2)


def create_gauntlet_parser(subparsers) -> argparse.ArgumentParser:
    """Create the gauntlet subcommand parser."""
    gauntlet_parser = subparsers.add_parser(
        "gauntlet",
        help="Adversarial stress-test a specification, architecture, or policy",
        description="""
Run comprehensive adversarial stress-testing on documents.

Gauntlet combines multiple validation techniques:
- Red-team attacks (logical fallacies, edge cases, security)
- Capability probing (hallucination, sycophancy, consistency)
- Deep audit (multi-round intensive analysis)
- Formal verification (Z3/Lean proofs where applicable)
- Risk assessment (domain-specific hazards)

Produces Decision Receipts - audit-ready artifacts for compliance.

Examples:
    aragora gauntlet spec.md --input-type spec
    aragora gauntlet architecture.md --input-type architecture --profile thorough
    aragora gauntlet policy.yaml --input-type policy --output receipt.html
    aragora gauntlet code.py --input-type code --profile code --verify
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gauntlet_parser.add_argument(
        "input",
        help="Path to input file (spec, architecture, policy, code)",
    )
    gauntlet_parser.add_argument(
        "--input-type",
        "-t",
        choices=["spec", "architecture", "policy", "code", "strategy", "contract"],
        default="spec",
        help="Type of input document (default: spec)",
    )
    gauntlet_parser.add_argument(
        "--agents",
        "-a",
        default="anthropic-api,openai-api",
        help="Comma-separated agents for stress-testing",
    )
    gauntlet_parser.add_argument(
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
    try:
        from aragora.gauntlet.personas import list_personas

        persona_choices = sorted(list_personas())
    except ImportError:
        # Gauntlet personas module not available - use defaults
        persona_choices = ["gdpr", "hipaa", "ai_act", "security", "sox"]
    except Exception as e:
        logger.debug(f"Could not load personas, using defaults: {e}")
        persona_choices = ["gdpr", "hipaa", "ai_act", "security", "sox"]
    gauntlet_parser.add_argument(
        "--persona",
        choices=persona_choices,
        help="Regulatory persona for compliance-focused stress testing",
    )
    gauntlet_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        help="Number of deep audit rounds (overrides profile)",
    )
    gauntlet_parser.add_argument(
        "--timeout",
        type=int,
        help="Maximum duration in seconds (overrides profile)",
    )
    gauntlet_parser.add_argument(
        "--output",
        "-o",
        help="Output path for Decision Receipt",
    )
    gauntlet_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "md", "html"],
        help="Output format (default: inferred from extension or html)",
    )
    gauntlet_parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable formal verification (Z3/Lean)",
    )
    gauntlet_parser.add_argument(
        "--no-redteam",
        action="store_true",
        help="Disable red-team attacks",
    )
    gauntlet_parser.add_argument(
        "--no-probing",
        action="store_true",
        help="Disable capability probing",
    )
    gauntlet_parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable deep audit",
    )
    gauntlet_parser.set_defaults(func=cmd_gauntlet)

    return gauntlet_parser


__all__ = ["cmd_gauntlet", "create_gauntlet_parser", "parse_agents"]
