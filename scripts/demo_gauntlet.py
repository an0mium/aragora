#!/usr/bin/env python3
"""
Aragora Gauntlet Demo - Adversarial Validation in Action.

This demo runs a complete Gauntlet stress-test on a sample policy document,
showing the full workflow from input to Decision Receipt.

Usage:
    python scripts/demo_gauntlet.py                     # Use sample policy
    python scripts/demo_gauntlet.py my_document.md      # Use your own document
    python scripts/demo_gauntlet.py --profile thorough  # Use thorough profile

Profiles:
    quick    - Fast stress-test (~2 min), good for demos
    thorough - Comprehensive analysis (~15 min), for production
    code     - Code review focused
    policy   - Policy/compliance focused
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.core import Agent, Critique, Message
from aragora.modes.gauntlet import (
    GauntletOrchestrator,
    GauntletConfig,
    GauntletResult,
    InputType,
    Verdict,
    QUICK_GAUNTLET,
    THOROUGH_GAUNTLET,
    CODE_REVIEW_GAUNTLET,
    POLICY_GAUNTLET,
)
from aragora.export.decision_receipt import DecisionReceiptGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ANSI colors for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def color(text: str, color_code: str) -> str:
    """Apply ANSI color to text."""
    return f"{color_code}{text}{Colors.ENDC}"


# Ultra-fast demo profile - skips deep audit for quick demonstrations
DEMO_GAUNTLET = GauntletConfig(
    deep_audit_rounds=0,  # Skip deep audit for fast demo
    parallel_attacks=2,
    parallel_probes=2,
    enable_redteam=True,
    enable_probing=True,
    enable_deep_audit=False,  # Skip for speed
    enable_verification=False,  # Skip for speed
    enable_risk_assessment=True,
    max_duration_seconds=30,
)

PROFILES = {
    "demo": DEMO_GAUNTLET,  # Ultra-fast for demonstrations
    "quick": QUICK_GAUNTLET,
    "thorough": THOROUGH_GAUNTLET,
    "code": CODE_REVIEW_GAUNTLET,
    "policy": POLICY_GAUNTLET,
}


# Sample policy document for demo
SAMPLE_POLICY = """
# API Rate Limiting Policy v2.1

## Purpose
This policy defines rate limiting rules for the Aragora public API to ensure fair usage and system stability.

## Rate Limits

### Standard Tier
- **Requests per minute**: 60
- **Requests per hour**: 1,000
- **Burst allowance**: 10 requests in 1 second
- **Reset behavior**: Rolling window

### Premium Tier
- **Requests per minute**: 300
- **Requests per hour**: 10,000
- **Burst allowance**: 50 requests in 1 second
- **Reset behavior**: Fixed window, resets at top of hour

## Enforcement

1. Clients exceeding limits receive HTTP 429 with `Retry-After` header
2. Repeated violations (>5 per hour) trigger temporary IP ban
3. Extreme abuse patterns may result in permanent API key revocation

## Exceptions

- Internal services are exempt from rate limiting
- Partner integrations may request custom limits via support ticket
- Emergency maintenance may temporarily reduce all limits

## Monitoring

Rate limit metrics are exposed at `/api/metrics/rate-limits` for authorized clients.
Dashboard available at admin.aragora.ai/rate-limits.

## Changelog

- v2.1: Added burst allowance, clarified reset behavior
- v2.0: Introduced Premium tier, increased standard limits
- v1.0: Initial policy

---

*Last updated: 2025-01-10*
*Policy owner: Platform Engineering*
"""


class DemoAgent(Agent):
    """Simple agent for demo that simulates responses."""

    def __init__(self, name: str, persona: str = ""):
        super().__init__(name=name, model="demo", role="analyst")
        self.persona = persona or f"I am {name}, an AI analyst."

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a simulated response for demo purposes."""
        # Simulate different agent behaviors
        if "attack" in prompt.lower() or "vulnerability" in prompt.lower():
            if "claude" in self.name.lower():
                return self._security_focused_response(prompt)
            elif "gpt" in self.name.lower():
                return self._edge_case_response(prompt)
            elif "gemini" in self.name.lower():
                return self._compliance_response(prompt)
            else:
                return self._general_analysis(prompt)
        return self._general_analysis(prompt)

    def _security_focused_response(self, prompt: str) -> str:
        return """
**Security Analysis**

I've identified several potential vulnerabilities:

1. **MEDIUM SEVERITY**: The policy mentions "internal services are exempt" but doesn't define authentication mechanism for internal services. This could allow attackers to spoof internal requests.

2. **HIGH SEVERITY**: The IP ban only triggers after "5 violations per hour" - this is generous enough to allow significant abuse before enforcement.

3. **LOW SEVERITY**: Rate limit metrics endpoint exposure could leak usage patterns to competitors.

Recommended mitigations:
- Implement mutual TLS for internal service authentication
- Consider progressive penalties instead of threshold-based bans
- Add authentication requirement for metrics endpoint
"""

    def _edge_case_response(self, prompt: str) -> str:
        return """
**Edge Case Analysis**

Examining boundary conditions and unexpected scenarios:

1. **MEDIUM SEVERITY**: What happens when a client is exactly at their limit and sends a burst? The interaction between per-minute and burst limits is undefined.

2. **LOW SEVERITY**: The "rolling window" vs "fixed window" difference could cause confusion. A client at 59 requests at 11:59 could make 60 more at 12:00 on fixed window.

3. **MEDIUM SEVERITY**: "Partner integrations may request custom limits" - no SLA defined for response time or approval criteria.
"""

    def _compliance_response(self, prompt: str) -> str:
        return """
**Compliance Review**

Checking against common regulatory frameworks:

1. **LOW SEVERITY**: No mention of geographic considerations - GDPR may require different handling for EU vs non-EU traffic.

2. **MEDIUM SEVERITY**: "Permanent API key revocation" without appeal process may violate fair service principles in some jurisdictions.

3. **INFO**: Policy changelog is good practice for audit trail, but should include approver signatures for compliance.
"""

    def _general_analysis(self, prompt: str) -> str:
        return """
**General Analysis**

The policy is well-structured overall, with clear tiers and enforcement rules.

Areas for improvement:
- Define escalation procedures for edge cases
- Add SLO commitments for rate limit consistency
- Consider adding rate limit headers to all responses for transparency
"""

    async def critique(
        self, proposal: str, task: str, context: list[Message] | None = None
    ) -> Critique:
        """Generate a simulated critique for demo purposes."""
        # Generate a response and parse it into a critique
        response = await self.generate(f"Critique this: {proposal[:500]}")

        return Critique(
            agent=self.name,
            target_agent="target",
            target_content=proposal[:200],
            issues=["Issue identified during analysis"],
            suggestions=["Consider adding more detail"],
            severity=0.5,
            reasoning=response[:500] if response else "Analysis complete.",
        )


def create_agents(use_real_apis: bool = False) -> list[Agent]:
    """Create agents for the demo."""
    if use_real_apis:
        # Try to import real API agents
        try:
            from aragora.agents.api_agents import (
                AnthropicAgent,
                OpenAIAgent,
                GeminiAgent,
            )

            agents = []
            if os.getenv("ANTHROPIC_API_KEY"):
                agents.append(AnthropicAgent("claude-adversary", model="claude-sonnet-4-20250514"))
            if os.getenv("OPENAI_API_KEY"):
                agents.append(OpenAIAgent("gpt-adversary", model="gpt-4o"))
            if os.getenv("GEMINI_API_KEY"):
                agents.append(GeminiAgent("gemini-adversary"))
            if agents:
                logger.info(f"Using {len(agents)} real API agents")
                return agents
        except ImportError:
            pass

    # Fall back to demo agents
    logger.info("Using simulated demo agents (set API keys for real agents)")
    return [
        DemoAgent("claude-demo", "Security-focused analyst"),
        DemoAgent("gpt-demo", "Edge case specialist"),
        DemoAgent("gemini-demo", "Compliance reviewer"),
        DemoAgent("mistral-demo", "Systems architect"),
    ]


def print_banner():
    """Print the Aragora Gauntlet banner."""
    banner = f"""
{color('=' * 60, Colors.CYAN)}
{color('  ARAGORA GAUNTLET', Colors.BOLD + Colors.CYAN)}
{color('  Adversarial Validation Engine', Colors.CYAN)}
{color('=' * 60, Colors.CYAN)}

{color('"Stress-test high-stakes decisions before they break your business"', Colors.BLUE)}

"""
    print(banner)


def print_progress(stage: str, status: str = "in_progress"):
    """Print progress update."""
    icon = {
        "in_progress": color(">>", Colors.BLUE),
        "complete": color("OK", Colors.GREEN),
        "warning": color("!!", Colors.WARNING),
        "error": color("XX", Colors.FAIL),
    }.get(status, "  ")
    print(f"  [{icon}] {stage}")


def print_result(result: GauntletResult):
    """Print the Gauntlet result summary."""
    verdict_color = {
        Verdict.APPROVED: Colors.GREEN,
        Verdict.APPROVED_WITH_CONDITIONS: Colors.WARNING,
        Verdict.NEEDS_REVIEW: Colors.WARNING,
        Verdict.REJECTED: Colors.FAIL,
    }.get(result.verdict, Colors.ENDC)

    print(f"\n{color('=' * 60, Colors.CYAN)}")
    print(color("  GAUNTLET RESULT", Colors.BOLD))
    print(f"{color('=' * 60, Colors.CYAN)}\n")

    print(
        f"  {color('VERDICT:', Colors.BOLD)} {color(result.verdict.value.upper(), verdict_color)}"
    )
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Risk Score: {result.risk_score:.0%}")
    print(f"  Robustness: {result.robustness_score:.0%}")
    print(f"  Coverage: {result.coverage_score:.0%}")

    print(f"\n  {color('FINDINGS:', Colors.BOLD)}")
    print(
        f"    Critical: {color(str(len(result.critical_findings)), Colors.FAIL if result.critical_findings else Colors.ENDC)}"
    )
    print(
        f"    High:     {color(str(len(result.high_findings)), Colors.WARNING if result.high_findings else Colors.ENDC)}"
    )
    print(f"    Medium:   {len(result.medium_findings)}")
    print(f"    Low:      {len(result.low_findings)}")

    if result.critical_findings:
        print(f"\n  {color('CRITICAL ISSUES:', Colors.FAIL)}")
        for f in result.critical_findings[:3]:
            print(f"    - {f.title}")

    if result.dissenting_views:
        print(f"\n  {color('DISSENTING VIEWS:', Colors.WARNING)} {len(result.dissenting_views)}")

    print(f"\n  Duration: {result.duration_seconds:.1f}s")
    print(f"  Agents: {', '.join(result.agents_involved)}")
    print(f"  Checksum: {result.checksum}")


async def run_demo(
    input_file: Optional[Path] = None,
    profile: str = "quick",
    output_dir: Optional[Path] = None,
    use_real_apis: bool = False,
):
    """Run the Gauntlet demo."""
    print_banner()

    # Load input
    if input_file and input_file.exists():
        input_content = input_file.read_text()
        input_name = input_file.name
        logger.info(f"Loaded input from: {input_file}")
    else:
        input_content = SAMPLE_POLICY
        input_name = "sample_policy.md"
        logger.info("Using sample rate limiting policy")

    # Get profile config
    if profile not in PROFILES:
        logger.error(f"Unknown profile: {profile}. Available: {list(PROFILES.keys())}")
        return None

    config = PROFILES[profile]

    # Update config with input
    config.input_content = input_content
    if "policy" in input_name.lower():
        config.input_type = InputType.POLICY
    elif "code" in input_name.lower() or input_name.endswith(".py"):
        config.input_type = InputType.CODE
    else:
        config.input_type = InputType.SPEC

    print(f"\n{color('Configuration:', Colors.BOLD)}")
    print(f"  Profile: {profile}")
    print(f"  Input Type: {config.input_type.value}")
    print(f"  Input Size: {len(input_content):,} chars")
    print(f"  Max Duration: {config.max_duration_seconds}s")
    print()

    # Create agents
    agents = create_agents(use_real_apis)

    # Create orchestrator
    orchestrator = GauntletOrchestrator(agents)

    # Run Gauntlet with progress updates
    print(f"{color('Running Gauntlet...', Colors.BOLD)}\n")

    print_progress("Initializing adversarial agents")
    await asyncio.sleep(0.3)  # Visual delay
    print_progress("Initializing adversarial agents", "complete")

    print_progress("Running red-team attacks")
    result = await orchestrator.run(config)
    print_progress("Running red-team attacks", "complete")

    print_progress("Aggregating findings")
    print_progress("Aggregating findings", "complete")

    print_progress("Generating verdict")
    print_progress("Generating verdict", "complete")

    # Print result
    print_result(result)

    # Generate and save receipt
    output_dir = output_dir or Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    receipt = DecisionReceiptGenerator.from_gauntlet_result(result)

    # Save in multiple formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"gauntlet_receipt_{timestamp}"

    html_path = output_dir / f"{base_name}.html"
    receipt.save(html_path, format="html")

    json_path = output_dir / f"{base_name}.json"
    receipt.save(json_path, format="json")

    md_path = output_dir / f"{base_name}.md"
    receipt.save(md_path, format="md")

    print(f"\n{color('Decision Receipt saved:', Colors.GREEN)}")
    print(f"  HTML: {html_path}")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    print(f"\n{color('Open the HTML report:', Colors.BOLD)}")
    print(f"  open {html_path}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aragora Gauntlet Demo - Adversarial Validation in Action",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_gauntlet.py                     # Fast demo (default)
  python scripts/demo_gauntlet.py my_spec.md          # Your own document
  python scripts/demo_gauntlet.py --profile quick     # Full quick analysis (~2min)
  python scripts/demo_gauntlet.py --profile thorough  # Comprehensive (~15min)
  python scripts/demo_gauntlet.py --real-apis         # Use real AI agents
        """,
    )
    parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        help="Path to input document (default: sample policy)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        choices=list(PROFILES.keys()),
        default="demo",
        help="Gauntlet profile (demo=fast, quick=2min, thorough=15min)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for receipts (default: current directory)",
    )
    parser.add_argument(
        "--real-apis",
        action="store_true",
        help="Use real API agents if API keys are available",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(
            run_demo(
                input_file=args.input_file,
                profile=args.profile,
                output_dir=args.output,
                use_real_apis=args.real_apis,
            )
        )
    except KeyboardInterrupt:
        print(f"\n{color('Gauntlet cancelled.', Colors.WARNING)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Gauntlet failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
