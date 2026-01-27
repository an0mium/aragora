#!/usr/bin/env python3
"""
Gauntlet Demo - Showcase adversarial stress-testing.

This demo runs the Gauntlet against intentionally flawed specifications
to demonstrate how it surfaces critical issues.

Usage:
    python examples/gauntlet_demo.py [--demo security|gdpr|scaling|all]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.gauntlet import (
    GauntletOrchestrator,
    OrchestratorConfig as GauntletConfig,
    InputType,
)


# Demo specs with intentional flaws
DEMO_SPECS = {
    "security": {
        "file": "sample_specs/insecure_api.md",
        "name": "Insecure API Specification",
        "persona": "security",
        "description": "API spec with SQL injection, plaintext passwords, and missing auth",
    },
    "gdpr": {
        "file": "sample_specs/gdpr_violation.md",
        "name": "GDPR Violation Spec",
        "persona": "gdpr",
        "description": "Analytics platform with consent, retention, and transfer violations",
    },
    "scaling": {
        "file": "sample_specs/scaling_timebomb.md",
        "name": "Scaling Timebomb Architecture",
        "persona": "security",  # Use security for architecture review
        "description": "Architecture with single points of failure and scaling issues",
    },
}


def print_banner(text: str, char: str = "=") -> None:
    """Print a banner."""


def print_finding(finding, index: int) -> None:
    """Print a finding with formatting."""
    severity_colors = {
        "CRITICAL": "\033[91m",  # Red
        "HIGH": "\033[93m",  # Yellow
        "MEDIUM": "\033[94m",  # Blue
        "LOW": "\033[92m",  # Green
    }

    level = finding.severity_level
    severity_colors.get(level, "")

    if finding.mitigation:
        pass


async def run_demo(demo_key: str, verbose: bool = False) -> dict:
    """Run a single demo."""
    demo = DEMO_SPECS[demo_key]
    spec_path = Path(__file__).parent / demo["file"]

    if not spec_path.exists():
        return {"error": f"Spec not found: {spec_path}"}

    print_banner(f"DEMO: {demo['name']}")

    input_content = spec_path.read_text()

    # Create mock agents for demo (no API calls needed)
    from aragora.core import Agent

    class DemoAgent(Agent):
        """Demo agent that simulates findings."""

        async def generate(self, prompt: str, history: list) -> str:
            # Simulate finding issues based on prompt content
            if "security" in prompt.lower() or "injection" in prompt.lower():
                return """
                Critical security violation found: SQL Injection vulnerability.

                The specification mentions using string concatenation for SQL queries:
                "SELECT * FROM users WHERE username = '" + username + "'"

                This is a critical SQL injection vulnerability that could allow
                attackers to:
                - Extract all user data
                - Modify or delete records
                - Bypass authentication
                - Execute administrative commands

                Severity: CRITICAL
                Risk: Data breach, full system compromise

                Additional findings:
                - Passwords stored in plaintext (High severity)
                - No HTTPS requirement (High severity)
                - Sequential user IDs (Medium severity - enumeration attack)
                - Admin endpoints without authentication (Critical severity)
                """
            elif "gdpr" in prompt.lower() or "consent" in prompt.lower():
                return """
                Critical GDPR violation found: Invalid consent mechanism.

                The specification uses pre-checked consent checkboxes and treats
                continued use as implied consent. This violates GDPR Article 7
                requirements for freely given, specific, informed consent.

                Critical issues:
                - Pre-checked consent boxes violate consent requirements
                - No lawful basis for processing special category data
                - Health data processed without explicit consent
                - No data subject rights implementation
                - Indefinite retention violates storage limitation

                Severity: CRITICAL
                Risk: GDPR fines up to 4% of global turnover
                """
            elif "architecture" in prompt.lower() or "scaling" in prompt.lower():
                return """
                High severity issue found: Single point of failure.

                The architecture relies on a single PostgreSQL database server
                with no replication. At 1M requests/second, this creates:

                - Database bottleneck (single-threaded queries)
                - No failover capability
                - Latency issues (100ms target impossible)
                - Global locks causing contention

                Additional concerns:
                - No geographic distribution
                - Synchronous processing at high volume
                - No circuit breakers for dependencies

                Severity: HIGH
                Risk: Complete service outage, data loss
                """
            return "Analysis complete. No critical issues identified in this area."

        def critique(self, response: str, context: str) -> str:
            return "Critique noted."

    # Create config
    config = GauntletConfig(
        input_type=InputType.SPEC,
        input_content=input_content,
        persona=demo["persona"],
        enable_redteam=False,  # Use persona attacks instead
        enable_probing=False,
        enable_deep_audit=False,
        enable_verification=False,
        enable_risk_assessment=True,
        max_duration_seconds=30,
    )

    # Run gauntlet

    orchestrator = GauntletOrchestrator(
        agents=[DemoAgent(name="demo_agent", model="demo")],
    )

    result = await orchestrator.run(config)

    # Print results
    print_banner("RESULTS", "-")

    # Verdict
    verdict_emoji = {
        "approved": "✅",
        "approved_with_conditions": "⚠️",
        "needs_review": "🔍",
        "rejected": "❌",
    }
    verdict_emoji.get(result.verdict.value, "❓")

    # Findings summary

    # Show critical findings
    if result.critical_findings:
        for i, finding in enumerate(result.critical_findings[:3], 1):
            print_finding(finding, i)

    if result.high_findings and verbose:
        for i, finding in enumerate(result.high_findings[:3], 1):
            print_finding(finding, i)

    return {
        "demo": demo_key,
        "verdict": result.verdict.value,
        "findings": result.total_findings,
        "critical": len(result.critical_findings),
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Gauntlet Demo - Showcase adversarial stress-testing"
    )
    parser.add_argument(
        "--demo",
        "-d",
        choices=["security", "gdpr", "scaling", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show more detailed findings",
    )

    args = parser.parse_args()

    print_banner("ARAGORA GAUNTLET DEMO", "=")

    if args.demo == "all":
        demos = list(DEMO_SPECS.keys())
    else:
        demos = [args.demo]

    results = []
    for demo in demos:
        result = await run_demo(demo, args.verbose)
        results.append(result)

    # Summary
    print_banner("DEMO SUMMARY", "=")
    for r in results:
        if "error" not in r:
            pass


if __name__ == "__main__":
    asyncio.run(main())
