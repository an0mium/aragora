"""
Example: Using Aragora as an MCP verification service.

This shows how external AI agents (Claude Desktop, custom agents, etc.)
can call Aragora's MCP tools to verify plans and retrieve decision receipts.

Setup:
    1. Install aragora: pip install aragora
    2. Configure API keys (at least one):
       export ANTHROPIC_API_KEY=sk-ant-...
       export OPENAI_API_KEY=sk-...
    3. Run this example: python examples/mcp_integration.py

MCP Server Configuration:
    Add to your Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):

    {
        "mcpServers": {
            "aragora": {
                "command": "python",
                "args": ["-m", "aragora.mcp.server"],
                "env": {
                    "ANTHROPIC_API_KEY": "sk-ant-...",
                    "OPENAI_API_KEY": "sk-..."
                }
            }
        }
    }

    Or run the MCP server standalone:

        python -m aragora.mcp.server --port 8765

Available Verification Tools:
    - verify_plan: Submit a plan/decision for multi-agent adversarial debate
    - get_receipt: Retrieve a decision receipt by ID (JSON, Markdown, SARIF)

These tools complement the existing MCP tools:
    - run_debate: General-purpose multi-agent debate
    - run_gauntlet: Adversarial stress-testing
    - get_decision_receipt: Receipt from a debate ID
    - verify_decision_receipt: Verify receipt signature/integrity
"""

from __future__ import annotations

import asyncio
import json


async def example_verify_plan() -> None:
    """Demonstrate the verify_plan tool for decision verification."""
    from aragora.mcp.tools_module.verification import verify_plan_tool

    print("=" * 60)
    print("Example 1: Verify a deployment plan")
    print("=" * 60)

    result = await verify_plan_tool(
        plan="""
        Deploy the new payment service to production:
        1. Run database migration to add payment_intents table
        2. Deploy payment-service v2.1.0 to all regions simultaneously
        3. Enable feature flag for new checkout flow
        4. Remove old payment endpoints after 24 hours
        """,
        context="E-commerce platform handling $2M daily transactions. "
        "PCI-DSS compliant environment. No maintenance window available.",
        rounds=2,
        focus="security,quality",
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        print("\nNote: This example requires configured API keys.")
        print("Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY to run.")
        return

    print(f"\nVerdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Agreement Score: {result['agreement_score']:.0%}")
    print(f"Findings: {result['findings_count']}")
    print(f"Receipt ID: {result['receipt_id']}")
    print(f"Duration: {result['duration_seconds']}s")

    if result.get("unanimous_issues"):
        print("\nUnanimous Issues (all agents agree):")
        for issue in result["unanimous_issues"]:
            print(f"  - {issue}")

    print(f"\nRisk Summary: {json.dumps(result.get('risk_summary', {}), indent=2)}")

    return result


async def example_get_receipt(receipt_id: str) -> None:
    """Demonstrate the get_receipt tool for retrieving verification results."""
    from aragora.mcp.tools_module.verification import get_receipt_tool

    print("\n" + "=" * 60)
    print("Example 2: Retrieve receipt in different formats")
    print("=" * 60)

    # JSON format
    json_result = await get_receipt_tool(
        receipt_id=receipt_id,
        format="json",
    )

    if "error" in json_result:
        print(f"Error: {json_result['error']}")
        return

    print(f"\nJSON Receipt for {json_result['receipt_id']}:")
    content = json_result.get("content", {})
    if isinstance(content, dict):
        print(f"  Verdict: {content.get('verdict', 'N/A')}")
        print(f"  Confidence: {content.get('confidence', 'N/A')}")

    # Markdown format
    md_result = await get_receipt_tool(
        receipt_id=receipt_id,
        format="markdown",
    )

    if "error" not in md_result:
        print("\nMarkdown Receipt (first 500 chars):")
        md_content = md_result.get("content", "")
        print(md_content[:500] if isinstance(md_content, str) else str(md_content)[:500])


async def example_programmatic_integration() -> None:
    """Show how to use verify_plan in a CI/CD pipeline or automation."""
    from aragora.mcp.tools_module.verification import verify_plan_tool

    print("\n" + "=" * 60)
    print("Example 3: CI/CD gate - block deployments on critical findings")
    print("=" * 60)

    # In a real CI/CD pipeline, you would read the plan from a PR or config
    plan = """
    API Migration Plan:
    1. Add new /v2/users endpoint alongside /v1/users
    2. Update client SDKs to use v2
    3. Deprecate v1 after 30 days
    4. Store user passwords in plaintext for faster auth
    """

    result = await verify_plan_tool(
        plan=plan,
        context="Production API serving 10k requests/second",
        focus="security,quality,performance",
        rounds=2,
    )

    if "error" in result:
        print(f"Skipping (no API keys): {result['error']}")
        return

    # Gate logic: block on FAIL verdict or critical issues
    verdict = result.get("verdict", "UNKNOWN")
    critical = result.get("critical_count", 0)

    if verdict == "FAIL" or critical > 0:
        print(f"\nBLOCKED: Verdict={verdict}, Critical issues={critical}")
        print("Deployment blocked. Fix issues before proceeding.")
        # In CI: sys.exit(1)
    elif verdict == "CONDITIONAL":
        print(f"\nWARNING: Verdict={verdict}")
        print("Deployment allowed with conditions. Review findings.")
    else:
        print(f"\nAPPROVED: Verdict={verdict}")
        print("Deployment approved.")

    print(f"\nReceipt ID for audit trail: {result.get('receipt_id')}")


async def main() -> None:
    """Run all examples."""
    print("Aragora MCP Decision Verification Examples")
    print("=" * 60)
    print()

    # Example 1: Verify a plan
    result = await example_verify_plan()

    # Example 2: Retrieve receipt (only if example 1 succeeded)
    if result and "receipt_id" in result:
        await example_get_receipt(result["receipt_id"])

    # Example 3: CI/CD integration pattern
    await example_programmatic_integration()

    print("\n" + "=" * 60)
    print("Done. See ~/.aragora/reviews/ for saved receipts.")


if __name__ == "__main__":
    asyncio.run(main())
