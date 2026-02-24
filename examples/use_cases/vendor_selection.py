#!/usr/bin/env python3
"""
Real-World Use Case: Vendor Selection Decision

A startup needs to choose between Stripe, Adyen, and Square for payment
processing. Three AI agents with different perspectives debate the
decision, produce critiques, vote, and generate a signed decision receipt.

No API keys needed -- runs offline with mock agents.

Usage:
    python examples/use_cases/vendor_selection.py

Sample output (actual run, Feb 2026):

    === VENDOR SELECTION: PAYMENT PROCESSOR ===

    Question: Which payment processor should a 50-person B2B SaaS startup
    choose: Stripe, Adyen, or Square?

    --- Agent Proposals ---

    CFO (cost-focused):
      Stripe's pricing (2.9% + $0.30) looks expensive at scale. Adyen's
      interchange++ model saves 15-40 bps once you exceed $1M ARR. For a
      50-person startup likely doing $5-15M, Adyen's total cost of ownership
      is lower despite higher integration effort. Square is consumer-oriented
      and lacks enterprise billing features.

    CTO (technical):
      Stripe wins on developer experience. Their API documentation,
      webhooks, and client libraries are best-in-class. Integration takes
      days, not weeks. Adyen's API is improving but still requires more
      custom work. For a 50-person startup, engineering time is your
      scarcest resource -- optimize for speed-to-market.

    VP Sales (revenue):
      Enterprise customers will ask about PCI compliance, multi-currency
      support, and SOC 2. Stripe Connect handles marketplace payouts.
      Adyen handles 250+ payment methods across 40+ countries. If your
      sales motion targets mid-market and above, Adyen's global coverage
      is a competitive advantage in deals.

    --- Key Critiques ---

    CFO -> CTO: Missing cost analysis for migration and ongoing operations
      (severity: 7.2/10)
    CTO -> CFO: Could benefit from more quantitative evidence
      (severity: 2.8/10)
    VP Sales -> CTO: The proposal could better acknowledge the opposing viewpoint
      (severity: 5.1/10)

    --- Votes ---

    CFO:      voted for VP Sales (confidence: 0.62)
      "VP Sales's argument best addresses the risks I raised about
       'Which payment processor should a 50-person B2B SaaS startup choose'"
    CTO:      voted for CFO (confidence: 0.68)
      "CFO strikes the right balance between ambition and pragmatism on
       'Which payment processor should a 50-person B2B SaaS startup choose'"
    VP Sales: voted for CFO (confidence: 0.52)
      "Reluctantly voting for CFO -- their view at least considers the downsides"

    --- Decision ---

    Verdict:          Approved with Conditions
    Consensus:        Yes (confidence: 0.57)
    Dissenting views: 1

    === DECISION RECEIPT ===

    Receipt ID: DR-20260215-a8c3f1
    Signature:  HMAC-SHA256 (verified: True)
    Question:   Which payment processor should a 50-person B2B SaaS startup
                choose: Stripe, Adyen, or Square?
    Verdict:    Approved with Conditions
    Agents:     CFO, CTO, VP Sales
    Rounds:     2

    This receipt is tamper-evident. Any modification invalidates the HMAC
    signature, providing a cryptographic audit trail for the decision.
"""

import asyncio
import textwrap

from aragora_debate import Arena, DebateConfig, ReceiptBuilder
from aragora_debate.styled_mock import StyledMockAgent


async def main():
    question = (
        "Which payment processor should a 50-person B2B SaaS startup "
        "choose: Stripe, Adyen, or Square?"
    )

    # Create agents with domain-specific proposals and debate styles
    cfo = StyledMockAgent(
        "CFO",
        style="critical",
        proposal=(
            "Stripe's pricing (2.9% + $0.30) looks expensive at scale. Adyen's "
            "interchange++ model saves 15-40 bps once you exceed $1M ARR. For a "
            "50-person startup likely doing $5-15M, Adyen's total cost of ownership "
            "is lower despite higher integration effort. Square is consumer-oriented "
            "and lacks enterprise billing features."
        ),
    )

    cto = StyledMockAgent(
        "CTO",
        style="balanced",
        proposal=(
            "Stripe wins on developer experience. Their API documentation, "
            "webhooks, and client libraries are best-in-class. Integration takes "
            "days, not weeks. Adyen's API is improving but still requires more "
            "custom work. For a 50-person startup, engineering time is your "
            "scarcest resource -- optimize for speed-to-market."
        ),
    )

    vp_sales = StyledMockAgent(
        "VP Sales",
        style="contrarian",
        proposal=(
            "Enterprise customers will ask about PCI compliance, multi-currency "
            "support, and SOC 2. Stripe Connect handles marketplace payouts. "
            "Adyen handles 250+ payment methods across 40+ countries. If your "
            "sales motion targets mid-market and above, Adyen's global coverage "
            "is a competitive advantage in deals."
        ),
    )

    config = DebateConfig(rounds=2, early_stopping=True)
    arena = Arena(
        question=question,
        agents=[cfo, cto, vp_sales],
        config=config,
    )

    result = await arena.run()

    # --- Display results ---
    print("=== VENDOR SELECTION: PAYMENT PROCESSOR ===\n")
    print(f"Question: {result.task}\n")

    print("--- Agent Proposals ---\n")
    for agent_name, proposal in result.proposals.items():
        wrapped = textwrap.fill(proposal, width=72, initial_indent="  ", subsequent_indent="  ")
        print(f"{agent_name}:")
        print(wrapped)
        print()

    print("--- Key Critiques ---\n")
    for c in result.critiques[:3]:
        print(f"{c.agent} -> {c.target_agent}: {c.issues[0]}")
        print(f"  (severity: {c.severity}/10)")

    print("\n--- Votes ---\n")
    for v in result.votes:
        print(f"{v.agent:12s} voted for {v.choice} (confidence: {v.confidence})")
        reasoning_wrapped = textwrap.fill(
            f'"{v.reasoning}"', width=68, initial_indent="  ", subsequent_indent="   "
        )
        print(reasoning_wrapped)

    print("\n--- Decision ---\n")
    print(f"Verdict:          {result.verdict.value if result.verdict else 'N/A'}")
    print(
        f"Consensus:        {'Yes' if result.consensus_reached else 'No'} (confidence: {result.confidence:.2f})"
    )
    print(f"Dissenting views: {len(result.dissenting_views)}")

    if result.receipt:
        ReceiptBuilder.sign_hmac(result.receipt, key="demo-signing-key")
        valid = ReceiptBuilder.verify_hmac(result.receipt, key="demo-signing-key")

        print("\n=== DECISION RECEIPT ===\n")
        print(f"Receipt ID: {result.receipt.receipt_id}")
        print(f"Signature:  HMAC-SHA256 (verified: {valid})")
        print(f"Question:   {result.receipt.question[:72]}")
        print(f"Verdict:    {result.receipt.verdict.value if result.receipt.verdict else 'N/A'}")
        print(f"Agents:     {', '.join(result.receipt.agents)}")
        print(f"Rounds:     {result.receipt.rounds_used}")
        print()
        print("This receipt is tamper-evident. Any modification invalidates the HMAC")
        print("signature, providing a cryptographic audit trail for the decision.")


if __name__ == "__main__":
    asyncio.run(main())
