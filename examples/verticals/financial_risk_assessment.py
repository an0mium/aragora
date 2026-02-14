#!/usr/bin/env python3
"""Financial vertical example: Credit risk assessment debate.

Demonstrates aragora-debate for a realistic financial decision where
risk analysts debate extending a credit facility to a mid-market company
showing mixed financial signals.

Usage:
    python examples/verticals/financial_risk_assessment.py

No API keys required — uses mock agents with realistic financial analysis.
"""

from __future__ import annotations

import asyncio
import textwrap

from aragora_debate import Debate, DebateConfig, create_agent


def main() -> None:
    """Run a credit risk assessment debate."""

    # ── Deal scenario ──────────────────────────────────────────────────
    scenario = textwrap.dedent("""\
        Borrower: NovaTech Manufacturing Inc. (ticker: private)
        Request: $25M revolving credit facility, 5-year term
        Purpose: Working capital and equipment modernization
        Current rating: BB+ (internal), BB (S&P equivalent)

        Financial highlights (FY2025):
        - Revenue: $180M (+12% YoY), EBITDA: $27M (15% margin)
        - Net debt/EBITDA: 3.2x (covenant threshold: 4.0x)
        - Interest coverage: 4.1x
        - Current ratio: 1.4x, Quick ratio: 0.9x
        - Free cash flow: $8M (down from $14M due to capex)
        - Accounts receivable DSO: 68 days (industry avg: 45 days)

        Risk factors:
        - Top 3 customers = 52% of revenue (concentration risk)
        - Capex cycle: $15M committed for new CNC equipment
        - Industry headwind: tariff uncertainty on imported steel (30% of COGS)
        - Management: CEO transition in Q3 2025, new CEO from competitor

        Mitigants:
        - $40M undrawn ABL facility (asset-backed, separate lender)
        - Order backlog: $95M (6+ months visibility)
        - Real estate collateral: two owned manufacturing facilities ($35M appraised)
    """)

    # ── Configure analyst agents ───────────────────────────────────────
    credit_analyst = create_agent(
        "mock",
        name="Sarah Kim (Senior Credit Analyst)",
        proposal=textwrap.dedent("""\
            Recommend APPROVAL with enhanced covenants.

            Credit assessment (Basel III standardized approach):
            - Probability of default (PD): 2.8% (BB+ calibration, Moody's CreditEdge)
            - Loss given default (LGD): 35% (senior secured with real estate)
            - Expected loss: $245K annually on $25M facility
            - Risk-adjusted return: 3.2% spread yields adequate RAROC of 18%

            Positive factors:
            1. Revenue growth trajectory: 12% YoY in cyclical downturn signals market share gains
            2. EBITDA margin stability: 15% vs industry median 11% — pricing power
            3. Collateral coverage: $35M real estate + $95M backlog provides 5.2x coverage
            4. Leverage headroom: 3.2x vs 4.0x covenant = 0.8x EBITDA ($13.5M) buffer

            Enhanced covenant package:
            - Max net debt/EBITDA: 3.75x (tighter than standard 4.0x)
            - Minimum DSCR: 1.25x (tested quarterly)
            - Customer concentration cap: no single customer >25% of revenue
            - Mandatory prepayment: 50% of excess cash flow above $10M
            - Change of control clause: 30-day acceleration trigger on CEO departure

            Pricing: SOFR + 275bps, 50bps commitment fee on undrawn.
        """),
        vote_for="Sarah Kim (Senior Credit Analyst)",
        critique_issues=["DSO of 68 days signals potential collection issues",
                         "FCF decline needs monitoring despite capex explanation"],
    )

    risk_manager = create_agent(
        "mock",
        name="David Chen (Portfolio Risk Manager)",
        proposal=textwrap.dedent("""\
            Recommend CONDITIONAL APPROVAL — reduce facility to $18M with step-up provision.

            Portfolio-level concerns:
            1. Manufacturing sector exposure: 23% of book (internal limit: 25%)
               - Adding $25M pushes to 24.8% — dangerously close to concentration limit
               - Prefer $18M to maintain 1.5% buffer

            2. Customer concentration (CRITICAL):
               - 52% revenue from top 3 customers is severe
               - If #1 customer (est. 22% of revenue) churns: revenue drops to $140M,
                 EBITDA ~$16M, leverage spikes to 5.4x → covenant breach
               - Stress test shows 40% probability of covenant breach within 24 months
                 under customer loss scenario

            3. CEO transition risk:
               - Management transitions in leveraged mid-market have 3.2x higher default
                 rate in first 18 months (S&P LCD data, 2019-2024)
               - New CEO from competitor = potential cultural friction, strategy pivot risk

            4. Steel tariff exposure:
               - 30% COGS exposure to imported steel
               - 25% tariff = $8.1M EBITDA hit = leverage jumps to 4.6x (COVENANT BREACH)
               - No evidence of hedging program or domestic supplier pivot

            Step-up provision: If NovaTech maintains covenants for 4 consecutive quarters
            AND reduces customer concentration below 45%, facility increases to $25M.
        """),
        vote_for="David Chen (Portfolio Risk Manager)",
        critique_issues=["Collateral provides meaningful downside protection",
                         "Order backlog mitigates near-term revenue risk"],
    )

    compliance_officer = create_agent(
        "mock",
        name="Maria Santos (Regulatory Compliance)",
        proposal=textwrap.dedent("""\
            From a regulatory and audit perspective, conditional approval is appropriate.

            SOX and regulatory considerations:
            1. Fair lending documentation: ✓ Consistent with peer group treatment
            2. BSA/AML screening: ✓ No adverse findings on NovaTech or principals
            3. CRA impact: Positive — supports manufacturing employment in LMI census tract

            Audit-readiness requirements:
            - Credit memo must document tariff stress scenario (OCC 2024-3 guidance)
            - Customer concentration must be flagged in quarterly watch list
            - CEO transition requires enhanced monitoring per internal policy CP-112
            - Quarterly financial covenant compliance must be independently verified

            Model risk (SR 11-7):
            - PD model last validated Q2 2025 — current
            - LGD estimate uses appraised values from 2024 — recommend updated appraisal
            - Stress testing covers interest rate and recession scenarios but NOT
              tariff-specific supply chain disruption — gap identified

            Recommendation: Approve at reduced amount ($18-20M) pending:
            1. Updated real estate appraisal (< 12 months old required)
            2. Tariff impact analysis added to stress testing framework
            3. Management transition monitoring protocol activated
        """),
        vote_for="David Chen (Portfolio Risk Manager)",
        critique_issues=["Documentation requirements should not delay credit decision",
                         "Enhanced monitoring addresses most identified risks"],
    )

    # ── Run the debate ─────────────────────────────────────────────────
    debate = Debate(
        topic="$25M revolving credit facility for NovaTech Manufacturing: Approve, Conditional, or Decline?",
        context=scenario,
        rounds=2,
        consensus="majority",
        enable_trickster=True,
        trickster_sensitivity=0.5,
    )

    debate.add_agent(credit_analyst)
    debate.add_agent(risk_manager)
    debate.add_agent(compliance_officer)

    result = asyncio.run(debate.run())

    # ── Format financial decision receipt ──────────────────────────────
    from aragora_debate.receipt import ReceiptBuilder

    receipt = result.receipt
    print("=" * 72)
    print("  CREDIT DECISION RECEIPT")
    print("  Aragora Decision Integrity Platform — Financial Services Vertical")
    print("=" * 72)
    print()
    print(f"  Decision ID:    {receipt.receipt_id}")
    print(f"  Question:       {receipt.question[:60]}...")
    print(f"  Verdict:        {receipt.verdict.value if receipt.verdict else 'N/A'}")
    print(f"  Confidence:     {receipt.confidence:.0%}")
    print(f"  Consensus:      {'Reached' if receipt.consensus.reached else 'Not reached'} ({receipt.consensus.method.value})")
    print(f"  Agents:         {len(receipt.agents)}")
    print(f"  Rounds:         {receipt.rounds_used}")
    print()

    # Supporting vs dissenting
    print("  VOTE BREAKDOWN")
    print("  " + "-" * 50)
    if receipt.consensus.supporting_agents:
        print(f"  Supporting:     {', '.join(receipt.consensus.supporting_agents)}")
    if receipt.consensus.dissenting_agents:
        print(f"  Dissenting:     {', '.join(receipt.consensus.dissenting_agents)}")
    print()

    # Dissenting opinions
    if receipt.consensus.dissents:
        print("  DISSENTING OPINIONS")
        print("  " + "-" * 50)
        for dissent in receipt.consensus.dissents:
            print(f"  Agent:   {dissent.agent}")
            if dissent.reasons:
                print(f"  Reason:  {dissent.reasons[0][:80]}")
            if dissent.alternative_view:
                print(f"  Alt:     {dissent.alternative_view[:80]}")
            print()

    # Integrity
    print("  AUDIT TRAIL")
    print("  " + "-" * 50)
    ReceiptBuilder.sign_hmac(receipt, "credit-committee-key-2025")
    print(f"  HMAC-SHA256:    {receipt.signature[:40]}...")
    print(f"  Timestamp:      {receipt.timestamp}")
    print(f"  Tamper-proof:   {ReceiptBuilder.verify_hmac(receipt, 'credit-committee-key-2025')}")
    print()

    # SOX compliance notes
    print("  SOX / REGULATORY COMPLIANCE NOTES")
    print("  " + "-" * 50)
    print("  - Decision documented per OCC 2024-3 credit risk guidance")
    print("  - Stress scenarios include tariff, customer loss, rate shock")
    print("  - Receipt immutable and auditable (SHA-256 signed)")
    print("  - Retained per SOX §802: minimum 7-year retention")
    print("  - Model risk documented per SR 11-7 / OCC 2011-12")
    print()
    print("=" * 72)

    # Export options
    md = receipt.to_markdown()
    print(f"\n  Receipt exported: {len(md)} chars Markdown")
    json_str = ReceiptBuilder.to_json(receipt)
    print(f"  Receipt exported: {len(json_str)} chars JSON")
    print(f"\n  Run with real LLMs:")
    print(f'  Replace create_agent("mock", ...) with create_agent("anthropic", ...)')


if __name__ == "__main__":
    main()
