#!/usr/bin/env python3
"""Legal vertical example: M&A due diligence risk assessment.

Demonstrates aragora-debate for a legal due diligence scenario where
attorneys debate material risks in a proposed acquisition, producing an
audit-ready decision receipt.

Usage:
    python examples/verticals/legal_due_diligence.py

No API keys required — uses mock agents with realistic legal analysis.
"""

from __future__ import annotations

import asyncio
import textwrap

from aragora_debate import Debate, DebateConfig, create_agent


def main() -> None:
    """Run a legal due diligence risk debate."""

    # ── Deal scenario ──────────────────────────────────────────────────
    scenario = textwrap.dedent("""\
        Transaction: Acquisition of DataVault SaaS Inc. by Meridian Corp.
        Deal value: $120M (8.5x ARR)
        Structure: Stock purchase, 100% acquisition
        Signing target: 45 days

        DataVault profile:
        - B2B SaaS platform for document management and compliance
        - ARR: $14.1M, 340 enterprise customers, 97% gross retention
        - 85 employees (62 engineering), founded 2019
        - Incorporated: Delaware, HQ: Austin, TX

        Key findings from due diligence:
        1. IP: Core ML model trained on dataset containing scraped web content
           - No license agreements for 30% of training data sources
           - 2 pending DMCA takedown requests from content publishers
        2. Customer contracts: 40% of ARR on legacy terms with unlimited liability clauses
           - No cap on consequential damages in 137 contracts
           - Indemnification obligations uncapped for data breaches
        3. Data privacy: Processes EU customer data in US-only infrastructure
           - No Standard Contractual Clauses (SCCs) executed
           - DPA signed with only 60% of data sub-processors
        4. Employment: 3 former engineers from competitor joined without non-compete clearance
           - Competitor (DocStream) sent cease-and-desist letter in Q2 2025
           - No formal trade secret audit conducted on new hires
        5. Regulatory: SOC 2 Type II audit in progress, estimated completion Q1 2026
           - 4 critical findings in most recent penetration test (Q3 2025)
           - No FedRAMP authorization (required for 15% of pipeline)
    """)

    # ── Configure legal analyst agents ─────────────────────────────────
    ma_counsel = create_agent(
        "mock",
        name="Jennifer Park (M&A Lead Counsel)",
        proposal=textwrap.dedent("""\
            Recommend PROCEED with renegotiated terms and specific indemnity escrow.

            Overall assessment: The identified risks are significant but manageable
            through deal structure adjustments. None rise to the level of deal-breakers
            given DataVault's strong retention metrics and product-market fit.

            Proposed deal modifications:
            1. Purchase price adjustment: Reduce from $120M to $108M (-10%)
               reflecting IP remediation costs ($3M) and contract migration ($5M)
               plus risk discount for pending litigation exposure ($4M)

            2. Indemnification escrow: $15M (12.5% of revised price) held for 24 months
               - IP claims: $8M basket (covers training data and DMCA exposure)
               - Contract liability: $5M basket (unlimited liability migration)
               - Employment: $2M basket (DocStream trade secret claim)

            3. Closing conditions:
               - SCCs executed for all EU data transfers
               - DocStream C&D resolved or indemnified
               - SOC 2 Type II completion timeline committed in SPA
               - Formal trade secret audit completed for flagged engineers

            4. Representations & warranties insurance: $20M policy recommended
               to supplement seller indemnification (est. premium: $400K)

            Risk-adjusted IRR with modifications: 22% (vs 26% at original price).
            Still above Meridian's 18% hurdle rate.
        """),
        vote_for="Jennifer Park (M&A Lead Counsel)",
        critique_issues=["IP training data risk may expand beyond current DMCA claims",
                         "Unlimited liability contracts could deter future acquirers"],
    )

    ip_specialist = create_agent(
        "mock",
        name="Robert Liu (IP & Technology Counsel)",
        proposal=textwrap.dedent("""\
            CONDITIONAL PROCEED — IP risks require pre-closing remediation, not just escrow.

            Critical IP analysis:

            1. Training data liability (SEVERE):
               - Post-NYT v. OpenAI (2023), scraping without license creates copyright
                 infringement exposure even for ML training (fair use defense uncertain)
               - 30% unlicensed data = substantial portion of model capability
               - Potential damages: statutory $150K per work × estimated 500+ works = $75M
                 theoretical maximum exposure (realistic settlement range: $2-8M)
               - DMCA takedowns signal active enforcement posture by rights holders
               - Recommendation: Commission independent IP audit of training pipeline,
                 begin retroactive licensing negotiations BEFORE closing

            2. Trade secret exposure (MODERATE-HIGH):
               - DocStream C&D is not frivolous — 3 engineers without clearance is a pattern
               - Under DTSA (18 U.S.C. §1836), injunctive relief could force code quarantine
               - Clean-room analysis required: identify which code contributions overlap
                 with DocStream IP using independent forensic review ($150K, 6-8 weeks)
               - Failure to conduct diligence on this pre-close could constitute willful
                 misappropriation post-close, tripling damages

            3. Patent risk:
               - No freedom-to-operate analysis conducted
               - Document management SaaS space has ~2,400 active patents
               - Recommend FTO search ($80K, 4 weeks) before closing

            Total pre-closing diligence budget: $230K, 8-week timeline.
            Deal should not close without items 1 and 2 above completed.
        """),
        vote_for="Robert Liu (IP & Technology Counsel)",
        critique_issues=["Pre-closing remediation adds timeline risk",
                         "Escrow alone may not cover IP claims that mature post-close"],
    )

    privacy_counsel = create_agent(
        "mock",
        name="Elena Rossi (Data Privacy & Regulatory)",
        proposal=textwrap.dedent("""\
            CONDITIONAL PROCEED — GDPR non-compliance is a regulatory time bomb.

            Privacy and regulatory analysis:

            1. GDPR cross-border transfer (CRITICAL, pre-closing blocker):
               - Processing EU data without SCCs violates GDPR Art. 46
               - Post-Schrems II, this is actively enforced — fines up to 4% of global revenue
               - 340 enterprise customers likely include EU data subjects
               - REMEDIATION REQUIRED BEFORE CLOSING:
                 a. Execute SCCs with all sub-processors (60% gap)
                 b. Conduct Transfer Impact Assessment (TIA) per EDPB guidance
                 c. Implement supplementary measures (encryption in transit + at rest)
               - Timeline: 6-8 weeks minimum for SCC execution alone
               - If we close without this, Meridian inherits the violation and liability

            2. Sub-processor DPA gaps (HIGH):
               - 40% of sub-processors without DPAs = GDPR Art. 28 violation
               - Each gap is an independent enforcement action risk
               - Meridian's existing DPA framework can be extended post-close (4 weeks)

            3. SOC 2 gap (MODERATE):
               - Not a legal requirement but contractual: 22% of customer MSAs require it
               - 4 critical pentest findings must be disclosed to affected customers
               - FedRAMP absence blocks 15% of pipeline — quantify revenue impact

            4. Uncapped liability contracts (MODERATE):
               - 137 contracts with unlimited consequential damages
               - Recommended: negotiate liability caps during standard renewal cycle
               - 18-month migration plan reduces exposure by 70% (est.)

            Budget impact: $1.2M in privacy remediation over 12 months.
            Closing should be conditioned on SCC execution (Item 1a above).
        """),
        vote_for="Robert Liu (IP & Technology Counsel)",
        critique_issues=["Privacy remediation is expensive but predictable",
                         "Regulatory risk is binary — either compliant or exposed"],
    )

    # ── Run the debate ─────────────────────────────────────────────────
    debate = Debate(
        topic="DataVault acquisition: Should Meridian proceed, renegotiate, or walk away?",
        context=scenario,
        rounds=2,
        consensus="majority",
        enable_trickster=True,
        trickster_sensitivity=0.5,
    )

    debate.add_agent(ma_counsel)
    debate.add_agent(ip_specialist)
    debate.add_agent(privacy_counsel)

    result = asyncio.run(debate.run())

    # ── Format legal decision receipt ──────────────────────────────────
    from aragora_debate.receipt import ReceiptBuilder

    receipt = result.receipt
    print("=" * 72)
    print("  LEGAL DUE DILIGENCE DECISION RECEIPT")
    print("  Aragora Decision Integrity Platform — Legal Vertical")
    print("=" * 72)
    print()
    print(f"  Decision ID:    {receipt.receipt_id}")
    print(f"  Transaction:    DataVault SaaS Inc. acquisition")
    print(f"  Verdict:        {receipt.verdict.value if receipt.verdict else 'N/A'}")
    print(f"  Confidence:     {receipt.confidence:.0%}")
    print(f"  Consensus:      {'Reached' if receipt.consensus.reached else 'Not reached'} ({receipt.consensus.method.value})")
    print(f"  Counsel:        {len(receipt.agents)}")
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
            print(f"  Counsel:  {dissent.agent}")
            if dissent.reasons:
                print(f"  Reason:   {dissent.reasons[0][:80]}")
            if dissent.alternative_view:
                print(f"  Alt:      {dissent.alternative_view[:80]}")
            print()

    # Integrity
    print("  AUDIT TRAIL")
    print("  " + "-" * 50)
    ReceiptBuilder.sign_hmac(receipt, "legal-dd-committee-key-2025")
    print(f"  HMAC-SHA256:    {receipt.signature[:40]}...")
    print(f"  Timestamp:      {receipt.timestamp}")
    print(f"  Tamper-proof:   {ReceiptBuilder.verify_hmac(receipt, 'legal-dd-committee-key-2025')}")
    print()

    # Legal privilege notes
    print("  PRIVILEGE & CONFIDENTIALITY NOTES")
    print("  " + "-" * 50)
    print("  - ATTORNEY-CLIENT PRIVILEGED AND CONFIDENTIAL")
    print("  - Prepared at direction of counsel for legal advice purposes")
    print("  - Receipt encrypted at rest (AES-256-GCM)")
    print("  - Access restricted to deal team (RBAC enforced)")
    print("  - Retention per firm document retention policy (10 years)")
    print("  - Do not forward without General Counsel approval")
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
