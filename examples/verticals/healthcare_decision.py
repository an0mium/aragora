#!/usr/bin/env python3
"""Healthcare vertical example: Clinical drug selection debate.

Demonstrates aragora-debate for a realistic clinical decision scenario
where multiple medical specialists debate drug selection for a patient
with comorbidities.

Usage:
    python examples/verticals/healthcare_decision.py

No API keys required — uses mock agents with realistic medical responses.
"""

from __future__ import annotations

import asyncio
import textwrap

from aragora_debate import Debate, DebateConfig, create_agent


def main() -> None:
    """Run a clinical drug selection debate."""

    # ── Clinical scenario ──────────────────────────────────────────────
    # NOTE: No real PHI. In production, patient data would be anonymized
    # via aragora.privacy.HIPAAAnonymizer before entering the debate.
    scenario = textwrap.dedent("""\
        Patient: 67-year-old male (ID: ANON-4821)
        Chief complaint: Persistent atrial fibrillation (AFib) with rapid ventricular response
        Comorbidities: Type 2 diabetes (HbA1c 7.8%), Stage 3a CKD (eGFR 52), mild hepatic steatosis
        Current medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily
        Question: Should we initiate rate control with Metoprolol or Diltiazem?

        Key considerations:
        - CKD requires dose adjustment for renally-cleared drugs
        - Beta-blocker vs calcium channel blocker efficacy in diabetic patients
        - Drug interaction profile with existing medications
        - Risk of hypoglycemia masking (beta-blockers) vs peripheral edema (CCBs)
    """)

    # ── Configure specialist agents ────────────────────────────────────
    cardiologist = create_agent(
        "mock",
        name="Dr. Chen (Cardiology)",
        proposal=textwrap.dedent("""\
            I recommend Metoprolol succinate 25mg daily, titrated to 50-100mg based on
            heart rate response.

            Evidence basis (AHA/ACC 2023 AFib Guidelines, Class I):
            - Beta-blockers are first-line for rate control in AFib
            - Metoprolol succinate (extended-release) provides stable 24-hour coverage
            - Target resting HR <110 bpm per RACE II trial (lenient strategy appropriate here)

            Addressing comorbidities:
            - CKD Stage 3a: Metoprolol is hepatically metabolized — no dose adjustment needed
            - Diabetes: While beta-blockers can mask hypoglycemia symptoms, metoprolol
              is beta-1 selective, minimizing this risk vs non-selective agents
            - Cardioprotective benefit: reduces cardiovascular mortality in diabetic patients
              (MERIT-HF subgroup analysis, HR 0.64)

            Drug interactions: No significant interaction with metformin, lisinopril, or atorvastatin.
            Monitoring: ECG at 2 weeks, renal function at 4 weeks, HbA1c at 3 months.
        """),
        vote_for="Dr. Chen (Cardiology)",
        critique_issues=["Diltiazem has faster onset for acute rate control",
                         "CYP3A4 interaction with atorvastatin is manageable"],
    )

    pharmacologist = create_agent(
        "mock",
        name="Dr. Patel (Clinical Pharmacology)",
        proposal=textwrap.dedent("""\
            I recommend Diltiazem extended-release 120mg daily as the preferred agent
            for this specific patient profile.

            Pharmacokinetic rationale:
            - Diltiazem is 77% hepatically metabolized — better choice with eGFR 52
            - Avoids beta-blocker hypoglycemia masking entirely (clinically significant
              in poorly controlled T2DM with HbA1c 7.8%)
            - Comparable rate control efficacy: RATAF trial showed no difference in
              resting HR between diltiazem and metoprolol at 12 months

            Key concern — CYP3A4 interaction:
            - Diltiazem inhibits CYP3A4, which metabolizes atorvastatin
            - Expected ~2.5x increase in atorvastatin AUC
            - Mitigation: reduce atorvastatin from 40mg to 20mg (equivalent LDL reduction
              with elevated statin levels)
            - Alternative: switch to rosuvastatin (minimal CYP3A4 metabolism)

            Monitoring: LFTs at 2 and 6 weeks, lipid panel at 8 weeks, ECG at 2 weeks.
        """),
        vote_for="Dr. Patel (Clinical Pharmacology)",
        critique_issues=["Drug interaction with atorvastatin requires medication change",
                         "Peripheral edema risk may compound with diabetes-related edema"],
    )

    safety_officer = create_agent(
        "mock",
        name="Dr. Williams (Patient Safety)",
        proposal=textwrap.dedent("""\
            From a patient safety perspective, I favor Metoprolol with enhanced monitoring.

            Risk-benefit analysis:
            1. Hypoglycemia masking (Metoprolol): MODERATE risk
               - Beta-1 selective agents have 60% lower masking incidence vs non-selective
               - Patient has HbA1c 7.8% — moderate risk, not on insulin/sulfonylureas
               - Mitigation: CGM referral, patient education on non-adrenergic hypo symptoms

            2. Drug interaction (Diltiazem + Atorvastatin): HIGH risk
               - Statin dose adjustment introduces new risk of under-treatment or myopathy
               - Medication changes increase cognitive load and non-adherence risk
               - FDA safety communication 2012: diltiazem + high-dose statins contraindicated

            3. CKD progression: EQUIVALENT
               - Neither agent has proven renal benefit in AFib
               - ACEi (lisinopril) is providing renal protection

            4. Fall risk: Diltiazem causes more orthostatic hypotension in elderly (OR 1.4)

            Recommendation: Metoprolol 25mg with CGM referral, titrate slowly with
            orthostatic vital signs. Avoids polypharmacy cascade from statin adjustment.
        """),
        vote_for="Dr. Chen (Cardiology)",
        critique_issues=["Both options require careful monitoring in this population"],
    )

    # ── Run the debate ─────────────────────────────────────────────────
    debate = Debate(
        topic="Rate control agent selection: Metoprolol vs Diltiazem for AFib with T2DM and CKD",
        context=scenario,
        rounds=2,
        consensus="majority",
        enable_trickster=True,
        trickster_sensitivity=0.6,
    )

    debate.add_agent(cardiologist)
    debate.add_agent(pharmacologist)
    debate.add_agent(safety_officer)

    result = asyncio.run(debate.run())

    # ── Format clinical decision receipt ───────────────────────────────
    from aragora_debate.receipt import ReceiptBuilder

    receipt = result.receipt
    print("=" * 72)
    print("  CLINICAL DECISION RECEIPT")
    print("  Aragora Decision Integrity Platform — Healthcare Vertical")
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
    ReceiptBuilder.sign_hmac(receipt, "clinical-decision-key-2024")
    print(f"  HMAC-SHA256:    {receipt.signature[:40]}...")
    print(f"  Timestamp:      {receipt.timestamp}")
    print(f"  Tamper-proof:   {ReceiptBuilder.verify_hmac(receipt, 'clinical-decision-key-2024')}")
    print()

    # HIPAA compliance notes
    print("  HIPAA COMPLIANCE NOTES")
    print("  " + "-" * 50)
    print("  - Patient identified by anonymized ID (ANON-4821)")
    print("  - No PHI in debate transcript (pre-anonymized)")
    print("  - Receipt stored with AES-256-GCM encryption at rest")
    print("  - Access logged via aragora.audit with user/role context")
    print("  - Retention policy: 7 years per HIPAA §164.530(j)")
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
