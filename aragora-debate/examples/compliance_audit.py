#!/usr/bin/env python3
"""Example: Compliance audit debate — SOC 2 gap analysis.

Four AI agents with different specializations assess whether a SaaS application
meets SOC 2 Type II controls, producing a decision receipt suitable for
sharing with auditors.

Usage:
    python examples/compliance_audit.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate import (
    Agent,
    Arena,
    Critique,
    DebateConfig,
    Message,
    ReceiptBuilder,
    Vote,
)
from aragora_debate.types import ConsensusMethod, Evidence, Claim


# ---------------------------------------------------------------------------
# Specialized compliance agents
# ---------------------------------------------------------------------------

SYSTEM_DESCRIPTION = """\
SaaS Application Security Posture:
- Authentication: OAuth 2.0 + TOTP MFA (optional, not enforced)
- Encryption: TLS 1.3 in transit, AES-256-GCM at rest
- Access control: RBAC with 5 predefined roles
- Logging: Application logs to CloudWatch, 30-day retention
- Backups: Daily automated, 7-day retention, no tested restore procedure
- Incident response: Documented runbook, no tabletop exercises conducted
- Vendor management: 3 sub-processors, 1 without current SOC 2 report
- Change management: PR reviews required, no formal CAB process
"""


class ComplianceAgent(Agent):
    """Mock compliance specialist agent."""

    def __init__(self, name: str, specialty: str, findings: dict[str, str]) -> None:
        super().__init__(name, stance="neutral")
        self.specialty = specialty
        self.findings = findings

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        round_num = 1 + (max((m.round for m in context), default=0) if context else 0)
        if round_num == 1:
            return self.findings.get("initial", "No findings.")
        return self.findings.get("revised", self.findings.get("initial", "No additional findings."))

    async def critique(self, proposal: str, task: str, **kw) -> Critique:
        target = kw.get("target_agent", "unknown")
        critique_data = self.findings.get("critique", {})
        if isinstance(critique_data, dict) and target in critique_data:
            issues = critique_data[target]
        else:
            issues = ["Analysis could be more specific about remediation timelines"]
        return Critique(
            agent=self.name, target_agent=target, target_content=proposal,
            issues=issues if isinstance(issues, list) else [issues],
            severity=5.0,
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        vote_data = self.findings.get("vote", {})
        return Vote(
            agent=self.name,
            choice=vote_data.get("choice", self.name),
            reasoning=vote_data.get("reasoning", "Based on my domain analysis"),
            confidence=vote_data.get("confidence", 0.7),
        )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

security_analyst = ComplianceAgent(
    name="security-analyst",
    specialty="Technical security controls",
    findings={
        "initial": (
            "## Technical Security Assessment\n\n"
            "**CC6.1 (Logical Access):** PARTIAL COMPLIANCE\n"
            "- RBAC is implemented but MFA is optional. SOC 2 CC6.1 requires "
            "MFA for all administrative access. This is a **finding**.\n"
            "- 5 predefined roles may be insufficient — need to verify least-privilege.\n\n"
            "**CC6.7 (Encryption):** COMPLIANT\n"
            "- TLS 1.3 + AES-256-GCM meets requirements.\n\n"
            "**CC7.2 (System Monitoring):** PARTIAL COMPLIANCE\n"
            "- 30-day log retention is below the 1-year minimum recommended by "
            "AICPA for SOC 2. Security events should be retained for at least "
            "90 days minimum, 1 year recommended.\n\n"
            "**Overall: CONDITIONAL PASS** — 2 findings require remediation."
        ),
        "revised": (
            "After reviewing other assessments, I want to add:\n\n"
            "The audit-analyst's point about backup testing is valid and I'm "
            "upgrading it from observation to finding. An untested backup is "
            "not a backup — it's a hope.\n\n"
            "**Updated assessment: 3 findings, CONDITIONAL PASS.**"
        ),
        "critique": {
            "audit-analyst": ["Good catch on backup testing, but you missed the MFA gap"],
            "risk-assessor": ["Risk quantification is useful but SOC 2 auditors want control-level assessment"],
            "vendor-manager": ["Sub-processor gap is critical — should be elevated to finding"],
        },
        "vote": {"choice": "security-analyst", "reasoning": "Control-level analysis most relevant for SOC 2", "confidence": 0.75},
    },
)

audit_analyst = ComplianceAgent(
    name="audit-analyst",
    specialty="Audit procedures and evidence",
    findings={
        "initial": (
            "## Audit Readiness Assessment\n\n"
            "**A1.2 (Recovery):** FINDING\n"
            "- Backups exist but restore has never been tested. SOC 2 A1.2 "
            "requires demonstrated recovery capability. Without a documented "
            "restore test, this fails the control.\n\n"
            "**CC8.1 (Change Management):** OBSERVATION\n"
            "- PR reviews provide some change control, but no formal Change "
            "Advisory Board (CAB) process exists. Small teams often skip CAB, "
            "but auditors may flag this.\n\n"
            "**CC9.1 (Risk Assessment):** OBSERVATION\n"
            "- No evidence of formal risk assessment process. SOC 2 requires "
            "annual risk assessment.\n\n"
            "**Evidence gaps:** I cannot find evidence of:\n"
            "- Annual access reviews\n"
            "- Security awareness training records\n"
            "- Penetration test results\n\n"
            "**Overall: NOT READY** — Too many evidence gaps for Type II."
        ),
        "critique": {
            "security-analyst": ["Technical controls look good but you need to address the evidence gaps I identified"],
            "risk-assessor": ["Your risk matrix should map to specific SOC 2 trust service criteria"],
            "vendor-manager": ["Vendor risk is important but I'd prioritize the evidence gaps first"],
        },
        "vote": {"choice": "audit-analyst", "reasoning": "Evidence gaps are the biggest blocker for Type II", "confidence": 0.8},
    },
)

risk_assessor = ComplianceAgent(
    name="risk-assessor",
    specialty="Risk quantification",
    findings={
        "initial": (
            "## Risk Assessment\n\n"
            "**High Risk:**\n"
            "1. MFA not enforced → Credential stuffing attacks (Impact: HIGH, "
            "Likelihood: MEDIUM). Estimated annual loss expectancy: $150K-500K.\n"
            "2. Untested backups → Extended outage in disaster scenario "
            "(Impact: CRITICAL, Likelihood: LOW). Worst case: 48-72hr recovery.\n\n"
            "**Medium Risk:**\n"
            "3. 30-day log retention → Inability to investigate historical "
            "incidents (Impact: MEDIUM, Likelihood: MEDIUM).\n"
            "4. Sub-processor without SOC 2 → Supply chain risk "
            "(Impact: MEDIUM, Likelihood: LOW).\n\n"
            "**Risk Score: 7.2/10** — Above acceptable threshold of 5.0.\n\n"
            "**Recommendation:** Address items 1 & 2 before audit engagement. "
            "Items 3 & 4 can be in-progress with documented remediation plans."
        ),
        "critique": {
            "security-analyst": ["Good technical depth but missing risk quantification — how much does each gap cost?"],
            "audit-analyst": ["Evidence gaps are important but need to be prioritized by risk impact"],
            "vendor-manager": ["Vendor risk score should factor into the overall risk calculation"],
        },
        "vote": {"choice": "audit-analyst", "reasoning": "Evidence gaps represent the highest audit risk", "confidence": 0.7},
    },
)

vendor_manager = ComplianceAgent(
    name="vendor-manager",
    specialty="Third-party risk management",
    findings={
        "initial": (
            "## Vendor & Sub-Processor Assessment\n\n"
            "**CC9.2 (Vendor Management):** FINDING\n"
            "- 1 of 3 sub-processors lacks a current SOC 2 report. Per AICPA "
            "guidance, the service organization must either:\n"
            "  a) Obtain the sub-processor's SOC 2 report, or\n"
            "  b) Perform due diligence and document complementary controls.\n\n"
            "Neither has been done. This will result in a qualified opinion "
            "or scope limitation if not addressed.\n\n"
            "**Remediation options:**\n"
            "1. Request SOC 2 report from sub-processor (2-4 weeks)\n"
            "2. Conduct independent security assessment (1-2 weeks)\n"
            "3. Replace sub-processor with SOC 2-compliant alternative\n\n"
            "**Overall: This single gap could block the entire audit.**"
        ),
        "critique": {
            "security-analyst": ["Technical controls are important but you missed the vendor risk"],
            "audit-analyst": ["Agree on evidence gaps — the vendor gap is one of the worst"],
            "risk-assessor": ["Your risk score should weight the vendor gap more heavily"],
        },
        "vote": {"choice": "audit-analyst", "reasoning": "Comprehensive view of readiness gaps", "confidence": 0.65},
    },
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    agents = [security_analyst, audit_analyst, risk_assessor, vendor_manager]

    arena = Arena(
        question=(
            "Is our SaaS application ready for a SOC 2 Type II audit? "
            "Identify all gaps, prioritize them, and recommend whether to "
            "proceed with the audit engagement or remediate first."
        ),
        agents=agents,
        config=DebateConfig(
            rounds=2,
            consensus_method=ConsensusMethod.SUPERMAJORITY,
            consensus_threshold=0.6,
        ),
        context=SYSTEM_DESCRIPTION,
    )

    print("=" * 60)
    print("COMPLIANCE AUDIT: SOC 2 Type II Readiness")
    print("=" * 60)
    print()

    result = await arena.run()

    # Print structured summary
    print(f"\nStatus: {result.status}")
    print(f"Consensus: {'Reached' if result.consensus_reached else 'Not reached'}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Rounds: {result.rounds_used}")
    print()

    # Print each agent's position
    for name, proposal in result.proposals.items():
        print(f"--- {name} ---")
        print(proposal[:300])
        print()

    # Print dissent
    if result.dissenting_views:
        print("DISSENTING VIEWS:")
        for dv in result.dissenting_views:
            print(f"  - {dv}")
        print()

    # Decision receipt
    print("=" * 60)
    print("DECISION RECEIPT")
    print("=" * 60)
    assert result.receipt is not None
    print(result.receipt.to_markdown())

    # Export HTML for auditors
    html = ReceiptBuilder.to_html(result.receipt)
    html_path = os.path.join(os.path.dirname(__file__), "soc2_receipt.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML receipt exported to: {html_path}")


if __name__ == "__main__":
    asyncio.run(main())
