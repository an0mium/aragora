#!/usr/bin/env python3
"""Decision Pipeline Demo: Debate -> Receipt -> Audit -> Attestation.

Demonstrates the full decision integrity pipeline:
1. Run an adversarial debate on a business decision
2. Generate a tamper-evident decision receipt
3. Export receipt in multiple formats (JSON, Markdown, HTML)
4. Show ERC-8004 attestation anchor (mock blockchain for demo)
5. Verify receipt integrity (tamper detection)

Usage:
    python examples/decision_pipeline.py --demo    # Mock mode (no API keys)
    python examples/decision_pipeline.py           # Same as --demo
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# =============================================================================
# Mock Debate Engine
# =============================================================================


@dataclass
class MockProposal:
    agent: str
    content: str
    round: int


@dataclass
class MockVote:
    agent: str
    choice: str
    confidence: float
    reasoning: str


@dataclass
class MockDebateResult:
    question: str
    verdict: str
    confidence: float
    rounds_completed: int
    proposals: list[MockProposal]
    votes: list[MockVote]
    dissent: list[dict[str, str]]
    agents: list[str]
    total_tokens: int
    debate_time_ms: float


def _run_mock_debate(question: str) -> MockDebateResult:
    """Simulate a 2-round adversarial debate."""
    agents = ["claude-analyst", "gpt4-challenger", "mistral-synthesizer"]

    proposals = [
        MockProposal(
            agent="claude-analyst",
            round=1,
            content=(
                "I recommend proceeding with the migration. The cost savings from "
                "reduced licensing fees (estimated $240K/year) outweigh the migration "
                "risk. Key mitigation: run parallel systems for 90 days."
            ),
        ),
        MockProposal(
            agent="gpt4-challenger",
            round=1,
            content=(
                "I challenge the timeline. Historical data shows 78% of similar "
                "migrations exceed their estimated timeline by 40-60%. Recommend "
                "a phased approach: migrate read replicas first, then write path."
            ),
        ),
        MockProposal(
            agent="mistral-synthesizer",
            round=2,
            content=(
                "Synthesizing both perspectives: proceed with migration using the "
                "phased approach (read replicas first). This captures the cost savings "
                "while managing the timeline risk. Set a hard 120-day gate."
            ),
        ),
    ]

    votes = [
        MockVote("claude-analyst", "mistral-synthesizer", 0.85,
                 "Phased approach addresses my concern about parallel costs."),
        MockVote("gpt4-challenger", "mistral-synthesizer", 0.78,
                 "Phased migration reduces blast radius, though 120 days is tight."),
        MockVote("mistral-synthesizer", "mistral-synthesizer", 0.92,
                 "Consensus on phased approach with hard gate."),
    ]

    dissent = [
        {
            "agent": "gpt4-challenger",
            "reason": "120-day timeline may be too aggressive; recommend 150 days.",
        },
    ]

    return MockDebateResult(
        question=question,
        verdict="Proceed with phased migration (read replicas first, 120-day gate)",
        confidence=0.85,
        rounds_completed=2,
        proposals=proposals,
        votes=votes,
        dissent=dissent,
        agents=agents,
        total_tokens=4250,
        debate_time_ms=3200.0,
    )


# =============================================================================
# Receipt Generation
# =============================================================================


def _generate_receipt(debate: MockDebateResult) -> dict[str, Any]:
    """Generate a tamper-evident decision receipt from debate results."""
    receipt = {
        "receipt_id": f"rcpt_{int(time.time())}",
        "version": "1.0.0",
        "question": debate.question,
        "verdict": debate.verdict,
        "confidence": debate.confidence,
        "agents": debate.agents,
        "rounds": debate.rounds_completed,
        "total_tokens": debate.total_tokens,
        "debate_time_ms": debate.debate_time_ms,
        "dissent": debate.dissent,
        "vote_summary": {
            v.agent: {"choice": v.choice, "confidence": v.confidence}
            for v in debate.votes
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute content hash for tamper detection
    content_str = json.dumps(receipt, sort_keys=True, default=str)
    receipt["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()

    return receipt


def _sign_receipt(receipt: dict[str, Any], secret: str) -> dict[str, Any]:
    """Add HMAC-SHA256 signature for integrity verification."""
    content = json.dumps(
        {k: v for k, v in receipt.items() if k != "hmac_signature"},
        sort_keys=True,
        default=str,
    )
    signature = hmac.new(secret.encode(), content.encode(), hashlib.sha256).hexdigest()
    receipt["hmac_signature"] = signature
    return receipt


def _verify_receipt(receipt: dict[str, Any], secret: str) -> bool:
    """Verify receipt HMAC signature. Returns True if valid."""
    stored_sig = receipt.get("hmac_signature")
    if not stored_sig:
        return False

    content = json.dumps(
        {k: v for k, v in receipt.items() if k != "hmac_signature"},
        sort_keys=True,
        default=str,
    )
    expected = hmac.new(secret.encode(), content.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(stored_sig, expected)


# =============================================================================
# Export Formats
# =============================================================================


def _export_json(receipt: dict[str, Any]) -> str:
    return json.dumps(receipt, indent=2, default=str)


def _export_markdown(receipt: dict[str, Any]) -> str:
    lines = [
        f"# Decision Receipt: {receipt['receipt_id']}",
        "",
        f"**Question:** {receipt['question']}",
        "",
        f"**Verdict:** {receipt['verdict']}",
        "",
        f"**Confidence:** {receipt['confidence']:.0%}",
        "",
        f"**Agents:** {', '.join(receipt['agents'])}",
        "",
        f"**Rounds:** {receipt['rounds']} | **Tokens:** {receipt['total_tokens']}",
        "",
        "## Vote Summary",
        "",
        "| Agent | Choice | Confidence |",
        "|-------|--------|------------|",
    ]
    for agent, vote in receipt.get("vote_summary", {}).items():
        lines.append(f"| {agent} | {vote['choice']} | {vote['confidence']:.0%} |")

    lines.extend(["", "## Dissent", ""])
    for d in receipt.get("dissent", []):
        lines.append(f"- **{d['agent']}:** {d['reason']}")

    lines.extend([
        "",
        "## Integrity",
        "",
        f"- Content hash: `{receipt.get('content_hash', 'N/A')}`",
        f"- HMAC signed: {'Yes' if receipt.get('hmac_signature') else 'No'}",
        f"- Generated: {receipt.get('generated_at', 'N/A')}",
    ])

    return "\n".join(lines)


def _export_html(receipt: dict[str, Any]) -> str:
    votes_html = ""
    for agent, vote in receipt.get("vote_summary", {}).items():
        votes_html += f"<tr><td>{agent}</td><td>{vote['choice']}</td><td>{vote['confidence']:.0%}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html><head><title>Decision Receipt {receipt['receipt_id']}</title>
<style>
body {{ font-family: system-ui; max-width: 800px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.verdict {{ font-size: 1.2em; color: #2563eb; font-weight: bold; }}
.hash {{ font-family: monospace; font-size: 0.85em; color: #666; }}
</style></head>
<body>
<h1>Decision Receipt</h1>
<p><strong>ID:</strong> {receipt['receipt_id']}</p>
<p><strong>Question:</strong> {receipt['question']}</p>
<p class="verdict">Verdict: {receipt['verdict']}</p>
<p><strong>Confidence:</strong> {receipt['confidence']:.0%}</p>
<p><strong>Agents:</strong> {', '.join(receipt['agents'])}</p>

<h2>Vote Summary</h2>
<table><tr><th>Agent</th><th>Choice</th><th>Confidence</th></tr>
{votes_html}</table>

<h2>Integrity</h2>
<p class="hash">SHA-256: {receipt.get('content_hash', 'N/A')}</p>
<p class="hash">HMAC signed: {'Yes' if receipt.get('hmac_signature') else 'No'}</p>
</body></html>"""


# =============================================================================
# ERC-8004 Mock Attestation
# =============================================================================


def _mock_erc8004_attestation(receipt: dict[str, Any]) -> dict[str, Any]:
    """Simulate anchoring the receipt hash on-chain via ERC-8004."""
    content_hash = receipt.get("content_hash", "")
    # Simulate a transaction hash
    tx_data = f"erc8004:attest:{content_hash}:{receipt['receipt_id']}"
    tx_hash = "0x" + hashlib.sha256(tx_data.encode()).hexdigest()

    return {
        "protocol": "ERC-8004",
        "chain": "ethereum-sepolia (testnet)",
        "contract": "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD0E",
        "transaction_hash": tx_hash,
        "content_hash": content_hash,
        "receipt_id": receipt["receipt_id"],
        "attestation_type": "decision_receipt",
        "block_number": 19_847_223,  # Mock
        "gas_used": 48_521,
        "status": "confirmed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Demo Runner
# =============================================================================


def _print_step(num: int, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Step {num}: {title}")
    print(f"{'─' * 60}")


def run_demo() -> None:
    """Run the full decision pipeline demo."""
    signing_secret = "demo-secret-key-do-not-use-in-production"

    print("=" * 60)
    print("  Decision Pipeline Demo")
    print("  Debate -> Receipt -> Audit -> Attestation")
    print("=" * 60)

    # Step 1: Run adversarial debate
    _print_step(1, "Run Adversarial Debate")
    question = "Should we migrate our primary database from Oracle to PostgreSQL?"
    print(f"  Question: {question}")
    print("  Agents: claude-analyst, gpt4-challenger, mistral-synthesizer")
    print("  Running 2-round debate...")

    debate = _run_mock_debate(question)

    print(f"\n  Verdict: {debate.verdict}")
    print(f"  Confidence: {debate.confidence:.0%}")
    print(f"  Tokens used: {debate.total_tokens}")
    print(f"  Time: {debate.debate_time_ms:.0f}ms")

    # Step 2: Generate signed receipt
    _print_step(2, "Generate Decision Receipt")
    receipt = _generate_receipt(debate)
    receipt = _sign_receipt(receipt, signing_secret)

    print(f"  Receipt ID: {receipt['receipt_id']}")
    print(f"  Content hash: {receipt['content_hash'][:32]}...")
    print(f"  HMAC signature: {receipt['hmac_signature'][:32]}...")

    # Step 3: Export in multiple formats
    _print_step(3, "Export Receipt (JSON / Markdown / HTML)")

    json_output = _export_json(receipt)
    md_output = _export_markdown(receipt)
    html_output = _export_html(receipt)

    print(f"  JSON: {len(json_output)} bytes")
    print(f"  Markdown: {len(md_output)} bytes")
    print(f"  HTML: {len(html_output)} bytes")
    print("\n  --- Markdown Preview ---")
    for line in md_output.split("\n")[:15]:
        print(f"  {line}")
    print("  ...")

    # Step 4: ERC-8004 attestation
    _print_step(4, "ERC-8004 On-Chain Attestation (mock)")
    attestation = _mock_erc8004_attestation(receipt)

    print(f"  Protocol: {attestation['protocol']}")
    print(f"  Chain: {attestation['chain']}")
    print(f"  Contract: {attestation['contract']}")
    print(f"  TX hash: {attestation['transaction_hash'][:42]}...")
    print(f"  Block: {attestation['block_number']}")
    print(f"  Gas: {attestation['gas_used']}")
    print(f"  Status: {attestation['status']}")

    # Step 5: Verify receipt integrity
    _print_step(5, "Verify Receipt Integrity")

    # Valid verification
    is_valid = _verify_receipt(receipt, signing_secret)
    print(f"  Original receipt valid: {is_valid}")

    # Tamper detection
    tampered = dict(receipt)
    tampered["verdict"] = "Do not migrate"
    is_tampered = _verify_receipt(tampered, signing_secret)
    print(f"  Tampered receipt valid: {is_tampered}  (tamper detected!)")

    # Wrong key
    wrong_key = _verify_receipt(receipt, "wrong-secret")
    print(f"  Wrong signing key:     {wrong_key}  (key mismatch detected!)")

    # Summary
    print(f"\n{'=' * 60}")
    print("  Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"\n  Debate:       {debate.rounds_completed} rounds, {len(debate.agents)} agents")
    print(f"  Receipt:      {receipt['receipt_id']}")
    print(f"  Attestation:  {attestation['transaction_hash'][:24]}...")
    print(f"  Integrity:    SHA-256 + HMAC-SHA256")
    print(f"  Dissent:      {len(debate.dissent)} dissenting opinion(s)")

    print("\nTo run with live agents:")
    print("  aragora decide 'Your question' --agents anthropic-api,openai-api")
    print("  aragora plans show <plan_id>")
    print("  aragora verify receipt.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decision Pipeline Demo: Debate -> Receipt -> Audit -> Attestation",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Use mock debate engine (default: True)",
    )
    args = parser.parse_args()
    run_demo()


if __name__ == "__main__":
    main()
