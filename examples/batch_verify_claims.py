#!/usr/bin/env python3
"""
Batch Claim Verification -- Run multiple debates to fact-check claims.

Given a list of claims, this script runs a debate for each one and
produces a verification report with confidence scores and receipts.

No API keys needed -- works offline with mock agents.

Usage:
    python examples/batch_verify_claims.py
    python examples/batch_verify_claims.py --claims "Water boils at 100C" "The sky is green"
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora_debate import Debate, ReceiptBuilder, create_agent


@dataclass
class VerificationResult:
    claim: str
    verdict: str  # "supported", "disputed", "uncertain"
    confidence: float
    receipt_id: str
    reasoning: str


SAMPLE_CLAIMS = [
    "Token bucket is the best algorithm for rate limiting at scale",
    "Microservices always outperform monoliths",
    "PostgreSQL is ACID-compliant",
    "Python is faster than C for numerical computing",
    "JWT tokens should be stored in localStorage",
]


async def verify_claim(claim: str) -> VerificationResult:
    """Run a debate to verify a single claim."""
    debate = Debate(
        topic=f"Evaluate this claim: '{claim}'. Is it accurate, misleading, or false?",
        rounds=2,
        consensus="majority",
    )

    debate.add_agent(
        create_agent(
            "mock",
            name="fact-checker",
            proposal=(
                f"Analyzing the claim: '{claim}'. Let me evaluate the evidence "
                "and identify any nuances or conditions that affect accuracy."
            ),
        )
    )
    debate.add_agent(
        create_agent(
            "mock",
            name="skeptic",
            proposal=(
                f"I'll challenge the claim: '{claim}'. What assumptions does "
                "it make? Are there counterexamples or missing context?"
            ),
        )
    )
    debate.add_agent(
        create_agent(
            "mock",
            name="synthesizer",
            proposal=(
                "Synthesizing both perspectives to reach a balanced verdict "
                "with appropriate confidence level and caveats."
            ),
        )
    )

    result = await debate.run()

    # Determine verdict from consensus
    if result.consensus_reached and result.confidence > 0.7:
        verdict = "supported"
    elif result.confidence < 0.4:
        verdict = "disputed"
    else:
        verdict = "uncertain"

    receipt_id = ""
    if result.receipt:
        ReceiptBuilder.sign_hmac(result.receipt, key="batch-verify-key")
        receipt_id = result.receipt.receipt_id

    return VerificationResult(
        claim=claim,
        verdict=verdict,
        confidence=result.confidence,
        receipt_id=receipt_id,
        reasoning=result.final_answer[:300] if result.final_answer else "",
    )


async def batch_verify(claims: list[str]) -> list[VerificationResult]:
    """Verify multiple claims concurrently."""
    tasks = [verify_claim(claim) for claim in claims]
    return await asyncio.gather(*tasks)


def print_report(results: list[VerificationResult]) -> None:
    """Print a human-readable verification report."""
    icons = {"supported": "+", "disputed": "x", "uncertain": "?"}

    print("=" * 60)
    print("CLAIM VERIFICATION REPORT")
    print("=" * 60)

    for r in results:
        icon = icons.get(r.verdict, "?")
        print(f"\n[{icon}] {r.claim}")
        print(f"    Verdict: {r.verdict.upper()} ({r.confidence:.0%} confidence)")
        if r.receipt_id:
            print(f"    Receipt: {r.receipt_id[:16]}...")

    # Summary
    supported = sum(1 for r in results if r.verdict == "supported")
    disputed = sum(1 for r in results if r.verdict == "disputed")
    uncertain = sum(1 for r in results if r.verdict == "uncertain")

    print(f"\n{'=' * 60}")
    print(f"Summary: {supported} supported, {disputed} disputed, {uncertain} uncertain")
    print(f"Total claims: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Claim Verification")
    parser.add_argument("--claims", nargs="+", help="Claims to verify")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    claims = args.claims or SAMPLE_CLAIMS
    results = asyncio.run(batch_verify(claims))

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print_report(results)
