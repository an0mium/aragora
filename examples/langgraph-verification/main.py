"""
LangGraph + Aragora Verification Example

Demonstrates adding Aragora as a verification node in a LangGraph workflow.
The verification node runs adversarial testing on the graph's output before
it reaches the end user.

Requirements:
    pip install aragora-sdk langgraph langchain-core

Usage:
    export ARAGORA_API_KEY=your-key
    python main.py
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from aragora_sdk import AragoraClient


# -- State Definition --


class WorkflowState(TypedDict, total=False):
    """State flowing through the LangGraph workflow."""

    task: str
    research: str
    draft: str
    verification: dict
    final_output: str


# -- Aragora Verification Node --


def create_verification_node(
    api_url: str | None = None,
    api_key: str | None = None,
    attack_rounds: int = 2,
):
    """
    Create a LangGraph node that verifies output via Aragora.

    Returns a function compatible with StateGraph.add_node().
    """
    client = AragoraClient(
        base_url=api_url or os.getenv("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=api_key or os.getenv("ARAGORA_API_KEY"),
    )

    def verify(state: WorkflowState) -> WorkflowState:
        """Run Aragora adversarial verification on the draft output."""
        draft = state.get("draft", "")
        task = state.get("task", "")

        result = client.gauntlet.run(
            task=f"Verify this output for the task '{task}':\n\n{draft}",
            attack_rounds=attack_rounds,
        )

        receipt = client.gauntlet.get_receipt(result["gauntlet_id"])
        findings = client.gauntlet.get_findings(result["gauntlet_id"])

        state["verification"] = {
            "verdict": result["verdict"],
            "gauntlet_id": result["gauntlet_id"],
            "receipt_hash": receipt["hash"],
            "findings_count": len(findings),
            "findings": findings,
        }

        return state

    return verify


def should_revise(state: WorkflowState) -> str:
    """Conditional edge: revise if verification fails, otherwise finalize."""
    verification = state.get("verification", {})
    if verification.get("verdict") == "PASS":
        return "finalize"
    return "revise"


def main():
    """Build and run a LangGraph workflow with Aragora verification."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        print("langgraph not installed. Showing standalone verification.")
        print()

        client = AragoraClient(
            base_url=os.getenv("ARAGORA_API_URL", "https://api.aragora.ai"),
            api_key=os.getenv("ARAGORA_API_KEY"),
        )

        mock_draft = (
            "We recommend implementing rate limiting using a token bucket algorithm "
            "with Redis as the backing store. Set limits at 100 req/min for free tier, "
            "1000 req/min for pro. Use sliding window counters for accurate tracking."
        )

        print(f"Draft output: {mock_draft[:80]}...")
        print("Running Aragora verification...")

        result = client.gauntlet.run(
            task=f"Verify this API design recommendation:\n\n{mock_draft}",
            attack_rounds=2,
        )

        print(f"  Verdict: {result['verdict']}")
        receipt = client.gauntlet.get_receipt(result["gauntlet_id"])
        print(f"  Receipt: {receipt['hash']}")
        return

    # Build the graph
    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("research", lambda s: {**s, "research": "Research results..."})
    graph.add_node(
        "draft", lambda s: {**s, "draft": f"Based on {s.get('research', '')}: recommendation..."}
    )
    graph.add_node("verify", create_verification_node(attack_rounds=2))
    graph.add_node("revise", lambda s: {**s, "draft": f"Revised: {s.get('draft', '')}"})
    graph.add_node("finalize", lambda s: {**s, "final_output": s.get("draft", "")})

    # Add edges
    graph.set_entry_point("research")
    graph.add_edge("research", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges(
        "verify", should_revise, {"finalize": "finalize", "revise": "revise"}
    )
    graph.add_edge("revise", "verify")  # Re-verify after revision
    graph.add_edge("finalize", END)

    # Run
    app = graph.compile()
    result = app.invoke({"task": "Design a rate limiting strategy for our API"})

    print(f"Final output: {result.get('final_output', '')[:100]}...")
    print(f"Verification: {result.get('verification', {}).get('verdict')}")
    print(f"Receipt: {result.get('verification', {}).get('receipt_hash')}")


if __name__ == "__main__":
    main()
