"""
AutoGen + Aragora Verification Example

Demonstrates registering Aragora verification as an AutoGen function tool.
Agents in the group chat can call `aragora_verify` to stress-test their
outputs before presenting them.

Requirements:
    pip install aragora-sdk pyautogen

Usage:
    export ARAGORA_API_KEY=your-key
    export OPENAI_API_KEY=your-key
    python main.py
"""

from __future__ import annotations

import os

from aragora_sdk import AragoraClient


def create_verification_functions(
    api_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Create AutoGen-compatible function tools for Aragora verification.

    Returns a dict of function name -> callable, ready for register_function().
    """
    client = AragoraClient(
        base_url=api_url or os.getenv("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=api_key or os.getenv("ARAGORA_API_KEY"),
    )

    def aragora_verify(content: str, context: str = "") -> str:
        """
        Verify content through Aragora's adversarial testing.

        Args:
            content: The content to verify (analysis, recommendation, etc.)
            context: Additional context about the task

        Returns:
            Verification result with verdict and findings summary.
        """
        task = f"Verify this content:\n\n{content}"
        if context:
            task += f"\n\nContext: {context}"

        result = client.gauntlet.run(task=task, attack_rounds=2)
        receipt = client.gauntlet.get_receipt(result["gauntlet_id"])
        findings = client.gauntlet.get_findings(result["gauntlet_id"])

        summary = f"Verdict: {result['verdict']}\nReceipt: {receipt['hash']}\n"
        if findings:
            summary += f"Findings ({len(findings)}):\n"
            for f in findings[:5]:
                summary += f"  - [{f.get('severity', 'INFO')}] {f.get('title', 'N/A')}\n"
        else:
            summary += "No issues found.\n"

        return summary

    def aragora_debate(question: str, agents: str = "claude,gpt,gemini") -> str:
        """
        Run a multi-agent debate on a question.

        Args:
            question: The question to debate
            agents: Comma-separated list of agent types

        Returns:
            Debate result with decision and confidence.
        """
        agent_list = [a.strip() for a in agents.split(",")]
        debate = client.debates.create(
            task=question,
            agents=agent_list,
            rounds=3,
        )
        return (
            f"Decision: {debate.get('decision', 'N/A')}\n"
            f"Confidence: {debate.get('confidence', 'N/A')}\n"
            f"Consensus: {debate.get('consensus_reached', False)}\n"
            f"Debate ID: {debate.get('debate_id', 'N/A')}"
        )

    return {
        "aragora_verify": aragora_verify,
        "aragora_debate": aragora_debate,
    }


def main():
    """Run an AutoGen group chat with Aragora verification tools."""
    try:
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    except ImportError:
        print("pyautogen not installed. Showing standalone verification.")
        print()

        functions = create_verification_functions()

        mock_analysis = (
            "Our security audit found 3 critical vulnerabilities: "
            "1) SQL injection in /api/users endpoint, "
            "2) Missing rate limiting on auth endpoints, "
            "3) JWT tokens never expire."
        )

        print(f"Agent output: {mock_analysis[:80]}...")
        print()
        print("Verifying with Aragora...")
        result = functions["aragora_verify"](mock_analysis, "Security audit report")
        print(result)
        return

    # Configure LLM
    llm_config = {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }

    # Create agents
    analyst = AssistantAgent(
        name="SecurityAnalyst",
        system_message=(
            "You are a security analyst. Analyze systems for vulnerabilities. "
            "Always verify your findings using aragora_verify before presenting them."
        ),
        llm_config=llm_config,
    )

    reviewer = AssistantAgent(
        name="ReviewerAgent",
        system_message=(
            "You review security analyses for completeness and accuracy. "
            "Use aragora_verify to validate the analyst's findings."
        ),
        llm_config=llm_config,
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
    )

    # Register Aragora functions
    functions = create_verification_functions()
    for name, func in functions.items():
        analyst.register_function({name: func})
        reviewer.register_function({name: func})

    # Run group chat
    chat = GroupChat(agents=[user_proxy, analyst, reviewer], messages=[], max_round=6)
    manager = GroupChatManager(groupchat=chat, llm_config=llm_config)

    user_proxy.initiate_chat(
        manager,
        message="Review the security of an API that uses JWT auth with no expiry and stores passwords in plaintext.",
    )


if __name__ == "__main__":
    main()
