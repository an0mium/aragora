"""
LangChain Integration for Aragora.

Exposes Aragora as LangChain-compatible tools and chains:
- AragoraDebateTool: LangChain Tool for running debates
- AragoraKnowledgeTool: LangChain Tool for querying Knowledge Mound
- AragoraDebateChain: LangChain Chain for multi-step debate workflows

Usage:
    from aragora.integrations.langchain import (
        AragoraDebateTool,
        AragoraKnowledgeTool,
    )

    # Use as LangChain tool
    debate_tool = AragoraDebateTool(aragora_url="http://localhost:8080")
    result = await debate_tool.arun("Should we use microservices?")

    # Use in an agent
    from langchain.agents import AgentExecutor
    agent = AgentExecutor(tools=[debate_tool], ...)
"""

from aragora.integrations.langchain.tools import (
    AragoraDebateTool,
    AragoraKnowledgeTool,
    AragoraDecisionTool,
)
from aragora.integrations.langchain.chains import AragoraDebateChain

__all__ = [
    "AragoraDebateTool",
    "AragoraKnowledgeTool",
    "AragoraDecisionTool",
    "AragoraDebateChain",
]
