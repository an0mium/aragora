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
    LANGCHAIN_AVAILABLE,
)
from aragora.integrations.langchain.chains import AragoraDebateChain
from aragora.integrations.langchain.retriever import AragoraRetriever
from aragora.integrations.langchain.callbacks import AragoraCallbackHandler

# Alias for backwards compatibility
AragoraTool = AragoraDebateTool


def is_langchain_available() -> bool:
    """Check if LangChain is available."""
    return LANGCHAIN_AVAILABLE


__all__ = [
    # Primary tools
    "AragoraDebateTool",
    "AragoraKnowledgeTool",
    "AragoraDecisionTool",
    "AragoraDebateChain",
    # Retriever and callbacks
    "AragoraRetriever",
    "AragoraCallbackHandler",
    # Backwards compatibility
    "AragoraTool",
    "LANGCHAIN_AVAILABLE",
    "is_langchain_available",
]
