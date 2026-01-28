"""
LangChain Tools for Aragora.

Provides LangChain-compatible Tool implementations for:
- Running debates with multiple AI agents
- Querying the Knowledge Mound
- Getting decisions with audit trails

These tools can be used with LangChain agents, chains, and workflows.

Usage:
    from aragora.integrations.langchain import AragoraDebateTool

    tool = AragoraDebateTool(
        aragora_url="http://localhost:8080",
        api_token="your-token",
    )

    # Synchronous
    result = tool.run("Should we adopt microservices?")

    # Asynchronous
    result = await tool.arun("Should we adopt microservices?")
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

logger = logging.getLogger(__name__)

# LangChain imports with fallback
try:
    from langchain.tools import BaseTool
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Provide stub classes for type hints
    class BaseTool:  # type: ignore
        """Stub BaseTool when LangChain not installed."""

        pass

    class BaseModel:  # type: ignore
        """Stub BaseModel."""

        pass

    def Field(*args: Any, **kwargs: Any) -> Any:
        """Stub Field."""
        return None


def get_langchain_version() -> Optional[str]:
    """Get the LangChain version if available."""
    try:
        import langchain

        return getattr(langchain, "__version__", "unknown")
    except ImportError:
        return None


class AragoraToolInput(BaseModel):
    """Input schema for Aragora debate tool (compatible API)."""

    question: str = Field(description="The question or task to debate")
    agents: Optional[List[str]] = Field(
        default=None,
        description="List of agents to participate",
    )
    rounds: int = Field(
        default=3,
        description="Number of debate rounds",
    )
    consensus_threshold: float = Field(
        default=0.8,
        description="Threshold for consensus (0.0-1.0)",
    )
    include_evidence: bool = Field(
        default=True,
        description="Whether to include evidence in response",
    )


class AragoraDebateInput(BaseModel):
    """Input schema for Aragora debate tool."""

    task: str = Field(description="The question or task to debate")
    agents: Optional[List[str]] = Field(
        default=None,
        description="List of agents to participate (e.g., ['claude', 'gpt-4']). If not specified, uses defaults.",
    )
    max_rounds: Optional[int] = Field(
        default=None,
        description="Maximum debate rounds (default: 5)",
    )


class AragoraDebateTool(BaseTool):
    """
    LangChain Tool for running Aragora debates.

    This tool runs a multi-agent debate on a given question or task
    and returns the consensus answer with confidence score.

    Example:
        tool = AragoraDebateTool(aragora_url="http://localhost:8080")
        result = tool.run("What's the best database for our use case?")
        # Returns: "Based on debate with 85% consensus: PostgreSQL is recommended..."
    """

    name: str = "aragora_debate"
    description: str = (
        "Run a multi-agent AI debate to get a well-reasoned answer. "
        "Use this when you need multiple perspectives on a complex question. "
        "The debate reaches consensus through structured argumentation."
    )
    args_schema: Type[BaseModel] = AragoraDebateInput

    # Configuration
    aragora_url: str = "http://localhost:8080"
    api_token: Optional[str] = None
    default_agents: List[str] = ["claude", "gpt-4", "gemini"]
    default_max_rounds: int = 5
    timeout_seconds: float = 120.0

    def __init__(
        self,
        aragora_url: str = "http://localhost:8080",
        api_token: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Aragora debate tool.

        Args:
            aragora_url: Base URL for Aragora API
            api_token: Optional API token for authentication
            **kwargs: Additional BaseTool arguments
        """
        super().__init__(**kwargs)
        self.aragora_url = aragora_url
        self.api_token = api_token

    def _run(
        self,
        task: str,
        agents: Optional[List[str]] = None,
        max_rounds: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the debate synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._arun(task, agents, max_rounds, None)
        )

    async def _arun(
        self,
        task: str,
        agents: Optional[List[str]] = None,
        max_rounds: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the debate asynchronously."""
        import httpx

        agents = agents or self.default_agents
        max_rounds = max_rounds or self.default_max_rounds

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        payload = {
            "task": task,
            "agents": agents,
            "max_rounds": max_rounds,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.aragora_url}/api/debate/start",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

            # Format result for LangChain
            if result.get("consensus_reached"):
                return (
                    f"Debate consensus ({result.get('confidence', 0):.0%} confidence): "
                    f"{result.get('final_answer', 'No answer')}"
                )
            else:
                return f"No consensus reached after {result.get('rounds', max_rounds)} rounds."

        except Exception as e:
            logger.error(f"[AragoraDebateTool] Error: {e}")
            return f"Error running debate: {e}"


class AragoraKnowledgeInput(BaseModel):
    """Input schema for Aragora knowledge tool."""

    query: str = Field(description="Search query for the knowledge base")
    limit: Optional[int] = Field(
        default=5,
        description="Maximum number of results to return",
    )


class AragoraKnowledgeTool(BaseTool):
    """
    LangChain Tool for querying Aragora Knowledge Mound.

    This tool searches the organization's knowledge base for relevant
    information, documents, and past debate conclusions.

    Example:
        tool = AragoraKnowledgeTool(aragora_url="http://localhost:8080")
        result = tool.run("previous decisions about database migrations")
    """

    name: str = "aragora_knowledge"
    description: str = (
        "Search the organization's knowledge base for relevant information. "
        "Use this to find documents, past decisions, and institutional knowledge. "
        "Returns the most relevant results with confidence scores."
    )
    args_schema: Type[BaseModel] = AragoraKnowledgeInput

    # Configuration
    aragora_url: str = "http://localhost:8080"
    api_token: Optional[str] = None
    timeout_seconds: float = 30.0

    def __init__(
        self,
        aragora_url: str = "http://localhost:8080",
        api_token: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the knowledge tool."""
        super().__init__(**kwargs)
        self.aragora_url = aragora_url
        self.api_token = api_token

    def _run(
        self,
        query: str,
        limit: Optional[int] = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the query synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self._arun(query, limit, None))

    async def _arun(
        self,
        query: str,
        limit: Optional[int] = 5,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the query asynchronously."""
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    f"{self.aragora_url}/api/v1/knowledge/search",
                    params={"q": query, "limit": limit},
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

            # Format results for LangChain
            items = result.get("items", [])
            if not items:
                return f"No knowledge found for: {query}"

            formatted = []
            for i, item in enumerate(items, 1):
                confidence = item.get("confidence", 0)
                title = item.get("title", "Untitled")
                content = item.get("content", "")[:200]
                formatted.append(f"{i}. [{confidence:.0%}] {title}: {content}...")

            return "\n".join(formatted)

        except Exception as e:
            logger.error(f"[AragoraKnowledgeTool] Error: {e}")
            return f"Error querying knowledge: {e}"


class AragoraDecisionInput(BaseModel):
    """Input schema for Aragora decision tool."""

    question: str = Field(description="The decision question")
    options: Optional[List[str]] = Field(
        default=None,
        description="List of options to choose from (optional)",
    )


class AragoraDecisionTool(BaseTool):
    """
    LangChain Tool for making decisions with Aragora.

    This tool uses multi-agent debate to make a decision and generates
    an auditable decision receipt.

    Example:
        tool = AragoraDecisionTool(aragora_url="http://localhost:8080")
        result = tool.run("Should we approve this budget increase?")
    """

    name: str = "aragora_decision"
    description: str = (
        "Make a decision using multi-agent deliberation. "
        "Use this for important decisions that need audit trails. "
        "Returns a decision with rationale and confidence."
    )
    args_schema: Type[BaseModel] = AragoraDecisionInput

    # Configuration
    aragora_url: str = "http://localhost:8080"
    api_token: Optional[str] = None
    timeout_seconds: float = 120.0

    def __init__(
        self,
        aragora_url: str = "http://localhost:8080",
        api_token: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the decision tool."""
        super().__init__(**kwargs)
        self.aragora_url = aragora_url
        self.api_token = api_token

    def _run(
        self,
        question: str,
        options: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the decision synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self._arun(question, options, None))

    async def _arun(
        self,
        question: str,
        options: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the decision asynchronously."""
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        payload = {"question": question}
        if options:
            payload["options"] = options

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.aragora_url}/api/v1/decisions/make",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

            # Format decision for LangChain
            decision = result.get("decision", "Unknown")
            confidence = result.get("confidence", 0)
            rationale = result.get("rationale", "No rationale provided")
            receipt_id = result.get("receipt_id", "N/A")

            return (
                f"Decision: {decision} ({confidence:.0%} confidence)\n"
                f"Rationale: {rationale}\n"
                f"Receipt ID: {receipt_id}"
            )

        except Exception as e:
            logger.error(f"[AragoraDecisionTool] Error: {e}")
            return f"Error making decision: {e}"


# Convenience function to get all tools
def get_aragora_tools(
    aragora_url: str = "http://localhost:8080",
    api_token: Optional[str] = None,
) -> List[BaseTool]:
    """
    Get all Aragora LangChain tools.

    Args:
        aragora_url: Base URL for Aragora API
        api_token: Optional API token

    Returns:
        List of configured Aragora tools
    """
    return [
        AragoraDebateTool(aragora_url=aragora_url, api_token=api_token),
        AragoraKnowledgeTool(aragora_url=aragora_url, api_token=api_token),
        AragoraDecisionTool(aragora_url=aragora_url, api_token=api_token),
    ]
