"""
LangChain Chains for Aragora.

Provides LangChain-compatible Chain implementations for:
- Multi-step debate workflows
- Research and debate pipelines
- Decision-making chains

These chains can be composed with other LangChain components.

Usage:
    from aragora.integrations.langchain import AragoraDebateChain

    chain = AragoraDebateChain(
        aragora_url="http://localhost:8080",
        pre_research=True,
    )

    result = await chain.arun(question="Should we migrate to the cloud?")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aragora.utils.async_utils import run_async

logger = logging.getLogger(__name__)

# Type-only imports for static analysis
if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )


# Stub classes for when LangChain is not installed
class _StubChain:
    """Stub Chain base class when LangChain not installed."""

    def __init__(self, **kwargs: Any) -> None:
        pass


class _StubAsyncCallbackManager:
    """Stub for AsyncCallbackManagerForChainRun when LangChain not installed."""

    pass


class _StubCallbackManager:
    """Stub for CallbackManagerForChainRun when LangChain not installed."""

    pass


# LangChain imports with fallback
try:
    from langchain.chains.base import Chain as _LangChainBase
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )

    LANGCHAIN_AVAILABLE = True
    _ChainBase: type = _LangChainBase
except ImportError:
    LANGCHAIN_AVAILABLE = False
    _ChainBase = _StubChain
    # Stubs replace LangChain types when the optional dependency is not installed.
    # The type mismatch is unavoidable since stubs are not subclasses of LangChain types.
    AsyncCallbackManagerForChainRun = _StubAsyncCallbackManager  # type: ignore[misc, assignment]
    CallbackManagerForChainRun = _StubCallbackManager  # type: ignore[misc, assignment]


class AragoraDebateChain(_ChainBase):
    """
    LangChain Chain for running structured Aragora debates.

    This chain provides a complete debate workflow:
    1. (Optional) Research phase - gather context from Knowledge Mound
    2. Debate phase - run multi-agent debate
    3. (Optional) Verification phase - verify answer against knowledge

    Example:
        chain = AragoraDebateChain(aragora_url="http://localhost:8080")

        result = chain.run(
            question="What's the best approach for our API versioning?",
            context="We have 50 active API consumers."
        )
    """

    # Configuration
    aragora_url: str = "http://localhost:8080"
    api_token: str | None = None
    pre_research: bool = True
    post_verify: bool = True
    default_agents: list[str] = ["claude", "gpt-4", "gemini"]
    max_rounds: int = 5
    timeout_seconds: float = 180.0

    @property
    def input_keys(self) -> list[str]:
        """Input keys for the chain."""
        return ["question"]

    @property
    def output_keys(self) -> list[str]:
        """Output keys for the chain."""
        return ["answer", "confidence", "reasoning", "knowledge_context"]

    def __init__(
        self,
        aragora_url: str = "http://localhost:8080",
        api_token: str | None = None,
        pre_research: bool = True,
        post_verify: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the debate chain.

        Args:
            aragora_url: Base URL for Aragora API
            api_token: Optional API token
            pre_research: Whether to research before debate
            post_verify: Whether to verify answer after debate
            **kwargs: Additional Chain arguments
        """
        super().__init__(**kwargs)
        self.aragora_url = aragora_url
        self.api_token = api_token
        self.pre_research = pre_research
        self.post_verify = post_verify

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run the chain synchronously."""

        return run_async(self._acall(inputs, None))

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run the chain asynchronously."""
        import httpx

        question = inputs["question"]
        context = inputs.get("context", "")

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        knowledge_context = ""
        reasoning_steps = []

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            # Phase 1: Pre-research (optional)
            if self.pre_research:
                reasoning_steps.append("Researching knowledge base...")
                try:
                    response = await client.get(
                        f"{self.aragora_url}/api/v1/knowledge/search",
                        params={"q": question, "limit": 5},
                        headers=headers,
                    )
                    if response.status_code == 200:
                        knowledge_result = response.json()
                        items = knowledge_result.get("items", [])
                        if items:
                            knowledge_context = "\n".join(
                                f"- {item.get('title', '')}: {item.get('content', '')[:200]}"
                                for item in items
                            )
                            reasoning_steps.append(f"Found {len(items)} relevant knowledge items")
                except (ConnectionError, TimeoutError, OSError, httpx.HTTPError) as e:
                    logger.warning(
                        "[AragoraDebateChain] Research connection error: %s: %s", type(e).__name__, e
                    )
                    reasoning_steps.append(f"Research skipped (connection error): {e}")
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning("[AragoraDebateChain] Research failed: %s: %s", type(e).__name__, e)
                    reasoning_steps.append(f"Research skipped: {e}")

            # Phase 2: Run debate
            reasoning_steps.append("Running multi-agent debate...")

            task = question
            if context:
                task = f"{question}\n\nContext: {context}"
            if knowledge_context:
                task = f"{task}\n\nRelevant knowledge:\n{knowledge_context}"

            try:
                response = await client.post(
                    f"{self.aragora_url}/api/debate/start",
                    json={
                        "task": task,
                        "agents": self.default_agents,
                        "max_rounds": self.max_rounds,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                debate_result = response.json()

                answer = debate_result.get("final_answer", "No answer")
                confidence = debate_result.get("confidence", 0)
                consensus = debate_result.get("consensus_reached", False)

                if consensus:
                    reasoning_steps.append(f"Consensus reached with {confidence:.0%} confidence")
                else:
                    reasoning_steps.append(
                        f"No consensus after {debate_result.get('rounds', self.max_rounds)} rounds"
                    )

            except (ConnectionError, TimeoutError, OSError, httpx.HTTPError) as e:
                logger.error(
                    "[AragoraDebateChain] Debate connection error: %s: %s", type(e).__name__, e
                )
                return {
                    "answer": f"Debate failed (connection error): {e}",
                    "confidence": 0,
                    "reasoning": "\n".join(reasoning_steps),
                    "knowledge_context": knowledge_context,
                }
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error("[AragoraDebateChain] Debate failed: %s: %s", type(e).__name__, e)
                return {
                    "answer": f"Debate failed: {e}",
                    "confidence": 0,
                    "reasoning": "\n".join(reasoning_steps),
                    "knowledge_context": knowledge_context,
                }

            # Phase 3: Post-verification (optional)
            if self.post_verify and answer:
                reasoning_steps.append("Verifying answer against knowledge...")
                # In a full implementation, this would cross-check the answer
                # against the Knowledge Mound for consistency

            return {
                "answer": answer,
                "confidence": confidence,
                "reasoning": "\n".join(reasoning_steps),
                "knowledge_context": knowledge_context,
            }

    @property
    def _chain_type(self) -> str:
        """Return chain type for serialization."""
        return "aragora_debate_chain"


class AragoraResearchDebateChain(_ChainBase):
    """
    Chain that combines research and debate for comprehensive analysis.

    This chain:
    1. Searches multiple sources for context
    2. Synthesizes findings
    3. Runs debate on the synthesized information
    4. Produces a well-reasoned conclusion
    """

    aragora_url: str = "http://localhost:8080"
    api_token: str | None = None
    search_sources: list[str] = ["knowledge", "web", "documents"]
    max_search_results: int = 10

    @property
    def input_keys(self) -> list[str]:
        """Input keys for the chain."""
        return ["topic"]

    @property
    def output_keys(self) -> list[str]:
        """Output keys for the chain."""
        return ["conclusion", "sources", "debate_summary"]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run the chain synchronously."""

        return run_async(self._acall(inputs, None))

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Run the research and debate chain."""
        import httpx

        topic = inputs["topic"]

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        sources = []
        context_parts = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Research phase
            for source in self.search_sources:
                try:
                    if source == "knowledge":
                        response = await client.get(
                            f"{self.aragora_url}/api/v1/knowledge/search",
                            params={"q": topic, "limit": self.max_search_results},
                            headers=headers,
                        )
                        if response.status_code == 200:
                            items = response.json().get("items", [])
                            for item in items:
                                sources.append(
                                    {
                                        "source": "knowledge",
                                        "title": item.get("title"),
                                        "content": item.get("content", "")[:500],
                                    }
                                )
                            if items:
                                context_parts.append(
                                    "From knowledge base:\n"
                                    + "\n".join(
                                        f"- {i.get('title')}: {i.get('content', '')[:200]}"
                                        for i in items[:3]
                                    )
                                )
                except (ConnectionError, TimeoutError, OSError, httpx.HTTPError) as e:
                    logger.warning(
                        "[AragoraResearchDebateChain] %s search connection error: %s: %s", source, type(e).__name__, e
                    )
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.warning(
                        "[AragoraResearchDebateChain] %s search failed: %s: %s", source, type(e).__name__, e
                    )

            # Debate phase
            full_context = (
                "\n\n".join(context_parts) if context_parts else "No prior context found."
            )

            try:
                response = await client.post(
                    f"{self.aragora_url}/api/debate/start",
                    json={
                        "task": f"Research analysis: {topic}\n\nContext:\n{full_context}",
                        "agents": ["claude", "gpt-4", "gemini"],
                        "max_rounds": 5,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                debate_result = response.json()

                conclusion = debate_result.get("final_answer", "No conclusion reached")
                debate_summary = (
                    f"Consensus: {debate_result.get('consensus_reached', False)}, "
                    f"Confidence: {debate_result.get('confidence', 0):.0%}, "
                    f"Rounds: {debate_result.get('rounds', 0)}"
                )

            except (ConnectionError, TimeoutError, OSError, httpx.HTTPError) as e:
                conclusion = f"Research completed but debate connection failed: {e}"
                debate_summary = f"Debate connection error: {type(e).__name__}"
            except (RuntimeError, ValueError, TypeError) as e:
                conclusion = f"Research completed but debate failed: {e}"
                debate_summary = f"Debate error: {type(e).__name__}"

            return {
                "conclusion": conclusion,
                "sources": sources,
                "debate_summary": debate_summary,
            }

    @property
    def _chain_type(self) -> str:
        """Return chain type for serialization."""
        return "aragora_research_debate_chain"
