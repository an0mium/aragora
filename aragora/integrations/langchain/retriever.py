"""
LangChain Retriever for Aragora Knowledge Mound.

Provides LangChain-compatible Retriever implementations for
querying the Knowledge Mound.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aragora.utils.async_utils import run_async

logger = logging.getLogger(__name__)


# Stub classes for when LangChain is not installed
class _StubBaseRetriever:
    """Stub BaseRetriever when LangChain not installed."""

    pass


class _StubDocument:
    """Stub Document."""

    def __init__(self, page_content: str = "", metadata: dict[str, Any] | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _StubCallbackManager:
    """Stub callback manager when LangChain not installed."""

    pass


# LangChain imports with fallback
# We use conditional imports and assign to module-level variables.
# For type checking, we import the real types in TYPE_CHECKING block.
LANGCHAIN_AVAILABLE = False

try:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun as _AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun as _CallbackManagerForRetrieverRun,
    )
    from langchain.schema import BaseRetriever as _BaseRetriever
    from langchain.schema import Document as _Document

    LANGCHAIN_AVAILABLE = True
    _ActualBaseRetriever: Any = _BaseRetriever
    _ActualDocument: Any = _Document
    _ActualCallbackManager: Any = _CallbackManagerForRetrieverRun
    _ActualAsyncCallbackManager: Any = _AsyncCallbackManagerForRetrieverRun
except ImportError:
    _ActualBaseRetriever = _StubBaseRetriever
    _ActualDocument = _StubDocument
    _ActualCallbackManager = _StubCallbackManager
    _ActualAsyncCallbackManager = _StubCallbackManager


if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain.schema import Document


class AragoraRetriever(_ActualBaseRetriever):
    """
    LangChain Retriever for Aragora Knowledge Mound.

    Retrieves relevant documents from the organization's
    knowledge base for use in RAG pipelines.

    Example:
        retriever = AragoraRetriever(aragora_url="http://localhost:8080")
        docs = retriever.get_relevant_documents("database migrations")

        # Use with LangChain
        from langchain.chains import RetrievalQA
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
        )
    """

    aragora_url: str = "http://localhost:8080"
    api_token: str | None = None
    max_results: int = 5
    timeout_seconds: float = 30.0
    include_metadata: bool = True

    def __init__(
        self,
        aragora_url: str = "http://localhost:8080",
        api_token: str | None = None,
        max_results: int = 5,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Aragora retriever.

        Args:
            aragora_url: Base URL for Aragora API
            api_token: Optional API token for authentication
            max_results: Maximum number of documents to retrieve
            **kwargs: Additional BaseRetriever arguments
        """
        super().__init__(**kwargs)
        self.aragora_url = aragora_url
        self.api_token = api_token
        self.max_results = max_results

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve documents synchronously."""

        return run_async(self._aget_relevant_documents(query, run_manager=None))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve documents asynchronously."""
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    f"{self.aragora_url}/api/v1/knowledge/search",
                    params={"q": query, "limit": self.max_results},
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

            documents: list[Document] = []
            for item in result.get("items", []):
                metadata: dict[str, Any] = {}
                if self.include_metadata:
                    metadata = {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "source": item.get("source", "knowledge_mound"),
                        "confidence": item.get("confidence", 0),
                        "created_at": item.get("created_at"),
                    }

                doc = _ActualDocument(
                    page_content=item.get("content", ""),
                    metadata=metadata,
                )
                documents.append(doc)

            return documents

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"[AragoraRetriever] connection error: {type(e).__name__}: {e}")
            return []
        except (ValueError, KeyError) as e:
            logger.error(f"[AragoraRetriever] response parse error: {type(e).__name__}: {e}")
            return []
