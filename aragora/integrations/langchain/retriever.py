"""
LangChain Retriever for Aragora Knowledge Mound.

Provides LangChain-compatible Retriever implementations for
querying the Knowledge Mound.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# LangChain imports with fallback
try:
    from langchain.schema import BaseRetriever, Document
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    class BaseRetriever:  # type: ignore
        """Stub BaseRetriever when LangChain not installed."""

        pass

    class Document:  # type: ignore
        """Stub Document."""

        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class AragoraRetriever(BaseRetriever):
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
    api_token: Optional[str] = None
    max_results: int = 5
    timeout_seconds: float = 30.0
    include_metadata: bool = True

    def __init__(
        self,
        aragora_url: str = "http://localhost:8080",
        api_token: Optional[str] = None,
        max_results: int = 5,
        **kwargs: Any,
    ):
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
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Retrieve documents synchronously."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(
            self._aget_relevant_documents(query, run_manager=None)
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
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

            documents = []
            for item in result.get("items", []):
                metadata = {}
                if self.include_metadata:
                    metadata = {
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "source": item.get("source", "knowledge_mound"),
                        "confidence": item.get("confidence", 0),
                        "created_at": item.get("created_at"),
                    }

                doc = Document(
                    page_content=item.get("content", ""),
                    metadata=metadata,
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"[AragoraRetriever] Error: {e}")
            return []
