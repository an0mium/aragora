"""
Document NL Query endpoint handlers.

Endpoints:
- POST /api/documents/query - Ask questions about documents
- POST /api/documents/summarize - Summarize documents
- POST /api/documents/compare - Compare multiple documents
- POST /api/documents/extract - Extract structured information
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aragora.server.http_utils import run_async as _run_async

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
    safe_error_message,
)

logger = logging.getLogger(__name__)


class DocumentQueryHandler(BaseHandler):
    """Handler for natural language document query endpoints."""

    ROUTES = [
        "/api/v1/documents/query",
        "/api/v1/documents/summarize",
        "/api/v1/documents/compare",
        "/api/v1/documents/extract",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle GET requests - not supported for query endpoints."""
        return error_response("Use POST method for document queries", 405)

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST requests to appropriate methods."""
        if path == "/api/v1/documents/query":
            return self._query_documents(handler)
        elif path == "/api/v1/documents/summarize":
            return self._summarize_documents(handler)
        elif path == "/api/v1/documents/compare":
            return self._compare_documents(handler)
        elif path == "/api/v1/documents/extract":
            return self._extract_information(handler)
        return None

    @require_user_auth
    @handle_errors("document query")
    def _query_documents(self, handler, user=None) -> HandlerResult:
        """
        Answer a natural language question about documents.

        Request body:
        {
            "question": "What are the payment terms?",
            "document_ids": ["doc1", "doc2"],  // Optional: scope to specific docs
            "workspace_id": "ws_123",  // Optional
            "conversation_id": "conv_123",  // Optional: for multi-turn
            "config": {  // Optional
                "max_chunks": 10,
                "include_quotes": true
            }
        }

        Response:
        {
            "query_id": "query_abc123",
            "question": "...",
            "answer": "...",
            "confidence": "high",
            "citations": [...],
            "processing_time_ms": 1234
        }
        """
        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        question = body.get("question", "").strip()
        if not question:
            return error_response("'question' field is required", 400)

        document_ids = body.get("document_ids")
        workspace_id = body.get("workspace_id")
        conversation_id = body.get("conversation_id")
        config_dict = body.get("config", {})

        # Run async query
        try:
            result = _run_async(
                self._run_query(
                    question=question,
                    document_ids=document_ids,
                    workspace_id=workspace_id,
                    conversation_id=conversation_id,
                    config_dict=config_dict,
                )
            )
            return json_response(result)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return error_response(safe_error_message(e, "Query"), 500)

    async def _run_query(
        self,
        question: str,
        document_ids: Optional[list[str]],
        workspace_id: Optional[str],
        conversation_id: Optional[str],
        config_dict: dict,
    ) -> dict[str, Any]:
        """Run the document query asynchronously."""
        from aragora.analysis.nl_query import DocumentQueryEngine, QueryConfig

        # Build config from request
        config = QueryConfig(
            max_chunks=config_dict.get("max_chunks", 10),
            include_quotes=config_dict.get("include_quotes", True),
            max_answer_length=config_dict.get("max_answer_length", 500),
        )

        engine = await DocumentQueryEngine.create(config=config)
        result = await engine.query(
            question=question,
            workspace_id=workspace_id,
            document_ids=document_ids,
            conversation_id=conversation_id,
        )

        return result.to_dict()

    @require_user_auth
    @handle_errors("document summarize")
    def _summarize_documents(self, handler, user=None) -> HandlerResult:
        """
        Summarize one or more documents.

        Request body:
        {
            "document_ids": ["doc1", "doc2"],
            "focus": "financial terms",  // Optional: focus area
            "config": {...}  // Optional
        }

        Response:
        {
            "query_id": "...",
            "answer": "Summary...",
            "confidence": "high",
            "citations": [...]
        }
        """
        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        document_ids = body.get("document_ids", [])
        if not document_ids:
            return error_response("'document_ids' field is required", 400)

        focus = body.get("focus")
        config_dict = body.get("config", {})

        try:
            result = _run_async(
                self._run_summarize(
                    document_ids=document_ids,
                    focus=focus,
                    config_dict=config_dict,
                )
            )
            return json_response(result)
        except Exception as e:
            logger.error(f"Summarize failed: {e}")
            return error_response(safe_error_message(e, "Summarize"), 500)

    async def _run_summarize(
        self,
        document_ids: list[str],
        focus: Optional[str],
        config_dict: dict,
    ) -> dict[str, Any]:
        """Run document summarization asynchronously."""
        from aragora.analysis.nl_query import DocumentQueryEngine, QueryConfig

        config = QueryConfig(
            **{k: v for k, v in config_dict.items() if k in QueryConfig.__annotations__}
        )
        engine = await DocumentQueryEngine.create(config=config)
        result = await engine.summarize_documents(
            document_ids=document_ids,
            focus=focus,
        )

        return result.to_dict()

    @require_user_auth
    @handle_errors("document compare")
    def _compare_documents(self, handler, user=None) -> HandlerResult:
        """
        Compare multiple documents.

        Request body:
        {
            "document_ids": ["doc1", "doc2"],  // At least 2
            "aspects": ["pricing", "terms"],  // Optional: specific aspects
            "config": {...}
        }

        Response:
        {
            "query_id": "...",
            "answer": "Comparison...",
            "confidence": "high",
            "citations": [...]
        }
        """
        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        document_ids = body.get("document_ids", [])
        if len(document_ids) < 2:
            return error_response("At least 2 document_ids required for comparison", 400)

        aspects = body.get("aspects")
        config_dict = body.get("config", {})

        try:
            result = _run_async(
                self._run_compare(
                    document_ids=document_ids,
                    aspects=aspects,
                    config_dict=config_dict,
                )
            )
            return json_response(result)
        except Exception as e:
            logger.error(f"Compare failed: {e}")
            return error_response(safe_error_message(e, "Compare"), 500)

    async def _run_compare(
        self,
        document_ids: list[str],
        aspects: Optional[list[str]],
        config_dict: dict,
    ) -> dict[str, Any]:
        """Run document comparison asynchronously."""
        from aragora.analysis.nl_query import DocumentQueryEngine, QueryConfig

        config = QueryConfig(
            **{k: v for k, v in config_dict.items() if k in QueryConfig.__annotations__}
        )
        engine = await DocumentQueryEngine.create(config=config)
        result = await engine.compare_documents(
            document_ids=document_ids,
            aspects=aspects,
        )

        return result.to_dict()

    @require_user_auth
    @handle_errors("document extract")
    def _extract_information(self, handler, user=None) -> HandlerResult:
        """
        Extract structured information from documents.

        Request body:
        {
            "document_ids": ["doc1"],
            "fields": {
                "parties": "Who are the parties to this agreement?",
                "effective_date": "What is the effective date?",
                "term": "What is the term or duration?",
                "payment_terms": "What are the payment terms?"
            },
            "config": {...}
        }

        Response:
        {
            "document_ids": ["doc1"],
            "extractions": {
                "parties": {"answer": "...", "confidence": "high", "citations": [...]},
                "effective_date": {...},
                ...
            }
        }
        """
        body = self.read_json_body(handler)
        if not body:
            return error_response("Request body required", 400)

        document_ids = body.get("document_ids", [])
        if not document_ids:
            return error_response("'document_ids' field is required", 400)

        fields = body.get("fields", {})
        if not fields:
            return error_response("'fields' dict is required with extraction queries", 400)

        config_dict = body.get("config", {})

        try:
            result = _run_async(
                self._run_extract(
                    document_ids=document_ids,
                    fields=fields,
                    config_dict=config_dict,
                )
            )
            return json_response(result)
        except Exception as e:
            logger.error(f"Extract failed: {e}")
            return error_response(safe_error_message(e, "Extract"), 500)

    async def _run_extract(
        self,
        document_ids: list[str],
        fields: dict[str, str],
        config_dict: dict,
    ) -> dict[str, Any]:
        """Run structured extraction asynchronously."""
        from aragora.analysis.nl_query import DocumentQueryEngine, QueryConfig

        config = QueryConfig(
            **{k: v for k, v in config_dict.items() if k in QueryConfig.__annotations__}
        )
        engine = await DocumentQueryEngine.create(config=config)
        results = await engine.extract_information(
            document_ids=document_ids,
            extraction_template=fields,
        )

        return {
            "document_ids": document_ids,
            "extractions": {field: result.to_dict() for field, result in results.items()},
        }
