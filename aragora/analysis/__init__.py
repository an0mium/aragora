"""
Aragora Analysis Module.

Provides natural language querying and analysis capabilities for documents.

Usage:
    from aragora.analysis import DocumentQueryEngine, query_documents

    # Quick query
    result = await query_documents(
        question="What are the key terms in this contract?",
        document_ids=["doc_123"]
    )
    print(result.answer)

    # Full engine with config
    engine = await DocumentQueryEngine.create()
    result = await engine.query("Compare these two agreements")
"""

from .nl_query import (
    DocumentQueryEngine,
    QueryConfig,
    QueryResult,
    QueryMode,
    AnswerConfidence,
    Citation,
    StreamingChunk,
    query_documents,
    summarize_document,
)

__all__ = [
    "DocumentQueryEngine",
    "QueryConfig",
    "QueryResult",
    "QueryMode",
    "AnswerConfidence",
    "Citation",
    "StreamingChunk",
    "query_documents",
    "summarize_document",
]
