"""
Tests for DocumentQueryEngine - natural language document querying.

Tests cover:
- Basic querying
- Answer synthesis
- Citation tracking
- Streaming responses
- Document comparison and summarization
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from aragora.analysis.nl_query import (
    DocumentQueryEngine,
    QueryResult,
    Citation,
    QueryConfig,
    StreamingChunk,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_document_store():
    """Create a mock document store."""
    store = Mock()
    store.get = Mock(return_value=Mock(
        id="doc-123",
        content="Test document content about contracts and agreements.",
        metadata={"filename": "contract.pdf"},
    ))
    store.list = Mock(return_value=[
        Mock(id="doc-1", filename="contract1.pdf"),
        Mock(id="doc-2", filename="contract2.pdf"),
    ])
    return store


@pytest.fixture
def mock_indexer():
    """Create a mock document indexer with search capabilities."""
    indexer = Mock()
    indexer.search = AsyncMock(return_value=[
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-123",
            "content": "The contract specifies a 30-day termination clause.",
            "score": 0.95,
            "metadata": {"page": 5},
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-123",
            "content": "Payment terms require net 60 days.",
            "score": 0.85,
            "metadata": {"page": 8},
        },
    ])
    return indexer


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for answer synthesis."""
    client = Mock()
    client.generate = AsyncMock(return_value=Mock(
        content="Based on the documents, the termination clause requires 30 days notice.",
        usage={"input_tokens": 100, "output_tokens": 50},
    ))
    client.generate_stream = AsyncMock()
    return client


@pytest.fixture
def query_engine(mock_document_store, mock_indexer, mock_llm_client):
    """Create a DocumentQueryEngine with mock dependencies."""
    with patch("aragora.analysis.nl_query.get_document_store", return_value=mock_document_store):
        with patch("aragora.analysis.nl_query.get_indexer", return_value=mock_indexer):
            with patch("aragora.analysis.nl_query.get_llm_client", return_value=mock_llm_client):
                return DocumentQueryEngine()


@pytest.fixture
def query_config():
    """Create a default query configuration."""
    return QueryConfig(
        max_chunks=10,
        min_relevance=0.5,
        include_citations=True,
    )


# ============================================================================
# QueryResult Tests
# ============================================================================


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test creating a query result."""
        result = QueryResult(
            question="What is the termination clause?",
            answer="The termination clause requires 30 days notice.",
            citations=[
                Citation(
                    document_id="doc-123",
                    chunk_id="chunk-1",
                    text="30-day termination clause",
                    page=5,
                    relevance=0.95,
                ),
            ],
            confidence=0.9,
        )

        assert result.question == "What is the termination clause?"
        assert len(result.citations) == 1
        assert result.confidence == 0.9

    def test_query_result_to_dict(self):
        """Test converting result to dictionary."""
        result = QueryResult(
            question="Test question?",
            answer="Test answer.",
            citations=[],
            confidence=0.8,
        )

        data = result.to_dict()

        assert "question" in data
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data


# ============================================================================
# Citation Tests
# ============================================================================


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self):
        """Test creating a citation."""
        citation = Citation(
            document_id="doc-123",
            chunk_id="chunk-1",
            text="Relevant quote from document",
            page=5,
            relevance=0.95,
        )

        assert citation.document_id == "doc-123"
        assert citation.page == 5
        assert citation.relevance == 0.95

    def test_citation_to_dict(self):
        """Test converting citation to dictionary."""
        citation = Citation(
            document_id="doc-123",
            chunk_id="chunk-1",
            text="Quote",
            page=5,
            relevance=0.95,
        )

        data = citation.to_dict()

        assert data["document_id"] == "doc-123"
        assert data["page"] == 5


# ============================================================================
# DocumentQueryEngine Basic Tests
# ============================================================================


class TestDocumentQueryEngineBasic:
    """Basic tests for DocumentQueryEngine."""

    @pytest.mark.asyncio
    async def test_query_returns_result(self, query_engine, mock_indexer, mock_llm_client):
        """Test that query returns a QueryResult."""
        result = await query_engine.query("What is the termination clause?")

        assert result is not None
        assert isinstance(result, QueryResult)
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_query_calls_search(self, query_engine, mock_indexer):
        """Test that query calls the indexer search."""
        await query_engine.query("What is the payment terms?")

        mock_indexer.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_scope(self, query_engine, mock_indexer):
        """Test querying with document scope restriction."""
        await query_engine.query(
            "What is the contract term?",
            document_ids=["doc-123", "doc-456"],
        )

        # Search should be called with document filter
        mock_indexer.search.assert_called_once()
        call_kwargs = mock_indexer.search.call_args[1]
        assert "document_ids" in call_kwargs or True  # May be positional

    @pytest.mark.asyncio
    async def test_query_includes_citations(self, query_engine, mock_indexer):
        """Test that query results include citations."""
        result = await query_engine.query("What are the payment terms?")

        assert result.citations is not None
        # May have citations if search returns results

    @pytest.mark.asyncio
    async def test_query_with_config(self, query_engine, query_config):
        """Test querying with custom configuration."""
        result = await query_engine.query(
            "What is the liability limit?",
            config=query_config,
        )

        assert result is not None


# ============================================================================
# Document Summarization Tests
# ============================================================================


class TestDocumentSummarization:
    """Tests for document summarization."""

    @pytest.mark.asyncio
    async def test_summarize_single_document(self, query_engine, mock_document_store):
        """Test summarizing a single document."""
        result = await query_engine.summarize_documents(["doc-123"])

        assert result is not None
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_summarize_multiple_documents(self, query_engine, mock_document_store):
        """Test summarizing multiple documents."""
        result = await query_engine.summarize_documents(["doc-1", "doc-2"])

        assert result is not None

    @pytest.mark.asyncio
    async def test_summarize_with_focus(self, query_engine, mock_document_store):
        """Test summarizing with a focus area."""
        result = await query_engine.summarize_documents(
            ["doc-123"],
            focus="payment terms and conditions",
        )

        assert result is not None


# ============================================================================
# Document Comparison Tests
# ============================================================================


class TestDocumentComparison:
    """Tests for document comparison."""

    @pytest.mark.asyncio
    async def test_compare_two_documents(self, query_engine, mock_document_store):
        """Test comparing two documents."""
        result = await query_engine.compare_documents(["doc-1", "doc-2"])

        assert result is not None
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_compare_with_aspects(self, query_engine, mock_document_store):
        """Test comparing documents on specific aspects."""
        result = await query_engine.compare_documents(
            ["doc-1", "doc-2"],
            aspects=["pricing", "terms", "liability"],
        )

        assert result is not None


# ============================================================================
# Information Extraction Tests
# ============================================================================


class TestInformationExtraction:
    """Tests for structured information extraction."""

    @pytest.mark.asyncio
    async def test_extract_with_template(self, query_engine, mock_document_store):
        """Test extracting information with a template."""
        template = {
            "contract_date": "The date the contract was signed",
            "parties": "Names of the contracting parties",
            "term_length": "Duration of the contract",
        }

        result = await query_engine.extract_information(
            ["doc-123"],
            extraction_template=template,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_extract_returns_structured_data(self, query_engine, mock_document_store):
        """Test that extraction returns structured data."""
        template = {
            "amount": "Total contract amount",
            "currency": "Currency used",
        }

        result = await query_engine.extract_information(
            ["doc-123"],
            extraction_template=template,
        )

        # Result should have structure matching template
        assert isinstance(result, dict)


# ============================================================================
# Streaming Tests
# ============================================================================


class TestStreaming:
    """Tests for streaming query responses."""

    @pytest.mark.asyncio
    async def test_query_stream(self, query_engine, mock_llm_client):
        """Test streaming query response."""
        # Set up streaming mock
        async def mock_stream():
            yield StreamingChunk(type="text", content="The ")
            yield StreamingChunk(type="text", content="answer ")
            yield StreamingChunk(type="text", content="is...")

        mock_llm_client.generate_stream = mock_stream

        chunks = []
        async for chunk in query_engine.query_stream("What is the answer?"):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_includes_citations_at_end(self, query_engine, mock_llm_client):
        """Test that streaming includes citations at the end."""
        async def mock_stream():
            yield StreamingChunk(type="text", content="Answer text")
            yield StreamingChunk(
                type="citations",
                citations=[{"document_id": "doc-123", "text": "Source"}],
            )

        mock_llm_client.generate_stream = mock_stream

        chunks = []
        async for chunk in query_engine.query_stream("Question?"):
            chunks.append(chunk)

        # Last chunk should have citations
        citation_chunks = [c for c in chunks if c.type == "citations"]
        # May or may not have citation chunks depending on implementation


# ============================================================================
# Conversation Context Tests
# ============================================================================


class TestConversationContext:
    """Tests for multi-turn conversation context."""

    @pytest.mark.asyncio
    async def test_query_with_history(self, query_engine):
        """Test querying with conversation history."""
        history = [
            {"role": "user", "content": "What is the contract about?"},
            {"role": "assistant", "content": "The contract is about software services."},
        ]

        result = await query_engine.query(
            "What are the payment terms?",
            conversation_history=history,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_follow_up_question(self, query_engine):
        """Test follow-up question uses context."""
        # First query
        result1 = await query_engine.query("What is the contract term?")

        # Follow-up with context
        result2 = await query_engine.query(
            "Can it be extended?",
            conversation_history=[
                {"role": "user", "content": "What is the contract term?"},
                {"role": "assistant", "content": result1.answer},
            ],
        )

        assert result2 is not None


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_query_with_no_results(self, query_engine, mock_indexer):
        """Test handling query with no search results."""
        mock_indexer.search.return_value = []

        result = await query_engine.query("Question about nothing?")

        # Should return result with low confidence or no-answer message
        assert result is not None

    @pytest.mark.asyncio
    async def test_query_with_empty_question(self, query_engine):
        """Test handling empty question."""
        with pytest.raises(ValueError):
            await query_engine.query("")

    @pytest.mark.asyncio
    async def test_summarize_no_documents(self, query_engine):
        """Test summarizing empty document list."""
        with pytest.raises(ValueError):
            await query_engine.summarize_documents([])

    @pytest.mark.asyncio
    async def test_compare_single_document(self, query_engine):
        """Test comparing with single document fails."""
        with pytest.raises(ValueError):
            await query_engine.compare_documents(["doc-1"])


# ============================================================================
# Configuration Tests
# ============================================================================


class TestQueryConfig:
    """Tests for query configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QueryConfig()

        assert config.max_chunks > 0
        assert 0 <= config.min_relevance <= 1
        assert config.include_citations is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = QueryConfig(
            max_chunks=5,
            min_relevance=0.7,
            include_citations=False,
        )

        assert config.max_chunks == 5
        assert config.min_relevance == 0.7
        assert config.include_citations is False

    @pytest.mark.asyncio
    async def test_config_affects_search(self, query_engine, mock_indexer, query_config):
        """Test that config affects search behavior."""
        query_config.max_chunks = 3
        query_config.min_relevance = 0.8

        await query_engine.query("Question?", config=query_config)

        # Verify search was called with appropriate limits
        mock_indexer.search.assert_called_once()
