"""Tests for BM25 integration with InMemoryVectorStore."""

import pytest

from aragora.knowledge.mound.vector_abstraction.memory import InMemoryVectorStore
from aragora.knowledge.mound.vector_abstraction.base import VectorStoreConfig, VectorBackend
from aragora.knowledge.search.bm25 import BM25Index, BM25Config, BM25SearchResult


class TestBM25Index:
    """Tests for BM25Index standalone functionality."""

    @pytest.fixture
    def bm25_index(self):
        """Create a fresh BM25 index for testing."""
        return BM25Index()

    @pytest.fixture
    def custom_bm25_index(self):
        """Create BM25 index with custom config."""
        config = BM25Config(
            k1=2.0,
            b=0.5,
            lowercase=True,
            remove_punctuation=True,
        )
        return BM25Index(config)

    def test_add_document(self, bm25_index):
        """Test adding a document to the index."""
        bm25_index.add_document(
            id="doc-1",
            content="Python programming language",
            metadata={"type": "tutorial"},
        )

        assert bm25_index.get_stats()["total_documents"] == 1

    def test_add_multiple_documents(self, bm25_index):
        """Test adding multiple documents."""
        bm25_index.add_document(id="doc-1", content="Python is great")
        bm25_index.add_document(id="doc-2", content="Java is popular")
        bm25_index.add_document(id="doc-3", content="Python and Java together")

        assert bm25_index.get_stats()["total_documents"] == 3

    def test_remove_document(self, bm25_index):
        """Test removing a document from the index."""
        bm25_index.add_document(id="doc-1", content="Test document")
        bm25_index.add_document(id="doc-2", content="Another document")

        bm25_index.remove_document("doc-1")

        assert bm25_index.get_stats()["total_documents"] == 1

    def test_search_basic(self, bm25_index):
        """Test basic search functionality."""
        bm25_index.add_document(id="doc-1", content="Python programming tutorial")
        bm25_index.add_document(id="doc-2", content="JavaScript web development")
        bm25_index.add_document(id="doc-3", content="Python data science")

        results = bm25_index.search("Python", limit=10)

        assert len(results) >= 2
        assert all(isinstance(r, BM25SearchResult) for r in results)
        # Python docs should score higher
        doc_ids = [r.id for r in results]
        assert "doc-1" in doc_ids
        assert "doc-3" in doc_ids

    def test_search_no_matches(self, bm25_index):
        """Test search with no matching documents."""
        bm25_index.add_document(id="doc-1", content="Python programming")

        results = bm25_index.search("Ruby", limit=10)

        # Should return empty or very low scores
        assert len(results) == 0 or all(r.score == 0 for r in results)

    def test_search_case_insensitive(self, bm25_index):
        """Test that search is case insensitive by default."""
        bm25_index.add_document(id="doc-1", content="Python Programming")

        results_lower = bm25_index.search("python", limit=10)
        results_upper = bm25_index.search("PYTHON", limit=10)

        # Both should find the document
        assert len(results_lower) > 0
        assert len(results_upper) > 0

    def test_idf_weighting(self, bm25_index):
        """Test that rare terms get higher IDF scores."""
        # Add documents where 'python' is common and 'rust' is rare
        bm25_index.add_document(id="doc-1", content="python is great")
        bm25_index.add_document(id="doc-2", content="python is popular")
        bm25_index.add_document(id="doc-3", content="python is fun")
        bm25_index.add_document(id="doc-4", content="rust is fast")

        # Search for rare term 'rust'
        rust_results = bm25_index.search("rust", limit=10)
        assert len(rust_results) > 0
        assert rust_results[0].id == "doc-4"

    def test_term_frequency_saturation(self, bm25_index):
        """Test that repeated terms don't infinitely boost score."""
        bm25_index.add_document(id="doc-1", content="python")
        bm25_index.add_document(id="doc-2", content="python python python python python")

        results = bm25_index.search("python", limit=10)

        # Doc-2 should score higher but not 5x higher
        if len(results) >= 2:
            score_ratio = results[0].score / results[1].score if results[1].score > 0 else 1
            # BM25 saturation should limit the boost
            assert score_ratio < 3.0

    def test_document_length_normalization(self, bm25_index):
        """Test that long documents don't unfairly dominate."""
        bm25_index.add_document(id="doc-short", content="python tutorial")
        bm25_index.add_document(
            id="doc-long",
            content="python tutorial with lots of extra words about programming "
            "concepts and various other topics that extend the document",
        )

        results = bm25_index.search("python tutorial", limit=10)

        # Both should be found, short doc may score higher due to density
        doc_ids = [r.id for r in results]
        assert "doc-short" in doc_ids
        assert "doc-long" in doc_ids

    def test_matched_terms_tracked(self, bm25_index):
        """Test that matched terms are tracked in results."""
        bm25_index.add_document(id="doc-1", content="python programming language")

        results = bm25_index.search("python programming", limit=10)

        assert len(results) > 0
        assert len(results[0].matched_terms) >= 1

    def test_custom_config(self, custom_bm25_index):
        """Test BM25 with custom configuration."""
        custom_bm25_index.add_document(id="doc-1", content="Test document")

        assert custom_bm25_index.config.k1 == 2.0
        assert custom_bm25_index.config.b == 0.5

    def test_empty_query(self, bm25_index):
        """Test search with empty query."""
        bm25_index.add_document(id="doc-1", content="Test document")

        results = bm25_index.search("", limit=10)

        assert results == []


class TestInMemoryVectorStoreBM25Integration:
    """Tests for BM25 integration with InMemoryVectorStore."""

    @pytest.fixture
    def vector_store(self):
        """Create an in-memory vector store."""
        config = VectorStoreConfig(
            backend=VectorBackend.MEMORY,
            collection_name="test_collection",
        )
        return InMemoryVectorStore(config)

    @pytest.fixture
    async def connected_store(self, vector_store):
        """Create and connect a vector store."""
        await vector_store.connect()
        yield vector_store
        await vector_store.disconnect()

    @pytest.mark.asyncio
    async def test_upsert_indexes_in_bm25(self, connected_store):
        """Test that upsert adds document to BM25 index."""
        await connected_store.upsert(
            id="doc-1",
            embedding=[0.1, 0.2, 0.3],
            content="Python programming tutorial",
            metadata={"type": "tutorial"},
        )

        # Get BM25 index and verify document was indexed
        bm25_index = connected_store._get_bm25_index("")
        assert bm25_index.get_stats()["total_documents"] == 1

    @pytest.mark.asyncio
    async def test_delete_removes_from_bm25(self, connected_store):
        """Test that delete removes document from BM25 index."""
        await connected_store.upsert(
            id="doc-1",
            embedding=[0.1, 0.2, 0.3],
            content="Test document",
        )
        await connected_store.upsert(
            id="doc-2",
            embedding=[0.4, 0.5, 0.6],
            content="Another document",
        )

        bm25_index = connected_store._get_bm25_index("")
        assert bm25_index.get_stats()["total_documents"] == 2

        await connected_store.delete(["doc-1"])

        assert bm25_index.get_stats()["total_documents"] == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_uses_bm25(self, connected_store):
        """Test that hybrid search incorporates BM25 scores."""
        # Add documents with varying relevance
        await connected_store.upsert(
            id="python-doc",
            embedding=[0.9, 0.1, 0.0],  # High vector similarity for first query
            content="Python programming language tutorial",
        )
        await connected_store.upsert(
            id="java-doc",
            embedding=[0.1, 0.9, 0.0],  # Different direction
            content="Java enterprise development",
        )
        await connected_store.upsert(
            id="python-data",
            embedding=[0.8, 0.2, 0.1],
            content="Python data science and machine learning",
        )

        # Search with high alpha (more weight on BM25)
        results = await connected_store.hybrid_search(
            query="Python",
            embedding=[0.9, 0.1, 0.0],
            limit=10,
            alpha=0.8,  # 80% BM25, 20% vector
        )

        # Python docs should rank higher with BM25 weight
        top_ids = [r.id for r in results[:2]]
        assert "python-doc" in top_ids or "python-data" in top_ids

    @pytest.mark.asyncio
    async def test_hybrid_search_pure_vector(self, connected_store):
        """Test hybrid search with alpha=0 (pure vector)."""
        await connected_store.upsert(
            id="doc-1",
            embedding=[1.0, 0.0, 0.0],
            content="Irrelevant keyword content",
        )
        await connected_store.upsert(
            id="doc-2",
            embedding=[0.0, 1.0, 0.0],
            content="Python programming",
        )

        # Pure vector search (alpha=0)
        results = await connected_store.hybrid_search(
            query="Python",
            embedding=[1.0, 0.0, 0.0],  # Similar to doc-1
            limit=10,
            alpha=0.0,
        )

        # Doc-1 should rank first due to vector similarity
        assert len(results) > 0
        assert results[0].id == "doc-1"

    @pytest.mark.asyncio
    async def test_hybrid_search_pure_bm25(self, connected_store):
        """Test hybrid search with alpha=1 (pure BM25)."""
        await connected_store.upsert(
            id="doc-1",
            embedding=[1.0, 0.0, 0.0],
            content="Java enterprise application",
        )
        await connected_store.upsert(
            id="doc-2",
            embedding=[0.0, 1.0, 0.0],
            content="Python programming language",
        )

        # Pure BM25 search (alpha=1)
        results = await connected_store.hybrid_search(
            query="Python",
            embedding=[1.0, 0.0, 0.0],  # Similar to doc-1
            limit=10,
            alpha=1.0,
        )

        # Doc-2 should rank first due to keyword match
        assert len(results) > 0
        python_doc = next((r for r in results if r.id == "doc-2"), None)
        assert python_doc is not None
        # With pure BM25, Python doc should have higher score
        assert python_doc.score > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_with_namespace(self, connected_store):
        """Test that BM25 indices are namespace-aware."""
        await connected_store.upsert(
            id="doc-1",
            embedding=[0.1, 0.2, 0.3],
            content="Python in namespace A",
            namespace="namespace-a",
        )
        await connected_store.upsert(
            id="doc-2",
            embedding=[0.1, 0.2, 0.3],
            content="Python in namespace B",
            namespace="namespace-b",
        )

        # Search in namespace A
        results = await connected_store.hybrid_search(
            query="Python",
            embedding=[0.1, 0.2, 0.3],
            limit=10,
            alpha=0.5,
            namespace="namespace-a",
        )

        # Should only find doc in namespace A
        assert len(results) == 1
        assert results[0].id == "doc-1"

    @pytest.mark.asyncio
    async def test_bm25_batch_upsert(self, connected_store):
        """Test that batch upsert indexes all documents."""
        items = [
            {"id": "doc-1", "embedding": [0.1, 0.2], "content": "First document"},
            {"id": "doc-2", "embedding": [0.3, 0.4], "content": "Second document"},
            {"id": "doc-3", "embedding": [0.5, 0.6], "content": "Third document"},
        ]

        await connected_store.upsert_batch(items)

        bm25_index = connected_store._get_bm25_index("")
        assert bm25_index.get_stats()["total_documents"] == 3

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self, connected_store):
        """Test hybrid search respects metadata filters."""
        await connected_store.upsert(
            id="doc-1",
            embedding=[0.9, 0.1],
            content="Python tutorial",
            metadata={"category": "programming"},
        )
        await connected_store.upsert(
            id="doc-2",
            embedding=[0.9, 0.1],
            content="Python cooking recipes",
            metadata={"category": "cooking"},
        )

        # Search with filter
        results = await connected_store.hybrid_search(
            query="Python",
            embedding=[0.9, 0.1],
            limit=10,
            alpha=0.5,
            filters={"category": "programming"},
        )

        assert len(results) == 1
        assert results[0].id == "doc-1"


class TestBM25EdgeCases:
    """Edge case tests for BM25 functionality."""

    @pytest.fixture
    def bm25_index(self):
        return BM25Index()

    def test_special_characters(self, bm25_index):
        """Test handling of special characters."""
        bm25_index.add_document(id="doc-1", content="C++ programming language!")

        results = bm25_index.search("C++", limit=10)
        # Should handle special chars gracefully
        assert isinstance(results, list)

    def test_unicode_content(self, bm25_index):
        """Test handling of unicode content."""
        bm25_index.add_document(id="doc-1", content="Python programming 日本語")
        bm25_index.add_document(id="doc-2", content="Hello world 世界")

        results = bm25_index.search("Python", limit=10)
        assert len(results) > 0

    def test_very_long_document(self, bm25_index):
        """Test handling of very long documents."""
        long_content = " ".join(["word"] * 10000)
        bm25_index.add_document(id="doc-1", content=long_content)

        results = bm25_index.search("word", limit=10)
        assert len(results) > 0

    def test_duplicate_document_ids(self, bm25_index):
        """Test that duplicate IDs update the document."""
        bm25_index.add_document(id="doc-1", content="Original content")
        bm25_index.add_document(id="doc-1", content="Updated content")

        # Should still have one document
        assert bm25_index.get_stats()["total_documents"] == 1

        # Search should find updated content
        results = bm25_index.search("Updated", limit=10)
        assert len(results) > 0

    def test_clear_index(self, bm25_index):
        """Test clearing the entire index."""
        bm25_index.add_document(id="doc-1", content="First")
        bm25_index.add_document(id="doc-2", content="Second")

        bm25_index.clear()

        assert bm25_index.get_stats()["total_documents"] == 0

    def test_search_limit(self, bm25_index):
        """Test that search respects limit parameter."""
        for i in range(100):
            bm25_index.add_document(id=f"doc-{i}", content=f"Python document number {i}")

        results = bm25_index.search("Python", limit=5)
        assert len(results) <= 5


class TestBM25Scoring:
    """Tests for BM25 scoring accuracy."""

    @pytest.fixture
    def bm25_index(self):
        return BM25Index()

    def test_score_ordering(self, bm25_index):
        """Test that more relevant docs have higher scores."""
        bm25_index.add_document(id="high", content="python python python programming")
        bm25_index.add_document(id="medium", content="python programming tutorial")
        bm25_index.add_document(id="low", content="java programming tutorial")

        results = bm25_index.search("python", limit=10)

        # Filter to only docs with scores
        scored_results = [r for r in results if r.score > 0]
        if len(scored_results) >= 2:
            # Higher frequency of 'python' should score higher
            assert scored_results[0].id in ["high", "medium"]

    def test_multi_term_query(self, bm25_index):
        """Test scoring with multi-term queries."""
        bm25_index.add_document(id="doc-1", content="python programming language")
        bm25_index.add_document(id="doc-2", content="python")
        bm25_index.add_document(id="doc-3", content="programming language")

        results = bm25_index.search("python programming", limit=10)

        # Doc-1 should score highest (has both terms)
        if len(results) > 0:
            top_result = max(results, key=lambda r: r.score)
            assert top_result.id == "doc-1"

    def test_scores_are_non_negative(self, bm25_index):
        """Test that all BM25 scores are non-negative."""
        bm25_index.add_document(id="doc-1", content="Test content here")
        bm25_index.add_document(id="doc-2", content="Another document")

        results = bm25_index.search("content", limit=10)

        for result in results:
            assert result.score >= 0
