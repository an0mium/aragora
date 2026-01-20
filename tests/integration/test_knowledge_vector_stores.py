"""
Knowledge System Vector Store Integration Tests.

These tests validate the Knowledge System against real vector stores.
Skip if vector stores are not available.

Run with:
    WEAVIATE_URL=http://localhost:8080 pytest tests/integration/test_knowledge_vector_stores.py -v
    QDRANT_URL=http://localhost:6333 pytest tests/integration/test_knowledge_vector_stores.py -v
"""

import asyncio
import os
import uuid
from typing import Any

import pytest

# Check vector store availability
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
SKIP_WEAVIATE = "Weaviate not available (set WEAVIATE_URL env var)"
SKIP_QDRANT = "Qdrant not available (set QDRANT_URL env var)"

# Try importing vector store clients
try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    WEAVIATE_AVAILABLE = bool(WEAVIATE_URL)
except ImportError:
    WEAVIATE_AVAILABLE = False
    SKIP_WEAVIATE = "weaviate-client package not installed"

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    QDRANT_AVAILABLE = bool(QDRANT_URL)
except ImportError:
    QDRANT_AVAILABLE = False
    SKIP_QDRANT = "qdrant-client package not installed"


def generate_embedding(dim: int = 384, seed: int = 0) -> list[float]:
    """Generate a deterministic embedding for testing."""
    import hashlib

    h = hashlib.sha256(str(seed).encode()).hexdigest()
    values = [int(h[i : i + 2], 16) / 255.0 for i in range(0, min(len(h), dim * 2), 2)]
    # Pad if needed
    while len(values) < dim:
        values.extend(values[: dim - len(values)])
    return values[:dim]


class TestWeaviateIntegration:
    """Test Knowledge Mound with Weaviate backend."""

    pytestmark = pytest.mark.skipif(not WEAVIATE_AVAILABLE, reason=SKIP_WEAVIATE)

    @pytest.fixture
    async def weaviate_client(self):
        """Create Weaviate client for tests."""
        client = weaviate.connect_to_local(
            host=WEAVIATE_URL.replace("http://", "").split(":")[0],
            port=int(WEAVIATE_URL.split(":")[-1]) if ":" in WEAVIATE_URL else 8080,
        )

        # Create test collection
        collection_name = f"AragoraTest_{uuid.uuid4().hex[:8]}"

        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.OBJECT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
        )

        yield client, collection_name

        # Cleanup
        client.collections.delete(collection_name)
        client.close()

    @pytest.mark.asyncio
    async def test_document_ingestion(self, weaviate_client):
        """Test ingesting documents into Weaviate."""
        client, collection_name = weaviate_client
        collection = client.collections.get(collection_name)

        # Ingest documents
        documents = [
            {"content": f"Document {i} content about AI systems", "source": f"doc_{i}.txt"}
            for i in range(100)
        ]

        with collection.batch.dynamic() as batch:
            for i, doc in enumerate(documents):
                batch.add_object(
                    properties=doc,
                    vector=generate_embedding(seed=i),
                )

        # Verify count
        response = collection.aggregate.over_all(total_count=True)
        assert response.total_count == 100

    @pytest.mark.asyncio
    async def test_semantic_search(self, weaviate_client):
        """Test semantic search in Weaviate."""
        client, collection_name = weaviate_client
        collection = client.collections.get(collection_name)

        # Add diverse documents
        topics = ["machine learning", "database systems", "network security", "cloud computing"]
        for i, topic in enumerate(topics):
            collection.data.insert(
                properties={
                    "content": f"A comprehensive guide to {topic}",
                    "source": f"{topic}.md",
                },
                vector=generate_embedding(seed=i * 100),
            )

        # Search for similar document
        query_vector = generate_embedding(seed=0)  # Similar to machine learning doc
        results = collection.query.near_vector(
            near_vector=query_vector,
            limit=2,
        )

        assert len(results.objects) == 2
        assert "machine learning" in results.objects[0].properties["content"]

    @pytest.mark.asyncio
    async def test_filtered_search(self, weaviate_client):
        """Test filtered semantic search."""
        client, collection_name = weaviate_client
        collection = client.collections.get(collection_name)

        # Add documents with different sources
        sources = ["internal", "external", "internal", "external"]
        for i, source in enumerate(sources):
            collection.data.insert(
                properties={"content": f"Document {i}", "source": source},
                vector=generate_embedding(seed=i),
            )

        # Search with filter
        from weaviate.classes.query import Filter

        results = collection.query.near_vector(
            near_vector=generate_embedding(seed=0),
            filters=Filter.by_property("source").equal("internal"),
            limit=10,
        )

        assert len(results.objects) == 2
        for obj in results.objects:
            assert obj.properties["source"] == "internal"

    @pytest.mark.asyncio
    async def test_large_scale_ingestion(self, weaviate_client):
        """Test ingesting 10K+ documents."""
        client, collection_name = weaviate_client
        collection = client.collections.get(collection_name)

        # Ingest 10K documents in batches
        total_docs = 10000
        batch_size = 1000

        for batch_start in range(0, total_docs, batch_size):
            with collection.batch.fixed_size(batch_size) as batch:
                for i in range(batch_start, min(batch_start + batch_size, total_docs)):
                    batch.add_object(
                        properties={"content": f"Document {i}", "source": "bulk"},
                        vector=generate_embedding(seed=i),
                    )

        # Verify count
        response = collection.aggregate.over_all(total_count=True)
        assert response.total_count == total_docs


class TestQdrantIntegration:
    """Test Knowledge Mound with Qdrant backend."""

    pytestmark = pytest.mark.skipif(not QDRANT_AVAILABLE, reason=SKIP_QDRANT)

    @pytest.fixture
    async def qdrant_client(self):
        """Create Qdrant client for tests."""
        client = QdrantClient(url=QDRANT_URL)

        # Create test collection
        collection_name = f"aragora_test_{uuid.uuid4().hex[:8]}"

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        yield client, collection_name

        # Cleanup
        client.delete_collection(collection_name=collection_name)
        client.close()

    @pytest.mark.asyncio
    async def test_document_ingestion(self, qdrant_client):
        """Test ingesting documents into Qdrant."""
        client, collection_name = qdrant_client

        # Ingest documents
        points = [
            PointStruct(
                id=i,
                vector=generate_embedding(seed=i),
                payload={"content": f"Document {i}", "source": f"doc_{i}.txt"},
            )
            for i in range(100)
        ]

        client.upsert(collection_name=collection_name, points=points)

        # Verify count
        info = client.get_collection(collection_name=collection_name)
        assert info.points_count == 100

    @pytest.mark.asyncio
    async def test_semantic_search(self, qdrant_client):
        """Test semantic search in Qdrant."""
        client, collection_name = qdrant_client

        # Add documents
        topics = ["machine learning", "database systems", "network security", "cloud computing"]
        points = [
            PointStruct(
                id=i,
                vector=generate_embedding(seed=i * 100),
                payload={"content": f"Guide to {topic}", "topic": topic},
            )
            for i, topic in enumerate(topics)
        ]
        client.upsert(collection_name=collection_name, points=points)

        # Search
        results = client.search(
            collection_name=collection_name,
            query_vector=generate_embedding(seed=0),
            limit=2,
        )

        assert len(results) == 2
        assert results[0].payload["topic"] == "machine learning"

    @pytest.mark.asyncio
    async def test_filtered_search(self, qdrant_client):
        """Test filtered semantic search in Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client, collection_name = qdrant_client

        # Add documents with different categories
        categories = ["tech", "science", "tech", "business"]
        points = [
            PointStruct(
                id=i,
                vector=generate_embedding(seed=i),
                payload={"content": f"Document {i}", "category": cat},
            )
            for i, cat in enumerate(categories)
        ]
        client.upsert(collection_name=collection_name, points=points)

        # Search with filter
        results = client.search(
            collection_name=collection_name,
            query_vector=generate_embedding(seed=0),
            query_filter=Filter(
                must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
            ),
            limit=10,
        )

        assert len(results) == 2
        for result in results:
            assert result.payload["category"] == "tech"

    @pytest.mark.asyncio
    async def test_large_scale_ingestion(self, qdrant_client):
        """Test ingesting 10K+ documents."""
        client, collection_name = qdrant_client

        # Ingest 10K documents in batches
        total_docs = 10000
        batch_size = 1000

        for batch_start in range(0, total_docs, batch_size):
            points = [
                PointStruct(
                    id=i,
                    vector=generate_embedding(seed=i),
                    payload={"content": f"Document {i}"},
                )
                for i in range(batch_start, min(batch_start + batch_size, total_docs))
            ]
            client.upsert(collection_name=collection_name, points=points)

        # Verify count
        info = client.get_collection(collection_name=collection_name)
        assert info.points_count == total_docs

    @pytest.mark.asyncio
    async def test_concurrent_search(self, qdrant_client):
        """Test concurrent search performance."""
        client, collection_name = qdrant_client

        # Add documents
        points = [
            PointStruct(
                id=i,
                vector=generate_embedding(seed=i),
                payload={"content": f"Document {i}"},
            )
            for i in range(1000)
        ]
        client.upsert(collection_name=collection_name, points=points)

        async def search_task(query_id: int):
            """Perform a search."""
            results = client.search(
                collection_name=collection_name,
                query_vector=generate_embedding(seed=query_id),
                limit=10,
            )
            return len(results)

        # Run 50 concurrent searches
        tasks = [search_task(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        assert all(r == 10 for r in results)


class TestMemoryVectorStore:
    """Test in-memory vector store for development."""

    @pytest.mark.asyncio
    async def test_memory_store_basic(self):
        """Test basic operations with in-memory store."""
        from aragora.knowledge.mound.vector_abstraction.memory import InMemoryVectorStore
        from aragora.knowledge.mound.vector_abstraction.base import VectorStoreConfig

        config = VectorStoreConfig(embedding_dimensions=384)
        store = InMemoryVectorStore(config)
        await store.connect()

        # Add vectors
        for i in range(100):
            await store.upsert(
                id=f"doc_{i}",
                embedding=generate_embedding(seed=i),
                content=f"Document {i}",
                metadata={"index": i},
            )

        # Search
        results = await store.search(
            embedding=generate_embedding(seed=0),
            limit=5,
        )

        assert len(results) == 5
        # First result should be the same document
        assert results[0].id == "doc_0"

    @pytest.mark.asyncio
    async def test_memory_store_filtering(self):
        """Test filtering with in-memory store."""
        from aragora.knowledge.mound.vector_abstraction.memory import InMemoryVectorStore
        from aragora.knowledge.mound.vector_abstraction.base import VectorStoreConfig

        config = VectorStoreConfig(embedding_dimensions=384)
        store = InMemoryVectorStore(config)
        await store.connect()

        # Add documents with categories
        for i in range(20):
            category = "A" if i % 2 == 0 else "B"
            await store.upsert(
                id=f"doc_{i}",
                embedding=generate_embedding(seed=i),
                content=f"Document {i}",
                metadata={"category": category},
            )

        # Search with filter
        results = await store.search(
            embedding=generate_embedding(seed=0),
            limit=10,
            filters={"category": "A"},
        )

        assert len(results) == 10
        for r in results:
            assert r.metadata["category"] == "A"

    @pytest.mark.asyncio
    async def test_memory_store_namespaces(self):
        """Test namespace isolation in memory store."""
        from aragora.knowledge.mound.vector_abstraction.memory import InMemoryVectorStore
        from aragora.knowledge.mound.vector_abstraction.base import VectorStoreConfig

        config = VectorStoreConfig(embedding_dimensions=384)
        store = InMemoryVectorStore(config)
        await store.connect()

        # Add to different namespaces
        for i in range(10):
            await store.upsert(
                id=f"doc_{i}",
                embedding=generate_embedding(seed=i),
                content=f"Doc {i}",
                metadata={},
                namespace="ns_a",
            )
            await store.upsert(
                id=f"doc_{i}",
                embedding=generate_embedding(seed=i + 100),
                content=f"Doc {i} B",
                metadata={},
                namespace="ns_b",
            )

        # Search in namespace A
        results_a = await store.search(
            embedding=generate_embedding(seed=0),
            limit=5,
            namespace="ns_a",
        )

        # Search in namespace B
        results_b = await store.search(
            embedding=generate_embedding(seed=100),
            limit=5,
            namespace="ns_b",
        )

        # Results should be from respective namespaces
        assert all("Doc" in r.content and "B" not in r.content for r in results_a)
        assert all("B" in r.content for r in results_b)
