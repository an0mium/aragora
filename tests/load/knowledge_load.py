"""
Knowledge Management Load Testing Suite for Aragora.

Tests Knowledge Mound and related subsystems under load including:
- Knowledge node creation and retrieval
- Semantic search operations
- Knowledge federation
- Cross-debate learning
- Evidence collection

Run with:
    pytest tests/load/knowledge_load.py -v --asyncio-mode=auto

For stress testing:
    pytest tests/load/knowledge_load.py -v -k stress --asyncio-mode=auto

Environment Variables:
    ARAGORA_API_URL: API base URL (default: http://localhost:8080)
    ARAGORA_API_TOKEN: Authentication token (optional)
    ARAGORA_KM_CONCURRENT: Concurrent KM operations (default: 20)
    ARAGORA_KM_DURATION: Test duration in seconds (default: 60)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

# Configuration
API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")
API_TOKEN = os.environ.get("ARAGORA_API_TOKEN", "")
CONCURRENT_KM = int(os.environ.get("ARAGORA_KM_CONCURRENT", "20"))
TEST_DURATION = int(os.environ.get("ARAGORA_KM_DURATION", "60"))


@dataclass
class KnowledgeLoadMetrics:
    """Metrics collected during knowledge management load test."""

    # Node operations
    nodes_created: int = 0
    nodes_retrieved: int = 0
    nodes_updated: int = 0
    nodes_deleted: int = 0

    # Search operations
    searches_performed: int = 0
    search_results_total: int = 0

    # Federation operations
    federation_syncs: int = 0
    cross_workspace_queries: int = 0

    # Evidence operations
    evidence_collected: int = 0
    evidence_validated: int = 0

    # Error tracking
    api_errors: int = 0
    rate_limited: int = 0
    response_times: list[float] = field(default_factory=list)
    search_latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def avg_response_time_ms(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times) * 1000

    @property
    def p95_response_time_ms(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)] * 1000

    @property
    def p99_response_time_ms(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)] * 1000

    @property
    def avg_search_latency_ms(self) -> float:
        if not self.search_latencies:
            return 0.0
        return sum(self.search_latencies) / len(self.search_latencies) * 1000

    @property
    def p95_search_latency_ms(self) -> float:
        if not self.search_latencies:
            return 0.0
        sorted_times = sorted(self.search_latencies)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)] * 1000

    @property
    def total_operations(self) -> int:
        return (
            self.nodes_created
            + self.nodes_retrieved
            + self.nodes_updated
            + self.nodes_deleted
            + self.searches_performed
            + self.federation_syncs
            + self.evidence_collected
        )

    @property
    def operations_per_second(self) -> float:
        if self.duration == 0:
            return 0.0
        return self.total_operations / self.duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes_created": self.nodes_created,
            "nodes_retrieved": self.nodes_retrieved,
            "nodes_updated": self.nodes_updated,
            "nodes_deleted": self.nodes_deleted,
            "searches_performed": self.searches_performed,
            "search_results_total": self.search_results_total,
            "federation_syncs": self.federation_syncs,
            "cross_workspace_queries": self.cross_workspace_queries,
            "evidence_collected": self.evidence_collected,
            "evidence_validated": self.evidence_validated,
            "api_errors": self.api_errors,
            "rate_limited": self.rate_limited,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "p95_response_time_ms": round(self.p95_response_time_ms, 2),
            "p99_response_time_ms": round(self.p99_response_time_ms, 2),
            "avg_search_latency_ms": round(self.avg_search_latency_ms, 2),
            "p95_search_latency_ms": round(self.p95_search_latency_ms, 2),
            "total_operations": self.total_operations,
            "operations_per_second": round(self.operations_per_second, 2),
            "duration_seconds": round(self.duration, 2),
            "error_count": len(self.errors),
        }


def random_string(length: int = 10) -> str:
    """Generate random string."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


def generate_sample_knowledge_content() -> str:
    """Generate sample knowledge content for testing."""
    topics = [
        "artificial intelligence and machine learning",
        "climate change mitigation strategies",
        "distributed systems architecture",
        "quantum computing applications",
        "sustainable energy solutions",
        "blockchain technology use cases",
        "cybersecurity best practices",
        "data privacy regulations",
        "agile software development",
        "microservices design patterns",
    ]

    topic = random.choice(topics)
    paragraphs = random.randint(1, 3)

    content = f"Knowledge about {topic}.\n\n"

    for i in range(paragraphs):
        content += (
            f"This is paragraph {i + 1} discussing various aspects of {topic}. "
            f"The information presented here is based on current understanding and "
            f"may include multiple perspectives from different sources. "
            f"Key considerations include implementation challenges, potential benefits, "
            f"and relevant industry standards. Test ID: {random_string(6)}\n\n"
        )

    return content


class KnowledgeLoadClient:
    """Client for knowledge management load testing."""

    def __init__(self, base_url: str, metrics: KnowledgeLoadMetrics, token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.metrics = metrics
        self.token = token
        self.session: Optional[Any] = None
        self.created_node_ids: list[str] = []

    async def _ensure_session(self) -> Any:
        """Ensure aiohttp session exists."""
        if self.session is None:
            import aiohttp

            self.session = aiohttp.ClientSession()
        return self.session

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def create_knowledge_node(
        self,
        content: str,
        node_type: str = "insight",
        workspace_id: Optional[str] = None,
    ) -> Optional[str]:
        """Create a knowledge node."""
        session = await self._ensure_session()

        payload = {
            "content": content,
            "node_type": node_type,
            "metadata": {
                "source": "load_test",
                "timestamp": time.time(),
            },
        }

        if workspace_id:
            payload["workspace_id"] = workspace_id

        try:
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/knowledge/nodes",
                json=payload,
                headers=self._get_headers(),
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status in (200, 201):
                    data = await response.json()
                    node_id = data.get("node_id") or data.get("id")
                    self.metrics.nodes_created += 1
                    if node_id:
                        self.created_node_ids.append(node_id)
                    return node_id
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return None
                else:
                    self.metrics.api_errors += 1
                    return None

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Create node error: {str(e)[:100]}")
            return None

    async def get_knowledge_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a knowledge node."""
        session = await self._ensure_session()

        try:
            start_time = time.time()

            async with session.get(
                f"{self.base_url}/api/knowledge/nodes/{node_id}",
                headers=self._get_headers(),
                timeout=10,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status == 200:
                    self.metrics.nodes_retrieved += 1
                    return await response.json()
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return None
                else:
                    self.metrics.api_errors += 1
                    return None

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Get node error: {str(e)[:100]}")
            return None

    async def update_knowledge_node(
        self,
        node_id: str,
        content: str,
    ) -> bool:
        """Update a knowledge node."""
        session = await self._ensure_session()

        payload = {
            "content": content,
            "metadata": {
                "updated_at": time.time(),
                "source": "load_test_update",
            },
        }

        try:
            start_time = time.time()

            async with session.put(
                f"{self.base_url}/api/knowledge/nodes/{node_id}",
                json=payload,
                headers=self._get_headers(),
                timeout=10,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status in (200, 204):
                    self.metrics.nodes_updated += 1
                    return True
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return False
                else:
                    self.metrics.api_errors += 1
                    return False

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Update node error: {str(e)[:100]}")
            return False

    async def delete_knowledge_node(self, node_id: str) -> bool:
        """Delete a knowledge node."""
        session = await self._ensure_session()

        try:
            start_time = time.time()

            async with session.delete(
                f"{self.base_url}/api/knowledge/nodes/{node_id}",
                headers=self._get_headers(),
                timeout=10,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status in (200, 204):
                    self.metrics.nodes_deleted += 1
                    if node_id in self.created_node_ids:
                        self.created_node_ids.remove(node_id)
                    return True
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return False
                else:
                    self.metrics.api_errors += 1
                    return False

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Delete node error: {str(e)[:100]}")
            return False

    async def search_knowledge(
        self,
        query: str,
        limit: int = 10,
        workspace_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search knowledge base."""
        session = await self._ensure_session()

        params = {"q": query, "limit": limit}
        if workspace_id:
            params["workspace_id"] = workspace_id

        try:
            start_time = time.time()

            async with session.get(
                f"{self.base_url}/api/knowledge/search",
                params=params,
                headers=self._get_headers(),
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)
                self.metrics.search_latencies.append(elapsed)

                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", data) if isinstance(data, dict) else data
                    if isinstance(results, list):
                        self.metrics.searches_performed += 1
                        self.metrics.search_results_total += len(results)
                        return results
                    return []
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return []
                else:
                    self.metrics.api_errors += 1
                    return []

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Search error: {str(e)[:100]}")
            return []

    async def query_knowledge(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Query knowledge using semantic understanding."""
        session = await self._ensure_session()

        payload = {
            "query": query,
            "context": context or {},
        }

        try:
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/knowledge/query",
                json=payload,
                headers=self._get_headers(),
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)
                self.metrics.search_latencies.append(elapsed)

                if response.status == 200:
                    self.metrics.searches_performed += 1
                    return await response.json()
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return None
                else:
                    self.metrics.api_errors += 1
                    return None

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Query error: {str(e)[:100]}")
            return None

    async def sync_federation(
        self,
        source_workspace: str,
        target_workspace: str,
    ) -> bool:
        """Trigger federation sync between workspaces."""
        session = await self._ensure_session()

        payload = {
            "source_workspace_id": source_workspace,
            "target_workspace_id": target_workspace,
            "sync_type": "incremental",
        }

        try:
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/knowledge/federation/sync",
                json=payload,
                headers=self._get_headers(),
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status in (200, 202):
                    self.metrics.federation_syncs += 1
                    return True
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return False
                else:
                    self.metrics.api_errors += 1
                    return False

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Federation sync error: {str(e)[:100]}")
            return False

    async def cross_workspace_query(
        self,
        query: str,
        workspace_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Query across multiple workspaces."""
        session = await self._ensure_session()

        payload = {
            "query": query,
            "workspace_ids": workspace_ids,
        }

        try:
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/knowledge/federation/query",
                json=payload,
                headers=self._get_headers(),
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)
                self.metrics.search_latencies.append(elapsed)

                if response.status == 200:
                    data = await response.json()
                    self.metrics.cross_workspace_queries += 1
                    return data.get("results", [])
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return []
                else:
                    self.metrics.api_errors += 1
                    return []

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Cross-workspace query error: {str(e)[:100]}")
            return []

    async def collect_evidence(
        self,
        claim: str,
        source_debate_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Collect evidence for a claim."""
        session = await self._ensure_session()

        payload = {
            "claim": claim,
            "search_strategy": "comprehensive",
        }

        if source_debate_id:
            payload["debate_id"] = source_debate_id

        try:
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/knowledge/evidence/collect",
                json=payload,
                headers=self._get_headers(),
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status == 200:
                    self.metrics.evidence_collected += 1
                    return await response.json()
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return None
                else:
                    self.metrics.api_errors += 1
                    return None

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Evidence collection error: {str(e)[:100]}")
            return None

    async def validate_evidence(
        self,
        evidence_id: str,
    ) -> Optional[dict[str, Any]]:
        """Validate collected evidence."""
        session = await self._ensure_session()

        try:
            start_time = time.time()

            async with session.post(
                f"{self.base_url}/api/knowledge/evidence/{evidence_id}/validate",
                headers=self._get_headers(),
                timeout=30,
            ) as response:
                elapsed = time.time() - start_time
                self.metrics.response_times.append(elapsed)

                if response.status == 200:
                    self.metrics.evidence_validated += 1
                    return await response.json()
                elif response.status == 429:
                    self.metrics.rate_limited += 1
                    return None
                else:
                    self.metrics.api_errors += 1
                    return None

        except Exception as e:
            self.metrics.api_errors += 1
            self.metrics.errors.append(f"Evidence validation error: {str(e)[:100]}")
            return None

    async def close(self) -> None:
        """Close session."""
        if self.session:
            await self.session.close()
            self.session = None


async def run_knowledge_load_test(
    base_url: str,
    token: str,
    concurrent: int,
    duration: float,
    include_federation: bool = True,
    include_evidence: bool = True,
) -> KnowledgeLoadMetrics:
    """
    Run knowledge management load test.

    Args:
        base_url: API base URL
        token: Authentication token
        concurrent: Number of concurrent operations
        duration: Test duration in seconds
        include_federation: Include federation operations
        include_evidence: Include evidence operations

    Returns:
        KnowledgeLoadMetrics with test results
    """
    metrics = KnowledgeLoadMetrics()
    metrics.start_time = time.time()

    clients: list[KnowledgeLoadClient] = []

    for _ in range(concurrent):
        clients.append(KnowledgeLoadClient(base_url, metrics, token))

    search_queries = [
        "artificial intelligence",
        "machine learning applications",
        "climate change solutions",
        "software architecture",
        "data privacy",
        "security best practices",
        "distributed systems",
        "cloud computing",
    ]

    evidence_claims = [
        "AI systems can improve decision making accuracy",
        "Remote work increases productivity",
        "Renewable energy is cost-effective",
        "Microservices improve scalability",
    ]

    async def knowledge_workload(client: KnowledgeLoadClient) -> None:
        """Simulate knowledge management workload."""
        # Create a node
        content = generate_sample_knowledge_content()
        node_id = await client.create_knowledge_node(content)

        # Search knowledge
        query = random.choice(search_queries)
        await client.search_knowledge(query)

        # Query knowledge
        await client.query_knowledge(f"What are best practices for {random_string(6)}?")

        # Occasionally update or delete nodes
        if node_id and random.random() < 0.3:
            await client.update_knowledge_node(node_id, content + "\nUpdated content.")

        if node_id and random.random() < 0.1:
            await client.delete_knowledge_node(node_id)

        # Federation operations
        if include_federation and random.random() < 0.1:
            await client.sync_federation(
                f"workspace_{random.randint(1, 5)}",
                f"workspace_{random.randint(6, 10)}",
            )

            await client.cross_workspace_query(
                random.choice(search_queries),
                [f"workspace_{i}" for i in random.sample(range(1, 10), 3)],
            )

        # Evidence operations
        if include_evidence and random.random() < 0.2:
            claim = random.choice(evidence_claims)
            evidence_result = await client.collect_evidence(claim)

            if evidence_result and evidence_result.get("evidence_id"):
                await client.validate_evidence(evidence_result["evidence_id"])

    # Run workloads for the specified duration
    end_time = time.time() + duration
    tasks: list[asyncio.Task] = []

    while time.time() < end_time:
        for client in clients:
            if len(tasks) < concurrent * 2:
                task = asyncio.create_task(knowledge_workload(client))
                tasks.append(task)

        done_tasks = [t for t in tasks if t.done()]
        for t in done_tasks:
            tasks.remove(t)

        await asyncio.sleep(0.1)

    # Wait for remaining tasks
    if tasks:
        await asyncio.wait(tasks, timeout=10.0)

    # Close all clients
    for client in clients:
        await client.close()

    metrics.end_time = time.time()
    return metrics


# =============================================================================
# SLO Validation Thresholds
# =============================================================================


class KnowledgeSLOThresholds:
    """SLO thresholds for knowledge management operations."""

    # Response time thresholds
    NODE_CRUD_P95_MS = 500  # Node operations should complete within 500ms at p95
    NODE_CRUD_P99_MS = 1000  # Node operations should complete within 1s at p99
    SEARCH_P95_MS = 1000  # Search should complete within 1s at p95
    SEARCH_P99_MS = 2000  # Search should complete within 2s at p99

    # Throughput thresholds
    MIN_OPS_PER_SECOND = 5  # Should handle at least 5 operations per second

    # Error rate thresholds
    MAX_ERROR_RATE = 0.05  # Max 5% error rate


# =============================================================================
# Pytest Test Cases
# =============================================================================


@pytest.mark.asyncio
async def test_single_node_lifecycle():
    """Test single knowledge node creation, retrieval, and deletion."""
    metrics = KnowledgeLoadMetrics()
    client = KnowledgeLoadClient(API_URL, metrics, API_TOKEN)

    # Create
    content = generate_sample_knowledge_content()
    node_id = await client.create_knowledge_node(content)

    if node_id:
        # Retrieve
        await client.get_knowledge_node(node_id)

        # Update
        await client.update_knowledge_node(node_id, content + "\nUpdated.")

        # Delete
        await client.delete_knowledge_node(node_id)

    await client.close()

    print(
        f"\nNode operations: created={metrics.nodes_created}, "
        f"retrieved={metrics.nodes_retrieved}, "
        f"updated={metrics.nodes_updated}, "
        f"deleted={metrics.nodes_deleted}"
    )


@pytest.mark.asyncio
async def test_search_throughput():
    """Test knowledge search throughput."""
    metrics = KnowledgeLoadMetrics()
    client = KnowledgeLoadClient(API_URL, metrics, API_TOKEN)

    queries = [
        "machine learning",
        "data analysis",
        "security protocols",
        "software design",
        "cloud architecture",
    ]

    for query in queries:
        await client.search_knowledge(query)

    await client.close()

    print(f"\nSearches performed: {metrics.searches_performed}")
    print(f"Avg search latency: {metrics.avg_search_latency_ms:.1f}ms")
    print(f"p95 search latency: {metrics.p95_search_latency_ms:.1f}ms")


@pytest.mark.asyncio
async def test_concurrent_node_operations():
    """Test concurrent knowledge node operations."""
    metrics = await run_knowledge_load_test(
        base_url=API_URL,
        token=API_TOKEN,
        concurrent=5,
        duration=10.0,
        include_federation=False,
        include_evidence=False,
    )

    print(f"\nResults: {json.dumps(metrics.to_dict(), indent=2)}")

    if metrics.total_operations == 0:
        pytest.skip("No operations completed (API may not be available)")

    # Should have performed some operations
    assert metrics.total_operations > 0


@pytest.mark.asyncio
async def test_semantic_query():
    """Test semantic knowledge queries."""
    metrics = KnowledgeLoadMetrics()
    client = KnowledgeLoadClient(API_URL, metrics, API_TOKEN)

    queries = [
        "What are the best practices for securing API endpoints?",
        "How can machine learning improve decision making?",
        "What strategies help with distributed system scaling?",
    ]

    for query in queries:
        await client.query_knowledge(query)

    await client.close()

    print(f"\nSemantic queries: {metrics.searches_performed}")


@pytest.mark.asyncio
async def test_federation_operations():
    """Test knowledge federation operations."""
    metrics = KnowledgeLoadMetrics()
    client = KnowledgeLoadClient(API_URL, metrics, API_TOKEN)

    # Sync between workspaces
    await client.sync_federation("workspace_1", "workspace_2")

    # Cross-workspace query
    await client.cross_workspace_query(
        "data governance",
        ["workspace_1", "workspace_2", "workspace_3"],
    )

    await client.close()

    print(f"\nFederation syncs: {metrics.federation_syncs}")
    print(f"Cross-workspace queries: {metrics.cross_workspace_queries}")


@pytest.mark.asyncio
async def test_evidence_collection():
    """Test evidence collection operations."""
    metrics = KnowledgeLoadMetrics()
    client = KnowledgeLoadClient(API_URL, metrics, API_TOKEN)

    claims = [
        "Automated testing reduces bugs",
        "Continuous integration improves code quality",
    ]

    for claim in claims:
        result = await client.collect_evidence(claim)
        if result and result.get("evidence_id"):
            await client.validate_evidence(result["evidence_id"])

    await client.close()

    print(f"\nEvidence collected: {metrics.evidence_collected}")
    print(f"Evidence validated: {metrics.evidence_validated}")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_knowledge_stress():
    """Stress test knowledge management system."""
    metrics = await run_knowledge_load_test(
        base_url=API_URL,
        token=API_TOKEN,
        concurrent=CONCURRENT_KM,
        duration=TEST_DURATION,
        include_federation=True,
        include_evidence=True,
    )

    print(f"\n{'=' * 60}")
    print("Knowledge Management Stress Test Results")
    print("=" * 60)
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60)

    if metrics.total_operations == 0:
        pytest.skip("No operations completed")

    # Validate SLO thresholds
    assert metrics.p95_response_time_ms < KnowledgeSLOThresholds.NODE_CRUD_P95_MS, (
        f"p95 response time {metrics.p95_response_time_ms}ms exceeds "
        f"{KnowledgeSLOThresholds.NODE_CRUD_P95_MS}ms threshold"
    )

    assert metrics.p99_response_time_ms < KnowledgeSLOThresholds.NODE_CRUD_P99_MS, (
        f"p99 response time {metrics.p99_response_time_ms}ms exceeds "
        f"{KnowledgeSLOThresholds.NODE_CRUD_P99_MS}ms threshold"
    )

    # Check error rate
    total_requests = metrics.total_operations + metrics.api_errors
    if total_requests > 0:
        error_rate = metrics.api_errors / total_requests
        assert error_rate < KnowledgeSLOThresholds.MAX_ERROR_RATE, (
            f"Error rate {error_rate:.2%} exceeds "
            f"{KnowledgeSLOThresholds.MAX_ERROR_RATE:.2%} threshold"
        )


@pytest.mark.asyncio
async def test_search_slo_compliance():
    """Test that search operations meet SLO requirements."""
    metrics = KnowledgeLoadMetrics()
    client = KnowledgeLoadClient(API_URL, metrics, API_TOKEN)

    # Perform many searches to get meaningful latency data
    queries = ["AI", "ML", "security", "data", "cloud"] * 10

    for query in queries:
        await client.search_knowledge(query)
        await asyncio.sleep(0.05)  # Small delay to avoid overwhelming

    await client.close()

    if metrics.searches_performed == 0:
        pytest.skip("No searches completed")

    print("\nSearch SLO compliance:")
    print(f"  Searches performed: {metrics.searches_performed}")
    print(f"  Avg latency: {metrics.avg_search_latency_ms:.1f}ms")
    print(f"  p95 latency: {metrics.p95_search_latency_ms:.1f}ms")

    # Validate search SLO
    assert metrics.p95_search_latency_ms < KnowledgeSLOThresholds.SEARCH_P95_MS, (
        f"Search p95 latency {metrics.p95_search_latency_ms}ms exceeds "
        f"{KnowledgeSLOThresholds.SEARCH_P95_MS}ms threshold"
    )


if __name__ == "__main__":
    # Run standalone for quick testing
    async def main() -> None:
        print(f"Running knowledge load test against {API_URL}")
        print(f"Concurrent operations: {CONCURRENT_KM}")
        print(f"Duration: {TEST_DURATION}s")
        print()

        metrics = await run_knowledge_load_test(
            base_url=API_URL,
            token=API_TOKEN,
            concurrent=CONCURRENT_KM,
            duration=TEST_DURATION,
            include_federation=True,
            include_evidence=True,
        )

        print("\nResults:")
        print(json.dumps(metrics.to_dict(), indent=2))

        if metrics.errors[:5]:
            print("\nSample errors:")
            for error in metrics.errors[:5]:
                print(f"  - {error}")

    asyncio.run(main())
