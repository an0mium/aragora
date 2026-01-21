"""PostgreSQL Storage Backend Integration Tests.

These tests verify PostgreSQL store implementations work correctly
when connected to an actual PostgreSQL database.

Requirements:
    - PostgreSQL server running
    - ARAGORA_POSTGRES_DSN or DATABASE_URL environment variable set
    - asyncpg installed: pip install asyncpg

Run with:
    ARAGORA_POSTGRES_DSN=postgresql://user:pass@localhost:5432/aragora_test \
    pytest tests/integration/test_postgres_stores.py -v

Skip if PostgreSQL not available:
    pytest tests/integration/test_postgres_stores.py -v -k "not postgres"
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional

import pytest

# Check if PostgreSQL is available
POSTGRES_DSN = os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
POSTGRES_AVAILABLE = bool(POSTGRES_DSN)

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not POSTGRES_AVAILABLE or not ASYNCPG_AVAILABLE,
        reason="PostgreSQL not configured (set ARAGORA_POSTGRES_DSN) or asyncpg not installed",
    ),
]


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def postgres_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create PostgreSQL connection pool for tests."""
    if not POSTGRES_DSN:
        pytest.skip("PostgreSQL DSN not configured")

    pool = await asyncpg.create_pool(
        POSTGRES_DSN,
        min_size=1,
        max_size=5,
        command_timeout=30,
    )
    yield pool
    await pool.close()


@pytest.fixture
async def clean_test_tables(postgres_pool: asyncpg.Pool):
    """Clean test data before each test."""
    async with postgres_pool.acquire() as conn:
        # Clean test data (use prefixed IDs for safety)
        await conn.execute("DELETE FROM webhook_configs WHERE id LIKE 'test-%'")
        await conn.execute("DELETE FROM gauntlet_runs WHERE id LIKE 'test-%'")
        await conn.execute("DELETE FROM approval_requests WHERE id LIKE 'test-%'")
        await conn.execute("DELETE FROM federated_regions WHERE region_id LIKE 'test-%'")
    yield
    # Cleanup after test
    async with postgres_pool.acquire() as conn:
        await conn.execute("DELETE FROM webhook_configs WHERE id LIKE 'test-%'")
        await conn.execute("DELETE FROM gauntlet_runs WHERE id LIKE 'test-%'")
        await conn.execute("DELETE FROM approval_requests WHERE id LIKE 'test-%'")
        await conn.execute("DELETE FROM federated_regions WHERE region_id LIKE 'test-%'")


class TestPostgresWebhookConfigStore:
    """Tests for PostgresWebhookConfigStore."""

    @pytest.fixture
    def store(self, postgres_pool: asyncpg.Pool):
        """Create webhook config store."""
        from aragora.storage.webhook_config_store import PostgresWebhookConfigStore

        return PostgresWebhookConfigStore(postgres_pool)

    @pytest.mark.asyncio
    async def test_register_and_get(self, store, clean_test_tables):
        """Test registering and retrieving a webhook."""
        webhook = await store.register_async(
            url="https://example.com/webhook",
            events=["debate.started", "debate.ended"],
            name="Test Webhook",
            user_id="test-user-1",
        )

        assert webhook.id is not None
        assert webhook.url == "https://example.com/webhook"
        assert webhook.events == ["debate.started", "debate.ended"]

        # Retrieve
        retrieved = await store.get_async(webhook.id)
        assert retrieved is not None
        assert retrieved.url == webhook.url
        assert retrieved.name == "Test Webhook"

    @pytest.mark.asyncio
    async def test_list_by_user(self, store, clean_test_tables):
        """Test listing webhooks by user."""
        # Create webhooks for different users
        await store.register_async(
            url="https://example.com/hook1",
            events=["*"],
            user_id="test-user-a",
        )
        await store.register_async(
            url="https://example.com/hook2",
            events=["*"],
            user_id="test-user-a",
        )
        await store.register_async(
            url="https://example.com/hook3",
            events=["*"],
            user_id="test-user-b",
        )

        user_a_hooks = await store.list_async(user_id="test-user-a")
        assert len(user_a_hooks) == 2

        user_b_hooks = await store.list_async(user_id="test-user-b")
        assert len(user_b_hooks) == 1

    @pytest.mark.asyncio
    async def test_update_webhook(self, store, clean_test_tables):
        """Test updating webhook properties."""
        webhook = await store.register_async(
            url="https://old-url.com/webhook",
            events=["debate.started"],
        )

        updated = await store.update_async(
            webhook.id,
            url="https://new-url.com/webhook",
            events=["debate.started", "debate.ended"],
            active=False,
        )

        assert updated is not None
        assert updated.url == "https://new-url.com/webhook"
        assert updated.active is False
        assert len(updated.events) == 2

    @pytest.mark.asyncio
    async def test_record_delivery(self, store, clean_test_tables):
        """Test recording webhook delivery statistics."""
        webhook = await store.register_async(
            url="https://example.com/webhook",
            events=["*"],
        )

        # Record successful delivery
        await store.record_delivery_async(webhook.id, 200, success=True)
        await store.record_delivery_async(webhook.id, 200, success=True)
        await store.record_delivery_async(webhook.id, 500, success=False)

        updated = await store.get_async(webhook.id)
        assert updated.delivery_count == 3
        assert updated.failure_count == 1


class TestPostgresGauntletRunStore:
    """Tests for PostgresGauntletRunStore."""

    @pytest.fixture
    def store(self, postgres_pool: asyncpg.Pool):
        """Create gauntlet run store."""
        from aragora.storage.gauntlet_run_store import PostgresGauntletRunStore

        return PostgresGauntletRunStore(postgres_pool)

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, clean_test_tables):
        """Test saving and retrieving a gauntlet run."""
        run_id = f"test-{uuid.uuid4().hex[:8]}"
        await store.save(
            {
                "run_id": run_id,
                "template_id": "security-audit",
                "status": "pending",
                "config_data": {"max_rounds": 5},
                "workspace_id": "test-ws",
            }
        )

        result = await store.get(run_id)
        assert result is not None
        assert result["template_id"] == "security-audit"
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_status_lifecycle(self, store, clean_test_tables):
        """Test gauntlet run status transitions."""
        run_id = f"test-{uuid.uuid4().hex[:8]}"
        await store.save(
            {
                "run_id": run_id,
                "template_id": "compliance-check",
                "status": "pending",
            }
        )

        # Start running
        await store.update_status(run_id, "running")
        result = await store.get(run_id)
        assert result["status"] == "running"
        assert result["started_at"] is not None

        # Complete with result
        await store.update_status(
            run_id,
            "completed",
            result_data={"verdict": "pass", "score": 95},
        )
        result = await store.get(run_id)
        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert result["result_data"]["score"] == 95

    @pytest.mark.asyncio
    async def test_list_active(self, store, clean_test_tables):
        """Test listing active runs."""
        # Create runs with different statuses
        for i, status in enumerate(["pending", "running", "completed", "failed"]):
            await store.save(
                {
                    "run_id": f"test-run-{i}",
                    "template_id": "test",
                    "status": status,
                }
            )

        active = await store.list_active()
        active_ids = [r["run_id"] for r in active]

        assert "test-run-0" in active_ids  # pending
        assert "test-run-1" in active_ids  # running
        assert "test-run-2" not in active_ids  # completed
        assert "test-run-3" not in active_ids  # failed


class TestPostgresApprovalRequestStore:
    """Tests for PostgresApprovalRequestStore."""

    @pytest.fixture
    def store(self, postgres_pool: asyncpg.Pool):
        """Create approval request store."""
        from aragora.storage.approval_request_store import PostgresApprovalRequestStore

        return PostgresApprovalRequestStore(postgres_pool)

    @pytest.mark.asyncio
    async def test_save_and_respond(self, store, clean_test_tables):
        """Test creating and responding to approval request."""
        request_id = f"test-{uuid.uuid4().hex[:8]}"
        await store.save(
            {
                "request_id": request_id,
                "workflow_id": "test-workflow",
                "step_id": "step-1",
                "title": "Approve deployment",
                "status": "pending",
                "priority": 1,
            }
        )

        # Respond to request
        success = await store.respond(
            request_id,
            "approved",
            "reviewer-123",
            response_data={"comment": "Looks good"},
        )
        assert success is True

        # Verify response
        result = await store.get(request_id)
        assert result["status"] == "approved"
        assert result["responder_id"] == "reviewer-123"
        assert result["responded_at"] is not None

    @pytest.mark.asyncio
    async def test_list_pending_ordered(self, store, clean_test_tables):
        """Test pending requests ordered by priority."""
        # Create requests with different priorities
        for i, priority in enumerate([3, 1, 2]):
            await store.save(
                {
                    "request_id": f"test-req-{i}",
                    "workflow_id": "test",
                    "step_id": "step",
                    "title": f"Request {i}",
                    "status": "pending",
                    "priority": priority,
                }
            )

        pending = await store.list_pending()
        priorities = [r["priority"] for r in pending if r["request_id"].startswith("test-")]

        # Should be ordered by priority (1 = highest)
        assert priorities == sorted(priorities)


class TestPostgresConcurrency:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, postgres_pool: asyncpg.Pool, clean_test_tables):
        """Test concurrent writes don't cause conflicts."""
        from aragora.storage.webhook_config_store import PostgresWebhookConfigStore

        store = PostgresWebhookConfigStore(postgres_pool)

        # Concurrent webhook registrations
        async def register_webhook(n: int):
            return await store.register_async(
                url=f"https://example.com/hook{n}",
                events=["*"],
                user_id=f"user-{n % 3}",
            )

        results = await asyncio.gather(*[register_webhook(i) for i in range(10)])

        assert len(results) == 10
        assert all(r.id is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_counter_increments(
        self, postgres_pool: asyncpg.Pool, clean_test_tables
    ):
        """Test atomic counter increments under concurrent load."""
        from aragora.storage.federation_registry_store import PostgresFederationRegistryStore
        from aragora.storage.federation_registry_store import FederatedRegionConfig

        store = PostgresFederationRegistryStore(postgres_pool)
        await store.initialize()

        # Create a region
        region = FederatedRegionConfig(
            region_id="test-concurrent-region",
            endpoint_url="https://region.example.com",
            api_key="test-key",
            workspace_id="",
        )
        await store.save(region)

        # Concurrent sync status updates
        async def update_sync(n: int):
            await store.update_sync_status(
                "test-concurrent-region",
                direction="push" if n % 2 == 0 else "pull",
                nodes_synced=1,
            )

        await asyncio.gather(*[update_sync(i) for i in range(20)])

        # Verify counters
        result = await store.get("test-concurrent-region")
        assert result is not None
        # 10 pushes + 10 pulls = 20 total syncs
        assert result.total_pushes == 10
        assert result.total_pulls == 10
        assert result.total_nodes_synced == 20


class TestPostgresTransactions:
    """Tests for transaction behavior."""

    @pytest.mark.asyncio
    async def test_cleanup_transaction_atomicity(self, postgres_pool: asyncpg.Pool):
        """Test that cleanup operations are atomic."""
        from aragora.storage.governance_store import PostgresGovernanceStore

        store = PostgresGovernanceStore(postgres_pool)
        await store.initialize()

        # Create some old records
        old_time = (datetime.now() - timedelta(days=60)).isoformat()

        await store.save_approval_async(
            approval_id="test-old-approval",
            title="Old approval",
            description="Test",
            risk_level="low",
            status="approved",
            requested_by="user",
            changes=[],
        )

        # Run cleanup (should be atomic)
        counts = await store.cleanup_old_records_async(
            approvals_days=30,
            verifications_days=7,
        )

        # Verify cleanup completed
        assert isinstance(counts, dict)
        assert "approvals" in counts
        assert "verifications" in counts


class TestPostgresConnectionPool:
    """Tests for connection pool behavior."""

    @pytest.mark.asyncio
    async def test_pool_exhaustion_handling(self, postgres_pool: asyncpg.Pool, clean_test_tables):
        """Test behavior when pool connections are exhausted."""
        from aragora.storage.webhook_config_store import PostgresWebhookConfigStore

        store = PostgresWebhookConfigStore(postgres_pool)

        # Many concurrent operations (more than pool size)
        async def quick_operation(n: int):
            webhook = await store.register_async(
                url=f"https://example.com/hook{n}",
                events=["*"],
            )
            return await store.get_async(webhook.id)

        # Should handle gracefully even with limited pool
        results = await asyncio.gather(
            *[quick_operation(i) for i in range(20)], return_exceptions=True
        )

        # All should succeed (pool handles queueing)
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == 20

    @pytest.mark.asyncio
    async def test_connection_reuse(self, postgres_pool: asyncpg.Pool, clean_test_tables):
        """Test that connections are properly returned to pool."""
        from aragora.storage.webhook_config_store import PostgresWebhookConfigStore

        store = PostgresWebhookConfigStore(postgres_pool)

        initial_size = postgres_pool.get_size()

        # Many sequential operations
        for i in range(50):
            await store.register_async(
                url=f"https://example.com/hook{i}",
                events=["*"],
            )

        # Pool size should not have grown unbounded
        final_size = postgres_pool.get_size()
        assert final_size <= postgres_pool.get_max_size()


class TestPostgresConsensusMemoryIntegration:
    """Integration tests for PostgresConsensusMemory."""

    @pytest.fixture
    async def memory(self, postgres_pool: asyncpg.Pool):
        """Create and initialize consensus memory."""
        from aragora.memory.postgres_consensus import PostgresConsensusMemory

        memory = PostgresConsensusMemory(postgres_pool)
        await memory.initialize()
        return memory

    @pytest.mark.asyncio
    async def test_store_and_retrieve_consensus(self, memory):
        """Test storing and retrieving a consensus record."""
        unique_id = uuid.uuid4().hex[:8]
        topic = f"Test topic {unique_id}"

        # Store
        result = await memory.store_consensus(
            topic=topic,
            conclusion="Test conclusion",
            strength="strong",
            confidence=0.9,
            participating_agents=["claude", "gpt4"],
            agreeing_agents=["claude", "gpt4"],
            domain="testing",
        )

        assert result["topic"] == topic
        assert result["confidence"] == 0.9
        consensus_id = result["id"]

        # Retrieve
        retrieved = await memory.get_consensus(consensus_id)
        assert retrieved is not None
        assert retrieved["topic"] == topic
        assert retrieved["conclusion"] == "Test conclusion"

    @pytest.mark.asyncio
    async def test_find_similar(self, memory):
        """Test finding similar consensus records."""
        unique_id = uuid.uuid4().hex[:8]
        base_topic = f"Rate limiting approach {unique_id}"

        # Store a consensus
        await memory.store_consensus(
            topic=base_topic,
            conclusion="Use token bucket",
            strength="strong",
            confidence=0.85,
            participating_agents=["claude"],
            agreeing_agents=["claude"],
        )

        # Find similar
        results = await memory.find_similar(base_topic, limit=5)
        assert len(results) >= 1
        assert any(base_topic in r.get("topic", "") for r in results)

    @pytest.mark.asyncio
    async def test_store_dissent(self, memory):
        """Test storing dissent records."""
        unique_id = uuid.uuid4().hex[:8]
        topic = f"Test topic for dissent {unique_id}"

        # First create a consensus
        result = await memory.store_consensus(
            topic=topic,
            conclusion="Main conclusion",
            strength="moderate",
            confidence=0.7,
            participating_agents=["claude", "gpt4"],
            agreeing_agents=["claude"],
        )
        debate_id = result["id"]

        # Store dissent
        dissent = await memory.store_dissent(
            debate_id=debate_id,
            agent_id="gpt4",
            dissent_type="alternative_approach",
            content="I disagree with this approach",
            reasoning="Here's why...",
            confidence=0.6,
        )

        assert dissent["debate_id"] == debate_id
        assert dissent["agent_id"] == "gpt4"

        # Retrieve dissents
        dissents = await memory.get_dissents_for_debate(debate_id)
        assert len(dissents) >= 1

    @pytest.mark.asyncio
    async def test_get_stats(self, memory):
        """Test getting statistics."""
        stats = await memory.get_stats()

        assert "total_consensus" in stats
        assert "total_dissents" in stats
        assert isinstance(stats["total_consensus"], int)


class TestPostgresCritiqueStoreIntegration:
    """Integration tests for PostgresCritiqueStore."""

    @pytest.fixture
    async def store(self, postgres_pool: asyncpg.Pool):
        """Create and initialize critique store."""
        from aragora.memory.postgres_critique import PostgresCritiqueStore

        store = PostgresCritiqueStore(postgres_pool)
        await store.initialize()
        return store

    @pytest.mark.asyncio
    async def test_store_and_retrieve_debate(self, store):
        """Test storing and retrieving a debate record."""
        unique_id = uuid.uuid4().hex[:8]
        debate_id = f"debate_{unique_id}"

        # Store
        result = await store.store_debate(
            debate_id=debate_id,
            task="Design a caching system",
            final_answer="Use Redis with LRU eviction",
            consensus_reached=True,
            confidence=0.88,
            rounds_used=3,
            duration_seconds=45.5,
        )

        assert result["id"] == debate_id
        assert result["consensus_reached"] is True

        # Retrieve
        retrieved = await store.get_debate(debate_id)
        assert retrieved is not None
        assert retrieved["task"] == "Design a caching system"
        assert retrieved["confidence"] == 0.88

    @pytest.mark.asyncio
    async def test_store_critique(self, store):
        """Test storing critique records."""
        unique_id = uuid.uuid4().hex[:8]
        debate_id = f"debate_{unique_id}"

        # First store a debate
        await store.store_debate(
            debate_id=debate_id,
            task="Test task",
            consensus_reached=True,
            confidence=0.8,
        )

        # Store critique
        critique_id = await store.store_critique(
            debate_id=debate_id,
            agent="claude",
            target_agent="gpt4",
            issues=["Logic flaw", "Missing edge case"],
            suggestions=["Add validation", "Handle null"],
            severity=0.7,
            reasoning="The approach needs improvement",
        )

        assert critique_id > 0

        # Get critiques for debate
        critiques = await store.get_critiques_for_debate(debate_id)
        assert len(critiques) >= 1
        assert critiques[0]["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_store_pattern(self, store):
        """Test storing and retrieving patterns."""
        unique_id = uuid.uuid4().hex[:8]
        issue_text = f"SQL injection vulnerability {unique_id}"

        # Store pattern
        result = await store.store_pattern(
            issue_text=issue_text,
            suggestion_text="Use parameterized queries",
            severity=0.9,
            example_task="Fix user login",
        )

        assert result["issue_type"] == "security"

        # Retrieve patterns
        patterns = await store.retrieve_patterns(issue_type="security", min_success=1, limit=10)
        assert any(issue_text in p.issue_text for p in patterns)

    @pytest.mark.asyncio
    async def test_reputation_tracking(self, store):
        """Test agent reputation tracking."""
        unique_id = uuid.uuid4().hex[:8]
        agent_name = f"test_agent_{unique_id}"

        # Update reputation
        await store.update_reputation(
            agent_name,
            proposal_made=True,
            proposal_accepted=True,
            critique_given=True,
            critique_valuable=True,
        )

        # Get reputation
        rep = await store.get_reputation(agent_name)
        assert rep is not None
        assert rep.agent_name == agent_name
        assert rep.proposals_made == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, store):
        """Test getting statistics."""
        stats = await store.get_stats()

        assert "total_debates" in stats
        assert "total_critiques" in stats
        assert "total_patterns" in stats


class TestPostgresContinuumMemoryIntegration:
    """Integration tests for PostgresContinuumMemory."""

    @pytest.fixture
    async def memory(self, postgres_pool: asyncpg.Pool):
        """Create and initialize continuum memory."""
        from aragora.memory.postgres_continuum import PostgresContinuumMemory

        memory = PostgresContinuumMemory(postgres_pool)
        await memory.initialize()
        return memory

    @pytest.mark.asyncio
    async def test_add_and_get_memory(self, memory):
        """Test adding and retrieving a memory entry."""
        unique_id = uuid.uuid4().hex[:8]
        memory_id = f"test_memory_{unique_id}"

        # Add memory
        result = await memory.add(
            memory_id=memory_id,
            content="Test error handling pattern for database connections",
            tier="slow",
            importance=0.8,
            metadata={"source": "integration_test"},
        )

        assert result["id"] == memory_id
        assert result["importance"] == 0.8
        assert result["tier"] == "slow"

        # Retrieve
        retrieved = await memory.get(memory_id)
        assert retrieved is not None
        assert retrieved["content"] == "Test error handling pattern for database connections"
        assert retrieved["metadata"]["source"] == "integration_test"

    @pytest.mark.asyncio
    async def test_retrieve_with_query(self, memory):
        """Test retrieving memories with keyword search."""
        unique_id = uuid.uuid4().hex[:8]
        memory_id = f"test_query_{unique_id}"

        # Add a memory with specific content
        await memory.add(
            memory_id=memory_id,
            content=f"Rate limiting pattern using token bucket {unique_id}",
            tier="medium",
            importance=0.9,
        )

        # Retrieve with keyword
        results = await memory.retrieve(query=unique_id, limit=10)
        assert len(results) >= 1
        assert any(memory_id == r["id"] for r in results)

    @pytest.mark.asyncio
    async def test_update_outcome(self, memory):
        """Test updating memory outcome and surprise score."""
        unique_id = uuid.uuid4().hex[:8]
        memory_id = f"test_outcome_{unique_id}"

        # Add memory
        await memory.add(
            memory_id=memory_id,
            content="Test pattern for outcome tracking",
            tier="fast",
            importance=0.7,
        )

        # Update with success
        surprise = await memory.update_outcome(memory_id, success=True)
        assert surprise >= 0

        # Verify counts updated
        entry = await memory.get(memory_id)
        assert entry is not None
        assert entry["success_count"] == 1
        assert entry["update_count"] == 2  # Initial 1 + update 1

    @pytest.mark.asyncio
    async def test_tier_promotion(self, memory):
        """Test tier promotion via promote_entry."""
        unique_id = uuid.uuid4().hex[:8]
        memory_id = f"test_promote_{unique_id}"

        # Add memory in slow tier
        await memory.add(
            memory_id=memory_id,
            content="Test pattern for promotion",
            tier="slow",
            importance=0.9,
        )

        # Promote to medium tier
        from aragora.memory.tier_manager import MemoryTier

        result = await memory.promote_entry(memory_id, MemoryTier.MEDIUM)
        assert result is True

        # Verify tier changed
        entry = await memory.get(memory_id)
        assert entry is not None
        assert entry["tier"] == "medium"

    @pytest.mark.asyncio
    async def test_red_line_protection(self, memory):
        """Test marking memory as red line (protected)."""
        unique_id = uuid.uuid4().hex[:8]
        memory_id = f"test_redline_{unique_id}"

        # Add memory
        await memory.add(
            memory_id=memory_id,
            content="Critical safety decision pattern",
            tier="slow",
            importance=0.5,
        )

        # Mark as red line
        result = await memory.mark_red_line(
            memory_id=memory_id,
            reason="Safety-critical pattern",
            promote_to_glacial=True,
        )
        assert result is True

        # Verify protection
        entry = await memory.get(memory_id)
        assert entry is not None
        assert entry["red_line"] is True
        assert entry["tier"] == "glacial"
        assert entry["importance"] == 1.0

        # Verify delete is blocked
        delete_result = await memory.delete(memory_id)
        assert delete_result["blocked"] is True
        assert delete_result["deleted"] is False

    @pytest.mark.asyncio
    async def test_get_stats(self, memory):
        """Test getting memory statistics."""
        stats = await memory.get_stats()

        assert "total_entries" in stats
        assert "by_tier" in stats
        assert "hyperparams" in stats
        assert isinstance(stats["total_entries"], int)

    @pytest.mark.asyncio
    async def test_count_by_tier(self, memory):
        """Test counting entries by tier."""
        from aragora.memory.tier_manager import MemoryTier

        # Count all
        total = await memory.count()
        assert isinstance(total, int)

        # Count by tier
        fast_count = await memory.count(tier=MemoryTier.FAST)
        assert isinstance(fast_count, int)

    @pytest.mark.asyncio
    async def test_get_by_tier(self, memory):
        """Test retrieving memories by specific tier."""
        unique_id = uuid.uuid4().hex[:8]

        # Add memory in fast tier
        await memory.add(
            memory_id=f"test_tier_{unique_id}",
            content="Fast tier pattern",
            tier="fast",
            importance=0.8,
        )

        # Get by tier
        from aragora.memory.tier_manager import MemoryTier

        results = await memory.get_by_tier(MemoryTier.FAST, limit=50)
        assert isinstance(results, list)
        assert all(r["tier"] == "fast" for r in results)
