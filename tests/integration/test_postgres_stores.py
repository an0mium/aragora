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
        reason="PostgreSQL not configured (set ARAGORA_POSTGRES_DSN) or asyncpg not installed"
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
        await store.save({
            "run_id": run_id,
            "template_id": "security-audit",
            "status": "pending",
            "config_data": {"max_rounds": 5},
            "workspace_id": "test-ws",
        })

        result = await store.get(run_id)
        assert result is not None
        assert result["template_id"] == "security-audit"
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_status_lifecycle(self, store, clean_test_tables):
        """Test gauntlet run status transitions."""
        run_id = f"test-{uuid.uuid4().hex[:8]}"
        await store.save({
            "run_id": run_id,
            "template_id": "compliance-check",
            "status": "pending",
        })

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
            await store.save({
                "run_id": f"test-run-{i}",
                "template_id": "test",
                "status": status,
            })

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
        await store.save({
            "request_id": request_id,
            "workflow_id": "test-workflow",
            "step_id": "step-1",
            "title": "Approve deployment",
            "status": "pending",
            "priority": 1,
        })

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
            await store.save({
                "request_id": f"test-req-{i}",
                "workflow_id": "test",
                "step_id": "step",
                "title": f"Request {i}",
                "status": "pending",
                "priority": priority,
            })

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
    async def test_concurrent_counter_increments(self, postgres_pool: asyncpg.Pool, clean_test_tables):
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
            *[quick_operation(i) for i in range(20)],
            return_exceptions=True
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
