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
from typing import Optional
from collections.abc import AsyncGenerator

import pytest

# Check if PostgreSQL is available
POSTGRES_DSN = os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
POSTGRES_AVAILABLE = bool(POSTGRES_DSN)


import asyncpg


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason="PostgreSQL not configured (set ARAGORA_POSTGRES_DSN)",
    ),
]

# Global pool to avoid recreating for each test
_global_pool: asyncpg.Pool | None = None


@pytest.fixture
async def postgres_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create PostgreSQL connection pool for tests (reuses global pool)."""
    global _global_pool
    if not POSTGRES_DSN:
        pytest.skip("PostgreSQL DSN not configured")

    if _global_pool is None or _global_pool._closed:
        _global_pool = await asyncpg.create_pool(
            POSTGRES_DSN,
            min_size=1,
            max_size=5,
            command_timeout=30,
        )
    yield _global_pool
    # Don't close - reuse across tests


@pytest.fixture
async def clean_test_tables(postgres_pool: asyncpg.Pool):
    """Reset tested store schemas to current definitions before each test."""

    async def _reset_schema(conn: asyncpg.Connection) -> None:
        from aragora.storage.approval_request_store import PostgresApprovalRequestStore
        from aragora.storage.federation_registry_store import PostgresFederationRegistryStore
        from aragora.storage.gauntlet_run_store import PostgresGauntletRunStore
        from aragora.storage.webhook_config_store import PostgresWebhookConfigStore

        # Keep integration tests resilient against stale/shared CI databases.
        await conn.execute("DROP TABLE IF EXISTS webhook_configs CASCADE")
        await conn.execute("DROP TABLE IF EXISTS gauntlet_runs CASCADE")
        await conn.execute("DROP TABLE IF EXISTS approval_requests CASCADE")
        await conn.execute("DROP TABLE IF EXISTS federated_regions CASCADE")

        await conn.execute(PostgresWebhookConfigStore.INITIAL_SCHEMA)
        await conn.execute(PostgresGauntletRunStore.SCHEMA_SQL)
        await conn.execute(PostgresApprovalRequestStore.SCHEMA_SQL)
        await conn.execute(PostgresFederationRegistryStore.INITIAL_SCHEMA)

    async with postgres_pool.acquire() as conn:
        await _reset_schema(conn)
    yield


class TestPostgresWebhookConfigStore:
    """Tests for PostgresWebhookConfigStore."""

    @pytest.fixture
    async def store(self, postgres_pool: asyncpg.Pool):
        """Create webhook config store."""
        from aragora.storage.webhook_config_store import PostgresWebhookConfigStore

        store = PostgresWebhookConfigStore(postgres_pool)
        await store.initialize()
        return store

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
    async def store(self, postgres_pool: asyncpg.Pool):
        """Create gauntlet run store."""
        from aragora.storage.gauntlet_run_store import PostgresGauntletRunStore

        store = PostgresGauntletRunStore(postgres_pool)
        await store.initialize()
        return store

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
    async def store(self, postgres_pool: asyncpg.Pool):
        """Create approval request store."""
        from aragora.storage.approval_request_store import PostgresApprovalRequestStore

        store = PostgresApprovalRequestStore(postgres_pool)
        await store.initialize()
        return store

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


class TestPostgresEloDatabaseIntegration:
    """Integration tests for PostgresEloDatabase."""

    @pytest.fixture
    async def db(self, postgres_pool: asyncpg.Pool):
        """Create and initialize ELO database."""
        from aragora.ranking.postgres_database import PostgresEloDatabase

        db = PostgresEloDatabase(postgres_pool)
        await db.initialize()
        return db

    @pytest.mark.asyncio
    async def test_set_and_get_rating(self, db):
        """Test setting and retrieving agent ratings."""
        unique_id = uuid.uuid4().hex[:8]
        agent_name = f"test_agent_{unique_id}"

        # Set rating
        await db.set_rating(
            agent_name=agent_name,
            elo=1600.0,
            domain_elos={"coding": 1700.0, "debate": 1550.0},
            wins=10,
            losses=5,
            draws=2,
        )

        # Get rating
        rating = await db.get_rating(agent_name)
        assert rating is not None
        assert rating["agent_name"] == agent_name
        assert rating["elo"] == 1600.0
        assert rating["wins"] == 10
        assert rating["losses"] == 5

    @pytest.mark.asyncio
    async def test_record_match(self, db):
        """Test recording match results."""
        unique_id = uuid.uuid4().hex[:8]
        winner = f"winner_{unique_id}"
        loser = f"loser_{unique_id}"

        # Initialize ratings
        await db.set_rating(agent_name=winner, elo=1500.0)
        await db.set_rating(agent_name=loser, elo=1500.0)

        # Record match
        match_id = await db.record_match(
            winner=winner,
            loser=loser,
            domain="coding",
            debate_id=f"debate_{unique_id}",
            winner_elo_before=1500.0,
            loser_elo_before=1500.0,
            winner_elo_after=1516.0,
            loser_elo_after=1484.0,
        )

        assert match_id is not None
        assert isinstance(match_id, int)

    @pytest.mark.asyncio
    async def test_get_leaderboard(self, db):
        """Test retrieving the leaderboard."""
        unique_id = uuid.uuid4().hex[:8]

        # Create a few agents with different ratings
        for i, elo in enumerate([1800, 1600, 1400]):
            await db.set_rating(
                agent_name=f"leaderboard_{unique_id}_{i}",
                elo=float(elo),
            )

        # Get leaderboard
        leaderboard = await db.get_leaderboard(limit=10)
        assert isinstance(leaderboard, list)

    @pytest.mark.asyncio
    async def test_update_rating(self, db):
        """Test updating an existing rating."""
        unique_id = uuid.uuid4().hex[:8]
        agent_name = f"update_test_{unique_id}"

        # Set initial rating
        await db.set_rating(agent_name=agent_name, elo=1500.0, wins=0, losses=0)

        # Update with new values
        await db.set_rating(agent_name=agent_name, elo=1550.0, wins=5, losses=2)

        # Verify update
        rating = await db.get_rating(agent_name)
        assert rating is not None
        assert rating["elo"] == 1550.0
        assert rating["wins"] == 5

    @pytest.mark.asyncio
    async def test_get_match_history(self, db):
        """Test retrieving match history for an agent."""
        unique_id = uuid.uuid4().hex[:8]
        agent = f"history_{unique_id}"
        opponent = f"opponent_{unique_id}"

        # Initialize ratings
        await db.set_rating(agent_name=agent, elo=1500.0)
        await db.set_rating(agent_name=opponent, elo=1500.0)

        # Record some matches
        await db.record_match(
            winner=agent,
            loser=opponent,
            domain="general",
            debate_id=f"match1_{unique_id}",
            winner_elo_before=1500.0,
            loser_elo_before=1500.0,
            winner_elo_after=1516.0,
            loser_elo_after=1484.0,
        )

        # Get history
        history = await db.get_match_history(agent, limit=10)
        assert isinstance(history, list)
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_get_stats(self, db):
        """Test getting database statistics."""
        stats = await db.get_stats()

        assert "total_agents" in stats
        assert "total_matches" in stats
        assert isinstance(stats["total_agents"], int)
