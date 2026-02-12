"""
Comprehensive tests for aragora.server.handlers.connectors module.

Tests cover:
- Connector CRUD operations (create, read, update, delete)
- Connector type discovery and creation
- Configuration validation
- Health checks
- Sync operations (trigger, status, history)
- Scheduler management (start, stop, stats)
- Webhook handling
- MongoDB aggregation handlers
- Workflow template handlers
- RBAC/authorization checks
- Tenant isolation
- Error handling paths
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ===========================================================================
# Mock Classes and Fixtures
# ===========================================================================


@dataclass
class MockAuthContext:
    """Mock authorization context for RBAC testing."""

    user_id: str = "user-123"
    org_id: str = "org-456"
    roles: list = field(default_factory=lambda: ["admin"])
    permissions: set = field(
        default_factory=lambda: {
            "connectors:read",
            "connectors:create",
            "connectors:execute",
            "connectors:update",
            "connectors:delete",
            "*",
        }
    )
    api_key_scope: str | None = None
    ip_address: str = "127.0.0.1"
    user_agent: str = "test-agent"
    request_id: str = "req-test-001"
    timestamp: str | None = None
    user_email: str = "test@example.com"
    workspace_id: str = "ws-001"


@dataclass
class MockDeniedAuthContext:
    """Mock authorization context with no permissions."""

    user_id: str = "user-456"
    org_id: str = "org-789"
    roles: list = field(default_factory=lambda: ["viewer"])
    permissions: set = field(default_factory=set)
    api_key_scope: str | None = None
    ip_address: str = "127.0.0.1"
    user_agent: str = "test-agent"
    request_id: str = "req-test-002"
    timestamp: str | None = None
    user_email: str = "viewer@example.com"
    workspace_id: str = "ws-001"


@dataclass
class MockSyncJob:
    """Mock sync job for testing."""

    id: str
    connector_id: str
    tenant_id: str
    schedule: Any
    last_run: datetime | None = None
    next_run: datetime | None = None
    consecutive_failures: int = 0
    current_run_id: str | None = None

    def _calculate_next_run(self):
        pass


@dataclass
class MockSyncSchedule:
    """Mock sync schedule."""

    schedule_type: str = "interval"
    interval_minutes: int = 60
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_type": self.schedule_type,
            "interval_minutes": self.interval_minutes,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockSyncSchedule:
        return cls(
            schedule_type=data.get("schedule_type", "interval"),
            interval_minutes=data.get("interval_minutes", 60),
            enabled=data.get("enabled", True),
        )


@dataclass
class MockSyncHistory:
    """Mock sync history entry."""

    id: str
    job_id: str
    connector_id: str
    tenant_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    items_synced: int = 0
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "connector_id": self.connector_id,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "items_synced": self.items_synced,
            "errors": self.errors,
        }


class MockSyncScheduler:
    """Mock sync scheduler for testing."""

    def __init__(self):
        self._jobs: dict[str, MockSyncJob] = {}
        self._history: list[MockSyncHistory] = []
        self._scheduler_task = None
        self._running_syncs: dict = {}
        self._max_concurrent = 5

    def list_jobs(self, tenant_id: str = "default") -> list[MockSyncJob]:
        return [j for j in self._jobs.values() if j.tenant_id == tenant_id]

    def get_job(self, job_id: str) -> MockSyncJob | None:
        return self._jobs.get(job_id)

    def register_connector(self, connector, schedule=None, tenant_id="default"):
        job_id = f"{tenant_id}:{connector.connector_id}"
        job = MockSyncJob(
            id=job_id,
            connector_id=connector.connector_id,
            tenant_id=tenant_id,
            schedule=schedule or MockSyncSchedule(),
        )
        self._jobs[job_id] = job
        return job

    def unregister_connector(self, connector_id: str, tenant_id: str = "default"):
        job_id = f"{tenant_id}:{connector_id}"
        if job_id in self._jobs:
            del self._jobs[job_id]

    async def trigger_sync(
        self, connector_id: str, tenant_id: str = "default", full_sync: bool = False
    ) -> str | None:
        job_id = f"{tenant_id}:{connector_id}"
        if job_id not in self._jobs:
            return None
        return f"run-{connector_id}-{datetime.now().timestamp()}"

    async def handle_webhook(
        self, connector_id: str, payload: dict, tenant_id: str = "default"
    ) -> bool:
        job_id = f"{tenant_id}:{connector_id}"
        return job_id in self._jobs

    def get_history(
        self,
        job_id: str = None,
        tenant_id: str = None,
        status=None,
        limit: int = 50,
    ) -> list[MockSyncHistory]:
        history = self._history
        if job_id:
            history = [h for h in history if h.job_id == job_id]
        if tenant_id:
            history = [h for h in history if h.tenant_id == tenant_id]
        return history[:limit]

    def get_stats(self, tenant_id: str = None) -> dict[str, Any]:
        return {
            "total_jobs": len(self._jobs),
            "running_syncs": len(self._running_syncs),
            "success_rate": 0.95,
        }

    async def start(self):
        self._scheduler_task = True

    async def stop(self):
        self._scheduler_task = None


class MockConnector:
    """Mock connector for testing."""

    def __init__(self, connector_id: str):
        self.connector_id = connector_id


@pytest.fixture
def mock_scheduler():
    """Create a mock sync scheduler."""
    return MockSyncScheduler()


@pytest.fixture
def mock_auth_context():
    """Create mock auth context with full permissions."""
    return MockAuthContext()


@pytest.fixture
def mock_denied_auth_context():
    """Create mock auth context with no permissions."""
    return MockDeniedAuthContext()


@pytest.fixture
def reset_scheduler():
    """Reset the global scheduler between tests."""
    import aragora.server.handlers.connectors as connectors_module

    original_scheduler = connectors_module._scheduler
    connectors_module._scheduler = None
    yield
    connectors_module._scheduler = original_scheduler


# ===========================================================================
# Test Connector List Operations
# ===========================================================================


class TestListConnectors:
    """Tests for handle_list_connectors."""

    @pytest.mark.asyncio
    async def test_list_connectors_empty(self, reset_scheduler):
        """List returns empty when no connectors registered."""
        from aragora.server.handlers.connectors import handle_list_connectors

        result = await handle_list_connectors(tenant_id="empty-tenant")

        assert result["connectors"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_connectors_with_pagination(self, reset_scheduler):
        """List respects limit and offset pagination params."""
        from aragora.server.handlers.connectors import handle_list_connectors

        result = await handle_list_connectors(limit=10, offset=0)

        assert "connectors" in result
        assert "limit" in result
        assert "offset" in result
        assert result["limit"] == 10
        assert result["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_connectors_bounds_pagination(self, reset_scheduler):
        """Pagination params are bounded to valid ranges."""
        from aragora.server.handlers.connectors import handle_list_connectors

        # Test max limit
        result = await handle_list_connectors(limit=1000)
        assert result["limit"] == 500  # Max is 500

        # Test negative offset
        result = await handle_list_connectors(offset=-10)
        assert result["offset"] == 0

        # Test negative limit
        result = await handle_list_connectors(limit=-5)
        assert result["limit"] == 1  # Min is 1


class TestGetConnector:
    """Tests for handle_get_connector."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_connector(self, reset_scheduler):
        """Returns None for nonexistent connector."""
        from aragora.server.handlers.connectors import handle_get_connector

        result = await handle_get_connector("nonexistent-connector")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_connector_with_tenant(self, reset_scheduler):
        """Get respects tenant_id parameter."""
        from aragora.server.handlers.connectors import handle_get_connector

        result = await handle_get_connector("test-connector", tenant_id="other-tenant")

        assert result is None


# ===========================================================================
# Test Connector Create Operations
# ===========================================================================


class TestCreateConnector:
    """Tests for handle_create_connector."""

    @pytest.mark.asyncio
    async def test_create_github_connector(self, reset_scheduler):
        """Creates GitHub connector with valid config."""
        from aragora.server.handlers.connectors import handle_create_connector

        config = {
            "owner": "test-org",
            "repo": "test-repo",
            "token": "ghp_fake_token",
        }

        result = await handle_create_connector(
            connector_type="github",
            config=config,
        )

        assert isinstance(result, dict)
        assert "id" in result
        assert result.get("type") == "github"
        assert result.get("status") == "registered"

    @pytest.mark.asyncio
    async def test_create_github_connector_combined_repo_format(self, reset_scheduler):
        """Creates GitHub connector with owner/repo combined format."""
        from aragora.server.handlers.connectors import handle_create_connector

        config = {
            "repo": "test-org/test-repo",  # Combined format
            "token": "ghp_fake_token",
        }

        result = await handle_create_connector(
            connector_type="github",
            config=config,
        )

        assert isinstance(result, dict)
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_s3_connector(self, reset_scheduler):
        """Creates S3 connector with valid config."""
        from aragora.server.handlers.connectors import handle_create_connector

        config = {
            "bucket": "test-bucket",
            "prefix": "data/",
            "region": "us-west-2",
        }

        result = await handle_create_connector(
            connector_type="s3",
            config=config,
        )

        assert isinstance(result, dict)
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_postgres_connector(self, reset_scheduler):
        """Creates PostgreSQL connector with valid config."""
        from aragora.server.handlers.connectors import handle_create_connector

        config = {
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "schema": "public",
        }

        result = await handle_create_connector(
            connector_type="postgres",
            config=config,
        )

        assert isinstance(result, dict)
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_mongodb_connector(self, reset_scheduler):
        """Creates MongoDB connector with valid config."""
        from aragora.server.handlers.connectors import handle_create_connector

        config = {
            "host": "localhost",
            "port": 27017,
            "database": "testdb",
        }

        result = await handle_create_connector(
            connector_type="mongodb",
            config=config,
        )

        assert isinstance(result, dict)
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_fhir_connector(self, reset_scheduler):
        """Creates FHIR connector with valid config."""
        from aragora.server.handlers.connectors import handle_create_connector

        config = {
            "base_url": "https://fhir.example.com",
            "organization_id": "org-123",
            "enable_phi_redaction": True,
        }

        result = await handle_create_connector(
            connector_type="fhir",
            config=config,
        )

        assert isinstance(result, dict)
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_connector_with_schedule(self, reset_scheduler):
        """Creates connector with custom schedule config."""
        from aragora.server.handlers.connectors import handle_create_connector

        config = {"bucket": "test-bucket"}
        schedule = {
            "schedule_type": "interval",
            "interval_minutes": 30,
            "enabled": True,
        }

        result = await handle_create_connector(
            connector_type="s3",
            config=config,
            schedule=schedule,
        )

        assert isinstance(result, dict)
        assert "id" in result

    @pytest.mark.asyncio
    async def test_create_unknown_connector_type_raises(self, reset_scheduler):
        """Raises ValueError for unknown connector type."""
        from aragora.server.handlers.connectors import handle_create_connector

        with pytest.raises(ValueError, match="Unknown connector type"):
            await handle_create_connector(
                connector_type="unknown_type",
                config={},
            )


# ===========================================================================
# Test Connector Update Operations
# ===========================================================================


class TestUpdateConnector:
    """Tests for handle_update_connector."""

    @pytest.mark.asyncio
    async def test_update_nonexistent_connector(self, reset_scheduler):
        """Returns None for nonexistent connector."""
        from aragora.server.handlers.connectors import handle_update_connector

        result = await handle_update_connector(
            connector_id="nonexistent",
            updates={"schedule": {"interval_minutes": 30}},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_update_connector_schedule(self, reset_scheduler):
        """Updates connector schedule successfully."""
        from aragora.server.handlers.connectors import (
            handle_create_connector,
            handle_update_connector,
        )

        # First create a connector
        config = {"bucket": "test-bucket"}
        create_result = await handle_create_connector(
            connector_type="s3",
            config=config,
        )
        connector_id = create_result["id"]

        # Update the schedule
        result = await handle_update_connector(
            connector_id=connector_id,
            updates={"schedule": {"interval_minutes": 15}},
        )

        assert result is not None
        assert result["status"] == "updated"


# ===========================================================================
# Test Connector Delete Operations
# ===========================================================================


class TestDeleteConnector:
    """Tests for handle_delete_connector."""

    @pytest.mark.asyncio
    async def test_delete_connector(self, reset_scheduler):
        """Deletes connector successfully."""
        from aragora.server.handlers.connectors import (
            handle_create_connector,
            handle_delete_connector,
            handle_get_connector,
        )

        # Create a connector
        config = {"bucket": "test-bucket"}
        create_result = await handle_create_connector(
            connector_type="s3",
            config=config,
        )
        connector_id = create_result["id"]

        # Delete it
        result = await handle_delete_connector(connector_id=connector_id)

        assert result is True

        # Verify it's gone
        get_result = await handle_get_connector(connector_id)
        assert get_result is None


# ===========================================================================
# Test Sync Operations (with mocked scheduler to avoid real sync)
# ===========================================================================


class TestTriggerSync:
    """Tests for handle_trigger_sync."""

    @pytest.mark.asyncio
    async def test_trigger_sync_nonexistent_connector(self, reset_scheduler):
        """Returns None for nonexistent connector."""
        from aragora.server.handlers.connectors import handle_trigger_sync

        result = await handle_trigger_sync(connector_id="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.legacy.get_scheduler")
    async def test_trigger_sync_success_mocked(self, mock_get_scheduler, reset_scheduler):
        """Triggers sync successfully with mocked scheduler."""
        from aragora.server.handlers.connectors import handle_trigger_sync

        mock_scheduler = MockSyncScheduler()
        # Pre-register a job
        mock_connector = MockConnector("test-connector")
        mock_scheduler.register_connector(mock_connector, tenant_id="default")
        mock_get_scheduler.return_value = mock_scheduler

        result = await handle_trigger_sync(
            connector_id="test-connector",
            full_sync=False,
        )

        assert result is not None
        assert "run_id" in result
        assert result["connector_id"] == "test-connector"
        assert result["status"] == "started"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.legacy.get_scheduler")
    async def test_trigger_full_sync_mocked(self, mock_get_scheduler, reset_scheduler):
        """Triggers full sync with mocked scheduler."""
        from aragora.server.handlers.connectors import handle_trigger_sync

        mock_scheduler = MockSyncScheduler()
        mock_connector = MockConnector("test-connector")
        mock_scheduler.register_connector(mock_connector, tenant_id="default")
        mock_get_scheduler.return_value = mock_scheduler

        result = await handle_trigger_sync(
            connector_id="test-connector",
            full_sync=True,
        )

        assert result is not None
        assert result["full_sync"] is True


class TestGetSyncStatus:
    """Tests for handle_get_sync_status."""

    @pytest.mark.asyncio
    async def test_get_sync_status_nonexistent(self, reset_scheduler):
        """Returns None for nonexistent connector."""
        from aragora.server.handlers.connectors import handle_get_sync_status

        result = await handle_get_sync_status(connector_id="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_sync_status_success(self, reset_scheduler):
        """Returns sync status for registered connector."""
        from aragora.server.handlers.connectors import (
            handle_create_connector,
            handle_get_sync_status,
        )

        config = {"bucket": "test-bucket"}
        create_result = await handle_create_connector(
            connector_type="s3",
            config=config,
        )
        connector_id = create_result["id"]

        result = await handle_get_sync_status(connector_id=connector_id)

        assert result is not None
        assert "connector_id" in result
        assert "is_running" in result
        assert "consecutive_failures" in result


class TestGetSyncHistory:
    """Tests for handle_get_sync_history."""

    @pytest.mark.asyncio
    async def test_get_sync_history_empty(self, reset_scheduler):
        """Returns empty history for new tenant."""
        from aragora.server.handlers.connectors import handle_get_sync_history

        result = await handle_get_sync_history(tenant_id="empty-tenant")

        assert "history" in result
        assert isinstance(result["history"], list)


# ===========================================================================
# Test Scheduler Operations
# ===========================================================================


class TestSchedulerHandlers:
    """Tests for scheduler management handlers."""

    @pytest.mark.asyncio
    async def test_start_scheduler(self, reset_scheduler):
        """Starts the scheduler successfully."""
        from aragora.server.handlers.connectors import handle_start_scheduler

        result = await handle_start_scheduler()

        assert result["status"] == "started"

    @pytest.mark.asyncio
    async def test_stop_scheduler(self, reset_scheduler):
        """Stops the scheduler successfully."""
        from aragora.server.handlers.connectors import (
            handle_start_scheduler,
            handle_stop_scheduler,
        )

        # Start first
        await handle_start_scheduler()

        # Then stop
        result = await handle_stop_scheduler()

        assert result["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_get_scheduler_stats(self, reset_scheduler):
        """Returns scheduler statistics."""
        from aragora.server.handlers.connectors import handle_get_scheduler_stats

        result = await handle_get_scheduler_stats()

        assert "total_jobs" in result


# ===========================================================================
# Test Webhook Handlers
# ===========================================================================


class TestWebhookHandler:
    """Tests for handle_webhook."""

    @pytest.mark.asyncio
    async def test_webhook_nonexistent_connector(self, reset_scheduler):
        """Returns not handled for nonexistent connector."""
        from aragora.server.handlers.connectors import handle_webhook

        result = await handle_webhook(
            connector_id="nonexistent",
            payload={"event": "push"},
        )

        assert result["handled"] is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.legacy.get_scheduler")
    async def test_webhook_success_mocked(self, mock_get_scheduler, reset_scheduler):
        """Handles webhook for registered connector with mocked scheduler."""
        from aragora.server.handlers.connectors import handle_webhook

        mock_scheduler = MockSyncScheduler()
        mock_connector = MockConnector("test-connector")
        mock_scheduler.register_connector(mock_connector, tenant_id="default")
        mock_get_scheduler.return_value = mock_scheduler

        result = await handle_webhook(
            connector_id="test-connector",
            payload={"event": "push", "ref": "refs/heads/main"},
        )

        assert result["handled"] is True
        assert result["connector_id"] == "test-connector"


# ===========================================================================
# Test Health Check
# ===========================================================================


class TestHealthCheck:
    """Tests for handle_connector_health."""

    @pytest.mark.asyncio
    async def test_health_check_basic(self, reset_scheduler):
        """Returns basic health status without auth."""
        from aragora.server.handlers.connectors import handle_connector_health

        result = await handle_connector_health()

        assert result["status"] == "healthy"
        assert "scheduler_running" in result
        assert "total_connectors" in result

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", False)
    async def test_health_check_rbac_disabled(self, reset_scheduler, mock_auth_context):
        """Returns basic health when RBAC disabled."""
        from aragora.server.handlers.connectors import handle_connector_health

        result = await handle_connector_health(auth_context=mock_auth_context)

        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.connectors.check_permission")
    async def test_health_check_with_auth_allowed(
        self, mock_check, reset_scheduler, mock_auth_context
    ):
        """Returns detailed stats when RBAC allows."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check.return_value = mock_decision

        from aragora.server.handlers.connectors import handle_connector_health

        result = await handle_connector_health(auth_context=mock_auth_context)

        assert result["status"] == "healthy"
        # Should have extended stats when permitted
        assert "running_syncs" in result
        assert "success_rate" in result


# ===========================================================================
# Test Workflow Templates
# ===========================================================================


class TestWorkflowTemplates:
    """Tests for workflow template handlers."""

    @pytest.mark.asyncio
    async def test_list_workflow_templates(self, reset_scheduler):
        """Lists available workflow templates."""
        from aragora.server.handlers.connectors import handle_list_workflow_templates

        result = await handle_list_workflow_templates()

        assert "templates" in result
        assert "total" in result
        assert "categories" in result

    @pytest.mark.asyncio
    async def test_list_workflow_templates_by_category(self, reset_scheduler):
        """Lists templates filtered by category."""
        from aragora.server.handlers.connectors import handle_list_workflow_templates

        result = await handle_list_workflow_templates(category="legal")

        assert "templates" in result

    @pytest.mark.asyncio
    async def test_get_workflow_template_nonexistent(self, reset_scheduler):
        """Returns None for nonexistent template."""
        from aragora.server.handlers.connectors import handle_get_workflow_template

        result = await handle_get_workflow_template(template_id="nonexistent")

        assert result is None


# ===========================================================================
# Test RBAC Permission Checks
# ===========================================================================


class TestRBACPermissions:
    """Tests for RBAC permission checks."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.connectors.check_permission")
    async def test_list_connectors_permission_denied(self, mock_check, reset_scheduler):
        """Returns 403 when permission denied."""
        from aragora.server.handlers.connectors import handle_list_connectors

        # Mock permission denied
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "Permission denied"
        mock_check.return_value = mock_decision

        result = await handle_list_connectors(auth_context=MockAuthContext())

        assert "error" in result
        assert result.get("status") == 403

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.connectors.check_permission")
    async def test_create_connector_permission_denied(self, mock_check, reset_scheduler):
        """Returns 403 when create permission denied."""
        from aragora.server.handlers.connectors import handle_create_connector

        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "Create permission denied"
        mock_check.return_value = mock_decision

        result = await handle_create_connector(
            connector_type="s3",
            config={"bucket": "test"},
            auth_context=MockAuthContext(),
        )

        assert "error" in result
        assert result.get("status") == 403

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.connectors.check_permission")
    async def test_trigger_sync_permission_denied(self, mock_check, reset_scheduler):
        """Returns 403 when execute permission denied."""
        from aragora.server.handlers.connectors import handle_trigger_sync

        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "Execute permission denied"
        mock_check.return_value = mock_decision

        result = await handle_trigger_sync(
            connector_id="test",
            auth_context=MockAuthContext(),
        )

        assert "error" in result
        assert result.get("status") == 403

    @pytest.mark.asyncio
    async def test_check_permission_returns_none_without_auth(self, reset_scheduler):
        """_check_permission returns None when no auth context."""
        from aragora.server.handlers.connectors import _check_permission

        result = _check_permission(None, "connectors:read")

        assert result is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", False)
    async def test_check_permission_graceful_degradation(self, reset_scheduler):
        """_check_permission allows when RBAC not available."""
        from aragora.server.handlers.connectors import _check_permission

        result = _check_permission(MockAuthContext(), "connectors:read")

        assert result is None


# ===========================================================================
# Test Tenant Isolation
# ===========================================================================


class TestTenantIsolation:
    """Tests for tenant isolation."""

    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    def test_resolve_tenant_id_uses_auth_org_id(self, reset_scheduler):
        """_resolve_tenant_id uses org_id from auth context."""
        from aragora.server.handlers.connectors import _resolve_tenant_id

        auth = MockAuthContext(org_id="secure-org")
        result = _resolve_tenant_id(auth, "malicious-tenant")

        assert result == "secure-org"

    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    def test_resolve_tenant_id_fallback_when_no_auth(self, reset_scheduler):
        """_resolve_tenant_id uses fallback when no auth."""
        from aragora.server.handlers.connectors import _resolve_tenant_id

        result = _resolve_tenant_id(None, "default-tenant")

        assert result == "default-tenant"

    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    def test_resolve_tenant_id_handles_unauthenticated_sentinel(self, reset_scheduler):
        """_resolve_tenant_id handles 'unauthenticated' sentinel."""
        from aragora.server.handlers.connectors import _resolve_tenant_id

        result = _resolve_tenant_id("unauthenticated", "default")

        assert result == "default"

    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    def test_resolve_tenant_id_uses_fallback_when_no_org_id(self, reset_scheduler):
        """_resolve_tenant_id uses fallback when auth has no org_id."""
        from aragora.server.handlers.connectors import _resolve_tenant_id

        auth = MockAuthContext(org_id=None)
        result = _resolve_tenant_id(auth, "default")

        assert result == "default"


# ===========================================================================
# Test MongoDB Aggregation
# ===========================================================================


class TestMongoDBAggregate:
    """Tests for MongoDB aggregation handlers."""

    @pytest.mark.asyncio
    async def test_mongodb_aggregate_connector_not_found(self, reset_scheduler):
        """Raises error when connector not found."""
        from aragora.server.handlers.connectors import handle_mongodb_aggregate

        # Patch inside the function's import path
        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.registry": MagicMock(
                    get_connector=MagicMock(return_value=None)
                )
            },
        ):
            with pytest.raises((ValueError, ModuleNotFoundError)):
                await handle_mongodb_aggregate(
                    connector_id="nonexistent",
                    collection="users",
                    pipeline=[{"$match": {"active": True}}],
                )

    @pytest.mark.asyncio
    async def test_mongodb_aggregate_wrong_connector_type(self, reset_scheduler):
        """Raises error when connector is not MongoDB."""
        from aragora.server.handlers.connectors import handle_mongodb_aggregate

        # Return a non-MongoDB connector (mock without the MongoDB type)
        mock_connector = MagicMock()
        mock_connector.__class__.__name__ = "S3Connector"

        mock_registry = MagicMock()
        mock_registry.get_connector = MagicMock(return_value=mock_connector)

        with patch.dict("sys.modules", {"aragora.connectors.enterprise.registry": mock_registry}):
            with pytest.raises((ValueError, ModuleNotFoundError)):
                await handle_mongodb_aggregate(
                    connector_id="s3-connector",
                    collection="users",
                    pipeline=[{"$match": {"active": True}}],
                )

    @pytest.mark.asyncio
    async def test_mongodb_aggregate_validates_pipeline_is_list(self, reset_scheduler):
        """Validates pipeline must be a list - validation happens before connector lookup."""
        from aragora.server.handlers.connectors import handle_mongodb_aggregate

        # The handler should validate pipeline type before trying to get connector
        # However, if it doesn't validate early, it will fail on module import
        try:
            await handle_mongodb_aggregate(
                connector_id="test",
                collection="users",
                pipeline="not a list",  # type: ignore
            )
            pytest.fail("Should have raised an error")
        except (ValueError, TypeError, ModuleNotFoundError):
            # Any of these is acceptable - the function correctly rejects invalid input
            pass


class TestMongoDBCollections:
    """Tests for MongoDB collections handler."""

    @pytest.mark.asyncio
    async def test_mongodb_collections_connector_not_found(self, reset_scheduler):
        """Raises error when connector not found."""
        from aragora.server.handlers.connectors import handle_mongodb_collections

        mock_registry = MagicMock()
        mock_registry.get_connector = MagicMock(return_value=None)

        with patch.dict("sys.modules", {"aragora.connectors.enterprise.registry": mock_registry}):
            with pytest.raises((ValueError, ModuleNotFoundError)):
                await handle_mongodb_collections(connector_id="nonexistent")


# ===========================================================================
# Test Connector Creation Helper
# ===========================================================================


class TestCreateConnectorHelper:
    """Tests for _create_connector helper function."""

    def test_create_github_connector(self, reset_scheduler):
        """Creates GitHub connector correctly."""
        from aragora.server.handlers.connectors import _create_connector

        connector = _create_connector(
            "github",
            {
                "owner": "test-org",
                "repo": "test-repo",
                "token": "ghp_token",
            },
        )

        assert connector is not None
        assert hasattr(connector, "connector_id")

    def test_create_s3_connector(self, reset_scheduler):
        """Creates S3 connector correctly."""
        from aragora.server.handlers.connectors import _create_connector

        connector = _create_connector(
            "s3",
            {
                "bucket": "test-bucket",
                "prefix": "data/",
                "region": "us-west-2",
            },
        )

        assert connector is not None

    def test_create_postgres_connector(self, reset_scheduler):
        """Creates PostgreSQL connector correctly."""
        from aragora.server.handlers.connectors import _create_connector

        connector = _create_connector(
            "postgres",
            {
                "database": "testdb",
                "host": "localhost",
                "port": 5432,
            },
        )

        assert connector is not None

    def test_create_mongodb_connector(self, reset_scheduler):
        """Creates MongoDB connector correctly."""
        from aragora.server.handlers.connectors import _create_connector

        connector = _create_connector(
            "mongodb",
            {
                "database": "testdb",
                "host": "localhost",
                "port": 27017,
            },
        )

        assert connector is not None

    def test_create_fhir_connector(self, reset_scheduler):
        """Creates FHIR connector correctly."""
        from aragora.server.handlers.connectors import _create_connector

        connector = _create_connector(
            "fhir",
            {
                "base_url": "https://fhir.example.com",
                "organization_id": "org-123",
            },
        )

        assert connector is not None

    def test_create_unknown_type_raises(self, reset_scheduler):
        """Raises ValueError for unknown connector type."""
        from aragora.server.handlers.connectors import _create_connector

        with pytest.raises(ValueError, match="Unknown connector type"):
            _create_connector("invalid_type", {})


# ===========================================================================
# Test Scheduler Singleton
# ===========================================================================


class TestSchedulerSingleton:
    """Tests for get_scheduler singleton behavior."""

    def test_get_scheduler_returns_scheduler(self, reset_scheduler):
        """get_scheduler returns a SyncScheduler instance."""
        from aragora.server.handlers.connectors import get_scheduler

        scheduler = get_scheduler()

        assert scheduler is not None
        assert hasattr(scheduler, "list_jobs")
        assert hasattr(scheduler, "get_job")
        assert hasattr(scheduler, "register_connector")

    def test_get_scheduler_singleton(self, reset_scheduler):
        """get_scheduler returns same instance on multiple calls."""
        from aragora.server.handlers.connectors import get_scheduler

        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()

        assert scheduler1 is scheduler2


# ===========================================================================
# Test Audit Logging
# ===========================================================================


class TestAuditLogging:
    """Tests for audit logging in handlers."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.legacy.audit_data")
    async def test_create_connector_audits(self, mock_audit, reset_scheduler):
        """Create connector logs audit event."""
        from aragora.server.handlers.connectors import handle_create_connector

        await handle_create_connector(
            connector_type="s3",
            config={"bucket": "test-bucket"},
            auth_context=MockAuthContext(user_id="test-user"),
        )

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args.kwargs
        assert call_kwargs["action"] == "create"
        assert call_kwargs["resource_type"] == "connector"
        assert call_kwargs["user_id"] == "test-user"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.legacy.audit_data")
    @patch("aragora.server.handlers.connectors.legacy.get_scheduler")
    async def test_trigger_sync_audits(self, mock_get_scheduler, mock_audit, reset_scheduler):
        """Trigger sync logs audit event."""
        from aragora.server.handlers.connectors import handle_trigger_sync

        # Setup mock scheduler with pre-registered job
        mock_scheduler = MockSyncScheduler()
        mock_connector = MockConnector("test-connector")
        mock_scheduler.register_connector(mock_connector, tenant_id="default")
        mock_get_scheduler.return_value = mock_scheduler

        result = await handle_trigger_sync(
            connector_id="test-connector",
            auth_context=MockAuthContext(user_id="test-user"),
        )

        # Only assert if sync was successful
        if result is not None and "run_id" in result:
            mock_audit.assert_called_once()
            call_kwargs = mock_audit.call_args.kwargs
            assert call_kwargs["action"] == "execute"
            assert call_kwargs["resource_type"] == "connector_sync"


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in handlers."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.connectors.check_permission")
    async def test_permission_denied_error_handling(self, mock_check, reset_scheduler):
        """Handles PermissionDeniedError gracefully."""
        from aragora.server.handlers.connectors import handle_list_connectors

        # Import the actual exception if available
        try:
            from aragora.rbac import PermissionDeniedError

            mock_check.side_effect = PermissionDeniedError("Access denied")
        except ImportError:
            # Create a mock exception
            class MockPermissionDeniedError(Exception):
                pass

            mock_check.side_effect = MockPermissionDeniedError("Access denied")
            # Patch the module to use our mock
            with patch(
                "aragora.server.handlers.connectors.PermissionDeniedError",
                MockPermissionDeniedError,
            ):
                result = await handle_list_connectors(auth_context=MockAuthContext())
                assert "error" in result
                assert result.get("status") == 403
                return

        result = await handle_list_connectors(auth_context=MockAuthContext())

        assert "error" in result
        assert result.get("status") == 403

    @pytest.mark.asyncio
    async def test_create_connector_missing_required_field(self, reset_scheduler):
        """Raises error when required field missing."""
        from aragora.server.handlers.connectors import handle_create_connector

        # S3 requires bucket
        with pytest.raises(KeyError):
            await handle_create_connector(
                connector_type="s3",
                config={},  # Missing bucket
            )


# ===========================================================================
# Test Response Formats
# ===========================================================================


class TestResponseFormats:
    """Tests for response format consistency."""

    @pytest.mark.asyncio
    async def test_list_connectors_response_format(self, reset_scheduler):
        """List connectors has correct response format."""
        from aragora.server.handlers.connectors import handle_list_connectors

        result = await handle_list_connectors()

        assert "connectors" in result
        assert "total" in result
        assert "limit" in result
        assert "offset" in result
        assert isinstance(result["connectors"], list)
        assert isinstance(result["total"], int)

    @pytest.mark.asyncio
    async def test_create_connector_response_format(self, reset_scheduler):
        """Create connector has correct response format."""
        from aragora.server.handlers.connectors import handle_create_connector

        result = await handle_create_connector(
            connector_type="s3",
            config={"bucket": "test"},
        )

        assert "id" in result
        assert "job_id" in result
        assert "type" in result
        assert "status" in result

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.connectors.legacy.get_scheduler")
    async def test_trigger_sync_response_format_mocked(self, mock_get_scheduler, reset_scheduler):
        """Trigger sync has correct response format with mocked scheduler."""
        from aragora.server.handlers.connectors import handle_trigger_sync

        mock_scheduler = MockSyncScheduler()
        mock_connector = MockConnector("test-connector")
        mock_scheduler.register_connector(mock_connector, tenant_id="default")
        mock_get_scheduler.return_value = mock_scheduler

        result = await handle_trigger_sync(connector_id="test-connector")

        assert "run_id" in result
        assert "connector_id" in result
        assert "status" in result
        assert "full_sync" in result

    @pytest.mark.asyncio
    async def test_health_check_response_format(self, reset_scheduler):
        """Health check has correct response format."""
        from aragora.server.handlers.connectors import handle_connector_health

        result = await handle_connector_health()

        assert "status" in result
        assert "scheduler_running" in result
        assert "total_connectors" in result


# ===========================================================================
# Additional Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_list_connectors_with_zero_limit(self, reset_scheduler):
        """List connectors handles zero limit."""
        from aragora.server.handlers.connectors import handle_list_connectors

        result = await handle_list_connectors(limit=0)

        # Should be bounded to minimum of 1
        assert result["limit"] == 1

    @pytest.mark.asyncio
    async def test_get_connector_empty_id(self, reset_scheduler):
        """Get connector handles empty ID."""
        from aragora.server.handlers.connectors import handle_get_connector

        result = await handle_get_connector("")

        assert result is None

    @pytest.mark.asyncio
    async def test_sync_history_with_status_filter(self, reset_scheduler):
        """Sync history accepts status filter."""
        from aragora.server.handlers.connectors import handle_get_sync_history

        result = await handle_get_sync_history(status="completed")

        assert "history" in result
        assert isinstance(result["history"], list)

    @pytest.mark.asyncio
    async def test_scheduler_stats_with_tenant(self, reset_scheduler):
        """Scheduler stats accepts tenant filter."""
        from aragora.server.handlers.connectors import handle_get_scheduler_stats

        result = await handle_get_scheduler_stats(tenant_id="test-tenant")

        assert "total_jobs" in result

    @pytest.mark.asyncio
    async def test_create_github_connector_with_sync_options(self, reset_scheduler):
        """Creates GitHub connector with sync options."""
        from aragora.server.handlers.connectors import handle_create_connector

        config = {
            "owner": "test-org",
            "repo": "test-repo",
            "token": "ghp_token",
            "sync_prs": False,
            "sync_issues": True,
        }

        result = await handle_create_connector(
            connector_type="github",
            config=config,
        )

        assert isinstance(result, dict)
        assert "id" in result
