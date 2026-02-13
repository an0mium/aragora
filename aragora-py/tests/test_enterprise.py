"""Tests for enterprise API modules (Admin, Analytics, Connectors)."""

from unittest.mock import AsyncMock

import pytest

from aragora_client import AdminAPI, AnalyticsAPI, AragoraClient, ConnectorsAPI


class TestAdminAPI:
    """Tests for AdminAPI."""

    @pytest.fixture
    def client(self):
        """Create a mock client."""
        client = AsyncMock(spec=AragoraClient)
        client._get = AsyncMock()
        client._post = AsyncMock()
        client._patch = AsyncMock()
        client._request = AsyncMock()
        return client

    @pytest.fixture
    def admin_api(self, client):
        """Create AdminAPI with mock client."""
        return AdminAPI(client)

    @pytest.mark.asyncio
    async def test_list_organizations(self, admin_api, client):
        """Test listing organizations."""
        client._get.return_value = {"organizations": [], "total": 0}
        result = await admin_api.list_organizations(limit=10, tier="enterprise")
        client._get.assert_called_once_with(
            "/api/v1/admin/organizations",
            params={"limit": 10, "offset": 0, "tier": "enterprise"},
        )
        assert result == {"organizations": [], "total": 0}

    @pytest.mark.asyncio
    async def test_list_all_users(self, admin_api, client):
        """Test listing all users."""
        client._get.return_value = {"users": [], "total": 0}
        await admin_api.list_all_users(org_id="org-123", active_only=True)
        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[0][0] == "/api/v1/admin/users"
        assert call_args[1]["params"]["org_id"] == "org-123"
        assert call_args[1]["params"]["active_only"] == "true"

    @pytest.mark.asyncio
    async def test_get_stats(self, admin_api, client):
        """Test getting admin stats."""
        client._get.return_value = {"users": 100, "orgs": 10}
        await admin_api.get_stats()
        client._get.assert_called_once_with("/api/v1/admin/stats")

    @pytest.mark.asyncio
    async def test_impersonate_user(self, admin_api, client):
        """Test user impersonation."""
        client._post.return_value = {"token": "temp-token"}
        await admin_api.impersonate_user("user-123")
        client._post.assert_called_once_with(
            "/api/v1/admin/users/user-123/impersonate", {}
        )

    @pytest.mark.asyncio
    async def test_deactivate_user(self, admin_api, client):
        """Test user deactivation."""
        client._post.return_value = {"status": "deactivated"}
        await admin_api.deactivate_user("user-123")
        client._post.assert_called_once_with(
            "/api/v1/admin/users/user-123/deactivate", {}
        )

    @pytest.mark.asyncio
    async def test_get_nomic_status(self, admin_api, client):
        """Test getting Nomic status."""
        client._get.return_value = {"phase": 1, "active": True}
        await admin_api.get_nomic_status()
        client._get.assert_called_once_with("/api/v1/admin/nomic/status")

    @pytest.mark.asyncio
    async def test_reset_nomic(self, admin_api, client):
        """Test resetting Nomic."""
        client._post.return_value = {"status": "reset"}
        await admin_api.reset_nomic(target_phase=0, clear_errors=True, reason="Testing")
        client._post.assert_called_once_with(
            "/api/v1/admin/nomic/reset",
            {"target_phase": 0, "clear_errors": True, "reason": "Testing"},
        )

    @pytest.mark.asyncio
    async def test_healthz(self, admin_api, client):
        """Test liveness probe."""
        client._get.return_value = {"status": "ok"}
        await admin_api.healthz()
        client._get.assert_called_once_with("/healthz")

    @pytest.mark.asyncio
    async def test_get_detailed_health(self, admin_api, client):
        """Test detailed health check."""
        client._get.return_value = {"subsystems": {}}
        await admin_api.get_detailed_health()
        client._get.assert_called_once_with("/api/v1/health/detailed")

    @pytest.mark.asyncio
    async def test_rotate_encryption_key(self, admin_api, client):
        """Test encryption key rotation."""
        client._post.return_value = {"rotated": 100}
        await admin_api.rotate_encryption_key(
            dry_run=True, stores=["memory"], force=True
        )
        client._post.assert_called_once_with(
            "/api/v1/admin/security/rotate-key",
            {"dry_run": True, "force": True, "stores": ["memory"]},
        )

    @pytest.mark.asyncio
    async def test_issue_credits(self, admin_api, client):
        """Test issuing credits."""
        client._post.return_value = {"transaction_id": "tx-123"}
        await admin_api.issue_credits(
            "org-123",
            10000,
            credit_type="promotional",
            description="Welcome bonus",
            expires_days=30,
        )
        client._post.assert_called_once()
        call_args = client._post.call_args
        assert call_args[0][0] == "/api/v1/admin/credits/issue"
        assert call_args[0][1]["org_id"] == "org-123"
        assert call_args[0][1]["amount_cents"] == 10000

    @pytest.mark.asyncio
    async def test_get_debate_dashboard(self, admin_api, client):
        """Test getting debate dashboard."""
        client._get.return_value = {"active": 5, "completed": 10}
        await admin_api.get_debate_dashboard(domain="tech", hours=12)
        client._get.assert_called_once_with(
            "/api/v1/admin/dashboard/debates",
            params={"limit": 10, "hours": 12, "domain": "tech"},
        )


class TestAnalyticsAPI:
    """Tests for AnalyticsAPI."""

    @pytest.fixture
    def client(self):
        """Create a mock client."""
        client = AsyncMock(spec=AragoraClient)
        client._get = AsyncMock()
        client._post = AsyncMock()
        return client

    @pytest.fixture
    def analytics_api(self, client):
        """Create AnalyticsAPI with mock client."""
        return AnalyticsAPI(client)

    @pytest.mark.asyncio
    async def test_get_disagreement_stats(self, analytics_api, client):
        """Test getting disagreement stats."""
        client._get.return_value = {"rate": 0.15}
        await analytics_api.get_disagreement_stats(time_range="7d")
        client._get.assert_called_once_with(
            "/api/v1/analytics/disagreements", params={"time_range": "7d"}
        )

    @pytest.mark.asyncio
    async def test_get_consensus_quality_stats(self, analytics_api, client):
        """Test getting consensus quality stats."""
        client._get.return_value = {"strength": 0.85}
        await analytics_api.get_consensus_quality_stats()
        client._get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_learning_efficiency(self, analytics_api, client):
        """Test getting learning efficiency."""
        client._get.return_value = {"efficiency": 0.7}
        await analytics_api.get_learning_efficiency(agent="claude", domain="tech")
        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[1]["params"]["agent"] == "claude"
        assert call_args[1]["params"]["domain"] == "tech"

    @pytest.mark.asyncio
    async def test_get_debates_overview(self, analytics_api, client):
        """Test getting debates overview."""
        client._get.return_value = {"total": 100, "completed": 90}
        await analytics_api.get_debates_overview(org_id="org-123")
        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[1]["params"]["org_id"] == "org-123"

    @pytest.mark.asyncio
    async def test_get_debates_trends(self, analytics_api, client):
        """Test getting debate trends."""
        client._get.return_value = {"data": []}
        await analytics_api.get_debates_trends(granularity="weekly")
        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[1]["params"]["granularity"] == "weekly"

    @pytest.mark.asyncio
    async def test_get_agents_leaderboard(self, analytics_api, client):
        """Test getting agent leaderboard."""
        client._get.return_value = {"agents": []}
        await analytics_api.get_agents_leaderboard(limit=10, domain="tech")
        client._get.assert_called_once()

    @pytest.mark.asyncio
    async def test_compare_agents(self, analytics_api, client):
        """Test comparing agents."""
        client._post.return_value = {"comparison": {}}
        await analytics_api.compare_agents(["claude", "gpt4"])
        client._post.assert_called_once_with(
            "/api/v1/analytics/agents/compare", {"agents": ["claude", "gpt4"]}
        )

    @pytest.mark.asyncio
    async def test_get_token_usage(self, analytics_api, client):
        """Test getting token usage."""
        client._get.return_value = {"tokens": 1000000}
        await analytics_api.get_token_usage("org-123", granularity="hourly")
        client._get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_flip_summary(self, analytics_api, client):
        """Test getting flip summary."""
        client._get.return_value = {"rate": 0.1}
        await analytics_api.get_flip_summary()
        client._get.assert_called_once_with("/api/v1/analytics/flips/summary")

    @pytest.mark.asyncio
    async def test_get_deliberation_analytics(self, analytics_api, client):
        """Test getting deliberation analytics."""
        client._get.return_value = {"patterns": []}
        await analytics_api.get_deliberation_analytics("org-123", days=7)
        client._get.assert_called_once_with(
            "/api/v1/analytics/deliberations/org-123", params={"days": 7}
        )


class TestConnectorsAPI:
    """Tests for ConnectorsAPI."""

    @pytest.fixture
    def client(self):
        """Create a mock client."""
        client = AsyncMock(spec=AragoraClient)
        client._get = AsyncMock()
        client._post = AsyncMock()
        client._patch = AsyncMock()
        client._request = AsyncMock()
        return client

    @pytest.fixture
    def connectors_api(self, client):
        """Create ConnectorsAPI with mock client."""
        return ConnectorsAPI(client)

    @pytest.mark.asyncio
    async def test_list(self, connectors_api, client):
        """Test listing connectors."""
        client._get.return_value = {"connectors": [], "total": 0}
        await connectors_api.list(status="active", connector_type="slack")
        client._get.assert_called_once()
        call_args = client._get.call_args
        assert call_args[1]["params"]["status"] == "active"
        assert call_args[1]["params"]["type"] == "slack"

    @pytest.mark.asyncio
    async def test_create(self, connectors_api, client):
        """Test creating a connector."""
        client._post.return_value = {"id": "conn-123"}
        await connectors_api.create(
            "slack", name="My Slack", config={"webhook_url": "https://..."}
        )
        client._post.assert_called_once_with(
            "/api/v1/connectors",
            {
                "type": "slack",
                "name": "My Slack",
                "config": {"webhook_url": "https://..."},
            },
        )

    @pytest.mark.asyncio
    async def test_get(self, connectors_api, client):
        """Test getting a connector."""
        client._get.return_value = {"id": "conn-123", "type": "slack"}
        await connectors_api.get("conn-123")
        client._get.assert_called_once_with("/api/v1/connectors/conn-123")

    @pytest.mark.asyncio
    async def test_update(self, connectors_api, client):
        """Test updating a connector."""
        client._patch.return_value = {"id": "conn-123"}
        await connectors_api.update("conn-123", name="Updated Name")
        client._patch.assert_called_once_with(
            "/api/v1/connectors/conn-123", {"name": "Updated Name"}
        )

    @pytest.mark.asyncio
    async def test_delete(self, connectors_api, client):
        """Test deleting a connector."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"deleted": True}
        client._request.return_value = mock_response
        await connectors_api.delete("conn-123")
        client._request.assert_called_once_with("DELETE", "/api/v1/connectors/conn-123")

    @pytest.mark.asyncio
    async def test_sync(self, connectors_api, client):
        """Test triggering a sync."""
        client._post.return_value = {"sync_id": "sync-123"}
        await connectors_api.sync("conn-123")
        client._post.assert_called_once_with("/api/v1/connectors/conn-123/sync", {})

    @pytest.mark.asyncio
    async def test_test_existing(self, connectors_api, client):
        """Test testing an existing connector."""
        client._post.return_value = {"connected": True}
        await connectors_api.test(connector_id="conn-123")
        client._post.assert_called_once_with("/api/v1/connectors/conn-123/test", {})

    @pytest.mark.asyncio
    async def test_test_new(self, connectors_api, client):
        """Test testing a new connector config."""
        client._post.return_value = {"connected": True}
        await connectors_api.test(config={"url": "https://..."})
        client._post.assert_called_once_with(
            "/api/v1/connectors/test", {"config": {"url": "https://..."}}
        )

    @pytest.mark.asyncio
    async def test_get_sync_history(self, connectors_api, client):
        """Test getting sync history."""
        client._get.return_value = {"syncs": []}
        await connectors_api.get_sync_history(connector_id="conn-123")
        client._get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, connectors_api, client):
        """Test getting connector stats."""
        client._get.return_value = {"total": 10, "active": 8}
        await connectors_api.get_stats()
        client._get.assert_called_once_with("/api/v1/connectors/stats")

    @pytest.mark.asyncio
    async def test_list_types(self, connectors_api, client):
        """Test listing connector types."""
        client._get.return_value = {"types": ["slack", "github", "jira"]}
        await connectors_api.list_types()
        client._get.assert_called_once_with("/api/v1/connectors/types")
