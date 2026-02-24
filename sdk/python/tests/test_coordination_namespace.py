"""Tests for Coordination namespace API.

Covers all 13 endpoints:
- register_workspace, list_workspaces, unregister_workspace
- create_federation_policy, list_federation_policies
- execute_cross_workspace, list_executions
- grant_consent, revoke_consent, list_consents
- approve_request
- get_stats, get_health
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient
from aragora_sdk.namespaces.coordination import AsyncCoordinationAPI, CoordinationAPI


class TestCoordinationWiring:
    """Verify the namespace is wired into both sync and async clients."""

    def test_sync_client_has_coordination(self) -> None:
        client = AragoraClient(base_url="http://localhost")
        assert isinstance(client.coordination, CoordinationAPI)
        client.close()

    @pytest.mark.asyncio
    async def test_async_client_has_coordination(self) -> None:
        async with AragoraAsyncClient(base_url="http://localhost") as client:
            assert isinstance(client.coordination, AsyncCoordinationAPI)


class TestWorkspaceEndpoints:
    """Tests for workspace registration, listing, and unregistration."""

    def test_register_workspace(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "ws-1",
                "name": "Primary",
                "org_id": "org-1",
                "federation_mode": "readonly",
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.register_workspace(
                id="ws-1", name="Primary", org_id="org-1"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/coordination/workspaces",
                json={
                    "id": "ws-1",
                    "name": "Primary",
                    "org_id": "org-1",
                    "federation_mode": "readonly",
                    "supports_agent_execution": True,
                    "supports_workflow_execution": True,
                    "supports_knowledge_query": True,
                },
            )
            assert result["id"] == "ws-1"
            client.close()

    def test_list_workspaces(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "workspaces": [{"id": "ws-1"}, {"id": "ws-2"}],
                "total": 2,
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.list_workspaces()

            mock_request.assert_called_once_with("GET", "/api/v1/coordination/workspaces")
            assert result["total"] == 2
            assert len(result["workspaces"]) == 2
            client.close()

    def test_unregister_workspace(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"unregistered": True}
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.unregister_workspace("ws-1")

            mock_request.assert_called_once_with("DELETE", "/api/v1/coordination/workspaces/ws-1")
            assert result["unregistered"] is True
            client.close()


class TestFederationPolicyEndpoints:
    """Tests for federation policy creation and listing."""

    def test_create_federation_policy(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "name": "default-policy",
                "mode": "readonly",
                "sharing_scope": "metadata",
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.create_federation_policy(
                name="default-policy",
                mode="readonly",
                sharing_scope="metadata",
                allowed_operations=["knowledge_query"],
            )

            call_args = mock_request.call_args
            assert call_args[0] == ("POST", "/api/v1/coordination/federation")
            body = call_args[1]["json"]
            assert body["name"] == "default-policy"
            assert body["mode"] == "readonly"
            assert body["sharing_scope"] == "metadata"
            assert body["allowed_operations"] == ["knowledge_query"]
            assert result["name"] == "default-policy"
            client.close()

    def test_list_federation_policies(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "policies": [{"name": "default", "scope": "default"}],
                "total": 1,
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.list_federation_policies()

            mock_request.assert_called_once_with("GET", "/api/v1/coordination/federation")
            assert result["total"] == 1
            client.close()

    def test_backward_compatible_aliases(self) -> None:
        """Verify create_policy and list_policies still work."""
        assert CoordinationAPI.create_policy is CoordinationAPI.create_federation_policy
        assert CoordinationAPI.list_policies is CoordinationAPI.list_federation_policies


class TestExecutionEndpoints:
    """Tests for cross-workspace execution and listing."""

    def test_execute_cross_workspace(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "request_id": "req-1",
                "success": True,
                "data": {"answer": 42},
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.execute_cross_workspace(
                operation="knowledge_query",
                source_workspace_id="ws-1",
                target_workspace_id="ws-2",
                payload={"query": "test"},
            )

            call_args = mock_request.call_args
            assert call_args[0] == ("POST", "/api/v1/coordination/execute")
            body = call_args[1]["json"]
            assert body["operation"] == "knowledge_query"
            assert body["source_workspace_id"] == "ws-1"
            assert body["target_workspace_id"] == "ws-2"
            assert body["payload"] == {"query": "test"}
            assert result["success"] is True
            client.close()

    def test_execute_alias(self) -> None:
        """Verify execute is an alias for execute_cross_workspace."""
        assert CoordinationAPI.execute is CoordinationAPI.execute_cross_workspace

    def test_list_executions(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "executions": [{"request_id": "req-1"}],
                "total": 1,
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.list_executions(workspace_id="ws-1")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/coordination/executions",
                params={"workspace_id": "ws-1"},
            )
            assert result["total"] == 1
            client.close()

    def test_list_executions_no_filter(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"executions": [], "total": 0}
            client = AragoraClient(base_url="http://localhost")
            client.coordination.list_executions()

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/coordination/executions",
                params={},
            )
            client.close()


class TestConsentEndpoints:
    """Tests for consent grant, revoke, and listing."""

    def test_grant_consent(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "consent-1",
                "source_workspace_id": "ws-1",
                "target_workspace_id": "ws-2",
                "scope": "metadata",
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.grant_consent(
                source_workspace_id="ws-1",
                target_workspace_id="ws-2",
                scope="metadata",
                data_types=["debates", "receipts"],
                expires_in_days=30,
            )

            call_args = mock_request.call_args
            assert call_args[0] == ("POST", "/api/v1/coordination/consent")
            body = call_args[1]["json"]
            assert body["source_workspace_id"] == "ws-1"
            assert body["target_workspace_id"] == "ws-2"
            assert body["data_types"] == ["debates", "receipts"]
            assert body["expires_in_days"] == 30
            assert result["id"] == "consent-1"
            client.close()

    def test_revoke_consent(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"revoked": True}
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.revoke_consent("consent-1")

            mock_request.assert_called_once_with("DELETE", "/api/v1/coordination/consent/consent-1")
            assert result["revoked"] is True
            client.close()

    def test_list_consents(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "consents": [{"id": "consent-1"}],
                "total": 1,
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.list_consents(workspace_id="ws-1")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/coordination/consent",
                params={"workspace_id": "ws-1"},
            )
            assert result["total"] == 1
            client.close()


class TestApprovalEndpoint:
    """Test for the approve_request endpoint."""

    def test_approve_request(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"approved": True}
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.approve_request("req-1", approved_by="admin-user")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/coordination/approve/req-1",
                json={"approved_by": "admin-user"},
            )
            assert result["approved"] is True
            client.close()


class TestStatsAndHealthEndpoints:
    """Tests for stats and health endpoints."""

    def test_get_stats(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_workspaces": 5,
                "pending_requests": 2,
                "valid_consents": 3,
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.get_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/coordination/stats")
            assert result["total_workspaces"] == 5
            client.close()

    def test_get_health(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "healthy",
                "total_workspaces": 5,
                "pending_requests": 0,
            }
            client = AragoraClient(base_url="http://localhost")
            result = client.coordination.get_health()

            mock_request.assert_called_once_with("GET", "/api/v1/coordination/health")
            assert result["status"] == "healthy"
            client.close()


class TestAsyncCoordination:
    """Tests for async coordination namespace methods."""

    @pytest.mark.asyncio
    async def test_async_register_workspace(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "ws-1", "name": "Primary"}
            async with AragoraAsyncClient(base_url="http://localhost") as client:
                result = await client.coordination.register_workspace(id="ws-1", name="Primary")
                assert result["id"] == "ws-1"

    @pytest.mark.asyncio
    async def test_async_create_federation_policy(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"name": "policy-1"}
            async with AragoraAsyncClient(base_url="http://localhost") as client:
                result = await client.coordination.create_federation_policy(name="policy-1")
                assert result["name"] == "policy-1"

    @pytest.mark.asyncio
    async def test_async_execute_cross_workspace(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "data": {}}
            async with AragoraAsyncClient(base_url="http://localhost") as client:
                result = await client.coordination.execute_cross_workspace(
                    operation="knowledge_query",
                    source_workspace_id="ws-1",
                    target_workspace_id="ws-2",
                )
                assert result["success"] is True

    def test_async_backward_compatible_aliases(self) -> None:
        """Verify async aliases at class level."""
        assert AsyncCoordinationAPI.create_policy is AsyncCoordinationAPI.create_federation_policy
        assert AsyncCoordinationAPI.list_policies is AsyncCoordinationAPI.list_federation_policies
        assert AsyncCoordinationAPI.execute is AsyncCoordinationAPI.execute_cross_workspace
