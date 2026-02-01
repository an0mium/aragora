"""Tests for Policies namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Policy CRUD Operations
# =========================================================================


class TestPoliciesCRUD:
    """Tests for policy CRUD operations."""

    def test_list_policies_default(self) -> None:
        """List policies with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "policies": [{"id": "pol_1", "name": "Data Retention"}],
                "total": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.list()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/policies", params={"limit": 100, "offset": 0}
            )
            assert result["total"] == 1
            client.close()

    def test_list_policies_with_filters(self) -> None:
        """List policies with all filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"policies": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.policies.list(
                workspace_id="ws_123",
                vertical_id="finance",
                framework_id="soc2",
                enabled_only=True,
                limit=50,
                offset=10,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["workspace_id"] == "ws_123"
            assert params["vertical_id"] == "finance"
            assert params["framework_id"] == "soc2"
            assert params["enabled_only"] == "true"
            assert params["limit"] == 50
            assert params["offset"] == 10
            client.close()

    def test_get_policy(self) -> None:
        """Get a specific policy."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "pol_123",
                "name": "Data Retention",
                "enabled": True,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.get("pol_123")

            mock_request.assert_called_once_with("GET", "/api/v1/policies/pol_123")
            assert result["id"] == "pol_123"
            client.close()

    def test_create_policy_minimal(self) -> None:
        """Create a policy with minimal required fields."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "pol_new", "name": "New Policy"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.create(
                name="New Policy",
                framework_id="soc2",
                vertical_id="finance",
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/policies",
                json={
                    "name": "New Policy",
                    "framework_id": "soc2",
                    "vertical_id": "finance",
                    "workspace_id": "default",
                    "level": "recommended",
                    "enabled": True,
                    "rules": [],
                    "metadata": {},
                },
            )
            assert result["name"] == "New Policy"
            client.close()

    def test_create_policy_full(self) -> None:
        """Create a policy with all fields."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "pol_new"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            rules = [{"type": "retention", "days": 90}]
            metadata = {"owner": "compliance-team"}
            client.policies.create(
                name="Data Retention Policy",
                framework_id="gdpr",
                vertical_id="healthcare",
                description="Enforce 90-day retention",
                workspace_id="ws_456",
                level="required",
                enabled=False,
                rules=rules,
                metadata=metadata,
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["description"] == "Enforce 90-day retention"
            assert json_data["workspace_id"] == "ws_456"
            assert json_data["level"] == "required"
            assert json_data["enabled"] is False
            assert json_data["rules"] == rules
            assert json_data["metadata"] == metadata
            client.close()

    def test_update_policy(self) -> None:
        """Update a policy."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "pol_123", "name": "Updated Policy"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.update(
                "pol_123",
                name="Updated Policy",
                description="Updated description",
                level="optional",
                enabled=False,
            )

            mock_request.assert_called_once_with(
                "PATCH",
                "/api/v1/policies/pol_123",
                json={
                    "name": "Updated Policy",
                    "description": "Updated description",
                    "level": "optional",
                    "enabled": False,
                },
            )
            assert result["name"] == "Updated Policy"
            client.close()

    def test_update_policy_rules(self) -> None:
        """Update policy rules."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "pol_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            new_rules = [{"type": "encryption", "algorithm": "AES-256"}]
            client.policies.update("pol_123", rules=new_rules)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["rules"] == new_rules
            client.close()

    def test_delete_policy(self) -> None:
        """Delete a policy."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.delete("pol_123")

            mock_request.assert_called_once_with("DELETE", "/api/v1/policies/pol_123")
            assert result["deleted"] is True
            client.close()

    def test_toggle_policy(self) -> None:
        """Toggle a policy's enabled status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"enabled": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.toggle("pol_123", enabled=True)

            mock_request.assert_called_once_with(
                "POST", "/api/v1/policies/pol_123/toggle", json={"enabled": True}
            )
            assert result["enabled"] is True
            client.close()

    def test_toggle_policy_no_value(self) -> None:
        """Toggle a policy without specifying value (server toggles)."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"enabled": False}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.policies.toggle("pol_123")

            mock_request.assert_called_once_with("POST", "/api/v1/policies/pol_123/toggle", json={})
            client.close()


# =========================================================================
# Policy Violations
# =========================================================================


class TestPolicyViolations:
    """Tests for policy-specific violation operations."""

    def test_get_policy_violations_default(self) -> None:
        """Get violations for a policy with defaults."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"violations": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.get_policy_violations("pol_123")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/policies/pol_123/violations",
                params={"limit": 100, "offset": 0},
            )
            assert "violations" in result
            client.close()

    def test_get_policy_violations_with_filters(self) -> None:
        """Get violations for a policy with filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"violations": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.policies.get_policy_violations(
                "pol_123", status="open", severity="critical", limit=25, offset=5
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["status"] == "open"
            assert params["severity"] == "critical"
            assert params["limit"] == 25
            assert params["offset"] == 5
            client.close()


# =========================================================================
# Violations (Global)
# =========================================================================


class TestViolations:
    """Tests for global violation operations."""

    def test_list_violations_default(self) -> None:
        """List all violations with defaults."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"violations": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.list_violations()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/compliance/violations", params={"limit": 100, "offset": 0}
            )
            assert "violations" in result
            client.close()

    def test_list_violations_with_filters(self) -> None:
        """List violations with all filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"violations": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.policies.list_violations(
                workspace_id="ws_123",
                vertical_id="finance",
                framework_id="soc2",
                status="investigating",
                severity="high",
                limit=50,
                offset=10,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["workspace_id"] == "ws_123"
            assert params["vertical_id"] == "finance"
            assert params["framework_id"] == "soc2"
            assert params["status"] == "investigating"
            assert params["severity"] == "high"
            client.close()

    def test_get_violation(self) -> None:
        """Get a specific violation."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "vio_123",
                "status": "open",
                "severity": "high",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.get_violation("vio_123")

            mock_request.assert_called_once_with("GET", "/api/v1/compliance/violations/vio_123")
            assert result["id"] == "vio_123"
            client.close()

    def test_update_violation_status(self) -> None:
        """Update a violation's status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "vio_123", "status": "resolved"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.update_violation("vio_123", status="resolved")

            mock_request.assert_called_once_with(
                "PATCH",
                "/api/v1/compliance/violations/vio_123",
                json={"status": "resolved"},
            )
            assert result["status"] == "resolved"
            client.close()

    def test_update_violation_with_notes(self) -> None:
        """Update a violation with resolution notes."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "vio_123", "status": "resolved"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.policies.update_violation(
                "vio_123",
                status="resolved",
                resolution_notes="Fixed by updating encryption config",
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["resolution_notes"] == "Fixed by updating encryption config"
            client.close()


# =========================================================================
# Compliance Check
# =========================================================================


class TestComplianceCheck:
    """Tests for compliance check operations."""

    def test_check_compliance_minimal(self) -> None:
        """Run compliance check with minimal parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "compliant": True,
                "score": 95,
                "issues": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.check_compliance("Some content to check")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/compliance/check",
                json={
                    "content": "Some content to check",
                    "min_severity": "low",
                    "store_violations": False,
                    "workspace_id": "default",
                    "source": "manual_check",
                },
            )
            assert result["compliant"] is True
            assert result["score"] == 95
            client.close()

    def test_check_compliance_full(self) -> None:
        """Run compliance check with all parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"compliant": False, "issues": [{}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.policies.check_compliance(
                content="Sensitive data to check",
                frameworks=["gdpr", "hipaa"],
                min_severity="high",
                store_violations=True,
                workspace_id="ws_123",
                source="api_integration",
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["content"] == "Sensitive data to check"
            assert json_data["frameworks"] == ["gdpr", "hipaa"]
            assert json_data["min_severity"] == "high"
            assert json_data["store_violations"] is True
            assert json_data["workspace_id"] == "ws_123"
            assert json_data["source"] == "api_integration"
            client.close()


# =========================================================================
# Statistics
# =========================================================================


class TestComplianceStats:
    """Tests for compliance statistics operations."""

    def test_get_stats_default(self) -> None:
        """Get compliance statistics with no filter."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_policies": 10,
                "enabled_policies": 8,
                "total_violations": 5,
                "open_violations": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.policies.get_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/compliance/stats", params={})
            assert result["total_policies"] == 10
            client.close()

    def test_get_stats_with_workspace(self) -> None:
        """Get compliance statistics for a workspace."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_policies": 5}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.policies.get_stats(workspace_id="ws_123")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["workspace_id"] == "ws_123"
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncPolicies:
    """Tests for async Policies API."""

    @pytest.mark.asyncio
    async def test_async_list_policies(self) -> None:
        """List policies asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"policies": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.list()

                mock_request.assert_called_once_with(
                    "GET", "/api/v1/policies", params={"limit": 100, "offset": 0}
                )
                assert "policies" in result

    @pytest.mark.asyncio
    async def test_async_get_policy(self) -> None:
        """Get policy asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "pol_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.get("pol_123")

                mock_request.assert_called_once_with("GET", "/api/v1/policies/pol_123")
                assert result["id"] == "pol_123"

    @pytest.mark.asyncio
    async def test_async_create_policy(self) -> None:
        """Create policy asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "pol_new"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.create(
                    name="New Policy", framework_id="soc2", vertical_id="tech"
                )

                call_args = mock_request.call_args
                assert call_args[1]["json"]["name"] == "New Policy"
                assert result["id"] == "pol_new"

    @pytest.mark.asyncio
    async def test_async_update_policy(self) -> None:
        """Update policy asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "pol_123", "enabled": False}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.update("pol_123", enabled=False)

                mock_request.assert_called_once_with(
                    "PATCH", "/api/v1/policies/pol_123", json={"enabled": False}
                )
                assert result["enabled"] is False

    @pytest.mark.asyncio
    async def test_async_delete_policy(self) -> None:
        """Delete policy asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.delete("pol_123")

                mock_request.assert_called_once_with("DELETE", "/api/v1/policies/pol_123")
                assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_async_list_violations(self) -> None:
        """List violations asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"violations": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.list_violations()

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/compliance/violations",
                    params={"limit": 100, "offset": 0},
                )
                assert "violations" in result

    @pytest.mark.asyncio
    async def test_async_check_compliance(self) -> None:
        """Check compliance asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"compliant": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.check_compliance("Test content")

                call_args = mock_request.call_args
                assert call_args[1]["json"]["content"] == "Test content"
                assert result["compliant"] is True

    @pytest.mark.asyncio
    async def test_async_get_stats(self) -> None:
        """Get stats asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_policies": 10}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.get_stats()

                mock_request.assert_called_once_with("GET", "/api/v1/compliance/stats", params={})
                assert result["total_policies"] == 10

    @pytest.mark.asyncio
    async def test_async_toggle_policy(self) -> None:
        """Toggle policy asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"enabled": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.policies.toggle("pol_123", enabled=True)

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/policies/pol_123/toggle", json={"enabled": True}
                )
                assert result["enabled"] is True
