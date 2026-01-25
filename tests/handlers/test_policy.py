"""
Tests for PolicyHandler - policy and compliance endpoints.

Tests all 12 endpoints:
- Policy CRUD: list, get, create, update, delete, toggle
- Policy violations: get violations for a policy
- Compliance violations: list, get, update status
- Compliance check: run compliance validation
- Compliance stats: get statistics
"""

import io
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.server.handlers.policy import PolicyHandler


@pytest.fixture
def handler():
    """Create handler instance with empty server context."""
    return PolicyHandler({})


class TestPolicyHandlerRouting:
    """Test can_handle routing logic."""

    def test_can_handle_policies_list(self, handler):
        """Test routing for policy list endpoint."""
        assert handler.can_handle("/api/v1/policies", "GET") is True
        assert handler.can_handle("/api/v1/policies", "POST") is True

    def test_can_handle_policies_with_id(self, handler):
        """Test routing for policy ID endpoints."""
        assert handler.can_handle("/api/v1/policies/pol_123abc", "GET") is True
        assert handler.can_handle("/api/v1/policies/pol_123abc", "PATCH") is True
        assert handler.can_handle("/api/v1/policies/pol_123abc", "DELETE") is True

    def test_can_handle_policy_toggle(self, handler):
        """Test routing for policy toggle endpoint."""
        assert handler.can_handle("/api/v1/policies/pol_123abc/toggle", "POST") is True

    def test_can_handle_policy_violations(self, handler):
        """Test routing for policy violations endpoint."""
        assert handler.can_handle("/api/v1/policies/pol_123abc/violations", "GET") is True

    def test_can_handle_compliance_endpoints(self, handler):
        """Test routing for compliance endpoints."""
        assert handler.can_handle("/api/v1/compliance/violations", "GET") is True
        assert handler.can_handle("/api/v1/compliance/violations/viol_123", "GET") is True
        assert handler.can_handle("/api/v1/compliance/violations/viol_123", "PATCH") is True
        assert handler.can_handle("/api/v1/compliance/check", "POST") is True
        assert handler.can_handle("/api/v1/compliance/stats", "GET") is True

    def test_cannot_handle_unrelated_paths(self, handler):
        """Test that unrelated paths are rejected."""
        assert handler.can_handle("/api/v1/debates", "GET") is False
        assert handler.can_handle("/api/v1/users", "GET") is False
        assert handler.can_handle("/health", "GET") is False


def _create_mock_policy(policy_id: str, name: str, enabled: bool = True):
    """Create a mock policy object."""
    mock = MagicMock()
    mock.id = policy_id
    mock.name = name
    mock.enabled = enabled
    mock.to_dict.return_value = {
        "id": policy_id,
        "name": name,
        "enabled": enabled,
        "framework_id": "soc2",
        "vertical_id": "healthcare",
    }
    return mock


def _create_mock_violation(violation_id: str, severity: str = "high"):
    """Create a mock violation object."""
    mock = MagicMock()
    mock.id = violation_id
    mock.severity = severity
    mock.to_dict.return_value = {
        "id": violation_id,
        "severity": severity,
        "status": "open",
    }
    return mock


def _create_mock_handler_with_body(body_data, path: str = "/api/v1/policies"):
    """Create mock handler with proper headers and rfile."""
    body_bytes = json.dumps(body_data).encode("utf-8")
    mock_handler = MagicMock()
    mock_handler.headers = {"Content-Length": str(len(body_bytes))}
    mock_handler.rfile = io.BytesIO(body_bytes)
    mock_handler.path = path
    mock_handler.user_context = None
    return mock_handler


class TestPolicyListEndpoint:
    """Test GET /api/v1/policies endpoint."""

    @pytest.mark.asyncio
    async def test_list_policies_success(self, handler):
        """Test successful policy listing."""
        mock_store = MagicMock()
        policies = [
            _create_mock_policy("pol_1", "HIPAA Compliance"),
            _create_mock_policy("pol_2", "SOC2 Audit"),
        ]
        mock_store.list_policies.return_value = policies

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies", "GET", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 2
        assert len(body["policies"]) == 2

    @pytest.mark.asyncio
    async def test_list_policies_with_filters(self, handler):
        """Test policy listing with query filters."""
        mock_store = MagicMock()
        mock_store.list_policies.return_value = []

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies?workspace_id=ws1&enabled_only=true&limit=10"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies", "GET", mock_http)

        assert result.status_code == 200
        mock_store.list_policies.assert_called_once()
        call_kwargs = mock_store.list_policies.call_args[1]
        assert call_kwargs["workspace_id"] == "ws1"
        assert call_kwargs["enabled_only"] is True
        assert call_kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_list_policies_store_unavailable(self, handler):
        """Test policy listing when store is unavailable."""
        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies"

        with patch.object(handler, "_get_policy_store", return_value=None):
            result = await handler.handle("/api/v1/policies", "GET", mock_http)

        assert result.status_code == 503
        body = json.loads(result.body)
        assert "not available" in body["error"]


class TestPolicyGetEndpoint:
    """Test GET /api/v1/policies/:id endpoint."""

    @pytest.mark.asyncio
    async def test_get_policy_success(self, handler):
        """Test successful policy retrieval."""
        mock_store = MagicMock()
        mock_policy = MagicMock()
        mock_policy.to_dict.return_value = {
            "id": "pol_123",
            "name": "Test Policy",
            "enabled": True,
        }
        mock_store.get_policy.return_value = mock_policy

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies/pol_123"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123", "GET", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["policy"]["id"] == "pol_123"

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, handler):
        """Test policy not found."""
        mock_store = MagicMock()
        mock_store.get_policy.return_value = None

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies/pol_nonexistent"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_nonexistent", "GET", mock_http)

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "not found" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_get_policy_invalid_id(self, handler):
        """Test invalid policy ID validation (path traversal)."""
        # Use a valid path structure but with an invalid ID containing traversal
        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies/..%2fetc"

        result = await handler.handle("/api/v1/policies/..%2fetc", "GET", mock_http)

        # The validation should reject this - either 400 or the handler won't match (None)
        # Since the ID contains special characters, it should be rejected
        assert result is not None
        assert result.status_code == 400


class TestPolicyCreateEndpoint:
    """Test POST /api/v1/policies endpoint."""

    @pytest.mark.asyncio
    async def test_create_policy_success(self, handler):
        """Test successful policy creation."""
        mock_store = MagicMock()
        mock_policy = MagicMock()
        mock_policy.to_dict.return_value = {
            "id": "pol_new123",
            "name": "New Policy",
            "framework_id": "soc2",
            "vertical_id": "fintech",
            "enabled": True,
        }
        mock_store.create_policy.return_value = mock_policy

        body_data = {
            "name": "New Policy",
            "framework_id": "soc2",
            "vertical_id": "fintech",
            "description": "Test policy",
        }
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/policies")

        # Mock the lazy import of Policy and PolicyRule
        mock_policy_class = MagicMock()
        mock_policy_rule_class = MagicMock()
        mock_policy_rule_class.from_dict = MagicMock(return_value=MagicMock())

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.compliance.policy_store": MagicMock(
                        Policy=mock_policy_class,
                        PolicyRule=mock_policy_rule_class,
                    )
                },
            ):
                result = await handler.handle("/api/v1/policies", "POST", mock_http)

        assert result.status_code == 201
        body = json.loads(result.body)
        assert "policy" in body
        assert body["message"] == "Policy created successfully"

    @pytest.mark.asyncio
    async def test_create_policy_missing_required_fields(self, handler):
        """Test policy creation with missing required fields."""
        mock_store = MagicMock()

        body_data = {"name": "Test Policy"}  # Missing framework_id and vertical_id
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/policies")

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies", "POST", mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "required field" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_create_policy_invalid_json(self, handler):
        """Test policy creation with invalid JSON."""
        mock_store = MagicMock()

        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "12"}
        mock_http.rfile = io.BytesIO(b"invalid json")
        mock_http.path = "/api/v1/policies"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies", "POST", mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "json" in body["error"].lower()


class TestPolicyUpdateEndpoint:
    """Test PATCH /api/v1/policies/:id endpoint."""

    @pytest.mark.asyncio
    async def test_update_policy_success(self, handler):
        """Test successful policy update."""
        mock_store = MagicMock()
        mock_policy = MagicMock()
        mock_policy.to_dict.return_value = {
            "id": "pol_123",
            "name": "Updated Policy",
            "enabled": True,
        }
        mock_store.update_policy.return_value = mock_policy

        body_data = {"name": "Updated Policy", "description": "Updated description"}
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/policies/pol_123")

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123", "PATCH", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["message"] == "Policy updated successfully"

    @pytest.mark.asyncio
    async def test_update_policy_not_found(self, handler):
        """Test updating non-existent policy."""
        mock_store = MagicMock()
        mock_store.update_policy.return_value = None

        body_data = {"name": "Updated"}
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/policies/pol_123")

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_nonexistent", "PATCH", mock_http)

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_update_policy_no_data(self, handler):
        """Test updating policy with no data."""
        mock_store = MagicMock()

        body_data = {}
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/policies/pol_123")

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123", "PATCH", mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "no update data" in body["error"].lower()


class TestPolicyDeleteEndpoint:
    """Test DELETE /api/v1/policies/:id endpoint."""

    @pytest.mark.asyncio
    async def test_delete_policy_success(self, handler):
        """Test successful policy deletion."""
        mock_store = MagicMock()
        mock_store.delete_policy.return_value = True

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies/pol_123"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123", "DELETE", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["message"] == "Policy deleted successfully"

    @pytest.mark.asyncio
    async def test_delete_policy_not_found(self, handler):
        """Test deleting non-existent policy."""
        mock_store = MagicMock()
        mock_store.delete_policy.return_value = False

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies/pol_nonexistent"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_nonexistent", "DELETE", mock_http)

        assert result.status_code == 404


class TestPolicyToggleEndpoint:
    """Test POST /api/v1/policies/:id/toggle endpoint."""

    @pytest.mark.asyncio
    async def test_toggle_policy_enable(self, handler):
        """Test enabling a policy."""
        mock_store = MagicMock()
        mock_store.toggle_policy.return_value = True

        body_data = {"enabled": True}
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/policies/pol_123/toggle")

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123/toggle", "POST", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["enabled"] is True
        assert "enabled successfully" in body["message"]

    @pytest.mark.asyncio
    async def test_toggle_policy_disable(self, handler):
        """Test disabling a policy."""
        mock_store = MagicMock()
        mock_store.toggle_policy.return_value = True

        body_data = {"enabled": False}
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/policies/pol_123/toggle")

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123/toggle", "POST", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["enabled"] is False
        assert "disabled successfully" in body["message"]

    @pytest.mark.asyncio
    async def test_toggle_policy_auto_toggle(self, handler):
        """Test auto-toggle when no enabled value provided."""
        mock_store = MagicMock()
        mock_policy = MagicMock()
        mock_policy.enabled = True  # Currently enabled
        mock_store.get_policy.return_value = mock_policy
        mock_store.toggle_policy.return_value = True

        body_data = {}  # No enabled field
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/policies/pol_123/toggle")

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123/toggle", "POST", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["enabled"] is False  # Should be toggled to False

    @pytest.mark.asyncio
    async def test_toggle_policy_not_found(self, handler):
        """Test toggling non-existent policy."""
        mock_store = MagicMock()
        mock_store.toggle_policy.return_value = False

        body_data = {"enabled": True}
        mock_http = _create_mock_handler_with_body(
            body_data, "/api/v1/policies/pol_nonexistent/toggle"
        )

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle(
                "/api/v1/policies/pol_nonexistent/toggle", "POST", mock_http
            )

        assert result.status_code == 404


class TestPolicyViolationsEndpoint:
    """Test GET /api/v1/policies/:id/violations endpoint."""

    @pytest.mark.asyncio
    async def test_get_policy_violations_success(self, handler):
        """Test successful policy violations retrieval."""
        mock_store = MagicMock()
        mock_policy = MagicMock()
        mock_store.get_policy.return_value = mock_policy

        violations = [
            _create_mock_violation("viol_1", "critical"),
            _create_mock_violation("viol_2", "high"),
        ]
        mock_store.list_violations.return_value = violations

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies/pol_123/violations"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123/violations", "GET", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 2
        assert body["policy_id"] == "pol_123"

    @pytest.mark.asyncio
    async def test_get_policy_violations_policy_not_found(self, handler):
        """Test violations for non-existent policy."""
        mock_store = MagicMock()
        mock_store.get_policy.return_value = None

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies/pol_nonexistent/violations"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle(
                "/api/v1/policies/pol_nonexistent/violations", "GET", mock_http
            )

        assert result.status_code == 404


class TestComplianceViolationsListEndpoint:
    """Test GET /api/v1/compliance/violations endpoint."""

    @pytest.mark.asyncio
    async def test_list_violations_success(self, handler):
        """Test successful violations listing."""
        mock_store = MagicMock()
        violations = [
            _create_mock_violation("viol_1"),
            _create_mock_violation("viol_2"),
        ]
        mock_store.list_violations.return_value = violations

        mock_http = MagicMock()
        mock_http.path = "/api/v1/compliance/violations"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/compliance/violations", "GET", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_list_violations_with_filters(self, handler):
        """Test violations listing with query filters."""
        mock_store = MagicMock()
        mock_store.list_violations.return_value = []

        mock_http = MagicMock()
        mock_http.path = "/api/v1/compliance/violations?status=open&severity=critical"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/compliance/violations", "GET", mock_http)

        assert result.status_code == 200
        call_kwargs = mock_store.list_violations.call_args[1]
        assert call_kwargs["status"] == "open"
        assert call_kwargs["severity"] == "critical"


class TestComplianceViolationGetEndpoint:
    """Test GET /api/v1/compliance/violations/:id endpoint."""

    @pytest.mark.asyncio
    async def test_get_violation_success(self, handler):
        """Test successful violation retrieval."""
        mock_store = MagicMock()
        mock_violation = MagicMock()
        mock_violation.to_dict.return_value = {
            "id": "viol_123",
            "status": "open",
            "severity": "high",
        }
        mock_store.get_violation.return_value = mock_violation

        mock_http = MagicMock()
        mock_http.path = "/api/v1/compliance/violations/viol_123"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle(
                "/api/v1/compliance/violations/viol_123", "GET", mock_http
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["violation"]["id"] == "viol_123"

    @pytest.mark.asyncio
    async def test_get_violation_not_found(self, handler):
        """Test violation not found."""
        mock_store = MagicMock()
        mock_store.get_violation.return_value = None

        mock_http = MagicMock()
        mock_http.path = "/api/v1/compliance/violations/viol_nonexistent"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle(
                "/api/v1/compliance/violations/viol_nonexistent", "GET", mock_http
            )

        assert result.status_code == 404


class TestComplianceViolationUpdateEndpoint:
    """Test PATCH /api/v1/compliance/violations/:id endpoint."""

    @pytest.mark.asyncio
    async def test_update_violation_status_success(self, handler):
        """Test successful violation status update."""
        mock_store = MagicMock()
        mock_violation = MagicMock()
        mock_violation.to_dict.return_value = {
            "id": "viol_123",
            "status": "resolved",
        }
        mock_store.update_violation_status.return_value = mock_violation

        body_data = {"status": "resolved", "resolution_notes": "Fixed the issue"}
        mock_http = _create_mock_handler_with_body(
            body_data, "/api/v1/compliance/violations/viol_123"
        )

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle(
                "/api/v1/compliance/violations/viol_123", "PATCH", mock_http
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "resolved" in body["message"]

    @pytest.mark.asyncio
    async def test_update_violation_missing_status(self, handler):
        """Test update with missing status field."""
        mock_store = MagicMock()

        body_data = {"resolution_notes": "Some notes"}  # No status
        mock_http = _create_mock_handler_with_body(
            body_data, "/api/v1/compliance/violations/viol_123"
        )

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle(
                "/api/v1/compliance/violations/viol_123", "PATCH", mock_http
            )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "status" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_update_violation_invalid_status(self, handler):
        """Test update with invalid status value."""
        mock_store = MagicMock()

        body_data = {"status": "invalid_status"}
        mock_http = _create_mock_handler_with_body(
            body_data, "/api/v1/compliance/violations/viol_123"
        )

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle(
                "/api/v1/compliance/violations/viol_123", "PATCH", mock_http
            )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "invalid status" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_update_violation_all_valid_statuses(self, handler):
        """Test that all valid statuses are accepted."""
        valid_statuses = ["open", "investigating", "resolved", "false_positive"]
        mock_store = MagicMock()
        mock_violation = MagicMock()
        mock_violation.to_dict.return_value = {"id": "viol_123", "status": "open"}
        mock_store.update_violation_status.return_value = mock_violation

        for status in valid_statuses:
            body_data = {"status": status}
            mock_http = _create_mock_handler_with_body(
                body_data, "/api/v1/compliance/violations/viol_123"
            )

            with patch.object(handler, "_get_policy_store", return_value=mock_store):
                result = await handler.handle(
                    "/api/v1/compliance/violations/viol_123", "PATCH", mock_http
                )

            assert result.status_code == 200, f"Status {status} should be valid"


class TestComplianceCheckEndpoint:
    """Test POST /api/v1/compliance/check endpoint."""

    @pytest.mark.asyncio
    async def test_compliance_check_success(self, handler):
        """Test successful compliance check."""
        mock_manager = MagicMock()
        mock_result = MagicMock()
        mock_result.compliant = True
        mock_result.score = 95
        mock_result.issues = []
        mock_result.to_dict.return_value = {
            "compliant": True,
            "score": 95,
            "issues": [],
        }
        mock_manager.check.return_value = mock_result

        body_data = {"content": "Test content for compliance check"}
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/compliance/check")

        with patch.object(handler, "_get_compliance_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/compliance/check", "POST", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["compliant"] is True
        assert body["score"] == 95

    @pytest.mark.asyncio
    async def test_compliance_check_with_violations(self, handler):
        """Test compliance check that finds violations."""
        mock_manager = MagicMock()
        mock_issue = MagicMock()
        mock_issue.rule_id = "rule_1"
        mock_issue.description = "PII detected"
        mock_issue.framework = "gdpr"
        mock_issue.severity = MagicMock()
        mock_issue.severity.value = "high"
        mock_issue.metadata = {}

        mock_result = MagicMock()
        mock_result.compliant = False
        mock_result.score = 40
        mock_result.issues = [mock_issue]
        mock_result.to_dict.return_value = {
            "compliant": False,
            "score": 40,
            "issues": [{"rule_id": "rule_1", "description": "PII detected"}],
        }
        mock_manager.check.return_value = mock_result

        body_data = {
            "content": "Email: test@example.com, SSN: 123-45-6789",
            "store_violations": False,
        }
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/compliance/check")

        with patch.object(handler, "_get_compliance_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/compliance/check", "POST", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["compliant"] is False
        assert body["issue_count"] == 1

    @pytest.mark.asyncio
    async def test_compliance_check_missing_content(self, handler):
        """Test compliance check without content."""
        mock_manager = MagicMock()

        body_data = {}  # No content
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/compliance/check")

        with patch.object(handler, "_get_compliance_manager", return_value=mock_manager):
            result = await handler.handle("/api/v1/compliance/check", "POST", mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "content" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_compliance_check_invalid_severity(self, handler):
        """Test compliance check with invalid severity level."""
        mock_manager = MagicMock()

        body_data = {"content": "Test content", "min_severity": "ultra_high"}
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/compliance/check")

        # Create a mock ComplianceSeverity that raises ValueError for invalid values
        mock_severity_class = MagicMock()
        mock_severity_class.side_effect = ValueError("Invalid severity")

        with patch.object(handler, "_get_compliance_manager", return_value=mock_manager):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.compliance.framework": MagicMock(
                        ComplianceSeverity=mock_severity_class,
                    )
                },
            ):
                result = await handler.handle("/api/v1/compliance/check", "POST", mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "min_severity" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_compliance_check_manager_unavailable(self, handler):
        """Test compliance check when manager is unavailable."""
        body_data = {"content": "Test content"}
        mock_http = _create_mock_handler_with_body(body_data, "/api/v1/compliance/check")

        with patch.object(handler, "_get_compliance_manager", return_value=None):
            result = await handler.handle("/api/v1/compliance/check", "POST", mock_http)

        assert result.status_code == 503


class TestComplianceStatsEndpoint:
    """Test GET /api/v1/compliance/stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler):
        """Test successful stats retrieval."""
        mock_store = MagicMock()

        # Mock policies
        policies = [
            _create_mock_policy("pol_1", "Policy 1", enabled=True),
            _create_mock_policy("pol_2", "Policy 2", enabled=True),
            _create_mock_policy("pol_3", "Policy 3", enabled=False),
        ]
        mock_store.list_policies.return_value = policies

        # Mock violation counts
        mock_store.count_violations.side_effect = [
            {"total": 5, "critical": 1, "high": 2, "medium": 1, "low": 1},  # open
            {"total": 10, "critical": 2, "high": 3, "medium": 3, "low": 2},  # all
        ]

        mock_http = MagicMock()
        mock_http.path = "/api/v1/compliance/stats"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/compliance/stats", "GET", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["policies"]["total"] == 3
        assert body["policies"]["enabled"] == 2
        assert body["violations"]["total"] == 10
        assert body["violations"]["open"] == 5
        assert "risk_score" in body

    @pytest.mark.asyncio
    async def test_get_stats_risk_score_calculation(self, handler):
        """Test risk score calculation logic."""
        mock_store = MagicMock()
        mock_store.list_policies.return_value = []

        # High risk scenario: 4 critical, 3 high violations open
        mock_store.count_violations.side_effect = [
            {"total": 7, "critical": 4, "high": 3, "medium": 0, "low": 0},  # open
            {"total": 7, "critical": 4, "high": 3, "medium": 0, "low": 0},  # all
        ]

        mock_http = MagicMock()
        mock_http.path = "/api/v1/compliance/stats"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/compliance/stats", "GET", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        # Risk score: 4*25 + 3*10 = 100 + 30 = 130, capped at 100
        assert body["risk_score"] == 100


class TestPolicyHandlerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unmatched_route(self, handler):
        """Test that handle returns None for unmatched routes."""
        mock_http = MagicMock()
        mock_http.path = "/api/v1/unknown"

        result = await handler.handle("/api/v1/unknown", "GET", mock_http)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_handling_in_list_policies(self, handler):
        """Test exception handling in list policies."""
        mock_store = MagicMock()
        mock_store.list_policies.side_effect = Exception("Database error")

        mock_http = MagicMock()
        mock_http.path = "/api/v1/policies"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies", "GET", mock_http)

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_user_context_extraction_for_audit(self, handler):
        """Test that user context is extracted for audit trails."""
        mock_store = MagicMock()
        mock_policy = MagicMock()
        mock_policy.to_dict.return_value = {"id": "pol_123"}
        mock_store.update_policy.return_value = mock_policy

        body_bytes = json.dumps({"name": "Updated"}).encode("utf-8")
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": str(len(body_bytes))}
        mock_http.rfile = io.BytesIO(body_bytes)
        mock_http.path = "/api/v1/policies/pol_123"

        # Set up user context
        mock_http.user_context = MagicMock()
        mock_http.user_context.user_id = "user_456"

        with patch.object(handler, "_get_policy_store", return_value=mock_store):
            result = await handler.handle("/api/v1/policies/pol_123", "PATCH", mock_http)

        assert result.status_code == 200
        # Verify user_id was passed to update_policy
        call_kwargs = mock_store.update_policy.call_args[1]
        assert call_kwargs["changed_by"] == "user_456"


# Export test classes
__all__ = [
    "TestPolicyHandlerRouting",
    "TestPolicyListEndpoint",
    "TestPolicyGetEndpoint",
    "TestPolicyCreateEndpoint",
    "TestPolicyUpdateEndpoint",
    "TestPolicyDeleteEndpoint",
    "TestPolicyToggleEndpoint",
    "TestPolicyViolationsEndpoint",
    "TestComplianceViolationsListEndpoint",
    "TestComplianceViolationGetEndpoint",
    "TestComplianceViolationUpdateEndpoint",
    "TestComplianceCheckEndpoint",
    "TestComplianceStatsEndpoint",
    "TestPolicyHandlerEdgeCases",
]
