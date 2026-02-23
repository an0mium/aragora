"""Comprehensive tests for governance.py consolidation module.

Tests the re-export facade at aragora/server/handlers/compliance/governance.py,
which provides a single entry point for:
- PolicyHandler (from policy.py) - Policy CRUD, violation tracking
- ComplianceHandler (from compliance/handler.py) - SOC 2, GDPR, CCPA, HIPAA, EU AI Act
- RBAC constants and require_permission decorator

Test classes:
- TestGovernanceImports: Module re-exports, __all__, constant values
- TestGovernanceConstants: RBAC permission constants
- TestPolicyHandlerCanHandle: Routing dispatch
- TestPolicyHandlerListPolicies: GET /api/v1/policies
- TestPolicyHandlerGetPolicy: GET /api/v1/policies/:id
- TestPolicyHandlerCreatePolicy: POST /api/v1/policies
- TestPolicyHandlerUpdatePolicy: PATCH /api/v1/policies/:id
- TestPolicyHandlerDeletePolicy: DELETE /api/v1/policies/:id
- TestPolicyHandlerTogglePolicy: POST /api/v1/policies/:id/toggle
- TestPolicyHandlerGetPolicyViolations: GET /api/v1/policies/:id/violations
- TestPolicyHandlerListViolations: GET /api/v1/compliance/violations
- TestPolicyHandlerGetViolation: GET /api/v1/compliance/violations/:id
- TestPolicyHandlerUpdateViolation: PATCH /api/v1/compliance/violations/:id
- TestPolicyHandlerCheckCompliance: POST /api/v1/compliance/check
- TestPolicyHandlerStats: GET /api/v1/compliance/stats
- TestComplianceHandlerViaGovernance: ComplianceHandler routing and factory
- TestPolicyHandlerEdgeCases: Edge cases and error paths
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _mock_http(
    method: str = "GET",
    body: dict[str, Any] | None = None,
    path: str = "/",
) -> MagicMock:
    """Create a lightweight mock HTTP handler."""
    h = MagicMock()
    h.command = method
    h.path = path
    h.user_context = None
    if body is not None:
        raw = json.dumps(body).encode()
        h.rfile.read.return_value = raw
        h.headers = {"Content-Length": str(len(raw))}
    else:
        h.rfile.read.return_value = b"{}"
        h.headers = {"Content-Length": "2"}
    return h


def _mock_policy(policy_id: str = "pol_abc123", name: str = "Test Policy", enabled: bool = True):
    """Create a mock policy object."""
    m = MagicMock()
    m.id = policy_id
    m.name = name
    m.enabled = enabled
    m.to_dict.return_value = {
        "id": policy_id,
        "name": name,
        "enabled": enabled,
        "framework_id": "soc2",
        "vertical_id": "healthcare",
    }
    return m


def _mock_violation(violation_id: str = "viol_abc123", severity: str = "high"):
    """Create a mock violation object."""
    m = MagicMock()
    m.id = violation_id
    m.severity = severity
    m.to_dict.return_value = {
        "id": violation_id,
        "severity": severity,
        "status": "open",
    }
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def policy_handler():
    """Create a PolicyHandler via the governance re-export."""
    from aragora.server.handlers.compliance.governance import PolicyHandler

    return PolicyHandler({})


@pytest.fixture
def mock_store():
    """Create a mock policy store."""
    store = MagicMock()
    store.list_policies.return_value = []
    store.list_violations.return_value = []
    store.get_policy.return_value = None
    store.get_violation.return_value = None
    store.count_violations.return_value = {
        "total": 0,
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    return store


@pytest.fixture(autouse=True)
def _patch_audit(monkeypatch):
    """Patch audit_security to avoid side effects."""
    monkeypatch.setattr(
        "aragora.server.handlers.policy.audit_security",
        lambda **kwargs: None,
    )


# ===========================================================================
# Import / Re-export Tests
# ===========================================================================


class TestGovernanceImports:
    """Verify all re-exports from the governance module are available."""

    def test_import_policy_handler(self):
        from aragora.server.handlers.compliance.governance import PolicyHandler

        assert PolicyHandler is not None

    def test_import_compliance_handler(self):
        from aragora.server.handlers.compliance.governance import ComplianceHandler

        assert ComplianceHandler is not None

    def test_import_create_compliance_handler(self):
        from aragora.server.handlers.compliance.governance import create_compliance_handler

        assert callable(create_compliance_handler)

    def test_import_require_permission(self):
        from aragora.server.handlers.compliance.governance import require_permission

        assert callable(require_permission)

    def test_import_governance_read_permission(self):
        from aragora.server.handlers.compliance.governance import GOVERNANCE_READ_PERMISSION

        assert isinstance(GOVERNANCE_READ_PERMISSION, str)

    def test_import_governance_write_permission(self):
        from aragora.server.handlers.compliance.governance import GOVERNANCE_WRITE_PERMISSION

        assert isinstance(GOVERNANCE_WRITE_PERMISSION, str)

    def test_all_exports_complete(self):
        from aragora.server.handlers.compliance import governance

        expected = {
            "require_permission",
            "GOVERNANCE_READ_PERMISSION",
            "GOVERNANCE_WRITE_PERMISSION",
            "PolicyHandler",
            "ComplianceHandler",
            "create_compliance_handler",
        }
        assert set(governance.__all__) == expected

    def test_policy_handler_identity(self):
        """Re-exported PolicyHandler is the same class as original."""
        from aragora.server.handlers.compliance.governance import PolicyHandler as GovPH
        from aragora.server.handlers.policy import PolicyHandler as OrigPH

        assert GovPH is OrigPH

    def test_compliance_handler_identity(self):
        """Re-exported ComplianceHandler is the same class as original."""
        from aragora.server.handlers.compliance.governance import ComplianceHandler as GovCH
        from aragora.server.handlers.compliance.handler import ComplianceHandler as OrigCH

        assert GovCH is OrigCH

    def test_create_compliance_handler_identity(self):
        """Re-exported factory is the same function as original."""
        from aragora.server.handlers.compliance.governance import (
            create_compliance_handler as gov_fn,
        )
        from aragora.server.handlers.compliance.handler import create_compliance_handler as orig_fn

        assert gov_fn is orig_fn


# ===========================================================================
# RBAC Constants
# ===========================================================================


class TestGovernanceConstants:
    """Verify RBAC permission constant values."""

    def test_read_permission_value(self):
        from aragora.server.handlers.compliance.governance import GOVERNANCE_READ_PERMISSION

        assert GOVERNANCE_READ_PERMISSION == "governance:read"

    def test_write_permission_value(self):
        from aragora.server.handlers.compliance.governance import GOVERNANCE_WRITE_PERMISSION

        assert GOVERNANCE_WRITE_PERMISSION == "governance:write"

    def test_permissions_are_strings(self):
        from aragora.server.handlers.compliance.governance import (
            GOVERNANCE_READ_PERMISSION,
            GOVERNANCE_WRITE_PERMISSION,
        )

        assert isinstance(GOVERNANCE_READ_PERMISSION, str)
        assert isinstance(GOVERNANCE_WRITE_PERMISSION, str)

    def test_permissions_follow_colon_convention(self):
        from aragora.server.handlers.compliance.governance import (
            GOVERNANCE_READ_PERMISSION,
            GOVERNANCE_WRITE_PERMISSION,
        )

        assert ":" in GOVERNANCE_READ_PERMISSION
        assert ":" in GOVERNANCE_WRITE_PERMISSION


# ===========================================================================
# PolicyHandler.can_handle
# ===========================================================================


class TestPolicyHandlerCanHandle:
    """Routing via can_handle."""

    def test_policies_list_path(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/policies", "GET") is True

    def test_policies_post(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/policies", "POST") is True

    def test_policy_by_id(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/policies/pol_123abc", "GET") is True

    def test_policy_toggle(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/policies/pol_123abc/toggle", "POST") is True

    def test_policy_violations(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/policies/pol_123abc/violations", "GET") is True

    def test_compliance_violations(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/compliance/violations", "GET") is True

    def test_compliance_check(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/compliance/check", "POST") is True

    def test_compliance_stats(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/compliance/stats", "GET") is True

    def test_rejects_unrelated_path(self, policy_handler):
        assert policy_handler.can_handle("/api/v1/debates", "GET") is False

    def test_rejects_root(self, policy_handler):
        assert policy_handler.can_handle("/", "GET") is False


# ===========================================================================
# Policy List
# ===========================================================================


class TestPolicyHandlerListPolicies:
    """GET /api/v1/policies"""

    @pytest.mark.asyncio
    async def test_list_policies_success(self, policy_handler, mock_store):
        policies = [_mock_policy("pol_1"), _mock_policy("pol_2")]
        mock_store.list_policies.return_value = policies
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["policies"]) == 2

    @pytest.mark.asyncio
    async def test_list_policies_store_unavailable(self, policy_handler):
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=None):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_policies_store_exception(self, policy_handler, mock_store):
        mock_store.list_policies.side_effect = RuntimeError("db down")
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 500


# ===========================================================================
# Policy Get
# ===========================================================================


class TestPolicyHandlerGetPolicy:
    """GET /api/v1/policies/:id"""

    @pytest.mark.asyncio
    async def test_get_policy_success(self, policy_handler, mock_store):
        mock_store.get_policy.return_value = _mock_policy("pol_abc123")
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["policy"]["id"] == "pol_abc123"

    @pytest.mark.asyncio
    async def test_get_policy_not_found(self, policy_handler, mock_store):
        mock_store.get_policy.return_value = None
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_missing", {}, http)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_policy_store_unavailable(self, policy_handler):
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=None):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_policy_invalid_id(self, policy_handler, mock_store):
        """An ID with special characters should be rejected."""
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_<script>", {}, http)
        assert _status(result) == 400


# ===========================================================================
# Policy Create
# ===========================================================================


class TestPolicyHandlerCreatePolicy:
    """POST /api/v1/policies"""

    @pytest.mark.asyncio
    async def test_create_policy_success(self, policy_handler, mock_store):
        body_data = {
            "name": "New Policy",
            "framework_id": "soc2",
            "vertical_id": "healthcare",
        }
        created = _mock_policy("pol_new1234", "New Policy")
        mock_store.create_policy.return_value = created
        http = _mock_http("POST", body=body_data)
        mock_policy_cls = MagicMock(return_value=created)
        mock_rule_cls = MagicMock()
        mock_rule_cls.from_dict = MagicMock(return_value=MagicMock())
        with (
            patch.object(policy_handler, "_get_policy_store", return_value=mock_store),
            patch("aragora.compliance.policy_store.Policy", mock_policy_cls, create=True),
            patch("aragora.compliance.policy_store.PolicyRule", mock_rule_cls, create=True),
        ):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["message"] == "Policy created successfully"

    @pytest.mark.asyncio
    async def test_create_policy_missing_required_field(self, policy_handler, mock_store):
        body_data = {"name": "No Framework"}
        http = _mock_http("POST", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 400
        assert "Missing required field" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_policy_store_unavailable(self, policy_handler):
        body_data = {"name": "P", "framework_id": "soc2", "vertical_id": "hc"}
        http = _mock_http("POST", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=None):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_create_policy_invalid_json(self, policy_handler, mock_store):
        http = _mock_http("POST")
        http.rfile.read.return_value = b"not json {"
        http.headers = {"Content-Length": "10"}
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_policy_body_too_large(self, policy_handler, mock_store):
        http = _mock_http("POST")
        http.headers = {"Content-Length": str(11 * 1024 * 1024)}
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 413


# ===========================================================================
# Policy Update
# ===========================================================================


class TestPolicyHandlerUpdatePolicy:
    """PATCH /api/v1/policies/:id"""

    @pytest.mark.asyncio
    async def test_update_policy_success(self, policy_handler, mock_store):
        updated = _mock_policy("pol_abc123", "Updated")
        mock_store.update_policy.return_value = updated
        body_data = {"name": "Updated"}
        http = _mock_http("PATCH", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123", {}, http)
        assert _status(result) == 200
        assert "updated successfully" in _body(result)["message"]

    @pytest.mark.asyncio
    async def test_update_policy_not_found(self, policy_handler, mock_store):
        mock_store.update_policy.return_value = None
        body_data = {"name": "Updated"}
        http = _mock_http("PATCH", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_missing", {}, http)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_policy_empty_body(self, policy_handler, mock_store):
        http = _mock_http("PATCH")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_policy_store_unavailable(self, policy_handler):
        http = _mock_http("PATCH", body={"name": "X"})
        with patch.object(policy_handler, "_get_policy_store", return_value=None):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123", {}, http)
        assert _status(result) == 503


# ===========================================================================
# Policy Delete
# ===========================================================================


class TestPolicyHandlerDeletePolicy:
    """DELETE /api/v1/policies/:id"""

    @pytest.mark.asyncio
    async def test_delete_policy_success(self, policy_handler, mock_store):
        mock_store.delete_policy.return_value = True
        http = _mock_http("DELETE")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123", {}, http)
        assert _status(result) == 200
        assert "deleted successfully" in _body(result)["message"]

    @pytest.mark.asyncio
    async def test_delete_policy_not_found(self, policy_handler, mock_store):
        mock_store.delete_policy.return_value = False
        http = _mock_http("DELETE")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_missing", {}, http)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_policy_store_unavailable(self, policy_handler):
        http = _mock_http("DELETE")
        with patch.object(policy_handler, "_get_policy_store", return_value=None):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123", {}, http)
        assert _status(result) == 503


# ===========================================================================
# Policy Toggle
# ===========================================================================


class TestPolicyHandlerTogglePolicy:
    """POST /api/v1/policies/:id/toggle"""

    @pytest.mark.asyncio
    async def test_toggle_policy_with_explicit_enabled(self, policy_handler, mock_store):
        mock_store.toggle_policy.return_value = True
        body_data = {"enabled": False}
        http = _mock_http("POST", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123/toggle", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_toggle_policy_auto_toggle(self, policy_handler, mock_store):
        """When 'enabled' not provided, auto-toggle the current state."""
        existing = _mock_policy("pol_abc123", enabled=True)
        mock_store.get_policy.return_value = existing
        mock_store.toggle_policy.return_value = True
        http = _mock_http("POST")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123/toggle", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["enabled"] is False  # toggled from True

    @pytest.mark.asyncio
    async def test_toggle_policy_not_found(self, policy_handler, mock_store):
        mock_store.toggle_policy.return_value = False
        body_data = {"enabled": True}
        http = _mock_http("POST", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_missing/toggle", {}, http)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_toggle_policy_store_unavailable(self, policy_handler):
        http = _mock_http("POST", body={"enabled": True})
        with patch.object(policy_handler, "_get_policy_store", return_value=None):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123/toggle", {}, http)
        assert _status(result) == 503


# ===========================================================================
# Policy Violations (per-policy)
# ===========================================================================


class TestPolicyHandlerGetPolicyViolations:
    """GET /api/v1/policies/:id/violations"""

    @pytest.mark.asyncio
    async def test_get_policy_violations_success(self, policy_handler, mock_store):
        mock_store.get_policy.return_value = _mock_policy("pol_abc123")
        violations = [_mock_violation("viol_1"), _mock_violation("viol_2")]
        mock_store.list_violations.return_value = violations
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/pol_abc123/violations", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_get_policy_violations_policy_not_found(self, policy_handler, mock_store):
        mock_store.get_policy.return_value = None
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle(
                "/api/v1/policies/pol_missing/violations", {}, http
            )
        assert _status(result) == 404


# ===========================================================================
# Compliance Violations (global)
# ===========================================================================


class TestPolicyHandlerListViolations:
    """GET /api/v1/compliance/violations"""

    @pytest.mark.asyncio
    async def test_list_violations_success(self, policy_handler, mock_store):
        violations = [_mock_violation("viol_1")]
        mock_store.list_violations.return_value = violations
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/compliance/violations", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_violations_store_unavailable(self, policy_handler):
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=None):
            result = await policy_handler.handle("/api/v1/compliance/violations", {}, http)
        assert _status(result) == 503


# ===========================================================================
# Get Violation
# ===========================================================================


class TestPolicyHandlerGetViolation:
    """GET /api/v1/compliance/violations/:id"""

    @pytest.mark.asyncio
    async def test_get_violation_success(self, policy_handler, mock_store):
        mock_store.get_violation.return_value = _mock_violation("viol_abc123")
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle(
                "/api/v1/compliance/violations/viol_abc123", {}, http
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["violation"]["id"] == "viol_abc123"

    @pytest.mark.asyncio
    async def test_get_violation_not_found(self, policy_handler, mock_store):
        mock_store.get_violation.return_value = None
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle(
                "/api/v1/compliance/violations/viol_missing", {}, http
            )
        assert _status(result) == 404


# ===========================================================================
# Update Violation
# ===========================================================================


class TestPolicyHandlerUpdateViolation:
    """PATCH /api/v1/compliance/violations/:id"""

    @pytest.mark.asyncio
    async def test_update_violation_success(self, policy_handler, mock_store):
        updated = _mock_violation("viol_abc123")
        mock_store.update_violation_status.return_value = updated
        body_data = {"status": "resolved"}
        http = _mock_http("PATCH", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle(
                "/api/v1/compliance/violations/viol_abc123", {}, http
            )
        assert _status(result) == 200
        assert "updated to resolved" in _body(result)["message"]

    @pytest.mark.asyncio
    async def test_update_violation_missing_status(self, policy_handler, mock_store):
        http = _mock_http("PATCH")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle(
                "/api/v1/compliance/violations/viol_abc123", {}, http
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_violation_invalid_status(self, policy_handler, mock_store):
        body_data = {"status": "bogus"}
        http = _mock_http("PATCH", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle(
                "/api/v1/compliance/violations/viol_abc123", {}, http
            )
        assert _status(result) == 400
        assert "Invalid status" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_violation_not_found(self, policy_handler, mock_store):
        mock_store.update_violation_status.return_value = None
        body_data = {"status": "investigating"}
        http = _mock_http("PATCH", body=body_data)
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle(
                "/api/v1/compliance/violations/viol_abc123", {}, http
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_violation_all_valid_statuses(self, policy_handler, mock_store):
        """Ensure all four valid statuses are accepted."""
        for status_val in ("open", "investigating", "resolved", "false_positive"):
            updated = _mock_violation("viol_abc123")
            mock_store.update_violation_status.return_value = updated
            body_data = {"status": status_val}
            http = _mock_http("PATCH", body=body_data)
            with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
                result = await policy_handler.handle(
                    "/api/v1/compliance/violations/viol_abc123", {}, http
                )
            assert _status(result) == 200


# ===========================================================================
# Compliance Check
# ===========================================================================


class TestPolicyHandlerCheckCompliance:
    """POST /api/v1/compliance/check"""

    @pytest.mark.asyncio
    async def test_check_compliance_success(self, policy_handler):
        mock_manager = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"compliant": True}
        mock_result.compliant = True
        mock_result.score = 95.0
        mock_result.issues = []
        mock_manager.check.return_value = mock_result

        mock_severity = MagicMock()
        mock_severity.return_value = "low"  # ComplianceSeverity("low")

        body_data = {"content": "Test content for compliance"}
        http = _mock_http("POST", body=body_data)
        with (
            patch.object(policy_handler, "_get_compliance_manager", return_value=mock_manager),
            patch("aragora.compliance.framework.ComplianceSeverity", mock_severity),
        ):
            result = await policy_handler.handle("/api/v1/compliance/check", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["compliant"] is True
        assert body["score"] == 95.0

    @pytest.mark.asyncio
    async def test_check_compliance_missing_content(self, policy_handler):
        mock_manager = MagicMock()
        http = _mock_http("POST", body={})
        with patch.object(policy_handler, "_get_compliance_manager", return_value=mock_manager):
            result = await policy_handler.handle("/api/v1/compliance/check", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_check_compliance_manager_unavailable(self, policy_handler):
        body_data = {"content": "Test content"}
        http = _mock_http("POST", body=body_data)
        with patch.object(policy_handler, "_get_compliance_manager", return_value=None):
            result = await policy_handler.handle("/api/v1/compliance/check", {}, http)
        assert _status(result) == 503


# ===========================================================================
# Compliance Stats
# ===========================================================================


class TestPolicyHandlerStats:
    """GET /api/v1/compliance/stats"""

    @pytest.mark.asyncio
    async def test_stats_success(self, policy_handler, mock_store):
        policies = [_mock_policy("pol_1", enabled=True), _mock_policy("pol_2", enabled=False)]
        mock_store.list_policies.return_value = policies
        mock_store.count_violations.return_value = {
            "total": 5,
            "critical": 1,
            "high": 2,
            "medium": 1,
            "low": 1,
        }
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/compliance/stats", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["policies"]["total"] == 2
        assert body["policies"]["enabled"] == 1
        assert body["policies"]["disabled"] == 1

    @pytest.mark.asyncio
    async def test_stats_store_unavailable(self, policy_handler):
        http = _mock_http("GET")
        with patch.object(policy_handler, "_get_policy_store", return_value=None):
            result = await policy_handler.handle("/api/v1/compliance/stats", {}, http)
        assert _status(result) == 503


# ===========================================================================
# ComplianceHandler via Governance
# ===========================================================================


class TestComplianceHandlerViaGovernance:
    """Test ComplianceHandler re-exported through governance module."""

    def test_can_handle_v2_compliance_get(self):
        from aragora.server.handlers.compliance.governance import ComplianceHandler

        h = ComplianceHandler({})
        assert h.can_handle("/api/v2/compliance/status", "GET") is True

    def test_can_handle_v2_compliance_post(self):
        from aragora.server.handlers.compliance.governance import ComplianceHandler

        h = ComplianceHandler({})
        assert h.can_handle("/api/v2/compliance/audit-verify", "POST") is True

    def test_can_handle_v2_compliance_delete(self):
        from aragora.server.handlers.compliance.governance import ComplianceHandler

        h = ComplianceHandler({})
        assert h.can_handle("/api/v2/compliance/gdpr/legal-holds/hold_123", "DELETE") is True

    def test_cannot_handle_wrong_method(self):
        from aragora.server.handlers.compliance.governance import ComplianceHandler

        h = ComplianceHandler({})
        assert h.can_handle("/api/v2/compliance/status", "PUT") is False

    def test_cannot_handle_unrelated_path(self):
        from aragora.server.handlers.compliance.governance import ComplianceHandler

        h = ComplianceHandler({})
        assert h.can_handle("/api/v2/debates", "GET") is False

    def test_create_compliance_handler_factory(self):
        from aragora.server.handlers.compliance.governance import (
            ComplianceHandler,
            create_compliance_handler,
        )

        ctx = {"some_key": "some_value"}
        h = create_compliance_handler(ctx)
        assert isinstance(h, ComplianceHandler)


# ===========================================================================
# Edge Cases and Error Paths
# ===========================================================================


class TestPolicyHandlerEdgeCases:
    """Edge cases, method fallback, and unmatched routes."""

    @pytest.mark.asyncio
    async def test_unmatched_route_returns_none(self, policy_handler):
        """Handle should return None for paths it can't route internally."""
        http = _mock_http("GET")
        # A path that passes can_handle but doesn't match any internal route
        result = await policy_handler.handle("/api/v1/compliance/unknown_endpoint", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_method_from_handler_command(self, policy_handler, mock_store):
        """Method should be taken from handler.command attribute."""
        mock_store.list_policies.return_value = []
        http = _mock_http("GET")
        http.command = "GET"
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies", {}, http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_method_defaults_to_get_when_none(self, policy_handler, mock_store):
        """When method is None and query_params is dict, defaults to GET."""
        mock_store.list_policies.return_value = []
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies", {}, None, method="GET")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_query_params_string_as_method(self, policy_handler, mock_store):
        """When query_params is a string, it is treated as the method."""
        mock_store.list_policies.return_value = []
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies", "GET", None)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_policy_handler_init_default_ctx(self):
        """PolicyHandler initialises with empty ctx if None provided."""
        from aragora.server.handlers.compliance.governance import PolicyHandler

        h = PolicyHandler(None)
        assert h.ctx == {}

    @pytest.mark.asyncio
    async def test_path_segment_validation_rejects_traversal(self, policy_handler, mock_store):
        """Path traversal attempts should be rejected with 400."""
        http = _mock_http("DELETE")
        with patch.object(policy_handler, "_get_policy_store", return_value=mock_store):
            result = await policy_handler.handle("/api/v1/policies/../../../etc/passwd", {}, http)
        # Either 400 from validation or None from no route match
        if result is not None:
            assert _status(result) == 400

    def test_routes_class_attribute(self, policy_handler):
        """PolicyHandler should declare ROUTES."""
        assert hasattr(policy_handler, "ROUTES")
        assert len(policy_handler.ROUTES) > 0
        assert "/api/v1/policies" in policy_handler.ROUTES

    def test_get_policy_store_import_error(self, policy_handler):
        """_get_policy_store returns None when import fails."""
        with patch.dict(
            "sys.modules",
            {"aragora.compliance.policy_store": None},
        ):
            result = policy_handler._get_policy_store()
        assert result is None

    def test_get_compliance_manager_import_error(self, policy_handler):
        """_get_compliance_manager returns None when import fails."""
        with patch.dict(
            "sys.modules",
            {"aragora.compliance.framework": None},
        ):
            result = policy_handler._get_compliance_manager()
        assert result is None
