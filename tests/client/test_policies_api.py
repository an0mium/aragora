"""Tests for PoliciesAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.policies import (
    ComplianceCheckResult,
    ComplianceStats,
    PoliciesAPI,
    Policy,
    PolicyRule,
    PolicyViolation,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> PoliciesAPI:
    return PoliciesAPI(mock_client)


SAMPLE_RULE = {
    "id": "rule-1",
    "name": "No PII exposure",
    "description": "Prevents PII from leaking",
    "condition": "contains_pii(content)",
    "severity": "critical",
    "enabled": True,
    "metadata": {"category": "privacy"},
}

SAMPLE_POLICY = {
    "id": "pol-123",
    "name": "GDPR Policy",
    "description": "GDPR compliance rules",
    "framework_id": "gdpr",
    "workspace_id": "ws-1",
    "vertical_id": "healthcare",
    "level": "required",
    "enabled": True,
    "rules": [SAMPLE_RULE],
    "created_at": "2026-01-15T10:00:00Z",
    "updated_at": "2026-01-15T12:00:00Z",
    "metadata": {"version": "1.0"},
}

SAMPLE_VIOLATION = {
    "id": "viol-456",
    "policy_id": "pol-123",
    "rule_id": "rule-1",
    "rule_name": "No PII exposure",
    "framework_id": "gdpr",
    "vertical_id": "healthcare",
    "workspace_id": "ws-1",
    "severity": "critical",
    "status": "open",
    "description": "PII detected in output",
    "source": "debate-789",
    "created_at": "2026-01-15T11:00:00Z",
    "resolved_at": None,
    "resolved_by": None,
    "resolution_notes": None,
    "metadata": {"field": "response"},
}

SAMPLE_CHECK_RESULT = {
    "compliant": False,
    "score": 72.5,
    "issues": [{"rule": "pii_check", "severity": "high", "message": "PII found"}],
    "checked_at": "2026-01-15T11:30:00Z",
}

SAMPLE_STATS = {
    "policies": {"total": 10, "enabled": 8, "disabled": 2},
    "violations": {
        "total": 25,
        "open": 5,
        "by_severity": {"critical": 2, "high": 3},
    },
    "risk_score": 35,
}


# =========================================================================
# Policy CRUD
# =========================================================================


class TestPoliciesList:
    def test_list_default(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"policies": [SAMPLE_POLICY], "total": 1}
        policies, total = api.list()
        assert len(policies) == 1
        assert total == 1
        assert isinstance(policies[0], Policy)
        assert policies[0].id == "pol-123"
        assert policies[0].name == "GDPR Policy"
        mock_client._get.assert_called_once()
        params = mock_client._get.call_args[1]["params"]
        assert params["limit"] == 100
        assert params["offset"] == 0

    def test_list_with_filters(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"policies": [], "total": 0}
        api.list(
            workspace_id="ws-1",
            vertical_id="healthcare",
            framework_id="gdpr",
            enabled_only=True,
            limit=10,
            offset=5,
        )
        params = mock_client._get.call_args[1]["params"]
        assert params["workspace_id"] == "ws-1"
        assert params["vertical_id"] == "healthcare"
        assert params["framework_id"] == "gdpr"
        assert params["enabled_only"] is True
        assert params["limit"] == 10
        assert params["offset"] == 5

    def test_list_empty(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"policies": [], "total": 0}
        policies, total = api.list()
        assert policies == []
        assert total == 0

    def test_list_total_fallback(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        """When 'total' is missing from response, fall back to len(policies)."""
        mock_client._get.return_value = {"policies": [SAMPLE_POLICY]}
        policies, total = api.list()
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_async(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"policies": [SAMPLE_POLICY], "total": 1}
        )
        policies, total = await api.list_async()
        assert len(policies) == 1
        assert total == 1
        assert policies[0].id == "pol-123"

    @pytest.mark.asyncio
    async def test_list_async_with_filters(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"policies": [], "total": 0}
        )
        await api.list_async(
            workspace_id="ws-2", framework_id="soc2", enabled_only=True
        )
        params = mock_client._get_async.call_args[1]["params"]
        assert params["workspace_id"] == "ws-2"
        assert params["framework_id"] == "soc2"
        assert params["enabled_only"] is True


class TestPoliciesGet:
    def test_get(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"policy": SAMPLE_POLICY}
        policy = api.get("pol-123")
        assert isinstance(policy, Policy)
        assert policy.id == "pol-123"
        assert policy.framework_id == "gdpr"
        mock_client._get.assert_called_once_with("/api/v1/policies/pol-123")

    def test_get_unwrapped_response(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        """When response has no 'policy' key, use the response dict itself."""
        mock_client._get.return_value = SAMPLE_POLICY
        policy = api.get("pol-123")
        assert policy.id == "pol-123"

    @pytest.mark.asyncio
    async def test_get_async(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"policy": SAMPLE_POLICY})
        policy = await api.get_async("pol-123")
        assert policy.name == "GDPR Policy"


class TestPoliciesCreate:
    def test_create_minimal(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"policy": SAMPLE_POLICY}
        policy = api.create(
            name="GDPR Policy", framework_id="gdpr", vertical_id="healthcare"
        )
        assert isinstance(policy, Policy)
        assert policy.id == "pol-123"
        mock_client._post.assert_called_once()
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "GDPR Policy"
        assert body["framework_id"] == "gdpr"
        assert body["vertical_id"] == "healthcare"
        assert body["workspace_id"] == "default"
        assert body["level"] == "recommended"
        assert body["enabled"] is True

    def test_create_full(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"policy": SAMPLE_POLICY}
        rules = [{"id": "r1", "name": "test", "condition": "always", "severity": "low"}]
        api.create(
            name="SOC2 Policy",
            framework_id="soc2",
            vertical_id="finance",
            description="SOC2 compliance",
            workspace_id="ws-2",
            level="required",
            enabled=False,
            rules=rules,
            metadata={"author": "admin"},
        )
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "SOC2 Policy"
        assert body["description"] == "SOC2 compliance"
        assert body["workspace_id"] == "ws-2"
        assert body["level"] == "required"
        assert body["enabled"] is False
        assert body["rules"] == rules
        assert body["metadata"] == {"author": "admin"}

    def test_create_no_optional_fields(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        """When rules and metadata are None, they should not be in the body."""
        mock_client._post.return_value = {"policy": SAMPLE_POLICY}
        api.create(name="Test", framework_id="f1", vertical_id="v1")
        body = mock_client._post.call_args[0][1]
        assert "rules" not in body
        assert "metadata" not in body

    @pytest.mark.asyncio
    async def test_create_async(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"policy": SAMPLE_POLICY})
        policy = await api.create_async(
            name="GDPR Policy", framework_id="gdpr", vertical_id="healthcare"
        )
        assert policy.id == "pol-123"

    @pytest.mark.asyncio
    async def test_create_async_with_rules(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"policy": SAMPLE_POLICY})
        rules = [{"id": "r1", "name": "check", "condition": "x > 1", "severity": "high"}]
        await api.create_async(
            name="Custom", framework_id="custom", vertical_id="v1", rules=rules
        )
        body = mock_client._post_async.call_args[0][1]
        assert body["rules"] == rules


class TestPoliciesUpdate:
    def test_update_name(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"policy": SAMPLE_POLICY}
        policy = api.update("pol-123", name="Updated Name")
        assert isinstance(policy, Policy)
        mock_client._patch.assert_called_once()
        body = mock_client._patch.call_args[0][1]
        assert body["name"] == "Updated Name"

    def test_update_multiple_fields(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch.return_value = {"policy": SAMPLE_POLICY}
        api.update(
            "pol-123",
            description="New desc",
            level="optional",
            enabled=False,
            metadata={"updated": True},
        )
        body = mock_client._patch.call_args[0][1]
        assert body["description"] == "New desc"
        assert body["level"] == "optional"
        assert body["enabled"] is False
        assert body["metadata"] == {"updated": True}

    def test_update_with_rules(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"policy": SAMPLE_POLICY}
        new_rules = [{"id": "r2", "name": "new rule", "condition": "c", "severity": "low"}]
        api.update("pol-123", rules=new_rules)
        body = mock_client._patch.call_args[0][1]
        assert body["rules"] == new_rules

    def test_update_no_fields(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        """When no fields are provided, body should be empty."""
        mock_client._patch.return_value = {"policy": SAMPLE_POLICY}
        api.update("pol-123")
        body = mock_client._patch.call_args[0][1]
        assert body == {}

    @pytest.mark.asyncio
    async def test_update_async(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._patch_async = AsyncMock(return_value={"policy": SAMPLE_POLICY})
        policy = await api.update_async("pol-123", name="Async Update")
        assert policy.id == "pol-123"
        body = mock_client._patch_async.call_args[0][1]
        assert body["name"] == "Async Update"


class TestPoliciesDelete:
    def test_delete(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = None
        result = api.delete("pol-123")
        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/policies/pol-123")

    @pytest.mark.asyncio
    async def test_delete_async(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._delete_async = AsyncMock(return_value=None)
        result = await api.delete_async("pol-123")
        assert result is True


class TestPoliciesToggle:
    def test_toggle_explicit(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"policy": SAMPLE_POLICY}
        policy = api.toggle("pol-123", enabled=False)
        assert isinstance(policy, Policy)
        body = mock_client._post.call_args[0][1]
        assert body["enabled"] is False
        mock_client._post.assert_called_once()
        url = mock_client._post.call_args[0][0]
        assert url == "/api/v1/policies/pol-123/toggle"

    def test_toggle_implicit(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        """When enabled is None, body should be empty (server toggles)."""
        mock_client._post.return_value = {"policy": SAMPLE_POLICY}
        api.toggle("pol-123")
        body = mock_client._post.call_args[0][1]
        assert body == {}

    @pytest.mark.asyncio
    async def test_toggle_async(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"policy": SAMPLE_POLICY})
        policy = await api.toggle_async("pol-123", enabled=True)
        assert policy.id == "pol-123"
        body = mock_client._post_async.call_args[0][1]
        assert body["enabled"] is True


# =========================================================================
# Violations
# =========================================================================


class TestViolationsList:
    def test_list_violations_default(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"violations": [SAMPLE_VIOLATION], "total": 1}
        violations, total = api.list_violations()
        assert len(violations) == 1
        assert total == 1
        assert isinstance(violations[0], PolicyViolation)
        assert violations[0].id == "viol-456"
        assert violations[0].severity == "critical"
        mock_client._get.assert_called_once()

    def test_list_violations_with_filters(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"violations": [], "total": 0}
        api.list_violations(
            policy_id="pol-123",
            workspace_id="ws-1",
            status="open",
            severity="critical",
            limit=50,
            offset=10,
        )
        params = mock_client._get.call_args[1]["params"]
        assert params["policy_id"] == "pol-123"
        assert params["workspace_id"] == "ws-1"
        assert params["status"] == "open"
        assert params["severity"] == "critical"
        assert params["limit"] == 50
        assert params["offset"] == 10

    def test_list_violations_total_fallback(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"violations": [SAMPLE_VIOLATION]}
        _, total = api.list_violations()
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_violations_async(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"violations": [SAMPLE_VIOLATION], "total": 1}
        )
        violations, total = await api.list_violations_async()
        assert len(violations) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_violations_async_with_filters(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"violations": [], "total": 0}
        )
        await api.list_violations_async(status="resolved", severity="high")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["status"] == "resolved"
        assert params["severity"] == "high"


class TestViolationsGet:
    def test_get_violation(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"violation": SAMPLE_VIOLATION}
        violation = api.get_violation("viol-456")
        assert isinstance(violation, PolicyViolation)
        assert violation.id == "viol-456"
        assert violation.policy_id == "pol-123"
        assert violation.status == "open"
        mock_client._get.assert_called_once_with("/api/v1/compliance/violations/viol-456")

    def test_get_violation_unwrapped(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_VIOLATION
        violation = api.get_violation("viol-456")
        assert violation.id == "viol-456"

    @pytest.mark.asyncio
    async def test_get_violation_async(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"violation": SAMPLE_VIOLATION}
        )
        violation = await api.get_violation_async("viol-456")
        assert violation.rule_name == "No PII exposure"


class TestViolationsUpdate:
    def test_update_violation_status(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        resolved_data = {**SAMPLE_VIOLATION, "status": "resolved"}
        mock_client._patch.return_value = {"violation": resolved_data}
        violation = api.update_violation("viol-456", status="resolved")
        assert isinstance(violation, PolicyViolation)
        body = mock_client._patch.call_args[0][1]
        assert body["status"] == "resolved"
        assert "resolution_notes" not in body

    def test_update_violation_with_notes(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch.return_value = {"violation": SAMPLE_VIOLATION}
        api.update_violation(
            "viol-456", status="false_positive", resolution_notes="Not applicable"
        )
        body = mock_client._patch.call_args[0][1]
        assert body["status"] == "false_positive"
        assert body["resolution_notes"] == "Not applicable"

    @pytest.mark.asyncio
    async def test_update_violation_async(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch_async = AsyncMock(
            return_value={"violation": SAMPLE_VIOLATION}
        )
        violation = await api.update_violation_async(
            "viol-456", status="investigating", resolution_notes="Looking into it"
        )
        assert violation.id == "viol-456"
        body = mock_client._patch_async.call_args[0][1]
        assert body["status"] == "investigating"
        assert body["resolution_notes"] == "Looking into it"


# =========================================================================
# Compliance Checking
# =========================================================================


class TestComplianceCheck:
    def test_check_minimal(self, api: PoliciesAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_CHECK_RESULT
        result = api.check("Some content to check")
        assert isinstance(result, ComplianceCheckResult)
        assert result.compliant is False
        assert result.score == 72.5
        assert len(result.issues) == 1
        body = mock_client._post.call_args[0][1]
        assert body["content"] == "Some content to check"
        assert body["min_severity"] == "low"
        assert body["store_violations"] is False
        assert body["workspace_id"] == "default"

    def test_check_with_options(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_CHECK_RESULT
        api.check(
            content="Check this",
            frameworks=["gdpr", "soc2"],
            min_severity="high",
            store_violations=True,
            workspace_id="ws-1",
        )
        body = mock_client._post.call_args[0][1]
        assert body["frameworks"] == ["gdpr", "soc2"]
        assert body["min_severity"] == "high"
        assert body["store_violations"] is True
        assert body["workspace_id"] == "ws-1"

    def test_check_no_frameworks(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        """When frameworks is None, it should not be in the body."""
        mock_client._post.return_value = SAMPLE_CHECK_RESULT
        api.check("content")
        body = mock_client._post.call_args[0][1]
        assert "frameworks" not in body

    @pytest.mark.asyncio
    async def test_check_async(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_CHECK_RESULT)
        result = await api.check_async("Check this async")
        assert result.compliant is False
        assert result.score == 72.5

    @pytest.mark.asyncio
    async def test_check_async_with_frameworks(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_CHECK_RESULT)
        await api.check_async("content", frameworks=["hipaa"])
        body = mock_client._post_async.call_args[0][1]
        assert body["frameworks"] == ["hipaa"]


class TestComplianceStats:
    def test_get_stats_default(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_STATS
        stats = api.get_stats()
        assert isinstance(stats, ComplianceStats)
        assert stats.policies_total == 10
        assert stats.policies_enabled == 8
        assert stats.policies_disabled == 2
        assert stats.violations_total == 25
        assert stats.violations_open == 5
        assert stats.violations_by_severity == {"critical": 2, "high": 3}
        assert stats.risk_score == 35

    def test_get_stats_with_workspace(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_STATS
        api.get_stats(workspace_id="ws-1")
        params = mock_client._get.call_args[1]["params"]
        assert params["workspace_id"] == "ws-1"

    def test_get_stats_empty(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        stats = api.get_stats()
        assert stats.policies_total == 0
        assert stats.violations_total == 0
        assert stats.risk_score == 0

    @pytest.mark.asyncio
    async def test_get_stats_async(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_STATS)
        stats = await api.get_stats_async()
        assert stats.policies_total == 10
        assert stats.risk_score == 35

    @pytest.mark.asyncio
    async def test_get_stats_async_with_workspace(
        self, api: PoliciesAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_STATS)
        await api.get_stats_async(workspace_id="ws-2")
        params = mock_client._get_async.call_args[1]["params"]
        assert params["workspace_id"] == "ws-2"


# =========================================================================
# Parser / Dataclass Tests
# =========================================================================


class TestParsePolicy:
    def test_parse_datetime_iso(self, api: PoliciesAPI) -> None:
        policy = api._parse_policy(SAMPLE_POLICY)
        assert policy.created_at is not None
        assert policy.created_at.year == 2026
        assert policy.updated_at is not None
        assert policy.updated_at.year == 2026

    def test_parse_missing_datetimes(self, api: PoliciesAPI) -> None:
        data = {
            "id": "p1",
            "name": "Test",
            "description": "",
            "framework_id": "f1",
            "vertical_id": "v1",
        }
        policy = api._parse_policy(data)
        assert policy.created_at is None
        assert policy.updated_at is None

    def test_parse_invalid_datetime(self, api: PoliciesAPI) -> None:
        data = {**SAMPLE_POLICY, "created_at": "not-a-date", "updated_at": "bad"}
        policy = api._parse_policy(data)
        assert policy.created_at is None
        assert policy.updated_at is None

    def test_parse_policy_defaults(self, api: PoliciesAPI) -> None:
        """Missing fields should fall back to defaults."""
        policy = api._parse_policy({})
        assert policy.id == ""
        assert policy.name == ""
        assert policy.workspace_id == "default"
        assert policy.level == "recommended"
        assert policy.enabled is True
        assert policy.rules == []
        assert policy.metadata == {}

    def test_parse_policy_with_rules(self, api: PoliciesAPI) -> None:
        policy = api._parse_policy(SAMPLE_POLICY)
        assert len(policy.rules) == 1
        rule = policy.rules[0]
        assert isinstance(rule, PolicyRule)
        assert rule.id == "rule-1"
        assert rule.severity == "critical"


class TestParseRule:
    def test_parse_rule_full(self, api: PoliciesAPI) -> None:
        rule = api._parse_rule(SAMPLE_RULE)
        assert rule.id == "rule-1"
        assert rule.name == "No PII exposure"
        assert rule.description == "Prevents PII from leaking"
        assert rule.condition == "contains_pii(content)"
        assert rule.severity == "critical"
        assert rule.enabled is True
        assert rule.metadata == {"category": "privacy"}

    def test_parse_rule_defaults(self, api: PoliciesAPI) -> None:
        rule = api._parse_rule({})
        assert rule.id == ""
        assert rule.name == ""
        assert rule.severity == "medium"
        assert rule.enabled is True
        assert rule.metadata == {}


class TestParseViolation:
    def test_parse_violation_full(self, api: PoliciesAPI) -> None:
        violation = api._parse_violation(SAMPLE_VIOLATION)
        assert violation.id == "viol-456"
        assert violation.policy_id == "pol-123"
        assert violation.status == "open"
        assert violation.created_at is not None
        assert violation.created_at.year == 2026
        assert violation.resolved_at is None

    def test_parse_violation_with_resolution(self, api: PoliciesAPI) -> None:
        data = {
            **SAMPLE_VIOLATION,
            "status": "resolved",
            "resolved_at": "2026-01-16T09:00:00Z",
            "resolved_by": "admin",
            "resolution_notes": "Fixed the issue",
        }
        violation = api._parse_violation(data)
        assert violation.resolved_at is not None
        assert violation.resolved_at.year == 2026
        assert violation.resolved_by == "admin"
        assert violation.resolution_notes == "Fixed the issue"

    def test_parse_violation_defaults(self, api: PoliciesAPI) -> None:
        violation = api._parse_violation({})
        assert violation.id == ""
        assert violation.workspace_id == "default"
        assert violation.severity == "medium"
        assert violation.status == "open"
        assert violation.created_at is None
        assert violation.resolved_at is None

    def test_parse_violation_invalid_datetime(self, api: PoliciesAPI) -> None:
        data = {**SAMPLE_VIOLATION, "created_at": "bad-date", "resolved_at": "nope"}
        violation = api._parse_violation(data)
        assert violation.created_at is None
        assert violation.resolved_at is None


class TestParseCheckResult:
    def test_parse_check_result_direct(self, api: PoliciesAPI) -> None:
        result = api._parse_check_result(SAMPLE_CHECK_RESULT)
        assert result.compliant is False
        assert result.score == 72.5
        assert len(result.issues) == 1
        assert result.checked_at is not None
        assert result.checked_at.year == 2026

    def test_parse_check_result_nested(self, api: PoliciesAPI) -> None:
        """When data has a 'result' key, parse from nested dict."""
        data = {
            "result": {"compliant": True, "score": 100.0, "issues": []},
            "checked_at": "2026-01-15T12:00:00Z",
        }
        result = api._parse_check_result(data)
        assert result.compliant is True
        assert result.score == 100.0
        assert result.issues == []

    def test_parse_check_result_defaults(self, api: PoliciesAPI) -> None:
        result = api._parse_check_result({})
        assert result.compliant is True
        assert result.score == 100.0
        assert result.issues == []
        assert result.checked_at is None

    def test_parse_check_result_invalid_datetime(self, api: PoliciesAPI) -> None:
        data = {**SAMPLE_CHECK_RESULT, "checked_at": "invalid"}
        result = api._parse_check_result(data)
        assert result.checked_at is None


class TestParseStats:
    def test_parse_stats_full(self, api: PoliciesAPI) -> None:
        stats = api._parse_stats(SAMPLE_STATS)
        assert stats.policies_total == 10
        assert stats.policies_enabled == 8
        assert stats.policies_disabled == 2
        assert stats.violations_total == 25
        assert stats.violations_open == 5
        assert stats.violations_by_severity == {"critical": 2, "high": 3}
        assert stats.risk_score == 35

    def test_parse_stats_empty(self, api: PoliciesAPI) -> None:
        stats = api._parse_stats({})
        assert stats.policies_total == 0
        assert stats.policies_enabled == 0
        assert stats.policies_disabled == 0
        assert stats.violations_total == 0
        assert stats.violations_open == 0
        assert stats.violations_by_severity == {}
        assert stats.risk_score == 0


class TestDataclasses:
    def test_policy_rule_construction(self) -> None:
        rule = PolicyRule(
            id="r1",
            name="Rule 1",
            description="Test rule",
            condition="x > 0",
            severity="high",
        )
        assert rule.id == "r1"
        assert rule.enabled is True
        assert rule.metadata == {}

    def test_policy_construction(self) -> None:
        policy = Policy(
            id="p1",
            name="Test Policy",
            description="A test",
            framework_id="gdpr",
            workspace_id="ws-1",
            vertical_id="v1",
            level="required",
        )
        assert policy.enabled is True
        assert policy.rules == []
        assert policy.created_at is None
        assert policy.updated_at is None
        assert policy.metadata == {}

    def test_policy_violation_construction(self) -> None:
        violation = PolicyViolation(
            id="v1",
            policy_id="p1",
            rule_id="r1",
            rule_name="Rule 1",
            framework_id="gdpr",
            vertical_id="v1",
            workspace_id="ws-1",
            severity="critical",
            status="open",
            description="Violation found",
            source="debate-1",
        )
        assert violation.created_at is None
        assert violation.resolved_at is None
        assert violation.resolved_by is None
        assert violation.resolution_notes is None
        assert violation.metadata == {}

    def test_compliance_check_result_defaults(self) -> None:
        result = ComplianceCheckResult(compliant=True, score=100.0)
        assert result.issues == []
        assert result.checked_at is None

    def test_compliance_stats_defaults(self) -> None:
        stats = ComplianceStats()
        assert stats.policies_total == 0
        assert stats.policies_enabled == 0
        assert stats.policies_disabled == 0
        assert stats.violations_total == 0
        assert stats.violations_open == 0
        assert stats.violations_by_severity == {}
        assert stats.risk_score == 0
