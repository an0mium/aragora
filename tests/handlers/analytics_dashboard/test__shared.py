"""Comprehensive tests for analytics_dashboard/_shared.py.

Tests the shared constants, RBAC guards, utility functions, and stub responses
used across the analytics dashboard handler suite.

Covers:
- Permission constants (9 constants)
- RBAC import/availability guards
- Metrics import guards
- _run_async helper
- _build_analytics_stub_responses() structure and data integrity
- ANALYTICS_STUB_RESPONSES pre-built dict
- __all__ exports completeness
- Re-exported symbols (BaseHandler, json_response, etc.)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_shared():
    """Import the shared module fresh."""
    import aragora.server.handlers.analytics_dashboard._shared as shared

    return shared


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shared():
    """Return the _shared module."""
    return _import_shared()


# =========================================================================
# 1. PERMISSION CONSTANTS
# =========================================================================


class TestPermissionConstants:
    """Verify all nine permission constant values and naming."""

    def test_perm_analytics_read(self, shared):
        assert shared.PERM_ANALYTICS_READ == "analytics:dashboard:read"

    def test_perm_analytics_write(self, shared):
        assert shared.PERM_ANALYTICS_WRITE == "analytics:dashboard:write"

    def test_perm_analytics_export(self, shared):
        assert shared.PERM_ANALYTICS_EXPORT == "analytics:export"

    def test_perm_analytics_admin(self, shared):
        assert shared.PERM_ANALYTICS_ADMIN == "analytics:admin"

    def test_perm_analytics_cost(self, shared):
        assert shared.PERM_ANALYTICS_COST == "analytics:cost:read"

    def test_perm_analytics_compliance(self, shared):
        assert shared.PERM_ANALYTICS_COMPLIANCE == "analytics:compliance:read"

    def test_perm_analytics_tokens(self, shared):
        assert shared.PERM_ANALYTICS_TOKENS == "analytics:tokens:read"

    def test_perm_analytics_flips(self, shared):
        assert shared.PERM_ANALYTICS_FLIPS == "analytics:flips:read"

    def test_perm_analytics_deliberations(self, shared):
        assert shared.PERM_ANALYTICS_DELIBERATIONS == "analytics:deliberations:read"

    def test_all_permissions_are_strings(self, shared):
        perms = [
            shared.PERM_ANALYTICS_READ,
            shared.PERM_ANALYTICS_WRITE,
            shared.PERM_ANALYTICS_EXPORT,
            shared.PERM_ANALYTICS_ADMIN,
            shared.PERM_ANALYTICS_COST,
            shared.PERM_ANALYTICS_COMPLIANCE,
            shared.PERM_ANALYTICS_TOKENS,
            shared.PERM_ANALYTICS_FLIPS,
            shared.PERM_ANALYTICS_DELIBERATIONS,
        ]
        for perm in perms:
            assert isinstance(perm, str), f"Permission {perm!r} is not a string"

    def test_all_permissions_use_colon_separator(self, shared):
        perms = [
            shared.PERM_ANALYTICS_READ,
            shared.PERM_ANALYTICS_WRITE,
            shared.PERM_ANALYTICS_EXPORT,
            shared.PERM_ANALYTICS_ADMIN,
            shared.PERM_ANALYTICS_COST,
            shared.PERM_ANALYTICS_COMPLIANCE,
            shared.PERM_ANALYTICS_TOKENS,
            shared.PERM_ANALYTICS_FLIPS,
            shared.PERM_ANALYTICS_DELIBERATIONS,
        ]
        for perm in perms:
            assert ":" in perm, f"Permission {perm!r} missing colon separator"

    def test_all_permissions_start_with_analytics(self, shared):
        perms = [
            shared.PERM_ANALYTICS_READ,
            shared.PERM_ANALYTICS_WRITE,
            shared.PERM_ANALYTICS_EXPORT,
            shared.PERM_ANALYTICS_ADMIN,
            shared.PERM_ANALYTICS_COST,
            shared.PERM_ANALYTICS_COMPLIANCE,
            shared.PERM_ANALYTICS_TOKENS,
            shared.PERM_ANALYTICS_FLIPS,
            shared.PERM_ANALYTICS_DELIBERATIONS,
        ]
        for perm in perms:
            assert perm.startswith("analytics:"), (
                f"Permission {perm!r} doesn't start with 'analytics:'"
            )

    def test_permissions_are_unique(self, shared):
        perms = [
            shared.PERM_ANALYTICS_READ,
            shared.PERM_ANALYTICS_WRITE,
            shared.PERM_ANALYTICS_EXPORT,
            shared.PERM_ANALYTICS_ADMIN,
            shared.PERM_ANALYTICS_COST,
            shared.PERM_ANALYTICS_COMPLIANCE,
            shared.PERM_ANALYTICS_TOKENS,
            shared.PERM_ANALYTICS_FLIPS,
            shared.PERM_ANALYTICS_DELIBERATIONS,
        ]
        assert len(perms) == len(set(perms)), "Duplicate permission values found"

    def test_exactly_nine_permissions_defined(self, shared):
        perm_names = [k for k in dir(shared) if k.startswith("PERM_ANALYTICS_")]
        assert len(perm_names) == 9


# =========================================================================
# 2. RBAC AVAILABILITY
# =========================================================================


class TestRBACAvailability:
    """Verify RBAC imports and availability flags."""

    def test_rbac_available_is_bool(self, shared):
        assert isinstance(shared.RBAC_AVAILABLE, bool)

    def test_rbac_available_true_when_imports_succeed(self, shared):
        # In test env the RBAC module is importable
        assert shared.RBAC_AVAILABLE is True

    def test_authorization_context_importable(self, shared):
        assert shared.AuthorizationContext is not None

    def test_check_permission_importable(self, shared):
        assert shared.check_permission is not None

    def test_permission_denied_error_importable(self, shared):
        assert shared.PermissionDeniedError is not None

    def test_extract_user_from_request_importable(self, shared):
        assert shared.extract_user_from_request is not None

    def test_rbac_fail_closed_callable(self, shared):
        assert callable(shared.rbac_fail_closed)

    def test_rbac_fail_closed_returns_bool(self, shared):
        result = shared.rbac_fail_closed()
        assert isinstance(result, bool)


# =========================================================================
# 3. METRICS AVAILABILITY
# =========================================================================


class TestMetricsAvailability:
    """Verify metrics imports and fallback."""

    def test_metrics_available_is_bool(self, shared):
        assert isinstance(shared.METRICS_AVAILABLE, bool)

    def test_record_rbac_check_callable(self, shared):
        assert callable(shared.record_rbac_check)

    def test_record_rbac_check_accepts_args(self, shared):
        """record_rbac_check should accept (permission, granted) args."""
        # When METRICS_AVAILABLE is True, this is record_rbac_decision(permission, granted)
        # When False, it is a no-op that accepts *args, **kwargs.
        # Either way, calling with the real signature should not raise.
        shared.record_rbac_check("some_permission", True)

    def test_record_rbac_check_with_kwargs(self, shared):
        """record_rbac_check should accept keyword arguments."""
        shared.record_rbac_check(permission="read", granted=True)


# =========================================================================
# 4. _run_async HELPER
# =========================================================================


class TestRunAsync:
    """Test the _run_async wrapper function."""

    def test_run_async_executes_coroutine(self, shared):
        async def _coro():
            return 42

        result = shared._run_async(_coro())
        assert result == 42

    def test_run_async_returns_string(self, shared):
        async def _coro():
            return "hello"

        result = shared._run_async(_coro())
        assert result == "hello"

    def test_run_async_returns_dict(self, shared):
        async def _coro():
            return {"key": "value"}

        result = shared._run_async(_coro())
        assert result == {"key": "value"}

    def test_run_async_returns_none(self, shared):
        async def _coro():
            return None

        result = shared._run_async(_coro())
        assert result is None

    def test_run_async_returns_list(self, shared):
        async def _coro():
            return [1, 2, 3]

        result = shared._run_async(_coro())
        assert result == [1, 2, 3]

    def test_run_async_propagates_exception(self, shared):
        async def _coro():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            shared._run_async(_coro())


# =========================================================================
# 5. _build_analytics_stub_responses
# =========================================================================


class TestBuildAnalyticsStubResponses:
    """Test the stub response builder function."""

    def test_returns_dict(self, shared):
        result = shared._build_analytics_stub_responses()
        assert isinstance(result, dict)

    def test_all_keys_are_api_paths(self, shared):
        result = shared._build_analytics_stub_responses()
        for key in result:
            assert key.startswith("/api/analytics/"), f"Key {key!r} is not an analytics API path"

    def test_all_values_are_dicts(self, shared):
        result = shared._build_analytics_stub_responses()
        for key, val in result.items():
            assert isinstance(val, dict), f"Value for {key!r} is not a dict"

    def test_returns_fresh_copy_each_call(self, shared):
        """Each call should return an independent dict."""
        a = shared._build_analytics_stub_responses()
        b = shared._build_analytics_stub_responses()
        assert a is not b
        assert a == b

    def test_mutating_result_does_not_affect_next_call(self, shared):
        a = shared._build_analytics_stub_responses()
        a["/api/analytics/summary"]["summary"]["total_debates"] = 999
        b = shared._build_analytics_stub_responses()
        assert b["/api/analytics/summary"]["summary"]["total_debates"] == 47


# =========================================================================
# 6. ANALYTICS_STUB_RESPONSES — Structure
# =========================================================================


class TestStubResponsesStructure:
    """Test that the pre-built ANALYTICS_STUB_RESPONSES is well-formed."""

    def test_is_dict(self, shared):
        assert isinstance(shared.ANALYTICS_STUB_RESPONSES, dict)

    def test_has_expected_endpoint_count(self, shared):
        # 19 stub endpoint keys
        assert len(shared.ANALYTICS_STUB_RESPONSES) == 19

    def test_summary_endpoint_present(self, shared):
        assert "/api/analytics/summary" in shared.ANALYTICS_STUB_RESPONSES

    def test_trends_findings_endpoint_present(self, shared):
        assert "/api/analytics/trends/findings" in shared.ANALYTICS_STUB_RESPONSES

    def test_remediation_endpoint_present(self, shared):
        assert "/api/analytics/remediation" in shared.ANALYTICS_STUB_RESPONSES

    def test_agents_endpoint_present(self, shared):
        assert "/api/analytics/agents" in shared.ANALYTICS_STUB_RESPONSES

    def test_cost_endpoint_present(self, shared):
        assert "/api/analytics/cost" in shared.ANALYTICS_STUB_RESPONSES

    def test_cost_breakdown_endpoint_present(self, shared):
        assert "/api/analytics/cost/breakdown" in shared.ANALYTICS_STUB_RESPONSES

    def test_compliance_endpoint_present(self, shared):
        assert "/api/analytics/compliance" in shared.ANALYTICS_STUB_RESPONSES

    def test_heatmap_endpoint_present(self, shared):
        assert "/api/analytics/heatmap" in shared.ANALYTICS_STUB_RESPONSES

    def test_tokens_endpoint_present(self, shared):
        assert "/api/analytics/tokens" in shared.ANALYTICS_STUB_RESPONSES

    def test_tokens_trends_endpoint_present(self, shared):
        assert "/api/analytics/tokens/trends" in shared.ANALYTICS_STUB_RESPONSES

    def test_tokens_providers_endpoint_present(self, shared):
        assert "/api/analytics/tokens/providers" in shared.ANALYTICS_STUB_RESPONSES

    def test_flips_summary_endpoint_present(self, shared):
        assert "/api/analytics/flips/summary" in shared.ANALYTICS_STUB_RESPONSES

    def test_flips_recent_endpoint_present(self, shared):
        assert "/api/analytics/flips/recent" in shared.ANALYTICS_STUB_RESPONSES

    def test_flips_consistency_endpoint_present(self, shared):
        assert "/api/analytics/flips/consistency" in shared.ANALYTICS_STUB_RESPONSES

    def test_flips_trends_endpoint_present(self, shared):
        assert "/api/analytics/flips/trends" in shared.ANALYTICS_STUB_RESPONSES

    def test_deliberations_endpoint_present(self, shared):
        assert "/api/analytics/deliberations" in shared.ANALYTICS_STUB_RESPONSES

    def test_deliberations_channels_endpoint_present(self, shared):
        assert "/api/analytics/deliberations/channels" in shared.ANALYTICS_STUB_RESPONSES

    def test_deliberations_consensus_endpoint_present(self, shared):
        assert "/api/analytics/deliberations/consensus" in shared.ANALYTICS_STUB_RESPONSES

    def test_deliberations_performance_endpoint_present(self, shared):
        assert "/api/analytics/deliberations/performance" in shared.ANALYTICS_STUB_RESPONSES

    def test_all_responses_are_json_serializable(self, shared):
        for key, val in shared.ANALYTICS_STUB_RESPONSES.items():
            try:
                json.dumps(val)
            except (TypeError, ValueError) as exc:
                pytest.fail(f"Response for {key!r} is not JSON-serializable: {exc}")


# =========================================================================
# 7. STUB RESPONSE DATA — Summary
# =========================================================================


class TestStubSummary:
    """Validate /api/analytics/summary stub data."""

    def test_summary_has_summary_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]
        assert "summary" in data

    def test_summary_total_debates(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"]
        assert s["total_debates"] == 47

    def test_summary_total_messages(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"]
        assert s["total_messages"] == 312

    def test_summary_consensus_rate(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"]
        assert s["consensus_rate"] == 72.3

    def test_summary_avg_duration(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"]
        assert s["avg_debate_duration_ms"] == 45200

    def test_summary_active_users(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"]
        assert s["active_users_24h"] == 3

    def test_summary_top_topics_is_list(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"]
        assert isinstance(s["top_topics"], list)

    def test_summary_top_topics_count(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"]
        assert len(s["top_topics"]) == 4

    def test_summary_top_topics_have_required_keys(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"]
        for topic in s["top_topics"]:
            assert "topic" in topic
            assert "count" in topic


# =========================================================================
# 8. STUB RESPONSE DATA — Trends/Findings
# =========================================================================


class TestStubTrendsFindings:
    """Validate /api/analytics/trends/findings stub data."""

    def test_has_trends_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/trends/findings"]
        assert "trends" in data

    def test_trends_is_list(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/trends/findings"]
        assert isinstance(data["trends"], list)

    def test_trends_has_five_entries(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/trends/findings"]
        assert len(data["trends"]) == 5

    def test_trends_entries_have_date(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/trends/findings"]
        for entry in data["trends"]:
            assert "date" in entry

    def test_trends_entries_have_findings(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/trends/findings"]
        for entry in data["trends"]:
            assert "findings" in entry
            assert isinstance(entry["findings"], int)

    def test_trends_entries_have_resolved(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/trends/findings"]
        for entry in data["trends"]:
            assert "resolved" in entry
            assert isinstance(entry["resolved"], int)


# =========================================================================
# 9. STUB RESPONSE DATA — Remediation
# =========================================================================


class TestStubRemediation:
    """Validate /api/analytics/remediation stub data."""

    def test_has_metrics_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/remediation"]
        assert "metrics" in data

    def test_remediation_total_findings(self, shared):
        m = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/remediation"]["metrics"]
        assert m["total_findings"] == 21

    def test_remediation_remediated(self, shared):
        m = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/remediation"]["metrics"]
        assert m["remediated"] == 18

    def test_remediation_pending(self, shared):
        m = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/remediation"]["metrics"]
        assert m["pending"] == 3

    def test_remediation_rate(self, shared):
        m = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/remediation"]["metrics"]
        assert m["remediation_rate"] == 85.7

    def test_remediation_avg_time(self, shared):
        m = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/remediation"]["metrics"]
        assert m["avg_remediation_time_hours"] == 2.4


# =========================================================================
# 10. STUB RESPONSE DATA — Agents
# =========================================================================


class TestStubAgents:
    """Validate /api/analytics/agents stub data."""

    def test_has_agents_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]
        assert "agents" in data

    def test_agents_is_list(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]
        assert isinstance(data["agents"], list)

    def test_agents_has_five_entries(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]
        assert len(data["agents"]) == 5

    def test_agents_entries_have_required_keys(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]
        for agent in data["agents"]:
            assert "agent_id" in agent
            assert "name" in agent
            assert "debates" in agent
            assert "win_rate" in agent
            assert "elo" in agent

    def test_agents_win_rates_between_0_and_1(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]
        for agent in data["agents"]:
            assert 0.0 <= agent["win_rate"] <= 1.0, (
                f"Win rate {agent['win_rate']} for {agent['agent_id']} out of range"
            )

    def test_agents_elo_positive(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]
        for agent in data["agents"]:
            assert agent["elo"] > 0

    def test_agents_debates_positive(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]
        for agent in data["agents"]:
            assert agent["debates"] > 0

    def test_first_agent_is_claude_opus(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]
        assert data["agents"][0]["agent_id"] == "claude-opus"


# =========================================================================
# 11. STUB RESPONSE DATA — Cost
# =========================================================================


class TestStubCost:
    """Validate /api/analytics/cost stub data."""

    def test_has_analysis_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]
        assert "analysis" in data

    def test_total_cost(self, shared):
        a = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]["analysis"]
        assert a["total_cost_usd"] == 12.47

    def test_cost_by_model_is_dict(self, shared):
        a = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]["analysis"]
        assert isinstance(a["cost_by_model"], dict)

    def test_cost_by_model_five_models(self, shared):
        a = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]["analysis"]
        assert len(a["cost_by_model"]) == 5

    def test_cost_by_debate_type_is_dict(self, shared):
        a = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]["analysis"]
        assert isinstance(a["cost_by_debate_type"], dict)

    def test_projected_monthly_cost(self, shared):
        a = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]["analysis"]
        assert a["projected_monthly_cost"] == 18.70

    def test_cost_trend_is_list(self, shared):
        a = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]["analysis"]
        assert isinstance(a["cost_trend"], list)
        assert len(a["cost_trend"]) == 5

    def test_cost_trend_entries_have_date_and_cost(self, shared):
        a = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]["analysis"]
        for entry in a["cost_trend"]:
            assert "date" in entry
            assert "cost_usd" in entry


# =========================================================================
# 12. STUB RESPONSE DATA — Cost Breakdown
# =========================================================================


class TestStubCostBreakdown:
    """Validate /api/analytics/cost/breakdown stub data."""

    def test_has_breakdown_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost/breakdown"]
        assert "breakdown" in data

    def test_total_spend(self, shared):
        b = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost/breakdown"]["breakdown"]
        assert b["total_spend_usd"] == 12.47

    def test_agents_is_list(self, shared):
        b = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost/breakdown"]["breakdown"]
        assert isinstance(b["agents"], list)

    def test_agents_count(self, shared):
        b = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost/breakdown"]["breakdown"]
        assert len(b["agents"]) == 5

    def test_agents_have_required_keys(self, shared):
        b = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost/breakdown"]["breakdown"]
        for agent in b["agents"]:
            assert "agent" in agent
            assert "spend_usd" in agent
            assert "debates" in agent

    def test_budget_utilization(self, shared):
        b = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost/breakdown"]["breakdown"]
        assert b["budget_utilization_pct"] == 62.4


# =========================================================================
# 13. STUB RESPONSE DATA — Compliance
# =========================================================================


class TestStubCompliance:
    """Validate /api/analytics/compliance stub data."""

    def test_has_compliance_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/compliance"]
        assert "compliance" in data

    def test_overall_score(self, shared):
        c = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/compliance"]["compliance"]
        assert c["overall_score"] == 94

    def test_categories_is_list(self, shared):
        c = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/compliance"]["compliance"]
        assert isinstance(c["categories"], list)

    def test_categories_count(self, shared):
        c = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/compliance"]["compliance"]
        assert len(c["categories"]) == 4

    def test_categories_have_required_keys(self, shared):
        c = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/compliance"]["compliance"]
        for cat in c["categories"]:
            assert "name" in cat
            assert "score" in cat
            assert "status" in cat

    def test_all_categories_pass(self, shared):
        c = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/compliance"]["compliance"]
        for cat in c["categories"]:
            assert cat["status"] == "pass"

    def test_last_audit_present(self, shared):
        c = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/compliance"]["compliance"]
        assert "last_audit" in c
        assert "2026" in c["last_audit"]


# =========================================================================
# 14. STUB RESPONSE DATA — Heatmap
# =========================================================================


class TestStubHeatmap:
    """Validate /api/analytics/heatmap stub data."""

    def test_has_heatmap_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/heatmap"]
        assert "heatmap" in data

    def test_x_labels_are_weekdays(self, shared):
        h = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/heatmap"]["heatmap"]
        assert h["x_labels"] == ["Mon", "Tue", "Wed", "Thu", "Fri"]

    def test_y_labels_are_times(self, shared):
        h = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/heatmap"]["heatmap"]
        assert h["y_labels"] == ["9AM", "12PM", "3PM", "6PM"]

    def test_values_is_2d_grid(self, shared):
        h = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/heatmap"]["heatmap"]
        assert isinstance(h["values"], list)
        assert len(h["values"]) == 4  # 4 time slots
        for row in h["values"]:
            assert len(row) == 5  # 5 weekdays

    def test_max_value(self, shared):
        h = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/heatmap"]["heatmap"]
        assert h["max_value"] == 8

    def test_max_value_consistent_with_grid(self, shared):
        h = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/heatmap"]["heatmap"]
        actual_max = max(max(row) for row in h["values"])
        assert actual_max == h["max_value"]


# =========================================================================
# 15. STUB RESPONSE DATA — Tokens
# =========================================================================


class TestStubTokens:
    """Validate /api/analytics/tokens stub data."""

    def test_has_summary_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]
        assert "summary" in data

    def test_has_by_agent_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]
        assert "by_agent" in data

    def test_has_by_model_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]
        assert "by_model" in data

    def test_total_tokens_sum(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]["summary"]
        assert s["total_tokens"] == s["total_tokens_in"] + s["total_tokens_out"]

    def test_total_tokens_in(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]["summary"]
        assert s["total_tokens_in"] == 284500

    def test_total_tokens_out(self, shared):
        s = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]["summary"]
        assert s["total_tokens_out"] == 142300

    def test_by_agent_five_agents(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]
        assert len(data["by_agent"]) == 5

    def test_by_model_five_models(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]
        assert len(data["by_model"]) == 5


# =========================================================================
# 16. STUB RESPONSE DATA — Token Trends
# =========================================================================


class TestStubTokenTrends:
    """Validate /api/analytics/tokens/trends stub data."""

    def test_has_trends_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens/trends"]
        assert "trends" in data

    def test_trends_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens/trends"]
        assert len(data["trends"]) == 5

    def test_trends_entries_have_required_keys(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens/trends"]
        for entry in data["trends"]:
            assert "date" in entry
            assert "tokens_in" in entry
            assert "tokens_out" in entry


# =========================================================================
# 17. STUB RESPONSE DATA — Token Providers
# =========================================================================


class TestStubTokenProviders:
    """Validate /api/analytics/tokens/providers stub data."""

    def test_has_providers_key(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens/providers"]
        assert "providers" in data

    def test_providers_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens/providers"]
        assert len(data["providers"]) == 4

    def test_providers_have_required_keys(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens/providers"]
        for p in data["providers"]:
            assert "provider" in p
            assert "tokens" in p
            assert "pct" in p

    def test_percentages_sum_approximately_100(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens/providers"]
        total = sum(p["pct"] for p in data["providers"])
        assert 99.0 <= total <= 101.0, f"Percentages sum to {total}, not ~100"


# =========================================================================
# 18. STUB RESPONSE DATA — Flips
# =========================================================================


class TestStubFlips:
    """Validate /api/analytics/flips/* stub data."""

    def test_flips_summary_total(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/summary"]
        assert data["summary"]["total"] == 14

    def test_flips_summary_consistent_and_inconsistent(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/summary"]
        s = data["summary"]
        assert s["consistent"] + s["inconsistent"] == s["total"]

    def test_flips_recent_is_list(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/recent"]
        assert isinstance(data["flips"], list)

    def test_flips_recent_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/recent"]
        assert len(data["flips"]) == 3

    def test_flips_recent_entries_have_required_keys(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/recent"]
        for flip in data["flips"]:
            assert "agent" in flip
            assert "topic" in flip
            assert "from" in flip
            assert "to" in flip
            assert "date" in flip

    def test_flips_consistency_is_list(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/consistency"]
        assert isinstance(data["consistency"], list)

    def test_flips_consistency_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/consistency"]
        assert len(data["consistency"]) == 5

    def test_flips_consistency_scores_between_0_and_1(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/consistency"]
        for entry in data["consistency"]:
            assert 0.0 <= entry["consistency_score"] <= 1.0

    def test_flips_trends_is_list(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/trends"]
        assert isinstance(data["trends"], list)

    def test_flips_trends_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/trends"]
        assert len(data["trends"]) == 5


# =========================================================================
# 19. STUB RESPONSE DATA — Deliberations
# =========================================================================


class TestStubDeliberations:
    """Validate /api/analytics/deliberations/* stub data."""

    def test_deliberations_summary(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations"]
        assert data["summary"]["total"] == 47
        assert data["summary"]["consensus_rate"] == 72.3

    def test_deliberations_channels_is_list(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/channels"]
        assert isinstance(data["channels"], list)

    def test_deliberations_channels_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/channels"]
        assert len(data["channels"]) == 3

    def test_deliberations_channels_have_required_keys(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/channels"]
        for ch in data["channels"]:
            assert "channel" in ch
            assert "count" in ch
            assert "consensus_rate" in ch

    def test_deliberations_consensus_is_list(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/consensus"]
        assert isinstance(data["consensus"], list)

    def test_deliberations_consensus_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/consensus"]
        assert len(data["consensus"]) == 3

    def test_deliberations_consensus_have_required_keys(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/consensus"]
        for entry in data["consensus"]:
            assert "method" in entry
            assert "count" in entry
            assert "avg_rounds" in entry

    def test_deliberations_performance_is_list(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/performance"]
        assert isinstance(data["performance"], list)

    def test_deliberations_performance_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/performance"]
        assert len(data["performance"]) == 4

    def test_deliberations_performance_have_required_keys(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations/performance"]
        for entry in data["performance"]:
            assert "metric" in entry
            assert "value" in entry


# =========================================================================
# 20. __all__ EXPORTS
# =========================================================================


class TestAllExports:
    """Verify __all__ is complete and consistent."""

    def test_all_is_list(self, shared):
        assert isinstance(shared.__all__, list)

    def test_all_contains_permission_constants(self, shared):
        expected = [
            "PERM_ANALYTICS_READ",
            "PERM_ANALYTICS_WRITE",
            "PERM_ANALYTICS_EXPORT",
            "PERM_ANALYTICS_ADMIN",
            "PERM_ANALYTICS_COST",
            "PERM_ANALYTICS_COMPLIANCE",
            "PERM_ANALYTICS_TOKENS",
            "PERM_ANALYTICS_FLIPS",
            "PERM_ANALYTICS_DELIBERATIONS",
        ]
        for name in expected:
            assert name in shared.__all__, f"{name} missing from __all__"

    def test_all_contains_rbac_symbols(self, shared):
        expected = [
            "RBAC_AVAILABLE",
            "AuthorizationContext",
            "check_permission",
            "PermissionDeniedError",
            "extract_user_from_request",
            "rbac_fail_closed",
        ]
        for name in expected:
            assert name in shared.__all__, f"{name} missing from __all__"

    def test_all_contains_metrics_symbols(self, shared):
        assert "METRICS_AVAILABLE" in shared.__all__
        assert "record_rbac_check" in shared.__all__

    def test_all_contains_base_handler_exports(self, shared):
        expected = [
            "BaseHandler",
            "HandlerResult",
            "error_response",
            "json_response",
            "handle_errors",
            "rate_limit",
            "require_permission",
            "require_user_auth",
            "get_clamped_int_param",
            "get_string_param",
        ]
        for name in expected:
            assert name in shared.__all__, f"{name} missing from __all__"

    def test_all_contains_utility_symbols(self, shared):
        expected = [
            "_run_async",
            "ANALYTICS_STUB_RESPONSES",
            "logger",
            "safe_error_message",
            "strip_version_prefix",
            "cached_analytics",
            "cached_analytics_org",
        ]
        for name in expected:
            assert name in shared.__all__, f"{name} missing from __all__"

    def test_all_entries_resolve_to_attributes(self, shared):
        for name in shared.__all__:
            assert hasattr(shared, name), (
                f"{name} listed in __all__ but not an attribute of the module"
            )

    def test_no_duplicates_in_all(self, shared):
        assert len(shared.__all__) == len(set(shared.__all__)), "Duplicates in __all__"


# =========================================================================
# 21. RE-EXPORTED SYMBOLS
# =========================================================================


class TestReExportedSymbols:
    """Verify that re-exported symbols are properly importable."""

    def test_base_handler_is_class(self, shared):
        assert isinstance(shared.BaseHandler, type)

    def test_handler_result_is_class(self, shared):
        assert isinstance(shared.HandlerResult, type)

    def test_json_response_callable(self, shared):
        assert callable(shared.json_response)

    def test_error_response_callable(self, shared):
        assert callable(shared.error_response)

    def test_handle_errors_callable(self, shared):
        assert callable(shared.handle_errors)

    def test_rate_limit_callable(self, shared):
        assert callable(shared.rate_limit)

    def test_require_permission_callable(self, shared):
        assert callable(shared.require_permission)

    def test_require_user_auth_callable(self, shared):
        assert callable(shared.require_user_auth)

    def test_get_clamped_int_param_callable(self, shared):
        assert callable(shared.get_clamped_int_param)

    def test_get_string_param_callable(self, shared):
        assert callable(shared.get_string_param)

    def test_safe_error_message_callable(self, shared):
        assert callable(shared.safe_error_message)

    def test_strip_version_prefix_callable(self, shared):
        assert callable(shared.strip_version_prefix)

    def test_cached_analytics_callable(self, shared):
        assert callable(shared.cached_analytics)

    def test_cached_analytics_org_callable(self, shared):
        assert callable(shared.cached_analytics_org)

    def test_logger_is_logger_instance(self, shared):
        import logging

        assert isinstance(shared.logger, logging.Logger)

    def test_logger_name(self, shared):
        assert shared.logger.name == "aragora.server.handlers.analytics_dashboard._shared"


# =========================================================================
# 22. DATA CONSISTENCY ACROSS ENDPOINTS
# =========================================================================


class TestCrossEndpointConsistency:
    """Verify data consistency between related stub responses."""

    def test_cost_total_matches_breakdown_total(self, shared):
        cost_total = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost"]["analysis"][
            "total_cost_usd"
        ]
        breakdown_total = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost/breakdown"][
            "breakdown"
        ]["total_spend_usd"]
        assert cost_total == breakdown_total

    def test_agent_count_consistent_across_endpoints(self, shared):
        agents_count = len(shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]["agents"])
        breakdown_count = len(
            shared.ANALYTICS_STUB_RESPONSES["/api/analytics/cost/breakdown"]["breakdown"]["agents"]
        )
        assert agents_count == breakdown_count

    def test_token_agents_and_models_same_count(self, shared):
        data = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/tokens"]
        assert len(data["by_agent"]) == len(data["by_model"])

    def test_deliberation_total_consistent(self, shared):
        summary_total = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations"]["summary"][
            "total"
        ]
        # The summary total debates and deliberations total should be the same
        main_total = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"][
            "total_debates"
        ]
        assert summary_total == main_total

    def test_flips_consistency_agent_count_matches_agents(self, shared):
        flip_agents = len(
            shared.ANALYTICS_STUB_RESPONSES["/api/analytics/flips/consistency"]["consistency"]
        )
        all_agents = len(shared.ANALYTICS_STUB_RESPONSES["/api/analytics/agents"]["agents"])
        assert flip_agents == all_agents

    def test_consensus_rate_consistent(self, shared):
        summary_rate = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/summary"]["summary"][
            "consensus_rate"
        ]
        deliberation_rate = shared.ANALYTICS_STUB_RESPONSES["/api/analytics/deliberations"][
            "summary"
        ]["consensus_rate"]
        assert summary_rate == deliberation_rate
