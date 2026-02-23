"""Comprehensive tests for cost visibility helper functions.

Tests cover aragora/server/handlers/costs/helpers.py (198 lines):

  TestGenerateMockSummary         - _generate_mock_summary() for all time ranges
  TestGenerateMockSummaryContent  - Verifies structure, providers, features, alerts
  TestBuildExportRows             - _build_export_rows() for provider/feature/daily/default
  TestExportCsvResponse           - _export_csv_response() CSV generation and headers
  TestGetImplementationSteps      - _get_implementation_steps() known + unknown types
  TestGetImplementationDifficulty - _get_implementation_difficulty() known + unknown types
  TestGetImplementationTime       - _get_implementation_time() known + unknown types
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from aragora.server.handlers.costs.helpers import (
    _build_export_rows,
    _export_csv_response,
    _generate_mock_summary,
    _get_implementation_difficulty,
    _get_implementation_steps,
    _get_implementation_time,
)
from aragora.server.handlers.costs.models import CostSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now():
    return datetime.now(timezone.utc)


def _make_summary(**overrides) -> CostSummary:
    """Create a CostSummary with sensible defaults."""
    now = _now()
    defaults = dict(
        total_cost=125.50,
        budget=500.00,
        tokens_used=3_125_000,
        api_calls=12_550,
        last_updated=now,
        cost_by_provider=[
            {"name": "Anthropic", "cost": 77.31, "percentage": 61.6},
            {"name": "OpenAI", "cost": 34.64, "percentage": 27.6},
        ],
        cost_by_feature=[
            {"name": "Debates", "cost": 54.22, "percentage": 43.2},
            {"name": "Code Review", "cost": 22.46, "percentage": 17.9},
        ],
        daily_costs=[
            {"date": "2026-02-20", "cost": 18.50, "tokens": 462500},
            {"date": "2026-02-21", "cost": 21.00, "tokens": 525000},
            {"date": "2026-02-22", "cost": 16.00, "tokens": 400000},
        ],
        alerts=[
            {
                "id": "1",
                "type": "budget_warning",
                "message": "80% usage",
                "severity": "warning",
                "timestamp": now.isoformat(),
            }
        ],
    )
    defaults.update(overrides)
    return CostSummary(**defaults)


# ===========================================================================
# TestGenerateMockSummary -- time range dispatch and data shape
# ===========================================================================


class TestGenerateMockSummary:
    """Tests for _generate_mock_summary()."""

    def test_returns_cost_summary_instance(self):
        result = _generate_mock_summary("7d")
        assert isinstance(result, CostSummary)

    def test_24h_produces_1_day(self):
        result = _generate_mock_summary("24h")
        assert len(result.daily_costs) == 1

    def test_7d_produces_7_days(self):
        result = _generate_mock_summary("7d")
        assert len(result.daily_costs) == 7

    def test_30d_produces_30_days(self):
        result = _generate_mock_summary("30d")
        assert len(result.daily_costs) == 30

    def test_90d_produces_90_days(self):
        result = _generate_mock_summary("90d")
        assert len(result.daily_costs) == 90

    def test_unknown_range_defaults_to_7_days(self):
        result = _generate_mock_summary("unknown")
        assert len(result.daily_costs) == 7

    def test_empty_string_defaults_to_7_days(self):
        result = _generate_mock_summary("")
        assert len(result.daily_costs) == 7

    def test_budget_is_500(self):
        result = _generate_mock_summary("7d")
        assert result.budget == 500.00

    def test_total_cost_positive(self):
        result = _generate_mock_summary("7d")
        assert result.total_cost > 0

    def test_total_tokens_positive(self):
        result = _generate_mock_summary("7d")
        assert result.tokens_used > 0

    def test_api_calls_positive(self):
        result = _generate_mock_summary("7d")
        assert result.api_calls > 0

    def test_total_cost_equals_sum_of_daily_costs(self):
        result = _generate_mock_summary("7d")
        daily_sum = sum(entry["cost"] for entry in result.daily_costs)
        # total_cost is rounded, so allow small rounding tolerance
        assert abs(result.total_cost - round(daily_sum, 2)) < 0.02

    def test_total_tokens_equals_sum_of_daily_tokens(self):
        result = _generate_mock_summary("7d")
        daily_sum = sum(entry["tokens"] for entry in result.daily_costs)
        assert result.tokens_used == daily_sum

    def test_api_calls_derived_from_total_cost(self):
        result = _generate_mock_summary("7d")
        assert result.api_calls == int(result.total_cost * 100)


class TestGenerateMockSummaryContent:
    """Tests for the content structure of _generate_mock_summary() results."""

    def test_has_four_providers(self):
        result = _generate_mock_summary("7d")
        assert len(result.cost_by_provider) == 4

    def test_provider_names(self):
        result = _generate_mock_summary("7d")
        names = [p["name"] for p in result.cost_by_provider]
        assert names == ["Anthropic", "OpenAI", "Mistral", "OpenRouter"]

    def test_provider_percentages_sum_to_100(self):
        result = _generate_mock_summary("7d")
        total_pct = sum(p["percentage"] for p in result.cost_by_provider)
        assert abs(total_pct - 100.0) < 0.01

    def test_provider_costs_are_positive(self):
        result = _generate_mock_summary("7d")
        for p in result.cost_by_provider:
            assert p["cost"] > 0

    def test_has_four_features(self):
        result = _generate_mock_summary("7d")
        assert len(result.cost_by_feature) == 4

    def test_feature_names(self):
        result = _generate_mock_summary("7d")
        names = [f["name"] for f in result.cost_by_feature]
        assert names == ["Debates", "Email Triage", "Code Review", "Knowledge Work"]

    def test_feature_percentages_sum_to_100(self):
        result = _generate_mock_summary("7d")
        total_pct = sum(f["percentage"] for f in result.cost_by_feature)
        assert abs(total_pct - 100.0) < 0.01

    def test_feature_costs_are_positive(self):
        result = _generate_mock_summary("7d")
        for f in result.cost_by_feature:
            assert f["cost"] > 0

    def test_has_two_alerts(self):
        result = _generate_mock_summary("7d")
        assert len(result.alerts) == 2

    def test_alert_types(self):
        result = _generate_mock_summary("7d")
        types = [a["type"] for a in result.alerts]
        assert "budget_warning" in types
        assert "spike_detected" in types

    def test_alert_severities(self):
        result = _generate_mock_summary("7d")
        severities = {a["severity"] for a in result.alerts}
        assert severities == {"warning", "info"}

    def test_alerts_have_required_fields(self):
        result = _generate_mock_summary("7d")
        for alert in result.alerts:
            assert "id" in alert
            assert "type" in alert
            assert "message" in alert
            assert "severity" in alert
            assert "timestamp" in alert

    def test_daily_costs_have_date_cost_tokens(self):
        result = _generate_mock_summary("7d")
        for entry in result.daily_costs:
            assert "date" in entry
            assert "cost" in entry
            assert "tokens" in entry

    def test_daily_cost_values_are_numeric(self):
        result = _generate_mock_summary("7d")
        for entry in result.daily_costs:
            assert isinstance(entry["cost"], (int, float))
            assert isinstance(entry["tokens"], int)

    def test_daily_dates_are_chronological(self):
        result = _generate_mock_summary("7d")
        dates = [entry["date"] for entry in result.daily_costs]
        assert dates == sorted(dates)

    def test_last_updated_is_recent(self):
        before = _now()
        result = _generate_mock_summary("7d")
        after = _now()
        assert before <= result.last_updated <= after


# ===========================================================================
# TestBuildExportRows -- grouping by provider, feature, daily
# ===========================================================================


class TestBuildExportRows:
    """Tests for _build_export_rows()."""

    def test_provider_grouping(self):
        summary = _make_summary()
        rows = _build_export_rows(summary, "provider")
        assert len(rows) == 2
        assert rows[0]["name"] == "Anthropic"
        assert rows[1]["name"] == "OpenAI"

    def test_provider_rows_have_name_cost_percentage(self):
        summary = _make_summary()
        rows = _build_export_rows(summary, "provider")
        for row in rows:
            assert set(row.keys()) == {"name", "cost", "percentage"}

    def test_feature_grouping(self):
        summary = _make_summary()
        rows = _build_export_rows(summary, "feature")
        assert len(rows) == 2
        assert rows[0]["name"] == "Debates"
        assert rows[1]["name"] == "Code Review"

    def test_feature_rows_have_name_cost_percentage(self):
        summary = _make_summary()
        rows = _build_export_rows(summary, "feature")
        for row in rows:
            assert set(row.keys()) == {"name", "cost", "percentage"}

    def test_daily_grouping(self):
        summary = _make_summary()
        rows = _build_export_rows(summary, "daily")
        assert len(rows) == 3

    def test_daily_rows_have_date_cost_tokens(self):
        summary = _make_summary()
        rows = _build_export_rows(summary, "daily")
        for row in rows:
            assert set(row.keys()) == {"date", "cost", "tokens"}

    def test_default_grouping_uses_daily(self):
        """Any unknown group_by falls through to daily."""
        summary = _make_summary()
        rows_default = _build_export_rows(summary, "anything_else")
        rows_daily = _build_export_rows(summary, "daily")
        assert rows_default == rows_daily

    def test_empty_provider_list(self):
        summary = _make_summary(cost_by_provider=[])
        rows = _build_export_rows(summary, "provider")
        assert rows == []

    def test_empty_feature_list(self):
        summary = _make_summary(cost_by_feature=[])
        rows = _build_export_rows(summary, "feature")
        assert rows == []

    def test_empty_daily_list(self):
        summary = _make_summary(daily_costs=[])
        rows = _build_export_rows(summary, "daily")
        assert rows == []

    def test_daily_cost_values_match_summary(self):
        summary = _make_summary()
        rows = _build_export_rows(summary, "daily")
        assert rows[0]["cost"] == 18.50
        assert rows[0]["tokens"] == 462500
        assert rows[0]["date"] == "2026-02-20"

    def test_provider_cost_values_match_summary(self):
        summary = _make_summary()
        rows = _build_export_rows(summary, "provider")
        assert rows[0]["cost"] == 77.31
        assert rows[0]["percentage"] == 61.6

    def test_daily_missing_keys_default(self):
        """daily_costs entries with missing keys get defaults."""
        summary = _make_summary(daily_costs=[{"extra": "field"}])
        rows = _build_export_rows(summary, "daily")
        assert rows[0] == {"date": "", "cost": 0, "tokens": 0}


# ===========================================================================
# TestExportCsvResponse -- CSV generation and HTTP headers
# ===========================================================================


class TestExportCsvResponse:
    """Tests for _export_csv_response()."""

    def test_empty_rows_returns_empty_body(self):
        resp = _export_csv_response([], "ws-1", "7d")
        assert resp.text == ""

    def test_empty_rows_content_type(self):
        resp = _export_csv_response([], "ws-1", "7d")
        assert resp.content_type == "text/csv"

    def test_empty_rows_has_content_disposition(self):
        resp = _export_csv_response([], "ws-1", "7d")
        assert "Content-Disposition" in resp.headers
        assert "costs_ws-1_7d.csv" in resp.headers["Content-Disposition"]

    def test_non_empty_rows_contain_header(self):
        rows = [{"name": "Anthropic", "cost": 77.31, "percentage": 61.6}]
        resp = _export_csv_response(rows, "ws-1", "7d")
        lines = resp.text.strip().splitlines()
        assert lines[0] == "name,cost,percentage"

    def test_non_empty_rows_contain_data(self):
        rows = [{"name": "Anthropic", "cost": 77.31, "percentage": 61.6}]
        resp = _export_csv_response(rows, "ws-1", "7d")
        lines = resp.text.strip().splitlines()
        assert len(lines) == 2
        assert "Anthropic" in lines[1]
        assert "77.31" in lines[1]

    def test_multiple_rows(self):
        rows = [
            {"name": "Anthropic", "cost": 77.31, "percentage": 61.6},
            {"name": "OpenAI", "cost": 34.64, "percentage": 27.6},
        ]
        resp = _export_csv_response(rows, "ws-1", "7d")
        lines = resp.text.strip().splitlines()
        # header + 2 data rows
        assert len(lines) == 3

    def test_content_type_is_csv(self):
        rows = [{"name": "A", "cost": 1.0, "percentage": 100.0}]
        resp = _export_csv_response(rows, "ws-1", "7d")
        assert resp.content_type == "text/csv"

    def test_content_disposition_includes_workspace_and_range(self):
        rows = [{"name": "A", "cost": 1.0, "percentage": 100.0}]
        resp = _export_csv_response(rows, "my-workspace", "30d")
        header = resp.headers["Content-Disposition"]
        assert "costs_my-workspace_30d.csv" in header
        assert header.startswith("attachment;")

    def test_csv_is_parseable(self):
        rows = [
            {"date": "2026-02-20", "cost": 18.50, "tokens": 462500},
            {"date": "2026-02-21", "cost": 21.00, "tokens": 525000},
        ]
        resp = _export_csv_response(rows, "ws-1", "7d")
        reader = csv.DictReader(io.StringIO(resp.text))
        parsed = list(reader)
        assert len(parsed) == 2
        assert parsed[0]["date"] == "2026-02-20"
        assert parsed[0]["cost"] == "18.5"
        assert parsed[1]["tokens"] == "525000"

    def test_daily_rows_fieldnames(self):
        rows = [{"date": "2026-02-20", "cost": 18.50, "tokens": 462500}]
        resp = _export_csv_response(rows, "ws-1", "7d")
        reader = csv.DictReader(io.StringIO(resp.text))
        assert reader.fieldnames == ["date", "cost", "tokens"]


# ===========================================================================
# TestGetImplementationSteps -- known + unknown recommendation types
# ===========================================================================


class TestGetImplementationSteps:
    """Tests for _get_implementation_steps()."""

    def test_model_downgrade_returns_list(self):
        steps = _get_implementation_steps("model_downgrade")
        assert isinstance(steps, list)
        assert len(steps) == 4

    def test_model_downgrade_first_step(self):
        steps = _get_implementation_steps("model_downgrade")
        assert "lower-tier" in steps[0].lower() or "identify" in steps[0].lower()

    def test_caching_returns_list(self):
        steps = _get_implementation_steps("caching")
        assert isinstance(steps, list)
        assert len(steps) == 4

    def test_caching_mentions_cache(self):
        steps = _get_implementation_steps("caching")
        joined = " ".join(steps).lower()
        assert "cache" in joined or "cach" in joined

    def test_batching_returns_list(self):
        steps = _get_implementation_steps("batching")
        assert isinstance(steps, list)
        assert len(steps) == 4

    def test_batching_mentions_batch(self):
        steps = _get_implementation_steps("batching")
        joined = " ".join(steps).lower()
        assert "batch" in joined

    def test_rate_limiting_returns_list(self):
        steps = _get_implementation_steps("rate_limiting")
        assert isinstance(steps, list)
        assert len(steps) == 4

    def test_rate_limiting_mentions_rate(self):
        steps = _get_implementation_steps("rate_limiting")
        joined = " ".join(steps).lower()
        assert "rate" in joined or "limit" in joined

    def test_unknown_type_returns_default_steps(self):
        steps = _get_implementation_steps("unknown_type")
        assert isinstance(steps, list)
        assert len(steps) == 4
        assert steps[0] == "Review recommendation details"

    def test_empty_string_returns_default_steps(self):
        steps = _get_implementation_steps("")
        assert steps[0] == "Review recommendation details"

    def test_all_known_types_return_unique_steps(self):
        known = ["model_downgrade", "caching", "batching", "rate_limiting"]
        all_steps = [tuple(_get_implementation_steps(t)) for t in known]
        # All should be different from each other
        assert len(set(all_steps)) == 4

    def test_default_steps_differ_from_all_known(self):
        default = _get_implementation_steps("nonexistent")
        for known_type in ["model_downgrade", "caching", "batching", "rate_limiting"]:
            assert default != _get_implementation_steps(known_type)


# ===========================================================================
# TestGetImplementationDifficulty -- known + unknown recommendation types
# ===========================================================================


class TestGetImplementationDifficulty:
    """Tests for _get_implementation_difficulty()."""

    def test_model_downgrade_is_easy(self):
        assert _get_implementation_difficulty("model_downgrade") == "easy"

    def test_caching_is_medium(self):
        assert _get_implementation_difficulty("caching") == "medium"

    def test_batching_is_medium(self):
        assert _get_implementation_difficulty("batching") == "medium"

    def test_rate_limiting_is_easy(self):
        assert _get_implementation_difficulty("rate_limiting") == "easy"

    def test_unknown_type_defaults_to_medium(self):
        assert _get_implementation_difficulty("something_else") == "medium"

    def test_empty_string_defaults_to_medium(self):
        assert _get_implementation_difficulty("") == "medium"

    def test_return_type_is_string(self):
        for t in ["model_downgrade", "caching", "batching", "rate_limiting", "xyz"]:
            assert isinstance(_get_implementation_difficulty(t), str)


# ===========================================================================
# TestGetImplementationTime -- known + unknown recommendation types
# ===========================================================================


class TestGetImplementationTime:
    """Tests for _get_implementation_time()."""

    def test_model_downgrade_under_1_hour(self):
        assert _get_implementation_time("model_downgrade") == "< 1 hour"

    def test_caching_2_to_4_hours(self):
        assert _get_implementation_time("caching") == "2-4 hours"

    def test_batching_4_to_8_hours(self):
        assert _get_implementation_time("batching") == "4-8 hours"

    def test_rate_limiting_1_to_2_hours(self):
        assert _get_implementation_time("rate_limiting") == "1-2 hours"

    def test_unknown_type_defaults_to_2_to_4_hours(self):
        assert _get_implementation_time("nonexistent") == "2-4 hours"

    def test_empty_string_defaults_to_2_to_4_hours(self):
        assert _get_implementation_time("") == "2-4 hours"

    def test_return_type_is_string(self):
        for t in ["model_downgrade", "caching", "batching", "rate_limiting", "xyz"]:
            assert isinstance(_get_implementation_time(t), str)


# ===========================================================================
# TestIntegration -- end-to-end helper combinations
# ===========================================================================


class TestIntegration:
    """Integration tests combining multiple helpers."""

    def test_mock_summary_through_export_provider(self):
        """Generate mock summary then export by provider produces valid CSV."""
        summary = _generate_mock_summary("7d")
        rows = _build_export_rows(summary, "provider")
        resp = _export_csv_response(rows, "ws-1", "7d")
        assert resp.content_type == "text/csv"
        reader = csv.DictReader(io.StringIO(resp.text))
        parsed = list(reader)
        assert len(parsed) == 4
        assert parsed[0]["name"] == "Anthropic"

    def test_mock_summary_through_export_feature(self):
        """Generate mock summary then export by feature produces valid CSV."""
        summary = _generate_mock_summary("30d")
        rows = _build_export_rows(summary, "feature")
        resp = _export_csv_response(rows, "ws-2", "30d")
        reader = csv.DictReader(io.StringIO(resp.text))
        parsed = list(reader)
        assert len(parsed) == 4
        names = [r["name"] for r in parsed]
        assert "Debates" in names

    def test_mock_summary_through_export_daily(self):
        """Generate mock summary then export daily produces correct row count."""
        summary = _generate_mock_summary("90d")
        rows = _build_export_rows(summary, "daily")
        resp = _export_csv_response(rows, "ws-3", "90d")
        reader = csv.DictReader(io.StringIO(resp.text))
        parsed = list(reader)
        assert len(parsed) == 90

    def test_all_recommendation_types_have_consistent_metadata(self):
        """Each known rec type has steps, difficulty, and time."""
        for rec_type in ["model_downgrade", "caching", "batching", "rate_limiting"]:
            steps = _get_implementation_steps(rec_type)
            diff = _get_implementation_difficulty(rec_type)
            time = _get_implementation_time(rec_type)
            assert len(steps) == 4
            assert diff in ("easy", "medium", "hard")
            assert isinstance(time, str)
            assert "hour" in time
