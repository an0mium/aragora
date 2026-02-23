"""
Tests for OutcomeHandler - Decision outcome tracking REST endpoints.

Tests cover:
- POST /api/v1/decisions/{id}/outcome - Record an outcome
- GET /api/v1/decisions/{id}/outcomes - List outcomes for a decision
- GET /api/v1/outcomes/search - Search outcomes
- GET /api/v1/outcomes/impact - Impact analytics
- Input validation and error handling
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from collections import OrderedDict

from aragora.server.handlers.governance.outcomes import (
    OutcomeHandler,
    _outcome_store,
)


@pytest.fixture(autouse=True)
def clear_outcome_store():
    """Clear the in-memory outcome store before each test."""
    _outcome_store.clear()
    yield
    _outcome_store.clear()


def _make_handler_with_body(body: dict) -> MagicMock:
    """Create a mock handler with JSON body."""
    handler = MagicMock()
    handler.rfile = MagicMock()
    handler.headers = {"Content-Length": str(len(json.dumps(body)))}
    handler.rfile.read.return_value = json.dumps(body).encode()
    return handler


def _make_handler_with_query(query: str = "") -> MagicMock:
    """Create a mock handler with query params."""
    handler = MagicMock()
    parsed_url = MagicMock()
    parsed_url.query = query
    handler.parsed_url = parsed_url
    return handler


class TestOutcomeHandlerInit:
    """Tests for handler initialization."""

    def test_init_default(self):
        h = OutcomeHandler()
        assert h.ctx == {}

    def test_init_with_ctx(self):
        h = OutcomeHandler(ctx={"key": "val"})
        assert h.ctx["key"] == "val"

    def test_routes_defined(self):
        assert len(OutcomeHandler.ROUTES) > 0
        assert "/api/v1/decisions/*/outcome" in OutcomeHandler.ROUTES
        assert "/api/v1/outcomes/search" in OutcomeHandler.ROUTES


class TestCanHandle:
    """Tests for route matching."""

    def test_post_outcome(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/v1/decisions/dec123/outcome") is True

    def test_get_outcomes(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/v1/decisions/dec123/outcomes") is True

    def test_get_search(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/v1/outcomes/search") is True

    def test_get_impact(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/v1/outcomes/impact") is True

    def test_unrelated_path(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/v1/debates") is False


class TestRecordOutcome:
    """Tests for POST /api/v1/decisions/{id}/outcome."""

    @patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter")
    def test_record_success(self, mock_get_adapter):
        mock_adapter = MagicMock()
        mock_adapter.ingest.return_value = True
        mock_get_adapter.return_value = mock_adapter

        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_123",
            "outcome_type": "success",
            "outcome_description": "It worked",
            "impact_score": 0.8,
        })

        result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)
        assert result["status"] == 201
        body = json.loads(result["body"])
        assert body["decision_id"] == "dec_abc"
        assert body["status"] == "recorded"
        assert "outcome_id" in body

    def test_record_missing_debate_id(self):
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "outcome_type": "success",
            "outcome_description": "test",
            "impact_score": 0.5,
        })

        result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)
        assert result["status"] == 400

    def test_record_invalid_outcome_type(self):
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "invalid_type",
            "outcome_description": "test",
            "impact_score": 0.5,
        })

        result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)
        assert result["status"] == 400

    def test_record_missing_description(self):
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "success",
            "impact_score": 0.5,
        })

        result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)
        assert result["status"] == 400

    def test_record_impact_out_of_range(self):
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "success",
            "outcome_description": "test",
            "impact_score": 1.5,
        })

        result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)
        assert result["status"] == 400

    def test_record_stores_in_memory(self):
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "success",
            "outcome_description": "Worked",
            "impact_score": 0.7,
        })

        with patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter") as m:
            m.return_value = MagicMock(ingest=MagicMock(return_value=True))
            h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)

        assert len(_outcome_store) == 1
        stored = list(_outcome_store.values())[0]
        assert stored["decision_id"] == "dec_abc"
        assert stored["outcome_type"] == "success"


class TestListOutcomes:
    """Tests for GET /api/v1/decisions/{id}/outcomes."""

    def test_list_empty(self):
        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_list_outcomes("/api/v1/decisions/dec_abc/outcomes", handler)
        body = json.loads(result["body"])
        assert body["count"] == 0
        assert body["outcomes"] == []

    def test_list_with_matching_outcomes(self):
        _outcome_store["o1"] = {"decision_id": "dec_abc", "outcome_type": "success"}
        _outcome_store["o2"] = {"decision_id": "dec_abc", "outcome_type": "failure"}
        _outcome_store["o3"] = {"decision_id": "dec_other", "outcome_type": "success"}

        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_list_outcomes("/api/v1/decisions/dec_abc/outcomes", handler)
        body = json.loads(result["body"])
        assert body["count"] == 2
        assert body["decision_id"] == "dec_abc"


class TestSearchOutcomes:
    """Tests for GET /api/v1/outcomes/search."""

    def test_search_empty_store(self):
        h = OutcomeHandler()
        handler = _make_handler_with_query("")

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 0

    def test_search_by_query(self):
        _outcome_store["o1"] = {
            "outcome_description": "Vendor delivered on time",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "success",
        }
        _outcome_store["o2"] = {
            "outcome_description": "Budget was exceeded",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "failure",
        }

        h = OutcomeHandler()
        handler = _make_handler_with_query("q=vendor")

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 1

    def test_search_by_type(self):
        _outcome_store["o1"] = {
            "outcome_description": "A",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "success",
        }
        _outcome_store["o2"] = {
            "outcome_description": "B",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "failure",
        }

        h = OutcomeHandler()
        handler = _make_handler_with_query("type=failure")

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 1

    def test_search_by_tags(self):
        _outcome_store["o1"] = {
            "outcome_description": "A",
            "lessons_learned": "",
            "tags": ["vendor", "procurement"],
            "outcome_type": "success",
        }
        _outcome_store["o2"] = {
            "outcome_description": "B",
            "lessons_learned": "",
            "tags": ["hiring"],
            "outcome_type": "success",
        }

        h = OutcomeHandler()
        handler = _make_handler_with_query("tags=vendor")

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 1


class TestImpactAnalytics:
    """Tests for GET /api/v1/outcomes/impact."""

    def test_impact_empty(self):
        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_impact_analytics(handler)
        body = json.loads(result["body"])
        assert body["total_outcomes"] == 0
        assert body["avg_impact_score"] == 0.0

    def test_impact_with_outcomes(self):
        _outcome_store["o1"] = {
            "outcome_type": "success",
            "impact_score": 0.8,
            "lessons_learned": "Early screening helps",
        }
        _outcome_store["o2"] = {
            "outcome_type": "success",
            "impact_score": 0.6,
            "lessons_learned": "",
        }
        _outcome_store["o3"] = {
            "outcome_type": "failure",
            "impact_score": 0.3,
            "lessons_learned": "Need better criteria",
        }

        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_impact_analytics(handler)
        body = json.loads(result["body"])

        assert body["total_outcomes"] == 3
        assert body["by_type"]["success"]["count"] == 2
        assert body["by_type"]["failure"]["count"] == 1
        assert body["avg_impact_score"] > 0
        assert len(body["top_lessons"]) == 2  # Only outcomes with lessons

    def test_impact_avg_computation_correct(self):
        """Verify average impact score is computed correctly."""
        _outcome_store["o1"] = {
            "outcome_type": "success",
            "impact_score": 1.0,
            "lessons_learned": "",
        }
        _outcome_store["o2"] = {
            "outcome_type": "failure",
            "impact_score": 0.0,
            "lessons_learned": "",
        }

        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_impact_analytics(handler)
        body = json.loads(result["body"])
        assert body["avg_impact_score"] == 0.5

    def test_impact_top_lessons_sorted_by_score(self):
        """Top lessons should be sorted by impact score descending."""
        _outcome_store["o1"] = {
            "outcome_type": "success",
            "impact_score": 0.3,
            "lessons_learned": "Low impact lesson",
        }
        _outcome_store["o2"] = {
            "outcome_type": "success",
            "impact_score": 0.9,
            "lessons_learned": "High impact lesson",
        }

        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_impact_analytics(handler)
        body = json.loads(result["body"])
        lessons = body["top_lessons"]
        assert lessons[0]["lesson"] == "High impact lesson"
        assert lessons[1]["lesson"] == "Low impact lesson"


class TestRecordOutcomeEdgeCases:
    """Additional edge case tests for record endpoint."""

    def test_record_missing_decision_id_in_path(self):
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "success",
            "outcome_description": "test",
            "impact_score": 0.5,
        })

        result = h._handle_record_outcome("/api/v1/outcomes/search", handler)
        assert result["status"] == 400

    def test_record_negative_impact_score(self):
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "success",
            "outcome_description": "test",
            "impact_score": -0.5,
        })

        result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)
        assert result["status"] == 400

    def test_record_impact_score_string(self):
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "success",
            "outcome_description": "test",
            "impact_score": "not_a_number",
        })

        result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)
        assert result["status"] == 400

    def test_record_with_optional_fields(self):
        """Recording with all optional fields should succeed."""
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "partial",
            "outcome_description": "Partially met goals",
            "impact_score": 0.6,
            "kpis_before": {"revenue": 100000},
            "kpis_after": {"revenue": 120000},
            "lessons_learned": "Set clearer milestones",
            "tags": ["strategy", "q4"],
        })

        with patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter") as m:
            m.return_value = MagicMock(ingest=MagicMock(return_value=True))
            result = h._handle_record_outcome("/api/v1/decisions/dec_xyz/outcome", handler)

        assert result["status"] == 201
        stored = list(_outcome_store.values())[0]
        assert stored["lessons_learned"] == "Set clearer milestones"
        assert "strategy" in stored["tags"]

    def test_record_km_adapter_failure_still_stores(self):
        """Even if KM ingestion fails, the outcome should still be stored."""
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_1",
            "outcome_type": "success",
            "outcome_description": "test",
            "impact_score": 0.5,
        })

        with patch(
            "aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter",
            side_effect=RuntimeError("adapter unavailable"),
        ):
            result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)

        assert result["status"] == 201
        assert len(_outcome_store) == 1

    def test_record_builds_outcome_record_type(self):
        """Verify the handler creates an OutcomeRecord and stores its dict form."""
        h = OutcomeHandler()
        handler = _make_handler_with_body({
            "debate_id": "dbt_typed",
            "outcome_type": "failure",
            "outcome_description": "Deployment failed",
            "impact_score": 0.9,
            "kpis_before": {"errors": 0},
            "kpis_after": {"errors": 42},
        })

        with patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter") as m:
            m.return_value = MagicMock(ingest=MagicMock(return_value=True))
            result = h._handle_record_outcome("/api/v1/decisions/dec_typed/outcome", handler)

        assert result["status"] == 201
        stored = list(_outcome_store.values())[0]
        assert stored["outcome_type"] == "failure"
        assert stored["kpis_before"]["errors"] == 0
        assert stored["kpis_after"]["errors"] == 42

    def test_record_invalid_json_body(self):
        """Non-JSON body should return 400."""
        h = OutcomeHandler()
        handler = MagicMock()
        handler.headers = {"Content-Length": "5"}
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = b"notjson"

        result = h._handle_record_outcome("/api/v1/decisions/dec_abc/outcome", handler)
        assert result["status"] == 400


class TestListOutcomesEdgeCases:
    """Additional tests for list outcomes endpoint."""

    def test_list_missing_decision_id(self):
        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_list_outcomes("/api/v1/outcomes/search", handler)
        assert result["status"] == 400

    def test_list_no_matching_outcomes(self):
        _outcome_store["o1"] = {"decision_id": "dec_other", "outcome_type": "success"}

        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_list_outcomes("/api/v1/decisions/dec_abc/outcomes", handler)
        body = json.loads(result["body"])
        assert body["count"] == 0

    def test_list_legacy_path(self):
        """Legacy unversioned path should also work."""
        _outcome_store["o1"] = {"decision_id": "dec_abc", "outcome_type": "success"}

        h = OutcomeHandler()
        handler = MagicMock()

        result = h._handle_list_outcomes("/api/decisions/dec_abc/outcomes", handler)
        body = json.loads(result["body"])
        assert body["count"] == 1


class TestSearchOutcomesEdgeCases:
    """Additional tests for search endpoint."""

    def test_search_matches_lessons_learned(self):
        """Search query should match in lessons_learned field too."""
        _outcome_store["o1"] = {
            "outcome_description": "Something happened",
            "lessons_learned": "Always validate vendor references",
            "tags": [],
            "outcome_type": "failure",
        }

        h = OutcomeHandler()
        handler = _make_handler_with_query("q=vendor")

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 1

    def test_search_with_limit(self):
        """Limit parameter should cap the number of results."""
        for i in range(10):
            _outcome_store[f"o{i}"] = {
                "outcome_description": f"Outcome {i}",
                "lessons_learned": "",
                "tags": [],
                "outcome_type": "success",
            }

        h = OutcomeHandler()
        handler = _make_handler_with_query("limit=3")

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 3

    def test_search_combined_filters(self):
        """Multiple filters should be applied together (AND logic)."""
        _outcome_store["o1"] = {
            "outcome_description": "Vendor delivered on time",
            "lessons_learned": "",
            "tags": ["vendor"],
            "outcome_type": "success",
        }
        _outcome_store["o2"] = {
            "outcome_description": "Vendor was late",
            "lessons_learned": "",
            "tags": ["vendor"],
            "outcome_type": "failure",
        }

        h = OutcomeHandler()
        handler = _make_handler_with_query("q=vendor&type=failure")

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 1

    def test_search_case_insensitive(self):
        """Query matching should be case-insensitive."""
        _outcome_store["o1"] = {
            "outcome_description": "VENDOR Delivered",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "success",
        }

        h = OutcomeHandler()
        handler = _make_handler_with_query("q=vendor")

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 1

    def test_search_without_parsed_url(self):
        """Handler without parsed_url should return all outcomes."""
        _outcome_store["o1"] = {
            "outcome_description": "test",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "success",
        }

        h = OutcomeHandler()
        handler = MagicMock(spec=[])  # No parsed_url attribute

        result = h._handle_search_outcomes(handler)
        body = json.loads(result["body"])
        assert body["count"] == 1


class TestCanHandleEdgeCases:
    """Additional route matching tests."""

    def test_trailing_slash_on_outcome(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/v1/decisions/dec123/outcome/") is True

    def test_legacy_outcome_path(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/decisions/dec123/outcome") is True

    def test_legacy_search_path(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/outcomes/search") is True

    def test_legacy_impact_path(self):
        h = OutcomeHandler()
        assert h.can_handle("/api/outcomes/impact") is True
