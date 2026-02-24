"""
Tests for OutcomeHandler - Decision outcome tracking REST endpoints.

Tests cover all 4 endpoints:
- POST /api/v1/decisions/{id}/outcome  - Record an outcome
- GET  /api/v1/decisions/{id}/outcomes - List outcomes for a decision
- GET  /api/v1/outcomes/search         - Search outcomes by topic/tags
- GET  /api/v1/outcomes/impact         - Impact analytics

Plus parameter validation, error handling, and empty-result edge cases.
"""

from __future__ import annotations

import json

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.governance.outcomes import (
    OutcomeHandler,
    _outcome_store,
    _extract_decision_id,
    VALID_OUTCOME_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_outcome_store():
    """Clear the in-memory outcome store before and after each test."""
    _outcome_store.clear()
    yield
    _outcome_store.clear()


@pytest.fixture
def handler() -> OutcomeHandler:
    """Return a fresh OutcomeHandler instance."""
    return OutcomeHandler(ctx={})


def _make_http_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP request handler with optional JSON body."""
    mock = MagicMock()
    if body is not None:
        raw = json.dumps(body).encode()
        mock.headers = {"Content-Length": str(len(raw))}
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = raw
    else:
        mock.headers = {"Content-Length": "0"}
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = b""
    return mock


def _body(result: dict) -> dict:
    """Parse the JSON body from a HandlerResult dict."""
    return json.loads(result["body"])


# ---------------------------------------------------------------------------
# Initialization & can_handle
# ---------------------------------------------------------------------------


class TestOutcomeHandlerInit:
    """Tests for handler construction and route matching."""

    def test_init_default_ctx(self, handler: OutcomeHandler):
        assert handler.ctx == {}

    def test_init_with_ctx(self):
        h = OutcomeHandler(ctx={"storage": "mock"})
        assert h.ctx["storage"] == "mock"

    def test_can_handle_post_outcome(self, handler: OutcomeHandler):
        assert handler.can_handle("/api/v1/decisions/dec_abc/outcome") is True

    def test_can_handle_get_outcomes(self, handler: OutcomeHandler):
        assert handler.can_handle("/api/v1/decisions/dec_abc/outcomes") is True

    def test_can_handle_search(self, handler: OutcomeHandler):
        assert handler.can_handle("/api/v1/outcomes/search") is True

    def test_can_handle_impact(self, handler: OutcomeHandler):
        assert handler.can_handle("/api/v1/outcomes/impact") is True

    def test_can_handle_rejects_unrelated(self, handler: OutcomeHandler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_can_handle_legacy_paths(self, handler: OutcomeHandler):
        assert handler.can_handle("/api/decisions/dec123/outcome") is True
        assert handler.can_handle("/api/outcomes/search") is True

    def test_can_handle_trailing_slash(self, handler: OutcomeHandler):
        assert handler.can_handle("/api/v1/decisions/dec123/outcome/") is True


# ---------------------------------------------------------------------------
# POST /api/v1/decisions/{id}/outcome
# ---------------------------------------------------------------------------


class TestRecordOutcome:
    """Tests for the record-outcome POST endpoint."""

    @patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter")
    def test_record_success(self, mock_adapter_fn, handler: OutcomeHandler):
        mock_adapter_fn.return_value = MagicMock(ingest=MagicMock())
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "success",
                "outcome_description": "Revenue increased",
                "impact_score": 0.85,
            }
        )
        result = handler.handle_post("/api/v1/decisions/dec_abc/outcome", {}, http)
        assert result is not None
        assert result["status"] == 201
        body = _body(result)
        assert body["decision_id"] == "dec_abc"
        assert body["status"] == "recorded"
        assert body["outcome_id"].startswith("out_")

    def test_record_missing_debate_id(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "outcome_type": "success",
                "outcome_description": "test",
                "impact_score": 0.5,
            }
        )
        result = handler.handle_post("/api/v1/decisions/dec_abc/outcome", {}, http)
        assert result["status"] == 400
        assert "debate_id" in _body(result).get("error", "")

    def test_record_invalid_outcome_type(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "bogus",
                "outcome_description": "test",
                "impact_score": 0.5,
            }
        )
        result = handler.handle_post("/api/v1/decisions/dec_abc/outcome", {}, http)
        assert result["status"] == 400

    def test_record_missing_description(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "failure",
                "impact_score": 0.3,
            }
        )
        result = handler.handle_post("/api/v1/decisions/dec_abc/outcome", {}, http)
        assert result["status"] == 400

    def test_record_impact_out_of_range(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "success",
                "outcome_description": "test",
                "impact_score": 1.5,
            }
        )
        result = handler.handle_post("/api/v1/decisions/dec_abc/outcome", {}, http)
        assert result["status"] == 400

    def test_record_negative_impact(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "success",
                "outcome_description": "test",
                "impact_score": -0.1,
            }
        )
        result = handler.handle_post("/api/v1/decisions/dec_abc/outcome", {}, http)
        assert result["status"] == 400

    def test_record_impact_string_rejected(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "success",
                "outcome_description": "test",
                "impact_score": "high",
            }
        )
        result = handler.handle_post("/api/v1/decisions/dec_abc/outcome", {}, http)
        assert result["status"] == 400

    def test_record_stores_outcome(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "partial",
                "outcome_description": "Mixed results",
                "impact_score": 0.5,
            }
        )
        with patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter") as m:
            m.return_value = MagicMock(ingest=MagicMock())
            handler.handle_post("/api/v1/decisions/dec_store/outcome", {}, http)

        assert len(_outcome_store) == 1
        stored = list(_outcome_store.values())[0]
        assert stored["decision_id"] == "dec_store"

    def test_record_with_optional_fields(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "success",
                "outcome_description": "Good",
                "impact_score": 0.9,
                "kpis_before": {"revenue": 100},
                "kpis_after": {"revenue": 200},
                "lessons_learned": "Act fast",
                "tags": ["strategy", "q1"],
            }
        )
        with patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter") as m:
            m.return_value = MagicMock(ingest=MagicMock())
            result = handler.handle_post("/api/v1/decisions/dec_opt/outcome", {}, http)

        assert result["status"] == 201
        stored = list(_outcome_store.values())[0]
        assert stored["lessons_learned"] == "Act fast"
        assert "strategy" in stored["tags"]

    def test_record_km_failure_still_stores(self, handler: OutcomeHandler):
        """KM ingestion failure should not prevent outcome storage."""
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "success",
                "outcome_description": "test",
                "impact_score": 0.5,
            }
        )
        with patch(
            "aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter",
            side_effect=RuntimeError("KM unavailable"),
        ):
            result = handler.handle_post("/api/v1/decisions/dec_km/outcome", {}, http)

        assert result["status"] == 201
        assert len(_outcome_store) == 1

    def test_record_invalid_json_body(self, handler: OutcomeHandler):
        http = MagicMock()
        http.headers = {"Content-Length": "7"}
        http.rfile = MagicMock()
        http.rfile.read.return_value = b"not{json"
        result = handler.handle_post("/api/v1/decisions/dec_abc/outcome", {}, http)
        assert result["status"] == 400

    def test_record_missing_decision_id_in_path(self, handler: OutcomeHandler):
        http = _make_http_handler(
            {
                "debate_id": "dbt_1",
                "outcome_type": "success",
                "outcome_description": "test",
                "impact_score": 0.5,
            }
        )
        # Path without /decisions/ segment
        result = handler.handle_post("/api/v1/outcomes/outcome", {}, http)
        # handle_post checks path.endswith("/outcome"), but _extract_decision_id
        # will fail. handle_post wraps _handle_record_outcome which returns 400.
        assert result is not None
        assert result["status"] == 400


# ---------------------------------------------------------------------------
# GET /api/v1/decisions/{id}/outcomes
# ---------------------------------------------------------------------------


class TestListOutcomes:
    """Tests for listing outcomes for a specific decision."""

    def test_list_empty(self, handler: OutcomeHandler):
        result = handler.handle("/api/v1/decisions/dec_abc/outcomes", {}, MagicMock())
        body = _body(result)
        assert body["count"] == 0
        assert body["outcomes"] == []

    def test_list_returns_matching(self, handler: OutcomeHandler):
        _outcome_store["o1"] = {"decision_id": "dec_abc", "outcome_type": "success"}
        _outcome_store["o2"] = {"decision_id": "dec_abc", "outcome_type": "failure"}
        _outcome_store["o3"] = {"decision_id": "dec_other", "outcome_type": "success"}

        result = handler.handle("/api/v1/decisions/dec_abc/outcomes", {}, MagicMock())
        body = _body(result)
        assert body["count"] == 2
        assert body["decision_id"] == "dec_abc"

    def test_list_no_matching(self, handler: OutcomeHandler):
        _outcome_store["o1"] = {"decision_id": "dec_other", "outcome_type": "success"}
        result = handler.handle("/api/v1/decisions/dec_abc/outcomes", {}, MagicMock())
        body = _body(result)
        assert body["count"] == 0

    def test_list_missing_decision_id(self, handler: OutcomeHandler):
        result = handler.handle("/api/v1/outcomes/outcomes", {}, MagicMock())
        # This path won't match /outcomes but rather would go through search
        # or return None because it doesn't match the patterns correctly.
        # Let's test a path that matches endswith("/outcomes") but has no decisions/
        result = handler._handle_list_outcomes("/api/v1/stuff/outcomes")
        assert result["status"] == 400


# ---------------------------------------------------------------------------
# GET /api/v1/outcomes/search
# ---------------------------------------------------------------------------


class TestSearchOutcomes:
    """Tests for searching outcomes."""

    def test_search_empty_store(self, handler: OutcomeHandler):
        result = handler.handle("/api/v1/outcomes/search", {}, MagicMock())
        body = _body(result)
        assert body["count"] == 0

    def test_search_by_query(self, handler: OutcomeHandler):
        _outcome_store["o1"] = {
            "outcome_description": "Vendor delivered on time",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "success",
        }
        _outcome_store["o2"] = {
            "outcome_description": "Budget exceeded",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "failure",
        }
        result = handler.handle("/api/v1/outcomes/search", {"q": "vendor"}, MagicMock())
        body = _body(result)
        assert body["count"] == 1

    def test_search_by_tags(self, handler: OutcomeHandler):
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
        result = handler.handle("/api/v1/outcomes/search", {"tags": "vendor"}, MagicMock())
        body = _body(result)
        assert body["count"] == 1

    def test_search_by_type(self, handler: OutcomeHandler):
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
        result = handler.handle("/api/v1/outcomes/search", {"type": "failure"}, MagicMock())
        body = _body(result)
        assert body["count"] == 1

    def test_search_with_limit(self, handler: OutcomeHandler):
        for i in range(10):
            _outcome_store[f"o{i}"] = {
                "outcome_description": f"Outcome {i}",
                "lessons_learned": "",
                "tags": [],
                "outcome_type": "success",
            }
        result = handler.handle("/api/v1/outcomes/search", {"limit": "3"}, MagicMock())
        body = _body(result)
        assert body["count"] == 3

    def test_search_case_insensitive(self, handler: OutcomeHandler):
        _outcome_store["o1"] = {
            "outcome_description": "VENDOR Delivered",
            "lessons_learned": "",
            "tags": [],
            "outcome_type": "success",
        }
        result = handler.handle("/api/v1/outcomes/search", {"q": "vendor"}, MagicMock())
        body = _body(result)
        assert body["count"] == 1

    def test_search_matches_lessons_learned(self, handler: OutcomeHandler):
        _outcome_store["o1"] = {
            "outcome_description": "Something happened",
            "lessons_learned": "Always validate vendor references",
            "tags": [],
            "outcome_type": "failure",
        }
        result = handler.handle("/api/v1/outcomes/search", {"q": "vendor"}, MagicMock())
        body = _body(result)
        assert body["count"] == 1


# ---------------------------------------------------------------------------
# GET /api/v1/outcomes/impact
# ---------------------------------------------------------------------------


class TestImpactAnalytics:
    """Tests for the impact analytics endpoint."""

    def test_impact_empty(self, handler: OutcomeHandler):
        result = handler.handle("/api/v1/outcomes/impact", {}, MagicMock())
        body = _body(result)
        assert body["total_outcomes"] == 0
        assert body["avg_impact_score"] == 0.0
        assert body["top_lessons"] == []

    def test_impact_aggregates(self, handler: OutcomeHandler):
        _outcome_store["o1"] = {
            "outcome_type": "success",
            "impact_score": 0.8,
            "lessons_learned": "Ship fast",
        }
        _outcome_store["o2"] = {
            "outcome_type": "success",
            "impact_score": 0.6,
            "lessons_learned": "",
        }
        _outcome_store["o3"] = {
            "outcome_type": "failure",
            "impact_score": 0.3,
            "lessons_learned": "Test more",
        }

        result = handler.handle("/api/v1/outcomes/impact", {}, MagicMock())
        body = _body(result)
        assert body["total_outcomes"] == 3
        assert body["by_type"]["success"]["count"] == 2
        assert body["by_type"]["failure"]["count"] == 1
        assert body["avg_impact_score"] > 0

    def test_impact_avg_correct(self, handler: OutcomeHandler):
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
        result = handler.handle("/api/v1/outcomes/impact", {}, MagicMock())
        body = _body(result)
        assert body["avg_impact_score"] == 0.5

    def test_impact_top_lessons_sorted(self, handler: OutcomeHandler):
        _outcome_store["o1"] = {
            "outcome_type": "success",
            "impact_score": 0.3,
            "lessons_learned": "Low impact",
        }
        _outcome_store["o2"] = {
            "outcome_type": "success",
            "impact_score": 0.9,
            "lessons_learned": "High impact",
        }
        result = handler.handle("/api/v1/outcomes/impact", {}, MagicMock())
        body = _body(result)
        assert body["top_lessons"][0]["lesson"] == "High impact"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExtractDecisionId:
    """Tests for the _extract_decision_id helper."""

    def test_versioned_path(self):
        assert _extract_decision_id("/api/v1/decisions/dec_abc/outcome") == "dec_abc"

    def test_legacy_path(self):
        assert _extract_decision_id("/api/decisions/dec_xyz/outcomes") == "dec_xyz"

    def test_no_decisions_segment(self):
        assert _extract_decision_id("/api/v1/outcomes/search") is None

    def test_decisions_at_end(self):
        assert _extract_decision_id("/api/v1/decisions") is None
