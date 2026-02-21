"""
Tests for the Outcome API endpoints via OutcomeHandler.

Focuses on end-to-end routing, edge cases, @handle_errors decorator behavior,
KM adapter integration, eviction, and registration verification.

Endpoints under test:
- POST /api/v1/decisions/{id}/outcome
- GET  /api/v1/decisions/{id}/outcomes
- GET  /api/v1/outcomes/search
- GET  /api/v1/outcomes/impact
"""

from __future__ import annotations

import json
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.governance.outcomes import (
    MAX_OUTCOMES,
    OutcomeHandler,
    VALID_OUTCOME_TYPES,
    _outcome_store,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_store():
    """Clear the in-memory outcome store before and after each test."""
    _outcome_store.clear()
    yield
    _outcome_store.clear()


@pytest.fixture
def handler():
    """Create an OutcomeHandler instance."""
    return OutcomeHandler()


def _mock_post_handler(body: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON POST body."""
    raw = json.dumps(body).encode()
    mock = MagicMock()
    mock.headers = {"Content-Length": str(len(raw))}
    mock.rfile.read.return_value = raw
    return mock


def _mock_get_handler(query: str = "") -> MagicMock:
    """Create a mock HTTP handler with query params."""
    mock = MagicMock()
    parsed_url = MagicMock()
    parsed_url.query = query
    mock.parsed_url = parsed_url
    return mock


def _valid_outcome_body(**overrides) -> dict:
    """Return a valid outcome request body, with optional overrides."""
    body = {
        "debate_id": "dbt_test_001",
        "outcome_type": "success",
        "outcome_description": "Revenue increased by 12%",
        "impact_score": 0.75,
        "kpis_before": {"revenue": 100000},
        "kpis_after": {"revenue": 112000},
        "lessons_learned": "Early vendor engagement pays off",
        "tags": ["vendor", "revenue"],
    }
    body.update(overrides)
    return body


# ============================================================================
# Registration Tests
# ============================================================================


class TestOutcomeHandlerRegistration:
    """Verify that the handler is properly registered in the lazy import system."""

    def test_in_handler_modules(self):
        """OutcomeHandler must appear in HANDLER_MODULES."""
        from aragora.server.handlers._lazy_imports import HANDLER_MODULES

        assert "OutcomeHandler" in HANDLER_MODULES
        assert HANDLER_MODULES["OutcomeHandler"] == "aragora.server.handlers.governance"

    def test_in_all_handler_names(self):
        """OutcomeHandler must appear in ALL_HANDLER_NAMES dispatch list."""
        from aragora.server.handlers._lazy_imports import ALL_HANDLER_NAMES

        assert "OutcomeHandler" in ALL_HANDLER_NAMES

    def test_lazy_import_works(self):
        """Importing OutcomeHandler via the handlers package should succeed."""
        from aragora.server.handlers import OutcomeHandler as LazyLoaded

        assert LazyLoaded is OutcomeHandler


# ============================================================================
# Routing via handle()
# ============================================================================


class TestHandleDispatch:
    """Tests for the top-level handle() method that dispatches to sub-handlers."""

    @patch("aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter")
    def test_post_outcome_dispatches(self, mock_get_adapter, handler):
        """POST to /outcome path dispatches to _handle_record_outcome."""
        mock_get_adapter.return_value = MagicMock(ingest=MagicMock(return_value=True))
        mock_http = _mock_post_handler(_valid_outcome_body())

        result = handler.handle("POST", "/api/v1/decisions/dec_x/outcome", mock_http)
        assert result["status"] == 201

    def test_get_outcomes_dispatches(self, handler):
        """GET to /outcomes path dispatches to _handle_list_outcomes."""
        mock_http = MagicMock()
        result = handler.handle("GET", "/api/v1/decisions/dec_x/outcomes", mock_http)
        assert result["status"] == 200
        body = json.loads(result["body"])
        assert body["decision_id"] == "dec_x"

    def test_search_dispatches(self, handler):
        """GET to /outcomes/search dispatches to _handle_search_outcomes."""
        mock_http = _mock_get_handler("")
        result = handler.handle("GET", "/api/v1/outcomes/search", mock_http)
        assert result["status"] == 200

    def test_impact_dispatches(self, handler):
        """GET to /outcomes/impact dispatches to _handle_impact_analytics."""
        mock_http = MagicMock()
        result = handler.handle("GET", "/api/v1/outcomes/impact", mock_http)
        assert result["status"] == 200

    def test_unknown_path_returns_404(self, handler):
        """Unrecognized paths return 404."""
        mock_http = MagicMock()
        result = handler.handle("GET", "/api/v1/unknown/path", mock_http)
        assert result["status"] == 404


# ============================================================================
# POST /api/v1/decisions/{id}/outcome - Edge Cases
# ============================================================================


class TestRecordOutcomeEdgeCases:
    """Edge cases for recording outcomes."""

    def test_missing_decision_id_in_path(self, handler):
        """A path with no decision segment returns 400."""
        mock_http = _mock_post_handler(_valid_outcome_body())
        result = handler._handle_record_outcome("/api/v1/outcome", mock_http)
        assert result["status"] == 400

    def test_impact_score_zero_is_valid(self, handler):
        """An impact_score of exactly 0.0 is a valid value."""
        mock_http = _mock_post_handler(_valid_outcome_body(impact_score=0.0))
        with patch(
            "aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter"
        ) as m:
            m.return_value = MagicMock(ingest=MagicMock(return_value=True))
            result = handler._handle_record_outcome(
                "/api/v1/decisions/dec_z/outcome", mock_http
            )
        assert result["status"] == 201

    def test_impact_score_one_is_valid(self, handler):
        """An impact_score of exactly 1.0 is a valid value."""
        mock_http = _mock_post_handler(_valid_outcome_body(impact_score=1.0))
        with patch(
            "aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter"
        ) as m:
            m.return_value = MagicMock(ingest=MagicMock(return_value=True))
            result = handler._handle_record_outcome(
                "/api/v1/decisions/dec_z/outcome", mock_http
            )
        assert result["status"] == 201

    def test_impact_score_negative_rejected(self, handler):
        """A negative impact_score is rejected."""
        mock_http = _mock_post_handler(_valid_outcome_body(impact_score=-0.1))
        result = handler._handle_record_outcome(
            "/api/v1/decisions/dec_z/outcome", mock_http
        )
        assert result["status"] == 400

    def test_impact_score_not_a_number(self, handler):
        """A non-numeric impact_score is rejected."""
        mock_http = _mock_post_handler(_valid_outcome_body(impact_score="high"))
        result = handler._handle_record_outcome(
            "/api/v1/decisions/dec_z/outcome", mock_http
        )
        assert result["status"] == 400

    def test_oversized_body_rejected(self, handler):
        """A body exceeding MAX_BODY_SIZE returns 400."""
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": str(handler.MAX_BODY_SIZE + 1)}
        mock_http.rfile.read.return_value = b"x" * (handler.MAX_BODY_SIZE + 1)
        result = handler._handle_record_outcome(
            "/api/v1/decisions/dec_z/outcome", mock_http
        )
        assert result["status"] == 400

    def test_all_valid_outcome_types_accepted(self, handler):
        """Each value in VALID_OUTCOME_TYPES is accepted."""
        for otype in sorted(VALID_OUTCOME_TYPES):
            _outcome_store.clear()
            mock_http = _mock_post_handler(
                _valid_outcome_body(outcome_type=otype)
            )
            with patch(
                "aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter"
            ) as m:
                m.return_value = MagicMock(ingest=MagicMock(return_value=True))
                result = handler._handle_record_outcome(
                    "/api/v1/decisions/dec_z/outcome", mock_http
                )
            assert result["status"] == 201, f"outcome_type={otype} should be accepted"


# ============================================================================
# KM Adapter Integration
# ============================================================================


class TestKMAdapterIntegration:
    """Verify graceful handling when the KM adapter is unavailable."""

    def test_km_import_failure_does_not_block_recording(self, handler):
        """If the KM adapter import fails, the outcome is still recorded."""
        mock_http = _mock_post_handler(_valid_outcome_body())

        with patch(
            "aragora.server.handlers.governance.outcomes.get_outcome_adapter",
            side_effect=ImportError("No KM"),
            create=True,
        ):
            # Patch at the import site inside the handler method
            with patch.dict(
                "sys.modules",
                {"aragora.knowledge.mound.adapters.outcome_adapter": None},
            ):
                result = handler._handle_record_outcome(
                    "/api/v1/decisions/dec_km/outcome", mock_http
                )

        # Recording should still succeed (201) even when KM is unavailable
        assert result["status"] == 201
        assert len(_outcome_store) == 1


# ============================================================================
# Eviction
# ============================================================================


class TestEviction:
    """Verify the LRU eviction behaviour when the store is full."""

    def test_eviction_at_max_capacity(self, handler):
        """When the store exceeds MAX_OUTCOMES, the oldest entries are removed."""
        # Pre-fill the store to MAX_OUTCOMES
        for i in range(MAX_OUTCOMES):
            _outcome_store[f"pre_{i}"] = {
                "decision_id": "dec_old",
                "outcome_type": "success",
                "impact_score": 0.5,
            }
        assert len(_outcome_store) == MAX_OUTCOMES

        # Record one more outcome to trigger eviction
        mock_http = _mock_post_handler(_valid_outcome_body())
        with patch(
            "aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter"
        ) as m:
            m.return_value = MagicMock(ingest=MagicMock(return_value=True))
            result = handler._handle_record_outcome(
                "/api/v1/decisions/dec_new/outcome", mock_http
            )

        assert result["status"] == 201
        # Store should not exceed MAX_OUTCOMES
        assert len(_outcome_store) <= MAX_OUTCOMES
        # The first pre-filled entry should have been evicted
        assert "pre_0" not in _outcome_store


# ============================================================================
# Impact Analytics Aggregation
# ============================================================================


class TestImpactAnalyticsAggregation:
    """Detailed tests for the impact analytics endpoint."""

    def test_avg_impact_score_computed_correctly(self, handler):
        """The overall average impact score is correct."""
        _outcome_store["o1"] = {"outcome_type": "success", "impact_score": 0.9, "lessons_learned": ""}
        _outcome_store["o2"] = {"outcome_type": "failure", "impact_score": 0.1, "lessons_learned": ""}

        result = handler._handle_impact_analytics(MagicMock())
        body = json.loads(result["body"])
        assert body["avg_impact_score"] == 0.5

    def test_by_type_avg_impact(self, handler):
        """Per-type average impact scores are computed correctly."""
        _outcome_store["o1"] = {"outcome_type": "success", "impact_score": 0.8, "lessons_learned": ""}
        _outcome_store["o2"] = {"outcome_type": "success", "impact_score": 0.6, "lessons_learned": ""}

        result = handler._handle_impact_analytics(MagicMock())
        body = json.loads(result["body"])
        assert body["by_type"]["success"]["avg_impact"] == 0.7
        assert body["by_type"]["success"]["count"] == 2

    def test_top_lessons_sorted_by_impact(self, handler):
        """Top lessons are ordered by descending impact score."""
        _outcome_store["o1"] = {"outcome_type": "success", "impact_score": 0.3, "lessons_learned": "Low impact lesson"}
        _outcome_store["o2"] = {"outcome_type": "success", "impact_score": 0.9, "lessons_learned": "High impact lesson"}
        _outcome_store["o3"] = {"outcome_type": "partial", "impact_score": 0.6, "lessons_learned": "Mid impact lesson"}

        result = handler._handle_impact_analytics(MagicMock())
        body = json.loads(result["body"])

        lessons = body["top_lessons"]
        assert len(lessons) == 3
        assert lessons[0]["lesson"] == "High impact lesson"
        assert lessons[1]["lesson"] == "Mid impact lesson"
        assert lessons[2]["lesson"] == "Low impact lesson"
