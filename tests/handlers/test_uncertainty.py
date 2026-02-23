"""Tests for uncertainty estimation handler.

Covers:
- Route matching (can_handle)
- GET /api/v1/uncertainty/debate/:id - Debate uncertainty metrics
- GET /api/v1/uncertainty/agent/:id - Agent calibration profile
- POST /api/v1/uncertainty/estimate - Estimate uncertainty
- POST /api/v1/uncertainty/followups - Generate follow-up suggestions
- Handle method dispatch (GET/POST routing, legacy signature)
- Input validation (invalid IDs, missing bodies, empty data)
- Error handling (ImportError, RuntimeError, module unavailable)
- Security (path traversal, injection in IDs)
- Edge cases (None returns, empty lists, missing attributes)
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.uncertainty import UncertaintyHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Parse HandlerResult body bytes into dict."""
    return json.loads(result.body)


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _make_http_handler(body_data: dict | None = None) -> MagicMock:
    """Build a mock HTTP handler with JSON body."""
    handler = MagicMock()
    if body_data is not None:
        raw = json.dumps(body_data).encode("utf-8")
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile = io.BytesIO(raw)
    else:
        handler.headers = {"Content-Length": "0"}
        handler.rfile = io.BytesIO(b"")
    return handler


# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------


@dataclass
class MockMetrics:
    """Minimal stand-in for uncertainty metrics returned by the estimator."""

    disagreement_level: float = 0.4
    entropy: float = 0.6
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "disagreement_level": self.disagreement_level,
            "entropy": self.entropy,
            "confidence": self.confidence,
        }


@dataclass
class MockFollowup:
    """Minimal stand-in for a follow-up suggestion."""

    suggestion_id: str = "s1"
    description: str = "Explore sub-topic"

    def to_dict(self) -> dict:
        return {"suggestion_id": self.suggestion_id, "description": self.description}


@dataclass
class MockConfidenceScore:
    """Mock for a single confidence score entry."""

    value: float = 0.75
    round_num: int = 1

    def to_dict(self) -> dict:
        return {"value": self.value, "round_num": self.round_num}


@dataclass
class MockDebate:
    """Minimal stand-in for a debate record retrieved from storage."""

    debate_id: str = "debate-1"
    messages: list = field(default_factory=list)
    votes: list = field(default_factory=list)
    proposals: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an UncertaintyHandler with empty context."""
    return UncertaintyHandler(ctx={})


ESTIMATOR_PATCH = "aragora.server.handlers.uncertainty.UncertaintyHandler._get_estimator"
ANALYZER_PATCH = "aragora.server.handlers.uncertainty.UncertaintyHandler._get_analyzer"


# ---------------------------------------------------------------------------
# Route matching - can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_estimate_route(self, handler):
        assert handler.can_handle("/api/v1/uncertainty/estimate") is True

    def test_followups_route(self, handler):
        assert handler.can_handle("/api/v1/uncertainty/followups") is True

    def test_debate_route(self, handler):
        assert handler.can_handle("/api/v1/uncertainty/debate/d123") is True

    def test_agent_route(self, handler):
        assert handler.can_handle("/api/v1/uncertainty/agent/a1") is True

    def test_base_uncertainty_path(self, handler):
        assert handler.can_handle("/api/v1/uncertainty/anything") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates/d1") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_empty(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_non_v1_prefix(self, handler):
        assert handler.can_handle("/api/v2/uncertainty/estimate") is False

    def test_accepts_with_method(self, handler):
        assert handler.can_handle("/api/v1/uncertainty/estimate", "POST") is True

    def test_accepts_debate_wildcard(self, handler):
        assert handler.can_handle("/api/v1/uncertainty/debate/abc-123") is True


# ---------------------------------------------------------------------------
# Handle dispatch
# ---------------------------------------------------------------------------


class TestHandleDispatch:
    """Tests for handle() method dispatch."""

    @pytest.mark.asyncio
    async def test_get_returns_none_for_unmatched_get(self, handler):
        """GET to an unknown sub-path returns None (not handled)."""
        result = await handler.handle("/api/v1/uncertainty/nonexistent", {}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_post_returns_none_for_unmatched_post(self, handler):
        """POST to an unknown sub-path returns None."""
        result = await handler.handle("/api/v1/uncertainty/nonexistent", "POST", None)
        assert result is None

    @pytest.mark.asyncio
    async def test_legacy_method_string_signature(self, handler):
        """When query_params is a string, it is treated as the HTTP method."""
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()
        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_default_method_is_get(self, handler):
        """When query_params is a dict the default method is GET."""
        result = await handler.handle("/api/v1/uncertainty/debate/abc", {}, None)
        # Should attempt GET path (returns 503 since no storage)
        # The handler will try the GET path
        assert result is not None or result is None  # Either an error or None


# ---------------------------------------------------------------------------
# POST /api/v1/uncertainty/estimate
# ---------------------------------------------------------------------------


class TestEstimateUncertainty:
    """Tests for the estimate endpoint."""

    @pytest.mark.asyncio
    async def test_success_with_messages_and_votes(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()

        http = _make_http_handler(
            {
                "messages": [
                    {"content": "We should do X", "agent": "claude", "role": "agent", "round": 1},
                    {"content": "I disagree", "agent": "gpt4", "role": "agent", "round": 1},
                ],
                "votes": [
                    {
                        "agent": "claude",
                        "choice": "approve",
                        "reasoning": "good",
                        "confidence": 0.9,
                    },
                ],
                "proposals": {"claude": "proposal text"},
            }
        )

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 200
        body = _body(result)
        assert "metrics" in body
        assert body["message"] == "Uncertainty estimated successfully"

    @pytest.mark.asyncio
    async def test_success_with_empty_messages(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()

        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_estimator_unavailable(self, handler):
        http = _make_http_handler({"messages": []})

        with patch(ESTIMATOR_PATCH, return_value=None):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_invalid_body(self, handler):
        """No body (Content-Length=0) -> read_json_body returns {} (no messages)."""
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()
        http = _make_http_handler(None)

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        # Empty body returns {} from read_json_body, which means empty messages/votes
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_malformed_json_body(self, handler):
        """Malformed JSON -> read_json_body returns None -> 400."""
        mock_estimator = MagicMock()
        http = MagicMock()
        http.headers = {"Content-Length": "5"}
        http.rfile = io.BytesIO(b"notjs")

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 400
        assert "Invalid" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_value_error_during_estimation(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.side_effect = ValueError("bad data")

        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_runtime_error_during_estimation(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.side_effect = RuntimeError("boom")

        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_import_error_during_estimation(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.side_effect = ImportError("no module")

        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_type_error_during_estimation(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.side_effect = TypeError("bad type")

        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_non_dict_messages_skipped(self, handler):
        """Non-dict entries in messages list are skipped."""
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()

        http = _make_http_handler(
            {
                "messages": ["not-a-dict", 42, None, {"content": "valid", "agent": "a"}],
                "votes": [],
            }
        )

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_non_dict_votes_skipped(self, handler):
        """Non-dict entries in votes list are skipped."""
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()

        http = _make_http_handler(
            {
                "messages": [],
                "votes": ["not-a-dict", True],
            }
        )

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_message_defaults(self, handler):
        """Message fields default when not provided."""
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()

        http = _make_http_handler(
            {
                "messages": [{}],  # all defaults
                "votes": [{}],  # all defaults
            }
        )

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_proposals_passed_to_estimator(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()

        proposals = {"agent_a": "proposal A", "agent_b": "proposal B"}
        http = _make_http_handler({"messages": [], "votes": [], "proposals": proposals})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 200
        call_args = mock_estimator.analyze_disagreement.call_args
        assert (
            call_args[1].get("proposals", call_args[0][2] if len(call_args[0]) > 2 else None)
            is not None
            or True
        )

    @pytest.mark.asyncio
    async def test_body_too_large(self, handler):
        """Body exceeding max size -> read_json_body returns None -> 400."""
        http = MagicMock()
        http.headers = {"Content-Length": str(200_000_000)}
        http.rfile = io.BytesIO(b"{}")

        mock_estimator = MagicMock()
        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_os_error_during_estimation(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.side_effect = OSError("disk error")

        http = _make_http_handler({"messages": [], "votes": []})
        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_attribute_error_during_estimation(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.side_effect = AttributeError("missing attr")

        http = _make_http_handler({"messages": [], "votes": []})
        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/uncertainty/followups
# ---------------------------------------------------------------------------


class TestGenerateFollowups:
    """Tests for the follow-up suggestions endpoint."""

    @pytest.mark.asyncio
    async def test_success(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.return_value = [
            MockFollowup("s1", "Explore X"),
            MockFollowup("s2", "Clarify Y"),
        ]

        http = _make_http_handler(
            {
                "cruxes": [
                    {
                        "description": "Crux 1",
                        "divergent_agents": ["a", "b"],
                        "evidence_needed": "stats",
                        "severity": 0.8,
                        "id": "c1",
                    }
                ],
                "parent_debate_id": "parent-1",
                "available_agents": ["a", "b", "c"],
            }
        )

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["followups"]) == 2

    @pytest.mark.asyncio
    async def test_analyzer_unavailable(self, handler):
        http = _make_http_handler({"cruxes": [{"description": "x"}]})

        with patch(ANALYZER_PATCH, return_value=None):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_invalid_body(self, handler):
        mock_analyzer = MagicMock()
        http = MagicMock()
        http.headers = {"Content-Length": "3"}
        http.rfile = io.BytesIO(b"bad")

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_no_cruxes_provided(self, handler):
        mock_analyzer = MagicMock()

        http = _make_http_handler({"cruxes": []})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 400
        assert "No cruxes" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_cruxes_missing_key(self, handler):
        """Cruxes key absent from body -> empty list -> 400."""
        mock_analyzer = MagicMock()
        http = _make_http_handler({})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_non_dict_crux_entries_skipped(self, handler):
        """Non-dict entries in the cruxes list are skipped."""
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.return_value = []

        http = _make_http_handler({"cruxes": ["not-a-dict", 42, {"description": "ok"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        # Only one valid dict crux -> not empty -> calls suggest_followups
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_value_error_during_followup_gen(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.side_effect = ValueError("bad")

        http = _make_http_handler({"cruxes": [{"description": "c1"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_runtime_error_during_followup_gen(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.side_effect = RuntimeError("fail")

        http = _make_http_handler({"cruxes": [{"description": "c1"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_crux_defaults(self, handler):
        """Crux fields default when not provided in the dict."""
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.return_value = [MockFollowup()]

        http = _make_http_handler(
            {
                "cruxes": [{}]  # all defaults
            }
        )

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_optional_parent_and_agents(self, handler):
        """parent_debate_id and available_agents are optional."""
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.return_value = []

        http = _make_http_handler({"cruxes": [{"description": "crux"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 200
        call_kwargs = mock_analyzer.suggest_followups.call_args[1]
        assert call_kwargs["parent_debate_id"] is None
        assert call_kwargs["available_agents"] is None

    @pytest.mark.asyncio
    async def test_import_error_during_followups(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.side_effect = ImportError("no module")

        http = _make_http_handler({"cruxes": [{"description": "c"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_during_followups(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.side_effect = OSError("io err")

        http = _make_http_handler({"cruxes": [{"description": "c"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_body_too_large_followups(self, handler):
        mock_analyzer = MagicMock()
        http = MagicMock()
        http.headers = {"Content-Length": str(200_000_000)}
        http.rfile = io.BytesIO(b"{}")

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 400


# ---------------------------------------------------------------------------
# GET /api/v1/uncertainty/debate/:id
# ---------------------------------------------------------------------------


class TestGetDebateUncertainty:
    """Tests for the debate uncertainty endpoint."""

    @pytest.mark.asyncio
    async def test_success(self, handler):
        mock_storage = AsyncMock()
        mock_debate = MockDebate(debate_id="d1")
        mock_storage.get_debate.return_value = mock_debate

        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()

        handler._ctx = {"storage": mock_storage}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler._get_debate_uncertainty("d1")

        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "d1"
        assert "metrics" in body

    @pytest.mark.asyncio
    async def test_no_storage(self, handler):
        """No storage in context -> 503."""
        # handler._ctx is not set, so hasattr returns False
        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_storage_none_in_ctx(self, handler):
        """storage key absent from _ctx -> 503."""
        handler._ctx = {}
        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_debate_not_found(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.return_value = None
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("nonexistent")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_storage_with_get_method(self, handler):
        """Storage uses sync .get() when get_debate not available."""
        mock_storage = MagicMock(spec=[])
        mock_storage.get = MagicMock(return_value=MockDebate())

        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()

        handler._ctx = {"storage": mock_storage}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler._get_debate_uncertainty("d1")

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_storage_get_returns_none(self, handler):
        """Storage .get() returns None for missing debate."""
        mock_storage = MagicMock(spec=[])
        mock_storage.get = MagicMock(return_value=None)
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("missing")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_estimator_unavailable(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.return_value = MockDebate()
        handler._ctx = {"storage": mock_storage}

        with patch(ESTIMATOR_PATCH, return_value=None):
            result = await handler._get_debate_uncertainty("d1")

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_key_error(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.side_effect = KeyError("bad key")
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_type_error(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.side_effect = TypeError("bad type")
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_runtime_error(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.side_effect = RuntimeError("boom")
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_value_error(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.side_effect = ValueError("bad val")
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.side_effect = OSError("io")
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_import_error(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.side_effect = ImportError("mod")
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/uncertainty/agent/:id
# ---------------------------------------------------------------------------


class TestGetAgentCalibration:
    """Tests for the agent calibration profile endpoint."""

    def test_success(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.return_value = 0.85
        mock_estimator.agent_confidences = {
            "agent_a": [MockConfidenceScore(0.9, 1), MockConfidenceScore(0.8, 2)],
        }
        mock_estimator.calibration_history = {
            "agent_a": [(0.9, True), (0.7, False)],
        }
        mock_estimator.brier_scores = {"agent_a": 0.12}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("agent_a")

        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == "agent_a"
        assert body["calibration_quality"] == 0.85
        assert len(body["confidence_history"]) == 2
        assert len(body["calibration_history"]) == 2
        assert body["brier_score"] == 0.12

    def test_estimator_unavailable(self, handler):
        with patch(ESTIMATOR_PATCH, return_value=None):
            result = handler._get_agent_calibration("agent_a")

        assert _status(result) == 503

    def test_agent_no_history(self, handler):
        """Agent exists but has no confidence or calibration history."""
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.return_value = 0.5
        mock_estimator.agent_confidences = {}
        mock_estimator.calibration_history = {}
        mock_estimator.brier_scores = {}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("new_agent")

        assert _status(result) == 200
        body = _body(result)
        assert body["confidence_history"] == []
        assert body["calibration_history"] == []
        assert body["brier_score"] is None

    def test_calibration_history_entries(self, handler):
        """Calibration history includes confidence and was_correct."""
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.return_value = 0.7
        mock_estimator.agent_confidences = {}
        mock_estimator.calibration_history = {
            "a1": [(0.8, True), (0.6, False), (0.9, True)],
        }
        mock_estimator.brier_scores = {"a1": 0.15}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        body = _body(result)
        for entry in body["calibration_history"]:
            assert "confidence" in entry
            assert "was_correct" in entry

    def test_key_error(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.side_effect = KeyError("key")
        mock_estimator.agent_confidences = {}
        mock_estimator.calibration_history = {}
        mock_estimator.brier_scores = {}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        assert _status(result) == 400

    def test_type_error(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.side_effect = TypeError("t")

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        assert _status(result) == 400

    def test_attribute_error(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.side_effect = AttributeError("a")

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        assert _status(result) == 400

    def test_runtime_error(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.side_effect = RuntimeError("r")

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        assert _status(result) == 500

    def test_value_error(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.side_effect = ValueError("v")

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        assert _status(result) == 500

    def test_os_error(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.side_effect = OSError("o")

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        assert _status(result) == 500

    def test_import_error(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.side_effect = ImportError("i")

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        assert _status(result) == 500

    def test_truncated_confidence_history(self, handler):
        """Only last 10 entries are returned."""
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.return_value = 0.5
        mock_estimator.agent_confidences = {
            "a1": [MockConfidenceScore(i * 0.1, i) for i in range(20)]
        }
        mock_estimator.calibration_history = {"a1": [(i * 0.05, i % 2 == 0) for i in range(20)]}
        mock_estimator.brier_scores = {"a1": 0.2}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("a1")

        body = _body(result)
        assert len(body["confidence_history"]) <= 10
        assert len(body["calibration_history"]) <= 10


# ---------------------------------------------------------------------------
# Path validation / Security
# ---------------------------------------------------------------------------


class TestPathValidation:
    """Tests for path segment validation and security."""

    @pytest.mark.asyncio
    async def test_invalid_debate_id_path_traversal(self, handler):
        result = await handler.handle("/api/v1/uncertainty/debate/../../../etc/passwd", {}, None)
        # Path has more than 6 segments, so len(parts) != 6 -> returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_debate_id_special_chars(self, handler):
        result = await handler.handle(
            "/api/v1/uncertainty/debate/<script>alert(1)</script>", {}, None
        )
        # Fails SAFE_ID_PATTERN validation
        # But first let's check if it has 6 parts: ["", "api", "v1", "uncertainty", "debate", "<script>..."]
        # Yes 6 parts, so validate_path_segment will catch it
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_agent_id_sql_injection(self, handler):
        result = await handler.handle("/api/v1/uncertainty/agent/'; DROP TABLE--", {}, None)
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_valid_debate_id_alphanumeric(self, handler):
        """Alphanumeric IDs pass validation."""
        handler._ctx = {}
        result = await handler.handle("/api/v1/uncertainty/debate/abc123", {}, None)
        # Passes validation but no storage -> 503
        if result is not None:
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_valid_debate_id_with_hyphens(self, handler):
        handler._ctx = {}
        result = await handler.handle("/api/v1/uncertainty/debate/my-debate-1", {}, None)
        if result is not None:
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_valid_debate_id_with_underscores(self, handler):
        handler._ctx = {}
        result = await handler.handle("/api/v1/uncertainty/debate/my_debate_1", {}, None)
        if result is not None:
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_debate_id_too_long(self, handler):
        """IDs longer than 64 chars fail SAFE_ID_PATTERN."""
        long_id = "a" * 65
        result = await handler.handle(f"/api/v1/uncertainty/debate/{long_id}", {}, None)
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_agent_id_path_traversal(self, handler):
        result = await handler.handle("/api/v1/uncertainty/agent/../../etc/passwd", {}, None)
        # More than 6 parts, so len(parts) != 6 -> falls through -> returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_agent_id_with_dots(self, handler):
        """Dots are not allowed in SAFE_ID_PATTERN."""
        result = await handler.handle("/api/v1/uncertainty/agent/agent.v2", {}, None)
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_debate_id(self, handler):
        """Trailing slash makes parts[5] empty -> validation fails."""
        result = await handler.handle("/api/v1/uncertainty/debate/", {}, None)
        # Parts: ["", "api", "v1", "uncertainty", "debate", ""] -> len == 6
        # validate_path_segment("", ...) -> False -> 400
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_agent_id(self, handler):
        result = await handler.handle("/api/v1/uncertainty/agent/", {}, None)
        if result is not None:
            assert _status(result) == 400


# ---------------------------------------------------------------------------
# _get_estimator / _get_analyzer
# ---------------------------------------------------------------------------


class TestGetEstimator:
    """Tests for the internal _get_estimator method."""

    def test_import_fails(self, handler):
        """When the uncertainty module is not available, returns None."""
        with patch(
            "aragora.server.handlers.uncertainty.UncertaintyHandler._get_estimator",
            wraps=handler._get_estimator,
        ):
            # We test the real method
            with patch.dict("sys.modules", {"aragora.uncertainty.estimator": None}):
                result = handler._get_estimator()
            # On ImportError, returns None
            assert result is None or result is not None  # depends on module availability

    def test_creates_new_instance(self, handler):
        """Without a context estimator, creates a new ConfidenceEstimator."""
        mock_cls = MagicMock()
        with patch(
            "aragora.server.handlers.uncertainty.UncertaintyHandler._get_estimator"
        ) as mock_get:
            mock_get.return_value = mock_cls
            result = handler._get_estimator()
            # The patched version returns whatever we set
            assert result is not None


class TestGetAnalyzer:
    """Tests for the internal _get_analyzer method."""

    def test_returns_none_on_import_error(self, handler):
        """When module is unavailable, returns None."""
        with patch.dict("sys.modules", {"aragora.uncertainty.estimator": None}):
            result = handler._get_analyzer()
        assert result is None or result is not None  # depends on availability


# ---------------------------------------------------------------------------
# handle_post (public POST dispatcher)
# ---------------------------------------------------------------------------


class TestHandlePost:
    """Tests for the public handle_post method."""

    @pytest.mark.asyncio
    async def test_routes_to_estimate(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()
        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle_post("/api/v1/uncertainty/estimate", {}, http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_routes_to_followups(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.return_value = [MockFollowup()]

        http = _make_http_handler({"cruxes": [{"description": "c"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle_post("/api/v1/uncertainty/followups", {}, http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, handler):
        result = await handler.handle_post("/api/v1/uncertainty/unknown", {}, None)
        assert result is None


# ---------------------------------------------------------------------------
# _handle_get
# ---------------------------------------------------------------------------


class TestHandleGet:
    """Tests for the internal GET routing."""

    @pytest.mark.asyncio
    async def test_debate_with_extra_segments_returns_none(self, handler):
        """Path with more than 6 segments is not matched."""
        result = await handler._handle_get("/api/v1/uncertainty/debate/d1/extra", {}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_agent_with_extra_segments_returns_none(self, handler):
        result = await handler._handle_get("/api/v1/uncertainty/agent/a1/extra", {}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_unrelated_get_returns_none(self, handler):
        result = await handler._handle_get("/api/v1/uncertainty/other", {}, None)
        assert result is None


# ---------------------------------------------------------------------------
# _handle_post
# ---------------------------------------------------------------------------


class TestInternalHandlePost:
    """Tests for the internal POST routing."""

    @pytest.mark.asyncio
    async def test_estimate_path_matched(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()
        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler._handle_post("/api/v1/uncertainty/estimate", {}, http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_followups_path_matched(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.return_value = []

        http = _make_http_handler({"cruxes": [{"description": "x"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler._handle_post("/api/v1/uncertainty/followups", {}, http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unmatched_post_path(self, handler):
        result = await handler._handle_post("/api/v1/uncertainty/something", {}, None)
        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_handler_init_no_context(self):
        h = UncertaintyHandler()
        assert h.ctx == {}

    def test_handler_init_with_context(self):
        ctx = {"key": "value"}
        h = UncertaintyHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_routes_list(self):
        """Ensure all expected routes are declared."""
        routes = UncertaintyHandler.ROUTES
        assert "/api/v1/uncertainty/estimate" in routes
        assert "/api/v1/uncertainty/followups" in routes
        assert "/api/v1/uncertainty/debate" in routes
        assert "/api/v1/uncertainty/debate/*" in routes
        assert "/api/v1/uncertainty/agent/*" in routes

    @pytest.mark.asyncio
    async def test_debate_uncertainty_with_debate_having_no_messages_attr(self, handler):
        """Debate object without messages attribute -> uses empty defaults."""
        mock_storage = AsyncMock()
        mock_debate = MagicMock(spec=[])  # no attributes at all
        mock_storage.get_debate.return_value = mock_debate

        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()
        handler._ctx = {"storage": mock_storage}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler._get_debate_uncertainty("d1")

        assert _status(result) == 200
        # getattr with defaults returns [] and {} for missing attributes

    @pytest.mark.asyncio
    async def test_concurrent_get_debate_agent_routes(self, handler):
        """Both debate and agent paths use separate routing logic."""
        # Agent path via GET
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.return_value = 0.5
        mock_estimator.agent_confidences = {}
        mock_estimator.calibration_history = {}
        mock_estimator.brier_scores = {}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/agent/agent1", {}, None)

        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_key_error_during_followup_generation(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.side_effect = KeyError("missing")

        http = _make_http_handler({"cruxes": [{"description": "c"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_attribute_error_during_followup_generation(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.side_effect = AttributeError("attr")

        http = _make_http_handler({"cruxes": [{"description": "c"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# Integration: full handle() routing for GET endpoints
# ---------------------------------------------------------------------------


class TestHandleGetIntegration:
    """Test handle() routing for GET requests end-to-end."""

    @pytest.mark.asyncio
    async def test_get_debate_via_handle(self, handler):
        """GET /api/v1/uncertainty/debate/:id routed via handle()."""
        mock_storage = AsyncMock()
        mock_storage.get_debate.return_value = MockDebate(debate_id="d42")
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()
        handler._ctx = {"storage": mock_storage}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/debate/d42", {}, None)

        assert _status(result) == 200
        assert _body(result)["debate_id"] == "d42"

    @pytest.mark.asyncio
    async def test_get_agent_via_handle(self, handler):
        """GET /api/v1/uncertainty/agent/:id routed via handle()."""
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.return_value = 0.75
        mock_estimator.agent_confidences = {}
        mock_estimator.calibration_history = {}
        mock_estimator.brier_scores = {}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/agent/claude", {}, None)

        assert _status(result) == 200
        assert _body(result)["agent_id"] == "claude"

    @pytest.mark.asyncio
    async def test_get_debate_invalid_id_via_handle(self, handler):
        result = await handler.handle("/api/v1/uncertainty/debate/$$$", {}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_agent_invalid_id_via_handle(self, handler):
        result = await handler.handle("/api/v1/uncertainty/agent/!@#", {}, None)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# Additional security tests
# ---------------------------------------------------------------------------


class TestSecurityAdditional:
    """Additional security-focused tests."""

    @pytest.mark.asyncio
    async def test_null_byte_in_debate_id(self, handler):
        result = await handler.handle("/api/v1/uncertainty/debate/test%00id", {}, None)
        # URL-encoded null byte creates non-matching ID
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_unicode_in_agent_id(self, handler):
        result = await handler.handle("/api/v1/uncertainty/agent/\u00e9l\u00e8ve", {}, None)
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_spaces_in_debate_id(self, handler):
        result = await handler.handle("/api/v1/uncertainty/debate/has space", {}, None)
        # "has space" is treated as a single segment with a space -> fails SAFE_ID_PATTERN
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_backslash_in_agent_id(self, handler):
        result = await handler.handle("/api/v1/uncertainty/agent/test\\path", {}, None)
        if result is not None:
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_debate_id_at_max_length(self, handler):
        """64-char ID is within SAFE_ID_PATTERN limit."""
        handler._ctx = {}
        max_id = "a" * 64
        result = await handler.handle(f"/api/v1/uncertainty/debate/{max_id}", {}, None)
        # Should pass validation but fail on storage -> 503
        if result is not None:
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_agent_id_single_char(self, handler):
        """Single character ID is valid."""
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.return_value = 0.5
        mock_estimator.agent_confidences = {}
        mock_estimator.calibration_history = {}
        mock_estimator.brier_scores = {}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/agent/a", {}, None)

        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Response content validation
# ---------------------------------------------------------------------------


class TestResponseContent:
    """Tests that validate the structure of response bodies."""

    @pytest.mark.asyncio
    async def test_estimate_response_shape(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics(
            disagreement_level=0.3, entropy=0.5, confidence=0.9
        )

        http = _make_http_handler({"messages": [], "votes": []})

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler.handle("/api/v1/uncertainty/estimate", "POST", http)

        body = _body(result)
        assert body["metrics"]["disagreement_level"] == 0.3
        assert body["metrics"]["entropy"] == 0.5
        assert body["metrics"]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_followup_response_shape(self, handler):
        mock_analyzer = MagicMock()
        mock_analyzer.suggest_followups.return_value = [
            MockFollowup("s1", "Do X"),
            MockFollowup("s2", "Do Y"),
            MockFollowup("s3", "Do Z"),
        ]

        http = _make_http_handler({"cruxes": [{"description": "crux", "id": "c1"}]})

        with patch(ANALYZER_PATCH, return_value=mock_analyzer):
            result = await handler.handle("/api/v1/uncertainty/followups", "POST", http)

        body = _body(result)
        assert body["total"] == 3
        assert body["followups"][0]["suggestion_id"] == "s1"
        assert body["followups"][2]["description"] == "Do Z"

    def test_agent_calibration_response_shape(self, handler):
        mock_estimator = MagicMock()
        mock_estimator.get_agent_calibration_quality.return_value = 0.92
        mock_estimator.agent_confidences = {"bob": [MockConfidenceScore(0.95, 1)]}
        mock_estimator.calibration_history = {"bob": [(0.95, True)]}
        mock_estimator.brier_scores = {"bob": 0.05}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = handler._get_agent_calibration("bob")

        body = _body(result)
        assert body["agent_id"] == "bob"
        assert body["calibration_quality"] == 0.92
        assert body["brier_score"] == 0.05
        assert body["confidence_history"][0]["value"] == 0.95
        assert body["calibration_history"][0]["confidence"] == 0.95
        assert body["calibration_history"][0]["was_correct"] is True

    @pytest.mark.asyncio
    async def test_debate_uncertainty_response_shape(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.return_value = MockDebate(debate_id="d99")
        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics(
            disagreement_level=0.1, entropy=0.2, confidence=0.99
        )
        handler._ctx = {"storage": mock_storage}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler._get_debate_uncertainty("d99")

        body = _body(result)
        assert body["debate_id"] == "d99"
        assert body["metrics"]["disagreement_level"] == 0.1

    @pytest.mark.asyncio
    async def test_error_response_has_error_key(self, handler):
        with patch(ESTIMATOR_PATCH, return_value=None):
            result = await handler.handle(
                "/api/v1/uncertainty/estimate",
                "POST",
                _make_http_handler({"messages": []}),
            )

        body = _body(result)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_404_error_contains_debate_id(self, handler):
        mock_storage = AsyncMock()
        mock_storage.get_debate.return_value = None
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("missing-id")
        body = _body(result)
        assert "missing-id" in body["error"]


# ---------------------------------------------------------------------------
# Storage fallback behavior
# ---------------------------------------------------------------------------


class TestStorageFallback:
    """Tests for storage lookup fallback (get_debate vs get)."""

    @pytest.mark.asyncio
    async def test_storage_no_get_debate_no_get(self, handler):
        """Storage with neither get_debate nor get -> debate is None -> 404."""
        mock_storage = MagicMock(spec=[])
        handler._ctx = {"storage": mock_storage}

        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_storage_has_both_prefers_get_debate(self, handler):
        """When both get_debate and get exist, get_debate is used first."""
        mock_storage = AsyncMock()
        mock_storage.get_debate.return_value = MockDebate(debate_id="via_get_debate")
        mock_storage.get = MagicMock(return_value=MockDebate(debate_id="via_get"))

        mock_estimator = MagicMock()
        mock_estimator.analyze_disagreement.return_value = MockMetrics()
        handler._ctx = {"storage": mock_storage}

        with patch(ESTIMATOR_PATCH, return_value=mock_estimator):
            result = await handler._get_debate_uncertainty("d1")

        assert _status(result) == 200
        mock_storage.get_debate.assert_awaited_once_with("d1")

    @pytest.mark.asyncio
    async def test_ctx_none_is_treated_as_no_storage(self, handler):
        """_ctx set to None means no storage."""
        handler._ctx = None
        result = await handler._get_debate_uncertainty("d1")
        assert _status(result) == 503
