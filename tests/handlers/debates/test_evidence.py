"""Tests for evidence operations handler mixin.

Tests the evidence and analysis API endpoints including:
- GET /api/v1/debates/{id}/impasse - Detect debate impasse
- GET /api/v1/debates/{id}/convergence - Get convergence status
- GET /api/v1/debates/{id}/verification-report - Get verification report
- GET /api/v1/debates/{id}/summary - Get debate summary
- GET /api/v1/debates/{id}/citations - Get evidence citations
- GET /api/v1/debates/{id}/evidence - Get comprehensive evidence trail
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.exceptions import (
    DatabaseError,
    RecordNotFoundError,
    StorageError,
)


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    return result.status_code


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_storage():
    """Create a mock storage with configurable get_debate return."""
    storage = MagicMock()
    storage.get_debate.return_value = None
    return storage


@pytest.fixture
def evidence_mixin(mock_storage):
    """Create evidence mixin instance backed by mock storage."""
    from aragora.server.handlers.debates.evidence import EvidenceOperationsMixin

    class TestHandler(EvidenceOperationsMixin):
        """Test handler that includes the mixin."""

        def __init__(self, storage):
            self._storage = storage
            self.ctx: dict[str, Any] = {}

        def get_storage(self):
            return self._storage

    return TestHandler(mock_storage)


@pytest.fixture
def evidence_mixin_no_storage():
    """Create evidence mixin with no storage (returns None)."""
    from aragora.server.handlers.debates.evidence import EvidenceOperationsMixin

    class TestHandler(EvidenceOperationsMixin):
        def __init__(self):
            self.ctx: dict[str, Any] = {}

        def get_storage(self):
            return None

    return TestHandler()


@pytest.fixture
def mock_http_handler():
    """Create a minimal mock HTTP handler."""
    handler = MagicMock()
    handler.headers = {"Content-Length": "0"}
    return handler


# =============================================================================
# Sample debate data factories
# =============================================================================


def _make_debate(
    *,
    consensus_reached: bool = False,
    critiques: list | None = None,
    convergence_status: str = "unknown",
    convergence_similarity: float = 0.0,
    rounds_used: int = 0,
    verification_results: dict | None = None,
    verification_bonuses: dict | None = None,
    winner: str | None = None,
    task: str = "Test debate task",
    final_answer: str = "Test conclusion",
    confidence: float = 0.0,
    grounded_verdict: Any = None,
) -> dict:
    """Build a debate dict with sensible defaults."""
    return {
        "consensus_reached": consensus_reached,
        "critiques": critiques or [],
        "convergence_status": convergence_status,
        "convergence_similarity": convergence_similarity,
        "rounds_used": rounds_used,
        "verification_results": verification_results or {},
        "verification_bonuses": verification_bonuses or {},
        "winner": winner,
        "task": task,
        "final_answer": final_answer,
        "confidence": confidence,
        "grounded_verdict": grounded_verdict,
    }


# =============================================================================
# Impasse Detection Tests
# =============================================================================


class TestGetImpasse:
    """Tests for _get_impasse endpoint."""

    def test_no_storage_returns_503(self, evidence_mixin_no_storage, mock_http_handler):
        """Storage unavailable returns 503."""
        result = evidence_mixin_no_storage._get_impasse(mock_http_handler, "debate-1")
        assert _status(result) == 503
        assert "storage" in _body(result).get("error", "").lower()

    def test_debate_not_found_returns_404(self, evidence_mixin, mock_storage, mock_http_handler):
        """Non-existent debate returns 404."""
        mock_storage.get_debate.return_value = None
        result = evidence_mixin._get_impasse(mock_http_handler, "no-such-debate")
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    def test_no_impasse_all_indicators_false(self, evidence_mixin, mock_storage, mock_http_handler):
        """Debate with consensus and no high-severity critiques is not an impasse."""
        mock_storage.get_debate.return_value = _make_debate(
            consensus_reached=True,
            critiques=[{"severity": 0.3}, {"severity": 0.5}],
        )
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-1"
        assert body["is_impasse"] is False
        assert body["indicators"]["no_convergence"] is False
        assert body["indicators"]["high_severity_critiques"] is False
        assert body["indicators"]["repeated_critiques"] is False

    def test_impasse_detected_two_indicators(self, evidence_mixin, mock_storage, mock_http_handler):
        """Impasse when 2+ indicators are true (no_convergence + high_severity)."""
        mock_storage.get_debate.return_value = _make_debate(
            consensus_reached=False,
            critiques=[{"severity": 0.9}],
        )
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["is_impasse"] is True
        assert body["indicators"]["no_convergence"] is True
        assert body["indicators"]["high_severity_critiques"] is True

    def test_no_impasse_only_one_indicator(self, evidence_mixin, mock_storage, mock_http_handler):
        """Not an impasse with only one indicator true."""
        mock_storage.get_debate.return_value = _make_debate(
            consensus_reached=False,
            critiques=[{"severity": 0.3}],
        )
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        assert _status(result) == 200
        body = _body(result)
        # Only no_convergence is True, high_severity is False
        assert body["is_impasse"] is False
        assert body["indicators"]["no_convergence"] is True
        assert body["indicators"]["high_severity_critiques"] is False

    def test_impasse_empty_critiques(self, evidence_mixin, mock_storage, mock_http_handler):
        """Empty critiques list means no high-severity or repeated critiques."""
        mock_storage.get_debate.return_value = _make_debate(
            consensus_reached=False,
            critiques=[],
        )
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["indicators"]["high_severity_critiques"] is False
        assert body["indicators"]["repeated_critiques"] is False
        # Only no_convergence is True, so not an impasse
        assert body["is_impasse"] is False

    def test_impasse_missing_critiques_key(self, evidence_mixin, mock_storage, mock_http_handler):
        """Debate dict without critiques key defaults to empty list."""
        mock_storage.get_debate.return_value = {"consensus_reached": False}
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["indicators"]["high_severity_critiques"] is False

    def test_impasse_severity_boundary_at_0_7(self, evidence_mixin, mock_storage, mock_http_handler):
        """Severity exactly 0.7 does not trigger high_severity (requires > 0.7)."""
        mock_storage.get_debate.return_value = _make_debate(
            consensus_reached=False,
            critiques=[{"severity": 0.7}],
        )
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        body = _body(result)
        assert body["indicators"]["high_severity_critiques"] is False

    def test_impasse_severity_above_0_7(self, evidence_mixin, mock_storage, mock_http_handler):
        """Severity above 0.7 triggers high_severity."""
        mock_storage.get_debate.return_value = _make_debate(
            consensus_reached=False,
            critiques=[{"severity": 0.71}],
        )
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        body = _body(result)
        assert body["indicators"]["high_severity_critiques"] is True

    def test_impasse_critique_missing_severity(self, evidence_mixin, mock_storage, mock_http_handler):
        """Critique without severity key defaults to 0 (not high severity)."""
        mock_storage.get_debate.return_value = _make_debate(
            consensus_reached=False,
            critiques=[{}],
        )
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        body = _body(result)
        assert body["indicators"]["high_severity_critiques"] is False

    def test_impasse_storage_exception_handled(self, evidence_mixin, mock_storage, mock_http_handler):
        """Unexpected storage exception is caught by handle_errors decorator."""
        mock_storage.get_debate.side_effect = RuntimeError("DB down")
        result = evidence_mixin._get_impasse(mock_http_handler, "debate-1")
        assert _status(result) == 500


# =============================================================================
# Convergence Status Tests
# =============================================================================


class TestGetConvergence:
    """Tests for _get_convergence endpoint."""

    def test_no_storage_returns_503(self, evidence_mixin_no_storage, mock_http_handler):
        """Storage unavailable returns 503."""
        result = evidence_mixin_no_storage._get_convergence(mock_http_handler, "debate-1")
        assert _status(result) == 503

    def test_debate_not_found_returns_404(self, evidence_mixin, mock_storage, mock_http_handler):
        """Non-existent debate returns 404."""
        mock_storage.get_debate.return_value = None
        result = evidence_mixin._get_convergence(mock_http_handler, "no-such-debate")
        assert _status(result) == 404

    def test_convergence_success(self, evidence_mixin, mock_storage, mock_http_handler):
        """Successful convergence status retrieval."""
        mock_storage.get_debate.return_value = _make_debate(
            convergence_status="converging",
            convergence_similarity=0.85,
            consensus_reached=True,
            rounds_used=3,
        )
        result = evidence_mixin._get_convergence(mock_http_handler, "debate-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-1"
        assert body["convergence_status"] == "converging"
        assert body["convergence_similarity"] == 0.85
        assert body["consensus_reached"] is True
        assert body["rounds_used"] == 3

    def test_convergence_defaults(self, evidence_mixin, mock_storage, mock_http_handler):
        """Missing fields use sensible defaults."""
        # Use a non-empty dict so `if not debate:` passes
        mock_storage.get_debate.return_value = {"id": "debate-1"}
        result = evidence_mixin._get_convergence(mock_http_handler, "debate-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["convergence_status"] == "unknown"
        assert body["convergence_similarity"] == 0.0
        assert body["consensus_reached"] is False
        assert body["rounds_used"] == 0

    def test_convergence_storage_exception(self, evidence_mixin, mock_storage, mock_http_handler):
        """Storage exception caught by handle_errors."""
        mock_storage.get_debate.side_effect = ValueError("bad data")
        result = evidence_mixin._get_convergence(mock_http_handler, "debate-1")
        assert _status(result) in (400, 500)


# =============================================================================
# Verification Report Tests
# =============================================================================


class TestGetVerificationReport:
    """Tests for _get_verification_report endpoint."""

    def test_no_storage_returns_503(self, evidence_mixin_no_storage, mock_http_handler):
        """Storage unavailable returns 503."""
        result = evidence_mixin_no_storage._get_verification_report(mock_http_handler, "d-1")
        assert _status(result) == 503

    def test_debate_not_found_returns_404(self, evidence_mixin, mock_storage, mock_http_handler):
        """Non-existent debate returns 404."""
        mock_storage.get_debate.return_value = None
        result = evidence_mixin._get_verification_report(mock_http_handler, "d-1")
        assert _status(result) == 404

    def test_verification_report_success(self, evidence_mixin, mock_storage, mock_http_handler):
        """Verification report with results and bonuses."""
        mock_storage.get_debate.return_value = _make_debate(
            verification_results={"agent_a": 3, "agent_b": 0, "agent_c": 2},
            verification_bonuses={"agent_a": 0.15, "agent_b": 0.0, "agent_c": 0.1},
            winner="agent_a",
            consensus_reached=True,
        )
        result = evidence_mixin._get_verification_report(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "d-1"
        assert body["verification_enabled"] is True
        assert body["verification_results"]["agent_a"] == 3
        assert body["verification_bonuses"]["agent_a"] == 0.15
        assert body["summary"]["total_verified_claims"] == 5  # 3 + 0 + 2
        assert body["summary"]["agents_with_verified_claims"] == 2  # agent_a and agent_c
        assert body["summary"]["total_bonus_applied"] == 0.25
        assert body["winner"] == "agent_a"
        assert body["consensus_reached"] is True

    def test_verification_report_no_results(self, evidence_mixin, mock_storage, mock_http_handler):
        """Verification report with empty results."""
        mock_storage.get_debate.return_value = _make_debate(
            verification_results={},
            verification_bonuses={},
        )
        result = evidence_mixin._get_verification_report(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["verification_enabled"] is False
        assert body["summary"]["total_verified_claims"] == 0
        assert body["summary"]["agents_with_verified_claims"] == 0
        assert body["summary"]["total_bonus_applied"] == 0.0

    def test_verification_report_defaults(self, evidence_mixin, mock_storage, mock_http_handler):
        """Missing verification keys default to empty dicts."""
        mock_storage.get_debate.return_value = {"id": "d-1"}
        result = evidence_mixin._get_verification_report(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["verification_enabled"] is False
        assert body["winner"] is None
        assert body["consensus_reached"] is False

    def test_verification_bonus_rounding(self, evidence_mixin, mock_storage, mock_http_handler):
        """Total bonus is rounded to 3 decimal places."""
        mock_storage.get_debate.return_value = _make_debate(
            verification_results={"a": 1},
            verification_bonuses={"a": 0.1, "b": 0.2, "c": 0.3333},
        )
        result = evidence_mixin._get_verification_report(mock_http_handler, "d-1")
        body = _body(result)
        # 0.1 + 0.2 + 0.3333 = 0.6333 -> round(0.6333, 3) = 0.633
        assert body["summary"]["total_bonus_applied"] == 0.633

    def test_verification_storage_exception(self, evidence_mixin, mock_storage, mock_http_handler):
        """Storage exception caught by handle_errors."""
        mock_storage.get_debate.side_effect = RuntimeError("DB error")
        result = evidence_mixin._get_verification_report(mock_http_handler, "d-1")
        assert _status(result) == 500


# =============================================================================
# Debate Summary Tests
# =============================================================================


class TestGetSummary:
    """Tests for _get_summary endpoint."""

    def test_no_storage_returns_503(self, evidence_mixin_no_storage, mock_http_handler):
        """Storage unavailable returns 503."""
        result = evidence_mixin_no_storage._get_summary(mock_http_handler, "d-1")
        assert _status(result) == 503

    def test_debate_not_found_returns_404(self, evidence_mixin, mock_storage, mock_http_handler):
        """Non-existent debate returns 404."""
        mock_storage.get_debate.return_value = None
        result = evidence_mixin._get_summary(mock_http_handler, "d-1")
        assert _status(result) == 404

    def test_summary_success(self, evidence_mixin, mock_storage, mock_http_handler):
        """Successful summary generation."""
        mock_storage.get_debate.return_value = _make_debate(
            task="Rate limiter design",
            consensus_reached=True,
            confidence=0.92,
        )

        mock_summary = MagicMock()
        mock_summary.to_dict.return_value = {
            "one_liner": "Sliding window is the recommended approach",
            "key_points": ["Low memory overhead", "Fair distribution"],
            "agreement_areas": ["Sliding window wins"],
            "disagreement_areas": [],
            "confidence": 0.92,
            "confidence_label": "high",
            "consensus_strength": "strong",
            "next_steps": ["Implement prototype"],
            "caveats": [],
            "rounds_used": 3,
            "agents_participated": 4,
            "duration_seconds": 45.0,
        }

        with patch(
            "aragora.debate.summarizer.summarize_debate",
            return_value=mock_summary,
        ):
            result = evidence_mixin._get_summary(mock_http_handler, "d-1")
            assert _status(result) == 200
            body = _body(result)
            assert body["debate_id"] == "d-1"
            assert body["task"] == "Rate limiter design"
            assert body["consensus_reached"] is True
            assert body["confidence"] == 0.92
            assert body["summary"]["one_liner"] == "Sliding window is the recommended approach"

    def test_summary_defaults_for_missing_fields(self, evidence_mixin, mock_storage, mock_http_handler):
        """Missing task and confidence use defaults."""
        mock_storage.get_debate.return_value = {"id": "d-1"}

        mock_summary = MagicMock()
        mock_summary.to_dict.return_value = {"one_liner": "Summary"}

        with patch(
            "aragora.debate.summarizer.summarize_debate",
            return_value=mock_summary,
        ):
            result = evidence_mixin._get_summary(mock_http_handler, "d-1")
            assert _status(result) == 200
            body = _body(result)
            assert body["task"] == ""
            assert body["confidence"] == 0.0
            assert body["consensus_reached"] is False

    def test_summary_summarizer_error(self, evidence_mixin, mock_storage, mock_http_handler):
        """Exception in summarize_debate is caught by handle_errors."""
        mock_storage.get_debate.return_value = _make_debate()

        with patch(
            "aragora.debate.summarizer.summarize_debate",
            side_effect=RuntimeError("Summarizer failed"),
        ):
            result = evidence_mixin._get_summary(mock_http_handler, "d-1")
            assert _status(result) == 500


# =============================================================================
# Evidence Citations Tests
# =============================================================================


class TestGetCitations:
    """Tests for _get_citations endpoint."""

    def test_no_storage_returns_503(self, evidence_mixin_no_storage, mock_http_handler):
        """Storage unavailable returns 503."""
        result = evidence_mixin_no_storage._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 503

    def test_debate_not_found_returns_404(self, evidence_mixin, mock_storage, mock_http_handler):
        """Non-existent debate returns 404."""
        mock_storage.get_debate.return_value = None
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 404

    def test_no_grounded_verdict(self, evidence_mixin, mock_storage, mock_http_handler):
        """Debate without grounded_verdict returns has_citations=False."""
        mock_storage.get_debate.return_value = _make_debate(grounded_verdict=None)
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "d-1"
        assert body["has_citations"] is False
        assert "No evidence citations" in body["message"]
        assert body["grounded_verdict"] is None

    def test_grounded_verdict_as_dict(self, evidence_mixin, mock_storage, mock_http_handler):
        """Grounded verdict stored as dict is returned properly."""
        verdict = {
            "grounding_score": 0.85,
            "confidence": 0.9,
            "claims": [
                {"claim": "Redis scales well", "evidence": "Benchmark data"},
            ],
            "all_citations": [
                {"source": "Redis docs", "url": "https://redis.io"},
            ],
            "verdict": "Redis is recommended for caching",
        }
        mock_storage.get_debate.return_value = _make_debate(grounded_verdict=verdict)
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["has_citations"] is True
        assert body["grounding_score"] == 0.85
        assert body["confidence"] == 0.9
        assert len(body["claims"]) == 1
        assert len(body["all_citations"]) == 1
        assert body["verdict"] == "Redis is recommended for caching"

    def test_grounded_verdict_as_json_string(self, evidence_mixin, mock_storage, mock_http_handler):
        """Grounded verdict stored as JSON string is parsed."""
        verdict = {
            "grounding_score": 0.7,
            "confidence": 0.8,
            "claims": [],
            "all_citations": [],
            "verdict": "Some verdict",
        }
        mock_storage.get_debate.return_value = _make_debate(
            grounded_verdict=json.dumps(verdict),
        )
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["has_citations"] is True
        assert body["grounding_score"] == 0.7
        assert body["verdict"] == "Some verdict"

    def test_grounded_verdict_unparseable_string(self, evidence_mixin, mock_storage, mock_http_handler):
        """Unparseable grounded_verdict returns has_citations=False."""
        mock_storage.get_debate.return_value = _make_debate(
            grounded_verdict="not valid json {{{",
        )
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["has_citations"] is False
        assert "could not be parsed" in body["message"]

    def test_grounded_verdict_empty_dict(self, evidence_mixin, mock_storage, mock_http_handler):
        """Empty dict grounded_verdict returns has_citations=True with defaults."""
        mock_storage.get_debate.return_value = _make_debate(grounded_verdict={})
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        # Empty dict is falsy, so safe_json_parse returns {} which is falsy
        # Therefore has_citations should be False
        assert body["has_citations"] is False

    def test_citations_record_not_found_error(self, evidence_mixin, mock_storage, mock_http_handler):
        """RecordNotFoundError returns 404."""
        mock_storage.get_debate.side_effect = RecordNotFoundError("debates", "d-1")
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 404

    def test_citations_storage_error(self, evidence_mixin, mock_storage, mock_http_handler):
        """StorageError returns 500."""
        mock_storage.get_debate.side_effect = StorageError("connection failed")
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 500
        body = _body(result)
        assert "database" in body.get("error", "").lower()

    def test_citations_database_error(self, evidence_mixin, mock_storage, mock_http_handler):
        """DatabaseError returns 500."""
        mock_storage.get_debate.side_effect = DatabaseError("timeout")
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 500
        body = _body(result)
        assert "database" in body.get("error", "").lower()

    def test_grounded_verdict_defaults_for_missing_keys(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """Grounded verdict dict missing keys uses get() defaults."""
        mock_storage.get_debate.return_value = _make_debate(
            grounded_verdict={"claims": [{"claim": "X"}]}
        )
        result = evidence_mixin._get_citations(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["has_citations"] is True
        assert body["grounding_score"] == 0
        assert body["confidence"] == 0
        assert body["verdict"] == ""
        assert body["all_citations"] == []


# =============================================================================
# Evidence Trail Tests
# =============================================================================


class TestGetEvidence:
    """Tests for _get_evidence endpoint."""

    def test_no_storage_returns_503(self, evidence_mixin_no_storage, mock_http_handler):
        """Storage unavailable returns 503."""
        result = evidence_mixin_no_storage._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 503

    def test_debate_not_found_returns_404(self, evidence_mixin, mock_storage, mock_http_handler):
        """Non-existent debate returns 404."""
        mock_storage.get_debate.return_value = None
        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 404

    def test_evidence_no_verdict_no_memory(self, evidence_mixin, mock_storage, mock_http_handler):
        """Debate with no grounded verdict and no continuum memory."""
        mock_storage.get_debate.return_value = _make_debate(
            task="Some task",
            grounded_verdict=None,
        )
        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "d-1"
        assert body["task"] == "Some task"
        assert body["has_evidence"] is False
        assert body["grounded_verdict"] is None
        assert body["claims"] == []
        assert body["citations"] == []
        assert body["related_evidence"] == []
        assert body["evidence_count"] == 0

    def test_evidence_with_grounded_verdict(self, evidence_mixin, mock_storage, mock_http_handler):
        """Grounded verdict data populates response fields."""
        verdict = {
            "grounding_score": 0.9,
            "confidence": 0.85,
            "claims": [{"claim": "A"}, {"claim": "B"}],
            "all_citations": [{"source": "S1"}],
            "verdict": "Decision X",
        }
        mock_storage.get_debate.return_value = _make_debate(
            task="Task Y",
            grounded_verdict=verdict,
        )
        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["has_evidence"] is True
        gv = body["grounded_verdict"]
        assert gv["grounding_score"] == 0.9
        assert gv["confidence"] == 0.85
        assert gv["claims_count"] == 2
        assert gv["citations_count"] == 1
        assert gv["verdict"] == "Decision X"
        assert len(body["claims"]) == 2
        assert len(body["citations"]) == 1

    def test_evidence_with_json_string_verdict(self, evidence_mixin, mock_storage, mock_http_handler):
        """Grounded verdict as JSON string is parsed via safe_json_parse."""
        verdict = {
            "grounding_score": 0.5,
            "confidence": 0.6,
            "claims": [],
            "all_citations": [],
            "verdict": "V",
        }
        mock_storage.get_debate.return_value = _make_debate(
            grounded_verdict=json.dumps(verdict),
        )
        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["has_evidence"] is True
        assert body["grounded_verdict"]["grounding_score"] == 0.5

    def test_evidence_with_continuum_memory(self, evidence_mixin, mock_storage, mock_http_handler):
        """Related evidence from ContinuumMemory is included."""
        mock_storage.get_debate.return_value = _make_debate(
            task="Evaluate caching strategies",
            grounded_verdict=None,
        )

        # Create mock memory with evidence-type items
        mock_memory_item = MagicMock()
        mock_memory_item.id = "mem-1"
        mock_memory_item.content = "Redis benchmarks show 100k ops/s"
        mock_memory_item.metadata = {"type": "evidence", "source": "benchmark"}
        mock_memory_item.importance = 0.8
        mock_memory_item.tier = "fast"

        mock_non_evidence_item = MagicMock()
        mock_non_evidence_item.id = "mem-2"
        mock_non_evidence_item.content = "General note"
        mock_non_evidence_item.metadata = {"type": "note"}

        mock_continuum = MagicMock()
        mock_continuum.search.return_value = [mock_memory_item, mock_non_evidence_item]

        evidence_mixin.ctx["continuum_memory"] = mock_continuum

        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["has_evidence"] is True
        assert body["evidence_count"] == 1
        assert body["related_evidence"][0]["id"] == "mem-1"
        assert body["related_evidence"][0]["content"] == "Redis benchmarks show 100k ops/s"
        assert body["related_evidence"][0]["source"] == "benchmark"
        assert body["related_evidence"][0]["importance"] == 0.8
        assert body["related_evidence"][0]["tier"] == "fast"

    def test_evidence_continuum_memory_error_graceful(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """ContinuumMemory errors are caught gracefully (not propagated)."""
        mock_storage.get_debate.return_value = _make_debate(
            task="Some task",
            grounded_verdict=None,
        )

        mock_continuum = MagicMock()
        mock_continuum.search.side_effect = RuntimeError("Memory backend down")
        evidence_mixin.ctx["continuum_memory"] = mock_continuum

        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["related_evidence"] == []
        assert body["evidence_count"] == 0

    def test_evidence_continuum_memory_not_configured(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """No continuum_memory in ctx is handled gracefully."""
        mock_storage.get_debate.return_value = _make_debate(task="Task")
        # ctx has no continuum_memory key
        evidence_mixin.ctx.pop("continuum_memory", None)
        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["related_evidence"] == []

    def test_evidence_continuum_search_params(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """ContinuumMemory.search is called with correct parameters."""
        task = "A" * 300  # Long task string
        mock_storage.get_debate.return_value = _make_debate(task=task)

        mock_continuum = MagicMock()
        mock_continuum.search.return_value = []
        evidence_mixin.ctx["continuum_memory"] = mock_continuum

        evidence_mixin._get_evidence(mock_http_handler, "d-1")
        mock_continuum.search.assert_called_once_with(
            query=task[:200],  # Truncated to 200 chars
            limit=10,
            min_importance=0.3,
        )

    def test_evidence_empty_task_skips_memory_search(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """Empty task string means memory search is skipped."""
        mock_storage.get_debate.return_value = _make_debate(task="")

        mock_continuum = MagicMock()
        mock_continuum.search.return_value = []
        evidence_mixin.ctx["continuum_memory"] = mock_continuum

        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        # search should not be called because task is empty
        mock_continuum.search.assert_not_called()

    def test_evidence_both_verdict_and_memory(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """Both grounded verdict and related memory evidence are returned."""
        verdict = {
            "grounding_score": 0.7,
            "confidence": 0.8,
            "claims": [{"claim": "C1"}],
            "all_citations": [{"source": "S1"}],
            "verdict": "V1",
        }
        mock_storage.get_debate.return_value = _make_debate(
            task="Evaluate X",
            grounded_verdict=verdict,
        )

        mock_mem = MagicMock()
        mock_mem.id = "m-1"
        mock_mem.content = "Evidence from memory"
        mock_mem.metadata = {"type": "evidence", "source": "docs"}
        mock_mem.importance = 0.6
        mock_mem.tier = "medium"

        mock_continuum = MagicMock()
        mock_continuum.search.return_value = [mock_mem]
        evidence_mixin.ctx["continuum_memory"] = mock_continuum

        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["has_evidence"] is True
        assert body["grounded_verdict"] is not None
        assert body["evidence_count"] == 1
        assert len(body["claims"]) == 1
        assert len(body["citations"]) == 1

    def test_evidence_record_not_found_error(self, evidence_mixin, mock_storage, mock_http_handler):
        """RecordNotFoundError returns 404."""
        mock_storage.get_debate.side_effect = RecordNotFoundError("debates", "d-1")
        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 404

    def test_evidence_storage_error(self, evidence_mixin, mock_storage, mock_http_handler):
        """StorageError returns 500."""
        mock_storage.get_debate.side_effect = StorageError("connection failed")
        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 500

    def test_evidence_database_error(self, evidence_mixin, mock_storage, mock_http_handler):
        """DatabaseError returns 500."""
        mock_storage.get_debate.side_effect = DatabaseError("timeout")
        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 500

    def test_evidence_memory_item_missing_metadata(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """Memory item with None metadata is handled gracefully."""
        mock_storage.get_debate.return_value = _make_debate(task="Task")

        mock_mem = MagicMock()
        mock_mem.id = "m-1"
        mock_mem.content = "content"
        mock_mem.metadata = None
        mock_mem.importance = 0.5
        mock_mem.tier = "slow"

        mock_continuum = MagicMock()
        mock_continuum.search.return_value = [mock_mem]
        evidence_mixin.ctx["continuum_memory"] = mock_continuum

        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        # metadata is None, so metadata.get("type") would fail
        # but the code uses `metadata = getattr(memory, "metadata", {}) or {}`
        # None or {} == {}, so type check returns False -- no evidence items
        assert body["evidence_count"] == 0

    def test_evidence_continuum_without_search_method(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """Continuum memory without search method is skipped."""
        mock_storage.get_debate.return_value = _make_debate(task="Task")

        mock_continuum = MagicMock(spec=[])  # No methods at all
        evidence_mixin.ctx["continuum_memory"] = mock_continuum

        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["related_evidence"] == []

    def test_evidence_memory_item_default_source(
        self, evidence_mixin, mock_storage, mock_http_handler
    ):
        """Memory item without source in metadata defaults to 'unknown'."""
        mock_storage.get_debate.return_value = _make_debate(task="Task")

        mock_mem = MagicMock()
        mock_mem.id = "m-1"
        mock_mem.content = "evidence text"
        mock_mem.metadata = {"type": "evidence"}  # No source key
        mock_mem.importance = 0.5
        mock_mem.tier = "medium"

        mock_continuum = MagicMock()
        mock_continuum.search.return_value = [mock_mem]
        evidence_mixin.ctx["continuum_memory"] = mock_continuum

        result = evidence_mixin._get_evidence(mock_http_handler, "d-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["evidence_count"] == 1
        assert body["related_evidence"][0]["source"] == "unknown"
