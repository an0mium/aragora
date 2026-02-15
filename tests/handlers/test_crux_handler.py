"""Tests for crux detection REST API in BeliefHandler."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class MockCruxClaim:
    claim_id: str
    statement: str
    author: str
    crux_score: float
    influence_score: float
    disagreement_score: float
    uncertainty_score: float
    centrality_score: float
    affected_claims: list[str] = field(default_factory=list)
    contesting_agents: list[str] = field(default_factory=list)
    resolution_impact: float = 0.0

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "author": self.author,
            "crux_score": round(self.crux_score, 4),
            "influence_score": round(self.influence_score, 4),
            "disagreement_score": round(self.disagreement_score, 4),
            "uncertainty_score": round(self.uncertainty_score, 4),
            "centrality_score": round(self.centrality_score, 4),
            "affected_claims": self.affected_claims,
            "contesting_agents": self.contesting_agents,
            "resolution_impact": round(self.resolution_impact, 4),
        }


@dataclass
class MockCruxAnalysisResult:
    cruxes: list[MockCruxClaim]
    total_claims: int
    total_disagreements: int
    average_uncertainty: float
    convergence_barrier: float
    recommended_focus: list[str]

    def to_dict(self) -> dict:
        return {
            "cruxes": [c.to_dict() for c in self.cruxes],
            "total_claims": self.total_claims,
            "total_disagreements": self.total_disagreements,
            "average_uncertainty": round(self.average_uncertainty, 4),
            "convergence_barrier": round(self.convergence_barrier, 4),
            "recommended_focus": self.recommended_focus,
        }


def _make_analysis(num_cruxes=3):
    """Build a mock CruxAnalysisResult."""
    cruxes = [
        MockCruxClaim(
            claim_id=f"claim_{i}",
            statement=f"Claim {i} statement",
            author=f"agent_{i}",
            crux_score=0.9 - i * 0.15,
            influence_score=0.8 - i * 0.1,
            disagreement_score=0.7 - i * 0.1,
            uncertainty_score=0.6 - i * 0.1,
            centrality_score=0.5,
            affected_claims=[f"claim_{i + 10}"],
            contesting_agents=["agent_a", "agent_b"],
            resolution_impact=0.3,
        )
        for i in range(num_cruxes)
    ]
    return MockCruxAnalysisResult(
        cruxes=cruxes,
        total_claims=10,
        total_disagreements=4,
        average_uncertainty=0.65,
        convergence_barrier=0.42,
        recommended_focus=[c.claim_id for c in cruxes],
    )


class MockTrace:
    def __init__(self, messages=None):
        self._messages = messages or []

    def to_debate_result(self):
        result = MagicMock()
        result.messages = self._messages
        return result

    @classmethod
    def load(cls, path):
        msg1 = MagicMock(agent="claude", content="We should use microservices")
        msg2 = MagicMock(agent="gpt4", content="Monolith is better")
        return cls([msg1, msg2])


@pytest.fixture
def handler():
    from aragora.server.handlers.belief import BeliefHandler

    h = BeliefHandler(server_context={})
    return h


def _setup_trace_dir(tmp_path, debate_id="debate-1"):
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir(exist_ok=True)
    (traces_dir / f"{debate_id}.json").write_text("{}")
    return tmp_path


_DETECTOR_PATCH = "aragora.reasoning.crux_detector.CruxDetector"
_TRACE_PATCH = "aragora.debate.traces.DebateTrace"
_CREATE_NETWORK_PATCH = "aragora.server.handlers.belief.BeliefHandler._create_belief_network"
_BELIEF_AVAILABLE_PATCH = "aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE"


class TestCruxDetectionAPI:
    """Tests for GET /api/v1/debates/{debate_id}/cruxes."""

    def test_cruxes_sorted_by_score(self, handler, tmp_path):
        handler.ctx["nomic_dir"] = _setup_trace_dir(tmp_path)
        analysis = _make_analysis(3)

        mock_detector = MagicMock()
        mock_detector.detect_cruxes.return_value = analysis
        mock_network = MagicMock()

        with (
            patch(_BELIEF_AVAILABLE_PATCH, True),
            patch(_TRACE_PATCH, MockTrace),
            patch(_CREATE_NETWORK_PATCH, return_value=mock_network),
            patch(_DETECTOR_PATCH, return_value=mock_detector),
        ):
            result = handler._get_crux_analysis(tmp_path, "debate-1", 5)

        body = result[0]
        assert body["debate_id"] == "debate-1"
        cruxes = body["cruxes"]
        assert len(cruxes) == 3
        # Sorted by descending crux_score
        for i in range(len(cruxes) - 1):
            assert cruxes[i]["crux_score"] >= cruxes[i + 1]["crux_score"]

    def test_includes_all_score_fields(self, handler, tmp_path):
        handler.ctx["nomic_dir"] = _setup_trace_dir(tmp_path)
        analysis = _make_analysis(1)

        mock_detector = MagicMock()
        mock_detector.detect_cruxes.return_value = analysis
        mock_network = MagicMock()

        with (
            patch(_BELIEF_AVAILABLE_PATCH, True),
            patch(_TRACE_PATCH, MockTrace),
            patch(_CREATE_NETWORK_PATCH, return_value=mock_network),
            patch(_DETECTOR_PATCH, return_value=mock_detector),
        ):
            result = handler._get_crux_analysis(tmp_path, "debate-1", 5)

        crux = result[0]["cruxes"][0]
        expected_fields = {
            "claim_id", "statement", "author", "crux_score",
            "influence_score", "disagreement_score", "uncertainty_score",
            "centrality_score", "affected_claims", "contesting_agents",
            "resolution_impact",
        }
        assert expected_fields.issubset(set(crux.keys()))

    def test_empty_network_returns_empty(self, handler, tmp_path):
        handler.ctx["nomic_dir"] = _setup_trace_dir(tmp_path)
        empty_analysis = MockCruxAnalysisResult(
            cruxes=[],
            total_claims=0,
            total_disagreements=0,
            average_uncertainty=0.0,
            convergence_barrier=0.0,
            recommended_focus=[],
        )

        mock_detector = MagicMock()
        mock_detector.detect_cruxes.return_value = empty_analysis
        mock_network = MagicMock()

        with (
            patch(_BELIEF_AVAILABLE_PATCH, True),
            patch(_TRACE_PATCH, MockTrace),
            patch(_CREATE_NETWORK_PATCH, return_value=mock_network),
            patch(_DETECTOR_PATCH, return_value=mock_detector),
        ):
            result = handler._get_crux_analysis(tmp_path, "debate-1", 5)

        body = result[0]
        assert body["cruxes"] == []
        assert body["total_claims"] == 0

    def test_limit_param_works(self, handler, tmp_path):
        handler.ctx["nomic_dir"] = _setup_trace_dir(tmp_path)
        analysis = _make_analysis(2)

        mock_detector = MagicMock()
        mock_detector.detect_cruxes.return_value = analysis
        mock_network = MagicMock()

        with (
            patch(_BELIEF_AVAILABLE_PATCH, True),
            patch(_TRACE_PATCH, MockTrace),
            patch(_CREATE_NETWORK_PATCH, return_value=mock_network),
            patch(_DETECTOR_PATCH, return_value=mock_detector),
        ):
            result = handler._get_crux_analysis(tmp_path, "debate-1", 2)

        # Verify limit was passed to detect_cruxes
        mock_detector.detect_cruxes.assert_called_once_with(top_k=2)

    def test_missing_debate_returns_404(self, handler, tmp_path):
        handler.ctx["nomic_dir"] = tmp_path
        (tmp_path / "traces").mkdir(exist_ok=True)

        with patch(_BELIEF_AVAILABLE_PATCH, True):
            result = handler._get_crux_analysis(tmp_path, "nonexistent", 5)
        assert result[1] == 404

    def test_no_nomic_dir_returns_503(self, handler):
        with patch(_BELIEF_AVAILABLE_PATCH, True):
            result = handler._get_crux_analysis(None, "debate-1", 5)
        assert result[1] == 503

    def test_analysis_metadata_present(self, handler, tmp_path):
        handler.ctx["nomic_dir"] = _setup_trace_dir(tmp_path)
        analysis = _make_analysis(2)

        mock_detector = MagicMock()
        mock_detector.detect_cruxes.return_value = analysis
        mock_network = MagicMock()

        with (
            patch(_BELIEF_AVAILABLE_PATCH, True),
            patch(_TRACE_PATCH, MockTrace),
            patch(_CREATE_NETWORK_PATCH, return_value=mock_network),
            patch(_DETECTOR_PATCH, return_value=mock_detector),
        ):
            result = handler._get_crux_analysis(tmp_path, "debate-1", 5)

        body = result[0]
        assert "total_disagreements" in body
        assert "average_uncertainty" in body
        assert "convergence_barrier" in body
        assert "recommended_focus" in body

    def test_belief_unavailable_returns_503(self, handler, tmp_path):
        handler.ctx["nomic_dir"] = _setup_trace_dir(tmp_path)

        with patch(_BELIEF_AVAILABLE_PATCH, False):
            result = handler._get_crux_analysis(tmp_path, "debate-1", 5)
        assert result[1] == 503


class TestCruxCanHandle:
    """Tests for route matching of versioned crux endpoint."""

    def test_can_handle_versioned_cruxes(self, handler):
        assert handler.can_handle("/api/v1/debates/debate-123/cruxes")

    def test_cannot_handle_unrelated(self, handler):
        assert not handler.can_handle("/api/v1/debates/debate-123/something-else")
