"""Tests for decision explain handler.

Tests the decision explainability API endpoints:
- GET /api/v1/decisions/{request_id}/explain - Get comprehensive decision explanation
- Supports JSON, Markdown, and HTML output formats
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


class MockDebateResult:
    """Mock debate result for testing."""

    def __init__(
        self,
        task: str = "Test decision",
        final_answer: str = "Yes, proceed",
        confidence: float = 0.85,
        consensus_reached: bool = True,
        messages: list = None,
        participants: list = None,
        rounds_used: int = 3,
        duration_seconds: float = 120.5,
        completed_at: datetime = None,
        consensus_proof: Any = None,
        agent_contributions: list = None,
    ):
        self.task = task
        self.final_answer = final_answer
        self.answer = final_answer
        self.confidence = confidence
        self.consensus_reached = consensus_reached
        self.messages = messages or []
        self.participants = participants or ["claude", "gpt-4"]
        self.rounds_used = rounds_used
        self.duration_seconds = duration_seconds
        self.completed_at = completed_at or datetime.now(timezone.utc)
        self.consensus_proof = consensus_proof
        self.agent_contributions = agent_contributions or []


class MockConsensusProof:
    """Mock consensus proof for testing."""

    def __init__(self):
        self.agreement_ratio = 0.9
        self.has_strong_consensus = True
        self.claims = []
        self.evidence_chain = []
        self.votes = []
        self.dissenting_agents = []
        self.dissents = []
        self.unresolved_tensions = []
        self.rounds_to_consensus = 3
        self.checksum = "abc123"


class MockMessage:
    """Mock message for testing."""

    def __init__(self, agent: str, content: str, role: str = "proposer", round_num: int = 1):
        self.agent = agent
        self.content = content
        self.role = role
        self.round = round_num


class MockVote:
    """Mock vote for testing."""

    def __init__(self, agent: str, vote: str, confidence: float, reasoning: str):
        self.agent = agent
        self.vote = vote
        self.confidence = confidence
        self.reasoning = reasoning


class MockDissent:
    """Mock dissent record for testing."""

    def __init__(self, agent: str, reasons: list, alternative_view: str, severity: float):
        self.agent = agent
        self.reasons = reasons
        self.alternative_view = alternative_view
        self.suggested_resolution = "Consider more data"
        self.severity = severity


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_nomic_dir():
    """Create temporary nomic directory with trace files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_path = Path(tmpdir)
        traces_dir = nomic_path / "traces"
        traces_dir.mkdir(parents=True)
        replays_dir = nomic_path / "replays"
        replays_dir.mkdir(parents=True)
        yield nomic_path


@pytest.fixture
def mock_trace_file(mock_nomic_dir):
    """Create a mock trace file."""
    trace_path = mock_nomic_dir / "traces" / "request-123.json"
    trace_data = {
        "debate_id": "request-123",
        "task": "Should we deploy to production?",
        "final_answer": "Yes, proceed with caution",
        "confidence": 0.85,
        "consensus_reached": True,
        "messages": [
            {
                "agent": "claude",
                "content": "I recommend proceeding",
                "role": "proposer",
                "round": 1,
            },
            {
                "agent": "gpt-4",
                "content": "I agree with conditions",
                "role": "proposer",
                "round": 1,
            },
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    trace_path.write_text(json.dumps(trace_data))
    return trace_path


@pytest.fixture
def mock_replay_dir(mock_nomic_dir):
    """Create mock replay events directory."""
    replay_dir = mock_nomic_dir / "replays" / "request-456"
    replay_dir.mkdir(parents=True)
    events_path = replay_dir / "events.jsonl"
    events = [
        {
            "type": "agent_message",
            "agent": "claude",
            "round": 1,
            "data": {"content": "I propose we proceed", "role": "proposer"},
        },
        {
            "type": "consensus",
            "data": {"reached": True, "confidence": 0.8, "answer": "Yes"},
        },
        {
            "type": "debate_end",
            "data": {"answer": "Yes, proceed"},
        },
    ]
    with events_path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return replay_dir


@pytest.fixture
def explain_handler(mock_nomic_dir):
    """Create explain handler instance."""
    from aragora.server.handlers.decisions.explain import DecisionExplainHandler

    ctx = {"nomic_dir": mock_nomic_dir}
    return DecisionExplainHandler(ctx)


@pytest.fixture
def explain_handler_no_dir():
    """Create explain handler without nomic directory."""
    from aragora.server.handlers.decisions.explain import DecisionExplainHandler

    ctx = {"nomic_dir": None}
    return DecisionExplainHandler(ctx)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    try:
        from aragora.server.handlers.decisions import explain

        explain._explain_limiter = explain.RateLimiter(requests_per_minute=30)
    except (ImportError, AttributeError):
        pass
    yield


# =============================================================================
# Routing Tests
# =============================================================================


class TestExplainRouting:
    """Tests for request routing and path handling."""

    def test_can_handle_v1_explain(self, explain_handler):
        """Test handler recognizes v1 explain paths."""
        assert explain_handler.can_handle("/api/v1/decisions/request-123/explain")
        assert explain_handler.can_handle("/api/v1/decisions/abc123/explain")

    def test_cannot_handle_other_paths(self, explain_handler):
        """Test handler rejects non-explain paths."""
        assert not explain_handler.can_handle("/api/v1/decisions/request-123")
        assert not explain_handler.can_handle("/api/v1/debates/request-123/explain")
        assert not explain_handler.can_handle("/api/v1/decisions")
        assert not explain_handler.can_handle("/api/health")

    def test_rate_limiting(self, explain_handler, mock_http_handler):
        """Test rate limiter enforces limits."""
        # Make many requests to trigger rate limit
        for i in range(35):
            explain_handler.handle(
                path=f"/api/v1/decisions/request-{i}/explain",
                query_params={},
                handler=mock_http_handler,
            )

        # Next request should be rate limited
        result = explain_handler.handle(
            path="/api/v1/decisions/request-new/explain",
            query_params={},
            handler=mock_http_handler,
        )

        assert result.status_code == 429


# =============================================================================
# Explanation Generation Tests
# =============================================================================


class TestExplanationGeneration:
    """Tests for explanation generation."""

    def test_explain_not_found(self, explain_handler, mock_http_handler):
        """Test explain returns 404 when decision not found."""
        result = explain_handler.handle(
            path="/api/v1/decisions/nonexistent/explain",
            query_params={},
            handler=mock_http_handler,
        )

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "not found" in body.get("error", "").lower()

    def test_explain_no_nomic_dir(self, explain_handler_no_dir, mock_http_handler):
        """Test explain returns 503 when nomic_dir not configured."""
        result = explain_handler_no_dir.handle(
            path="/api/v1/decisions/request-123/explain",
            query_params={},
            handler=mock_http_handler,
        )

        assert result.status_code == 503
        body = json.loads(result.body)
        assert "not configured" in body.get("error", "").lower()

    def test_explain_from_trace(self, explain_handler, mock_trace_file, mock_http_handler):
        """Test explanation loads from trace file."""
        mock_result = MockDebateResult(
            final_answer="Yes, proceed",
            confidence=0.85,
            consensus_reached=True,
            messages=[MockMessage("claude", "I recommend proceeding")],
            consensus_proof=MockConsensusProof(),
        )

        with patch("aragora.debate.traces.DebateTrace") as MockTrace:
            mock_trace = MagicMock()
            mock_trace.to_debate_result.return_value = mock_result
            MockTrace.load.return_value = mock_trace

            result = explain_handler.handle(
                path="/api/v1/decisions/request-123/explain",
                query_params={},
                handler=mock_http_handler,
            )

            assert result.status_code == 200
            body = json.loads(result.body)
            assert "summary" in body
            assert "reasoning" in body
            assert body["request_id"] == "request-123"

    def test_explain_from_replay(self, explain_handler, mock_replay_dir, mock_http_handler):
        """Test explanation falls back to replay events."""
        result = explain_handler.handle(
            path="/api/v1/decisions/request-456/explain",
            query_params={},
            handler=mock_http_handler,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "summary" in body


# =============================================================================
# Explanation Content Tests
# =============================================================================


class TestExplanationContent:
    """Tests for explanation content sections."""

    def test_summary_section(self, explain_handler):
        """Test summary section is built correctly."""
        result = MockDebateResult(
            final_answer="Yes",
            confidence=0.9,
            consensus_reached=True,
            consensus_proof=MockConsensusProof(),
        )

        summary = explain_handler._build_summary(result)

        assert summary["answer"] == "Yes"
        assert summary["confidence"] == 0.9
        assert summary["consensus_reached"] is True
        assert summary["agreement_ratio"] == 0.9
        assert summary["has_strong_consensus"] is True

    def test_votes_section(self, explain_handler):
        """Test votes section includes agent votes."""
        proof = MockConsensusProof()
        proof.votes = [
            MockVote("claude", "agree", 0.9, "Strong evidence"),
            MockVote("gpt-4", "agree", 0.8, "Reasonable conclusion"),
        ]
        result = MockDebateResult(consensus_proof=proof)

        votes = explain_handler._build_votes(result)

        assert len(votes) == 2

    def test_votes_from_contributions(self, explain_handler):
        """Test votes inferred from agent contributions when no proof."""
        result = MockDebateResult(
            consensus_proof=None,
            agent_contributions=[
                {"agent": "claude", "response": "I agree with the proposal"},
            ],
        )

        votes = explain_handler._build_votes(result)

        assert len(votes) == 1
        assert votes[0]["agent"] == "claude"

    def test_dissent_section(self, explain_handler):
        """Test dissent section captures minority views."""
        proof = MockConsensusProof()
        proof.dissenting_agents = ["gemini"]
        proof.dissents = [
            MockDissent("gemini", ["Insufficient data"], "We need more analysis", 0.6)
        ]
        result = MockDebateResult(consensus_proof=proof)

        dissent = explain_handler._build_dissent(result)

        assert "gemini" in dissent["dissenting_agents"]
        assert len(dissent["reasons"]) > 0
        assert len(dissent["alternative_views"]) > 0
        assert dissent["severity"] == 0.6

    def test_audit_trail_section(self, explain_handler):
        """Test audit trail includes timing and participants."""
        completed_at = datetime.now(timezone.utc)
        proof = MockConsensusProof()
        proof.checksum = "sha256:abc123"

        result = MockDebateResult(
            duration_seconds=150.5,
            rounds_used=4,
            participants=["claude", "gpt-4", "gemini"],
            completed_at=completed_at,
            consensus_proof=proof,
        )

        audit = explain_handler._build_audit_trail(result)

        assert audit["duration_seconds"] == 150.5
        assert audit["rounds_completed"] == 3  # From consensus proof
        assert "claude" in audit["agents_involved"]
        assert audit["checksum"] == "sha256:abc123"


# =============================================================================
# Output Format Tests
# =============================================================================


class TestOutputFormats:
    """Tests for different output formats."""

    def test_json_format_default(self, explain_handler, mock_trace_file, mock_http_handler):
        """Test JSON format is returned by default."""
        mock_result = MockDebateResult(consensus_proof=MockConsensusProof())

        with patch("aragora.debate.traces.DebateTrace") as MockTrace:
            mock_trace = MagicMock()
            mock_trace.to_debate_result.return_value = mock_result
            MockTrace.load.return_value = mock_trace

            result = explain_handler.handle(
                path="/api/v1/decisions/request-123/explain",
                query_params={},
                handler=mock_http_handler,
            )

            assert result.status_code == 200
            # Should be valid JSON
            body = json.loads(result.body)
            assert isinstance(body, dict)

    def test_markdown_format(self, explain_handler, mock_trace_file, mock_http_handler):
        """Test Markdown format output."""
        mock_result = MockDebateResult(
            final_answer="Proceed",
            confidence=0.85,
            consensus_proof=MockConsensusProof(),
        )

        with patch("aragora.debate.traces.DebateTrace") as MockTrace:
            mock_trace = MagicMock()
            mock_trace.to_debate_result.return_value = mock_result
            MockTrace.load.return_value = mock_trace

            result = explain_handler.handle(
                path="/api/v1/decisions/request-123/explain",
                query_params={"format": "md"},
                handler=mock_http_handler,
            )

            assert result.status_code == 200
            content_type = result.headers.get("Content-Type", "")
            assert "markdown" in content_type
            content = result.body.decode("utf-8")
            assert "# Decision Explanation" in content
            assert "## Summary" in content

    def test_html_format(self, explain_handler, mock_trace_file, mock_http_handler):
        """Test HTML format output."""
        mock_result = MockDebateResult(
            final_answer="Proceed",
            confidence=0.85,
            consensus_proof=MockConsensusProof(),
        )

        with patch("aragora.debate.traces.DebateTrace") as MockTrace:
            mock_trace = MagicMock()
            mock_trace.to_debate_result.return_value = mock_result
            MockTrace.load.return_value = mock_trace

            result = explain_handler.handle(
                path="/api/v1/decisions/request-123/explain",
                query_params={"format": "html"},
                handler=mock_http_handler,
            )

            assert result.status_code == 200
            content_type = result.headers.get("Content-Type", "")
            assert "html" in content_type
            content = result.body.decode("utf-8")
            assert "<!DOCTYPE html>" in content
            assert "<title>Decision Explanation" in content


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_invalid_request_id(self, explain_handler, mock_http_handler):
        """Test invalid request ID format returns 400."""
        result = explain_handler.handle(
            path="/api/v1/decisions//explain",  # Empty ID
            query_params={},
            handler=mock_http_handler,
        )

        assert result.status_code == 400

    def test_reasoning_section_empty(self, explain_handler, mock_nomic_dir):
        """Test reasoning section handles empty data."""
        result = MockDebateResult(consensus_proof=None, messages=[])

        reasoning = explain_handler._build_reasoning(mock_nomic_dir, "test-id", result)

        assert "key_claims" in reasoning
        assert "supporting_evidence" in reasoning
        assert "crux_claims" in reasoning
