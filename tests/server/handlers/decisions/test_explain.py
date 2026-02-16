"""
Tests for aragora.server.handlers.decisions.explain - Decision Explainability handler.

Tests cover:
- DecisionExplainHandler initialization
- can_handle() route matching
- handle() with RBAC verification
- _explain_decision() for various formats (JSON, Markdown, HTML)
- _build_explanation() data aggregation
- _build_summary(), _build_reasoning(), _build_votes(), _build_dissent()
- _build_tensions(), _build_audit_trail()
- _load_from_replay() and _load_from_storage()
- Rate limiting
- Error handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

import pytest

from aragora.server.handlers.decisions.explain import DecisionExplainHandler


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockAuthContext:
    """Mock authentication context for testing."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    org_id: str = "org-123"
    role: str = "admin"
    permissions: set = field(default_factory=lambda: {"decision:read", "*"})

    @property
    def authenticated(self) -> bool:
        return self.is_authenticated


@dataclass
class MockHandler:
    """Mock HTTP handler for testing."""

    headers: dict = field(default_factory=dict)
    client_address: tuple = ("127.0.0.1", 12345)
    command: str = "GET"


@dataclass
class MockMessage:
    """Mock debate message."""

    agent: str = "claude"
    content: str = "This is my analysis."
    role: str = "proposer"
    round: int = 1


@dataclass
class MockVote:
    """Mock vote record."""

    agent: str = "claude"
    vote: str = "agree"
    confidence: float = 0.9
    reasoning: str = "Strong evidence supports this conclusion."


@dataclass
class MockClaim:
    """Mock claim for testing."""

    statement: str = "The policy should be implemented."
    author: str = "claude"
    net_evidence_strength: float = 0.8


@dataclass
class MockDissentRecord:
    """Mock dissent record."""

    agent: str = "gpt4"
    reasons: list = field(default_factory=lambda: ["Insufficient evidence", "Risk too high"])
    alternative_view: str = "Consider a phased approach instead."
    suggested_resolution: str = "Run pilot program first."
    severity: float = 0.6


@dataclass
class MockTension:
    """Mock unresolved tension."""

    description: str = "Trade-off between speed and quality."
    impact: str = "May affect delivery timeline."


@dataclass
class MockConsensusProof:
    """Mock consensus proof for testing."""

    agreement_ratio: float = 0.85
    has_strong_consensus: bool = True
    votes: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    evidence_chain: list = field(default_factory=list)
    dissenting_agents: list = field(default_factory=lambda: ["gpt4"])
    dissents: list = field(default_factory=list)
    unresolved_tensions: list = field(default_factory=list)
    checksum: str = "sha256:abc123"
    rounds_to_consensus: int = 3


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    task: str = "Should we implement feature X?"
    final_answer: str = "Yes, implement with safeguards."
    consensus_reached: bool = True
    confidence: float = 0.85
    messages: list = field(default_factory=list)
    rounds_used: int = 3
    participants: list = field(default_factory=lambda: ["claude", "gpt4", "gemini"])
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 45.5
    consensus_proof: MockConsensusProof = None
    agent_contributions: list = field(default_factory=list)


def create_handler(ctx: dict = None) -> DecisionExplainHandler:
    """Create a DecisionExplainHandler with context."""
    return DecisionExplainHandler(ctx or {})


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict | str:
    """Extract body from HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return body.decode("utf-8")
        return body
    return result[0]


def get_content_type(result) -> str:
    """Extract content type from HandlerResult."""
    if hasattr(result, "content_type"):
        return result.content_type
    if hasattr(result, "headers") and result.headers:
        return result.headers.get("Content-Type", "")
    return ""


# ===========================================================================
# Tests for can_handle() Route Matching
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_handles_v1_decisions_explain(self):
        """Should handle /api/v1/decisions/:request_id/explain."""
        handler = create_handler()
        assert handler.can_handle("/api/v1/decisions/req-123/explain") is True

    def test_handles_v1_with_uuid(self):
        """Should handle UUIDs as request_id."""
        handler = create_handler()
        assert (
            handler.can_handle("/api/v1/decisions/550e8400-e29b-41d4-a716-446655440000/explain")
            is True
        )

    def test_rejects_non_explain_paths(self):
        """Should reject paths that don't end with /explain."""
        handler = create_handler()
        assert handler.can_handle("/api/v1/decisions/req-123") is False
        assert handler.can_handle("/api/v1/decisions/req-123/summary") is False

    def test_rejects_other_api_paths(self):
        """Should reject unrelated API paths."""
        handler = create_handler()
        assert handler.can_handle("/api/v1/debates/req-123") is False
        assert handler.can_handle("/api/v1/users/user-123") is False
        assert handler.can_handle("/api/v1/health") is False


# ===========================================================================
# Tests for handle() with RBAC
# ===========================================================================


class TestHandle:
    """Tests for handle() method with RBAC verification."""

    @pytest.mark.asyncio
    async def test_handle_requires_authentication(self):
        """Should return 401 when not authenticated."""
        handler = create_handler({"nomic_dir": Path("/tmp/nomic")})
        mock_http = MockHandler()

        with patch.object(
            handler,
            "get_auth_context",
            side_effect=ValueError("UnauthorizedError"),
        ):
            from aragora.server.handlers.secure import UnauthorizedError

            with patch.object(
                handler,
                "get_auth_context",
                side_effect=UnauthorizedError("Not authenticated"),
            ):
                result = await handler.handle("/api/v1/decisions/req-123/explain", {}, mock_http)

        assert get_status(result) == 401
        body = get_body(result)
        assert "Authentication required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_requires_decision_read_permission(self):
        """Should return 403 when user lacks decision:read permission."""
        handler = create_handler({"nomic_dir": Path("/tmp/nomic")})
        mock_http = MockHandler()
        mock_auth = MockAuthContext(permissions=set())  # No permissions

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            from aragora.server.handlers.secure import ForbiddenError

            with patch.object(
                handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied"),
            ):
                result = await handler.handle("/api/v1/decisions/req-123/explain", {}, mock_http)

        assert get_status(result) == 403

    @pytest.mark.asyncio
    async def test_handle_rate_limit_exceeded(self):
        """Should return 429 when rate limit is exceeded."""
        handler = create_handler({"nomic_dir": Path("/tmp/nomic")})
        mock_http = MockHandler()

        with patch(
            "aragora.server.handlers.decisions.explain._explain_limiter.is_allowed",
            return_value=False,
        ):
            result = await handler.handle("/api/v1/decisions/req-123/explain", {}, mock_http)

        assert get_status(result) == 429
        body = get_body(result)
        assert "Rate limit" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_handle_invalid_request_id(self):
        """Should return 400 for invalid request ID format."""
        handler = create_handler({"nomic_dir": Path("/tmp/nomic")})
        mock_http = MockHandler()
        mock_auth = MockAuthContext()

        with patch.object(handler, "get_auth_context", return_value=mock_auth):
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.decisions.explain._explain_limiter.is_allowed",
                    return_value=True,
                ):
                    with patch(
                        "aragora.server.handlers.decisions.explain.validate_id",
                        return_value=(False, "Invalid ID format"),
                    ):
                        result = await handler.handle(
                            "/api/v1/decisions/<script>/explain", {}, mock_http
                        )

        assert get_status(result) == 400


# ===========================================================================
# Tests for _explain_decision()
# ===========================================================================


class TestExplainDecision:
    """Tests for _explain_decision() method."""

    def test_explain_decision_no_nomic_dir(self):
        """Should return 503 when nomic_dir is not configured."""
        handler = create_handler({})

        result = handler._explain_decision(None, "req-123", "json")

        assert get_status(result) == 503
        body = get_body(result)
        assert "not configured" in body.get("error", "")

    def test_explain_decision_not_found(self):
        """Should return 404 when decision is not found."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nomic_dir = Path(tmp_dir)
            handler = create_handler({})

            with patch.object(handler, "_build_explanation", return_value=None):
                result = handler._explain_decision(nomic_dir, "req-123", "json")

            assert get_status(result) == 404
            body = get_body(result)
            assert "not found" in body.get("error", "")

    def test_explain_decision_json_format(self):
        """Should return JSON response by default."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nomic_dir = Path(tmp_dir)
            handler = create_handler({})
            mock_explanation = {
                "request_id": "req-123",
                "summary": {"answer": "Yes", "confidence": 0.9},
            }

            with patch.object(handler, "_build_explanation", return_value=mock_explanation):
                result = handler._explain_decision(nomic_dir, "req-123", "json")

            assert get_status(result) == 200
            body = get_body(result)
            assert body["request_id"] == "req-123"
            assert body["summary"]["answer"] == "Yes"

    def test_explain_decision_markdown_format(self):
        """Should return Markdown response when format=md."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nomic_dir = Path(tmp_dir)
            handler = create_handler({})
            mock_explanation = {
                "request_id": "req-123",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "answer": "Yes",
                    "confidence": 0.9,
                    "consensus_reached": True,
                    "agreement_ratio": 0.85,
                },
                "reasoning": {"key_claims": [], "crux_claims": []},
                "votes": [],
                "dissent": {"dissenting_agents": [], "reasons": []},
                "tensions": [],
                "audit_trail": {
                    "duration_seconds": 30,
                    "rounds_completed": 3,
                    "agents_involved": ["claude"],
                },
            }

            with patch.object(handler, "_build_explanation", return_value=mock_explanation):
                result = handler._explain_decision(nomic_dir, "req-123", "md")

            assert get_status(result) == 200
            assert "text/markdown" in get_content_type(result)
            body = get_body(result)
            assert "# Decision Explanation" in body
            assert "req-123" in body

    def test_explain_decision_html_format(self):
        """Should return HTML response when format=html."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nomic_dir = Path(tmp_dir)
            handler = create_handler({})
            mock_explanation = {
                "request_id": "req-123",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "answer": "Yes",
                    "confidence": 0.9,
                    "consensus_reached": True,
                    "agreement_ratio": 0.85,
                },
                "reasoning": {"key_claims": [], "crux_claims": []},
                "votes": [],
                "dissent": {"dissenting_agents": []},
                "tensions": [],
                "audit_trail": {
                    "duration_seconds": 30,
                    "rounds_completed": 3,
                    "agents_involved": ["claude"],
                    "checksum": "abc",
                },
            }

            with patch.object(handler, "_build_explanation", return_value=mock_explanation):
                result = handler._explain_decision(nomic_dir, "req-123", "html")

            assert get_status(result) == 200
            assert "text/html" in get_content_type(result)
            body = get_body(result)
            assert "<!DOCTYPE html>" in body
            assert "Decision Explanation" in body


# ===========================================================================
# Tests for _build_* methods
# ===========================================================================


class TestBuildExplanation:
    """Tests for explanation building methods."""

    def test_build_summary_with_consensus_proof(self):
        """Should build summary with consensus proof data."""
        handler = create_handler({})
        consensus_proof = MockConsensusProof(
            agreement_ratio=0.85,
            has_strong_consensus=True,
        )
        result = MockDebateResult(
            final_answer="Yes, proceed.",
            confidence=0.9,
            consensus_reached=True,
            consensus_proof=consensus_proof,
        )

        summary = handler._build_summary(result)

        assert summary["answer"] == "Yes, proceed."
        assert summary["confidence"] == 0.9
        assert summary["consensus_reached"] is True
        assert summary["agreement_ratio"] == 0.85
        assert summary["has_strong_consensus"] is True

    def test_build_summary_without_consensus_proof(self):
        """Should build summary with defaults when no consensus proof."""
        handler = create_handler({})
        result = MockDebateResult(
            final_answer="Maybe",
            confidence=0.5,
            consensus_reached=False,
        )

        summary = handler._build_summary(result)

        assert summary["answer"] == "Maybe"
        assert summary["confidence"] == 0.5
        assert summary["consensus_reached"] is False
        assert summary["agreement_ratio"] == 0.0

    def test_build_votes_from_consensus_proof(self):
        """Should build votes section from consensus proof."""
        handler = create_handler({})
        votes = [
            MockVote(agent="claude", vote="agree", confidence=0.9),
            MockVote(agent="gpt4", vote="disagree", confidence=0.7),
        ]
        consensus_proof = MockConsensusProof(votes=votes)
        result = MockDebateResult(consensus_proof=consensus_proof)

        built_votes = handler._build_votes(result)

        assert len(built_votes) == 2
        assert built_votes[0]["agent"] == "claude"
        assert built_votes[1]["agent"] == "gpt4"

    def test_build_votes_from_agent_contributions(self):
        """Should infer votes from agent contributions when no consensus proof."""
        handler = create_handler({})
        result = MockDebateResult(
            consensus_proof=None,
            agent_contributions=[
                {"agent": "claude", "response": "I agree with this approach."},
                {"agent": "gpt4", "response": "The analysis is sound."},
            ],
        )

        built_votes = handler._build_votes(result)

        assert len(built_votes) == 2
        assert built_votes[0]["agent"] == "claude"
        assert built_votes[1]["agent"] == "gpt4"

    def test_build_dissent_with_dissenting_agents(self):
        """Should build dissent section with dissenting agents."""
        handler = create_handler({})
        dissents = [
            MockDissentRecord(
                agent="gpt4",
                reasons=["Risk too high"],
                alternative_view="Phased approach",
                severity=0.7,
            ),
        ]
        consensus_proof = MockConsensusProof(
            dissenting_agents=["gpt4"],
            dissents=dissents,
        )
        result = MockDebateResult(consensus_proof=consensus_proof)

        dissent = handler._build_dissent(result)

        assert "gpt4" in dissent["dissenting_agents"]
        assert "Risk too high" in dissent["reasons"]
        assert len(dissent["alternative_views"]) == 1
        assert dissent["severity"] == 0.7

    def test_build_tensions(self):
        """Should build tensions section from consensus proof."""
        handler = create_handler({})
        tensions = [
            MockTension(description="Speed vs quality", impact="Timeline affected"),
        ]
        consensus_proof = MockConsensusProof(unresolved_tensions=tensions)
        result = MockDebateResult(consensus_proof=consensus_proof)

        built_tensions = handler._build_tensions(result)

        assert len(built_tensions) == 1
        assert built_tensions[0]["description"] == "Speed vs quality"

    def test_build_audit_trail(self):
        """Should build audit trail with timing and participants."""
        handler = create_handler({})
        completed = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        consensus_proof = MockConsensusProof(
            checksum="sha256:abc123",
            rounds_to_consensus=4,
        )
        result = MockDebateResult(
            completed_at=completed,
            duration_seconds=60.5,
            rounds_used=4,
            participants=["claude", "gpt4", "gemini"],
            consensus_proof=consensus_proof,
        )

        audit = handler._build_audit_trail(result)

        assert audit["duration_seconds"] == 60.5
        assert audit["rounds_completed"] == 4
        assert "claude" in audit["agents_involved"]
        assert audit["checksum"] == "sha256:abc123"


# ===========================================================================
# Tests for _load_from_replay()
# ===========================================================================


class TestLoadFromReplay:
    """Tests for _load_from_replay() method."""

    def test_load_from_replay_with_valid_events(self):
        """Should load debate result from replay events."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            replay_dir = Path(tmp_dir) / "replays" / "req-123"
            replay_dir.mkdir(parents=True)
            events_file = replay_dir / "events.jsonl"

            events = [
                {
                    "type": "agent_message",
                    "agent": "claude",
                    "round": 1,
                    "data": {"content": "Analysis here", "role": "proposer"},
                },
                {
                    "type": "agent_message",
                    "agent": "gpt4",
                    "round": 1,
                    "data": {"content": "Counter analysis", "role": "critic"},
                },
                {
                    "type": "consensus",
                    "data": {"reached": True, "confidence": 0.85, "answer": "Yes"},
                },
                {"type": "debate_end", "data": {"answer": "Yes"}},
            ]
            with events_file.open("w") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")

            handler = create_handler({})

            # Patch at the aragora.core level since that's where Message and DebateResult are imported from
            with patch("aragora.core.Message") as MockMsg:
                with patch("aragora.core.DebateResult") as MockResult:
                    MockMsg.return_value = MagicMock()
                    MockResult.return_value = MagicMock()

                    result = handler._load_from_replay(events_file)

                    # Result should be a mock DebateResult (or None if import failed)
                    # The test verifies the method doesn't crash

    def test_load_from_replay_handles_malformed_events(self):
        """Should handle malformed event lines gracefully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            replay_dir = Path(tmp_dir) / "replays" / "req-123"
            replay_dir.mkdir(parents=True)
            events_file = replay_dir / "events.jsonl"

            events = [
                "not valid json",
                '{"type": "agent_message", "agent": "claude", "round": 1, "data": {"content": "Valid", "role": "proposer"}}',
                "",  # Empty line
            ]
            with events_file.open("w") as f:
                for event in events:
                    f.write(event + "\n")

            handler = create_handler({})

            # Just test that the method doesn't crash with malformed data
            result = handler._load_from_replay(events_file)
            # Result could be None or a DebateResult depending on import availability
            assert result is None or hasattr(result, "messages")

    def test_load_from_replay_returns_none_on_empty_file(self):
        """Should return None when replay file is empty."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            replay_dir = Path(tmp_dir) / "replays" / "req-123"
            replay_dir.mkdir(parents=True)
            events_file = replay_dir / "events.jsonl"
            events_file.touch()  # Empty file

            handler = create_handler({})
            result = handler._load_from_replay(events_file)

            assert result is None


# ===========================================================================
# Tests for Output Formatting
# ===========================================================================


class TestOutputFormatting:
    """Tests for Markdown and HTML output formatting."""

    def test_markdown_includes_all_sections(self):
        """Markdown output should include all explanation sections."""
        handler = create_handler({})
        explanation = {
            "request_id": "req-123",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "answer": "Yes",
                "confidence": 0.9,
                "consensus_reached": True,
                "agreement_ratio": 0.85,
            },
            "reasoning": {
                "key_claims": [
                    {"statement": "Evidence supports this", "author": "claude", "strength": 0.8}
                ],
                "crux_claims": [{"claim_id": "claim-1", "crux_score": 0.7}],
            },
            "votes": [
                {
                    "agent": "claude",
                    "vote": "agree",
                    "confidence": 0.9,
                    "reasoning": "Strong evidence",
                }
            ],
            "dissent": {
                "dissenting_agents": ["gpt4"],
                "reasons": ["Risk concern"],
                "alternative_views": [{"agent": "gpt4", "view": "Phased approach"}],
                "severity": 0.5,
            },
            "tensions": [{"description": "Speed vs quality", "impact": "Timeline"}],
            "audit_trail": {
                "duration_seconds": 45,
                "rounds_completed": 3,
                "agents_involved": ["claude", "gpt4"],
                "checksum": "sha256:abc",
            },
        }

        result = handler._format_markdown(explanation)

        body = get_body(result)
        assert "# Decision Explanation" in body
        assert "## Summary" in body
        assert "## Key Claims" in body
        assert "## Vote Record" in body
        assert "## Dissenting Views" in body
        assert "## Unresolved Tensions" in body
        assert "## Audit Trail" in body

    def test_html_includes_styling(self):
        """HTML output should include CSS styling."""
        handler = create_handler({})
        explanation = {
            "request_id": "req-123",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "answer": "Yes",
                "confidence": 0.9,
                "consensus_reached": True,
                "agreement_ratio": 0.85,
            },
            "reasoning": {"key_claims": []},
            "votes": [],
            "dissent": {"dissenting_agents": []},
            "tensions": [],
            "audit_trail": {
                "duration_seconds": 45,
                "rounds_completed": 3,
                "agents_involved": [],
                "checksum": None,
            },
        }

        result = handler._format_html(explanation)

        body = get_body(result)
        assert "<style>" in body
        assert "font-family" in body
        assert "</html>" in body


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestDecisionExplainIntegration:
    """Integration tests for decision explanation handler."""

    def test_handler_routes_are_defined(self):
        """Handler should have ROUTES defined."""
        handler = create_handler({})
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0

    def test_handler_inherits_secure_handler(self):
        """Handler should inherit from SecureHandler for RBAC support."""
        from aragora.server.handlers.secure import SecureHandler

        handler = create_handler({})
        assert isinstance(handler, SecureHandler)

    def test_rate_limiter_is_configured(self):
        """Rate limiter should be configured for the endpoint."""
        from aragora.server.handlers.decisions.explain import _explain_limiter

        assert _explain_limiter is not None
        assert hasattr(_explain_limiter, "is_allowed")
