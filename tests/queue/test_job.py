"""
Tests for debate job definitions module.

Tests cover:
- DebateJobPayload creation, defaults, serialization via SerializableMixin
- create_debate_job factory function (defaults, custom values, truncation)
- get_debate_payload extraction from a Job
- DebateResult creation, to_dict, from_dict round-trip
"""

from __future__ import annotations

from aragora.queue.base import Job
from aragora.queue.job import (
    DebateJobPayload,
    DebateResult,
    create_debate_job,
    get_debate_payload,
)


class TestDebateJobPayload:
    """Tests for DebateJobPayload dataclass."""

    def test_creation_with_question(self):
        payload = DebateJobPayload(question="Should we adopt Rust?")
        assert payload.question == "Should we adopt Rust?"
        assert payload.protocol == "standard"
        assert payload.timeout_seconds is None
        assert payload.webhook_url is None
        assert payload.user_id is None
        assert payload.organization_id is None

    def test_default_agents(self):
        payload = DebateJobPayload(question="Test")
        assert isinstance(payload.agents, list)
        assert len(payload.agents) > 0

    def test_custom_agents(self):
        payload = DebateJobPayload(question="Test", agents=["claude", "gpt"])
        assert payload.agents == ["claude", "gpt"]

    def test_to_dict(self):
        payload = DebateJobPayload(
            question="Test question",
            agents=["claude"],
            rounds=5,
            consensus="majority",
        )
        data = payload.to_dict()
        assert data["question"] == "Test question"
        assert data["agents"] == ["claude"]
        assert data["rounds"] == 5
        assert data["consensus"] == "majority"

    def test_from_dict(self):
        data = {
            "question": "Restored question",
            "agents": ["gpt", "gemini"],
            "rounds": 3,
            "consensus": "unanimous",
            "protocol": "adversarial",
            "timeout_seconds": 120,
            "webhook_url": "https://example.com/hook",
            "user_id": "user-123",
            "organization_id": "org-456",
            "metadata": {"tag": "important"},
        }
        payload = DebateJobPayload.from_dict(data)
        assert payload.question == "Restored question"
        assert payload.agents == ["gpt", "gemini"]
        assert payload.rounds == 3
        assert payload.consensus == "unanimous"
        assert payload.protocol == "adversarial"
        assert payload.timeout_seconds == 120
        assert payload.webhook_url == "https://example.com/hook"
        assert payload.user_id == "user-123"
        assert payload.organization_id == "org-456"

    def test_round_trip(self):
        original = DebateJobPayload(
            question="Round-trip test",
            agents=["claude", "gpt"],
            rounds=7,
            consensus="judge",
            user_id="u-1",
        )
        data = original.to_dict()
        restored = DebateJobPayload.from_dict(data)
        assert restored.question == original.question
        assert restored.agents == original.agents
        assert restored.rounds == original.rounds
        assert restored.consensus == original.consensus
        assert restored.user_id == original.user_id


class TestCreateDebateJob:
    """Tests for create_debate_job factory function."""

    def test_basic_creation(self):
        job = create_debate_job(question="Is Python better than Java?")
        assert isinstance(job, Job)
        assert job.id is not None
        assert job.metadata["job_type"] == "debate"

    def test_payload_contains_question(self):
        job = create_debate_job(question="Test question")
        payload = DebateJobPayload.from_dict(job.payload)
        assert payload.question == "Test question"

    def test_custom_agents(self):
        job = create_debate_job(
            question="Test",
            agents=["claude", "gemini"],
        )
        payload = DebateJobPayload.from_dict(job.payload)
        assert payload.agents == ["claude", "gemini"]

    def test_custom_rounds_and_consensus(self):
        job = create_debate_job(
            question="Test",
            rounds=5,
            consensus="unanimous",
        )
        payload = DebateJobPayload.from_dict(job.payload)
        assert payload.rounds == 5
        assert payload.consensus == "unanimous"

    def test_priority(self):
        job = create_debate_job(question="Test", priority=10)
        assert job.priority == 10

    def test_max_attempts(self):
        job = create_debate_job(question="Test", max_attempts=5)
        assert job.max_attempts == 5

    def test_optional_fields(self):
        job = create_debate_job(
            question="Test",
            timeout_seconds=300,
            webhook_url="https://example.com/cb",
            user_id="user-1",
            organization_id="org-1",
        )
        payload = DebateJobPayload.from_dict(job.payload)
        assert payload.timeout_seconds == 300
        assert payload.webhook_url == "https://example.com/cb"
        assert payload.user_id == "user-1"
        assert payload.organization_id == "org-1"

    def test_question_preview_short(self):
        job = create_debate_job(question="Short question")
        assert job.metadata["question_preview"] == "Short question"

    def test_question_preview_truncated(self):
        long_question = "A" * 200
        job = create_debate_job(question=long_question)
        assert len(job.metadata["question_preview"]) == 100
        assert job.metadata["question_preview"] == "A" * 100

    def test_metadata_passed_through(self):
        job = create_debate_job(
            question="Test",
            metadata={"custom_key": "custom_value"},
        )
        payload = DebateJobPayload.from_dict(job.payload)
        assert payload.metadata == {"custom_key": "custom_value"}

    def test_unique_ids(self):
        job1 = create_debate_job(question="Q1")
        job2 = create_debate_job(question="Q2")
        assert job1.id != job2.id


class TestGetDebatePayload:
    """Tests for get_debate_payload extraction."""

    def test_extract_from_job(self):
        job = create_debate_job(
            question="Extract me",
            agents=["claude"],
            rounds=3,
        )
        payload = get_debate_payload(job)
        assert isinstance(payload, DebateJobPayload)
        assert payload.question == "Extract me"
        assert payload.agents == ["claude"]
        assert payload.rounds == 3

    def test_extract_preserves_all_fields(self):
        job = create_debate_job(
            question="Full extraction",
            agents=["gpt"],
            rounds=5,
            consensus="majority",
            protocol="adversarial",
            timeout_seconds=60,
            webhook_url="https://hook.example.com",
            user_id="u-42",
            organization_id="org-7",
        )
        payload = get_debate_payload(job)
        assert payload.consensus == "majority"
        assert payload.protocol == "adversarial"
        assert payload.timeout_seconds == 60
        assert payload.webhook_url == "https://hook.example.com"
        assert payload.user_id == "u-42"
        assert payload.organization_id == "org-7"


class TestDebateResult:
    """Tests for DebateResult dataclass."""

    def test_creation(self):
        result = DebateResult(
            debate_id="d-123",
            consensus_reached=True,
            final_answer="Yes, adopt Rust",
            confidence=0.95,
            rounds_used=5,
            participants=["claude", "gpt"],
            duration_seconds=45.5,
        )
        assert result.debate_id == "d-123"
        assert result.consensus_reached is True
        assert result.final_answer == "Yes, adopt Rust"
        assert result.confidence == 0.95
        assert result.rounds_used == 5
        assert result.participants == ["claude", "gpt"]
        assert result.duration_seconds == 45.5
        assert result.error is None

    def test_creation_with_error(self):
        result = DebateResult(
            debate_id="d-456",
            consensus_reached=False,
            final_answer=None,
            confidence=0.0,
            rounds_used=9,
            participants=["claude"],
            duration_seconds=120.0,
            error="Timeout exceeded",
        )
        assert result.consensus_reached is False
        assert result.final_answer is None
        assert result.error == "Timeout exceeded"

    def test_to_dict(self):
        result = DebateResult(
            debate_id="d-789",
            consensus_reached=True,
            final_answer="Answer",
            confidence=0.8,
            rounds_used=3,
            participants=["claude", "gpt"],
            duration_seconds=30.0,
            token_usage={"input": 1000, "output": 500},
        )
        data = result.to_dict()
        assert data["debate_id"] == "d-789"
        assert data["consensus_reached"] is True
        assert data["final_answer"] == "Answer"
        assert data["confidence"] == 0.8
        assert data["rounds_used"] == 3
        assert data["participants"] == ["claude", "gpt"]
        assert data["duration_seconds"] == 30.0
        assert data["token_usage"] == {"input": 1000, "output": 500}
        assert data["error"] is None

    def test_from_dict(self):
        data = {
            "debate_id": "d-abc",
            "consensus_reached": False,
            "final_answer": "Maybe",
            "confidence": 0.6,
            "rounds_used": 7,
            "participants": ["gemini"],
            "duration_seconds": 90.0,
            "token_usage": {"total": 2000},
            "error": None,
        }
        result = DebateResult.from_dict(data)
        assert result.debate_id == "d-abc"
        assert result.consensus_reached is False
        assert result.final_answer == "Maybe"
        assert result.confidence == 0.6
        assert result.rounds_used == 7
        assert result.participants == ["gemini"]

    def test_from_dict_defaults(self):
        data = {
            "debate_id": "d-min",
            "consensus_reached": True,
        }
        result = DebateResult.from_dict(data)
        assert result.debate_id == "d-min"
        assert result.consensus_reached is True
        assert result.final_answer is None
        assert result.confidence == 0.0
        assert result.rounds_used == 0
        assert result.participants == []
        assert result.duration_seconds == 0.0
        assert result.token_usage == {}
        assert result.error is None

    def test_round_trip(self):
        original = DebateResult(
            debate_id="d-rt",
            consensus_reached=True,
            final_answer="Round-trip",
            confidence=0.99,
            rounds_used=9,
            participants=["claude", "gpt", "gemini"],
            duration_seconds=180.0,
            token_usage={"input": 5000, "output": 2000},
        )
        data = original.to_dict()
        restored = DebateResult.from_dict(data)
        assert restored.debate_id == original.debate_id
        assert restored.consensus_reached == original.consensus_reached
        assert restored.final_answer == original.final_answer
        assert restored.confidence == original.confidence
        assert restored.rounds_used == original.rounds_used
        assert restored.participants == original.participants
        assert restored.duration_seconds == original.duration_seconds
        assert restored.token_usage == original.token_usage
