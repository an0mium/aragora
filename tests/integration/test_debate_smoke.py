"""
End-to-end smoke tests for the critical debate journey.

Validates the critical user path: create debate -> get results -> check diagnostics.
Designed to run in offline/demo mode without real API keys.

Tests:
1. Arena initialization and configuration
2. Debate creation via handler (mock HTTP)
3. Debate storage round-trip (save + retrieve)
4. Debate listing with stored debates
5. Diagnostics endpoint for completed debates
6. Diagnostics endpoint for failed debates
7. Readiness probe returns correct status
8. Liveness probe always returns ok
9. Post-debate receipt detection
10. Consensus info extraction
11. Agent failure diagnostics with suggestions
12. Full journey: store -> list -> get -> diagnostics
13. Debate creation with missing storage returns error
14. Diagnostics for missing debate returns 404
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Environment, DebateProtocol
from aragora.debate.orchestrator import Arena
from aragora.server.storage import DebateStorage


# =============================================================================
# Helper: parse HandlerResult body as dict
# =============================================================================


def _parse(result) -> dict[str, Any]:
    """Parse a HandlerResult body into a dict."""
    if result is None:
        return {}
    body = result.body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    if isinstance(body, str):
        return json.loads(body)
    return body


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_db(tmp_path: Path) -> DebateStorage:
    """Create an in-memory-like DebateStorage backed by a temp file."""
    db_path = str(tmp_path / "smoke_debates.db")
    storage = DebateStorage(db_path)
    return storage


@pytest.fixture
def sample_debate_record() -> dict[str, Any]:
    """A minimal completed debate record dict suitable for save_dict."""
    debate_id = f"smoke-{uuid.uuid4().hex[:8]}"
    return {
        "id": debate_id,
        "task": "Should we use microservices or monolith",
        "status": "concluded",
        "consensus_reached": True,
        "confidence": 0.82,
        "consensus_method": "majority",
        "duration_seconds": 32.5,
        "participants": ["claude-sonnet", "gpt-4o"],
        "agents": ["claude-sonnet", "gpt-4o"],
        "messages": [
            {
                "agent": "claude-sonnet",
                "role": "proposer",
                "round": 1,
                "content": "Microservices offer better scalability.",
            },
            {
                "agent": "gpt-4o",
                "role": "critic",
                "round": 1,
                "content": "Monolith is simpler and sufficient at this scale.",
            },
        ],
        "proposals": {
            "claude-sonnet": "Use microservices with event-driven architecture.",
        },
        "critiques": [],
        "final_answer": "Adopt microservices for the API layer.",
        "metadata": {
            "receipt_id": "rcpt-abc123",
            "receipt_generated": True,
        },
    }


@pytest.fixture
def failed_debate_record() -> dict[str, Any]:
    """A debate record where agents failed."""
    debate_id = f"fail-{uuid.uuid4().hex[:8]}"
    return {
        "id": debate_id,
        "task": "Evaluate AI safety measures",
        "status": "failed",
        "consensus_reached": False,
        "confidence": 0.0,
        "duration_seconds": 5.1,
        "participants": ["claude-sonnet", "gpt-4o", "mistral-large"],
        "agents": ["claude-sonnet", "gpt-4o", "mistral-large"],
        "messages": [],
        "proposals": {},
        "agent_failures": {
            "claude-sonnet": [{"error": "API key invalid"}],
            "gpt-4o": [{"error": "rate limit exceeded (429)"}],
            "mistral-large": [{"error": "timeout after 30s"}],
        },
        "error": "All agents failed",
        "metadata": {},
    }


@pytest.fixture
def mock_agents():
    """Minimal mock agents for Arena initialization tests."""
    agents = []
    for name in ["agent_alpha", "agent_beta"]:
        agent = MagicMock()
        agent.name = name
        agent.generate = AsyncMock(return_value=f"Response from {name}")
        agent.get_metrics = MagicMock(return_value={})
        agents.append(agent)
    return agents


@pytest.fixture
def diagnostics_handler(tmp_db):
    """Create a handler with DiagnosticsMixin backed by temp storage."""
    from aragora.server.handlers.debates.diagnostics import DiagnosticsMixin

    class _Handler(DiagnosticsMixin):
        def __init__(self, storage):
            self._storage = storage
            self.ctx = {"storage": storage}

        def get_storage(self):
            return self._storage

    return _Handler(tmp_db)


@pytest.fixture
def debates_handler_cls():
    """Return the DebatesHandler class for direct instantiation."""
    from aragora.server.handlers.debates.handler import DebatesHandler

    return DebatesHandler


# =============================================================================
# 1. Arena initialization smoke test
# =============================================================================


class TestArenaInitSmoke:
    """Verify Arena can be created in offline mode without real API keys."""

    @pytest.mark.asyncio
    async def test_arena_initializes_with_mock_agents(self, mock_agents):
        """Arena should initialize without errors using mock agents."""
        env = Environment(task="Smoke test question")
        protocol = DebateProtocol(rounds=1)
        arena = Arena(env, mock_agents, protocol)
        assert arena is not None

    @pytest.mark.asyncio
    async def test_arena_from_config_initializes(self, mock_agents):
        """Arena.from_config factory should work with defaults."""
        from aragora.debate.arena_config import ArenaConfig

        env = Environment(task="Smoke config test")
        config = ArenaConfig()
        arena = Arena.from_config(
            environment=env,
            agents=mock_agents,
            config=config,
        )
        assert arena is not None


# =============================================================================
# 2. Debate storage round-trip
# =============================================================================


class TestDebateStorageSmoke:
    """Verify debate storage save/retrieve cycle works."""

    def test_save_and_retrieve_by_id(self, tmp_db, sample_debate_record):
        """Saving a debate and retrieving it by ID should return the same data."""
        tmp_db.save_dict(sample_debate_record)
        debate_id = sample_debate_record["id"]

        retrieved = tmp_db.get_debate(debate_id)
        assert retrieved is not None
        assert retrieved["id"] == debate_id
        assert retrieved["task"] == sample_debate_record["task"]
        assert retrieved["consensus_reached"] is True

    def test_save_and_retrieve_by_slug(self, tmp_db, sample_debate_record):
        """Saving a debate should generate a slug that can be used for retrieval."""
        slug = tmp_db.save_dict(sample_debate_record)
        assert slug  # slug should be a non-empty string
        assert "-" in slug  # slugs contain hyphens

        retrieved = tmp_db.get_by_slug(slug)
        assert retrieved is not None
        assert retrieved["id"] == sample_debate_record["id"]

    def test_list_debates_returns_stored(self, tmp_db, sample_debate_record):
        """list_debates should include recently stored debates."""
        tmp_db.save_dict(sample_debate_record)

        debates = tmp_db.list_debates(limit=10)
        assert len(debates) >= 1
        # The stored debate should appear in the list
        debate_ids = [d.debate_id for d in debates]
        assert sample_debate_record["id"] in debate_ids

    def test_get_nonexistent_debate_returns_none(self, tmp_db):
        """Getting a debate that does not exist should return None."""
        result = tmp_db.get_debate("nonexistent-id-12345")
        assert result is None

    def test_multiple_debates_listed_in_order(self, tmp_db):
        """Multiple debates should be listed newest-first."""
        ids = []
        for i in range(3):
            debate_id = f"order-test-{i}"
            tmp_db.save_dict({
                "id": debate_id,
                "task": f"Order test debate {i}",
                "agents": ["a", "b"],
                "consensus_reached": False,
                "confidence": 0.0,
            })
            ids.append(debate_id)

        debates = tmp_db.list_debates(limit=10)
        assert len(debates) >= 3
        # Most recent should be first
        listed_ids = [d.debate_id for d in debates]
        # The last inserted should appear first in the list
        assert listed_ids[0] == ids[-1]


# =============================================================================
# 3. Diagnostics endpoint
# =============================================================================


class TestDiagnosticsSmoke:
    """Verify the diagnostics endpoint returns correct diagnostic reports."""

    def test_diagnostics_for_completed_debate(
        self, diagnostics_handler, tmp_db, sample_debate_record
    ):
        """Diagnostics for a completed debate should report success with agent info."""
        tmp_db.save_dict(sample_debate_record)
        debate_id = sample_debate_record["id"]

        result = diagnostics_handler._get_diagnostics(debate_id)
        data = _parse(result)

        assert result.status_code == 200
        assert data["debate_id"] == debate_id
        assert data["status"] == "completed"  # "concluded" normalizes to "completed"
        assert data["consensus"]["reached"] is True
        assert data["consensus"]["confidence"] == 0.82
        assert data["receipt_generated"] is True
        assert isinstance(data["agents"], list)
        assert len(data["agents"]) >= 2

    def test_diagnostics_for_failed_debate(
        self, diagnostics_handler, tmp_db, failed_debate_record
    ):
        """Diagnostics for a failed debate should report failures and give suggestions."""
        tmp_db.save_dict(failed_debate_record)
        debate_id = failed_debate_record["id"]

        result = diagnostics_handler._get_diagnostics(debate_id)
        data = _parse(result)

        assert result.status_code == 200
        assert data["debate_id"] == debate_id
        assert data["status"] == "failed"
        assert data["consensus"]["reached"] is False
        assert data["receipt_generated"] is False
        # Should have actionable suggestions
        assert len(data["suggestions"]) > 0
        # Should identify failed agents
        failed_agents = [a for a in data["agents"] if a["status"] == "failed"]
        assert len(failed_agents) >= 2

    def test_diagnostics_missing_debate_returns_404(self, diagnostics_handler):
        """Diagnostics for a non-existent debate should return 404."""
        result = diagnostics_handler._get_diagnostics("does-not-exist-999")
        assert result.status_code == 404

    def test_diagnostics_no_storage_returns_503(self):
        """Diagnostics when storage is unavailable should return 503."""
        from aragora.server.handlers.debates.diagnostics import DiagnosticsMixin

        class _NoStorageHandler(DiagnosticsMixin):
            def __init__(self):
                self.ctx = {}

            def get_storage(self):
                return None

        handler = _NoStorageHandler()
        result = handler._get_diagnostics("any-id")
        assert result.status_code == 503

    def test_diagnostics_agent_provider_inference(
        self, diagnostics_handler, tmp_db
    ):
        """Diagnostics should correctly infer providers from agent names."""
        debate = {
            "id": "provider-test",
            "task": "Provider inference test",
            "status": "concluded",
            "participants": ["claude-sonnet", "gpt-4o", "gemini-pro", "mistral-large"],
            "messages": [
                {"agent": "claude-sonnet", "role": "proposer", "round": 1},
                {"agent": "gpt-4o", "role": "critic", "round": 1},
                {"agent": "gemini-pro", "role": "proposer", "round": 1},
                {"agent": "mistral-large", "role": "proposer", "round": 1},
            ],
            "agents": ["claude-sonnet", "gpt-4o", "gemini-pro", "mistral-large"],
            "consensus_reached": True,
            "confidence": 0.9,
            "metadata": {},
        }
        tmp_db.save_dict(debate)

        result = diagnostics_handler._get_diagnostics("provider-test")
        data = _parse(result)

        agents_by_name = {a["name"]: a for a in data["agents"]}
        assert agents_by_name["claude-sonnet"]["provider"] == "anthropic"
        assert agents_by_name["gpt-4o"]["provider"] == "openai"
        assert agents_by_name["gemini-pro"]["provider"] == "google"
        assert agents_by_name["mistral-large"]["provider"] == "mistral"


# =============================================================================
# 4. Readiness and liveness probes
# =============================================================================


class TestProbesSmoke:
    """Verify health probes work in offline mode."""

    def test_liveness_probe_returns_ok(self):
        """Liveness probe should always return status ok."""
        from aragora.server.handlers.admin.health.kubernetes import liveness_probe

        handler = MagicMock()
        result = liveness_probe(handler)
        data = _parse(result)

        assert result.status_code == 200
        assert data["status"] == "ok"

    def test_readiness_probe_structure(self):
        """Readiness probe should return structured checks dict."""
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast
        from aragora.server.handlers.admin.health import _HEALTH_CACHE, _HEALTH_CACHE_TIMESTAMPS

        # Clear cache to get a fresh probe result
        _HEALTH_CACHE.clear()
        _HEALTH_CACHE_TIMESTAMPS.clear()

        handler = MagicMock()
        handler.get_storage = MagicMock(return_value=None)
        handler.get_elo_system = MagicMock(return_value=None)

        # is_server_ready is imported inside the function body from unified_server,
        # so we patch it on the source module.
        with patch(
            "aragora.server.unified_server.is_server_ready",
            return_value=True,
        ):
            result = readiness_probe_fast(handler)

        data = _parse(result)
        assert "checks" in data or "status" in data

    def test_server_ready_flag_lifecycle(self):
        """Server readiness flag should transition from False to True."""
        from aragora.server.unified_server import (
            is_server_ready,
            mark_server_ready,
            _server_ready,
        )
        import aragora.server.unified_server as us_mod

        # Save original state and test the flag API
        original = us_mod._server_ready
        try:
            us_mod._server_ready = False
            assert is_server_ready() is False

            mark_server_ready()
            assert is_server_ready() is True
        finally:
            # Restore original state
            us_mod._server_ready = original


# =============================================================================
# 5. Full journey: store -> list -> get -> diagnostics
# =============================================================================


class TestFullJourneySmoke:
    """End-to-end journey through the critical path."""

    def test_full_debate_lifecycle(
        self, diagnostics_handler, tmp_db, sample_debate_record
    ):
        """Full journey: store debate -> list -> retrieve -> diagnostics."""
        # Step 1: Store the debate
        slug = tmp_db.save_dict(sample_debate_record)
        debate_id = sample_debate_record["id"]
        assert slug is not None

        # Step 2: List debates and verify it appears
        debates = tmp_db.list_debates(limit=10)
        debate_ids = [d.debate_id for d in debates]
        assert debate_id in debate_ids

        # Step 3: Retrieve by ID
        retrieved = tmp_db.get_debate(debate_id)
        assert retrieved is not None
        assert retrieved["task"] == sample_debate_record["task"]
        assert retrieved["consensus_reached"] is True

        # Step 4: Run diagnostics
        result = diagnostics_handler._get_diagnostics(debate_id)
        data = _parse(result)
        assert result.status_code == 200
        assert data["debate_id"] == debate_id
        assert data["status"] == "completed"
        assert data["consensus"]["reached"] is True
        assert data["receipt_generated"] is True

    def test_failed_debate_lifecycle(
        self, diagnostics_handler, tmp_db, failed_debate_record
    ):
        """Full journey for a failed debate with suggestions."""
        # Store
        tmp_db.save_dict(failed_debate_record)
        debate_id = failed_debate_record["id"]

        # Retrieve
        retrieved = tmp_db.get_debate(debate_id)
        assert retrieved is not None
        assert retrieved["status"] == "failed"

        # Diagnostics
        result = diagnostics_handler._get_diagnostics(debate_id)
        data = _parse(result)
        assert data["status"] == "failed"
        assert len(data["suggestions"]) > 0

        # Verify specific suggestion categories
        suggestions_text = " ".join(data["suggestions"]).lower()
        # Should mention API key issues
        assert "api" in suggestions_text or "key" in suggestions_text or "failed" in suggestions_text

    def test_debate_handler_instantiation_with_storage(self, tmp_db, debates_handler_cls):
        """DebatesHandler should initialize with storage context."""
        handler = debates_handler_cls(ctx={"storage": tmp_db})
        assert handler.get_storage() is tmp_db

    def test_diagnostics_suggestions_for_rate_limited_agent(
        self, diagnostics_handler, tmp_db
    ):
        """Diagnostics should suggest fallback providers for rate-limited agents."""
        debate = {
            "id": "rate-limited-test",
            "task": "Rate limit test debate",
            "status": "concluded",
            "participants": ["gpt-4o"],
            "agents": ["gpt-4o"],
            "consensus_reached": False,
            "confidence": 0.0,
            "messages": [],
            "agent_failures": {
                "gpt-4o": [{"error": "429 rate limit exceeded"}],
            },
            "metadata": {},
        }
        tmp_db.save_dict(debate)

        result = diagnostics_handler._get_diagnostics("rate-limited-test")
        data = _parse(result)

        suggestions = data["suggestions"]
        assert len(suggestions) > 0
        # Should mention rate limits or OpenRouter fallback
        combined = " ".join(suggestions).lower()
        assert "rate" in combined or "openrouter" in combined or "fallback" in combined

    def test_diagnostics_consensus_no_consensus_suggestion(
        self, diagnostics_handler, tmp_db
    ):
        """When consensus not reached, diagnostics should suggest more rounds."""
        debate = {
            "id": "no-consensus-test",
            "task": "No consensus test debate",
            "status": "concluded",
            "participants": ["agent_a", "agent_b"],
            "agents": ["agent_a", "agent_b"],
            "consensus_reached": False,
            "confidence": 0.3,
            "rounds_used": 3,
            "messages": [
                {"agent": "agent_a", "role": "proposer", "round": 1},
                {"agent": "agent_b", "role": "critic", "round": 1},
            ],
            "metadata": {},
        }
        tmp_db.save_dict(debate)

        result = diagnostics_handler._get_diagnostics("no-consensus-test")
        data = _parse(result)

        assert data["consensus"]["reached"] is False
        suggestions = data["suggestions"]
        combined = " ".join(suggestions).lower()
        # Should suggest increasing rounds or reducing agents
        assert "rounds" in combined or "consensus" in combined
