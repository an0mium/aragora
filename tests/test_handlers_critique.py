"""
Tests for the Critique Handler endpoints.

Covers:
- GET /api/critiques/patterns - Get high-impact critique patterns
- GET /api/critiques/archive - Get archive statistics
- GET /api/reputation/all - Get all agent reputations
- GET /api/agent/:name/reputation - Get specific agent reputation
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from aragora.server.handlers.critique import CritiqueHandler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def critique_handler(handler_context):
    """Create a CritiqueHandler with mock context."""
    return CritiqueHandler(handler_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler object."""
    handler = Mock()
    handler.headers = {}
    handler.command = 'GET'
    return handler


@pytest.fixture
def temp_nomic_dir_with_db():
    """Create a temporary nomic directory with a debates.db."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)
        db_path = nomic_dir / "debates.db"

        # Create a minimal database with required tables
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS critique_patterns (
                id INTEGER PRIMARY KEY,
                issue_type TEXT,
                pattern_text TEXT,
                success_rate REAL,
                usage_count INTEGER
            )
        ''')

        # Create reputations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_reputations (
                agent_name TEXT PRIMARY KEY,
                reputation_score REAL,
                vote_weight REAL,
                proposal_acceptance_rate REAL,
                critique_value REAL,
                debates_participated INTEGER
            )
        ''')

        # Insert test data
        cursor.execute('''
            INSERT INTO critique_patterns (issue_type, pattern_text, success_rate, usage_count)
            VALUES (?, ?, ?, ?)
        ''', ('logic', 'Check assumptions', 0.8, 10))

        cursor.execute('''
            INSERT INTO agent_reputations 
            (agent_name, reputation_score, vote_weight, proposal_acceptance_rate, critique_value, debates_participated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('claude', 0.85, 1.2, 0.7, 0.9, 50))

        conn.commit()
        conn.close()

        yield nomic_dir


# =============================================================================
# can_handle Tests
# =============================================================================

class TestCanHandle:
    """Test route matching for CritiqueHandler."""

    def test_can_handle_patterns_route(self, critique_handler):
        """Test matching patterns endpoint."""
        assert critique_handler.can_handle("/api/critiques/patterns")

    def test_can_handle_archive_route(self, critique_handler):
        """Test matching archive endpoint."""
        assert critique_handler.can_handle("/api/critiques/archive")

    def test_can_handle_all_reputations_route(self, critique_handler):
        """Test matching all reputations endpoint."""
        assert critique_handler.can_handle("/api/reputation/all")

    def test_can_handle_agent_reputation_route(self, critique_handler):
        """Test matching agent reputation endpoint."""
        assert critique_handler.can_handle("/api/agent/claude/reputation")
        assert critique_handler.can_handle("/api/agent/gpt4/reputation")

    def test_cannot_handle_unknown_route(self, critique_handler):
        """Test rejection of unknown routes."""
        assert not critique_handler.can_handle("/api/unknown")
        assert not critique_handler.can_handle("/api/critiques")
        assert not critique_handler.can_handle("/api/agent/claude/other")


# =============================================================================
# Patterns Endpoint Tests
# =============================================================================

class TestPatternsEndpoint:
    """Test /api/critiques/patterns endpoint."""

    def test_patterns_critique_store_unavailable_returns_503(self, critique_handler, mock_http_handler):
        """Test error when critique store module unavailable."""
        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False):
            result = critique_handler.handle(
                "/api/critiques/patterns",
                {},
                mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_patterns_no_db_returns_empty(self, handler_context, mock_http_handler):
        """Test empty response when no database exists."""
        handler_context["nomic_dir"] = None
        handler = CritiqueHandler(handler_context)

        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True):
            result = handler.handle(
                "/api/critiques/patterns",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["patterns"] == []
        assert body["count"] == 0

    def test_patterns_with_limit_parameter(self, critique_handler, mock_http_handler):
        """Test limit parameter is respected."""
        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False):
            result = critique_handler.handle(
                "/api/critiques/patterns",
                {"limit": ["5"], "min_success": ["0.7"]},
                mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503  # Store unavailable

    def test_patterns_with_valid_db(self, handler_context, temp_nomic_dir_with_db, mock_http_handler):
        """Test patterns endpoint with valid database."""
        handler_context["nomic_dir"] = temp_nomic_dir_with_db
        handler = CritiqueHandler(handler_context)

        # Mock CritiqueStore
        mock_store = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.issue_type = "logic"
        mock_pattern.pattern_text = "Check assumptions"
        mock_pattern.success_rate = 0.8
        mock_pattern.usage_count = 10
        mock_store.retrieve_patterns.return_value = [mock_pattern]
        mock_store.get_stats.return_value = {"total": 1}

        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True), \
             patch("aragora.server.handlers.critique.CritiqueStore", return_value=mock_store):
            result = handler.handle(
                "/api/critiques/patterns",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["patterns"]) == 1
        assert body["patterns"][0]["issue_type"] == "logic"


# =============================================================================
# Archive Endpoint Tests
# =============================================================================

class TestArchiveEndpoint:
    """Test /api/critiques/archive endpoint."""

    def test_archive_critique_store_unavailable_returns_503(self, critique_handler, mock_http_handler):
        """Test error when critique store module unavailable."""
        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False):
            result = critique_handler.handle(
                "/api/critiques/archive",
                {},
                mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_archive_no_db_returns_empty(self, handler_context, mock_http_handler):
        """Test empty response when no database exists."""
        handler_context["nomic_dir"] = None
        handler = CritiqueHandler(handler_context)

        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True):
            result = handler.handle(
                "/api/critiques/archive",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["archived"] == 0


# =============================================================================
# All Reputations Endpoint Tests
# =============================================================================

class TestAllReputationsEndpoint:
    """Test /api/reputation/all endpoint."""

    def test_reputations_critique_store_unavailable_returns_503(self, critique_handler, mock_http_handler):
        """Test error when critique store module unavailable."""
        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False):
            result = critique_handler.handle(
                "/api/reputation/all",
                {},
                mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_reputations_no_db_returns_empty(self, handler_context, mock_http_handler):
        """Test empty response when no database exists."""
        handler_context["nomic_dir"] = None
        handler = CritiqueHandler(handler_context)

        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True):
            result = handler.handle(
                "/api/reputation/all",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["reputations"] == []
        assert body["count"] == 0

    def test_reputations_with_valid_db(self, handler_context, temp_nomic_dir_with_db, mock_http_handler):
        """Test reputations endpoint with valid database."""
        handler_context["nomic_dir"] = temp_nomic_dir_with_db
        handler = CritiqueHandler(handler_context)

        mock_store = MagicMock()
        mock_rep = MagicMock()
        mock_rep.agent_name = "claude"
        mock_rep.reputation_score = 0.85
        mock_rep.vote_weight = 1.2
        mock_rep.proposal_acceptance_rate = 0.7
        mock_rep.critique_value = 0.9
        mock_rep.debates_participated = 50
        mock_store.get_all_reputations.return_value = [mock_rep]

        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True), \
             patch("aragora.server.handlers.critique.CritiqueStore", return_value=mock_store):
            result = handler.handle(
                "/api/reputation/all",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["reputations"]) == 1
        assert body["reputations"][0]["agent"] == "claude"
        assert body["reputations"][0]["score"] == 0.85


# =============================================================================
# Agent Reputation Endpoint Tests
# =============================================================================

class TestAgentReputationEndpoint:
    """Test /api/agent/:name/reputation endpoint."""

    def test_agent_reputation_critique_store_unavailable_returns_503(self, critique_handler, mock_http_handler):
        """Test error when critique store module unavailable."""
        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False):
            result = critique_handler.handle(
                "/api/agent/claude/reputation",
                {},
                mock_http_handler
            )
        assert result is not None
        assert result.status_code == 503

    def test_agent_reputation_invalid_name_returns_400(self, critique_handler, mock_http_handler):
        """Test error on invalid agent name."""
        result = critique_handler.handle(
            "/api/agent/../passwd/reputation",
            {},
            mock_http_handler
        )
        assert result is not None
        assert result.status_code == 400

    def test_agent_reputation_no_db_returns_no_data(self, handler_context, mock_http_handler):
        """Test response when no database exists."""
        handler_context["nomic_dir"] = None
        handler = CritiqueHandler(handler_context)

        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True):
            result = handler.handle(
                "/api/agent/claude/reputation",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert body["reputation"] is None
        assert "message" in body

    def test_agent_reputation_found(self, handler_context, temp_nomic_dir_with_db, mock_http_handler):
        """Test successful reputation retrieval."""
        handler_context["nomic_dir"] = temp_nomic_dir_with_db
        handler = CritiqueHandler(handler_context)

        mock_store = MagicMock()
        mock_rep = MagicMock()
        mock_rep.reputation_score = 0.85
        mock_rep.vote_weight = 1.2
        mock_rep.proposal_acceptance_rate = 0.7
        mock_rep.critique_value = 0.9
        mock_rep.debates_participated = 50
        mock_store.get_reputation.return_value = mock_rep

        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True), \
             patch("aragora.server.handlers.critique.CritiqueStore", return_value=mock_store):
            result = handler.handle(
                "/api/agent/claude/reputation",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert body["reputation"]["score"] == 0.85

    def test_agent_reputation_not_found(self, handler_context, temp_nomic_dir_with_db, mock_http_handler):
        """Test response when agent not found."""
        handler_context["nomic_dir"] = temp_nomic_dir_with_db
        handler = CritiqueHandler(handler_context)

        mock_store = MagicMock()
        mock_store.get_reputation.return_value = None

        with patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True), \
             patch("aragora.server.handlers.critique.CritiqueStore", return_value=mock_store):
            result = handler.handle(
                "/api/agent/unknown_agent/reputation",
                {},
                mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["reputation"] is None
        assert "not found" in body.get("message", "").lower()


# =============================================================================
# Agent Name Extraction Tests
# =============================================================================

class TestAgentNameExtraction:
    """Test agent name extraction from paths."""

    def test_extract_valid_agent_name(self, critique_handler):
        """Test extraction of valid agent name."""
        agent = critique_handler._extract_agent_name("/api/agent/claude/reputation")
        assert agent == "claude"

    def test_extract_agent_with_version(self, critique_handler):
        """Test extraction of agent name with version suffix."""
        agent = critique_handler._extract_agent_name("/api/agent/gpt4_v2/reputation")
        assert agent == "gpt4_v2"

    def test_extract_path_traversal_returns_none(self, critique_handler):
        """Test path traversal attempts return None."""
        agent = critique_handler._extract_agent_name("/api/agent/../etc/passwd/reputation")
        assert agent is None

    def test_extract_from_short_path_returns_none(self, critique_handler):
        """Test extraction returns None for too-short paths."""
        agent = critique_handler._extract_agent_name("/api/agent")
        assert agent is None
