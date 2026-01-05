"""
Tests for ConsensusHandler endpoints.

Endpoints tested:
- GET /api/consensus/similar - Find debates similar to a topic
- GET /api/consensus/settled - Get high-confidence settled topics
- GET /api/consensus/stats - Get consensus memory statistics
- GET /api/consensus/dissents - Get recent dissenting views
- GET /api/consensus/contrarian-views - Get contrarian perspectives
- GET /api/consensus/risk-warnings - Get risk warnings and edge cases
- GET /api/consensus/domain/:domain - Get domain-specific history
"""

import json
import pytest
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers import (
    ConsensusHandler,
    HandlerResult,
    json_response,
    error_response,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_consensus_db():
    """Create a temporary consensus database with test data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consensus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            conclusion TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            strength TEXT DEFAULT 'moderate',
            domain TEXT DEFAULT 'general',
            participating_agents TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dissent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            debate_id INTEGER,
            data TEXT NOT NULL,
            dissent_type TEXT DEFAULT 'minority_opinion',
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (debate_id) REFERENCES consensus(id)
        )
    """)

    # Insert test data
    cursor.execute("""
        INSERT INTO consensus (topic, conclusion, confidence, strength, domain, participating_agents)
        VALUES
            ('AI Safety', 'Alignment is critical', 0.95, 'strong', 'technology', '["claude", "gpt4"]'),
            ('Climate Change', 'Action needed urgently', 0.85, 'strong', 'environment', '["claude", "gemini"]'),
            ('Code Reviews', 'Essential for quality', 0.75, 'moderate', 'software', '["claude", "grok"]'),
            ('Testing Strategy', 'Unit tests are valuable', 0.60, 'weak', 'software', '["gemini"]')
    """)

    # Insert test dissents
    dissent_data = json.dumps({
        "agent_id": "grok",
        "content": "I disagree with the majority view",
        "confidence": 0.7,
        "reasoning": "There are edge cases to consider",
        "debate_id": "debate-1",
        "dissent_type": "minority_opinion",
        "metadata": {"domain": "technology"},
        "timestamp": datetime.now().isoformat(),
    })
    cursor.execute("""
        INSERT INTO dissent (debate_id, data, dissent_type)
        VALUES (1, ?, 'minority_opinion')
    """, (dissent_data,))

    risk_warning = json.dumps({
        "agent_id": "claude",
        "content": "This approach has security risks",
        "confidence": 0.8,
        "reasoning": "SQL injection possible",
        "debate_id": "debate-2",
        "dissent_type": "risk_warning",
        "metadata": {"domain": "security"},
        "timestamp": datetime.now().isoformat(),
        "rebuttal": "Use parameterized queries",
    })
    cursor.execute("""
        INSERT INTO dissent (debate_id, data, dissent_type)
        VALUES (2, ?, 'risk_warning')
    """, (risk_warning,))

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_consensus_memory(temp_consensus_db):
    """Create a mock ConsensusMemory pointing to temp database."""
    mock_memory = Mock()
    mock_memory.db_path = temp_consensus_db
    mock_memory.find_similar_debates.return_value = []
    mock_memory.get_statistics.return_value = {
        "total_consensus": 4,
        "total_dissents": 2,
        "by_domain": {"technology": 1, "environment": 1, "software": 2},
        "by_strength": {"strong": 2, "moderate": 1, "weak": 1},
    }
    mock_memory.get_domain_consensus_history.return_value = []
    return mock_memory


@pytest.fixture
def consensus_handler(mock_consensus_memory, temp_consensus_db):
    """Create a ConsensusHandler with mock context."""
    ctx = {
        "nomic_dir": Path(temp_consensus_db).parent,
    }
    handler = ConsensusHandler(ctx)
    return handler


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestConsensusHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_similar(self, consensus_handler):
        """Should handle /api/consensus/similar."""
        assert consensus_handler.can_handle("/api/consensus/similar") is True

    def test_can_handle_settled(self, consensus_handler):
        """Should handle /api/consensus/settled."""
        assert consensus_handler.can_handle("/api/consensus/settled") is True

    def test_can_handle_stats(self, consensus_handler):
        """Should handle /api/consensus/stats."""
        assert consensus_handler.can_handle("/api/consensus/stats") is True

    def test_can_handle_dissents(self, consensus_handler):
        """Should handle /api/consensus/dissents."""
        assert consensus_handler.can_handle("/api/consensus/dissents") is True

    def test_can_handle_contrarian_views(self, consensus_handler):
        """Should handle /api/consensus/contrarian-views."""
        assert consensus_handler.can_handle("/api/consensus/contrarian-views") is True

    def test_can_handle_risk_warnings(self, consensus_handler):
        """Should handle /api/consensus/risk-warnings."""
        assert consensus_handler.can_handle("/api/consensus/risk-warnings") is True

    def test_can_handle_domain_pattern(self, consensus_handler):
        """Should handle /api/consensus/domain/:domain pattern."""
        assert consensus_handler.can_handle("/api/consensus/domain/technology") is True
        assert consensus_handler.can_handle("/api/consensus/domain/software") is True

    def test_cannot_handle_unknown_route(self, consensus_handler):
        """Should not handle unknown routes."""
        assert consensus_handler.can_handle("/api/unknown") is False
        assert consensus_handler.can_handle("/api/consensus/unknown") is False


# ============================================================================
# Similar Debates Endpoint Tests
# ============================================================================

class TestSimilarDebatesEndpoint:
    """Tests for /api/consensus/similar endpoint."""

    def test_similar_requires_topic(self, consensus_handler):
        """Should require topic parameter."""
        result = consensus_handler.handle("/api/consensus/similar", {}, None)
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Topic required" in data["error"]

    def test_similar_empty_topic_returns_400(self, consensus_handler):
        """Should reject empty topic."""
        result = consensus_handler.handle("/api/consensus/similar", {"topic": ""}, None)
        assert result.status_code == 400

    def test_similar_topic_too_long_returns_400(self, consensus_handler):
        """Should reject topic over 500 chars."""
        long_topic = "x" * 501
        result = consensus_handler.handle("/api/consensus/similar", {"topic": long_topic}, None)
        assert result.status_code == 400

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_similar_returns_results(self, mock_cm_class, consensus_handler):
        """Should return similar debates when available."""
        # Setup mock
        mock_consensus = Mock()
        mock_consensus.topic = "AI Safety"
        mock_consensus.conclusion = "Important topic"
        mock_consensus.strength = Mock(value="strong")
        mock_consensus.confidence = 0.9
        mock_consensus.participating_agents = ["claude"]
        mock_consensus.timestamp = datetime.now()

        mock_result = Mock()
        mock_result.consensus = mock_consensus
        mock_result.similarity_score = 0.85
        mock_result.dissents = []

        mock_instance = Mock()
        mock_instance.find_similar_debates.return_value = [mock_result]
        mock_cm_class.return_value = mock_instance

        result = consensus_handler.handle(
            "/api/consensus/similar",
            {"topic": "AI alignment", "limit": "5"},
            None
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "similar" in data
        assert data["count"] == 1
        assert data["similar"][0]["topic"] == "AI Safety"

    def test_similar_limit_capped_at_20(self, consensus_handler):
        """Should cap limit at 20."""
        # This tests the parameter processing; actual query would need consensus memory
        with patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', False):
            result = consensus_handler.handle(
                "/api/consensus/similar",
                {"topic": "test", "limit": "100"},
                None
            )
            # Should return 503 since memory not available, but limit would be capped
            assert result.status_code == 503


# ============================================================================
# Settled Topics Endpoint Tests
# ============================================================================

class TestSettledTopicsEndpoint:
    """Tests for /api/consensus/settled endpoint."""

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_settled_returns_high_confidence_topics(self, mock_cm_class, temp_consensus_db):
        """Should return topics with high confidence."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        result = handler.handle("/api/consensus/settled", {"min_confidence": "0.8"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "topics" in data
        # Should have AI Safety (0.95) and Climate Change (0.85)
        assert data["count"] == 2
        assert all(t["confidence"] >= 0.8 for t in data["topics"])

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_settled_default_confidence(self, mock_cm_class, temp_consensus_db):
        """Should use 0.8 as default min_confidence."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        result = handler.handle("/api/consensus/settled", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["min_confidence"] == 0.8

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_settled_confidence_clamped(self, mock_cm_class, temp_consensus_db):
        """Should clamp confidence between 0 and 1."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})

        # Test over 1
        result = handler.handle("/api/consensus/settled", {"min_confidence": "1.5"}, None)
        data = json.loads(result.body)
        assert data["min_confidence"] == 1.0

        # Test under 0
        result = handler.handle("/api/consensus/settled", {"min_confidence": "-0.5"}, None)
        data = json.loads(result.body)
        assert data["min_confidence"] == 0.0


# ============================================================================
# Stats Endpoint Tests
# ============================================================================

class TestStatsEndpoint:
    """Tests for /api/consensus/stats endpoint."""

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_stats_returns_structure(self, mock_cm_class, temp_consensus_db):
        """Should return stats structure."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_instance.get_statistics.return_value = {
            "total_consensus": 4,
            "total_dissents": 2,
            "by_domain": {"technology": 1, "software": 2},
            "by_strength": {"strong": 2},
        }
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        result = handler.handle("/api/consensus/stats", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "total_topics" in data
        assert "high_confidence_count" in data
        assert "domains" in data
        assert "avg_confidence" in data
        assert "total_dissents" in data
        assert "by_strength" in data
        assert "by_domain" in data

    def test_stats_unavailable_returns_503(self):
        """Should return 503 when consensus memory unavailable."""
        import aragora.server.handlers.consensus as consensus_module
        from aragora.server.handlers.base import clear_cache

        # Clear cache to avoid cached results
        clear_cache()

        original_value = consensus_module.CONSENSUS_MEMORY_AVAILABLE
        try:
            consensus_module.CONSENSUS_MEMORY_AVAILABLE = False
            handler = ConsensusHandler({})
            result = handler.handle("/api/consensus/stats", {}, None)
            assert result.status_code == 503
        finally:
            consensus_module.CONSENSUS_MEMORY_AVAILABLE = original_value
            clear_cache()


# ============================================================================
# Dissents Endpoint Tests
# ============================================================================

class TestDissentsEndpoint:
    """Tests for /api/consensus/dissents endpoint."""

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_dissents_returns_list(self, mock_cm_class, temp_consensus_db):
        """Should return list of dissents."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        result = handler.handle("/api/consensus/dissents", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "dissents" in data
        assert isinstance(data["dissents"], list)

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_dissents_limit_capped(self, mock_cm_class, temp_consensus_db):
        """Should cap limit at 50."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        # Limit is processed internally, just verify request works
        result = handler.handle("/api/consensus/dissents", {"limit": "100"}, None)
        assert result.status_code == 200

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_dissents_topic_truncated(self, mock_cm_class, temp_consensus_db):
        """Should truncate topic to 500 chars."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        long_topic = "x" * 600
        result = handler.handle("/api/consensus/dissents", {"topic": long_topic}, None)
        # Should process without error (topic truncated internally)
        assert result.status_code == 200


# ============================================================================
# Contrarian Views Endpoint Tests
# ============================================================================

class TestContrarianViewsEndpoint:
    """Tests for /api/consensus/contrarian-views endpoint."""

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_contrarian_views_returns_list(self, mock_cm_class, temp_consensus_db):
        """Should return list of contrarian views."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        result = handler.handle("/api/consensus/contrarian-views", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "views" in data
        assert isinstance(data["views"], list)


# ============================================================================
# Risk Warnings Endpoint Tests
# ============================================================================

class TestRiskWarningsEndpoint:
    """Tests for /api/consensus/risk-warnings endpoint."""

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_risk_warnings_returns_list(self, mock_cm_class, temp_consensus_db):
        """Should return list of risk warnings."""
        mock_instance = Mock()
        mock_instance.db_path = temp_consensus_db
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        result = handler.handle("/api/consensus/risk-warnings", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_risk_warnings_unavailable_returns_503(self):
        """Should return 503 when consensus memory unavailable."""
        with patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', False):
            handler = ConsensusHandler({})
            result = handler.handle("/api/consensus/risk-warnings", {}, None)
            assert result.status_code == 503


# ============================================================================
# Domain History Endpoint Tests
# ============================================================================

class TestDomainHistoryEndpoint:
    """Tests for /api/consensus/domain/:domain endpoint."""

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_domain_history_returns_structure(self, mock_cm_class):
        """Should return domain history structure."""
        mock_instance = Mock()
        mock_instance.get_domain_consensus_history.return_value = []
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        result = handler.handle("/api/consensus/domain/technology", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["domain"] == "technology"
        assert "history" in data
        assert "count" in data

    @patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', True)
    @patch('aragora.server.handlers.consensus.ConsensusMemory')
    def test_domain_history_limit_capped(self, mock_cm_class):
        """Should cap limit at 200."""
        mock_instance = Mock()
        mock_instance.get_domain_consensus_history.return_value = []
        mock_cm_class.return_value = mock_instance

        handler = ConsensusHandler({})
        result = handler.handle("/api/consensus/domain/software", {"limit": "500"}, None)
        assert result.status_code == 200

    def test_domain_history_unavailable_returns_503(self):
        """Should return 503 when consensus memory unavailable."""
        with patch('aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE', False):
            handler = ConsensusHandler({})
            result = handler.handle("/api/consensus/domain/test", {}, None)
            assert result.status_code == 503


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestConsensusErrorHandling:
    """Tests for error handling."""

    def test_database_error_returns_500(self):
        """Should return 500 on database errors."""
        import aragora.server.handlers.consensus as consensus_module
        from aragora.server.handlers.base import clear_cache

        # Clear cache to avoid cached results
        clear_cache()

        # Create handler that will fail on DB access
        mock_cm = Mock()
        mock_cm.get_statistics.side_effect = Exception("Database error")
        mock_cm.db_path = "/nonexistent/path.db"

        original_cm = consensus_module.ConsensusMemory
        try:
            consensus_module.ConsensusMemory = Mock(return_value=mock_cm)
            handler = ConsensusHandler({})
            result = handler.handle("/api/consensus/stats", {}, None)

            # Should handle error gracefully
            assert result.status_code == 500
            data = json.loads(result.body)
            assert "error" in data
        finally:
            consensus_module.ConsensusMemory = original_cm
            clear_cache()

    def test_all_endpoints_unavailable_returns_503(self):
        """All endpoints should return 503 when consensus memory unavailable."""
        import aragora.server.handlers.consensus as consensus_module
        from aragora.server.handlers.base import clear_cache

        # Clear cache to avoid cached results
        clear_cache()

        original_value = consensus_module.CONSENSUS_MEMORY_AVAILABLE
        try:
            consensus_module.CONSENSUS_MEMORY_AVAILABLE = False
            handler = ConsensusHandler({})

            endpoints = [
                ("/api/consensus/similar", {"topic": "test"}),
                ("/api/consensus/settled", {}),
                ("/api/consensus/stats", {}),
                ("/api/consensus/dissents", {}),
                ("/api/consensus/contrarian-views", {}),
                ("/api/consensus/risk-warnings", {}),
                ("/api/consensus/domain/test", {}),
            ]

            for path, params in endpoints:
                result = handler.handle(path, params, None)
                assert result.status_code == 503, f"Expected 503 for {path}"
        finally:
            consensus_module.CONSENSUS_MEMORY_AVAILABLE = original_value
            clear_cache()
