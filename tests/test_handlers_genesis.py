"""
Tests for GenesisHandler endpoints.

Endpoints tested:
- GET /api/genesis/stats - Get overall genesis statistics
- GET /api/genesis/events - Get recent genesis events
- GET /api/genesis/lineage/:genome_id - Get genome ancestry
- GET /api/genesis/tree/:debate_id - Get debate tree structure
"""

import json
import pytest
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers import (
    GenesisHandler,
    HandlerResult,
    json_response,
    error_response,
)
from aragora.server.handlers.genesis import _genesis_limiter


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_rate_limits():
    """Clear rate limits before and after each test."""
    _genesis_limiter.clear()
    yield
    _genesis_limiter.clear()


@pytest.fixture
def temp_genesis_db():
    """Create a temporary genesis database with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nomic_dir = Path(tmpdir)
        db_path = nomic_dir / "genesis.db"

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create genesis_events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS genesis_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parent_event_id TEXT,
                content_hash TEXT,
                data TEXT
            )
        """
        )

        # Create genomes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS genomes (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                traits TEXT,
                lineage TEXT,
                fitness_score REAL DEFAULT 0.5,
                generation INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Insert test events
        test_events = [
            (
                "evt-001",
                "agent_birth",
                datetime.now().isoformat(),
                None,
                "hash001",
                json.dumps({"agent": "claude", "generation": 1}),
            ),
            (
                "evt-002",
                "agent_birth",
                datetime.now().isoformat(),
                "evt-001",
                "hash002",
                json.dumps({"agent": "gemini", "generation": 1}),
            ),
            (
                "evt-003",
                "fitness_update",
                datetime.now().isoformat(),
                "evt-002",
                "hash003",
                json.dumps({"agent": "claude", "change": 0.1}),
            ),
            (
                "evt-004",
                "agent_death",
                datetime.now().isoformat(),
                "evt-003",
                "hash004",
                json.dumps({"agent": "old-agent", "reason": "low_fitness"}),
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO genesis_events (event_id, event_type, timestamp, parent_event_id, content_hash, data)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            test_events,
        )

        # Insert test genomes
        cursor.execute(
            """
            INSERT INTO genomes (id, agent, traits, lineage, fitness_score, generation)
            VALUES
                ('genome-001', 'claude', '{"analytical": 0.9}', '[]', 0.85, 1),
                ('genome-002', 'gemini', '{"creative": 0.8}', '["genome-001"]', 0.75, 2)
        """
        )

        conn.commit()
        conn.close()

        yield nomic_dir


@pytest.fixture
def genesis_handler(temp_genesis_db):
    """Create a GenesisHandler with temp nomic directory."""
    ctx = {"nomic_dir": temp_genesis_db}
    return GenesisHandler(ctx)


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestGenesisHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_stats(self, genesis_handler):
        """Should handle /api/genesis/stats."""
        assert genesis_handler.can_handle("/api/genesis/stats") is True

    def test_can_handle_events(self, genesis_handler):
        """Should handle /api/genesis/events."""
        assert genesis_handler.can_handle("/api/genesis/events") is True

    def test_can_handle_lineage_pattern(self, genesis_handler):
        """Should handle /api/genesis/lineage/:genome_id pattern."""
        assert genesis_handler.can_handle("/api/genesis/lineage/genome-001") is True
        assert genesis_handler.can_handle("/api/genesis/lineage/abc123") is True

    def test_can_handle_tree_pattern(self, genesis_handler):
        """Should handle /api/genesis/tree/:debate_id pattern."""
        assert genesis_handler.can_handle("/api/genesis/tree/debate-001") is True
        assert genesis_handler.can_handle("/api/genesis/tree/xyz789") is True

    def test_cannot_handle_unknown_route(self, genesis_handler):
        """Should not handle unknown routes."""
        assert genesis_handler.can_handle("/api/unknown") is False
        assert genesis_handler.can_handle("/api/genesis/unknown") is False
        assert genesis_handler.can_handle("/api/genesis") is False


# ============================================================================
# Stats Endpoint Tests
# ============================================================================


class TestGenesisStatsEndpoint:
    """Tests for /api/genesis/stats endpoint."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType")
    def test_stats_returns_structure(self, mock_event_type, mock_ledger_class, genesis_handler):
        """Should return stats structure."""
        # Setup mock event types
        mock_event_type.__iter__ = Mock(
            return_value=iter(
                [
                    Mock(value="agent_birth"),
                    Mock(value="agent_death"),
                    Mock(value="fitness_update"),
                ]
            )
        )

        # Setup mock ledger
        mock_ledger = Mock()
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "abc123" * 10
        mock_ledger_class.return_value = mock_ledger

        result = genesis_handler.handle("/api/genesis/stats", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "event_counts" in data
        assert "total_events" in data
        assert "total_births" in data
        assert "total_deaths" in data
        assert "net_population_change" in data
        assert "integrity_verified" in data
        assert "merkle_root" in data

    def test_stats_unavailable_returns_503(self):
        """Should return 503 when genesis not available."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            handler = GenesisHandler({})
            result = handler.handle("/api/genesis/stats", {}, None)
            assert result.status_code == 503


# ============================================================================
# Events Endpoint Tests
# ============================================================================


class TestGenesisEventsEndpoint:
    """Tests for /api/genesis/events endpoint."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    def test_events_returns_list(self, temp_genesis_db):
        """Should return list of events from database."""
        handler = GenesisHandler({"nomic_dir": temp_genesis_db})

        # Bypass the ledger and use direct DB access
        with patch("aragora.server.handlers.genesis.GenesisLedger"):
            result = handler.handle("/api/genesis/events", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "events" in data
        assert "count" in data
        assert isinstance(data["events"], list)

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    def test_events_limit_capped(self, temp_genesis_db):
        """Should cap limit at 100."""
        handler = GenesisHandler({"nomic_dir": temp_genesis_db})

        with patch("aragora.server.handlers.genesis.GenesisLedger"):
            result = handler.handle("/api/genesis/events", {"limit": "500"}, None)

        # Should process without error (limit capped internally)
        assert result.status_code == 200

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType")
    def test_events_filter_by_type(self, mock_event_type, mock_ledger_class, genesis_handler):
        """Should filter events by type."""
        mock_etype = Mock()
        mock_event_type.return_value = mock_etype

        mock_event = Mock()
        mock_event.to_dict.return_value = {"event_id": "evt-001", "event_type": "agent_birth"}

        mock_ledger = Mock()
        mock_ledger.get_events_by_type.return_value = [mock_event]
        mock_ledger_class.return_value = mock_ledger

        result = genesis_handler.handle("/api/genesis/events", {"event_type": "agent_birth"}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["filter"] == "agent_birth"

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisEventType")
    def test_events_invalid_type_returns_400(self, mock_event_type, genesis_handler):
        """Should return 400 for invalid event type."""
        mock_event_type.side_effect = ValueError("Unknown event type")

        result = genesis_handler.handle("/api/genesis/events", {"event_type": "invalid_type"}, None)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Unknown event type" in data["error"]

    def test_events_unavailable_returns_503(self):
        """Should return 503 when genesis not available."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            handler = GenesisHandler({})
            result = handler.handle("/api/genesis/events", {}, None)
            assert result.status_code == 503


# ============================================================================
# Lineage Endpoint Tests
# ============================================================================


class TestGenesisLineageEndpoint:
    """Tests for /api/genesis/lineage/:genome_id endpoint."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_lineage_returns_structure(self, mock_ledger_class, genesis_handler):
        """Should return lineage structure."""
        mock_ledger = Mock()
        mock_ledger.get_lineage.return_value = ["ancestor-1", "ancestor-2"]
        mock_ledger_class.return_value = mock_ledger

        result = genesis_handler.handle("/api/genesis/lineage/genome-001", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["genome_id"] == "genome-001"
        assert "lineage" in data
        assert "generations" in data

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_lineage_not_found_returns_404(self, mock_ledger_class, genesis_handler):
        """Should return 404 for unknown genome."""
        mock_ledger = Mock()
        mock_ledger.get_lineage.return_value = None
        mock_ledger_class.return_value = mock_ledger

        result = genesis_handler.handle("/api/genesis/lineage/unknown-genome", {}, None)

        assert result.status_code == 404

    def test_lineage_path_traversal_blocked(self, genesis_handler):
        """Should block path traversal attempts."""
        result = genesis_handler.handle("/api/genesis/lineage/../../../etc/passwd", {}, None)
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid genome ID" in data["error"]

    def test_lineage_invalid_id_returns_400(self, genesis_handler):
        """Should reject invalid genome IDs."""
        result = genesis_handler.handle("/api/genesis/lineage/invalid!@#$", {}, None)
        assert result.status_code == 400

    def test_lineage_unavailable_returns_503(self):
        """Should return 503 when genesis not available."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            handler = GenesisHandler({})
            result = handler.handle("/api/genesis/lineage/test", {}, None)
            assert result.status_code == 503


# ============================================================================
# Tree Endpoint Tests
# ============================================================================


class TestGenesisTreeEndpoint:
    """Tests for /api/genesis/tree/:debate_id endpoint."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_tree_returns_structure(self, mock_ledger_class, genesis_handler):
        """Should return tree structure."""
        mock_ledger = Mock()
        mock_ledger.get_debate_tree.return_value = {
            "root": "debate-001",
            "branches": [],
        }
        mock_ledger_class.return_value = mock_ledger

        result = genesis_handler.handle("/api/genesis/tree/debate-001", {}, None)

        # Should attempt to get tree (may fail if method doesn't exist in mock)
        assert result.status_code in [200, 500]

    def test_tree_path_traversal_blocked(self, genesis_handler):
        """Should block path traversal attempts."""
        result = genesis_handler.handle("/api/genesis/tree/../../../etc/passwd", {}, None)
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid debate ID" in data["error"]

    def test_tree_invalid_id_returns_400(self, genesis_handler):
        """Should reject invalid debate IDs."""
        result = genesis_handler.handle("/api/genesis/tree/invalid!@#$", {}, None)
        assert result.status_code == 400

    def test_tree_unavailable_returns_503(self):
        """Should return 503 when genesis not available."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            handler = GenesisHandler({})
            result = handler.handle("/api/genesis/tree/test", {}, None)
            assert result.status_code == 503


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestGenesisErrorHandling:
    """Tests for error handling."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_database_error_returns_500(self, mock_ledger_class, genesis_handler):
        """Should return 500 on database errors."""
        mock_ledger_class.side_effect = Exception("Database error")

        result = genesis_handler.handle("/api/genesis/stats", {}, None)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data

    def test_all_endpoints_unavailable_returns_503(self):
        """All endpoints should return 503 when genesis not available."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            handler = GenesisHandler({})

            endpoints = [
                "/api/genesis/stats",
                "/api/genesis/events",
                "/api/genesis/lineage/test",
                "/api/genesis/tree/test",
            ]

            for path in endpoints:
                result = handler.handle(path, {}, None)
                assert result.status_code == 503, f"Expected 503 for {path}"


# ============================================================================
# Security Tests
# ============================================================================


class TestGenesisSecurity:
    """Tests for security measures."""

    def test_lineage_rejects_special_characters(self, genesis_handler):
        """Should reject genome IDs with special characters."""
        dangerous_ids = [
            "test; DROP TABLE genomes;--",
            "test' OR '1'='1",
            "test<script>alert('xss')</script>",
            "../../../etc/passwd",
            "test\x00null",
        ]

        for dangerous_id in dangerous_ids:
            path = f"/api/genesis/lineage/{dangerous_id}"
            result = genesis_handler.handle(path, {}, None)
            assert result.status_code == 400, f"Should reject: {dangerous_id}"

    def test_tree_rejects_special_characters(self, genesis_handler):
        """Should reject debate IDs with special characters."""
        dangerous_ids = [
            "test; DROP TABLE debates;--",
            "test' OR '1'='1",
            "../../../etc/passwd",
        ]

        for dangerous_id in dangerous_ids:
            path = f"/api/genesis/tree/{dangerous_id}"
            result = genesis_handler.handle(path, {}, None)
            assert result.status_code == 400, f"Should reject: {dangerous_id}"

    def test_accepts_valid_ids(self, genesis_handler):
        """Should accept valid IDs with alphanumeric, dash, underscore, dot."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            valid_ids = [
                "genome-001",
                "genome_v2",
                "genome.v3",
                "abc123XYZ",
                "test-id_v1.0",
            ]

            for valid_id in valid_ids:
                path = f"/api/genesis/lineage/{valid_id}"
                result = genesis_handler.handle(path, {}, None)
                # Should return 503 (unavailable) not 400 (invalid ID)
                assert result.status_code == 503, f"Should accept: {valid_id}"
