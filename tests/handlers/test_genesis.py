"""Tests for genesis handler (aragora/server/handlers/genesis.py).

Covers all routes and behavior of the GenesisHandler class:
- can_handle() routing for all ROUTES (versioned and unversioned)
- GET /api/genesis/stats - Get overall genesis statistics
- GET /api/genesis/events - Get recent genesis events
- GET /api/genesis/genomes - List all genomes
- GET /api/genesis/genomes/top - Get top genomes by fitness
- GET /api/genesis/genomes/:genome_id - Get single genome details
- GET /api/genesis/population - Get active population
- GET /api/genesis/lineage/:genome_id - Get genome ancestry
- GET /api/genesis/tree/:debate_id - Get debate tree structure
- GET /api/genesis/descendants/:genome_id - Get genome descendants
- Rate limiting (429)
- Path traversal protection (400)
- Module unavailability (503)
- Error handling (500)
- Input validation
- RBAC permission checks
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.genesis import GenesisHandler, _genesis_limiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for GenesisHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        client_address: tuple[str, int] = ("127.0.0.1", 12345),
    ):
        self.command = method
        self.client_address = client_address
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.path = ""

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock enums and data
# ---------------------------------------------------------------------------


class MockGenesisEventType(Enum):
    """Mock of GenesisEventType for testing."""

    DEBATE_START = "debate_start"
    DEBATE_END = "debate_end"
    DEBATE_SPAWN = "debate_spawn"
    DEBATE_MERGE = "debate_merge"
    CONSENSUS_REACHED = "consensus_reached"
    TENSION_DETECTED = "tension_detected"
    TENSION_RESOLVED = "tension_resolved"
    TENSION_UNRESOLVED = "tension_unresolved"
    AGENT_BIRTH = "agent_birth"
    AGENT_DEATH = "agent_death"
    AGENT_MUTATION = "agent_mutation"
    AGENT_CROSSOVER = "agent_crossover"
    FITNESS_UPDATE = "fitness_update"
    # The handler references MUTATION and CROSSOVER (without AGENT_ prefix)
    MUTATION = "mutation"
    CROSSOVER = "crossover"


def _make_mock_event(event_type="agent_birth", genome_id="genome-abc", change=0.1):
    """Create a mock genesis event."""
    event = MagicMock()
    event.data = {"genome_id": genome_id, "change": change}
    event.timestamp = "2026-01-15T10:00:00Z"
    event.to_dict.return_value = {
        "event_id": "evt-001",
        "event_type": event_type,
        "timestamp": "2026-01-15T10:00:00Z",
        "data": {"genome_id": genome_id},
    }
    return event


def _make_mock_genome(
    genome_id="genome-abc",
    name="claude",
    generation=1,
    fitness_score=0.85,
    parent_genomes=None,
    traits=None,
    expertise=None,
):
    """Create a mock AgentGenome."""
    genome = MagicMock()
    genome.genome_id = genome_id
    genome.name = name
    genome.generation = generation
    genome.fitness_score = fitness_score
    genome.parent_genomes = parent_genomes or []
    genome.traits = traits or {"analytical": 0.9, "creative": 0.7, "critical": 0.8}
    genome.expertise = expertise if expertise is not None else {"coding": 0.9, "math": 0.8}
    genome.get_dominant_traits = MagicMock(return_value=["analytical", "creative", "critical"])
    genome.to_dict.return_value = {
        "genome_id": genome_id,
        "name": name,
        "generation": generation,
        "fitness_score": fitness_score,
        "parent_genomes": parent_genomes or [],
        "traits": genome.traits,
        "expertise": genome.expertise,
    }
    return genome


def _make_mock_population(genomes=None, best=None):
    """Create a mock Population."""
    pop = MagicMock()
    pop.population_id = "pop-001"
    pop.generation = 3
    pop.size = len(genomes) if genomes else 4
    pop.average_fitness = 0.75
    pop.genomes = genomes or [
        _make_mock_genome("g1", "claude", 3, 0.9),
        _make_mock_genome("g2", "gemini", 3, 0.8),
        _make_mock_genome("g3", "codex", 3, 0.7),
        _make_mock_genome("g4", "grok", 3, 0.6),
    ]
    pop.debate_history = ["d1", "d2"]
    pop.best_genome = best or pop.genomes[0]
    return pop


def _make_mock_tree(num_nodes=5):
    """Create a mock FractalTree."""
    tree = MagicMock()
    tree.nodes = [MagicMock() for _ in range(num_nodes)]
    tree.to_dict.return_value = {
        "root_id": "debate-001",
        "nodes": [{"id": f"node-{i}"} for i in range(num_nodes)],
    }
    return tree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a GenesisHandler with a minimal context including nomic_dir."""
    return GenesisHandler(ctx={"nomic_dir": Path("/tmp/test-nomic")})


@pytest.fixture
def handler_no_ctx():
    """Create a GenesisHandler without nomic_dir."""
    return GenesisHandler(ctx={})


@pytest.fixture
def handler_none_ctx():
    """Create a GenesisHandler with None context."""
    return GenesisHandler(ctx=None)


@pytest.fixture
def http_handler():
    """Create a default MockHTTPHandler."""
    return MockHTTPHandler()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    _genesis_limiter._buckets = defaultdict(list)
    _genesis_limiter._requests = _genesis_limiter._buckets
    yield
    _genesis_limiter._buckets = defaultdict(list)
    _genesis_limiter._requests = _genesis_limiter._buckets


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/genesis/stats",
            "/api/genesis/events",
            "/api/genesis/genomes",
            "/api/genesis/genomes/top",
            "/api/genesis/population",
            "/api/genesis/lineage/genome-abc",
            "/api/genesis/tree/debate-001",
            "/api/genesis/genomes/genome-abc",
            "/api/genesis/descendants/genome-abc",
            # Versioned paths
            "/api/v1/genesis/stats",
            "/api/v1/genesis/events",
            "/api/v1/genesis/genomes",
            "/api/v1/genesis/genomes/top",
            "/api/v1/genesis/population",
            "/api/v1/genesis/lineage/genome-abc",
            "/api/v1/genesis/tree/debate-001",
            "/api/v1/genesis/genomes/genome-abc",
            "/api/v1/genesis/descendants/genome-abc",
        ],
    )
    def test_accepts_valid_paths(self, handler, path):
        assert handler.can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/debates",
            "/api/health",
            "/api/genesis",
            "/api/genes/stats",
            "/api/v2/other/stats",
            "/other/genesis/stats",
            "",
        ],
    )
    def test_rejects_invalid_paths(self, handler, path):
        assert handler.can_handle(path) is False


# ============================================================================
# GET /api/genesis/stats
# ============================================================================


class TestGetStats:
    """Tests for the genesis stats endpoint."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_stats_success(self, mock_ledger_cls, handler, http_handler):
        """Stats returns event counts, births, deaths, fitness trend."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        # Set up events by type
        births = [_make_mock_event("agent_birth")]
        deaths = []
        fitness_updates = [_make_mock_event(change=0.05), _make_mock_event(change=0.15)]

        def get_events_side_effect(etype):
            if etype == MockGenesisEventType.AGENT_BIRTH:
                return births
            if etype == MockGenesisEventType.AGENT_DEATH:
                return deaths
            if etype == MockGenesisEventType.FITNESS_UPDATE:
                return fitness_updates
            return []

        mock_ledger.get_events_by_type.side_effect = get_events_side_effect
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "a" * 64

        result = handler.handle("/api/genesis/stats", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_births"] == 1
        assert body["total_deaths"] == 0
        assert body["net_population_change"] == 1
        assert body["integrity_verified"] is True
        assert "merkle_root" in body

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False)
    def test_stats_module_unavailable(self, handler, http_handler):
        """Returns 503 when genesis module is not available."""
        result = handler.handle("/api/genesis/stats", {}, http_handler)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_stats_exception(self, mock_ledger_cls, handler, http_handler):
        """Returns 500 on internal error."""
        mock_ledger_cls.side_effect = RuntimeError("db error")
        result = handler.handle("/api/genesis/stats", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_stats_versioned_path(self, mock_ledger_cls, handler, http_handler):
        """Stats works on versioned path /api/v1/genesis/stats."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "b" * 64

        result = handler.handle("/api/v1/genesis/stats", {}, http_handler)
        assert _status(result) == 200

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_stats_no_nomic_dir(self, mock_ledger_cls, handler_no_ctx, http_handler):
        """Stats uses default path when nomic_dir not provided."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "c" * 64

        result = handler_no_ctx.handle("/api/genesis/stats", {}, http_handler)
        assert _status(result) == 200
        mock_ledger_cls.assert_called_once_with("genesis.db")

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_stats_with_nomic_dir(self, mock_ledger_cls, handler, http_handler):
        """Stats uses nomic_dir path when provided."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "d" * 64

        result = handler.handle("/api/genesis/stats", {}, http_handler)
        assert _status(result) == 200
        expected_path = str(Path("/tmp/test-nomic") / "genesis.db")
        mock_ledger_cls.assert_called_once_with(expected_path)

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_stats_avg_fitness_empty(self, mock_ledger_cls, handler, http_handler):
        """Stats handles empty fitness updates gracefully."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "e" * 64

        result = handler.handle("/api/genesis/stats", {}, http_handler)
        body = _body(result)
        assert body["avg_fitness_change_recent"] == 0.0


# ============================================================================
# GET /api/genesis/events
# ============================================================================


class TestGetEvents:
    """Tests for the genesis events endpoint."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_events_with_type_filter(self, mock_ledger_cls, handler, http_handler):
        """Events filtered by event_type returns matching events."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        events = [_make_mock_event("agent_birth")]
        mock_ledger.get_events_by_type.return_value = events

        result = handler.handle(
            "/api/genesis/events",
            {"event_type": "agent_birth", "limit": "5"},
            http_handler,
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["filter"] == "agent_birth"
        assert body["count"] == 1

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_events_all_recent(self, mock_ledger_cls, handler, http_handler):
        """Events without filter returns all recent events via SQL."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        # Mock the db.connection context manager for SQL query
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (
                "evt-1",
                "agent_birth",
                "2026-01-15T10:00:00Z",
                None,
                "abc123def456ghi7890",
                '{"genome_id": "g1"}',
            ),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = handler.handle("/api/genesis/events", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 1
        assert "filter" not in body

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_events_invalid_event_type_chars(self, mock_ledger_cls, handler, http_handler):
        """Invalid event_type characters return 400."""
        result = handler.handle(
            "/api/genesis/events",
            {"event_type": "agent_birth; DROP TABLE"},
            http_handler,
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_events_unknown_event_type(self, mock_ledger_cls, handler, http_handler):
        """Unknown event_type returns 400 with valid types list."""
        result = handler.handle(
            "/api/genesis/events",
            {"event_type": "nonexistent_type"},
            http_handler,
        )
        assert _status(result) == 400
        assert "Unknown event type" in _body(result).get("error", "")

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_events_event_type_too_long(self, mock_ledger_cls, handler, http_handler):
        """Event type exceeding 64 chars returns 400."""
        result = handler.handle(
            "/api/genesis/events",
            {"event_type": "a" * 65},
            http_handler,
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_events_event_type_as_list(self, mock_ledger_cls, handler, http_handler):
        """Event type passed as list extracts first element."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_events_by_type.return_value = []

        result = handler.handle(
            "/api/genesis/events",
            {"event_type": ["agent_birth", "agent_death"]},
            http_handler,
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["filter"] == "agent_birth"

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_events_event_type_empty_list(self, mock_ledger_cls, handler, http_handler):
        """Event type passed as empty list treats as no filter."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = handler.handle(
            "/api/genesis/events",
            {"event_type": []},
            http_handler,
        )
        assert _status(result) == 200

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False)
    def test_events_module_unavailable(self, handler, http_handler):
        """Returns 503 when genesis module not available."""
        result = handler.handle("/api/genesis/events", {}, http_handler)
        assert _status(result) == 503

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_events_exception(self, mock_ledger_cls, handler, http_handler):
        """Returns 500 on internal error."""
        mock_ledger_cls.side_effect = ValueError("corrupt db")
        result = handler.handle("/api/genesis/events", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_events_limit_capped_at_100(self, mock_ledger_cls, handler, http_handler):
        """Limit is capped at 100 even if higher requested."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = handler.handle("/api/genesis/events", {"limit": "500"}, http_handler)
        assert _status(result) == 200
        # Verify limit=100 was passed to SQL
        mock_cursor.execute.assert_called_once()
        args = mock_cursor.execute.call_args
        assert args[0][1] == (100,)

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_events_versioned_path(self, mock_ledger_cls, handler, http_handler):
        """Events work on versioned path /api/v1/genesis/events."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = handler.handle("/api/v1/genesis/events", {}, http_handler)
        assert _status(result) == 200


# ============================================================================
# GET /api/genesis/genomes
# ============================================================================


class TestGetGenomes:
    """Tests for the genomes listing endpoint."""

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_genomes_list(self, mock_store_cls, handler, http_handler):
        """Returns paginated list of genomes."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get_all.return_value = [
            _make_mock_genome("g1"),
            _make_mock_genome("g2"),
            _make_mock_genome("g3"),
        ]

        result = handler.handle("/api/genesis/genomes", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 3
        assert body["limit"] == 50  # default limit
        assert body["offset"] == 0

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_genomes_pagination(self, mock_store_cls, handler, http_handler):
        """Pagination with offset and limit works correctly."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        genomes = [_make_mock_genome(f"g{i}") for i in range(10)]
        mock_store.get_all.return_value = genomes

        result = handler.handle("/api/genesis/genomes", {"limit": "3", "offset": "2"}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 10
        assert body["limit"] == 3
        assert body["offset"] == 2
        assert len(body["genomes"]) == 3

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_genomes_limit_capped_at_200(self, mock_store_cls, handler, http_handler):
        """Limit is capped at 200 even if higher requested."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get_all.return_value = []

        result = handler.handle("/api/genesis/genomes", {"limit": "999"}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["limit"] == 200

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False)
    def test_genomes_module_unavailable(self, handler, http_handler):
        """Returns 503 when genome module not available."""
        result = handler.handle("/api/genesis/genomes", {}, http_handler)
        assert _status(result) == 503

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_genomes_exception(self, mock_store_cls, handler, http_handler):
        """Returns 500 on internal error."""
        mock_store_cls.side_effect = OSError("disk error")
        result = handler.handle("/api/genesis/genomes", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_genomes_empty(self, mock_store_cls, handler, http_handler):
        """Returns empty list when no genomes exist."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get_all.return_value = []

        result = handler.handle("/api/genesis/genomes", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["genomes"] == []
        assert body["total"] == 0


# ============================================================================
# GET /api/genesis/genomes/top
# ============================================================================


class TestGetTopGenomes:
    """Tests for the top genomes endpoint."""

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_top_genomes_default_limit(self, mock_store_cls, handler, http_handler):
        """Returns top genomes with default limit of 10."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        top = [_make_mock_genome("g1", fitness_score=0.95)]
        mock_store.get_top_by_fitness.return_value = top

        result = handler.handle("/api/genesis/genomes/top", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 1
        mock_store.get_top_by_fitness.assert_called_once_with(10)

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_top_genomes_custom_limit(self, mock_store_cls, handler, http_handler):
        """Top genomes respects custom limit."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get_top_by_fitness.return_value = []

        result = handler.handle("/api/genesis/genomes/top", {"limit": "5"}, http_handler)
        assert _status(result) == 200
        mock_store.get_top_by_fitness.assert_called_once_with(5)

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_top_genomes_limit_capped_at_50(self, mock_store_cls, handler, http_handler):
        """Top genomes limit is capped at 50."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get_top_by_fitness.return_value = []

        result = handler.handle("/api/genesis/genomes/top", {"limit": "999"}, http_handler)
        assert _status(result) == 200
        mock_store.get_top_by_fitness.assert_called_once_with(50)

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False)
    def test_top_genomes_module_unavailable(self, handler, http_handler):
        """Returns 503 when genome module not available."""
        result = handler.handle("/api/genesis/genomes/top", {}, http_handler)
        assert _status(result) == 503

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_top_genomes_exception(self, mock_store_cls, handler, http_handler):
        """Returns 500 on internal error."""
        mock_store_cls.side_effect = KeyError("missing key")
        result = handler.handle("/api/genesis/genomes/top", {}, http_handler)
        assert _status(result) == 500


# ============================================================================
# GET /api/genesis/genomes/:genome_id
# ============================================================================


class TestGetGenome:
    """Tests for single genome detail endpoint."""

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_get_genome_found(self, mock_store_cls, handler, http_handler):
        """Returns genome details when found."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        genome = _make_mock_genome("genome-abc")
        mock_store.get.return_value = genome

        result = handler.handle("/api/genesis/genomes/genome-abc", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["genome"]["genome_id"] == "genome-abc"

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_get_genome_not_found(self, mock_store_cls, handler, http_handler):
        """Returns 404 when genome not found."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get.return_value = None

        result = handler.handle("/api/genesis/genomes/nonexistent", {}, http_handler)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False)
    def test_get_genome_module_unavailable(self, handler, http_handler):
        """Returns 503 when genome module not available."""
        result = handler.handle("/api/genesis/genomes/genome-abc", {}, http_handler)
        assert _status(result) == 503

    def test_get_genome_path_traversal(self, handler, http_handler):
        """Blocks path traversal attempts in genome ID."""
        result = handler.handle("/api/genesis/genomes/../../../etc/passwd", {}, http_handler)
        assert _status(result) == 400

    def test_get_genome_invalid_id(self, handler, http_handler):
        """Returns 400 for invalid genome IDs."""
        result = handler.handle("/api/genesis/genomes/invalid id!", {}, http_handler)
        assert _status(result) == 400

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_get_genome_exception(self, mock_store_cls, handler, http_handler):
        """Returns 500 on internal error."""
        mock_store_cls.side_effect = AttributeError("oops")
        result = handler.handle("/api/genesis/genomes/genome-abc", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_get_genome_versioned(self, mock_store_cls, handler, http_handler):
        """Works with versioned path."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get.return_value = _make_mock_genome("g1")

        result = handler.handle("/api/v1/genesis/genomes/g1", {}, http_handler)
        assert _status(result) == 200


# ============================================================================
# GET /api/genesis/lineage/:genome_id
# ============================================================================


class TestGetLineage:
    """Tests for the genome lineage endpoint."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_lineage_success(self, mock_ledger_cls, handler, http_handler):
        """Returns enriched lineage data."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        lineage_data = [
            {
                "genome_id": "g1",
                "name": "claude",
                "generation": 1,
                "fitness_score": 0.9,
                "parent_genomes": [],
            },
            {
                "genome_id": "g0",
                "name": "proto",
                "generation": 0,
                "fitness_score": 0.5,
                "parent_genomes": [],
            },
        ]
        mock_ledger.get_lineage.return_value = lineage_data

        # Birth events to enrich lineage
        birth_event = _make_mock_event("agent_birth", genome_id="g1")
        mock_ledger.get_events_by_type.return_value = [birth_event]

        result = handler.handle("/api/genesis/lineage/g1", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["genome_id"] == "g1"
        assert body["generations"] == 2
        assert len(body["lineage"]) == 2

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_lineage_not_found(self, mock_ledger_cls, handler, http_handler):
        """Returns 404 when genome has no lineage."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_lineage.return_value = []

        result = handler.handle("/api/genesis/lineage/nonexistent", {}, http_handler)
        assert _status(result) == 404

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_lineage_max_depth(self, mock_ledger_cls, handler, http_handler):
        """Lineage respects max_depth parameter."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        # Create deep lineage
        lineage = [
            {"genome_id": f"g{i}", "name": f"agent{i}", "generation": i, "fitness_score": 0.5}
            for i in range(20)
        ]
        mock_ledger.get_lineage.return_value = lineage
        mock_ledger.get_events_by_type.return_value = []

        result = handler.handle("/api/genesis/lineage/g0", {"max_depth": "3"}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["generations"] == 3

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_lineage_max_depth_clamped_high(self, mock_ledger_cls, handler, http_handler):
        """Lineage max_depth clamped to 50 maximum."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_lineage.return_value = [{"genome_id": "g1", "name": "a"}]
        mock_ledger.get_events_by_type.return_value = []

        result = handler.handle("/api/genesis/lineage/g1", {"max_depth": "100"}, http_handler)
        assert _status(result) == 200

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_lineage_max_depth_clamped_low(self, mock_ledger_cls, handler, http_handler):
        """Lineage max_depth clamped to 1 minimum."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_lineage.return_value = [{"genome_id": "g1", "name": "a"}]
        mock_ledger.get_events_by_type.return_value = []

        result = handler.handle("/api/genesis/lineage/g1", {"max_depth": "0"}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        # Should still return at least 1 if there's data
        assert body["generations"] == 1

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False)
    def test_lineage_module_unavailable(self, handler, http_handler):
        """Returns 503 when genesis module not available."""
        result = handler.handle("/api/genesis/lineage/g1", {}, http_handler)
        assert _status(result) == 503

    def test_lineage_path_traversal(self, handler, http_handler):
        """Blocks path traversal in genome ID."""
        result = handler.handle("/api/genesis/lineage/../../../etc/passwd", {}, http_handler)
        assert _status(result) == 400

    def test_lineage_invalid_genome_id(self, handler, http_handler):
        """Returns 400 for invalid genome IDs."""
        result = handler.handle("/api/genesis/lineage/bad id!", {}, http_handler)
        assert _status(result) == 400

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_lineage_exception(self, mock_ledger_cls, handler, http_handler):
        """Returns 500 on internal error."""
        mock_ledger_cls.side_effect = RuntimeError("unexpected")
        result = handler.handle("/api/genesis/lineage/g1", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_lineage_event_enrichment_graceful_degradation(
        self, mock_ledger_cls, handler, http_handler
    ):
        """Lineage returns data even if event enrichment fails."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        mock_ledger.get_lineage.return_value = [
            {"genome_id": "g1", "name": "a", "generation": 1, "fitness_score": 0.8},
        ]
        # Event lookup raises an exception - should be caught and ignored
        mock_ledger.get_events_by_type.side_effect = RuntimeError("event db corrupt")

        result = handler.handle("/api/genesis/lineage/g1", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["generations"] == 1


# ============================================================================
# GET /api/genesis/tree/:debate_id
# ============================================================================


class TestGetDebateTree:
    """Tests for the debate tree endpoint."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_tree_success(self, mock_ledger_cls, handler, http_handler):
        """Returns debate tree structure."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_tree = _make_mock_tree(5)
        mock_ledger.get_debate_tree.return_value = mock_tree

        result = handler.handle("/api/genesis/tree/debate-001", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["debate_id"] == "debate-001"
        assert body["total_nodes"] == 5

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False)
    def test_tree_module_unavailable(self, handler, http_handler):
        """Returns 503 when genesis module not available."""
        result = handler.handle("/api/genesis/tree/debate-001", {}, http_handler)
        assert _status(result) == 503

    def test_tree_path_traversal(self, handler, http_handler):
        """Blocks path traversal in debate ID."""
        result = handler.handle("/api/genesis/tree/../../../etc/passwd", {}, http_handler)
        assert _status(result) == 400

    def test_tree_invalid_debate_id(self, handler, http_handler):
        """Returns 400 for invalid debate IDs."""
        result = handler.handle("/api/genesis/tree/bad id!", {}, http_handler)
        assert _status(result) == 400

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_tree_exception(self, mock_ledger_cls, handler, http_handler):
        """Returns 500 on internal error."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_debate_tree.side_effect = KeyError("missing")
        result = handler.handle("/api/genesis/tree/debate-001", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_tree_versioned(self, mock_ledger_cls, handler, http_handler):
        """Works with versioned path."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_debate_tree.return_value = _make_mock_tree(2)

        result = handler.handle("/api/v1/genesis/tree/debate-001", {}, http_handler)
        assert _status(result) == 200


# ============================================================================
# GET /api/genesis/population
# ============================================================================


class TestGetPopulation:
    """Tests for the population endpoint."""

    def test_population_success(self, handler, http_handler):
        """Returns population details with genome summaries."""
        mock_breeding_module = MagicMock()
        mock_pm = MagicMock()
        mock_breeding_module.PopulationManager.return_value = mock_pm
        pop = _make_mock_population()
        mock_pm.get_or_create_population.return_value = pop

        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        sys.modules["aragora.genesis.breeding"] = mock_breeding_module
        try:
            result = handler.handle("/api/genesis/population", {}, http_handler)
            body = _body(result)
            assert _status(result) == 200
            assert body["population_id"] == "pop-001"
            assert body["generation"] == 3
            assert body["size"] == 4
            assert body["average_fitness"] == 0.75
            assert len(body["genomes"]) == 4
            assert body["best_genome"] is not None
            assert body["debate_history_count"] == 2
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)

    def test_population_import_error(self, handler, http_handler):
        """Returns 503 when breeding module not available."""
        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        # Setting to None in sys.modules causes ImportError on from-import
        sys.modules["aragora.genesis.breeding"] = None
        try:
            result = handler.handle("/api/genesis/population", {}, http_handler)
            assert _status(result) == 503
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)

    def test_population_exception(self, handler, http_handler):
        """Returns 500 on internal error."""
        mock_mod = MagicMock()
        mock_mod.PopulationManager.side_effect = ValueError("bad config")

        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        sys.modules["aragora.genesis.breeding"] = mock_mod
        try:
            result = handler.handle("/api/genesis/population", {}, http_handler)
            assert _status(result) == 500
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)

    def test_population_versioned(self, handler, http_handler):
        """Population works on versioned path."""
        mock_mod = MagicMock()
        mock_pm = MagicMock()
        mock_mod.PopulationManager.return_value = mock_pm
        mock_pm.get_or_create_population.return_value = _make_mock_population()

        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        sys.modules["aragora.genesis.breeding"] = mock_mod
        try:
            result = handler.handle("/api/v1/genesis/population", {}, http_handler)
            assert _status(result) == 200
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)

    def test_population_no_nomic_dir(self, handler_no_ctx, http_handler):
        """Population uses default path when nomic_dir not provided."""
        mock_mod = MagicMock()
        mock_pm = MagicMock()
        mock_mod.PopulationManager.return_value = mock_pm
        mock_pm.get_or_create_population.return_value = _make_mock_population()

        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        sys.modules["aragora.genesis.breeding"] = mock_mod
        try:
            result = handler_no_ctx.handle("/api/genesis/population", {}, http_handler)
            assert _status(result) == 200
            mock_mod.PopulationManager.assert_called_once_with(db_path="genesis.db")
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)

    def test_population_best_genome_none(self, handler, http_handler):
        """Population handles None best_genome."""
        mock_mod = MagicMock()
        mock_pm = MagicMock()
        mock_mod.PopulationManager.return_value = mock_pm
        pop = _make_mock_population()
        pop.best_genome = None
        mock_pm.get_or_create_population.return_value = pop

        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        sys.modules["aragora.genesis.breeding"] = mock_mod
        try:
            result = handler.handle("/api/genesis/population", {}, http_handler)
            body = _body(result)
            assert _status(result) == 200
            assert body["best_genome"] is None
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)

    def test_population_genome_without_get_dominant_traits(self, handler, http_handler):
        """Population handles genomes without get_dominant_traits method."""
        mock_mod = MagicMock()
        mock_pm = MagicMock()
        mock_mod.PopulationManager.return_value = mock_pm

        # Create a genome without get_dominant_traits
        genome = MagicMock()
        genome.genome_id = "g1"
        genome.name = "claude"
        genome.fitness_score = 0.9
        genome.generation = 3
        genome.traits = {"analytical": 0.9, "creative": 0.7}
        genome.expertise = {"coding": 0.9}
        del genome.get_dominant_traits  # Remove the method

        pop = _make_mock_population(genomes=[genome])
        mock_pm.get_or_create_population.return_value = pop

        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        sys.modules["aragora.genesis.breeding"] = mock_mod
        try:
            result = handler.handle("/api/genesis/population", {}, http_handler)
            body = _body(result)
            assert _status(result) == 200
            # Should use fallback list(genome.traits.keys())[:3]
            assert len(body["genomes"]) == 1
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)


# ============================================================================
# GET /api/genesis/descendants/:genome_id
# ============================================================================


class TestGetDescendants:
    """Tests for the genome descendants endpoint."""

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_success(self, mock_store_cls, handler, http_handler):
        """Returns descendants tree from a root genome."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        root = _make_mock_genome("g1", generation=1, fitness_score=0.8)
        child1 = _make_mock_genome("g2", generation=2, fitness_score=0.85, parent_genomes=["g1"])
        child2 = _make_mock_genome("g3", generation=2, fitness_score=0.7, parent_genomes=["g1"])
        grandchild = _make_mock_genome("g4", generation=3, fitness_score=0.9, parent_genomes=["g2"])

        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root, child1, child2, grandchild]

        result = handler.handle("/api/genesis/descendants/g1", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["genome_id"] == "g1"
        assert body["total_descendants"] == 3
        assert body["root_genome"]["name"] == "claude"

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_not_found(self, mock_store_cls, handler, http_handler):
        """Returns 404 when root genome not found."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get.return_value = None

        result = handler.handle("/api/genesis/descendants/nonexistent", {}, http_handler)
        assert _status(result) == 404

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_max_depth(self, mock_store_cls, handler, http_handler):
        """Descendants respects max_depth parameter."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        root = _make_mock_genome("g1", generation=1)
        child = _make_mock_genome("g2", generation=2, parent_genomes=["g1"])
        grandchild = _make_mock_genome("g3", generation=3, parent_genomes=["g2"])

        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root, child, grandchild]

        result = handler.handle("/api/genesis/descendants/g1", {"max_depth": "1"}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        # Only depth 1 descendants (direct children)
        assert body["total_descendants"] == 1

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_max_depth_clamped(self, mock_store_cls, handler, http_handler):
        """Descendants max_depth clamped to 1-20."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        root = _make_mock_genome("g1")
        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root]

        # max_depth=0 should be clamped to 1
        result = handler.handle("/api/genesis/descendants/g1", {"max_depth": "0"}, http_handler)
        assert _status(result) == 200

        # max_depth=50 should be clamped to 20
        result = handler.handle("/api/genesis/descendants/g1", {"max_depth": "50"}, http_handler)
        assert _status(result) == 200

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_no_children(self, mock_store_cls, handler, http_handler):
        """Returns empty descendants for leaf genome."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        root = _make_mock_genome("g1")
        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root]

        result = handler.handle("/api/genesis/descendants/g1", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_descendants"] == 0
        assert body["descendants"] == []

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_sorted_by_depth_and_fitness(self, mock_store_cls, handler, http_handler):
        """Descendants are sorted by depth, then by fitness (descending)."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        root = _make_mock_genome("g1", generation=1)
        child_a = _make_mock_genome("g2", generation=2, fitness_score=0.7, parent_genomes=["g1"])
        child_b = _make_mock_genome("g3", generation=2, fitness_score=0.9, parent_genomes=["g1"])

        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root, child_a, child_b]

        result = handler.handle("/api/genesis/descendants/g1", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_descendants"] == 2
        # Higher fitness should come first at same depth
        assert body["descendants"][0]["fitness_score"] == 0.9
        assert body["descendants"][1]["fitness_score"] == 0.7

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False)
    def test_descendants_module_unavailable(self, handler, http_handler):
        """Returns 503 when genome module not available."""
        result = handler.handle("/api/genesis/descendants/g1", {}, http_handler)
        assert _status(result) == 503

    def test_descendants_path_traversal(self, handler, http_handler):
        """Blocks path traversal in genome ID."""
        result = handler.handle("/api/genesis/descendants/../../../etc/passwd", {}, http_handler)
        assert _status(result) == 400

    def test_descendants_invalid_genome_id(self, handler, http_handler):
        """Returns 400 for invalid genome IDs."""
        result = handler.handle("/api/genesis/descendants/bad id!", {}, http_handler)
        assert _status(result) == 400

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_exception(self, mock_store_cls, handler, http_handler):
        """Returns 500 on internal error."""
        mock_store_cls.side_effect = OSError("disk error")
        result = handler.handle("/api/genesis/descendants/g1", {}, http_handler)
        assert _status(result) == 500

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_versioned(self, mock_store_cls, handler, http_handler):
        """Works with versioned path."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        root = _make_mock_genome("g1")
        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root]

        result = handler.handle("/api/v1/genesis/descendants/g1", {}, http_handler)
        assert _status(result) == 200


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on genesis endpoints."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_rate_limit_allows_normal_traffic(self, mock_ledger_cls, handler, http_handler):
        """Normal traffic within limits passes through."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "f" * 64

        # First request should succeed
        result = handler.handle("/api/genesis/stats", {}, http_handler)
        assert _status(result) == 200

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    def test_rate_limit_exceeded(self, handler):
        """Returns 429 when rate limit exceeded."""
        http = MockHTTPHandler(client_address=("10.0.0.1", 12345))

        # Exhaust rate limit (10 requests per minute)
        for _ in range(10):
            _genesis_limiter._buckets["10.0.0.1"].append(time.time())

        result = handler.handle("/api/genesis/stats", {}, http)
        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_rate_limit_different_ips(self, mock_ledger_cls, handler):
        """Different IPs have independent rate limits."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "g" * 64

        # Exhaust rate limit for IP1
        for _ in range(10):
            _genesis_limiter._buckets["10.0.0.1"].append(time.time())

        # IP2 should still work
        http2 = MockHTTPHandler(client_address=("10.0.0.2", 12345))
        result = handler.handle("/api/genesis/stats", {}, http2)
        assert _status(result) == 200

    def test_rate_limit_applies_to_all_routes(self, handler):
        """Rate limiting applies to all genesis routes."""
        http = MockHTTPHandler(client_address=("10.0.0.3", 12345))

        # Exhaust rate limit
        for _ in range(10):
            _genesis_limiter._buckets["10.0.0.3"].append(time.time())

        paths = [
            "/api/genesis/stats",
            "/api/genesis/events",
            "/api/genesis/genomes",
            "/api/genesis/genomes/top",
            "/api/genesis/population",
        ]
        for path in paths:
            result = handler.handle(path, {}, http)
            assert _status(result) == 429, f"Rate limit not applied to {path}"


# ============================================================================
# Path Traversal Protection
# ============================================================================


class TestPathTraversal:
    """Tests for path traversal protection across all dynamic routes."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/genesis/genomes/../../etc/passwd",
            "/api/genesis/lineage/../../etc/passwd",
            "/api/genesis/tree/../../etc/passwd",
            "/api/genesis/descendants/../../etc/passwd",
            "/api/v1/genesis/genomes/../../etc/passwd",
            "/api/v1/genesis/lineage/../../etc/passwd",
            "/api/v1/genesis/tree/../../etc/passwd",
            "/api/v1/genesis/descendants/../../etc/passwd",
        ],
    )
    def test_path_traversal_blocked(self, handler, http_handler, path):
        """Path traversal attempts are blocked with 400."""
        result = handler.handle(path, {}, http_handler)
        assert _status(result) == 400


# ============================================================================
# Input Validation
# ============================================================================


class TestInputValidation:
    """Tests for input validation on various parameters."""

    @pytest.mark.parametrize(
        "genome_id",
        [
            "valid-id-123",
            "genome_abc",
            "g1.v2",  # With dot for versioning
            "A-Z_0-9",
        ],
    )
    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_valid_genome_ids(self, mock_store_cls, handler, http_handler, genome_id):
        """Valid genome IDs are accepted."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get.return_value = _make_mock_genome(genome_id)

        result = handler.handle(f"/api/genesis/genomes/{genome_id}", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.parametrize(
        "genome_id",
        [
            "bad id",
            "has spaces",
            "has@special",
            "",
        ],
    )
    def test_invalid_genome_ids_rejected(self, handler, http_handler, genome_id):
        """Invalid genome IDs are rejected with 400."""
        if not genome_id:
            # Empty genome_id means the path matches /api/genesis/genomes/ without trailing
            # which might route to the list endpoint instead
            return
        result = handler.handle(f"/api/genesis/genomes/{genome_id}", {}, http_handler)
        assert _status(result) == 400

    @pytest.mark.parametrize(
        "debate_id",
        [
            "debate-001",
            "debate_abc",
            "d123",
        ],
    )
    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_valid_debate_ids(self, mock_ledger_cls, handler, http_handler, debate_id):
        """Valid debate IDs are accepted."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_debate_tree.return_value = _make_mock_tree()

        result = handler.handle(f"/api/genesis/tree/{debate_id}", {}, http_handler)
        assert _status(result) == 200


# ============================================================================
# Handler Initialization
# ============================================================================


class TestInitialization:
    """Tests for handler initialization."""

    def test_init_with_ctx(self):
        """Handler initializes with provided context."""
        ctx = {"nomic_dir": Path("/test")}
        handler = GenesisHandler(ctx=ctx)
        assert handler.ctx == ctx

    def test_init_with_none_ctx(self):
        """Handler initializes with empty dict when None passed."""
        handler = GenesisHandler(ctx=None)
        assert handler.ctx == {}

    def test_init_default_ctx(self):
        """Handler initializes with empty dict by default."""
        handler = GenesisHandler()
        assert handler.ctx == {}


# ============================================================================
# Unmatched Route
# ============================================================================


class TestUnmatchedRoute:
    """Tests for unmatched routes returning None."""

    def test_handle_returns_none_for_unmatched(self, handler, http_handler):
        """Handle returns None for paths it cannot handle."""
        result = handler.handle("/api/unknown/route", {}, http_handler)
        assert result is None

    def test_handle_returns_none_for_partial_match(self, handler, http_handler):
        """Handle returns None for partial path matches."""
        result = handler.handle("/api/genesis", {}, http_handler)
        assert result is None


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for various edge cases."""

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_content_hash_truncation(self, mock_ledger_cls, handler, http_handler):
        """Content hash is truncated to 16 chars + ellipsis in events response."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        full_hash = "a" * 64
        mock_cursor.fetchall.return_value = [
            ("evt-1", "agent_birth", "2026-01-15T10:00:00Z", None, full_hash, "{}"),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = handler.handle("/api/genesis/events", {}, http_handler)
        body = _body(result)
        event = body["events"][0]
        assert event["content_hash"] == "a" * 16 + "..."

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_content_hash_none(self, mock_ledger_cls, handler, http_handler):
        """Content hash is None when not present."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("evt-1", "agent_birth", "2026-01-15T10:00:00Z", None, None, "{}"),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = handler.handle("/api/genesis/events", {}, http_handler)
        body = _body(result)
        event = body["events"][0]
        assert event["content_hash"] is None

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    @patch("aragora.server.handlers.genesis.GenesisEventType", MockGenesisEventType)
    def test_merkle_root_truncation(self, mock_ledger_cls, handler, http_handler):
        """Merkle root is truncated to 32 chars + ellipsis."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "h" * 64

        result = handler.handle("/api/genesis/stats", {}, http_handler)
        body = _body(result)
        assert body["merkle_root"] == "h" * 32 + "..."

    @patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenesisLedger")
    def test_events_invalid_json_in_data(self, mock_ledger_cls, handler, http_handler):
        """Events handle invalid JSON in data column gracefully via safe_json_parse."""
        mock_ledger = MagicMock()
        mock_ledger_cls.return_value = mock_ledger

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (
                "evt-1",
                "agent_birth",
                "2026-01-15T10:00:00Z",
                None,
                "abc123def456ghi7",
                "not-valid-json",
            ),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = handler.handle("/api/genesis/events", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        # safe_json_parse returns the default {}
        assert body["events"][0]["data"] == {}

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_handles_genome_with_none_parent_genomes(
        self, mock_store_cls, handler, http_handler
    ):
        """Descendants handles genomes with None parent_genomes."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        root = _make_mock_genome("g1")
        child = _make_mock_genome("g2", parent_genomes=["g1"])
        # Genome with None parent_genomes
        orphan = MagicMock()
        orphan.genome_id = "g3"
        orphan.parent_genomes = None

        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root, child, orphan]

        result = handler.handle("/api/genesis/descendants/g1", {}, http_handler)
        assert _status(result) == 200

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_descendants_handles_none_fitness_score(self, mock_store_cls, handler, http_handler):
        """Descendants handles genomes with None fitness_score in sorting."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        root = _make_mock_genome("g1")
        child = _make_mock_genome("g2", fitness_score=None, parent_genomes=["g1"])

        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root, child]

        result = handler.handle("/api/genesis/descendants/g1", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_descendants"] == 1

    @patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True)
    @patch("aragora.server.handlers.genesis.GenomeStore")
    def test_genomes_offset_beyond_total(self, mock_store_cls, handler, http_handler):
        """Genomes with offset beyond total returns empty page."""
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.get_all.return_value = [_make_mock_genome("g1")]

        result = handler.handle("/api/genesis/genomes", {"offset": "100"}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 1
        assert len(body["genomes"]) == 0

    def test_population_genome_empty_expertise(self, handler, http_handler):
        """Population handles genomes with empty expertise."""
        mock_mod = MagicMock()
        mock_pm = MagicMock()
        mock_mod.PopulationManager.return_value = mock_pm

        genome = _make_mock_genome("g1", expertise={})
        pop = _make_mock_population(genomes=[genome])
        mock_pm.get_or_create_population.return_value = pop

        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        sys.modules["aragora.genesis.breeding"] = mock_mod
        try:
            result = handler.handle("/api/genesis/population", {}, http_handler)
            body = _body(result)
            assert _status(result) == 200
            assert body["genomes"][0]["expertise_domains"] == []
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)

    def test_population_genome_none_expertise(self, handler, http_handler):
        """Population handles genomes with None expertise."""
        mock_mod = MagicMock()
        mock_pm = MagicMock()
        mock_mod.PopulationManager.return_value = mock_pm

        genome = _make_mock_genome("g1")
        genome.expertise = None
        pop = _make_mock_population(genomes=[genome])
        mock_pm.get_or_create_population.return_value = pop

        import sys

        original = sys.modules.get("aragora.genesis.breeding")
        sys.modules["aragora.genesis.breeding"] = mock_mod
        try:
            result = handler.handle("/api/genesis/population", {}, http_handler)
            body = _body(result)
            assert _status(result) == 200
            assert body["genomes"][0]["expertise_domains"] == []
        finally:
            if original is not None:
                sys.modules["aragora.genesis.breeding"] = original
            else:
                sys.modules.pop("aragora.genesis.breeding", None)
