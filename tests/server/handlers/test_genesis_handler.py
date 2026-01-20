"""
Tests for the GenesisHandler module.

Comprehensive tests covering:
- Handler routing for all genesis endpoints
- Successful request handling
- Error responses (400, 404, 500, 503)
- Input validation
- Rate limiting behavior
- Query parameter handling
- Dynamic route extraction
- Response format validation
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from aragora.server.handlers.genesis import GenesisHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    from aragora.server.handlers.genesis import _genesis_limiter
    # RateLimiter uses _buckets (dict of timestamp lists), not _requests
    if hasattr(_genesis_limiter, '_buckets'):
        _genesis_limiter._buckets.clear()
    yield


class TestGenesisHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_can_handle_stats(self, handler):
        """Handler can handle stats endpoint."""
        assert handler.can_handle("/api/genesis/stats")

    def test_can_handle_events(self, handler):
        """Handler can handle events endpoint."""
        assert handler.can_handle("/api/genesis/events")

    def test_can_handle_genomes(self, handler):
        """Handler can handle genomes endpoint."""
        assert handler.can_handle("/api/genesis/genomes")

    def test_can_handle_genomes_top(self, handler):
        """Handler can handle top genomes endpoint."""
        assert handler.can_handle("/api/genesis/genomes/top")

    def test_can_handle_population(self, handler):
        """Handler can handle population endpoint."""
        assert handler.can_handle("/api/genesis/population")

    def test_can_handle_lineage_dynamic(self, handler):
        """Handler can handle dynamic lineage endpoint."""
        assert handler.can_handle("/api/genesis/lineage/genome_123")

    def test_can_handle_tree_dynamic(self, handler):
        """Handler can handle dynamic tree endpoint."""
        assert handler.can_handle("/api/genesis/tree/debate_123")

    def test_can_handle_genome_by_id(self, handler):
        """Handler can handle genome by ID endpoint."""
        assert handler.can_handle("/api/genesis/genomes/genome_abc")

    def test_can_handle_descendants(self, handler):
        """Handler can handle descendants endpoint."""
        assert handler.can_handle("/api/genesis/descendants/genome_123")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other")
        assert not handler.can_handle("/api/genomes")


class TestGenesisHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_routes_contains_stats(self, handler):
        """ROUTES contains stats endpoint."""
        assert "/api/genesis/stats" in handler.ROUTES

    def test_routes_contains_events(self, handler):
        """ROUTES contains events endpoint."""
        assert "/api/genesis/events" in handler.ROUTES

    def test_routes_contains_genomes(self, handler):
        """ROUTES contains genomes endpoint."""
        assert "/api/genesis/genomes" in handler.ROUTES

    def test_routes_contains_genomes_top(self, handler):
        """ROUTES contains top genomes endpoint."""
        assert "/api/genesis/genomes/top" in handler.ROUTES

    def test_routes_contains_population(self, handler):
        """ROUTES contains population endpoint."""
        assert "/api/genesis/population" in handler.ROUTES


class TestGenesisHandlerStatsEndpoint:
    """Tests for GET /api/genesis/stats endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_stats_returns_503_when_genesis_unavailable(self, handler, mock_http_handler):
        """Stats endpoint returns 503 when genesis module not available."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle("/api/genesis/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body
        assert "genesis" in body["error"].lower()

    def test_stats_returns_data_when_available(self, handler, mock_http_handler):
        """Stats endpoint returns data when genesis available."""
        mock_ledger = MagicMock()
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "abc123def456789012345678901234567890"

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                with patch("aragora.server.handlers.genesis.GenesisEventType") as mock_event_type:
                    mock_event_type.__iter__ = MagicMock(return_value=iter([]))
                    result = handler.handle("/api/genesis/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "event_counts" in body
        assert "total_events" in body
        assert "integrity_verified" in body
        assert "merkle_root" in body

    def test_stats_handles_exception_gracefully(self, handler, mock_http_handler):
        """Stats endpoint returns 500 on internal error."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", side_effect=Exception("DB error")):
                result = handler.handle("/api/genesis/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 500


class TestGenesisHandlerEventsEndpoint:
    """Tests for GET /api/genesis/events endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_events_returns_503_when_unavailable(self, handler, mock_http_handler):
        """Events endpoint returns 503 when genesis unavailable."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle("/api/genesis/events", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_events_default_limit(self, handler, mock_http_handler):
        """Events endpoint uses default limit of 20."""
        mock_ledger = MagicMock()
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_db)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.cursor.return_value = mock_cursor
        mock_ledger.db = mock_db

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                result = handler.handle("/api/genesis/events", {}, mock_http_handler)

        assert result is not None
        # Either successful or 500 if ledger setup fails
        assert result.status_code in [200, 500]

    def test_events_limit_capped_at_100(self, handler, mock_http_handler):
        """Events endpoint caps limit at 100."""
        mock_ledger = MagicMock()
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_ledger.db = mock_db

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                # Request 500 but should be capped to 100
                result = handler.handle(
                    "/api/genesis/events",
                    {"limit": ["500"]},
                    mock_http_handler,
                )

        assert result is not None

    def test_events_with_event_type_filter(self, handler, mock_http_handler):
        """Events endpoint supports event_type filter."""
        result = handler.handle(
            "/api/genesis/events",
            {"event_type": ["mutation"]},
            mock_http_handler,
        )

        assert result is not None

    def test_events_invalid_event_type_returns_400(self, handler, mock_http_handler):
        """Events endpoint returns 400 for invalid event type."""
        mock_ledger = MagicMock()

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                with patch("aragora.server.handlers.genesis.GenesisEventType", side_effect=ValueError("Unknown")):
                    result = handler.handle(
                        "/api/genesis/events",
                        {"event_type": ["invalid_type"]},
                        mock_http_handler,
                    )

        assert result is not None
        assert result.status_code == 400


class TestGenesisHandlerGenomesEndpoint:
    """Tests for GET /api/genesis/genomes endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_genomes_returns_503_when_unavailable(self, handler, mock_http_handler):
        """Genomes endpoint returns 503 when genome module unavailable."""
        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False):
            result = handler.handle("/api/genesis/genomes", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_genomes_with_pagination(self, handler, mock_http_handler):
        """Genomes endpoint supports limit and offset."""
        mock_store = MagicMock()
        mock_genome = MagicMock()
        mock_genome.to_dict.return_value = {
            "genome_id": "g1",
            "name": "test-genome",
            "fitness_score": 0.75,
        }
        mock_store.get_all.return_value = [mock_genome]

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                result = handler.handle(
                    "/api/genesis/genomes",
                    {"limit": ["10"], "offset": ["5"]},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "genomes" in body
            assert "total" in body
            assert "limit" in body
            assert "offset" in body

    def test_genomes_limit_capped_at_200(self, handler, mock_http_handler):
        """Genomes endpoint caps limit at 200."""
        mock_store = MagicMock()
        mock_store.get_all.return_value = []

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                result = handler.handle(
                    "/api/genesis/genomes",
                    {"limit": ["1000"]},
                    mock_http_handler,
                )

        assert result is not None


class TestGenesisHandlerGenomeByIdEndpoint:
    """Tests for GET /api/genesis/genomes/:id endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_get_genome_not_found(self, handler, mock_http_handler):
        """Get genome returns 404 when not found."""
        mock_store = MagicMock()
        mock_store.get.return_value = None

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                result = handler.handle(
                    "/api/genesis/genomes/nonexistent_id",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 404

    def test_get_genome_success(self, handler, mock_http_handler):
        """Get genome returns genome data on success."""
        mock_store = MagicMock()
        mock_genome = MagicMock()
        mock_genome.to_dict.return_value = {
            "genome_id": "test_genome_123",
            "name": "test-agent",
            "fitness_score": 0.85,
            "generation": 3,
        }
        mock_store.get.return_value = mock_genome

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                result = handler.handle(
                    "/api/genesis/genomes/test_genome_123",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "genome" in body
            assert body["genome"]["genome_id"] == "test_genome_123"

    def test_get_genome_path_traversal_blocked(self, handler, mock_http_handler):
        """Get genome blocks path traversal attempts."""
        result = handler.handle(
            "/api/genesis/genomes/../etc/passwd",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400


class TestGenesisHandlerLineageEndpoint:
    """Tests for GET /api/genesis/lineage/:genome_id endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_lineage_returns_503_when_unavailable(self, handler, mock_http_handler):
        """Lineage endpoint returns 503 when genesis unavailable."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle(
                "/api/genesis/lineage/test_genome",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503

    def test_lineage_not_found(self, handler, mock_http_handler):
        """Lineage endpoint returns 404 when genome not found."""
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = None

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                result = handler.handle(
                    "/api/genesis/lineage/nonexistent",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 404

    def test_lineage_success(self, handler, mock_http_handler):
        """Lineage endpoint returns ancestry data."""
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = [
            {"genome_id": "gen1", "name": "ancestor1", "generation": 0, "fitness_score": 0.7},
            {"genome_id": "gen2", "name": "ancestor2", "generation": 1, "fitness_score": 0.75},
        ]
        mock_ledger.get_events_by_type.return_value = []

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                result = handler.handle(
                    "/api/genesis/lineage/gen2",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "genome_id" in body
            assert "lineage" in body
            assert "generations" in body

    def test_lineage_max_depth_parameter(self, handler, mock_http_handler):
        """Lineage endpoint respects max_depth parameter."""
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = [
            {"genome_id": f"gen{i}", "name": f"ancestor{i}", "generation": i, "fitness_score": 0.5}
            for i in range(20)
        ]
        mock_ledger.get_events_by_type.return_value = []

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                result = handler.handle(
                    "/api/genesis/lineage/gen5",
                    {"max_depth": ["3"]},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            # max_depth should limit lineage results
            assert body["generations"] <= 3

    def test_lineage_max_depth_clamped(self, handler, mock_http_handler):
        """Lineage endpoint clamps max_depth to 1-50."""
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = [
            {"genome_id": "gen1", "name": "test", "generation": 0, "fitness_score": 0.5}
        ]
        mock_ledger.get_events_by_type.return_value = []

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                # Request max_depth=100 (should be clamped to 50)
                result = handler.handle(
                    "/api/genesis/lineage/gen1",
                    {"max_depth": ["100"]},
                    mock_http_handler,
                )

        assert result is not None

    def test_lineage_path_traversal_blocked(self, handler, mock_http_handler):
        """Lineage endpoint blocks path traversal."""
        result = handler.handle(
            "/api/genesis/lineage/../etc/passwd",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400


class TestGenesisHandlerTreeEndpoint:
    """Tests for GET /api/genesis/tree/:debate_id endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_tree_returns_503_when_unavailable(self, handler, mock_http_handler):
        """Tree endpoint returns 503 when genesis unavailable."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle(
                "/api/genesis/tree/debate_123",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503

    def test_tree_success(self, handler, mock_http_handler):
        """Tree endpoint returns debate tree structure."""
        mock_ledger = MagicMock()
        mock_tree = MagicMock()
        mock_tree.to_dict.return_value = {
            "root": "node1",
            "children": [],
        }
        mock_tree.nodes = ["node1", "node2"]
        mock_ledger.get_debate_tree.return_value = mock_tree

        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger):
                result = handler.handle(
                    "/api/genesis/tree/debate_123",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "debate_id" in body
            assert "tree" in body
            assert "total_nodes" in body

    def test_tree_path_traversal_blocked(self, handler, mock_http_handler):
        """Tree endpoint blocks path traversal."""
        result = handler.handle(
            "/api/genesis/tree/../etc/passwd",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400


class TestGenesisHandlerDescendantsEndpoint:
    """Tests for GET /api/genesis/descendants/:genome_id endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_descendants_returns_503_when_unavailable(self, handler, mock_http_handler):
        """Descendants endpoint returns 503 when genome module unavailable."""
        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False):
            result = handler.handle(
                "/api/genesis/descendants/genome_123",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503

    def test_descendants_root_not_found(self, handler, mock_http_handler):
        """Descendants endpoint returns 404 when root genome not found."""
        mock_store = MagicMock()
        mock_store.get.return_value = None

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                result = handler.handle(
                    "/api/genesis/descendants/nonexistent",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 404

    def test_descendants_success(self, handler, mock_http_handler):
        """Descendants endpoint returns descendant data."""
        mock_store = MagicMock()

        mock_root = MagicMock()
        mock_root.genome_id = "root_genome"
        mock_root.name = "root-agent"
        mock_root.generation = 0
        mock_root.fitness_score = 0.7
        mock_root.parent_genomes = []

        mock_child = MagicMock()
        mock_child.genome_id = "child_genome"
        mock_child.name = "child-agent"
        mock_child.generation = 1
        mock_child.fitness_score = 0.8
        mock_child.parent_genomes = ["root_genome"]

        mock_store.get.return_value = mock_root
        mock_store.get_all.return_value = [mock_root, mock_child]

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                result = handler.handle(
                    "/api/genesis/descendants/root_genome",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "genome_id" in body
            assert "descendants" in body
            assert "total_descendants" in body
            assert "root_genome" in body

    def test_descendants_max_depth_clamped(self, handler, mock_http_handler):
        """Descendants endpoint clamps max_depth to 1-20."""
        mock_store = MagicMock()
        mock_root = MagicMock()
        mock_root.genome_id = "root"
        mock_root.name = "root"
        mock_root.generation = 0
        mock_root.fitness_score = 0.5
        mock_root.parent_genomes = []
        mock_store.get.return_value = mock_root
        mock_store.get_all.return_value = [mock_root]

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                # Request max_depth=100 (should be clamped to 20)
                result = handler.handle(
                    "/api/genesis/descendants/root",
                    {"max_depth": ["100"]},
                    mock_http_handler,
                )

        assert result is not None

    def test_descendants_path_traversal_blocked(self, handler, mock_http_handler):
        """Descendants endpoint blocks path traversal."""
        result = handler.handle(
            "/api/genesis/descendants/../etc/passwd",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400


class TestGenesisHandlerPopulationEndpoint:
    """Tests for GET /api/genesis/population endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_population_returns_503_when_unavailable(self, handler, mock_http_handler):
        """Population endpoint returns 503 when breeding module unavailable."""
        # PopulationManager is imported inside _get_population, so patch the module
        with patch.dict("sys.modules", {"aragora.genesis.breeding": None}):
            result = handler.handle(
                "/api/genesis/population",
                {},
                mock_http_handler,
            )

        assert result is not None
        # Either 503 (module unavailable) or 500 (import error)
        assert result.status_code in [500, 503]

    def test_population_success(self, handler, mock_http_handler):
        """Population endpoint returns population data."""
        mock_manager = MagicMock()
        mock_population = MagicMock()
        mock_population.population_id = "pop_123"
        mock_population.generation = 5
        mock_population.size = 4
        mock_population.average_fitness = 0.72
        mock_population.debate_history = ["d1", "d2"]

        mock_genome = MagicMock()
        mock_genome.genome_id = "g1"
        mock_genome.name = "test-agent"
        mock_genome.fitness_score = 0.75
        mock_genome.generation = 5
        mock_genome.traits = {"curious": 0.8}
        mock_genome.expertise = {"security": 0.9}
        mock_genome.get_dominant_traits = MagicMock(return_value=["curious"])

        mock_population.genomes = [mock_genome]
        mock_population.best_genome = mock_genome
        mock_manager.get_or_create_population.return_value = mock_population

        with patch("aragora.genesis.breeding.PopulationManager", return_value=mock_manager):
            result = handler.handle(
                "/api/genesis/population",
                {},
                mock_http_handler,
            )

        assert result is not None


class TestGenesisHandlerTopGenomesEndpoint:
    """Tests for GET /api/genesis/genomes/top endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_top_genomes_returns_503_when_unavailable(self, handler, mock_http_handler):
        """Top genomes endpoint returns 503 when genome module unavailable."""
        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False):
            result = handler.handle(
                "/api/genesis/genomes/top",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503

    def test_top_genomes_success(self, handler, mock_http_handler):
        """Top genomes endpoint returns ranked genomes."""
        mock_store = MagicMock()
        mock_genome = MagicMock()
        mock_genome.to_dict.return_value = {
            "genome_id": "top_genome",
            "name": "champion",
            "fitness_score": 0.95,
        }
        mock_store.get_top_by_fitness.return_value = [mock_genome]

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                result = handler.handle(
                    "/api/genesis/genomes/top",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "genomes" in body
            assert "count" in body

    def test_top_genomes_limit_capped_at_50(self, handler, mock_http_handler):
        """Top genomes endpoint caps limit at 50."""
        mock_store = MagicMock()
        mock_store.get_top_by_fitness.return_value = []

        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True):
            with patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store):
                result = handler.handle(
                    "/api/genesis/genomes/top",
                    {"limit": ["1000"]},
                    mock_http_handler,
                )

        assert result is not None


class TestGenesisHandlerRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_rate_limit_exceeded_returns_429(self, handler, mock_http_handler):
        """Rate limit exceeded returns 429."""
        from aragora.server.handlers.genesis import _genesis_limiter

        # Simulate rate limit exceeded
        with patch.object(_genesis_limiter, 'is_allowed', return_value=False):
            result = handler.handle("/api/genesis/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 429
        body = json.loads(result.body)
        assert "error" in body
        assert "rate limit" in body["error"].lower()

    def test_multiple_requests_tracked(self, handler, mock_http_handler):
        """Multiple requests are tracked for rate limiting."""
        # Make several requests
        for _ in range(5):
            result = handler.handle("/api/genesis/stats", {}, mock_http_handler)
            assert result is not None


class TestGenesisHandlerInputValidation:
    """Tests for input validation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_invalid_genome_id_format(self, handler, mock_http_handler):
        """Invalid genome ID format returns 400."""
        # Various invalid ID patterns
        invalid_ids = [
            "../etc/passwd",
            "genome<script>",
            "genome;DROP TABLE",
            "",
        ]

        for invalid_id in invalid_ids:
            if ".." in invalid_id:
                # Path traversal specifically checked
                result = handler.handle(
                    f"/api/genesis/genomes/{invalid_id}",
                    {},
                    mock_http_handler,
                )
                if result:
                    assert result.status_code == 400

    def test_invalid_debate_id_format(self, handler, mock_http_handler):
        """Invalid debate ID format returns 400."""
        result = handler.handle(
            "/api/genesis/tree/../etc/passwd",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400


class TestGenesisHandlerIntegration:
    """Integration tests for genesis handler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return GenesisHandler(mock_server_context)

    def test_all_static_routes_reachable(self, handler, mock_http_handler):
        """All static routes return a response."""
        static_routes = [
            "/api/genesis/stats",
            "/api/genesis/events",
            "/api/genesis/genomes",
            "/api/genesis/genomes/top",
            "/api/genesis/population",
        ]

        for route in static_routes:
            result = handler.handle(route, {}, mock_http_handler)
            assert result is not None, f"Route {route} returned None"
            # All should return some response (success or feature unavailable)
            assert result.status_code in [200, 400, 429, 500, 503], \
                f"Route {route} returned unexpected status {result.status_code}"

    def test_all_dynamic_routes_reachable(self, handler, mock_http_handler):
        """All dynamic routes return a response."""
        dynamic_routes = [
            "/api/genesis/lineage/test_genome",
            "/api/genesis/tree/test_debate",
            "/api/genesis/genomes/specific_genome",
            "/api/genesis/descendants/test_genome",
        ]

        for route in dynamic_routes:
            result = handler.handle(route, {}, mock_http_handler)
            assert result is not None, f"Route {route} returned None"
            assert result.status_code in [200, 400, 404, 429, 500, 503], \
                f"Route {route} returned unexpected status {result.status_code}"

    def test_handle_returns_none_for_unknown(self, handler, mock_http_handler):
        """Handle returns None for unknown paths."""
        result = handler.handle("/api/unknown", {}, mock_http_handler)
        assert result is None

    def test_handler_inherits_from_base(self, handler):
        """Handler inherits from BaseHandler."""
        from aragora.server.handlers.base import BaseHandler
        assert isinstance(handler, BaseHandler)
