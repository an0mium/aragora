"""
Comprehensive tests for the GenesisHandler module.

Covers:
- All GET endpoints (stats, events, genomes, genomes/top, genome by ID,
  lineage, tree, population, descendants)
- Input validation (invalid event types, bad genome/debate IDs, pagination bounds)
- RBAC permission checks
- Rate limiting
- Error handling (module not available, DB errors)
- Edge cases (empty results, max limits, path traversal)
- Version prefix stripping (v1 routes)
- can_handle routing
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.genesis import GenesisHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """Minimal server context."""
    return {"nomic_dir": None}


@pytest.fixture
def ctx_with_dir(tmp_path):
    """Server context with a nomic_dir set."""
    return {"nomic_dir": tmp_path}


@pytest.fixture
def handler(ctx):
    return GenesisHandler(ctx)


@pytest.fixture
def handler_with_dir(ctx_with_dir):
    return GenesisHandler(ctx_with_dir)


@pytest.fixture
def http_handler():
    """Mock HTTP request handler with client address."""
    h = MagicMock()
    h.client_address = ("10.0.0.1", 54321)
    h.headers = {"Content-Length": "0"}
    h.command = "GET"
    return h


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear rate-limiter state before each test."""
    from aragora.server.handlers.genesis import _genesis_limiter

    if hasattr(_genesis_limiter, "_buckets"):
        _genesis_limiter._buckets.clear()
    yield


# Helpers -------------------------------------------------------------------


def _body(result):
    """Parse JSON body from a HandlerResult."""
    return json.loads(result.body)


def _mock_event(genome_id=None, data=None, timestamp="2026-01-15T00:00:00Z"):
    e = MagicMock()
    e.data = data or {}
    if genome_id:
        e.data["genome_id"] = genome_id
    e.timestamp = timestamp
    e.to_dict = MagicMock(return_value={"event_id": "ev1", **e.data})
    return e


def _mock_genome(
    genome_id="g1",
    name="agent1",
    fitness=0.8,
    generation=1,
    parent_genomes=None,
    traits=None,
    expertise=None,
):
    g = MagicMock()
    g.genome_id = genome_id
    g.name = name
    g.fitness_score = fitness
    g.generation = generation
    g.parent_genomes = parent_genomes or []
    g.traits = traits or {"analytical": 0.9}
    g.expertise = expertise or {"security": 0.85}
    g.get_dominant_traits = MagicMock(return_value=list(g.traits.keys())[:3])
    g.to_dict = MagicMock(
        return_value={
            "genome_id": genome_id,
            "name": name,
            "fitness_score": fitness,
            "generation": generation,
        }
    )
    return g


# ===========================================================================
# can_handle routing
# ===========================================================================


class TestCanHandle:
    """Verify can_handle returns True/False for expected paths."""

    @pytest.fixture
    def h(self, ctx):
        return GenesisHandler(ctx)

    # Static routes (both versioned and unversioned)
    @pytest.mark.parametrize(
        "path",
        [
            "/api/genesis/stats",
            "/api/genesis/events",
            "/api/genesis/genomes",
            "/api/genesis/genomes/top",
            "/api/genesis/population",
            "/api/v1/genesis/stats",
            "/api/v1/genesis/events",
            "/api/v1/genesis/genomes",
            "/api/v1/genesis/genomes/top",
            "/api/v1/genesis/population",
        ],
    )
    def test_static_routes(self, h, path):
        assert h.can_handle(path)

    # Dynamic routes
    @pytest.mark.parametrize(
        "path",
        [
            "/api/genesis/lineage/genome_abc",
            "/api/genesis/tree/debate_xyz",
            "/api/genesis/genomes/specific_id",
            "/api/genesis/descendants/genome_abc",
            "/api/v1/genesis/lineage/genome_abc",
            "/api/v1/genesis/tree/debate_xyz",
            "/api/v1/genesis/genomes/specific_id",
            "/api/v1/genesis/descendants/genome_abc",
        ],
    )
    def test_dynamic_routes(self, h, path):
        assert h.can_handle(path)

    # Unhandled routes
    @pytest.mark.parametrize(
        "path",
        [
            "/api/debates",
            "/api/v1/debates",
            "/api/v1/other",
            "/health",
            "/api/genesis",
        ],
    )
    def test_unhandled_routes(self, h, path):
        assert not h.can_handle(path)

    def test_genomes_top_is_not_dynamic_genome(self, h):
        """'/api/genesis/genomes/top' must NOT match the dynamic genome-by-id route."""
        # can_handle should return True (via static), but the routing inside
        # handle() should go to _get_top_genomes, not _get_genome.
        assert h.can_handle("/api/genesis/genomes/top")


# ===========================================================================
# RBAC permission checks
# ===========================================================================


class TestRBACPermissions:
    """Ensure genesis:read permission is enforced."""

    @pytest.mark.no_auto_auth
    def test_handle_raises_without_auth_context(self, handler, http_handler):
        """Without auth context the RBAC decorator should raise PermissionDeniedError."""
        from aragora.rbac.decorators import PermissionDeniedError
        from aragora.server.auth import auth_config

        orig = auth_config.enabled
        auth_config.enabled = True
        try:
            with pytest.raises(PermissionDeniedError):
                handler.handle("/api/genesis/stats", {}, http_handler)
        finally:
            auth_config.enabled = orig

    @pytest.mark.no_auto_auth
    def test_handle_raises_for_events_without_auth(self, handler, http_handler):
        from aragora.rbac.decorators import PermissionDeniedError
        from aragora.server.auth import auth_config

        orig = auth_config.enabled
        auth_config.enabled = True
        try:
            with pytest.raises(PermissionDeniedError):
                handler.handle("/api/genesis/events", {}, http_handler)
        finally:
            auth_config.enabled = orig

    @pytest.mark.no_auto_auth
    def test_handle_raises_for_genomes_without_auth(self, handler, http_handler):
        from aragora.rbac.decorators import PermissionDeniedError
        from aragora.server.auth import auth_config

        orig = auth_config.enabled
        auth_config.enabled = True
        try:
            with pytest.raises(PermissionDeniedError):
                handler.handle("/api/genesis/genomes", {}, http_handler)
        finally:
            auth_config.enabled = orig


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    def test_rate_limit_exceeded_returns_429(self, handler, http_handler):
        from aragora.server.handlers.genesis import _genesis_limiter

        with patch.object(_genesis_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/genesis/stats", {}, http_handler)

        assert result is not None
        assert result.status_code == 429
        body = _body(result)
        assert "rate limit" in body["error"].lower()

    def test_rate_limit_applies_to_all_endpoints(self, handler, http_handler):
        from aragora.server.handlers.genesis import _genesis_limiter

        paths = [
            "/api/genesis/stats",
            "/api/genesis/events",
            "/api/genesis/genomes",
            "/api/genesis/genomes/top",
            "/api/genesis/population",
            "/api/genesis/lineage/test_genome",
            "/api/genesis/tree/test_debate",
            "/api/genesis/descendants/test_genome",
        ]
        with patch.object(_genesis_limiter, "is_allowed", return_value=False):
            for path in paths:
                result = handler.handle(path, {}, http_handler)
                assert result is not None
                assert result.status_code == 429, f"{path} did not return 429"

    def test_rate_limit_allowed_continues(self, handler, http_handler):
        """When rate limit allows, handler continues to dispatch."""
        from aragora.server.handlers.genesis import _genesis_limiter

        with patch.object(_genesis_limiter, "is_allowed", return_value=True):
            # With genesis unavailable, should get 503 (not 429)
            with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
                result = handler.handle("/api/genesis/stats", {}, http_handler)
        assert result.status_code == 503


# ===========================================================================
# GET /api/genesis/stats
# ===========================================================================


class TestGetGenesisStats:
    def test_503_when_genesis_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle("/api/genesis/stats", {}, http_handler)
        assert result.status_code == 503
        assert "not available" in _body(result)["error"].lower()

    def test_success(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "a" * 64

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
            patch("aragora.server.handlers.genesis.GenesisEventType") as mock_et,
        ):
            mock_et.__iter__ = MagicMock(return_value=iter([]))
            result = handler.handle("/api/genesis/stats", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["event_counts"] == {}
        assert body["total_events"] == 0
        assert body["total_births"] == 0
        assert body["total_deaths"] == 0
        assert body["net_population_change"] == 0
        assert body["integrity_verified"] is True
        assert body["merkle_root"].endswith("...")

    def test_stats_with_events(self, handler, http_handler):
        """Stats correctly aggregates when there are birth/death/fitness events."""
        birth_event = _mock_event()
        death_event = _mock_event()
        fitness_event = _mock_event(data={"change": 0.05})

        mock_ledger = MagicMock()

        # Build enum members
        birth_type = MagicMock()
        birth_type.value = "agent_birth"
        death_type = MagicMock()
        death_type.value = "agent_death"
        fitness_type = MagicMock()
        fitness_type.value = "fitness_update"

        def side_effect(etype):
            if etype is birth_type:
                return [birth_event, birth_event]
            if etype is death_type:
                return [death_event]
            if etype is fitness_type:
                return [fitness_event]
            return []

        mock_ledger.get_events_by_type.side_effect = side_effect
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "b" * 64

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
            patch("aragora.server.handlers.genesis.GenesisEventType") as mock_et,
        ):
            mock_et.__iter__ = MagicMock(return_value=iter([birth_type, death_type, fitness_type]))
            mock_et.AGENT_BIRTH = birth_type
            mock_et.AGENT_DEATH = death_type
            mock_et.FITNESS_UPDATE = fitness_type
            result = handler.handle("/api/genesis/stats", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["total_births"] == 2
        assert body["total_deaths"] == 1
        assert body["net_population_change"] == 1

    def test_stats_db_error_returns_500(self, handler, http_handler):
        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch(
                "aragora.server.handlers.genesis.GenesisLedger",
                side_effect=OSError("DB connection failed"),
            ),
        ):
            result = handler.handle("/api/genesis/stats", {}, http_handler)
        assert result.status_code == 500

    def test_stats_with_nomic_dir(self, handler_with_dir, http_handler):
        """When nomic_dir is set, ledger uses that path."""
        mock_ledger = MagicMock()
        mock_ledger.get_events_by_type.return_value = []
        mock_ledger.verify_integrity.return_value = True
        mock_ledger.get_merkle_root.return_value = "c" * 64

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch(
                "aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger
            ) as ctor,
            patch("aragora.server.handlers.genesis.GenesisEventType") as mock_et,
        ):
            mock_et.__iter__ = MagicMock(return_value=iter([]))
            result = handler_with_dir.handle("/api/genesis/stats", {}, http_handler)

        assert result.status_code == 200
        # Verify the ledger was constructed with the expected path
        call_args = ctor.call_args[0][0]
        assert "genesis.db" in call_args

    def test_stats_via_v1_route(self, handler, http_handler):
        """v1 prefixed route works identically."""
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle("/api/v1/genesis/stats", {}, http_handler)
        assert result.status_code == 503


# ===========================================================================
# GET /api/genesis/events
# ===========================================================================


class TestGetGenesisEvents:
    def test_503_when_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle("/api/genesis/events", {}, http_handler)
        assert result.status_code == 503

    def test_default_limit_is_20(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/events", {}, http_handler)

        assert result.status_code == 200
        # The SQL should have been called with limit=20
        execute_call = mock_cursor.execute.call_args
        assert execute_call[0][1] == (20,)

    def test_limit_capped_at_100(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/events", {"limit": ["999"]}, http_handler)

        assert result.status_code == 200
        execute_call = mock_cursor.execute.call_args
        assert execute_call[0][1] == (100,)

    def test_events_returned_in_response(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("ev1", "mutation", "2026-01-15T00:00:00Z", None, "hash1234567890ab", '{"key":"val"}'),
            ("ev2", "agent_birth", "2026-01-15T01:00:00Z", "ev1", None, "{}"),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/events", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["count"] == 2
        assert len(body["events"]) == 2
        assert body["events"][0]["event_id"] == "ev1"
        assert body["events"][0]["content_hash"].endswith("...")
        assert body["events"][1]["content_hash"] is None

    def test_filter_by_valid_event_type(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_event = _mock_event()
        mock_ledger.get_events_by_type.return_value = [mock_event]

        valid_type = MagicMock()
        valid_type.value = "mutation"

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
            patch("aragora.server.handlers.genesis.GenesisEventType") as mock_et,
        ):
            mock_et.__iter__ = MagicMock(return_value=iter([valid_type]))
            mock_et.return_value = valid_type
            result = handler.handle(
                "/api/genesis/events", {"event_type": ["mutation"]}, http_handler
            )

        assert result.status_code == 200
        body = _body(result)
        assert body["filter"] == "mutation"
        assert body["count"] == 1

    def test_filter_event_type_as_list(self, handler, http_handler):
        """When event_type is a list, first element is used."""
        mock_ledger = MagicMock()
        mock_ledger.get_events_by_type.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisEventType") as mock_et,
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            valid_type = MagicMock()
            valid_type.value = "mutation"
            mock_et.__iter__ = MagicMock(return_value=iter([valid_type]))
            # The handler extracts first element from list if it's a list
            result = handler.handle(
                "/api/genesis/events",
                {"event_type": ["mutation", "agent_birth"]},
                http_handler,
            )
        assert result is not None

    def test_invalid_event_type_special_chars_returns_400(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            result = handler.handle(
                "/api/genesis/events",
                {"event_type": ["drop;--"]},
                http_handler,
            )
        assert result.status_code == 400
        assert "invalid event type" in _body(result)["error"].lower()

    def test_invalid_event_type_too_long_returns_400(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True):
            result = handler.handle(
                "/api/genesis/events",
                {"event_type": ["a" * 65]},
                http_handler,
            )
        assert result.status_code == 400

    def test_unknown_event_type_returns_400(self, handler, http_handler):
        valid_type = MagicMock()
        valid_type.value = "mutation"

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisEventType") as mock_et,
        ):
            mock_et.__iter__ = MagicMock(return_value=iter([valid_type]))
            result = handler.handle(
                "/api/genesis/events",
                {"event_type": ["nonexistent_type"]},
                http_handler,
            )
        assert result.status_code == 400
        assert "unknown event type" in _body(result)["error"].lower()

    def test_empty_event_type_list_treated_as_none(self, handler, http_handler):
        """Empty event_type list means no filter (event_type becomes None)."""
        mock_ledger = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_ledger.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ledger.db.connection.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/events", {"event_type": []}, http_handler)
        assert result.status_code == 200

    def test_db_error_returns_500(self, handler, http_handler):
        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch(
                "aragora.server.handlers.genesis.GenesisLedger", side_effect=OSError("DB down")
            ),
        ):
            result = handler.handle("/api/genesis/events", {}, http_handler)
        assert result.status_code == 500


# ===========================================================================
# GET /api/genesis/genomes
# ===========================================================================


class TestGetGenomes:
    def test_503_when_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False):
            result = handler.handle("/api/genesis/genomes", {}, http_handler)
        assert result.status_code == 503

    def test_default_pagination(self, handler, http_handler):
        mock_store = MagicMock()
        mock_store.get_all.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/genomes", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["limit"] == 50
        assert body["offset"] == 0
        assert body["total"] == 0
        assert body["genomes"] == []

    def test_pagination_with_params(self, handler, http_handler):
        genomes = [_mock_genome(genome_id=f"g{i}") for i in range(10)]
        mock_store = MagicMock()
        mock_store.get_all.return_value = genomes

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle(
                "/api/genesis/genomes",
                {"limit": ["3"], "offset": ["2"]},
                http_handler,
            )

        assert result.status_code == 200
        body = _body(result)
        assert body["total"] == 10
        assert body["limit"] == 3
        assert body["offset"] == 2
        assert len(body["genomes"]) == 3

    def test_limit_capped_at_200(self, handler, http_handler):
        mock_store = MagicMock()
        mock_store.get_all.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle(
                "/api/genesis/genomes",
                {"limit": ["5000"]},
                http_handler,
            )

        body = _body(result)
        assert body["limit"] == 200

    def test_offset_beyond_total(self, handler, http_handler):
        mock_store = MagicMock()
        mock_store.get_all.return_value = [_mock_genome()]

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle(
                "/api/genesis/genomes",
                {"offset": ["100"]},
                http_handler,
            )

        body = _body(result)
        assert body["total"] == 1
        assert body["genomes"] == []

    def test_db_error_returns_500(self, handler, http_handler):
        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch(
                "aragora.server.handlers.genesis.GenomeStore", side_effect=OSError("disk full")
            ),
        ):
            result = handler.handle("/api/genesis/genomes", {}, http_handler)
        assert result.status_code == 500


# ===========================================================================
# GET /api/genesis/genomes/top
# ===========================================================================


class TestGetTopGenomes:
    def test_503_when_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False):
            result = handler.handle("/api/genesis/genomes/top", {}, http_handler)
        assert result.status_code == 503

    def test_default_limit_10(self, handler, http_handler):
        mock_store = MagicMock()
        mock_store.get_top_by_fitness.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/genomes/top", {}, http_handler)

        assert result.status_code == 200
        mock_store.get_top_by_fitness.assert_called_once_with(10)

    def test_limit_capped_at_50(self, handler, http_handler):
        mock_store = MagicMock()
        mock_store.get_top_by_fitness.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/genomes/top", {"limit": ["1000"]}, http_handler)

        mock_store.get_top_by_fitness.assert_called_once_with(50)

    def test_success_with_results(self, handler, http_handler):
        g1 = _mock_genome(genome_id="top1", fitness=0.95)
        g2 = _mock_genome(genome_id="top2", fitness=0.90)
        mock_store = MagicMock()
        mock_store.get_top_by_fitness.return_value = [g1, g2]

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/genomes/top", {}, http_handler)

        body = _body(result)
        assert body["count"] == 2
        assert len(body["genomes"]) == 2

    def test_db_error_returns_500(self, handler, http_handler):
        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", side_effect=OSError("timeout")),
        ):
            result = handler.handle("/api/genesis/genomes/top", {}, http_handler)
        assert result.status_code == 500


# ===========================================================================
# GET /api/genesis/genomes/:genome_id
# ===========================================================================


class TestGetGenomeById:
    def test_503_when_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False):
            result = handler.handle("/api/genesis/genomes/genome_abc", {}, http_handler)
        assert result.status_code == 503

    def test_genome_found(self, handler, http_handler):
        g = _mock_genome(genome_id="genome_abc")
        mock_store = MagicMock()
        mock_store.get.return_value = g

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/genomes/genome_abc", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["genome"]["genome_id"] == "genome_abc"

    def test_genome_not_found_returns_404(self, handler, http_handler):
        mock_store = MagicMock()
        mock_store.get.return_value = None

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/genomes/nonexistent", {}, http_handler)

        assert result.status_code == 404

    def test_path_traversal_blocked(self, handler, http_handler):
        result = handler.handle("/api/genesis/genomes/../etc/passwd", {}, http_handler)
        assert result.status_code == 400

    def test_invalid_genome_id_returns_400(self, handler, http_handler):
        """IDs with special chars are rejected by validate_genome_id."""
        result = handler.handle("/api/genesis/genomes/<script>", {}, http_handler)
        assert result is not None
        assert result.status_code == 400

    def test_db_error_returns_500(self, handler, http_handler):
        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch(
                "aragora.server.handlers.genesis.GenomeStore", side_effect=OSError("corrupted")
            ),
        ):
            result = handler.handle("/api/genesis/genomes/valid_id", {}, http_handler)
        assert result.status_code == 500


# ===========================================================================
# GET /api/genesis/lineage/:genome_id
# ===========================================================================


class TestGetGenomeLineage:
    def test_503_when_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle("/api/genesis/lineage/genome_abc", {}, http_handler)
        assert result.status_code == 503

    def test_genome_not_found_returns_404(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = None

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/lineage/unknown", {}, http_handler)

        assert result.status_code == 404

    def test_empty_lineage_returns_404(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = []  # empty list is falsy

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/lineage/empty_genome", {}, http_handler)

        assert result.status_code == 404

    def test_success_with_lineage(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = [
            {
                "genome_id": "g1",
                "name": "ancestor",
                "generation": 0,
                "fitness_score": 0.7,
                "parent_genomes": [],
            },
            {
                "genome_id": "g2",
                "name": "child",
                "generation": 1,
                "fitness_score": 0.8,
                "parent_genomes": ["g1"],
            },
        ]
        mock_ledger.get_events_by_type.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/lineage/g2", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["genome_id"] == "g2"
        assert body["generations"] == 2
        assert len(body["lineage"]) == 2
        assert body["lineage"][0]["genome_id"] == "g1"

    def test_enrichment_with_birth_event(self, handler, http_handler):
        """Lineage nodes get enriched with event data when birth events match."""
        birth_event = _mock_event(genome_id="g1", timestamp="2026-01-10T00:00:00Z")

        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = [
            {"genome_id": "g1", "name": "original", "generation": 0, "fitness_score": 0.5},
        ]

        birth_type = MagicMock()
        mutation_type = MagicMock()
        crossover_type = MagicMock()

        def events_by_type(etype):
            if etype is birth_type:
                return [birth_event]
            return []

        mock_ledger.get_events_by_type.side_effect = events_by_type

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
            patch("aragora.server.handlers.genesis.GenesisEventType") as mock_et,
        ):
            mock_et.AGENT_BIRTH = birth_type
            mock_et.MUTATION = mutation_type
            mock_et.CROSSOVER = crossover_type
            result = handler.handle("/api/genesis/lineage/g1", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["lineage"][0]["event_type"] == "agent_birth"
        assert body["lineage"][0]["created_at"] == "2026-01-10T00:00:00Z"

    def test_max_depth_default_10(self, handler, http_handler):
        """Default max_depth truncates lineage to 10 items."""
        nodes = [
            {"genome_id": f"g{i}", "name": f"n{i}", "generation": i, "fitness_score": 0.5}
            for i in range(20)
        ]
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = nodes
        mock_ledger.get_events_by_type.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/lineage/g0", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["generations"] == 10

    def test_max_depth_parameter(self, handler, http_handler):
        nodes = [
            {"genome_id": f"g{i}", "name": f"n{i}", "generation": i, "fitness_score": 0.5}
            for i in range(20)
        ]
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = nodes
        mock_ledger.get_events_by_type.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/lineage/g0", {"max_depth": ["5"]}, http_handler)

        body = _body(result)
        assert body["generations"] == 5

    def test_max_depth_clamped_minimum_1(self, handler, http_handler):
        nodes = [{"genome_id": "g0", "name": "n0", "generation": 0, "fitness_score": 0.5}]
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = nodes
        mock_ledger.get_events_by_type.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/lineage/g0", {"max_depth": ["-5"]}, http_handler)

        body = _body(result)
        assert body["generations"] == 1

    def test_max_depth_clamped_maximum_50(self, handler, http_handler):
        nodes = [
            {"genome_id": f"g{i}", "name": f"n{i}", "generation": i, "fitness_score": 0.5}
            for i in range(60)
        ]
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = nodes
        mock_ledger.get_events_by_type.return_value = []

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/lineage/g0", {"max_depth": ["200"]}, http_handler)

        body = _body(result)
        assert body["generations"] == 50

    def test_path_traversal_blocked(self, handler, http_handler):
        result = handler.handle("/api/genesis/lineage/../etc/shadow", {}, http_handler)
        assert result.status_code == 400

    def test_db_error_returns_500(self, handler, http_handler):
        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch(
                "aragora.server.handlers.genesis.GenesisLedger", side_effect=OSError("corrupt")
            ),
        ):
            result = handler.handle("/api/genesis/lineage/valid_id", {}, http_handler)
        assert result.status_code == 500

    def test_enrichment_event_lookup_failure_is_silent(self, handler, http_handler):
        """If event enrichment fails, it should not break the response."""
        mock_ledger = MagicMock()
        mock_ledger.get_lineage.return_value = [
            {"genome_id": "g1", "name": "n1", "generation": 0, "fitness_score": 0.5},
        ]
        mock_ledger.get_events_by_type.side_effect = ValueError("event store down")

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/lineage/g1", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert len(body["lineage"]) == 1
        # No event_type or created_at because enrichment failed silently
        assert "event_type" not in body["lineage"][0]


# ===========================================================================
# GET /api/genesis/tree/:debate_id
# ===========================================================================


class TestGetDebateTree:
    def test_503_when_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = handler.handle("/api/genesis/tree/debate_abc", {}, http_handler)
        assert result.status_code == 503

    def test_success(self, handler, http_handler):
        mock_tree = MagicMock()
        mock_tree.to_dict.return_value = {"root": "n1", "children": ["n2"]}
        mock_tree.nodes = ["n1", "n2", "n3"]

        mock_ledger = MagicMock()
        mock_ledger.get_debate_tree.return_value = mock_tree

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/tree/debate_abc", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["debate_id"] == "debate_abc"
        assert body["total_nodes"] == 3
        assert body["tree"]["root"] == "n1"

    def test_path_traversal_blocked(self, handler, http_handler):
        result = handler.handle("/api/genesis/tree/../etc/hosts", {}, http_handler)
        assert result.status_code == 400

    def test_invalid_debate_id_returns_400(self, handler, http_handler):
        result = handler.handle("/api/genesis/tree/<script>alert(1)</script>", {}, http_handler)
        assert result is not None
        assert result.status_code == 400

    def test_db_error_returns_500(self, handler, http_handler):
        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch(
                "aragora.server.handlers.genesis.GenesisLedger",
                side_effect=OSError("ledger corrupted"),
            ),
        ):
            result = handler.handle("/api/genesis/tree/debate_abc", {}, http_handler)
        assert result.status_code == 500

    def test_tree_exception_during_fetch(self, handler, http_handler):
        mock_ledger = MagicMock()
        mock_ledger.get_debate_tree.side_effect = ValueError("no such debate")

        with (
            patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenesisLedger", return_value=mock_ledger),
        ):
            result = handler.handle("/api/genesis/tree/debate_abc", {}, http_handler)

        assert result.status_code == 500


# ===========================================================================
# GET /api/genesis/population
# ===========================================================================


class TestGetPopulation:
    def test_503_when_breeding_unavailable(self, handler, http_handler):
        with patch.dict("sys.modules", {"aragora.genesis.breeding": None}):
            result = handler.handle("/api/genesis/population", {}, http_handler)
        assert result.status_code in [500, 503]

    def test_success(self, handler, http_handler):
        mock_genome = _mock_genome(genome_id="pop_g1", name="pop_agent")
        mock_population = MagicMock()
        mock_population.population_id = "pop_001"
        mock_population.generation = 3
        mock_population.size = 4
        mock_population.average_fitness = 0.77
        mock_population.debate_history = ["d1"]
        mock_population.genomes = [mock_genome]
        mock_population.best_genome = mock_genome

        mock_manager = MagicMock()
        mock_manager.get_or_create_population.return_value = mock_population

        with patch(
            "aragora.genesis.breeding.PopulationManager",
            return_value=mock_manager,
        ):
            result = handler.handle("/api/genesis/population", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["population_id"] == "pop_001"
        assert body["generation"] == 3
        assert body["size"] == 4
        assert body["average_fitness"] == 0.77
        assert body["debate_history_count"] == 1
        assert len(body["genomes"]) == 1
        assert body["genomes"][0]["genome_id"] == "pop_g1"
        assert body["best_genome"]["genome_id"] == "pop_g1"

    def test_population_no_best_genome(self, handler, http_handler):
        mock_population = MagicMock()
        mock_population.population_id = "pop_002"
        mock_population.generation = 0
        mock_population.size = 0
        mock_population.average_fitness = 0.0
        mock_population.debate_history = []
        mock_population.genomes = []
        mock_population.best_genome = None

        mock_manager = MagicMock()
        mock_manager.get_or_create_population.return_value = mock_population

        with patch(
            "aragora.genesis.breeding.PopulationManager",
            return_value=mock_manager,
        ):
            result = handler.handle("/api/genesis/population", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["best_genome"] is None
        assert body["genomes"] == []

    def test_population_genome_without_get_dominant_traits(self, handler, http_handler):
        """Genome that lacks get_dominant_traits uses fallback."""
        mock_genome = MagicMock(spec=[])  # no attrs by default
        mock_genome.genome_id = "g_no_traits"
        mock_genome.name = "plain"
        mock_genome.fitness_score = 0.6
        mock_genome.generation = 1
        mock_genome.traits = {"bold": 0.9, "calm": 0.3}
        mock_genome.expertise = {"math": 0.8}
        # No get_dominant_traits on this mock

        mock_population = MagicMock()
        mock_population.population_id = "pop_003"
        mock_population.generation = 1
        mock_population.size = 1
        mock_population.average_fitness = 0.6
        mock_population.debate_history = []
        mock_population.genomes = [mock_genome]
        mock_population.best_genome = None

        mock_manager = MagicMock()
        mock_manager.get_or_create_population.return_value = mock_population

        with patch(
            "aragora.genesis.breeding.PopulationManager",
            return_value=mock_manager,
        ):
            result = handler.handle("/api/genesis/population", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        # Falls back to list(genome.traits.keys())[:3]
        assert len(body["genomes"][0]["personality_traits"]) <= 3

    def test_db_error_returns_500(self, handler, http_handler):
        with patch(
            "aragora.genesis.breeding.PopulationManager",
            side_effect=OSError("DB crash"),
        ):
            result = handler.handle("/api/genesis/population", {}, http_handler)
        assert result.status_code == 500


# ===========================================================================
# GET /api/genesis/descendants/:genome_id
# ===========================================================================


class TestGetGenomeDescendants:
    def test_503_when_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", False):
            result = handler.handle("/api/genesis/descendants/genome_abc", {}, http_handler)
        assert result.status_code == 503

    def test_root_not_found_returns_404(self, handler, http_handler):
        mock_store = MagicMock()
        mock_store.get.return_value = None

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/descendants/unknown", {}, http_handler)

        assert result.status_code == 404

    def test_no_descendants(self, handler, http_handler):
        root = _mock_genome(genome_id="root")
        mock_store = MagicMock()
        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root]

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/descendants/root", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["total_descendants"] == 0
        assert body["descendants"] == []
        assert body["genome_id"] == "root"
        assert body["root_genome"]["name"] == "agent1"

    def test_with_descendants(self, handler, http_handler):
        root = _mock_genome(genome_id="root", generation=0, fitness=0.7)
        child1 = _mock_genome(
            genome_id="child1", generation=1, fitness=0.8, parent_genomes=["root"]
        )
        child2 = _mock_genome(
            genome_id="child2", generation=1, fitness=0.75, parent_genomes=["root"]
        )
        grandchild = _mock_genome(
            genome_id="grandchild", generation=2, fitness=0.85, parent_genomes=["child1"]
        )

        mock_store = MagicMock()
        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root, child1, child2, grandchild]

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/descendants/root", {}, http_handler)

        assert result.status_code == 200
        body = _body(result)
        assert body["total_descendants"] == 3
        assert body["max_generation"] == 2
        # Should be sorted by depth first
        depths = [d["depth"] for d in body["descendants"]]
        assert depths == sorted(depths)

    def test_max_depth_default_5(self, handler, http_handler):
        """Descendants are only fetched up to default max_depth of 5."""
        # Build a chain: root -> g1 -> g2 -> ... -> g7
        genomes = [_mock_genome(genome_id="root", generation=0)]
        for i in range(1, 8):
            genomes.append(
                _mock_genome(
                    genome_id=f"g{i}",
                    generation=i,
                    parent_genomes=[f"g{i - 1}" if i > 1 else "root"],
                )
            )

        mock_store = MagicMock()
        mock_store.get.return_value = genomes[0]
        mock_store.get_all.return_value = genomes

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/descendants/root", {}, http_handler)

        body = _body(result)
        # With max_depth=5, only depths 1-5 are included (not 6, 7)
        max_found_depth = max(d["depth"] for d in body["descendants"])
        assert max_found_depth <= 5

    def test_max_depth_parameter(self, handler, http_handler):
        root = _mock_genome(genome_id="root", generation=0)
        child = _mock_genome(genome_id="child", generation=1, parent_genomes=["root"])
        grandchild = _mock_genome(genome_id="grandchild", generation=2, parent_genomes=["child"])

        mock_store = MagicMock()
        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root, child, grandchild]

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle(
                "/api/genesis/descendants/root", {"max_depth": ["1"]}, http_handler
            )

        body = _body(result)
        # Only depth=1 (direct children)
        assert body["total_descendants"] == 1
        assert body["descendants"][0]["genome_id"] == "child"

    def test_max_depth_clamped_to_1_20(self, handler, http_handler):
        root = _mock_genome(genome_id="root", generation=0)
        mock_store = MagicMock()
        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root]

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            # Negative value should be clamped to 1
            result = handler.handle(
                "/api/genesis/descendants/root", {"max_depth": ["-10"]}, http_handler
            )
            assert result.status_code == 200

            # Over 20 should be clamped to 20
            result = handler.handle(
                "/api/genesis/descendants/root", {"max_depth": ["999"]}, http_handler
            )
            assert result.status_code == 200

    def test_path_traversal_blocked(self, handler, http_handler):
        result = handler.handle("/api/genesis/descendants/../etc/passwd", {}, http_handler)
        assert result.status_code == 400

    def test_invalid_genome_id_returns_400(self, handler, http_handler):
        result = handler.handle("/api/genesis/descendants/bad;id", {}, http_handler)
        assert result is not None
        assert result.status_code == 400

    def test_db_error_returns_500(self, handler, http_handler):
        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", side_effect=OSError("IO error")),
        ):
            result = handler.handle("/api/genesis/descendants/valid_id", {}, http_handler)
        assert result.status_code == 500

    def test_descendants_sorted_by_depth_then_fitness(self, handler, http_handler):
        root = _mock_genome(genome_id="root", generation=0, fitness=0.5)
        c1 = _mock_genome(genome_id="c1", generation=1, fitness=0.6, parent_genomes=["root"])
        c2 = _mock_genome(genome_id="c2", generation=1, fitness=0.9, parent_genomes=["root"])

        mock_store = MagicMock()
        mock_store.get.return_value = root
        mock_store.get_all.return_value = [root, c1, c2]

        with (
            patch("aragora.server.handlers.genesis.GENOME_AVAILABLE", True),
            patch("aragora.server.handlers.genesis.GenomeStore", return_value=mock_store),
        ):
            result = handler.handle("/api/genesis/descendants/root", {}, http_handler)

        body = _body(result)
        # Same depth, sorted by fitness descending
        assert body["descendants"][0]["fitness_score"] == 0.9
        assert body["descendants"][1]["fitness_score"] == 0.6


# ===========================================================================
# handle() dispatch and edge cases
# ===========================================================================


class TestHandleDispatch:
    def test_unknown_path_returns_none(self, handler, http_handler):
        result = handler.handle("/api/genesis/unknown_endpoint", {}, http_handler)
        assert result is None

    def test_handler_with_no_context(self, http_handler):
        """GenesisHandler accepts None context."""
        h = GenesisHandler(None)
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = h.handle("/api/genesis/stats", {}, http_handler)
        assert result.status_code == 503

    def test_handler_with_empty_context(self, http_handler):
        h = GenesisHandler({})
        with patch("aragora.server.handlers.genesis.GENESIS_AVAILABLE", False):
            result = h.handle("/api/genesis/stats", {}, http_handler)
        assert result.status_code == 503

    def test_inherits_from_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_routes_class_attribute_has_versioned_and_unversioned(self, handler):
        assert "/api/genesis/stats" in handler.ROUTES
        assert "/api/v1/genesis/stats" in handler.ROUTES
        assert "/api/genesis/descendants/*" in handler.ROUTES
        assert "/api/v1/genesis/descendants/*" in handler.ROUTES
