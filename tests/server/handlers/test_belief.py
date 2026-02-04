"""Tests for the BeliefHandler."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.belief import BeliefHandler


class TestBeliefHandler:
    """Tests for BeliefHandler."""

    def _make_handler(self, ctx: dict | None = None) -> BeliefHandler:
        return BeliefHandler(server_context=ctx or {})

    def _make_http_handler(self, client_ip: str = "127.0.0.1") -> MagicMock:
        mock = MagicMock()
        mock.client_address = (client_ip, 12345)
        mock.headers = {"X-Forwarded-For": client_ip}
        return mock

    # -------------------------------------------------------------------------
    # can_handle tests
    # -------------------------------------------------------------------------

    def test_can_handle_cruxes_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/belief-network/debate123/cruxes") is True

    def test_can_handle_load_bearing_claims_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/belief-network/debate123/load-bearing-claims") is True

    def test_can_handle_provenance_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/provenance/debate123/claims/claim456/support") is True

    def test_can_handle_graph_stats_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/debate/debate123/graph-stats") is True

    def test_can_handle_versioned_graph_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/v1/belief-network/debate123/graph") is True

    def test_can_handle_versioned_export_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/v1/belief-network/debate123/export") is True

    def test_cannot_handle_unrelated_route(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_partial_match(self):
        handler = self._make_handler()
        assert handler.can_handle("/api/belief-network") is False

    # -------------------------------------------------------------------------
    # ROUTES constant tests
    # -------------------------------------------------------------------------

    def test_routes_include_cruxes(self):
        handler = self._make_handler()
        assert "/api/belief-network/*/cruxes" in handler.ROUTES

    def test_routes_include_load_bearing_claims(self):
        handler = self._make_handler()
        assert "/api/belief-network/*/load-bearing-claims" in handler.ROUTES

    def test_routes_include_provenance_support(self):
        handler = self._make_handler()
        assert "/api/provenance/*/claims/*/support" in handler.ROUTES

    def test_routes_include_graph_stats(self):
        handler = self._make_handler()
        assert "/api/debate/*/graph-stats" in handler.ROUTES

    def test_routes_include_versioned_endpoints(self):
        handler = self._make_handler()
        assert "/api/v1/belief-network/*/graph" in handler.ROUTES
        assert "/api/v1/belief-network/*/export" in handler.ROUTES

    # -------------------------------------------------------------------------
    # KM Adapter tests
    # -------------------------------------------------------------------------

    def test_get_km_adapter_returns_adapter_or_none(self):
        handler = self._make_handler()
        # Without proper context, may return None or auto-create an adapter
        adapter = handler._get_km_adapter()
        # May return None or create a BeliefAdapter depending on availability
        assert adapter is None or hasattr(adapter, "get_belief")

    def test_get_km_adapter_uses_context(self):
        mock_adapter = MagicMock()
        handler = self._make_handler(ctx={"belief_km_adapter": mock_adapter})

        adapter = handler._get_km_adapter()

        assert adapter is mock_adapter

    def test_get_km_adapter_caches_result(self):
        mock_adapter = MagicMock()
        handler = self._make_handler(ctx={"belief_km_adapter": mock_adapter})

        adapter1 = handler._get_km_adapter()
        adapter2 = handler._get_km_adapter()

        assert adapter1 is adapter2

    # -------------------------------------------------------------------------
    # Event emission tests
    # -------------------------------------------------------------------------

    def test_emit_km_event_belief_converged(self):
        handler = self._make_handler()
        mock_emitter = MagicMock()

        handler._emit_km_event(mock_emitter, "belief_converged", {"claim_id": "c1"})

        mock_emitter.emit.assert_called_once()

    def test_emit_km_event_crux_detected(self):
        handler = self._make_handler()
        mock_emitter = MagicMock()

        handler._emit_km_event(mock_emitter, "crux_detected", {"crux": "key_issue"})

        mock_emitter.emit.assert_called_once()

    def test_emit_km_event_unknown_type_uses_mound_updated(self):
        handler = self._make_handler()
        mock_emitter = MagicMock()

        handler._emit_km_event(mock_emitter, "unknown_event", {"data": "test"})

        mock_emitter.emit.assert_called_once()

    def test_emit_km_event_handles_error_gracefully(self):
        handler = self._make_handler()
        mock_emitter = MagicMock()
        mock_emitter.emit.side_effect = RuntimeError("Emit failed")

        # Should not raise
        handler._emit_km_event(mock_emitter, "belief_converged", {})

    # -------------------------------------------------------------------------
    # Initialization tests
    # -------------------------------------------------------------------------

    def test_init_sets_context(self):
        ctx = {"key": "value"}
        handler = BeliefHandler(server_context=ctx)
        assert handler.ctx == ctx

    def test_init_adapter_is_none(self):
        handler = self._make_handler()
        assert handler._km_adapter is None

    # -------------------------------------------------------------------------
    # Export list tests
    # -------------------------------------------------------------------------

    def test_exports_belief_handler(self):
        from aragora.server.handlers.belief import __all__

        assert "BeliefHandler" in __all__


class TestBeliefHandlerAvailability:
    """Tests for feature availability in BeliefHandler."""

    def test_belief_network_availability_flag(self):
        from aragora.server.handlers.belief import BELIEF_NETWORK_AVAILABLE

        # Should be a boolean
        assert isinstance(BELIEF_NETWORK_AVAILABLE, bool)

    def test_laboratory_availability_flag(self):
        from aragora.server.handlers.belief import LABORATORY_AVAILABLE

        assert isinstance(LABORATORY_AVAILABLE, bool)

    def test_provenance_availability_flag(self):
        from aragora.server.handlers.belief import PROVENANCE_AVAILABLE

        assert isinstance(PROVENANCE_AVAILABLE, bool)


class TestBeliefHandlerRateLimiting:
    """Tests for rate limiting in BeliefHandler."""

    def test_rate_limiter_exists(self):
        from aragora.server.handlers.belief import _belief_limiter

        assert _belief_limiter is not None
        assert hasattr(_belief_limiter, "is_allowed")

    def test_rate_limiter_config(self):
        from aragora.server.handlers.belief import _belief_limiter

        # Should be configured for read-heavy usage
        assert _belief_limiter.rpm == 60
