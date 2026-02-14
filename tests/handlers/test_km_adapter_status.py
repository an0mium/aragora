"""Tests for KM adapter status endpoint.

Tests:
- GET /api/v1/knowledge/adapters - list all adapters with status
- Adapter spec fields present
- Coordinator live status merge
- Module unavailable returns 503
- Unrelated paths return None
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge.adapters import KMAdapterStatusHandler


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture
def handler():
    """Create a KMAdapterStatusHandler instance."""
    return KMAdapterStatusHandler(ctx={})


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    return MagicMock()


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_adapters_path(self, handler):
        assert handler.can_handle("/api/v1/knowledge/adapters") is True

    def test_does_not_handle_other_path(self, handler):
        assert handler.can_handle("/api/v1/knowledge/search") is False

    def test_handles_without_version(self, handler):
        assert handler.can_handle("/api/knowledge/adapters") is True


class TestListAdapters:
    """Tests for GET /api/v1/knowledge/adapters."""

    def test_list_adapters_returns_200(self, handler, mock_handler):
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_list_adapters_has_expected_fields(self, handler, mock_handler):
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = parse_body(result)
        assert "adapters" in body
        assert "total" in body
        assert "coordinator_available" in body
        assert isinstance(body["adapters"], list)
        assert body["total"] >= 0

    def test_adapter_has_spec_fields(self, handler, mock_handler):
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = parse_body(result)
        if body["total"] > 0:
            adapter = body["adapters"][0]
            assert "name" in adapter
            assert "priority" in adapter
            assert "enabled_by_default" in adapter
            assert "required_deps" in adapter
            assert "status" in adapter

    def test_adapters_sorted_by_name(self, handler, mock_handler):
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = parse_body(result)
        names = [a["name"] for a in body["adapters"]]
        assert names == sorted(names)

    def test_coordinator_not_available(self, handler, mock_handler):
        """Without coordinator in context, adapters show as registered."""
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = parse_body(result)
        assert body["coordinator_available"] is False
        for adapter in body["adapters"]:
            assert adapter["status"] == "registered"

    def test_with_coordinator_context(self, mock_handler):
        """With coordinator in context, adapters get live status."""
        mock_coordinator = MagicMock()
        mock_coordinator.get_status.return_value = {
            "adapters": {
                "continuum": {
                    "enabled": True,
                    "has_reverse": True,
                    "forward_errors": 0,
                    "reverse_errors": 1,
                    "priority": 100,
                    "last_forward_sync": "2026-02-14T10:00:00",
                    "last_reverse_sync": None,
                },
            },
        }

        handler = KMAdapterStatusHandler(ctx={"km_coordinator": mock_coordinator})
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = parse_body(result)

        assert body["coordinator_available"] is True

        continuum = next(
            (a for a in body["adapters"] if a["name"] == "continuum"), None
        )
        if continuum:
            assert continuum["status"] == "active"
            assert continuum["enabled"] is True
            assert continuum["forward_errors"] == 0
            assert continuum["reverse_errors"] == 1

    def test_coordinator_error_handled_gracefully(self, mock_handler):
        """Coordinator errors should not crash the endpoint."""
        mock_coordinator = MagicMock()
        mock_coordinator.get_status.side_effect = RuntimeError("connection lost")

        handler = KMAdapterStatusHandler(ctx={"km_coordinator": mock_coordinator})
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = parse_body(result)

        assert result.status_code == 200
        assert body["coordinator_available"] is True

    def test_total_matches_adapter_count(self, handler, mock_handler):
        result = handler.handle("/api/v1/knowledge/adapters", {}, mock_handler)
        body = parse_body(result)
        assert body["total"] == len(body["adapters"])


class TestUnavailable:
    """Tests when KM system is unavailable."""

    def test_km_unavailable_returns_503(self, handler, mock_handler):
        with patch(
            "aragora.server.handlers.knowledge.adapters.KM_AVAILABLE", False
        ):
            result = handler.handle(
                "/api/v1/knowledge/adapters", {}, mock_handler
            )
            assert result.status_code == 503


class TestUnhandledRoutes:
    """Tests for paths not handled by this handler."""

    def test_unrelated_path_returns_none(self, handler, mock_handler):
        result = handler.handle("/api/v1/knowledge/search", {}, mock_handler)
        assert result is None
