"""Tests for ``aragora handlers`` CLI command (Gap 3).

Verifies list and routes subcommands with table/JSON output and tier filtering.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aragora.cli.commands.handlers import (
    _cmd_list_handlers,
    _cmd_list_routes,
    _get_registry_data,
)


class _FakeHandler:
    ROUTES = ["/api/v1/foo", "/api/v1/bar"]


class _FakeHandler2:
    ROUTES = ["/api/v1/baz"]


FAKE_REGISTRY = [
    ("_test_handler", _FakeHandler),
    ("_test_handler2", _FakeHandler2),
]

FAKE_TIERS = {
    "_test_handler": "core",
    "_test_handler2": "extended",
}


def _fake_get_registry_data(tier_filter=None):
    """Return fake registry data for testing."""
    all_data = [
        {
            "attr": "_test_handler",
            "class": "_FakeHandler",
            "tier": "core",
            "routes": ["/api/v1/foo", "/api/v1/bar"],
            "route_count": 2,
            "active": True,
        },
        {
            "attr": "_test_handler2",
            "class": "_FakeHandler2",
            "tier": "extended",
            "routes": ["/api/v1/baz"],
            "route_count": 1,
            "active": True,
        },
    ]
    if tier_filter is not None:
        return [e for e in all_data if e["tier"] == tier_filter]
    return all_data


@pytest.fixture(autouse=True)
def _mock_registry(monkeypatch):
    """Patch _get_registry_data with fake data for testing."""
    import aragora.cli.commands.handlers as mod

    monkeypatch.setattr(mod, "_get_registry_data", _fake_get_registry_data)


class TestHandlersList:
    """Test ``aragora handlers list``."""

    def test_table_output(self, capsys):
        args = type("A", (), {"tier": None, "json": False, "handlers_action": "list"})()
        result = _cmd_list_handlers(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "_FakeHandler" in out
        assert "core" in out
        assert "Total:" in out

    def test_json_output(self, capsys):
        args = type("A", (), {"tier": None, "json": True, "handlers_action": "list"})()
        result = _cmd_list_handlers(args)
        assert result == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_tier_filter(self, capsys):
        args = type("A", (), {"tier": "core", "json": False, "handlers_action": "list"})()
        result = _cmd_list_handlers(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "_FakeHandler" in out
        assert "_FakeHandler2" not in out


class TestHandlersRoutes:
    """Test ``aragora handlers routes``."""

    def test_routes_table_output(self, capsys):
        args = type("A", (), {"tier": None, "json": False, "handlers_action": "routes"})()
        result = _cmd_list_routes(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "/api/v1/foo" in out
        assert "/api/v1/baz" in out
        assert "Total:" in out

    def test_routes_json_output(self, capsys):
        args = type("A", (), {"tier": None, "json": True, "handlers_action": "routes"})()
        result = _cmd_list_routes(args)
        assert result == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)
        assert len(data) == 3
        paths = [r["path"] for r in data]
        assert "/api/v1/foo" in paths

    def test_routes_sorted_by_path(self, capsys):
        args = type("A", (), {"tier": None, "json": True, "handlers_action": "routes"})()
        _cmd_list_routes(args)
        data = json.loads(capsys.readouterr().out)
        paths = [r["path"] for r in data]
        assert paths == sorted(paths)


class TestParserRegistration:
    """Test that the handlers command is registered in the CLI parser."""

    def test_handlers_in_parser(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["handlers", "list"])
        assert args.command == "handlers"
