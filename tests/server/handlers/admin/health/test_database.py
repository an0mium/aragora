"""Tests for database health check implementations."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock
import sqlite3

import pytest


class MockHandler:
    """Mock handler for testing database health functions."""

    def __init__(
        self,
        storage: Any = None,
        elo_system: Any = None,
        nomic_dir: Path | None = None,
        ctx: Dict[str, Any] | None = None,
    ):
        self._storage = storage
        self._elo_system = elo_system
        self._nomic_dir = nomic_dir
        self.ctx = ctx or {}

    def get_storage(self) -> Any:
        return self._storage

    def get_elo_system(self) -> Any:
        return self._elo_system

    def get_nomic_dir(self) -> Path | None:
        return self._nomic_dir


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


class TestDatabaseSchemaHealth:
    """Tests for database_schema_health function."""

    def test_schema_healthy(self):
        """Test healthy schema returns 200."""
        from aragora.server.handlers.admin.health.database import database_schema_health

        handler = MockHandler()
        mock_health = {
            "status": "healthy",
            "databases": {
                "core.db": {"exists": True, "tables": ["debates", "traces"]},
            },
        }

        with patch(
            "aragora.persistence.validator.get_database_health",
            return_value=mock_health,
        ):
            result = database_schema_health(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"

    def test_schema_degraded(self):
        """Test degraded schema returns 503."""
        from aragora.server.handlers.admin.health.database import database_schema_health

        handler = MockHandler()
        mock_health = {
            "status": "degraded",
            "missing_tables": ["debates"],
        }

        with patch(
            "aragora.persistence.validator.get_database_health",
            return_value=mock_health,
        ):
            result = database_schema_health(handler)

        assert result.status_code == 503
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "degraded"

    def test_schema_validator_not_available(self):
        """Test returns 503 when validator module not available."""
        from aragora.server.handlers.admin.health.database import database_schema_health

        handler = MockHandler()

        with patch(
            "aragora.persistence.validator.get_database_health",
            side_effect=ImportError("Module not found"),
        ):
            result = database_schema_health(handler)

        assert result.status_code == 503
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "unavailable"

    def test_schema_check_error(self):
        """Test returns 500 on unexpected error."""
        from aragora.server.handlers.admin.health.database import database_schema_health

        handler = MockHandler()

        with patch(
            "aragora.persistence.validator.get_database_health",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = database_schema_health(handler)

        assert result.status_code == 500
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "error"


class TestDatabaseStoresHealth:
    """Tests for database_stores_health function."""

    def test_all_stores_healthy(self):
        """Test all stores healthy returns proper response."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = [{"id": "1"}]

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"agent": "claude", "elo": 1500}]

        handler = MockHandler(
            storage=mock_storage,
            elo_system=mock_elo,
            ctx={"insight_store": MagicMock()},
        )

        result = database_stores_health(handler)

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"
        assert "stores" in body
        assert body["stores"]["debate_storage"]["healthy"] is True
        assert body["stores"]["elo_system"]["healthy"] is True

    def test_storage_not_initialized(self):
        """Test storage not initialized is treated as healthy."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        handler = MockHandler(storage=None, elo_system=None)

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["debate_storage"]["healthy"] is True
        assert body["stores"]["debate_storage"]["status"] == "not_initialized"

    def test_storage_database_error(self):
        """Test storage database error is reported."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = sqlite3.Error("Database locked")

        handler = MockHandler(storage=mock_storage)

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["debate_storage"]["healthy"] is False
        assert body["stores"]["debate_storage"]["error_type"] == "database"

    def test_elo_system_error(self):
        """Test ELO system error is reported."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = ValueError("Invalid state")

        handler = MockHandler(elo_system=mock_elo)

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["elo_system"]["healthy"] is False

    def test_insight_store_connected(self):
        """Test insight store connected status."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        mock_insight = MagicMock()
        mock_insight.__class__.__name__ = "InsightStore"

        handler = MockHandler(ctx={"insight_store": mock_insight})

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["insight_store"]["healthy"] is True
        assert body["stores"]["insight_store"]["status"] == "connected"

    def test_flip_detector_not_initialized(self):
        """Test flip detector not initialized status."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        handler = MockHandler(ctx={})

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["flip_detector"]["healthy"] is True
        assert body["stores"]["flip_detector"]["status"] == "not_initialized"

    def test_user_store_connected(self):
        """Test user store connected status."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        mock_user_store = MagicMock()
        handler = MockHandler(ctx={"user_store": mock_user_store})

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["user_store"]["healthy"] is True
        assert body["stores"]["user_store"]["status"] == "connected"

    def test_consensus_memory_exists(self, tmp_path):
        """Test consensus memory exists status."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        # Create a mock consensus memory file
        consensus_file = tmp_path / "consensus_memory.db"
        consensus_file.touch()

        handler = MockHandler(nomic_dir=tmp_path)

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["consensus_memory"]["healthy"] is True
        assert body["stores"]["consensus_memory"]["status"] == "exists"

    def test_consensus_memory_not_initialized(self, tmp_path):
        """Test consensus memory not initialized."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        handler = MockHandler(nomic_dir=tmp_path)

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["consensus_memory"]["healthy"] is True
        assert body["stores"]["consensus_memory"]["status"] == "not_initialized"

    def test_agent_metadata_connected(self, tmp_path):
        """Test agent metadata connected status."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        # Create a mock ELO database with agent_metadata table
        elo_path = tmp_path / "elo.db"
        conn = sqlite3.connect(elo_path)
        conn.execute("CREATE TABLE agent_metadata (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO agent_metadata VALUES (1, 'claude')")
        conn.commit()
        conn.close()

        handler = MockHandler(nomic_dir=tmp_path)

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["agent_metadata"]["healthy"] is True
        assert body["stores"]["agent_metadata"]["status"] == "connected"
        assert body["stores"]["agent_metadata"]["agent_count"] == 1

    def test_agent_metadata_table_not_exists(self, tmp_path):
        """Test agent metadata table not exists."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        # Create ELO database without agent_metadata table
        elo_path = tmp_path / "elo.db"
        conn = sqlite3.connect(elo_path)
        conn.execute("CREATE TABLE ratings (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        handler = MockHandler(nomic_dir=tmp_path)

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["agent_metadata"]["healthy"] is True
        assert body["stores"]["agent_metadata"]["status"] == "table_not_exists"

    def test_summary_statistics(self):
        """Test summary statistics are calculated correctly."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = []

        handler = MockHandler(
            storage=mock_storage,
            ctx={
                "insight_store": MagicMock(),
                "user_store": MagicMock(),
            },
        )

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "summary" in body
        assert body["summary"]["total"] > 0
        assert body["summary"]["healthy"] >= 0
        assert body["summary"]["connected"] >= 0
        assert body["summary"]["not_initialized"] >= 0

    def test_elapsed_time_tracked(self):
        """Test elapsed time is tracked."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        handler = MockHandler()

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "elapsed_ms" in body
        assert body["elapsed_ms"] >= 0

    def test_integration_store_module_not_available(self):
        """Test integration store module not available."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        handler = MockHandler(ctx={})

        with patch.dict(
            "sys.modules",
            {"aragora.storage.integration_store": None},
        ):
            result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        # Module import error should be handled gracefully
        assert body["stores"]["integration_store"]["healthy"] is True

    def test_gmail_token_store_not_initialized(self):
        """Test gmail token store not initialized."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        handler = MockHandler(ctx={})

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["gmail_token_store"]["healthy"] is True

    def test_sync_store_not_initialized(self):
        """Test sync store not initialized."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        handler = MockHandler(ctx={})

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["sync_store"]["healthy"] is True

    def test_decision_result_store_connected(self):
        """Test decision result store connected."""
        from aragora.server.handlers.admin.health.database import database_stores_health

        mock_store = MagicMock()
        handler = MockHandler(ctx={"decision_result_store": mock_store})

        result = database_stores_health(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["decision_result_store"]["healthy"] is True
        assert body["stores"]["decision_result_store"]["status"] == "connected"
