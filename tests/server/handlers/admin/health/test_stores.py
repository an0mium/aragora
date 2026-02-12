"""Tests for StoresMixin health check implementations."""

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
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.stores import StoresMixin


class TestStoresHandler(StoresMixin):
    """Test handler implementing StoresMixin."""

    def __init__(
        self,
        storage: Any = None,
        elo_system: Any = None,
        nomic_dir: Path | None = None,
        ctx: dict[str, Any] | None = None,
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


class TestDatabaseStoresHealth:
    """Tests for database_stores_health method."""

    def test_all_stores_healthy(self):
        """Test returns healthy when all stores are working."""
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = []

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []

        handler = TestStoresHandler(
            storage=mock_storage,
            elo_system=mock_elo,
            ctx={"insight_store": MagicMock()},
        )

        result = handler.database_stores_health()

        assert result.status_code == 200
        body = json.loads(result.body.decode("utf-8"))
        assert body["status"] == "healthy"

    def test_storage_not_initialized(self):
        """Test storage not initialized is treated as healthy."""
        handler = TestStoresHandler(storage=None)

        result = handler.database_stores_health()

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["debate_storage"]["healthy"] is True
        assert body["stores"]["debate_storage"]["status"] == "not_initialized"

    def test_storage_database_error(self):
        """Test storage database error is reported."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = sqlite3.Error("Database locked")

        handler = TestStoresHandler(storage=mock_storage)

        result = handler.database_stores_health()

        body = json.loads(result.body.decode("utf-8"))
        assert body["stores"]["debate_storage"]["healthy"] is False
        assert body["stores"]["debate_storage"]["error_type"] == "database"
        assert body["status"] == "degraded"


class TestCheckDebateStorage:
    """Tests for _check_debate_storage method."""

    def test_storage_connected(self):
        """Test storage connected returns healthy."""
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = []
        mock_storage.__class__.__name__ = "SQLiteStorage"

        handler = TestStoresHandler(storage=mock_storage)

        result, healthy = handler._check_debate_storage()

        assert healthy is True
        assert result["status"] == "connected"
        assert result["type"] == "SQLiteStorage"

    def test_storage_io_error(self):
        """Test storage IO error is handled."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = OSError("Disk full")

        handler = TestStoresHandler(storage=mock_storage)

        result, healthy = handler._check_debate_storage()

        assert healthy is False
        assert result["error_type"] == "database"

    def test_storage_attribute_error(self):
        """Test storage attribute error is handled."""
        mock_storage = MagicMock()
        mock_storage.list_recent.side_effect = AttributeError("Missing method")

        handler = TestStoresHandler(storage=mock_storage)

        result, healthy = handler._check_debate_storage()

        assert healthy is False
        assert result["error_type"] == "data_access"


class TestCheckEloSystem:
    """Tests for _check_elo_system method."""

    def test_elo_connected(self):
        """Test ELO system connected returns healthy."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"agent": "claude", "elo": 1500}]

        handler = TestStoresHandler(elo_system=mock_elo)

        result, healthy = handler._check_elo_system()

        assert healthy is True
        assert result["status"] == "connected"
        assert result["agent_count"] == 1

    def test_elo_not_initialized(self):
        """Test ELO not initialized returns healthy."""
        handler = TestStoresHandler(elo_system=None)

        result, healthy = handler._check_elo_system()

        assert healthy is True
        assert result["status"] == "not_initialized"

    def test_elo_database_error(self):
        """Test ELO database error is handled."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = sqlite3.Error("Database error")

        handler = TestStoresHandler(elo_system=mock_elo)

        result, healthy = handler._check_elo_system()

        assert healthy is False
        assert result["error_type"] == "database"


class TestCheckInsightStore:
    """Tests for _check_insight_store method."""

    def test_insight_store_connected(self):
        """Test insight store connected."""
        mock_store = MagicMock()
        mock_store.__class__.__name__ = "InsightStore"

        handler = TestStoresHandler(ctx={"insight_store": mock_store})

        result = handler._check_insight_store()

        assert result["healthy"] is True
        assert result["status"] == "connected"
        assert result["type"] == "InsightStore"

    def test_insight_store_not_initialized(self):
        """Test insight store not initialized."""
        handler = TestStoresHandler(ctx={})

        result = handler._check_insight_store()

        assert result["healthy"] is True
        assert result["status"] == "not_initialized"


class TestCheckFlipDetector:
    """Tests for _check_flip_detector method."""

    def test_flip_detector_connected(self):
        """Test flip detector connected."""
        mock_detector = MagicMock()

        handler = TestStoresHandler(ctx={"flip_detector": mock_detector})

        result = handler._check_flip_detector()

        assert result["healthy"] is True
        assert result["status"] == "connected"

    def test_flip_detector_not_initialized(self):
        """Test flip detector not initialized."""
        handler = TestStoresHandler(ctx={})

        result = handler._check_flip_detector()

        assert result["healthy"] is True
        assert result["status"] == "not_initialized"


class TestCheckConsensusMemory:
    """Tests for _check_consensus_memory method."""

    def test_consensus_memory_exists(self, tmp_path):
        """Test consensus memory exists."""
        consensus_file = tmp_path / "consensus_memory.db"
        consensus_file.touch()

        handler = TestStoresHandler(nomic_dir=tmp_path)

        result = handler._check_consensus_memory()

        assert result["healthy"] is True
        assert result["status"] == "exists"

    def test_consensus_memory_not_initialized(self, tmp_path):
        """Test consensus memory not initialized."""
        handler = TestStoresHandler(nomic_dir=tmp_path)

        result = handler._check_consensus_memory()

        assert result["healthy"] is True
        assert result["status"] == "not_initialized"

    def test_consensus_memory_nomic_dir_not_set(self):
        """Test consensus memory nomic dir not set."""
        handler = TestStoresHandler(nomic_dir=None)

        result = handler._check_consensus_memory()

        assert result["healthy"] is True
        assert result["status"] == "nomic_dir_not_set"


class TestCheckAgentMetadata:
    """Tests for _check_agent_metadata method."""

    def test_agent_metadata_connected(self, tmp_path):
        """Test agent metadata table exists."""
        elo_path = tmp_path / "elo.db"
        conn = sqlite3.connect(elo_path)
        conn.execute("CREATE TABLE agent_metadata (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO agent_metadata VALUES (1, 'claude')")
        conn.execute("INSERT INTO agent_metadata VALUES (2, 'gemini')")
        conn.commit()
        conn.close()

        handler = TestStoresHandler(nomic_dir=tmp_path)

        result = handler._check_agent_metadata()

        assert result["healthy"] is True
        assert result["status"] == "connected"
        assert result["agent_count"] == 2

    def test_agent_metadata_table_not_exists(self, tmp_path):
        """Test agent metadata table not exists."""
        elo_path = tmp_path / "elo.db"
        conn = sqlite3.connect(elo_path)
        conn.execute("CREATE TABLE ratings (id INTEGER)")
        conn.commit()
        conn.close()

        handler = TestStoresHandler(nomic_dir=tmp_path)

        result = handler._check_agent_metadata()

        assert result["healthy"] is True
        assert result["status"] == "table_not_exists"

    def test_agent_metadata_database_not_exists(self, tmp_path):
        """Test agent metadata database not exists."""
        handler = TestStoresHandler(nomic_dir=tmp_path)

        result = handler._check_agent_metadata()

        assert result["healthy"] is True
        assert result["status"] == "database_not_exists"


class TestCheckIntegrationStore:
    """Tests for _check_integration_store method."""

    def test_integration_store_connected(self):
        """Test integration store connected."""
        mock_store = MagicMock()

        handler = TestStoresHandler(ctx={"integration_store": mock_store})

        # Mock the import to avoid ImportError
        with patch.dict("sys.modules", {"aragora.storage.integration_store": MagicMock()}):
            result = handler._check_integration_store()

        assert result["healthy"] is True
        assert result["status"] == "connected"

    def test_integration_store_not_initialized(self):
        """Test integration store not initialized."""
        handler = TestStoresHandler(ctx={})

        result = handler._check_integration_store()

        assert result["healthy"] is True
