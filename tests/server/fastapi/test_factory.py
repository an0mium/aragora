from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app
from aragora.server.fastapi import factory
from aragora.storage.connection_factory import DatabaseConfig, StorageBackendType


@contextmanager
def _patched_startup_dependencies():
    with ExitStack() as stack:
        mocked = {}
        mocked["storage"] = stack.enter_context(patch("aragora.server.storage.DebateStorage"))
        mocked["get_user_store"] = stack.enter_context(
            patch("aragora.storage.user_store.get_user_store", return_value=MagicMock())
        )
        stack.enter_context(patch("aragora.ranking.elo.EloSystem", return_value=MagicMock()))
        stack.enter_context(
            patch("aragora.memory.continuum.get_continuum_memory", return_value=MagicMock())
        )
        mock_cross_config = stack.enter_context(
            patch("aragora.memory.cross_debate_rlm.CrossDebateConfig")
        )
        mock_cross_config.return_value = MagicMock()
        stack.enter_context(
            patch("aragora.memory.cross_debate_rlm.CrossDebateMemory", return_value=MagicMock())
        )
        stack.enter_context(
            patch("aragora.knowledge.mound.get_knowledge_mound", return_value=MagicMock())
        )
        stack.enter_context(
            patch("aragora.rbac.checker.get_permission_checker", return_value=MagicMock())
        )
        stack.enter_context(
            patch("aragora.debate.decision_service.get_decision_service", return_value=MagicMock())
        )
        stack.enter_context(
            patch("aragora.server.middleware.deprecation_enforcer.register_default_deprecations")
        )
        yield mocked


def test_build_server_context_initializes_debate_storage_in_nomic_dir(tmp_path: Path):
    with _patched_startup_dependencies() as mocked:
        ctx = factory._build_server_context(tmp_path)

    mocked["storage"].assert_called_once_with(str(tmp_path / "debates.db"))
    assert ctx["storage"] is mocked["storage"].return_value


def test_create_app_lifespan_starts_with_fastapi_context(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARAGORA_NOMIC_DIR", str(tmp_path))

    with _patched_startup_dependencies() as mocked:
        mocked["storage"].return_value = object()
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/healthz")

    assert response.status_code == 200
    mocked["storage"].assert_called_once_with(str(tmp_path / "debates.db"))


def test_build_server_context_defers_postgres_user_store_in_async_context(tmp_path: Path):
    config = DatabaseConfig(
        backend_type=StorageBackendType.POSTGRES,
        dsn="postgresql://example",
        is_supabase=False,
    )

    with (
        _patched_startup_dependencies() as mocked,
        patch(
            "aragora.storage.connection_factory.resolve_database_config",
            return_value=config,
        ),
        patch(
            "aragora.storage.pool_manager.is_pool_initialized",
            return_value=False,
        ),
    ):
        ctx = asyncio.run(_async_build_server_context(tmp_path))

    assert ctx["user_store"] is None
    mocked["get_user_store"].assert_not_called()


async def _async_build_server_context(tmp_path: Path):
    return factory._build_server_context(tmp_path)
