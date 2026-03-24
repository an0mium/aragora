from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app
from aragora.server.fastapi import factory


@contextmanager
def _patched_startup_dependencies():
    with ExitStack() as stack:
        mocked = {}
        mocked["storage"] = stack.enter_context(patch("aragora.server.storage.DebateStorage"))
        stack.enter_context(patch("aragora.ranking.elo.EloSystem", return_value=MagicMock()))
        stack.enter_context(
            patch("aragora.storage.user_store.get_user_store", return_value=MagicMock())
        )
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
