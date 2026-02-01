"""Tests for canonical store path resolution."""

from __future__ import annotations

import os
from pathlib import Path

import tempfile

from aragora.nomic.stores.paths import (
    resolve_runtime_store_dir,
    resolve_store_dir,
    should_use_canonical_store,
)


def test_env_override_takes_precedence(tmp_path, monkeypatch):
    env_dir = tmp_path / "env-store"
    monkeypatch.setenv("ARAGORA_STORE_DIR", str(env_dir))
    try:
        resolved = resolve_store_dir()
        assert resolved == env_dir
    finally:
        monkeypatch.delenv("ARAGORA_STORE_DIR", raising=False)


def test_workspace_root_resolves_to_aragora_beads(tmp_path):
    resolved = resolve_store_dir(workspace_root=tmp_path)
    assert resolved == tmp_path / ".aragora_beads"


def test_legacy_gt_fallback(tmp_path, monkeypatch):
    cwd = Path.cwd()
    legacy = tmp_path / ".gt"
    legacy.mkdir()
    (legacy / "beads").mkdir()
    (legacy / "convoys").mkdir()

    monkeypatch.chdir(tmp_path)
    try:
        resolved = resolve_store_dir()
        assert resolved.resolve() == (legacy / "beads").resolve()
    finally:
        monkeypatch.chdir(cwd)


def test_runtime_store_defaults_to_persistent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARAGORA_CANONICAL_STORE_PERSIST", raising=False)
    monkeypatch.delenv("NOMIC_CANONICAL_STORE_PERSIST", raising=False)
    monkeypatch.delenv("ARAGORA_STORE_DIR", raising=False)
    path = resolve_runtime_store_dir()
    assert path.resolve() == (tmp_path / ".aragora_beads").resolve()


def test_runtime_store_uses_ephemeral_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARAGORA_CANONICAL_STORE_PERSIST", "0")
    path = resolve_runtime_store_dir()
    assert path.exists()
    assert str(path).startswith(tempfile.gettempdir())


def test_runtime_store_uses_env_dir(tmp_path, monkeypatch):
    env_dir = tmp_path / "env-store"
    monkeypatch.setenv("ARAGORA_STORE_DIR", str(env_dir))
    path = resolve_runtime_store_dir()
    assert path == env_dir


def test_create_bead_store_resolves_default(monkeypatch, tmp_path):
    """create_bead_store uses canonical resolution when no override set."""
    from aragora.nomic.beads import create_bead_store

    orig_cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)
    try:
        loop = __import__("asyncio").get_event_loop()
        store = loop.run_until_complete(create_bead_store())
        assert store.bead_dir == resolve_store_dir()
    finally:
        monkeypatch.chdir(orig_cwd)


def test_convoy_executor_env_override(monkeypatch, tmp_path):
    """Convoy executor uses canonical store when env toggle is set."""
    from aragora.nomic.convoy_executor import GastownConvoyExecutor

    monkeypatch.setenv("NOMIC_CONVOY_CANONICAL_STORE", "1")
    executor = GastownConvoyExecutor(
        repo_path=tmp_path,
        implementers=[],
        reviewers=[],
    )
    assert executor.bead_dir == tmp_path / ".aragora_beads"


def test_convoy_coordinator_default_storage(monkeypatch, tmp_path):
    """ConvoyCoordinator defaults to canonical storage when no override set."""
    from unittest.mock import MagicMock

    from aragora.nomic.convoy_coordinator import ConvoyCoordinator
    from aragora.nomic.stores.paths import resolve_store_dir

    monkeypatch.chdir(tmp_path)
    manager = MagicMock()
    manager.bead_store = MagicMock()
    coord = ConvoyCoordinator(convoy_manager=manager, hierarchy=MagicMock())
    assert coord.storage_dir.resolve() == resolve_store_dir().resolve()


def test_should_use_canonical_store_env(monkeypatch):
    monkeypatch.delenv("ARAGORA_CONVOY_CANONICAL_STORE", raising=False)
    monkeypatch.delenv("NOMIC_CONVOY_CANONICAL_STORE", raising=False)
    assert should_use_canonical_store(default=False) is False

    monkeypatch.setenv("ARAGORA_CONVOY_CANONICAL_STORE", "1")
    assert should_use_canonical_store(default=False) is True

    monkeypatch.setenv("ARAGORA_CONVOY_CANONICAL_STORE", "0")
    assert should_use_canonical_store(default=True) is False
