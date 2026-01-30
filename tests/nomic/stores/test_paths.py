"""Tests for canonical store path resolution."""

from __future__ import annotations

import os
from pathlib import Path

from aragora.nomic.stores.paths import resolve_store_dir


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
