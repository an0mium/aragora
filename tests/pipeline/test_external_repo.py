"""Tests for External Repository Targeting."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aragora.pipeline.external_repo import (
    ExternalRepoManager,
    RepoContext,
    RepoTarget,
)


class TestRepoTarget:
    def test_repo_name_from_url(self):
        target = RepoTarget(url="https://github.com/user/myrepo.git")
        assert target.repo_name == "myrepo"

    def test_repo_name_without_git(self):
        target = RepoTarget(url="https://github.com/user/myrepo")
        assert target.repo_name == "myrepo"

    def test_clone_url_with_token(self):
        target = RepoTarget(
            url="https://github.com/user/repo.git",
            auth_token="ghp_secret",
        )
        assert "x-access-token:ghp_secret" in target.clone_url

    def test_clone_url_without_token(self):
        target = RepoTarget(url="https://github.com/user/repo.git")
        assert target.clone_url == "https://github.com/user/repo.git"


class TestAnalyze:
    def test_analyze_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock repo structure
            (Path(tmpdir) / "main.py").write_text("print('hello')")
            (Path(tmpdir) / "tests").mkdir()
            (Path(tmpdir) / "tests" / "test_main.py").write_text("def test(): pass")
            (Path(tmpdir) / ".github" / "workflows").mkdir(parents=True)
            (Path(tmpdir) / "README.md").write_text("# My Project\nDescription here")

            manager = ExternalRepoManager()
            target = RepoTarget(url="https://example.com/repo.git")
            ctx = manager.analyze(target, Path(tmpdir))

            assert ctx.file_count >= 2
            assert ctx.has_tests
            assert ctx.has_ci
            assert "My Project" in ctx.readme_summary
            assert "main.py" in ctx.entry_points

    def test_analyze_language_breakdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                (Path(tmpdir) / f"file{i}.py").write_text(f"x = {i}")
            for i in range(3):
                (Path(tmpdir) / f"file{i}.js").write_text(f"const x = {i}")

            manager = ExternalRepoManager()
            target = RepoTarget(url="https://example.com/repo.git")
            ctx = manager.analyze(target, Path(tmpdir))

            assert ".py" in ctx.language_breakdown
            assert ctx.language_breakdown[".py"] == 5

    def test_analyze_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "src"
            sub.mkdir()
            (sub / "app.py").write_text("app = True")
            (Path(tmpdir) / "docs.md").write_text("docs")

            manager = ExternalRepoManager()
            target = RepoTarget(url="https://example.com/repo.git", subdirectory="src")
            ctx = manager.analyze(target, Path(tmpdir))

            # Only counts files in subdirectory
            assert ctx.file_count == 1


class TestRepoContext:
    def test_defaults(self):
        target = RepoTarget(url="https://example.com/repo.git")
        ctx = RepoContext(target=target, local_path=Path("/tmp/test"))
        assert ctx.file_count == 0
        assert not ctx.has_tests
        assert not ctx.has_ci
