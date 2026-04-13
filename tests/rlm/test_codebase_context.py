"""Unit tests for aragora.rlm.codebase_context.CodebaseContextBuilder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "ARAGORA_CODEBASE_MAX_CONTEXT_BYTES",
        "ARAGORA_NOMIC_MAX_CONTEXT_BYTES",
        "NOMIC_MAX_CONTEXT_BYTES",
        "ARAGORA_RLM_MAX_CONTEXT_BYTES",
        "ARAGORA_RLM_MAX_CONTENT_BYTES",
        "ARAGORA_CODEBASE_INCLUDE_TESTS",
        "ARAGORA_CODEBASE_RLM_FULL_CORPUS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def root_path(tmp_path):
    nomic_dir = tmp_path / ".nomic" / "context"
    nomic_dir.mkdir(parents=True)
    return tmp_path


class TestCodebaseContextBuilderInit:
    def test_defaults(self, root_path):
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            builder = CodebaseContextBuilder(root_path=root_path)
            mock_init.assert_called_once_with(
                aragora_path=root_path,
                max_context_bytes=0,
                include_tests=True,
                knowledge_mound=None,
                full_corpus=False,
            )

    def test_explicit_params_override_env(self, root_path, monkeypatch):
        monkeypatch.setenv("ARAGORA_CODEBASE_MAX_CONTEXT_BYTES", "999")
        monkeypatch.setenv("ARAGORA_CODEBASE_INCLUDE_TESTS", "0")
        monkeypatch.setenv("ARAGORA_CODEBASE_RLM_FULL_CORPUS", "1")
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            CodebaseContextBuilder(
                root_path=root_path,
                max_context_bytes=500,
                include_tests=True,
                full_corpus=False,
            )
            mock_init.assert_called_once_with(
                aragora_path=root_path,
                max_context_bytes=500,
                include_tests=True,
                knowledge_mound=None,
                full_corpus=False,
            )

    @pytest.mark.parametrize(
        "env_var",
        [
            "ARAGORA_CODEBASE_MAX_CONTEXT_BYTES",
            "ARAGORA_NOMIC_MAX_CONTEXT_BYTES",
            "NOMIC_MAX_CONTEXT_BYTES",
            "ARAGORA_RLM_MAX_CONTEXT_BYTES",
            "ARAGORA_RLM_MAX_CONTENT_BYTES",
        ],
    )
    def test_max_context_from_env(self, root_path, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "42000")
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            CodebaseContextBuilder(root_path=root_path)
            assert mock_init.call_args.kwargs["max_context_bytes"] == 42000

    def test_env_priority_order(self, root_path, monkeypatch):
        monkeypatch.setenv("ARAGORA_CODEBASE_MAX_CONTEXT_BYTES", "100")
        monkeypatch.setenv("NOMIC_MAX_CONTEXT_BYTES", "200")
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            CodebaseContextBuilder(root_path=root_path)
            assert mock_init.call_args.kwargs["max_context_bytes"] == 100

    def test_invalid_env_max_falls_back_to_zero(self, root_path, monkeypatch):
        monkeypatch.setenv("ARAGORA_CODEBASE_MAX_CONTEXT_BYTES", "not_a_number")
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            CodebaseContextBuilder(root_path=root_path)
            assert mock_init.call_args.kwargs["max_context_bytes"] == 0

    def test_include_tests_from_env(self, root_path, monkeypatch):
        monkeypatch.setenv("ARAGORA_CODEBASE_INCLUDE_TESTS", "0")
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            CodebaseContextBuilder(root_path=root_path)
            assert mock_init.call_args.kwargs["include_tests"] is False

    def test_full_corpus_from_env(self, root_path, monkeypatch):
        monkeypatch.setenv("ARAGORA_CODEBASE_RLM_FULL_CORPUS", "1")
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            CodebaseContextBuilder(root_path=root_path)
            assert mock_init.call_args.kwargs["full_corpus"] is True

    def test_knowledge_mound_passed_through(self, root_path):
        sentinel = object()
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            CodebaseContextBuilder(root_path=root_path, knowledge_mound=sentinel)
            assert mock_init.call_args.kwargs["knowledge_mound"] is sentinel

    def test_nonzero_max_context_skips_env(self, root_path, monkeypatch):
        monkeypatch.setenv("ARAGORA_CODEBASE_MAX_CONTEXT_BYTES", "999")
        with patch(
            "aragora.rlm.codebase_context.NomicContextBuilder.__init__", return_value=None
        ) as mock_init:
            from aragora.rlm.codebase_context import CodebaseContextBuilder

            CodebaseContextBuilder(root_path=root_path, max_context_bytes=123)
            assert mock_init.call_args.kwargs["max_context_bytes"] == 123


class TestModuleExports:
    def test_all_exports(self):
        from aragora.rlm import codebase_context

        assert codebase_context.__all__ == ["CodebaseContextBuilder"]

    def test_subclass_of_nomic_context_builder(self):
        from aragora.nomic.context_builder import NomicContextBuilder
        from aragora.rlm.codebase_context import CodebaseContextBuilder

        assert issubclass(CodebaseContextBuilder, NomicContextBuilder)
