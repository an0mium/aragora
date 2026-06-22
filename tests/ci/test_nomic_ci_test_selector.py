"""Unit tests for scripts/nomic_ci_test_selector.py infer_test_paths.

Covers the legacy root-path mapping (aragora/<x>.py -> tests/test_<x>.py)
AND the new relocated-subdir probes (tests/<module>/test_<x>.py and the
_root-suffixed variant) added by the misc-ci-test-selector-subdir-mapping fix.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTOR_PATH = REPO_ROOT / "scripts" / "nomic_ci_test_selector.py"

_spec = importlib.util.spec_from_file_location(
    "nomic_ci_test_selector", _SELECTOR_PATH
)
selector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(selector)

infer_test_paths = selector.infer_test_paths
changed_python_files = selector.changed_python_files


def _patch_exists(monkeypatch, fake_paths):
    """Monkeypatch Path.exists to return True for paths in fake_paths, else delegate."""
    orig_exists = Path.exists

    def fake_exists(self):
        if str(self) in fake_paths:
            return True
        return orig_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)


class TestInferTestPathsTopLevel:
    """Top-level aragora/<x>.py mapping: legacy root + relocated subdir probes."""

    def test_legacy_root_path_exists(self, monkeypatch):
        """When tests/test_<x>.py exists, it is selected (legacy behavior)."""
        _patch_exists(monkeypatch, {"tests/test_resilience_config.py"})

        result = infer_test_paths(["aragora/resilience_config.py"])
        assert "tests/test_resilience_config.py" in result

    def test_relocated_subdir_path_found(self, monkeypatch):
        """When the root test is missing but a subdir test exists, find it."""
        _patch_exists(monkeypatch, {"tests/resilience/test_http_client.py"})

        result = infer_test_paths(["aragora/http_client.py"])
        assert "tests/resilience/test_http_client.py" in result

    def test_root_suffix_variant_detected(self, monkeypatch):
        """When only the _root-suffixed variant exists, it is found."""
        _patch_exists(monkeypatch, {"tests/memory/test_continuum_root.py"})

        result = infer_test_paths(["aragora/continuum.py"])
        assert "tests/memory/test_continuum_root.py" in result

    def test_both_legacy_and_relocated_found(self, monkeypatch):
        """When both legacy root AND relocated subdir tests exist, both are selected."""
        _patch_exists(
            monkeypatch,
            {
                "tests/test_resilience_config.py",
                "tests/resilience/test_resilience_config.py",
            },
        )

        result = infer_test_paths(["aragora/resilience_config.py"])
        assert "tests/test_resilience_config.py" in result
        assert "tests/resilience/test_resilience_config.py" in result

    def test_no_test_found_returns_empty(self, monkeypatch):
        """When no test file exists anywhere, result is empty."""
        _patch_exists(monkeypatch, set())

        result = infer_test_paths(["aragora/nonexistent.py"])
        assert result == []

    def test_multiple_top_level_files(self, monkeypatch):
        """Multiple top-level changed files each resolve to their tests."""
        _patch_exists(
            monkeypatch,
            {
                "tests/test_resilience_config.py",
                "tests/resilience/test_http_client.py",
                "tests/memory/test_continuum_root.py",
            },
        )

        result = infer_test_paths(
            [
                "aragora/resilience_config.py",
                "aragora/http_client.py",
                "aragora/continuum.py",
            ]
        )
        assert "tests/test_resilience_config.py" in result
        assert "tests/resilience/test_http_client.py" in result
        assert "tests/memory/test_continuum_root.py" in result


class TestInferTestPathsSubdirectory:
    """Subdirectory aragora/<module>/<file>.py mapping (existing behavior, unchanged)."""

    def test_subdirectory_mapping(self, monkeypatch):
        """aragora/<module>/<file>.py maps to tests/<module>/test_<file>.py."""
        _patch_exists(monkeypatch, {"tests/debate/test_orchestrator.py"})

        result = infer_test_paths(["aragora/debate/orchestrator.py"])
        assert "tests/debate/test_orchestrator.py" in result

    def test_subdirectory_no_test_exists(self, monkeypatch):
        """When the subdirectory test doesn't exist, nothing is returned."""
        _patch_exists(monkeypatch, set())

        result = infer_test_paths(["aragora/debate/nonexistent.py"])
        assert result == []

    def test_subdirectory_with_root_suffix(self, monkeypatch):
        """aragora/<module>/<file>.py also probes _root variant for that module."""
        _patch_exists(monkeypatch, {"tests/connectors/test_twitter_poster_root.py"})

        result = infer_test_paths(["aragora/connectors/twitter_poster.py"])
        assert "tests/connectors/test_twitter_poster_root.py" in result


class TestInferTestPathsEdgeCases:
    """Edge cases: non-.py, tests/ passthrough, empty paths, dedup."""

    def test_tests_path_passthrough(self, monkeypatch):
        """Files under tests/ are passed through unchanged."""
        _patch_exists(monkeypatch, {"tests/debate/test_orchestrator.py"})

        result = infer_test_paths(["tests/debate/test_orchestrator.py"])
        assert result == ["tests/debate/test_orchestrator.py"]

    def test_non_py_source_ignored(self, monkeypatch):
        """Non-.py files under aragora/ are not mapped to test paths."""
        _patch_exists(monkeypatch, set())

        result = infer_test_paths(["aragora/config/settings.yaml"])
        assert result == []

    def test_empty_blank_paths_skipped(self, monkeypatch):
        """Empty or whitespace-only paths are skipped."""
        _patch_exists(monkeypatch, set())

        result = infer_test_paths(["", "  ", "\t"])
        assert result == []

    def test_deduplication(self, monkeypatch):
        """Duplicate test paths are deduplicated."""
        _patch_exists(monkeypatch, {"tests/debate/test_orchestrator.py"})

        # Both source files map to the same test
        result = infer_test_paths(
            ["aragora/debate/orchestrator.py", "aragora/debate/orchestrator.py"]
        )
        assert result.count("tests/debate/test_orchestrator.py") == 1

    def test_tests_path_not_double_counted(self, monkeypatch):
        """A test path passed directly + mapped from source is deduplicated."""
        _patch_exists(monkeypatch, {"tests/debate/test_orchestrator.py"})

        result = infer_test_paths(
            ["tests/debate/test_orchestrator.py", "aragora/debate/orchestrator.py"]
        )
        assert result.count("tests/debate/test_orchestrator.py") == 1


class TestChangedPythonFiles:
    """Tests for changed_python_files helper."""

    def test_filters_aragora_py_only(self):
        result = changed_python_files(
            ["aragora/foo.py", "tests/test_bar.py", "docs/README.md", "", "  "]
        )
        assert result == ["aragora/foo.py"]

    def test_excludes_non_py(self):
        result = changed_python_files(["aragora/config.yaml", "aragora/data.json"])
        assert result == []

    def test_excludes_non_aragora(self):
        result = changed_python_files(["scripts/util.py", "sdk/python/main.py"])
        assert result == []
