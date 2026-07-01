"""Unit tests for scripts/nomic_ci_test_selector.py infer_test_paths.

Covers the legacy root-path mapping (aragora/<x>.py -> tests/test_<x>.py)
AND the relocated-subdir probe via the pre-computed migration map
(tests/<module>/test_<x>.py) added by the misc-ci-test-selector-subdir-mapping
fix.  Also covers the _root-suffix variant for subdirectory files (batch-3
convention).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTOR_PATH = REPO_ROOT / "scripts" / "nomic_ci_test_selector.py"

_spec = importlib.util.spec_from_file_location("nomic_ci_test_selector", _SELECTOR_PATH)
selector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(selector)

infer_test_paths = selector.infer_test_paths
changed_python_files = selector.changed_python_files
_relocated_test_path = selector._relocated_test_path
_MIGRATED_TEST_MAP = selector._MIGRATED_TEST_MAP


def _patch_exists(monkeypatch, fake_paths):
    """Monkeypatch Path.exists for repo-relative fake paths, else delegate."""
    orig_exists = Path.exists

    def fake_exists(self):
        path_text = str(self)
        try:
            rel_text = str(self.relative_to(REPO_ROOT))
        except ValueError:
            rel_text = path_text
        if path_text in fake_paths or rel_text in fake_paths:
            return True
        return orig_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)


class TestRelocatedTestPath:
    """Tests for _relocated_test_path helper."""

    def test_known_migration_returns_new_path(self):
        """A mapped root test returns its new subdirectory home."""
        result = _relocated_test_path("tests/test_exceptions.py")
        assert result == "tests/agents/test_exceptions.py"

    def test_unknown_path_returns_none(self):
        """An unmapped root test returns None."""
        result = _relocated_test_path("tests/test_nonexistent.py")
        assert result is None

    def test_map_is_non_empty(self):
        """Every map entry is a well-formed root -> subdirectory relocation."""
        assert _MIGRATED_TEST_MAP
        for old, new in _MIGRATED_TEST_MAP.items():
            assert old.startswith("tests/test_")
            assert new.startswith("tests/")
            assert "/test_" in new

    def test_mapped_destinations_exist(self):
        """Every configured migrated target exists in the repo."""
        missing = [
            new_path
            for new_path in _MIGRATED_TEST_MAP.values()
            if not (REPO_ROOT / new_path).exists()
        ]
        assert missing == []


class TestInferTestPathsTopLevel:
    """Top-level aragora/<x>.py mapping: legacy root + migration-map probe."""

    def test_legacy_root_path_exists(self, monkeypatch):
        """When tests/test_<x>.py exists, it is selected (legacy behavior)."""
        _patch_exists(monkeypatch, {"tests/test_resilience_config.py"})

        result = infer_test_paths(["aragora/resilience_config.py"])
        assert "tests/test_resilience_config.py" in result

    def test_relocated_via_migration_map(self, monkeypatch):
        """When legacy root is missing but the migration map has it, the
        relocated path is selected."""
        # tests/test_exceptions.py was relocated to tests/agents/test_exceptions.py
        # Only the new path exists on disk
        _patch_exists(monkeypatch, {"tests/agents/test_exceptions.py"})

        result = infer_test_paths(["aragora/exceptions.py"])
        assert "tests/agents/test_exceptions.py" in result

    def test_both_legacy_and_map_entry_include_relocated(self, monkeypatch):
        """A legacy root stub must not hide the relocated mapped test."""
        _patch_exists(
            monkeypatch,
            {
                "tests/test_exceptions.py",
                "tests/agents/test_exceptions.py",
            },
        )

        result = infer_test_paths(["aragora/exceptions.py"])
        assert "tests/test_exceptions.py" in result
        assert "tests/agents/test_exceptions.py" in result

    def test_repo_root_anchoring_ignores_cwd(self, monkeypatch, tmp_path):
        """Path probes are anchored to the repository, not the caller cwd."""
        _patch_exists(monkeypatch, {"tests/agents/test_exceptions.py"})
        monkeypatch.chdir(tmp_path)

        result = infer_test_paths(["aragora/exceptions.py"])
        assert "tests/agents/test_exceptions.py" in result

    def test_no_test_found_returns_empty(self, monkeypatch):
        """When no test file exists anywhere, result is empty."""
        _patch_exists(monkeypatch, set())

        result = infer_test_paths(["aragora/nonexistent.py"])
        assert result == []


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

    def test_both_regular_and_root_variants(self, monkeypatch):
        """Both regular and _root suffix variants are found when both exist."""
        _patch_exists(
            monkeypatch,
            {
                "tests/connectors/test_twitter_poster.py",
                "tests/connectors/test_twitter_poster_root.py",
            },
        )

        result = infer_test_paths(["aragora/connectors/twitter_poster.py"])
        assert "tests/connectors/test_twitter_poster.py" in result
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

        result = infer_test_paths(
            [
                "aragora/debate/orchestrator.py",
                "aragora/debate/orchestrator.py",
            ]
        )
        assert result.count("tests/debate/test_orchestrator.py") == 1

    def test_tests_path_not_double_counted(self, monkeypatch):
        """A test path passed directly + mapped from source is deduplicated."""
        _patch_exists(monkeypatch, {"tests/debate/test_orchestrator.py"})

        result = infer_test_paths(
            [
                "tests/debate/test_orchestrator.py",
                "aragora/debate/orchestrator.py",
            ]
        )
        assert result.count("tests/debate/test_orchestrator.py") == 1


class TestChangedPythonFiles:
    """Tests for changed_python_files helper."""

    def test_filters_aragora_py_only(self):
        result = changed_python_files(
            [
                "aragora/foo.py",
                "tests/test_bar.py",
                "docs/README.md",
                "",
                "  ",
            ]
        )
        assert result == ["aragora/foo.py"]

    def test_excludes_non_py(self):
        result = changed_python_files(["aragora/config.yaml", "aragora/data.json"])
        assert result == []

    def test_excludes_non_aragora(self):
        result = changed_python_files(["scripts/util.py", "sdk/python/main.py"])
        assert result == []
