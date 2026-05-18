"""Tests for the canonical-metrics test-definitions counter.

These tests pin the regex used by
``scripts/check_canonical_metrics.py::_observe_test_definitions_count``
to count both sync and async test functions. Earlier versions of the
counter used a sync-only regex (``^\\s*def test_``) which missed
``async def test_`` entries and triggered a false-positive "stale docs" drift on
``canonical.test_definitions.count``.

Coverage targets:
  - sync ``def test_`` is counted
  - ``async def test_`` is counted
  - non-test definitions (``def helper_``, ``def Test_*``) are excluded
  - multi-file aggregation works
  - missing tests/ directory returns 0
  - method matches the regex documented in ``docs/METRICS.md``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "check_canonical_metrics.py"
    spec = importlib.util.spec_from_file_location("check_canonical_metrics_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ccm = _load_module()


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Repoint the module's REPO_ROOT at a tmp dir for isolated counting."""

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    monkeypatch.setattr(ccm, "REPO_ROOT", tmp_path)
    return tmp_path


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


class TestSyncTestsCounted:
    def test_module_level_sync_def(self, fake_repo: Path) -> None:
        _write(
            fake_repo / "tests" / "test_a.py",
            "def test_alpha():\n    pass\n",
        )
        assert ccm._observe_test_definitions_count() == 1

    def test_class_nested_sync_def(self, fake_repo: Path) -> None:
        _write(
            fake_repo / "tests" / "test_b.py",
            "class TestThing:\n"
            "    def test_one(self):\n        pass\n"
            "    def test_two(self):\n        pass\n",
        )
        assert ccm._observe_test_definitions_count() == 2


class TestAsyncTestsCounted:
    def test_module_level_async_def(self, fake_repo: Path) -> None:
        _write(
            fake_repo / "tests" / "test_c.py",
            "async def test_async_alpha():\n    pass\n",
        )
        assert ccm._observe_test_definitions_count() == 1

    def test_class_nested_async_def(self, fake_repo: Path) -> None:
        _write(
            fake_repo / "tests" / "test_d.py",
            "class TestAsync:\n"
            "    async def test_one(self):\n        pass\n"
            "    async def test_two(self):\n        pass\n",
        )
        assert ccm._observe_test_definitions_count() == 2

    def test_mixed_sync_and_async_in_one_file(self, fake_repo: Path) -> None:
        _write(
            fake_repo / "tests" / "test_e.py",
            "def test_sync_one():\n    pass\n"
            "async def test_async_one():\n    pass\n"
            "def test_sync_two():\n    pass\n",
        )
        assert ccm._observe_test_definitions_count() == 3


class TestNonTestsExcluded:
    def test_helper_functions_not_counted(self, fake_repo: Path) -> None:
        _write(
            fake_repo / "tests" / "test_f.py",
            "def helper_alpha():\n    pass\n"
            "def setup_module():\n    pass\n"
            "def test_real():\n    pass\n",
        )
        assert ccm._observe_test_definitions_count() == 1

    def test_capitalized_test_class_constructor_not_counted(self, fake_repo: Path) -> None:
        # `def Test_*` (capital T) is not pytest-discovered; the counter
        # must require lowercase 'test_'.
        _write(
            fake_repo / "tests" / "test_g.py",
            "def Test_Alpha():\n    pass\ndef test_alpha():\n    pass\n",
        )
        assert ccm._observe_test_definitions_count() == 1

    def test_test_string_in_body_not_counted(self, fake_repo: Path) -> None:
        # A function body that mentions 'def test_' as a string literal
        # must not be miscounted.
        _write(
            fake_repo / "tests" / "test_h.py",
            'def test_alpha():\n    return "def test_fake"\n',
        )
        assert ccm._observe_test_definitions_count() == 1


class TestAggregation:
    def test_multiple_files_summed(self, fake_repo: Path) -> None:
        _write(fake_repo / "tests" / "a" / "test_one.py", "def test_a():\n    pass\n")
        _write(fake_repo / "tests" / "b" / "test_two.py", "async def test_b():\n    pass\n")
        _write(fake_repo / "tests" / "c" / "test_three.py", "def test_c():\n    pass\n")
        assert ccm._observe_test_definitions_count() == 3

    def test_non_python_files_ignored(self, fake_repo: Path) -> None:
        _write(fake_repo / "tests" / "test_real.py", "def test_alpha():\n    pass\n")
        _write(fake_repo / "tests" / "fixture.txt", "def test_fake():\n    pass\n")
        _write(fake_repo / "tests" / "data.json", '{"def test_fake": 1}')
        assert ccm._observe_test_definitions_count() == 1


class TestEdgeCases:
    def test_missing_tests_dir_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # No tests/ subdir at all.
        monkeypatch.setattr(ccm, "REPO_ROOT", tmp_path)
        assert ccm._observe_test_definitions_count() == 0

    def test_empty_tests_dir_returns_zero(self, fake_repo: Path) -> None:
        # tests/ exists but no .py files.
        (fake_repo / "tests" / "README.md").write_text("hi")
        assert ccm._observe_test_definitions_count() == 0

    def test_unreadable_file_skipped_gracefully(
        self, fake_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate a UnicodeDecodeError by writing binary garbage.
        bad = fake_repo / "tests" / "test_binary.py"
        bad.write_bytes(b"\xff\xfe\x00\x01not valid utf-8")
        good = fake_repo / "tests" / "test_good.py"
        good.write_text("def test_alpha():\n    pass\n")
        # The counter must not raise; the binary file is skipped silently.
        assert ccm._observe_test_definitions_count() == 1


class TestDocumentedMethodAlignment:
    """Pins the counter's behavior to what METRICS.md documents.

    METRICS.md row for "Test functions (class + module level)" promises:
        git grep -E '^[[:space:]]*(async )?def test_' -- tests | wc -l

    Counter must mirror that regex semantics. Same input, same count.
    """

    def test_regex_matches_documented_git_grep(self, fake_repo: Path) -> None:
        # Examples drawn from the documented grep pattern + edge cases.
        body = (
            "def test_a():\n    pass\n"
            "    def test_b(self):\n        pass\n"
            "async def test_c():\n    pass\n"
            "    async def test_d(self):\n        pass\n"
            "\tasync def test_e(self):\n\t\tpass\n"
            "async\tdef test_tab_between_async_and_def():\n    pass\n"
            "async  def test_double_space_between_async_and_def():\n    pass\n"
            "def not_a_test():\n    pass\n"
            "async def helper():\n    pass\n"
        )
        _write(fake_repo / "tests" / "test_regex.py", body)
        # Expected: 5 (a, b, c, d, e); non-literal-space async forms,
        # not_a_test, and helper are excluded to match the documented grep.
        assert ccm._observe_test_definitions_count() == 5
