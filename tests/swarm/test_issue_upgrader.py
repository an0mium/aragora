"""Tests for the narrow heuristic issue upgrader."""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.swarm.issue_upgrader import UpgradedIssue, upgrade_issue_heuristic


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    (tmp_path / "aragora" / "swarm").mkdir(parents=True)
    return tmp_path


def _write_module(repo_root: Path, rel: str, content: str) -> Path:
    path = repo_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _upgrade(repo_root: Path, rel: str) -> UpgradedIssue | None:
    body = f"Focus on `{rel}` with a mirrored focused test file."
    return upgrade_issue_heuristic(f"Add tests for {Path(rel).name}", body, repo_root=repo_root)


class TestUpgradeAdmission:
    def test_upgrades_trivial_public_function_module(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/trivial_module.py",
            '"""Small helper module."""\n\n'
            "def normalize_name(value: str) -> str:\n"
            "    return value.strip().lower()\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/trivial_module.py")

        assert result is not None
        assert result.complexity == "trivial"
        assert result.upgrade_method == "heuristic"
        assert result.functions_found == ["normalize_name"]

    def test_upgrades_simple_class_module(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/simple_module.py",
            '"""Simple state holder."""\n\n'
            "class IssueState:\n"
            "    def status(self) -> str:\n"
            "        return 'ok'\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/simple_module.py")

        assert result is not None
        assert result.complexity in {"trivial", "simple"}
        assert "IssueState" in result.upgraded_body

    def test_skips_medium_module(self, repo_root: Path) -> None:
        body_lines = ['"""Module with too much surface."""', ""]
        for idx in range(10):
            body_lines.extend(
                [
                    f"def public_fn_{idx}(value: int) -> int:",
                    f"    return value + {idx}",
                    "",
                ]
            )
        _write_module(repo_root, "aragora/swarm/medium_module.py", "\n".join(body_lines))

        assert _upgrade(repo_root, "aragora/swarm/medium_module.py") is None

    def test_skips_complex_module(self, repo_root: Path) -> None:
        body_lines = ['"""Complex module."""', ""]
        for idx in range(18):
            body_lines.extend(
                [
                    f"def public_fn_{idx}(value: int) -> int:",
                    f"    return value + {idx}",
                    "",
                ]
            )
        _write_module(repo_root, "aragora/swarm/complex_module.py", "\n".join(body_lines))

        assert _upgrade(repo_root, "aragora/swarm/complex_module.py") is None

    def test_skips_private_only_module(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/private_only.py",
            "def _helper() -> str:\n    return 'x'\n",
        )

        assert _upgrade(repo_root, "aragora/swarm/private_only.py") is None

    def test_skips_constant_only_module(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/constants_only.py",
            '"""Constants only."""\n\nDEFAULT_TIMEOUT = 5\n',
        )

        assert _upgrade(repo_root, "aragora/swarm/constants_only.py") is None

    def test_skips_reexport_module(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/reexport_only.py",
            "from aragora.swarm.simple_module import IssueState\n\n__all__ = ['IssueState']\n",
        )

        assert _upgrade(repo_root, "aragora/swarm/reexport_only.py") is None

    def test_skips_empty_module(self, repo_root: Path) -> None:
        _write_module(repo_root, "aragora/swarm/empty_module.py", "")

        assert _upgrade(repo_root, "aragora/swarm/empty_module.py") is None

    def test_skips_init_modules(self, repo_root: Path) -> None:
        _write_module(repo_root, "aragora/swarm/__init__.py", "from .x import y\n")

        assert _upgrade(repo_root, "aragora/swarm/__init__.py") is None

    def test_skips_when_body_has_no_module_reference(self, repo_root: Path) -> None:
        result = upgrade_issue_heuristic(
            "No path", "This issue body names no module.", repo_root=repo_root
        )
        assert result is None

    def test_skips_when_module_is_missing(self, repo_root: Path) -> None:
        result = _upgrade(repo_root, "aragora/swarm/missing_module.py")
        assert result is None


class TestAsyncHandling:
    def test_async_module_is_upgraded_with_async_note(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/async_module.py",
            '"""Async helper."""\n\nasync def fetch_status() -> str:\n    return \'ready\'\n',
        )

        result = _upgrade(repo_root, "aragora/swarm/async_module.py")

        assert result is not None
        assert "pytest.mark.asyncio" in result.upgraded_body
        assert "fetch_status()" in result.upgraded_body

    def test_async_class_method_is_handled_sensibly(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/async_class_module.py",
            '"""Async class helper."""\n\n'
            "class AsyncRunner:\n"
            "    async def run(self) -> str:\n"
            "        return 'done'\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/async_class_module.py")

        assert result is not None
        assert "AsyncRunner" in result.upgraded_body
        assert "pytest.mark.asyncio" in result.upgraded_body


class TestDependencyWeight:
    def test_low_external_dependency_weight_with_concrete_guidance_is_allowed(
        self,
        repo_root: Path,
    ) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/http_client.py",
            '"""HTTP helper."""\n\nimport httpx\n\ndef fetch() -> str:\n    return \'ok\'\n',
        )

        result = _upgrade(repo_root, "aragora/swarm/http_client.py")

        assert result is not None
        assert "Patch `httpx` clients" in result.upgraded_body
        assert result.imports == ["httpx"]

    def test_high_external_dependency_weight_is_skipped(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/heavy_external.py",
            '"""Heavy external module."""\n\n'
            "import httpx\n"
            "import requests\n"
            "import boto3\n\n"
            "def fetch_all() -> str:\n"
            "    return 'ok'\n",
        )

        assert _upgrade(repo_root, "aragora/swarm/heavy_external.py") is None

    def test_unknown_external_dependency_is_skipped(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/unknown_external.py",
            '"""Unknown external dep."""\n\n'
            "import vendorlib\n\n"
            "def run() -> str:\n"
            "    return 'ok'\n",
        )

        assert _upgrade(repo_root, "aragora/swarm/unknown_external.py") is None

    def test_local_imports_do_not_count_as_external_weight(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/local_imports.py",
            '"""Local helper."""\n\n'
            "from aragora.swarm.spec import SwarmSpec\n\n"
            "def build() -> str:\n"
            "    return SwarmSpec.__name__\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/local_imports.py")

        assert result is not None
        assert result.imports == []


class TestOutputShape:
    def test_generated_test_path_is_correct_for_nested_module(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/nested/example_module.py",
            "def helper() -> str:\n    return 'x'\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/nested/example_module.py")

        assert result is not None
        assert "`tests/swarm/nested/test_example_module.py` (create)" in result.upgraded_body

    def test_upgraded_body_contains_concrete_file_scope_and_validation(
        self, repo_root: Path
    ) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/output_shape.py",
            "def helper() -> str:\n    return 'x'\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/output_shape.py")

        assert result is not None
        assert "### File Scope" in result.upgraded_body
        assert "### Validation" in result.upgraded_body
        assert "pytest tests/swarm/test_output_shape.py -q" in result.upgraded_body

    def test_upgraded_body_includes_public_api_section(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/public_api.py",
            "def alpha() -> str:\n    return 'a'\n\ndef beta() -> str:\n    return 'b'\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/public_api.py")

        assert result is not None
        assert "`alpha()`" in result.upgraded_body
        assert "`beta()`" in result.upgraded_body

    def test_module_docstring_is_used_in_summary(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/with_docstring.py",
            '"""Normalize worker-ready issue bodies into bounded specs."""\n\n'
            "def normalize() -> str:\n"
            "    return 'ok'\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/with_docstring.py")

        assert result is not None
        assert "Normalize worker-ready issue bodies" in result.module_summary
        assert "**Module purpose:** Normalize worker-ready issue bodies" in result.upgraded_body

    def test_output_is_stable_for_same_module(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/stable_module.py",
            "def normalize() -> str:\n    return 'ok'\n",
        )

        first = _upgrade(repo_root, "aragora/swarm/stable_module.py")
        second = _upgrade(repo_root, "aragora/swarm/stable_module.py")

        assert first == second

    def test_title_is_rewritten_to_nested_module_path(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/nested/title_module.py",
            "def normalize() -> str:\n    return 'ok'\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/nested/title_module.py")

        assert result is not None
        assert result.upgraded_title == "Add unit tests for swarm/nested/title_module.py"

    def test_simple_module_reports_simple_complexity(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/simple_complexity.py",
            "\n".join(
                ['"""Simple but not trivial."""', ""]
                + [
                    f"def fn_{idx}(value: int) -> int:\n    return value + {idx}\n"
                    for idx in range(5)
                ]
            ),
        )

        result = _upgrade(repo_root, "aragora/swarm/simple_complexity.py")

        assert result is not None
        assert result.complexity == "simple"

    def test_class_only_public_api_is_supported(self, repo_root: Path) -> None:
        _write_module(
            repo_root,
            "aragora/swarm/class_only.py",
            "class PublicTool:\n    def run(self) -> str:\n        return 'ok'\n",
        )

        result = _upgrade(repo_root, "aragora/swarm/class_only.py")

        assert result is not None
        assert "PublicTool" in result.upgraded_body
