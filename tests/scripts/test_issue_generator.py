from __future__ import annotations

from pathlib import Path

from scripts.issue_generator import IssueGenerator


def test_find_untested_modules_emits_module_aware_issue(tmp_path: Path) -> None:
    repo_root = tmp_path
    module_path = repo_root / "aragora" / "pkg" / "demo.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        '"""Demo service module for testing."""\n'
        "import requests\n\n"
        "class DemoClient:\n"
        "    pass\n\n"
        "def fetch_demo() -> str:\n"
        "    return 'ok'\n\n"
        + "\n".join(f"def extra_{idx}() -> int:\n    return {idx}\n" for idx in range(20)),
        encoding="utf-8",
    )
    (repo_root / "tests").mkdir()

    generator = IssueGenerator(repo_root)

    issues = generator._find_untested_modules()

    assert len(issues) == 1
    issue = issues[0]
    assert issue.title == "Add focused tests for aragora/pkg/demo.py"
    assert issue.file_hints == ["aragora/pkg/demo.py", "tests/pkg/test_demo.py"]
    assert issue.complexity in {"small", "medium", "large"}
    assert "Module purpose: Demo service module for testing." in issue.description
    assert "Public API candidates: fetch_demo()" in issue.description
    assert "Classes: DemoClient" in issue.description
    assert "Suggested test file: tests/pkg/test_demo.py" in issue.description
    assert "Mocking hints: Mock `requests`" in issue.description


def test_find_untested_modules_skips_small_and_already_tested_modules(tmp_path: Path) -> None:
    repo_root = tmp_path
    small_module = repo_root / "aragora" / "pkg" / "small.py"
    small_module.parent.mkdir(parents=True)
    small_module.write_text("def tiny() -> int:\n    return 1\n", encoding="utf-8")

    tested_module = repo_root / "scripts" / "already_tested.py"
    tested_module.parent.mkdir(parents=True)
    tested_module.write_text(
        '"""Already tested utility."""\n' + "\n".join("print('x')" for _ in range(60)),
        encoding="utf-8",
    )

    test_path = repo_root / "tests" / "scripts" / "test_already_tested.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")

    generator = IssueGenerator(repo_root)

    assert generator._find_untested_modules() == []
