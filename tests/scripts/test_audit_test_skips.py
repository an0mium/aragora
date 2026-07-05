"""Tests for scripts/audit_test_skips.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts import audit_test_skips  # noqa: E402


def test_emit_output_mutes_stdout_after_broken_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    muted = []

    class BrokenStdout:
        def write(self, _output: str) -> int:
            raise BrokenPipeError

        def flush(self) -> None:
            raise AssertionError("flush should not run after a broken write")

    monkeypatch.setattr(audit_test_skips.sys, "stdout", BrokenStdout())
    monkeypatch.setattr(
        audit_test_skips,
        "_mute_stdout_after_broken_pipe",
        lambda: muted.append(True),
    )

    audit_test_skips._emit_output("payload")

    assert muted == [True]


def test_main_json_uses_pipe_safe_emitter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    emitted = []
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_test_skips.py", "--json", "--tests-dir", str(tests_dir)],
    )
    monkeypatch.setattr(audit_test_skips, "audit_skips", lambda _tests_dir: [])
    monkeypatch.setattr(
        audit_test_skips,
        "generate_report",
        lambda _markers: {
            "total": 0,
            "justified_total": 0,
            "unjustified_total": 0,
            "by_category": {},
            "by_unjustified_category": {},
            "by_justification_category": {},
            "by_type": {},
            "by_file": {},
            "high_skip_files": [],
            "markers": [],
            "generated_at": "2026-06-13T00:00:00",
        },
    )
    monkeypatch.setattr(audit_test_skips, "_emit_output", emitted.append)

    audit_test_skips.main()

    assert json.loads(emitted[0])["total"] == 0


def test_main_default_summary_uses_pipe_safe_emitter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    emitted = []
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_test_skips.py", "--tests-dir", str(tests_dir)],
    )
    monkeypatch.setattr(audit_test_skips, "audit_skips", lambda _tests_dir: [])
    monkeypatch.setattr(
        audit_test_skips,
        "generate_report",
        lambda _markers: {
            "total": 2,
            "justified_total": 1,
            "unjustified_total": 1,
            "by_category": {"known_bug": 1, "optional_dependency": 1},
            "by_unjustified_category": {"known_bug": 1},
            "by_justification_category": {"optional_dependency": 1},
            "by_type": {"skip": 2},
            "by_file": {"tests/test_example.py": 2},
            "high_skip_files": [{"file": "tests/test_example.py", "count": 2}],
            "markers": [],
            "generated_at": "2026-06-13T00:00:00",
        },
    )
    monkeypatch.setattr(audit_test_skips, "_emit_output", emitted.append)

    audit_test_skips.main()

    assert "Total skip markers: 2" in emitted[0]
    assert "Justified skip markers: 1" in emitted[0]
    assert "Unjustified skip markers: 1" in emitted[0]
    assert "known_bug: 1" in emitted[0]
    assert "optional_dependency: 1" in emitted[0]
    assert "tests/test_example.py: 2" in emitted[0]


def test_generate_report_splits_justified_and_unjustified(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_example.py"
    test_file.write_text(
        "\n".join(
            [
                "import pytest",
                "",
                "@pytest.mark.skip(",
                '    reason="justified-skip[optional_dependency]: package not installed"',
                ")",
                "def test_optional_dep():",
                "    pass",
                "",
                '@pytest.mark.skip(reason="Known bug: GH-123")',
                "def test_known_bug():",
                "    pass",
                "",
                "def test_runtime_skip():",
                '    pytest.skip("justified_skip[platform_specific]: symlink unavailable")',
            ]
        ),
        encoding="utf-8",
    )

    report = audit_test_skips.generate_report(audit_test_skips.audit_skips(tests_dir))

    assert report["total"] == 3
    assert report["justified_total"] == 2
    assert report["unjustified_total"] == 1
    assert report["by_justification_category"] == {
        "optional_dependency": 1,
        "platform_specific": 1,
    }
    assert report["by_unjustified_category"] == {"known_bug": 1}

    justified = [marker for marker in report["markers"] if marker["justified"]]
    assert justified[0]["justification_rationale"] == "package not installed"


def test_main_unjustified_count_only_uses_report_metric(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    emitted = []
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_test_skips.py", "--unjustified-count-only", "--tests-dir", str(tests_dir)],
    )
    monkeypatch.setattr(audit_test_skips, "audit_skips", lambda _tests_dir: [])
    monkeypatch.setattr(
        audit_test_skips,
        "generate_report",
        lambda _markers: {
            "total": 9,
            "justified_total": 7,
            "unjustified_total": 2,
            "by_category": {},
            "by_unjustified_category": {},
            "by_justification_category": {},
            "by_type": {},
            "by_file": {},
            "high_skip_files": [],
            "markers": [],
            "generated_at": "2026-06-13T00:00:00",
        },
    )
    monkeypatch.setattr(audit_test_skips, "_emit_output", emitted.append)

    audit_test_skips.main()

    assert emitted == ["2"]
