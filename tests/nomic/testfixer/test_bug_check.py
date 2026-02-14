"""Tests for post-fix bug checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from aragora.nomic.testfixer.bug_check import BugCheckResult, PostFixBugChecker


@dataclass
class FakePatch:
    file_path: str = "src/module.py"
    original_content: str = "# original"
    patched_content: str = "# patched"


@dataclass
class FakeProposal:
    patches: list = field(default_factory=list)


@dataclass
class FakeBug:
    category: str = "logic_error"
    line_number: int = 10
    pattern_name: str = "unused_var"
    title: str = "Unused variable"
    severity: str = "low"


class TestBugCheckResult:
    def test_default_passes(self):
        result = BugCheckResult()
        assert result.passes is True

    def test_summary_with_new_bugs(self):
        result = BugCheckResult(
            new_bugs=[FakeBug()],
            resolved_bugs=[],
            passes=True,
            summary="1 new bug(s), 0 resolved bug(s)",
        )
        assert "1 new" in result.summary


class TestPostFixBugChecker:
    def test_passes_when_no_detector(self, tmp_path):
        checker = PostFixBugChecker(tmp_path, detector=None)
        # Force detector to None to simulate unavailable case
        checker._detector = None
        proposal = FakeProposal(patches=[FakePatch()])
        result = checker.check_patches(proposal)
        assert result.passes is True
        assert "unavailable" in result.summary

    def test_passes_when_no_new_bugs(self, tmp_path):
        detector = MagicMock()
        detector.detect_in_file.return_value = []

        # Create the target file
        target = tmp_path / "src" / "module.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# patched")

        checker = PostFixBugChecker(tmp_path, detector=detector)
        proposal = FakeProposal(patches=[FakePatch(file_path="src/module.py")])
        result = checker.check_patches(proposal)
        assert result.passes is True

    def test_fails_on_new_critical_bug(self, tmp_path):
        critical_bug = FakeBug(severity="critical", line_number=5, title="NPE")
        detector = MagicMock()
        # After scan finds a critical bug; before scan finds nothing
        detector.detect_in_file.side_effect = [
            [critical_bug],  # after (patched file)
            [],  # before (original content via temp file) — but we mock _scan_content
        ]

        target = tmp_path / "src" / "module.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# patched")

        checker = PostFixBugChecker(tmp_path, detector=detector)
        # Override _scan_content to return empty (no bugs in original)
        checker._scan_content = lambda content, path: []

        proposal = FakeProposal(patches=[FakePatch(file_path="src/module.py")])
        result = checker.check_patches(proposal)
        assert result.passes is False
        assert len(result.new_bugs) == 1

    def test_passes_when_bug_already_existed(self, tmp_path):
        existing_bug = FakeBug(severity="high", line_number=5, title="NPE")
        detector = MagicMock()
        detector.detect_in_file.return_value = [existing_bug]

        target = tmp_path / "src" / "module.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# patched")

        checker = PostFixBugChecker(tmp_path, detector=detector)
        # Both before and after have the same bug
        checker._scan_content = lambda content, path: [existing_bug]

        proposal = FakeProposal(patches=[FakePatch(file_path="src/module.py")])
        result = checker.check_patches(proposal)
        assert result.passes is True
        assert len(result.new_bugs) == 0

    def test_bug_key_stability(self, tmp_path):
        checker = PostFixBugChecker(tmp_path)
        bug = FakeBug(category="logic", line_number=10, pattern_name="p1", title="t1")
        key = checker._bug_key(bug)
        assert key == "logic:10:p1:t1"
