"""Tests for the batch pattern fixer."""

from __future__ import annotations

import textwrap

import pytest

from aragora.nomic.pattern_fixer import ANTIPATTERNS, PatternFixer, PatternMatch


@pytest.fixture
def tmp_codebase(tmp_path):
    """Create a temporary codebase with known antipatterns."""
    (tmp_path / "good.py").write_text(
        textwrap.dedent("""\
        import logging
        logger = logging.getLogger(__name__)

        def safe():
            try:
                do_something()
            except ValueError:
                logger.warning("oops")
        """)
    )
    (tmp_path / "bad_except.py").write_text(
        textwrap.dedent("""\
        def risky():
            try:
                do_something()
            except Exception: pass
        """)
    )
    (tmp_path / "bad_eval.py").write_text(
        textwrap.dedent("""\
        def dangerous(x):
            return eval(x)
        """)
    )
    (tmp_path / "str_leak.py").write_text(
        textwrap.dedent("""\
        def handler():
            try:
                work()
            except Exception as e:
                message = str(e)
                return {"error": message}
        """)
    )
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.py").write_text(
        textwrap.dedent("""\
        while True:
            data = read()
            if not data:
                break
        """)
    )
    return tmp_path


@pytest.fixture
def fixer(tmp_codebase) -> PatternFixer:
    return PatternFixer(str(tmp_codebase))


class TestFindAntipattern:
    def test_finds_bare_except(self, fixer: PatternFixer):
        matches = fixer.find_antipattern("bare_except")
        assert len(matches) >= 1
        assert any("bad_except.py" in m.file for m in matches)

    def test_finds_eval(self, fixer: PatternFixer):
        matches = fixer.find_antipattern("eval_usage")
        assert len(matches) >= 1
        assert any("bad_eval.py" in m.file for m in matches)

    def test_finds_str_e_leak(self, fixer: PatternFixer):
        matches = fixer.find_antipattern("str_e_leak")
        assert len(matches) >= 1
        assert any("str_leak.py" in m.file for m in matches)

    def test_finds_while_true(self, fixer: PatternFixer):
        matches = fixer.find_antipattern("while_true")
        assert len(matches) >= 1
        assert any("nested.py" in m.file for m in matches)

    def test_unknown_pattern_raises(self, fixer: PatternFixer):
        with pytest.raises(ValueError, match="Unknown antipattern"):
            fixer.find_antipattern("nonexistent")


class TestFindPattern:
    def test_custom_regex(self, fixer: PatternFixer):
        matches = fixer.find_pattern(r"def\s+\w+\(")
        assert len(matches) >= 3  # good.py, bad_except.py, bad_eval.py, str_leak.py

    def test_context_lines(self, fixer: PatternFixer):
        matches = fixer.find_pattern(r"eval\(")
        assert len(matches) >= 1
        m = matches[0]
        # context_before/after should have surrounding lines
        assert isinstance(m.context_before, list)
        assert isinstance(m.context_after, list)


class TestFixPattern:
    def test_fix_replaces_content(self, fixer: PatternFixer, tmp_codebase):
        matches = fixer.find_antipattern("eval_usage")
        result = fixer.fix_pattern(matches, "    return safe_eval(x)")
        assert result.matches_fixed >= 1
        assert len(result.files_changed) >= 1

        content = (tmp_codebase / "bad_eval.py").read_text()
        assert "safe_eval" in content

    def test_fix_with_callable(self, fixer: PatternFixer, tmp_codebase):
        matches = fixer.find_antipattern("bare_except")
        result = fixer.fix_pattern(
            matches,
            lambda m: m.content.replace(
                "except Exception: pass", "except Exception:\n        logger.warning('caught')"
            ),
        )
        assert result.matches_fixed >= 1

    def test_fix_nonexistent_file(self, fixer: PatternFixer):
        fake = PatternMatch(file="/nonexistent/path.py", line=1, content="x")
        result = fixer.fix_pattern([fake], "replacement")
        assert result.matches_fixed == 0
        assert len(result.errors) >= 1


class TestListAndCount:
    def test_list_antipatterns(self, fixer: PatternFixer):
        patterns = fixer.list_antipatterns()
        assert "bare_except" in patterns
        assert "eval_usage" in patterns
        assert len(patterns) == len(ANTIPATTERNS)

    def test_count_antipatterns(self, fixer: PatternFixer):
        counts = fixer.count_antipatterns()
        assert counts["bare_except"] >= 1
        assert counts["eval_usage"] >= 1
        assert isinstance(counts["shell_true"], int)


class TestReport:
    def test_generate_report(self, fixer: PatternFixer):
        report = fixer.generate_report()
        assert "# Antipattern Report" in report
        assert "bare_except" in report
        assert "eval_usage" in report
        assert "Total antipatterns found:" in report


class TestEdgeCases:
    def test_empty_codebase(self, tmp_path):
        fixer = PatternFixer(str(tmp_path))
        assert fixer.find_pattern(r"anything") == []

    def test_binary_file_skipped(self, tmp_path):
        (tmp_path / "binary.py").write_bytes(b"\x00\x01\x02eval(\x03")
        fixer = PatternFixer(str(tmp_path))
        # Should not crash; may or may not find match in binary content
        fixer.find_pattern(r"eval\(")
