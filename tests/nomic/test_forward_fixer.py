"""Tests for the forward-fix diagnostic and repair system."""

from __future__ import annotations

import textwrap

import pytest

from aragora.nomic.forward_fixer import (
    DiagnosisResult,
    FailureType,
    ForwardFix,
    ForwardFixer,
)


@pytest.fixture
def fixer() -> ForwardFixer:
    return ForwardFixer(max_attempts=3, min_confidence=0.7)


PYTEST_IMPORT_ERROR = textwrap.dedent("""\
    ============================= ERRORS =============================
    FAILED tests/sdk/test_client.py::test_init - ImportError: cannot import name 'Client' from 'aragora.sdk'
    E   ImportError: cannot import name 'Client' from 'aragora.sdk'
    =========================== short test summary info ===========================
    FAILED tests/sdk/test_client.py::test_init
    1 failed
""")

PYTEST_ASSERTION_ERROR = textwrap.dedent("""\
    FAILED tests/handlers/test_auth.py::test_login - AssertionError: assert 401 == 200
    E   AssertionError: assert 401 == 200
    =========================== short test summary info ===========================
    FAILED tests/handlers/test_auth.py::test_login
    1 failed
""")

PYTEST_ATTRIBUTE_ERROR = textwrap.dedent("""\
    FAILED tests/debate/test_arena.py::test_run - AttributeError: 'Arena' object has no attribute 'execute'
    E   AttributeError: 'Arena' object has no attribute 'execute'
    =========================== short test summary info ===========================
    FAILED tests/debate/test_arena.py::test_run
    1 failed
""")

PYTEST_TIMEOUT = textwrap.dedent("""\
    FAILED tests/connectors/test_slack.py::test_send - TimeoutError: timed out
    E   TimeoutError: timed out
    =========================== short test summary info ===========================
    FAILED tests/connectors/test_slack.py::test_send
    1 failed
""")

PYTEST_UNKNOWN = textwrap.dedent("""\
    FAILED tests/misc/test_thing.py::test_x - RuntimeError: something unexpected
    E   RuntimeError: something unexpected
    =========================== short test summary info ===========================
    FAILED tests/misc/test_thing.py::test_x
    1 failed
""")


class TestDiagnoseFailure:
    def test_classifies_import_error(self, fixer: ForwardFixer):
        result = fixer.diagnose_failure(PYTEST_IMPORT_ERROR)
        assert result.failure_type == FailureType.IMPORT_ERROR
        assert len(result.failed_tests) >= 1
        assert "test_client.py" in result.failed_tests[0]

    def test_classifies_assertion_error(self, fixer: ForwardFixer):
        result = fixer.diagnose_failure(PYTEST_ASSERTION_ERROR)
        assert result.failure_type == FailureType.ASSERTION_MISMATCH
        assert len(result.error_messages) >= 1

    def test_classifies_attribute_error(self, fixer: ForwardFixer):
        result = fixer.diagnose_failure(PYTEST_ATTRIBUTE_ERROR)
        assert result.failure_type == FailureType.ATTRIBUTE_ERROR

    def test_classifies_timeout(self, fixer: ForwardFixer):
        result = fixer.diagnose_failure(PYTEST_TIMEOUT)
        assert result.failure_type == FailureType.TIMEOUT

    def test_classifies_unknown(self, fixer: ForwardFixer):
        result = fixer.diagnose_failure(PYTEST_UNKNOWN)
        assert result.failure_type == FailureType.UNKNOWN
        assert result.confidence <= 0.3

    def test_diff_correlation_boosts_confidence(self, fixer: ForwardFixer):
        diff = textwrap.dedent("""\
            --- a/tests/sdk/test_client.py
            +++ b/tests/sdk/test_client.py
            @@ -1,3 +1,3 @@
            -from aragora.sdk import OldClient
            +from aragora.sdk import Client
        """)
        result = fixer.diagnose_failure(PYTEST_IMPORT_ERROR, diff=diff)
        assert result.confidence > 0.5
        assert result.likely_cause  # Should have a cause


class TestSuggestFix:
    def test_import_fix_suggestion(self, fixer: ForwardFixer):
        diag = fixer.diagnose_failure(PYTEST_IMPORT_ERROR)
        # Force confidence above threshold for test
        diag.confidence = 0.8
        fix = fixer.suggest_fix(diag)
        assert fix is not None
        assert fix.fix_type == "add_import"
        assert "Client" in fix.description

    def test_attribute_fix_suggestion(self, fixer: ForwardFixer):
        diag = fixer.diagnose_failure(PYTEST_ATTRIBUTE_ERROR)
        diag.confidence = 0.8
        fix = fixer.suggest_fix(diag)
        assert fix is not None
        assert fix.fix_type == "fix_attribute"
        assert "execute" in fix.description

    def test_assertion_fix_suggestion(self, fixer: ForwardFixer):
        diag = fixer.diagnose_failure(PYTEST_ASSERTION_ERROR)
        diag.confidence = 0.8
        fix = fixer.suggest_fix(diag)
        assert fix is not None
        assert fix.fix_type == "update_assertion"

    def test_returns_none_for_unknown(self, fixer: ForwardFixer):
        diag = fixer.diagnose_failure(PYTEST_UNKNOWN)
        diag.confidence = 0.9  # Even high confidence
        fix = fixer.suggest_fix(diag)
        assert fix is None

    def test_returns_none_for_low_confidence(self, fixer: ForwardFixer):
        diag = fixer.diagnose_failure(PYTEST_IMPORT_ERROR)
        diag.confidence = 0.3  # Below threshold
        fix = fixer.suggest_fix(diag)
        assert fix is None

    def test_returns_none_for_timeout(self, fixer: ForwardFixer):
        diag = fixer.diagnose_failure(PYTEST_TIMEOUT)
        diag.confidence = 0.9
        fix = fixer.suggest_fix(diag)
        assert fix is None  # No auto-fix for timeouts


class TestParsePytestOutput:
    def test_extracts_test_names(self, fixer: ForwardFixer):
        parsed = fixer._parse_pytest_output(PYTEST_IMPORT_ERROR)
        assert len(parsed) >= 1
        assert "test_client.py::test_init" in parsed[0]["test"]

    def test_extracts_error_messages(self, fixer: ForwardFixer):
        parsed = fixer._parse_pytest_output(PYTEST_ASSERTION_ERROR)
        assert len(parsed) >= 1
        assert "assert" in parsed[0]["error"].lower() or "AssertionError" in parsed[0]["error"]

    def test_empty_output(self, fixer: ForwardFixer):
        parsed = fixer._parse_pytest_output("")
        assert parsed == []


class TestCorrelateWithDiff:
    def test_finds_overlap(self, fixer: ForwardFixer):
        diff = textwrap.dedent("""\
            --- a/aragora/sdk/client.py
            +++ b/aragora/sdk/client.py
            @@ -1 +1 @@
            -old
            +new
        """)
        errors = [
            {
                "test": "tests/sdk/test_client.py::test_init",
                "error": "ImportError",
                "file": "aragora/sdk/client.py",
            }
        ]
        cause = fixer._correlate_with_diff(diff, errors)
        assert "client.py" in cause

    def test_no_overlap(self, fixer: ForwardFixer):
        diff = textwrap.dedent("""\
            --- a/unrelated.py
            +++ b/unrelated.py
        """)
        errors = [{"test": "t", "error": "e", "file": "other.py"}]
        cause = fixer._correlate_with_diff(diff, errors)
        assert cause == ""


class TestFailureTypeEnum:
    def test_all_types_have_values(self):
        assert len(FailureType) == 7
        for ft in FailureType:
            assert isinstance(ft.value, str)
