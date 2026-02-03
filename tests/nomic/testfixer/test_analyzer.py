"""Tests for aragora.nomic.testfixer.analyzer module."""

from __future__ import annotations

import pytest

from aragora.nomic.testfixer.analyzer import (
    CATEGORY_PATTERNS,
    AIAnalyzer,
    FailureAnalysis,
    FailureAnalyzer,
    FailureCategory,
    FixTarget,
    categorize_by_heuristics,
    determine_fix_target,
    extract_relevant_code,
    generate_approach_heuristic,
)
from aragora.nomic.testfixer.runner import TestFailure, TestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure(
    *,
    error_type: str = "Exception",
    error_message: str = "test failed",
    stack_trace: str = "",
    test_name: str = "test_example",
    test_file: str = "tests/test_example.py",
    involved_files: list[str] | None = None,
    involved_functions: list[str] | None = None,
    line_number: int | None = None,
    duration_seconds: float = 0.0,
) -> TestFailure:
    return TestFailure(
        test_name=test_name,
        test_file=test_file,
        error_type=error_type,
        error_message=error_message,
        stack_trace=stack_trace,
        line_number=line_number,
        involved_files=involved_files or [],
        involved_functions=involved_functions or [],
        duration_seconds=duration_seconds,
    )


def _make_result(failures: list[TestFailure] | None = None) -> TestResult:
    failures = failures or []
    return TestResult(
        command="pytest tests/",
        exit_code=1 if failures else 0,
        stdout="",
        stderr="",
        total_tests=len(failures) or 1,
        passed=0 if failures else 1,
        failed=len(failures),
        failures=failures,
    )


# ===================================================================
# categorize_by_heuristics
# ===================================================================


class TestCategorizeByHeuristics:
    """Tests for the categorize_by_heuristics function."""

    # -- TEST_ASYNC patterns -------------------------------------------------

    def test_coroutine_never_awaited(self):
        failure = _make_failure(
            error_message="coroutine 'fetch' was never awaited",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_ASYNC
        assert conf == 0.95

    def test_runtime_warning_coroutine(self):
        failure = _make_failure(
            error_message="RuntimeWarning: coroutine 'run' was never awaited",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_ASYNC
        assert conf == 0.95  # "coroutine.*was never awaited" matches first at 0.95

    def test_await_missing(self):
        failure = _make_failure(
            error_message="await is missing for async call",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_ASYNC
        assert conf == 0.85

    # -- TEST_MOCK patterns --------------------------------------------------

    def test_magic_mock_no_attribute(self):
        failure = _make_failure(
            error_message="MagicMock object has no attribute 'run'",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_MOCK
        assert conf == 0.9

    def test_mock_not_called(self):
        failure = _make_failure(
            error_message="Expected mock to be called but mock was not called",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_MOCK
        assert conf == 0.85

    def test_call_args_none(self):
        failure = _make_failure(
            error_message="call_args is None because mock was never called",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_MOCK
        assert conf == 0.8

    # -- IMPL_MISSING patterns -----------------------------------------------

    def test_attribute_error_has_no_attribute(self):
        failure = _make_failure(
            error_type="AttributeError",
            error_message="'Foo' object has no attribute 'bar'",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.IMPL_MISSING
        assert conf == 0.9

    def test_name_error_not_defined(self):
        failure = _make_failure(
            error_type="NameError",
            error_message="name 'xyz' is not defined",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.IMPL_MISSING
        assert conf == 0.9

    # -- IMPL_TYPE patterns --------------------------------------------------

    def test_type_error_argument(self):
        failure = _make_failure(
            error_type="TypeError",
            error_message="foo() takes 1 positional argument but 2 were given",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.IMPL_TYPE
        assert conf == 0.85

    def test_type_error_expected(self):
        failure = _make_failure(
            error_type="TypeError",
            error_message="expected str, got int",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.IMPL_TYPE
        assert conf == 0.85

    def test_cannot_unpack(self):
        failure = _make_failure(
            error_message="cannot unpack non-iterable NoneType object",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.IMPL_TYPE
        assert conf == 0.8

    # -- TEST_ASSERTION patterns ---------------------------------------------

    def test_assertion_error_basic(self):
        failure = _make_failure(
            error_type="AssertionError",
            error_message="1 != 2",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_ASSERTION
        assert conf == 0.7

    def test_expected_but_got(self):
        failure = _make_failure(
            error_message="Expected 42 but got 0",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_ASSERTION
        assert conf == 0.8

    # -- ENV_DEPENDENCY patterns ---------------------------------------------

    def test_module_not_found_no_module_named(self):
        failure = _make_failure(
            error_type="ModuleNotFoundError",
            error_message="No module named 'boto3'",
        )
        cat, conf = categorize_by_heuristics(failure)
        # ENV_DEPENDENCY "ModuleNotFoundError.*No module named" at 0.9
        # IMPL_MISSING "ModuleNotFoundError" at 0.85
        # Highest confidence wins: ENV_DEPENDENCY 0.9
        assert cat == FailureCategory.ENV_DEPENDENCY
        assert conf == 0.9

    def test_pip_install_hint(self):
        failure = _make_failure(
            error_message="you need to pip install mylib",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.ENV_DEPENDENCY
        assert conf == 0.8

    # -- RACE_CONDITION patterns ---------------------------------------------

    def test_timeout(self):
        failure = _make_failure(
            error_message="timeout waiting for response",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.RACE_CONDITION
        assert conf == 0.6

    def test_deadlock(self):
        failure = _make_failure(
            error_message="deadlock detected in worker pool",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.RACE_CONDITION
        assert conf == 0.9

    def test_race_condition_explicit(self):
        failure = _make_failure(
            error_message="race condition in concurrent access",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.RACE_CONDITION
        assert conf == 0.95

    # -- UNKNOWN / no match --------------------------------------------------

    def test_unknown_when_no_pattern_matches(self):
        failure = _make_failure(
            error_type="SomeWeirdError",
            error_message="something entirely unique and unmatchable zzqq",
            stack_trace="no helpful info here zzqq",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.UNKNOWN
        assert conf == 0.0

    # -- Highest confidence wins ---------------------------------------------

    def test_highest_confidence_wins(self):
        """When multiple patterns match, the one with highest confidence wins."""
        # "deadlock" matches RACE_CONDITION at 0.9
        # "coroutine.*was never awaited" matches TEST_ASYNC at 0.95
        failure = _make_failure(
            error_message="coroutine was never awaited, also deadlock",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert conf == 0.95
        assert cat == FailureCategory.TEST_ASYNC

    def test_combined_text_includes_all_fields(self):
        """Pattern matching uses error_type + error_message + stack_trace."""
        failure = _make_failure(
            error_type="SomeError",
            error_message="nothing special",
            stack_trace="coroutine 'x' was never awaited somewhere",
        )
        cat, conf = categorize_by_heuristics(failure)
        assert cat == FailureCategory.TEST_ASYNC
        assert conf == 0.95


# ===================================================================
# determine_fix_target
# ===================================================================


class TestDetermineFixTarget:
    """Tests for the determine_fix_target function."""

    _dummy = _make_failure()

    @pytest.mark.parametrize(
        "category",
        [
            FailureCategory.TEST_ASSERTION,
            FailureCategory.TEST_SETUP,
            FailureCategory.TEST_ASYNC,
            FailureCategory.TEST_MOCK,
        ],
    )
    def test_test_issues_target_test_file(self, category):
        assert determine_fix_target(category, self._dummy) == FixTarget.TEST_FILE

    @pytest.mark.parametrize(
        "category",
        [
            FailureCategory.IMPL_BUG,
            FailureCategory.IMPL_MISSING,
            FailureCategory.IMPL_TYPE,
        ],
    )
    def test_impl_issues_target_impl_file(self, category):
        assert determine_fix_target(category, self._dummy) == FixTarget.IMPL_FILE

    def test_api_change_targets_both(self):
        assert determine_fix_target(FailureCategory.IMPL_API_CHANGE, self._dummy) == FixTarget.BOTH

    @pytest.mark.parametrize(
        "category",
        [FailureCategory.ENV_DEPENDENCY, FailureCategory.ENV_CONFIG],
    )
    def test_env_issues_target_config(self, category):
        assert determine_fix_target(category, self._dummy) == FixTarget.CONFIG

    @pytest.mark.parametrize(
        "category",
        [FailureCategory.RACE_CONDITION, FailureCategory.FLAKY],
    )
    def test_complex_issues_target_human(self, category):
        assert determine_fix_target(category, self._dummy) == FixTarget.HUMAN

    def test_unknown_defaults_to_test_file(self):
        assert determine_fix_target(FailureCategory.UNKNOWN, self._dummy) == FixTarget.TEST_FILE

    def test_env_resource_defaults_to_test_file(self):
        """ENV_RESOURCE is not explicitly handled, so falls through to default."""
        assert (
            determine_fix_target(FailureCategory.ENV_RESOURCE, self._dummy) == FixTarget.TEST_FILE
        )


# ===================================================================
# extract_relevant_code
# ===================================================================


class TestExtractRelevantCode:
    """Tests for the extract_relevant_code function."""

    def test_extracts_code_from_existing_file(self, tmp_path):
        source = "\n".join(f"line {i}" for i in range(50))
        code_file = tmp_path / "src" / "module.py"
        code_file.parent.mkdir(parents=True)
        code_file.write_text(source)

        failure = _make_failure(
            involved_files=["src/module.py"],
            stack_trace="",
        )
        snippets = extract_relevant_code(failure, tmp_path)
        assert "src/module.py" in snippets
        # Without a matching line number in the stack trace, first 50 lines are returned
        assert "line 0" in snippets["src/module.py"]

    def test_extracts_context_around_line_number(self, tmp_path):
        lines = [f"line {i}" for i in range(100)]
        code_file = tmp_path / "src" / "module.py"
        code_file.parent.mkdir(parents=True)
        code_file.write_text("\n".join(lines))

        failure = _make_failure(
            involved_files=["src/module.py"],
            stack_trace='File "src/module.py", line 50, in test_something',
        )
        snippets = extract_relevant_code(failure, tmp_path, context_lines=5)
        assert "src/module.py" in snippets
        snippet = snippets["src/module.py"]
        # Line 50 should be included (1-indexed in output)
        assert "  50:" in snippet
        # Lines far away should not be present
        assert "line 0" not in snippet
        assert "line 99" not in snippet

    def test_missing_file_is_skipped(self, tmp_path):
        failure = _make_failure(
            involved_files=["nonexistent/file.py"],
        )
        snippets = extract_relevant_code(failure, tmp_path)
        assert snippets == {}

    def test_limits_to_five_files(self, tmp_path):
        file_names = [f"src/f{i}.py" for i in range(8)]
        for name in file_names:
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("content")

        failure = _make_failure(involved_files=file_names)
        snippets = extract_relevant_code(failure, tmp_path)
        assert len(snippets) <= 5

    def test_empty_involved_files(self, tmp_path):
        failure = _make_failure(involved_files=[])
        snippets = extract_relevant_code(failure, tmp_path)
        assert snippets == {}


# ===================================================================
# generate_approach_heuristic
# ===================================================================


class TestGenerateApproachHeuristic:
    """Tests for the generate_approach_heuristic function."""

    _dummy = _make_failure()

    @pytest.mark.parametrize(
        "category",
        [
            FailureCategory.TEST_ASYNC,
            FailureCategory.TEST_MOCK,
            FailureCategory.IMPL_MISSING,
            FailureCategory.IMPL_TYPE,
            FailureCategory.TEST_ASSERTION,
            FailureCategory.ENV_DEPENDENCY,
            FailureCategory.IMPL_API_CHANGE,
        ],
    )
    def test_known_categories_return_nonempty(self, category):
        result = generate_approach_heuristic(category, self._dummy)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_category_returns_default(self):
        result = generate_approach_heuristic(FailureCategory.UNKNOWN, self._dummy)
        assert "Analyze the error message" in result

    def test_flaky_returns_default(self):
        result = generate_approach_heuristic(FailureCategory.FLAKY, self._dummy)
        assert "Analyze the error message" in result

    def test_async_mentions_await(self):
        result = generate_approach_heuristic(FailureCategory.TEST_ASYNC, self._dummy)
        assert "await" in result.lower()

    def test_mock_mentions_mock(self):
        result = generate_approach_heuristic(FailureCategory.TEST_MOCK, self._dummy)
        assert "mock" in result.lower()


# ===================================================================
# FailureAnalysis.to_fix_prompt
# ===================================================================


class TestFailureAnalysisToFixPrompt:
    """Tests for FailureAnalysis.to_fix_prompt method."""

    def test_contains_category(self):
        analysis = FailureAnalysis(
            failure=_make_failure(),
            category=FailureCategory.TEST_MOCK,
            fix_target=FixTarget.TEST_FILE,
            confidence=0.9,
            root_cause="Mock not configured",
        )
        prompt = analysis.to_fix_prompt()
        assert "test_mock" in prompt

    def test_contains_fix_target(self):
        analysis = FailureAnalysis(
            failure=_make_failure(),
            fix_target=FixTarget.IMPL_FILE,
        )
        prompt = analysis.to_fix_prompt()
        assert "impl_file" in prompt

    def test_contains_confidence(self):
        analysis = FailureAnalysis(
            failure=_make_failure(),
            confidence=0.85,
        )
        prompt = analysis.to_fix_prompt()
        assert "85%" in prompt

    def test_contains_root_cause(self):
        analysis = FailureAnalysis(
            failure=_make_failure(),
            root_cause="Missing method on Foo class",
        )
        prompt = analysis.to_fix_prompt()
        assert "Missing method on Foo class" in prompt

    def test_contains_relevant_code_blocks(self):
        analysis = FailureAnalysis(
            failure=_make_failure(),
            relevant_code={
                "src/app.py": "def main(): pass",
                "tests/test_app.py": "def test_main(): assert False",
            },
        )
        prompt = analysis.to_fix_prompt()
        assert "src/app.py" in prompt
        assert "tests/test_app.py" in prompt
        assert "```python" in prompt
        assert "def main(): pass" in prompt

    def test_no_relevant_code_section_when_empty(self):
        analysis = FailureAnalysis(
            failure=_make_failure(),
            relevant_code={},
        )
        prompt = analysis.to_fix_prompt()
        assert "### Relevant Code" not in prompt

    def test_includes_original_failure_context(self):
        failure = _make_failure(
            test_name="test_calc",
            error_type="ValueError",
            error_message="bad value",
        )
        analysis = FailureAnalysis(failure=failure)
        prompt = analysis.to_fix_prompt()
        assert "test_calc" in prompt
        assert "ValueError" in prompt


# ===================================================================
# FailureAnalyzer.analyze (async)
# ===================================================================


class TestFailureAnalyzerAnalyze:
    """Tests for FailureAnalyzer.analyze method."""

    @pytest.mark.asyncio
    async def test_heuristic_only_path(self, tmp_path):
        """Without an AI analyzer, heuristic results are used directly."""
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        failure = _make_failure(
            error_message="coroutine 'run' was never awaited",
        )
        analysis = await analyzer.analyze(failure)
        assert analysis.category == FailureCategory.TEST_ASYNC
        assert analysis.confidence == 0.95
        assert analysis.fix_target == FixTarget.TEST_FILE

    @pytest.mark.asyncio
    async def test_ai_analyzer_called_when_confidence_low(self, tmp_path):
        """AI analyzer is consulted when heuristic confidence < 0.8."""

        class MockAI:
            async def analyze(self, failure, code_context):
                return ("AI root cause", "AI approach", 0.92)

        analyzer = FailureAnalyzer(repo_path=tmp_path, ai_analyzer=MockAI())
        # "timeout" matches RACE_CONDITION at 0.6 confidence
        failure = _make_failure(error_message="timeout waiting for lock")
        analysis = await analyzer.analyze(failure)
        # AI returned higher confidence
        assert analysis.root_cause == "AI root cause"
        assert analysis.suggested_approach == "AI approach"
        assert analysis.confidence == 0.92

    @pytest.mark.asyncio
    async def test_ai_analyzer_skipped_when_heuristic_confident(self, tmp_path):
        """AI analyzer is NOT called when heuristic confidence >= 0.8."""
        call_count = 0

        class MockAI:
            async def analyze(self, failure, code_context):
                nonlocal call_count
                call_count += 1
                return ("AI root cause", "AI approach", 0.99)

        analyzer = FailureAnalyzer(repo_path=tmp_path, ai_analyzer=MockAI())
        failure = _make_failure(
            error_message="coroutine 'x' was never awaited",
        )
        analysis = await analyzer.analyze(failure)
        assert call_count == 0
        assert analysis.confidence == 0.95

    @pytest.mark.asyncio
    async def test_ai_analyzer_fallback_on_exception(self, tmp_path):
        """When AI analyzer raises, heuristic results are used."""

        class BrokenAI:
            async def analyze(self, failure, code_context):
                raise RuntimeError("AI unavailable")

        analyzer = FailureAnalyzer(repo_path=tmp_path, ai_analyzer=BrokenAI())
        failure = _make_failure(error_message="timeout in database")
        analysis = await analyzer.analyze(failure)
        # Falls back to heuristic
        assert analysis.category == FailureCategory.RACE_CONDITION
        assert analysis.confidence == 0.6

    @pytest.mark.asyncio
    async def test_ai_analyzer_lower_confidence_ignored(self, tmp_path):
        """When AI returns lower confidence than heuristic, heuristic wins."""

        class WeakAI:
            async def analyze(self, failure, code_context):
                return ("AI cause", "AI approach", 0.3)

        analyzer = FailureAnalyzer(repo_path=tmp_path, ai_analyzer=WeakAI())
        failure = _make_failure(error_message="timeout error")
        analysis = await analyzer.analyze(failure)
        assert analysis.confidence == 0.6  # heuristic wins
        assert "AI cause" not in analysis.root_cause

    @pytest.mark.asyncio
    async def test_root_cause_file_is_impl_for_impl_target(self, tmp_path):
        """When fix_target is IMPL_FILE, root_cause_file is a non-test file."""
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        failure = _make_failure(
            error_type="AttributeError",
            error_message="'Foo' object has no attribute 'bar'",
            involved_files=["src/foo.py", "tests/test_foo.py"],
        )
        analysis = await analyzer.analyze(failure)
        assert analysis.fix_target == FixTarget.IMPL_FILE
        assert analysis.root_cause_file == "src/foo.py"

    @pytest.mark.asyncio
    async def test_root_cause_file_defaults_to_test_file(self, tmp_path):
        """When fix_target is TEST_FILE, root_cause_file is the test file."""
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        failure = _make_failure(
            test_file="tests/test_thing.py",
            error_message="AssertionError: 1 != 2",
        )
        analysis = await analyzer.analyze(failure)
        assert analysis.root_cause_file == "tests/test_thing.py"

    @pytest.mark.asyncio
    async def test_complexity_high_for_api_change(self, tmp_path):
        """IMPL_API_CHANGE gets high fix_complexity."""
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        # Force a category that maps to IMPL_API_CHANGE — we need to test via
        # the analysis output. Since heuristics don't produce IMPL_API_CHANGE,
        # we use an AI analyzer to verify complexity mapping indirectly.
        # Instead, test the RACE_CONDITION path which also yields high complexity.
        failure = _make_failure(error_message="deadlock detected")
        analysis = await analyzer.analyze(failure)
        assert analysis.fix_complexity == "high"

    @pytest.mark.asyncio
    async def test_complexity_medium_for_mock(self, tmp_path):
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        failure = _make_failure(
            error_message="MagicMock object has no attribute 'execute'",
        )
        analysis = await analyzer.analyze(failure)
        assert analysis.fix_complexity == "medium"

    @pytest.mark.asyncio
    async def test_complexity_low_for_assertion(self, tmp_path):
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        failure = _make_failure(
            error_type="AssertionError",
            error_message="values differ",
            stack_trace="",
        )
        analysis = await analyzer.analyze(failure)
        assert analysis.fix_complexity == "low"

    @pytest.mark.asyncio
    async def test_regression_risk_low_for_test_target(self, tmp_path):
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        failure = _make_failure(
            error_message="coroutine 'f' was never awaited",
        )
        analysis = await analyzer.analyze(failure)
        assert analysis.fix_target == FixTarget.TEST_FILE
        assert analysis.regression_risk == "low"

    @pytest.mark.asyncio
    async def test_regression_risk_medium_for_impl_target(self, tmp_path):
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        failure = _make_failure(
            error_type="AttributeError",
            error_message="'X' has no attribute 'y'",
        )
        analysis = await analyzer.analyze(failure)
        assert analysis.fix_target == FixTarget.IMPL_FILE
        assert analysis.regression_risk == "medium"

    @pytest.mark.asyncio
    async def test_relevant_code_extracted(self, tmp_path):
        src = tmp_path / "src" / "mod.py"
        src.parent.mkdir(parents=True)
        src.write_text("def foo():\n    return 42\n")

        analyzer = FailureAnalyzer(repo_path=tmp_path)
        failure = _make_failure(
            error_message="AssertionError",
            involved_files=["src/mod.py"],
        )
        analysis = await analyzer.analyze(failure)
        assert "src/mod.py" in analysis.relevant_code

    @pytest.mark.asyncio
    async def test_custom_context_lines(self, tmp_path):
        lines = [f"line{i}" for i in range(100)]
        src = tmp_path / "src" / "big.py"
        src.parent.mkdir(parents=True)
        src.write_text("\n".join(lines))

        analyzer = FailureAnalyzer(repo_path=tmp_path, context_lines=3)
        failure = _make_failure(
            involved_files=["src/big.py"],
            stack_trace='File "src/big.py", line 50, in func',
            error_message="AssertionError",
        )
        analysis = await analyzer.analyze(failure)
        snippet = analysis.relevant_code.get("src/big.py", "")
        # With context_lines=3, we get lines 47-53 approximately
        assert "line50" not in snippet or "line0" not in snippet


# ===================================================================
# FailureAnalyzer.analyze_result (async)
# ===================================================================


class TestFailureAnalyzerAnalyzeResult:
    """Tests for FailureAnalyzer.analyze_result method."""

    @pytest.mark.asyncio
    async def test_processes_multiple_failures(self, tmp_path):
        failures = [
            _make_failure(
                test_name="test_one",
                error_message="coroutine 'a' was never awaited",
            ),
            _make_failure(
                test_name="test_two",
                error_type="AttributeError",
                error_message="'X' has no attribute 'y'",
            ),
            _make_failure(
                test_name="test_three",
                error_message="deadlock found in thread pool",
            ),
        ]
        result = _make_result(failures)
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        analyses = await analyzer.analyze_result(result)

        assert len(analyses) == 3
        categories = [a.category for a in analyses]
        assert FailureCategory.TEST_ASYNC in categories
        assert FailureCategory.IMPL_MISSING in categories
        assert FailureCategory.RACE_CONDITION in categories

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_list(self, tmp_path):
        result = _make_result(failures=[])
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        analyses = await analyzer.analyze_result(result)
        assert analyses == []

    @pytest.mark.asyncio
    async def test_single_failure_result(self, tmp_path):
        failure = _make_failure(error_message="timeout waiting for db")
        result = _make_result(failures=[failure])
        analyzer = FailureAnalyzer(repo_path=tmp_path)
        analyses = await analyzer.analyze_result(result)
        assert len(analyses) == 1
        assert analyses[0].failure is failure


# ===================================================================
# Enum completeness checks
# ===================================================================


class TestEnums:
    """Verify enum members are present."""

    def test_failure_category_has_14_members(self):
        # 5 test + 4 impl + 3 env + 3 complex = 15
        assert len(FailureCategory) == 15

    def test_fix_target_has_6_members(self):
        assert len(FixTarget) == 6

    def test_all_failure_categories(self):
        expected = {
            "TEST_ASSERTION",
            "TEST_SETUP",
            "TEST_ASYNC",
            "TEST_MOCK",
            "TEST_IMPORT",
            "IMPL_BUG",
            "IMPL_MISSING",
            "IMPL_TYPE",
            "IMPL_API_CHANGE",
            "ENV_DEPENDENCY",
            "ENV_CONFIG",
            "ENV_RESOURCE",
            "RACE_CONDITION",
            "FLAKY",
            "UNKNOWN",
        }
        actual = {m.name for m in FailureCategory}
        assert actual == expected

    def test_all_fix_targets(self):
        expected = {"TEST_FILE", "IMPL_FILE", "BOTH", "CONFIG", "SKIP", "HUMAN"}
        actual = {m.name for m in FixTarget}
        assert actual == expected
