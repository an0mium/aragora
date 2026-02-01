"""
Edge case tests for the Smart Test Generator Module.

Covers gaps not addressed by the primary test file:
- Async function analysis and test generation
- Complex type annotations (dict[str, Any], Optional, Union, nested generics)
- Class methods and static methods extraction
- Decorated functions (@staticmethod, @property, @classmethod)
- Error recovery with invalid/malformed Python code
- Large functions with many branches and nested structures
"""

from __future__ import annotations

import pytest

from aragora.coding.test_generator import (
    TestCase,
    TestFramework,
    TestGenerator,
    TestSuite,
    TestType,
    generate_tests_for_function,
)


# =============================================================================
# Async Function Tests
# =============================================================================


class TestAsyncFunctionAnalysis:
    """Tests for async function analysis and test generation."""

    def test_analyze_async_with_branches_and_raises(self):
        """Should fully analyze an async function with branches and exceptions."""
        code = """
async def fetch_user(user_id: int) -> dict:
    if user_id <= 0:
        raise ValueError("Invalid user ID")
    result = await db_query(user_id)
    if result is None:
        raise KeyError("User not found")
    return result
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "fetch_user")

        assert analysis["function_name"] == "fetch_user"
        assert len(analysis["parameters"]) == 1
        assert analysis["parameters"][0]["name"] == "user_id"
        assert analysis["parameters"][0]["type"] == "int"
        assert analysis["return_type"] == "dict"
        assert analysis["branches"] == 2
        assert len(analysis["raises"]) == 2
        assert any("ValueError" in r for r in analysis["raises"])
        assert any("KeyError" in r for r in analysis["raises"])

    def test_async_function_generates_test_cases(self):
        """Should generate full test cases for an async function via analysis."""
        code = """
async def send_message(channel: str, text: str) -> bool:
    if not channel:
        raise ValueError("No channel")
    return True
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "send_message")
        cases = gen.generate_test_cases("send_message", analysis)

        # Should have happy path, None tests for each param, string boundary tests, and error test
        assert len(cases) >= 6
        func_names = {c.function_under_test for c in cases}
        assert func_names == {"send_message"}

        # Error handling for ValueError
        error_cases = [c for c in cases if c.test_type == TestType.ERROR]
        assert len(error_cases) == 1
        assert error_cases[0].expected_error == "ValueError"

    def test_async_function_extracted_by_name(self):
        """Should extract async functions from module-level code."""
        code = """
async def handle_request(request: str) -> str:
    return "ok"

async def process_batch(items: list) -> list:
    return items
"""
        gen = TestGenerator()
        names = gen._extract_function_names(code)

        assert "handle_request" in names
        assert "process_batch" in names


# =============================================================================
# Complex Type Annotation Tests
# =============================================================================


class TestComplexTypeAnnotations:
    """Tests for functions with complex type annotations."""

    def test_dict_str_any_annotation(self):
        """Should parse dict[str, Any] type annotation."""
        code = """
from typing import Any

def merge_configs(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return {**base, **overrides}
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "merge_configs")

        assert len(analysis["parameters"]) == 2
        assert "dict" in analysis["parameters"][0]["type"]
        assert analysis["return_type"] is not None
        assert "dict" in analysis["return_type"]

    def test_optional_int_annotation(self):
        """Should parse Optional[int] and generate appropriate boundary tests."""
        code = """
from typing import Optional

def get_timeout(override: Optional[int] = None) -> int:
    if override is not None:
        return override
    return 30
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "get_timeout")

        # Optional[int] contains 'int' so int-boundary tests should fire
        param_type = analysis["parameters"][0].get("type", "")
        assert "int" in param_type.lower() or "Optional" in param_type

        cases = gen.generate_test_cases("get_timeout", analysis)
        # Should include at least: happy path + none test + int boundary tests (zero, negative)
        assert len(cases) >= 3

    def test_union_type_annotation(self):
        """Should parse Union[str, int] without error."""
        code = """
from typing import Union

def normalize(value: Union[str, int]) -> str:
    return str(value)
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "normalize")

        param_type = analysis["parameters"][0].get("type", "")
        # The unparsed annotation should contain both types
        assert "str" in param_type or "Union" in param_type
        assert analysis["return_type"] == "str"

    def test_nested_generic_annotation(self):
        """Should parse list[tuple[int, ...]] annotation and handle type priority."""
        code = """
def flatten_pairs(pairs: list[tuple[int, ...]]) -> list[int]:
    result = []
    for pair in pairs:
        result.extend(pair)
    return result
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "flatten_pairs")

        param_type = analysis["parameters"][0].get("type", "")
        assert "list" in param_type.lower()
        assert "tuple" in param_type.lower()
        assert analysis["return_type"] is not None

        # Because the type string "list[tuple[int, ...]]" contains "int",
        # the implementation's elif chain matches the int branch first
        # (int check precedes list check). This produces int-boundary tests
        # rather than list-boundary tests -- verifying actual behavior.
        cases = gen.generate_test_cases("flatten_pairs", analysis)
        zero_cases = [c for c in cases if "zero" in c.tags]
        negative_cases = [c for c in cases if "negative" in c.tags]
        assert len(zero_cases) >= 1
        assert len(negative_cases) >= 1

    def test_multiple_complex_annotations_together(self):
        """Should handle a function with several complex annotations at once."""
        code = """
from typing import Any, Optional

def query(
    table: str,
    filters: dict[str, Any],
    limit: Optional[int] = None,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    pass
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "query")

        assert len(analysis["parameters"]) == 4
        param_names = [p["name"] for p in analysis["parameters"]]
        assert param_names == ["table", "filters", "limit", "columns"]
        assert analysis["return_type"] is not None


# =============================================================================
# Class Methods and Static Methods Tests
# =============================================================================


class TestClassMethodExtraction:
    """Tests for extracting and analyzing methods inside class bodies."""

    def test_extract_public_methods_from_class(self):
        """Should extract public methods defined inside a class."""
        code = """
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b

    def _internal_reset(self):
        pass
"""
        gen = TestGenerator()
        names = gen._extract_function_names(code)

        assert "add" in names
        assert "subtract" in names
        assert "_internal_reset" not in names

    def test_analyze_method_includes_self(self):
        """Should include 'self' in parameters when analyzing a method."""
        code = """
class Processor:
    def run(self, data: list[str]) -> int:
        return len(data)
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "run")

        param_names = [p["name"] for p in analysis["parameters"]]
        assert "self" in param_names
        assert "data" in param_names

    def test_classmethod_analysis(self):
        """Should analyze a classmethod, including cls parameter."""
        code = """
class Factory:
    @classmethod
    def create(cls, name: str) -> 'Factory':
        return cls()
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "create")

        param_names = [p["name"] for p in analysis["parameters"]]
        assert "cls" in param_names
        assert "name" in param_names

    def test_staticmethod_analysis(self):
        """Should analyze a staticmethod with no self/cls parameter."""
        code = """
class MathUtils:
    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        if value < low:
            return low
        if value > high:
            return high
        return value
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "clamp")

        param_names = [p["name"] for p in analysis["parameters"]]
        assert "self" not in param_names
        assert "cls" not in param_names
        assert param_names == ["value", "low", "high"]
        assert analysis["branches"] == 2


# =============================================================================
# Decorator Tests
# =============================================================================


class TestDecoratedFunctions:
    """Tests for functions/methods with decorators."""

    def test_property_method_extracted(self):
        """Should extract a property-decorated method as a public function."""
        code = """
class Config:
    @property
    def timeout(self) -> int:
        return self._timeout
"""
        gen = TestGenerator()
        names = gen._extract_function_names(code)

        assert "timeout" in names

    def test_multiple_decorators_do_not_break_analysis(self):
        """Should analyze a function with multiple stacked decorators."""
        code = """
import functools

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@log
def compute(x: int, y: int) -> int:
    if x < 0:
        raise ValueError("negative x")
    return x + y
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "compute")

        assert analysis["function_name"] == "compute"
        assert len(analysis["parameters"]) == 2
        assert analysis["branches"] >= 1
        assert len(analysis["raises"]) >= 1

    def test_decorated_function_generates_tests(self):
        """Should generate test cases for a decorated function."""
        code = """
def retry(func):
    return func

@retry
def fetch(url: str) -> str:
    if not url:
        raise ValueError("empty url")
    return "data"
"""
        gen = TestGenerator()
        suite = gen.generate_test_suite(code, "fetcher", functions=["fetch"])

        assert len(suite.tests) > 0
        func_names = {t.function_under_test for t in suite.tests}
        assert func_names == {"fetch"}

        error_tests = [t for t in suite.tests if t.test_type == TestType.ERROR]
        assert len(error_tests) >= 1


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for behavior with invalid or malformed Python code input."""

    def test_completely_empty_code(self):
        """Should handle empty string code gracefully."""
        gen = TestGenerator()
        analysis = gen.analyze_function("", "nonexistent")

        # Should parse successfully but find no matching function
        assert analysis["function_name"] == "nonexistent"
        assert analysis["parameters"] == []

    def test_non_python_code(self):
        """Should return error for non-Python content."""
        gen = TestGenerator()
        analysis = gen.analyze_function("function hello() { return 1; }", "hello")

        # JavaScript is not valid Python, should return parse error
        assert "error" in analysis

    def test_incomplete_function_definition(self):
        """Should return error for truncated code."""
        code = "def incomplete(x: int"
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "incomplete")

        assert "error" in analysis

    def test_generate_suite_with_syntax_error_code(self):
        """Should produce an empty suite when the code cannot be parsed."""
        gen = TestGenerator()
        suite = gen.generate_test_suite("def ???():", "broken_module")

        assert suite.name == "broken_module"
        assert suite.tests == []

    def test_analyze_function_not_found(self):
        """Should return default analysis when function name does not exist in code."""
        code = """
def existing_func(x: int) -> int:
    return x
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "nonexistent_func")

        # Should not error, just return empty analysis fields
        assert "error" not in analysis
        assert analysis["function_name"] == "nonexistent_func"
        assert analysis["parameters"] == []
        assert analysis["return_type"] is None


# =============================================================================
# Large / Complex Function Tests
# =============================================================================


class TestLargeComplexFunctions:
    """Tests for analysis of functions with high cyclomatic complexity."""

    def test_many_branches(self):
        """Should correctly count many if/elif branches."""
        code = """
def categorize(code: int) -> str:
    if code < 100:
        return "info"
    elif code < 200:
        return "success"
    elif code < 300:
        return "redirect"
    elif code < 400:
        return "client_error"
    elif code < 500:
        return "server_error"
    else:
        return "unknown"
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "categorize")

        # 5 branches: if + 4 elif
        assert analysis["branches"] == 5
        assert analysis["complexity"] >= 6  # 1 base + 5 branches

    def test_nested_branches_and_loops(self):
        """Should count nested if statements and loops independently."""
        code = """
def process_matrix(matrix: list, threshold: float) -> list:
    result = []
    for row in matrix:
        row_result = []
        for value in row:
            if value > threshold:
                if value > threshold * 2:
                    row_result.append("high")
                else:
                    row_result.append("medium")
            else:
                row_result.append("low")
        result.append(row_result)
    return result
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "process_matrix")

        assert analysis["loops"] == 2
        assert analysis["branches"] == 2
        assert analysis["complexity"] >= 5  # 1 base + 2 loops + 2 ifs

    def test_function_with_while_loop(self):
        """Should count while loops toward complexity."""
        code = """
def retry_operation(max_attempts: int) -> bool:
    attempts = 0
    while attempts < max_attempts:
        if try_once():
            return True
        attempts += 1
    return False
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "retry_operation")

        assert analysis["loops"] >= 1
        assert analysis["branches"] >= 1
        assert "try_once" in analysis["calls"]

    def test_complex_function_generates_many_tests(self):
        """A complex function with multiple typed params and exceptions should yield many test cases."""
        code = """
def transform(
    data: list,
    factor: float,
    label: str,
) -> dict:
    if not data:
        raise ValueError("No data")
    if factor <= 0:
        raise ValueError("Bad factor")
    result = []
    for item in data:
        if isinstance(item, str):
            result.append(item.upper())
        else:
            result.append(item * factor)
    return {"label": label, "result": result}
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "transform")
        cases = gen.generate_test_cases("transform", analysis)

        # Should have: happy path (1) + 3 none tests + list boundary for data (2)
        # + float boundary for factor (2) + str boundary for label (2) + 2 error tests = 12+
        assert len(cases) >= 10

        # Verify diversity of test types
        test_types = {c.test_type for c in cases}
        assert TestType.UNIT in test_types
        assert TestType.EDGE_CASE in test_types
        assert TestType.BOUNDARY in test_types
        assert TestType.ERROR in test_types

    def test_method_call_detection(self):
        """Should detect attribute-style method calls (e.g., obj.method())."""
        code = """
def pipeline(items: list) -> list:
    cleaned = items.copy()
    cleaned.sort()
    return cleaned
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "pipeline")

        # ast.Attribute calls: copy, sort
        assert "copy" in analysis["calls"]
        assert "sort" in analysis["calls"]
