"""
Tests for the Smart Test Generator Module.

Tests cover:
- TestFramework enum
- TestType enum
- TestCase dataclass
- TestSuite dataclass and code generation
- TestGenerator class with analysis and test generation
- Utility functions
- Convenience functions
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

from aragora.coding.test_generator import (
    TestCase,
    TestFramework,
    TestGenerator,
    TestSuite,
    TestType,
    generate_tests_for_file,
    generate_tests_for_function,
    _js_repr,
    _to_pascal_case,
    _to_snake_case,
)


# =============================================================================
# TestFramework Tests
# =============================================================================


class TestTestFramework:
    """Tests for TestFramework enum."""

    def test_all_values(self):
        """Should have all expected framework values."""
        assert TestFramework.PYTEST.value == "pytest"
        assert TestFramework.UNITTEST.value == "unittest"
        assert TestFramework.JEST.value == "jest"
        assert TestFramework.MOCHA.value == "mocha"
        assert TestFramework.VITEST.value == "vitest"
        assert TestFramework.RSPEC.value == "rspec"
        assert TestFramework.GO_TEST.value == "go_test"
        assert TestFramework.RUST_TEST.value == "rust_test"

    def test_is_string_enum(self):
        """Test that TestFramework is a string enum."""
        assert isinstance(TestFramework.PYTEST, str)
        assert TestFramework.PYTEST == "pytest"


# =============================================================================
# TestType Tests
# =============================================================================


class TestTestType:
    """Tests for TestType enum."""

    def test_all_values(self):
        """Should have all expected test type values."""
        assert TestType.UNIT.value == "unit"
        assert TestType.INTEGRATION.value == "integration"
        assert TestType.E2E.value == "e2e"
        assert TestType.PROPERTY.value == "property"
        assert TestType.EDGE_CASE.value == "edge_case"
        assert TestType.ERROR.value == "error"
        assert TestType.BOUNDARY.value == "boundary"

    def test_is_string_enum(self):
        """Test that TestType is a string enum."""
        assert isinstance(TestType.UNIT, str)
        assert TestType.UNIT == "unit"


# =============================================================================
# TestCase Tests
# =============================================================================


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_minimal_creation(self):
        """Should create TestCase with minimal required fields."""
        case = TestCase(
            name="test_example",
            description="An example test",
            test_type=TestType.UNIT,
            function_under_test="my_function",
        )

        assert case.name == "test_example"
        assert case.description == "An example test"
        assert case.test_type == TestType.UNIT
        assert case.function_under_test == "my_function"
        assert case.inputs == {}
        assert case.expected_output is None
        assert case.assertions == []
        assert case.priority == 5

    def test_full_creation(self):
        """Should create TestCase with all fields."""
        case = TestCase(
            name="test_complex",
            description="A complex test",
            test_type=TestType.INTEGRATION,
            function_under_test="process_data",
            inputs={"data": [1, 2, 3]},
            expected_output=6,
            expected_error=None,
            assertions=["result > 0", "result < 100"],
            setup="mock_db()",
            teardown="cleanup_db()",
            tags=["integration", "database"],
            priority=1,
        )

        assert case.inputs == {"data": [1, 2, 3]}
        assert case.expected_output == 6
        assert len(case.assertions) == 2
        assert case.setup == "mock_db()"
        assert case.teardown == "cleanup_db()"
        assert "integration" in case.tags
        assert case.priority == 1

    def test_to_dict(self):
        """Should convert to dictionary."""
        case = TestCase(
            name="test_dict",
            description="Dict conversion test",
            test_type=TestType.UNIT,
            function_under_test="my_func",
            inputs={"x": 1},
            expected_output=2,
            tags=["fast"],
        )

        data = case.to_dict()

        assert data["name"] == "test_dict"
        assert data["description"] == "Dict conversion test"
        assert data["test_type"] == "unit"
        assert data["function_under_test"] == "my_func"
        assert data["inputs"] == {"x": 1}
        assert data["expected_output"] == 2
        assert data["tags"] == ["fast"]
        assert data["priority"] == 5

    def test_to_dict_with_error(self):
        """Should include expected_error in dict."""
        case = TestCase(
            name="test_error",
            description="Error test",
            test_type=TestType.ERROR,
            function_under_test="raise_error",
            expected_error="ValueError",
        )

        data = case.to_dict()
        assert data["expected_error"] == "ValueError"


# =============================================================================
# TestSuite Tests
# =============================================================================


class TestTestSuite:
    """Tests for TestSuite dataclass."""

    def test_minimal_creation(self):
        """Should create TestSuite with minimal fields."""
        suite = TestSuite(
            name="my_module",
            file_path="test_my_module.py",
            framework=TestFramework.PYTEST,
        )

        assert suite.name == "my_module"
        assert suite.file_path == "test_my_module.py"
        assert suite.framework == TestFramework.PYTEST
        assert suite.tests == []
        assert suite.imports == []
        assert suite.coverage_target == 80.0

    def test_with_tests(self):
        """Should create TestSuite with tests."""
        test_case = TestCase(
            name="test_one",
            description="First test",
            test_type=TestType.UNIT,
            function_under_test="func_one",
        )

        suite = TestSuite(
            name="my_module",
            file_path="test_my_module.py",
            framework=TestFramework.PYTEST,
            tests=[test_case],
        )

        assert len(suite.tests) == 1
        assert suite.tests[0].name == "test_one"

    def test_to_dict(self):
        """Should convert suite to dictionary."""
        case = TestCase(
            name="test_one",
            description="First test",
            test_type=TestType.UNIT,
            function_under_test="func_one",
        )

        suite = TestSuite(
            name="my_module",
            file_path="test_my_module.py",
            framework=TestFramework.PYTEST,
            tests=[case],
            imports=["from my_module import func_one"],
            coverage_target=90.0,
        )

        data = suite.to_dict()

        assert data["name"] == "my_module"
        assert data["framework"] == "pytest"
        assert data["test_count"] == 1
        assert data["coverage_target"] == 90.0
        assert len(data["tests"]) == 1
        assert "generated_at" in data

    def test_to_code_pytest(self):
        """Should generate pytest code."""
        case = TestCase(
            name="test_add",
            description="Test addition",
            test_type=TestType.UNIT,
            function_under_test="add",
            inputs={"a": 1, "b": 2},
            expected_output=3,
        )

        suite = TestSuite(
            name="math_module",
            file_path="test_math.py",
            framework=TestFramework.PYTEST,
            tests=[case],
            imports=["from math_module import add"],
        )

        code = suite.to_code()

        assert "import pytest" in code
        assert "from math_module import add" in code
        assert "def test_test_add():" in code
        assert "Test addition" in code
        assert "a = 1" in code
        assert "b = 2" in code
        assert "result = add()" in code
        assert "assert result == 3" in code

    def test_to_code_pytest_with_error(self):
        """Should generate pytest code with error expectation."""
        case = TestCase(
            name="test_raises",
            description="Test error is raised",
            test_type=TestType.ERROR,
            function_under_test="divide_by_zero",
            expected_error="ZeroDivisionError",
        )

        suite = TestSuite(
            name="math_module",
            file_path="test_math.py",
            framework=TestFramework.PYTEST,
            tests=[case],
        )

        code = suite.to_code()

        assert "pytest.raises(ZeroDivisionError)" in code

    def test_to_code_pytest_with_setup_teardown(self):
        """Should generate pytest code with setup/teardown."""
        case = TestCase(
            name="test_db",
            description="Test with database",
            test_type=TestType.INTEGRATION,
            function_under_test="save_record",
            setup="db = connect_db()",
            teardown="db.close()",
        )

        suite = TestSuite(
            name="db_module",
            file_path="test_db.py",
            framework=TestFramework.PYTEST,
            tests=[case],
        )

        code = suite.to_code()

        assert "db = connect_db()" in code
        assert "db.close()" in code

    def test_to_code_jest(self):
        """Should generate Jest code."""
        case = TestCase(
            name="test_multiply",
            description="Test multiplication",
            test_type=TestType.UNIT,
            function_under_test="multiply",
            inputs={"x": 3, "y": 4},
            expected_output=12,
        )

        suite = TestSuite(
            name="math_utils",
            file_path="math.test.js",
            framework=TestFramework.JEST,
            tests=[case],
            imports=["import { multiply } from './math';"],
        )

        code = suite.to_code()

        assert "describe('math_utils'," in code
        assert "test('Test multiplication'," in code
        assert "const x = 3;" in code
        assert "const y = 4;" in code
        assert "expect(result).toEqual(12);" in code

    def test_to_code_jest_with_error(self):
        """Should generate Jest code with error expectation."""
        case = TestCase(
            name="test_throws",
            description="Test throws error",
            test_type=TestType.ERROR,
            function_under_test="throwError",
            expected_error="Error",
        )

        suite = TestSuite(
            name="error_utils",
            file_path="error.test.js",
            framework=TestFramework.JEST,
            tests=[case],
        )

        code = suite.to_code()

        assert "expect(() => throwError()).toThrow();" in code

    def test_to_code_unittest(self):
        """Should generate unittest code."""
        case = TestCase(
            name="test_subtract",
            description="Test subtraction",
            test_type=TestType.UNIT,
            function_under_test="subtract",
            inputs={"a": 10, "b": 3},
            expected_output=7,
        )

        suite = TestSuite(
            name="calc_module",
            file_path="test_calc.py",
            framework=TestFramework.UNITTEST,
            tests=[case],
        )

        code = suite.to_code()

        assert "import unittest" in code
        assert "class TestCalcModule(unittest.TestCase):" in code
        assert "def test_test_subtract(self):" in code
        assert "self.assertEqual(result, 7)" in code
        assert "if __name__ == '__main__':" in code

    def test_to_code_unittest_with_error(self):
        """Should generate unittest code with error expectation."""
        case = TestCase(
            name="test_value_error",
            description="Test ValueError is raised",
            test_type=TestType.ERROR,
            function_under_test="validate",
            expected_error="ValueError",
        )

        suite = TestSuite(
            name="validation",
            file_path="test_validation.py",
            framework=TestFramework.UNITTEST,
            tests=[case],
        )

        code = suite.to_code()

        assert "self.assertRaises(ValueError)" in code


# =============================================================================
# TestGenerator Tests
# =============================================================================


class TestTestGenerator:
    """Tests for TestGenerator class."""

    def test_init_defaults(self):
        """Should initialize with defaults."""
        gen = TestGenerator()

        assert gen.framework == TestFramework.PYTEST
        assert gen.coverage_target == 80.0

    def test_init_custom(self):
        """Should initialize with custom values."""
        gen = TestGenerator(
            framework=TestFramework.JEST,
            coverage_target=95.0,
        )

        assert gen.framework == TestFramework.JEST
        assert gen.coverage_target == 95.0

    def test_analyze_function_simple(self):
        """Should analyze a simple function."""
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "add")

        assert analysis["function_name"] == "add"
        assert len(analysis["parameters"]) == 2
        assert analysis["parameters"][0]["name"] == "a"
        assert analysis["parameters"][0]["type"] == "int"
        assert analysis["return_type"] == "int"
        assert analysis["branches"] == 0
        assert analysis["loops"] == 0

    def test_analyze_function_with_branches(self):
        """Should count branches."""
        code = """
def classify(x: int) -> str:
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "classify")

        assert analysis["branches"] == 2  # if and elif
        assert analysis["complexity"] >= 3

    def test_analyze_function_with_loops(self):
        """Should count loops."""
        code = """
def sum_list(items: list) -> int:
    total = 0
    for item in items:
        total += item
    return total
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "sum_list")

        assert analysis["loops"] == 1
        assert analysis["complexity"] >= 2

    def test_analyze_function_with_raises(self):
        """Should detect raised exceptions."""
        code = """
def validate(value: str) -> None:
    if not value:
        raise ValueError("Empty value")
    if len(value) > 100:
        raise ValueError("Too long")
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "validate")

        assert len(analysis["raises"]) == 2
        assert any("ValueError" in r for r in analysis["raises"])

    def test_analyze_function_with_calls(self):
        """Should detect function calls."""
        code = """
def process(data: dict) -> dict:
    validated = validate(data)
    transformed = transform(validated)
    return save(transformed)
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "process")

        assert "validate" in analysis["calls"]
        assert "transform" in analysis["calls"]
        assert "save" in analysis["calls"]

    def test_analyze_function_syntax_error(self):
        """Should handle syntax errors."""
        code = "def broken( -> int:"  # Invalid syntax
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "broken")

        assert "error" in analysis

    def test_analyze_async_function(self):
        """Should analyze async functions."""
        code = """
async def fetch_data(url: str) -> dict:
    return await http_get(url)
"""
        gen = TestGenerator()
        analysis = gen.analyze_function(code, "fetch_data")

        assert analysis["function_name"] == "fetch_data"
        assert analysis["parameters"][0]["name"] == "url"
        assert analysis["return_type"] == "dict"

    def test_generate_test_cases_happy_path(self):
        """Should generate happy path test."""
        gen = TestGenerator()
        analysis = {
            "function_name": "my_func",
            "parameters": [],
            "raises": [],
        }

        cases = gen.generate_test_cases("my_func", analysis)

        happy_path = [c for c in cases if "happy_path" in c.tags]
        assert len(happy_path) == 1
        assert happy_path[0].priority == 1

    def test_generate_test_cases_string_param(self):
        """Should generate string parameter tests."""
        gen = TestGenerator()
        analysis = {
            "function_name": "greet",
            "parameters": [{"name": "name", "type": "str"}],
            "raises": [],
        }

        cases = gen.generate_test_cases("greet", analysis)

        # Check for None test (tagged as "null" in the implementation)
        null_tests = [c for c in cases if "null" in c.tags]
        assert len(null_tests) == 1

        # Check for empty string test
        empty_tests = [c for c in cases if "empty" in c.tags]
        assert len(empty_tests) >= 1

        # Check for long string test
        large_tests = [c for c in cases if "large_input" in c.tags]
        assert len(large_tests) == 1

    def test_generate_test_cases_int_param(self):
        """Should generate integer parameter tests."""
        gen = TestGenerator()
        analysis = {
            "function_name": "square",
            "parameters": [{"name": "n", "type": "int"}],
            "raises": [],
        }

        cases = gen.generate_test_cases("square", analysis)

        # Check for zero test
        zero_tests = [c for c in cases if "zero" in c.tags]
        assert len(zero_tests) == 1

        # Check for negative test
        neg_tests = [c for c in cases if "negative" in c.tags]
        assert len(neg_tests) == 1

    def test_generate_test_cases_list_param(self):
        """Should generate list parameter tests."""
        gen = TestGenerator()
        analysis = {
            "function_name": "sum_items",
            "parameters": [{"name": "items", "type": "list"}],
            "raises": [],
        }

        cases = gen.generate_test_cases("sum_items", analysis)

        # Check for empty list test by tag
        empty_tests = [c for c in cases if "empty" in c.tags and c.inputs.get("items") == []]
        assert len(empty_tests) == 1

        # Check for boundary tests
        boundary_tests = [c for c in cases if "boundary" in c.tags]
        assert len(boundary_tests) >= 1

    def test_generate_test_cases_error_handling(self):
        """Should generate error handling tests."""
        gen = TestGenerator()
        analysis = {
            "function_name": "parse",
            "parameters": [],
            "raises": ["ValueError('Invalid input')", "TypeError"],
        }

        cases = gen.generate_test_cases("parse", analysis)

        error_tests = [c for c in cases if c.test_type == TestType.ERROR]
        assert len(error_tests) == 2

        # Check expected errors
        expected_errors = [c.expected_error for c in error_tests]
        assert "ValueError" in expected_errors
        assert "TypeError" in expected_errors

    def test_generate_test_suite(self):
        """Should generate complete test suite."""
        code = """
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b
"""
        gen = TestGenerator()
        suite = gen.generate_test_suite(code, "math_ops")

        assert suite.name == "math_ops"
        assert suite.file_path == "test_math_ops.py"
        assert suite.framework == TestFramework.PYTEST
        assert len(suite.tests) > 0

        # Should have tests for both functions
        funcs_tested = set(t.function_under_test for t in suite.tests)
        assert "add" in funcs_tested
        assert "subtract" in funcs_tested

    def test_generate_test_suite_specific_functions(self):
        """Should only generate tests for specified functions."""
        code = """
def func_a():
    pass

def func_b():
    pass

def func_c():
    pass
"""
        gen = TestGenerator()
        suite = gen.generate_test_suite(code, "multi_funcs", functions=["func_a", "func_c"])

        funcs_tested = set(t.function_under_test for t in suite.tests)
        assert "func_a" in funcs_tested
        assert "func_c" in funcs_tested
        assert "func_b" not in funcs_tested

    def test_extract_function_names(self):
        """Should extract public function names."""
        code = """
def public_func():
    pass

def _private_func():
    pass

async def async_public():
    pass
"""
        gen = TestGenerator()
        names = gen._extract_function_names(code)

        assert "public_func" in names
        assert "async_public" in names
        assert "_private_func" not in names

    def test_extract_function_names_syntax_error(self):
        """Should return empty list on syntax error."""
        code = "def broken("  # Invalid syntax
        gen = TestGenerator()
        names = gen._extract_function_names(code)

        assert names == []


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestToSnakeCase:
    """Tests for _to_snake_case function."""

    def test_simple(self):
        """Should handle simple strings."""
        assert _to_snake_case("hello") == "hello"
        assert _to_snake_case("Hello") == "hello"

    def test_camel_case(self):
        """Should convert camelCase."""
        assert _to_snake_case("camelCase") == "camel_case"
        assert _to_snake_case("myFunctionName") == "my_function_name"

    def test_pascal_case(self):
        """Should convert PascalCase."""
        assert _to_snake_case("PascalCase") == "pascal_case"
        assert _to_snake_case("MyClassName") == "my_class_name"

    def test_with_spaces(self):
        """Should handle spaces."""
        assert _to_snake_case("hello world") == "hello_world"
        # Multiple capitals after space creates extra underscores
        assert _to_snake_case("My Function Name") == "my__function__name"

    def test_abbreviations(self):
        """Should handle abbreviations."""
        # HTTP is treated as a single word, then Server
        assert _to_snake_case("HTTPServer") == "http_server"
        assert _to_snake_case("getURL") == "get_url"


class TestToPascalCase:
    """Tests for _to_pascal_case function."""

    def test_simple(self):
        """Should handle simple strings."""
        assert _to_pascal_case("hello") == "Hello"

    def test_snake_case(self):
        """Should convert snake_case."""
        assert _to_pascal_case("my_function") == "MyFunction"
        assert _to_pascal_case("some_long_name") == "SomeLongName"

    def test_with_spaces(self):
        """Should handle spaces."""
        assert _to_pascal_case("hello world") == "HelloWorld"

    def test_with_hyphens(self):
        """Should handle hyphens."""
        assert _to_pascal_case("my-component") == "MyComponent"


class TestJsRepr:
    """Tests for _js_repr function."""

    def test_none(self):
        """Should convert None to null."""
        assert _js_repr(None) == "null"

    def test_bool(self):
        """Should convert booleans."""
        assert _js_repr(True) == "true"
        assert _js_repr(False) == "false"

    def test_string(self):
        """Should quote strings."""
        assert _js_repr("hello") == '"hello"'
        assert _js_repr("") == '""'

    def test_numbers(self):
        """Should represent numbers."""
        assert _js_repr(42) == "42"
        assert _js_repr(3.14) == "3.14"

    def test_list(self):
        """Should convert lists to arrays."""
        assert _js_repr([1, 2, 3]) == "[1, 2, 3]"
        assert _js_repr(["a", "b"]) == '["a", "b"]'
        assert _js_repr([]) == "[]"

    def test_tuple(self):
        """Should convert tuples to arrays."""
        assert _js_repr((1, 2)) == "[1, 2]"

    def test_dict(self):
        """Should convert dicts to objects."""
        assert _js_repr({"a": 1}) == "{a: 1}"
        assert _js_repr({"x": "y"}) == '{x: "y"}'

    def test_nested(self):
        """Should handle nested structures."""
        result = _js_repr({"items": [1, 2], "active": True})
        assert "items: [1, 2]" in result
        assert "active: true" in result


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestGenerateTestsForFunction:
    """Tests for generate_tests_for_function convenience function."""

    def test_basic_generation(self):
        """Should generate tests for a function."""
        code = """
def double(x: int) -> int:
    return x * 2
"""
        cases, test_code = generate_tests_for_function(code, "double")

        assert len(cases) > 0
        assert any(c.function_under_test == "double" for c in cases)
        assert "import pytest" in test_code
        assert "def test_" in test_code

    def test_with_different_framework(self):
        """Should generate tests with specified framework."""
        code = """
def greet(name: str) -> str:
    return f"Hello, {name}!"
"""
        cases, test_code = generate_tests_for_function(
            code, "greet", framework=TestFramework.UNITTEST
        )

        assert "import unittest" in test_code
        assert "class Test" in test_code


class TestGenerateTestsForFile:
    """Tests for generate_tests_for_file convenience function."""

    def test_file_generation(self):
        """Should generate tests from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def multiply(a: int, b: int) -> int:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
""")
            temp_path = f.name

        try:
            suite = generate_tests_for_file(temp_path)

            assert suite.name == os.path.splitext(os.path.basename(temp_path))[0]
            assert len(suite.tests) > 0

            funcs_tested = set(t.function_under_test for t in suite.tests)
            assert "multiply" in funcs_tested
            assert "divide" in funcs_tested
        finally:
            os.unlink(temp_path)

    def test_file_generation_specific_functions(self):
        """Should only test specified functions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def func_one():
    pass

def func_two():
    pass
""")
            temp_path = f.name

        try:
            suite = generate_tests_for_file(temp_path, functions=["func_one"])

            funcs_tested = set(t.function_under_test for t in suite.tests)
            assert "func_one" in funcs_tested
            assert "func_two" not in funcs_tested
        finally:
            os.unlink(temp_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for test generator."""

    def test_full_workflow(self):
        """Test complete test generation workflow."""
        code = '''
def validate_email(email: str) -> bool:
    """Validate an email address."""
    if not email:
        raise ValueError("Email cannot be empty")
    if "@" not in email:
        return False
    if len(email) > 254:
        raise ValueError("Email too long")
    return True
'''
        gen = TestGenerator(framework=TestFramework.PYTEST, coverage_target=90.0)

        # Analyze
        analysis = gen.analyze_function(code, "validate_email")
        assert analysis["parameters"][0]["type"] == "str"
        assert len(analysis["raises"]) == 2

        # Generate test cases
        cases = gen.generate_test_cases("validate_email", analysis)
        assert len(cases) >= 5  # happy path + string tests + error tests

        # Generate suite
        suite = gen.generate_test_suite(code, "email_validator")
        assert suite.coverage_target == 90.0

        # Generate code
        test_code = suite.to_code()
        assert "def test_" in test_code
        assert "import pytest" in test_code
        assert "validate_email" in test_code

    def test_multiple_frameworks(self):
        """Test generation for multiple frameworks."""
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        for framework in [TestFramework.PYTEST, TestFramework.JEST, TestFramework.UNITTEST]:
            gen = TestGenerator(framework=framework)
            suite = gen.generate_test_suite(code, "math")
            test_code = suite.to_code()

            # Each framework should generate valid code
            assert len(test_code) > 0

            if framework == TestFramework.PYTEST:
                assert "import pytest" in test_code
            elif framework == TestFramework.JEST:
                assert "describe(" in test_code
            elif framework == TestFramework.UNITTEST:
                assert "import unittest" in test_code
