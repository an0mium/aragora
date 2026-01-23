"""
Tests for the test generator module.

Tests:
- Test case generation
- Function spec extraction
- Test code generation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.nomic.test_generator import (
    FunctionSpec,
    TestCase,
    TestCodeGenerator,
    TestGenerator,
    TestSuite,
    TestType,
    extract_function_specs,
    generate_tests_for_file,
)


class TestTestType:
    """Tests for TestType enum."""

    def test_type_values(self):
        """Test type value strings."""
        assert TestType.UNIT.value == "unit"
        assert TestType.INTEGRATION.value == "integration"
        assert TestType.EDGE_CASE.value == "edge_case"


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_create_test_case(self):
        """Test creating a test case."""
        tc = TestCase(
            name="test_add_numbers",
            description="Test adding two numbers",
            test_type=TestType.UNIT,
            function_under_test="math.add",
            input_values={"a": 1, "b": 2},
            expected_output=3,
        )
        assert tc.name == "test_add_numbers"
        assert tc.test_type == TestType.UNIT

    def test_test_case_to_dict(self):
        """Test serialization."""
        tc = TestCase(
            name="test_example",
            description="Example test",
            test_type=TestType.UNIT,
            function_under_test="module.func",
        )
        data = tc.to_dict()
        assert data["name"] == "test_example"
        assert data["test_type"] == "unit"


class TestFunctionSpec:
    """Tests for FunctionSpec dataclass."""

    def test_create_function_spec(self):
        """Test creating a function spec."""
        spec = FunctionSpec(
            name="calculate",
            module="math_utils",
            parameters=[{"name": "x", "type": "int"}],
            return_type="int",
            is_async=False,
        )
        assert spec.name == "calculate"
        assert spec.module == "math_utils"
        assert len(spec.parameters) == 1

    def test_function_spec_to_dict(self):
        """Test serialization."""
        spec = FunctionSpec(
            name="func",
            module="module",
            raises=["ValueError"],
        )
        data = spec.to_dict()
        assert data["name"] == "func"
        assert "ValueError" in data["raises"]


class TestTestSuite:
    """Tests for TestSuite dataclass."""

    def test_create_test_suite(self):
        """Test creating a test suite."""
        tc = TestCase(
            name="test_example",
            description="Example",
            test_type=TestType.UNIT,
            function_under_test="module.func",
        )
        suite = TestSuite(
            name="TestModule",
            description="Tests for module",
            test_cases=[tc],
        )
        assert suite.name == "TestModule"
        assert len(suite.test_cases) == 1

    def test_test_suite_to_dict(self):
        """Test serialization with coverage."""
        tc1 = TestCase(
            name="test_1",
            description="Test 1",
            test_type=TestType.UNIT,
            function_under_test="func",
        )
        tc2 = TestCase(
            name="test_2",
            description="Test 2",
            test_type=TestType.EDGE_CASE,
            function_under_test="func",
        )
        suite = TestSuite(
            name="TestModule",
            description="Tests",
            test_cases=[tc1, tc2],
        )
        data = suite.to_dict()
        assert data["coverage"]["unit"] == 1
        assert data["coverage"]["edge_case"] == 1


class TestTestGenerator:
    """Tests for TestGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a generator for testing."""
        return TestGenerator()

    @pytest.fixture
    def sample_spec(self):
        """Create a sample function spec."""
        return FunctionSpec(
            name="process_data",
            module="data_utils",
            parameters=[
                {"name": "data", "type": "list"},
                {"name": "limit", "type": "int"},
            ],
            return_type="dict",
            docstring="Process a list of data items.",
            raises=["ValueError"],
        )

    def test_generate_from_spec(self, generator, sample_spec):
        """Test generating tests from spec."""
        suite = generator.generate_from_spec(sample_spec)
        assert suite.name == "TestProcessData"
        assert len(suite.test_cases) > 0

    def test_generate_happy_path(self, generator, sample_spec):
        """Test happy path test generation."""
        suite = generator.generate_from_spec(sample_spec)
        happy_path = [tc for tc in suite.test_cases if "happy_path" in tc.tags]
        assert len(happy_path) == 1
        assert "basic" in happy_path[0].name

    def test_generate_edge_cases(self, generator, sample_spec):
        """Test edge case generation."""
        suite = generator.generate_from_spec(sample_spec)
        edge_cases = [tc for tc in suite.test_cases if tc.test_type == TestType.EDGE_CASE]
        assert len(edge_cases) > 0

    def test_generate_error_handling(self, generator, sample_spec):
        """Test error handling test generation."""
        suite = generator.generate_from_spec(sample_spec)
        error_tests = [tc for tc in suite.test_cases if tc.test_type == TestType.ERROR_HANDLING]
        assert len(error_tests) >= 1
        assert any("valueerror" in tc.name for tc in error_tests)

    def test_generate_without_edge_cases(self, generator, sample_spec):
        """Test generation without edge cases."""
        suite = generator.generate_from_spec(
            sample_spec,
            include_edge_cases=False,
        )
        edge_cases = [tc for tc in suite.test_cases if tc.test_type == TestType.EDGE_CASE]
        assert len(edge_cases) == 0

    def test_generate_imports(self, generator, sample_spec):
        """Test import generation."""
        suite = generator.generate_from_spec(sample_spec)
        assert "import pytest" in suite.imports
        assert any("from data_utils import" in imp for imp in suite.imports)


class TestTestCodeGenerator:
    """Tests for TestCodeGenerator."""

    @pytest.fixture
    def code_generator(self):
        """Create a code generator."""
        return TestCodeGenerator()

    def test_generate_suite(self, code_generator):
        """Test generating test code."""
        tc = TestCase(
            name="test_example",
            description="Example test",
            test_type=TestType.UNIT,
            function_under_test="module.func",
            input_values={"x": 1},
            assertions=["result == 1"],
        )
        suite = TestSuite(
            name="TestExample",
            description="Example test suite",
            test_cases=[tc],
            imports=["import pytest", "from module import func"],
        )
        code = code_generator.generate_suite(suite)
        assert "class TestExample:" in code
        assert "def test_example(self):" in code
        assert "result = func(x=1)" in code

    def test_generate_exception_test(self, code_generator):
        """Test generating exception test."""
        tc = TestCase(
            name="test_raises_error",
            description="Test exception",
            test_type=TestType.ERROR_HANDLING,
            function_under_test="module.func",
            expected_exception="ValueError",
        )
        suite = TestSuite(
            name="TestExample",
            description="Tests",
            test_cases=[tc],
            imports=["import pytest"],
        )
        code = code_generator.generate_suite(suite)
        assert "pytest.raises(ValueError)" in code


class TestExtractFunctionSpecs:
    """Tests for extract_function_specs."""

    def test_extract_simple_function(self):
        """Test extracting a simple function."""
        source = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
        specs = extract_function_specs(source, "greeting")
        assert len(specs) == 1
        assert specs[0].name == "hello"
        assert specs[0].module == "greeting"
        assert len(specs[0].parameters) == 1
        assert specs[0].return_type == "str"

    def test_extract_async_function(self):
        """Test extracting an async function."""
        source = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    pass
'''
        specs = extract_function_specs(source, "network")
        assert len(specs) == 1
        assert specs[0].is_async is True

    def test_skip_private_functions(self):
        """Test that private functions are skipped."""
        source = """
def public_func():
    pass

def _private_func():
    pass

def __dunder__():
    pass
"""
        specs = extract_function_specs(source, "module")
        names = [s.name for s in specs]
        assert "public_func" in names
        assert "_private_func" not in names
        assert "__dunder__" in names  # Dunder methods are kept

    def test_extract_raises(self):
        """Test extracting exceptions from docstring."""
        source = '''
def validate(data):
    """Validate data.

    Raises:
        ValueError: If data is invalid.
        TypeError: If data is wrong type.
    """
    pass
'''
        specs = extract_function_specs(source, "validator")
        assert len(specs) == 1
        assert "ValueError" in specs[0].raises
        assert "TypeError" in specs[0].raises

    def test_handle_syntax_error(self):
        """Test handling syntax errors gracefully."""
        source = "def broken("
        specs = extract_function_specs(source, "broken")
        assert len(specs) == 0


class TestGenerateTestsForFile:
    """Tests for generate_tests_for_file."""

    def test_generate_for_file(self, tmp_path):
        """Test generating tests for a file."""
        file_path = tmp_path / "utils.py"
        file_path.write_text('''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
''')
        suite = generate_tests_for_file(file_path)
        assert suite is not None
        assert suite.name == "TestUtils"
        assert len(suite.test_cases) >= 2

    def test_generate_for_nonexistent_file(self, tmp_path):
        """Test handling nonexistent files."""
        file_path = tmp_path / "does_not_exist.py"
        suite = generate_tests_for_file(file_path)
        assert suite is None

    def test_generate_for_empty_file(self, tmp_path):
        """Test handling empty files."""
        file_path = tmp_path / "empty.py"
        file_path.write_text("# Just a comment")
        suite = generate_tests_for_file(file_path)
        assert suite is None
