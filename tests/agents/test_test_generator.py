"""
Tests for the Test Generator Agent module.

Tests cover:
- TestType enum values
- FunctionSignature dataclass and extraction
- TestSuggestion dataclass
- CoverageGap dataclass
- TestGeneratorAgent class
- Test generation for different parameter types
- Edge case analysis based on parameter types
- Pytest code generation
- Coverage gap analysis
- Error handling
- Convenience functions
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.agents.test_generator import (
    TestType,
    FunctionSignature,
    TestSuggestion,
    CoverageGap,
    TestGeneratorAgent,
    generate_tests_for_code,
    AGENT_CONFIGS,
)
from aragora.core import Critique


# =============================================================================
# Sample Code Constants for Testing
# =============================================================================

SAMPLE_SIMPLE_CODE = """
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""

SAMPLE_ASYNC_CODE = """
async def fetch_data(url: str) -> dict:
    \"\"\"Fetch data from URL.\"\"\"
    pass
"""

SAMPLE_CLASS_CODE = """
class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        \"\"\"Add two numbers.\"\"\"
        return round(a + b, self.precision)

    def _internal_method(self):
        pass
"""

SAMPLE_CODE_VARIOUS_TYPES = """
def process_strings(name: str, items: list[str]) -> str:
    return name

def calculate(value: int, ratio: float = 0.5) -> float:
    return value * ratio

def get_user(user_id: Optional[str] = None) -> dict:
    return {}
"""

SAMPLE_CODE_MULTILINE_DOCSTRING = """
def complex_function(data: dict) -> list:
    \"\"\"
    Process complex data.

    This function does many things.
    \"\"\"
    return []
"""

SAMPLE_CODE_WITH_DECORATORS = """
@property
def value(self) -> int:
    return self._value

@staticmethod
def create() -> 'MyClass':
    return MyClass()
"""

SAMPLE_CODE_NO_TYPES = """
def untyped_function(x, y):
    return x + y
"""


# =============================================================================
# TestType Enum Tests
# =============================================================================


class TestTestTypeEnum:
    """Tests for TestType enum."""

    def test_unit_value(self):
        """TestType includes unit type."""
        assert TestType.UNIT.value == "unit"

    def test_integration_value(self):
        """TestType includes integration type."""
        assert TestType.INTEGRATION.value == "integration"

    def test_e2e_value(self):
        """TestType includes e2e type."""
        assert TestType.E2E.value == "e2e"

    def test_property_value(self):
        """TestType includes property type."""
        assert TestType.PROPERTY.value == "property"

    def test_snapshot_value(self):
        """TestType includes snapshot type."""
        assert TestType.SNAPSHOT.value == "snapshot"

    def test_performance_value(self):
        """TestType includes performance type."""
        assert TestType.PERFORMANCE.value == "performance"

    def test_security_value(self):
        """TestType includes security type."""
        assert TestType.SECURITY.value == "security"

    def test_test_type_is_string_enum(self):
        """TestType is a string enum."""
        assert isinstance(TestType.UNIT, str)
        assert TestType.UNIT == "unit"


# =============================================================================
# FunctionSignature Dataclass Tests
# =============================================================================


class TestFunctionSignature:
    """Tests for FunctionSignature dataclass."""

    def test_minimal_init(self):
        """FunctionSignature can be initialized with minimal params."""
        sig = FunctionSignature(
            name="test_func",
            parameters=[],
            return_type=None,
            docstring=None,
        )

        assert sig.name == "test_func"
        assert sig.parameters == []
        assert sig.return_type is None
        assert sig.docstring is None

    def test_default_values(self):
        """FunctionSignature has correct default values."""
        sig = FunctionSignature(
            name="func",
            parameters=[],
            return_type=None,
            docstring=None,
        )

        assert sig.decorators == []
        assert sig.is_async is False
        assert sig.class_name is None

    def test_full_init(self):
        """FunctionSignature can be initialized with all params."""
        sig = FunctionSignature(
            name="process",
            parameters=[{"name": "data", "type": "str", "default": None}],
            return_type="dict",
            docstring="Process data.",
            decorators=["@staticmethod"],
            is_async=True,
            class_name="DataProcessor",
        )

        assert sig.name == "process"
        assert len(sig.parameters) == 1
        assert sig.return_type == "dict"
        assert sig.docstring == "Process data."
        assert sig.is_async is True
        assert sig.class_name == "DataProcessor"

    def test_full_name_without_class(self):
        """full_name returns function name when no class."""
        sig = FunctionSignature(
            name="standalone_func",
            parameters=[],
            return_type=None,
            docstring=None,
        )

        assert sig.full_name == "standalone_func"

    def test_full_name_with_class(self):
        """full_name returns class.function format when in class."""
        sig = FunctionSignature(
            name="method",
            parameters=[],
            return_type=None,
            docstring=None,
            class_name="MyClass",
        )

        assert sig.full_name == "MyClass.method"


# =============================================================================
# TestSuggestion Dataclass Tests
# =============================================================================


class TestTestSuggestion:
    """Tests for TestSuggestion dataclass."""

    def test_minimal_init(self):
        """TestSuggestion can be initialized with minimal params."""
        suggestion = TestSuggestion(
            function_name="test_func",
            test_name="test_test_func_happy_path",
            test_type=TestType.UNIT,
            description="Test happy path",
            input_values={"x": 1},
        )

        assert suggestion.function_name == "test_func"
        assert suggestion.test_type == TestType.UNIT

    def test_default_values(self):
        """TestSuggestion has correct default values."""
        suggestion = TestSuggestion(
            function_name="func",
            test_name="test_func",
            test_type=TestType.UNIT,
            description="Test",
            input_values={},
        )

        assert suggestion.expected_output is None
        assert suggestion.edge_case is False
        assert suggestion.priority == 1
        assert suggestion.rationale == ""

    def test_full_init(self):
        """TestSuggestion can be initialized with all params."""
        suggestion = TestSuggestion(
            function_name="calculate",
            test_name="test_calculate_zero",
            test_type=TestType.UNIT,
            description="Test with zero input",
            input_values={"value": 0},
            expected_output=0,
            edge_case=True,
            priority=2,
            rationale="Zero is a boundary condition",
        )

        assert suggestion.expected_output == 0
        assert suggestion.edge_case is True
        assert suggestion.priority == 2
        assert "boundary" in suggestion.rationale


# =============================================================================
# CoverageGap Dataclass Tests
# =============================================================================


class TestCoverageGap:
    """Tests for CoverageGap dataclass."""

    def test_minimal_init(self):
        """CoverageGap can be initialized with minimal params."""
        gap = CoverageGap(
            file_path="/test/module.py",
            function_name="untested_func",
            gap_type="no_tests",
            description="Function has no tests",
        )

        assert gap.file_path == "/test/module.py"
        assert gap.function_name == "untested_func"
        assert gap.gap_type == "no_tests"

    def test_default_values(self):
        """CoverageGap has correct default values."""
        gap = CoverageGap(
            file_path="/test.py",
            function_name="func",
            gap_type="missing_edge_cases",
            description="Missing edge cases",
        )

        assert gap.suggested_tests == []
        assert gap.priority == 1

    def test_full_init(self):
        """CoverageGap can be initialized with all params."""
        gap = CoverageGap(
            file_path="/src/utils.py",
            function_name="parse_data",
            gap_type="low_coverage",
            description="Only 50% coverage",
            suggested_tests=["test_parse_data_empty", "test_parse_data_invalid"],
            priority=2,
        )

        assert len(gap.suggested_tests) == 2
        assert gap.priority == 2


# =============================================================================
# TestGeneratorAgent Initialization Tests
# =============================================================================


class TestTestGeneratorAgentInit:
    """Tests for TestGeneratorAgent initialization."""

    def test_default_initialization(self):
        """Test default agent initialization."""
        agent = TestGeneratorAgent()

        assert agent.name == "test_generator"

    def test_has_system_prompt(self):
        """Agent has detailed system prompt."""
        agent = TestGeneratorAgent()

        assert "Test Generation Specialist" in agent._system_prompt
        assert "test suggestions" in agent._system_prompt.lower()

    def test_system_prompt_includes_response_format(self):
        """System prompt includes expected response format."""
        agent = TestGeneratorAgent()

        assert "TEST_NAME:" in agent._system_prompt
        assert "TEST_TYPE:" in agent._system_prompt
        assert "PRIORITY:" in agent._system_prompt


# =============================================================================
# TestGeneratorAgent generate and critique Tests
# =============================================================================


class TestTestGeneratorAgentMethods:
    """Tests for TestGeneratorAgent generate and critique methods."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    @pytest.mark.asyncio
    async def test_generate_returns_empty_string(self, agent):
        """generate method returns empty string (stub)."""
        result = await agent.generate("Generate tests for function")

        assert result == ""

    @pytest.mark.asyncio
    async def test_generate_with_context(self, agent):
        """generate method returns empty string even with context."""
        result = await agent.generate("Generate tests", context=["context"])

        assert result == ""

    @pytest.mark.asyncio
    async def test_critique_returns_critique_object(self, agent):
        """critique method returns a Critique object."""
        result = await agent.critique(
            proposal="def test(): pass",
            task="Generate tests",
            target_agent="developer",
        )

        assert isinstance(result, Critique)
        assert result.agent == "test_generator"

    @pytest.mark.asyncio
    async def test_critique_has_correct_message(self, agent):
        """critique method returns message about using assess_test_quality."""
        result = await agent.critique(
            proposal="test code",
            task="task",
            target_agent="target",
        )

        assert "assess_test_quality" in result.reasoning

    @pytest.mark.asyncio
    async def test_critique_default_target(self, agent):
        """critique method handles missing target_agent."""
        result = await agent.critique(
            proposal="test code",
            task="task",
        )

        assert result.target_agent == "unknown"

    @pytest.mark.asyncio
    async def test_critique_truncates_long_proposal(self, agent):
        """critique method truncates long proposals in target_content."""
        long_proposal = "x" * 500
        result = await agent.critique(
            proposal=long_proposal,
            task="task",
        )

        assert len(result.target_content) <= 200


# =============================================================================
# Function Signature Extraction Tests
# =============================================================================


class TestFunctionSignatureExtraction:
    """Tests for extract_function_signatures method."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_extract_simple_function(self, agent):
        """Extract signature from simple function."""
        signatures = agent.extract_function_signatures(SAMPLE_SIMPLE_CODE)

        assert len(signatures) == 1
        assert signatures[0].name == "add"
        assert len(signatures[0].parameters) == 2
        assert signatures[0].return_type == "int"

    def test_extract_async_function(self, agent):
        """Extract signature from async function."""
        signatures = agent.extract_function_signatures(SAMPLE_ASYNC_CODE)

        assert len(signatures) == 1
        assert signatures[0].is_async is True
        assert signatures[0].name == "fetch_data"

    def test_extract_class_methods(self, agent):
        """Extract signatures from class methods."""
        signatures = agent.extract_function_signatures(SAMPLE_CLASS_CODE)

        # Should include __init__ and add (public methods)
        names = [s.name for s in signatures]
        assert "__init__" in names
        assert "add" in names
        # Private method should be excluded
        assert "_internal_method" not in names

    def test_extract_method_with_class_name(self, agent):
        """Extracted methods include class name."""
        signatures = agent.extract_function_signatures(SAMPLE_CLASS_CODE)

        add_sig = next((s for s in signatures if s.name == "add"), None)
        assert add_sig is not None
        assert add_sig.class_name == "Calculator"
        assert add_sig.full_name == "Calculator.add"

    def test_extract_docstring(self, agent):
        """Extract docstring from function."""
        signatures = agent.extract_function_signatures(SAMPLE_SIMPLE_CODE)

        assert signatures[0].docstring is not None
        assert "Add two numbers" in signatures[0].docstring

    def test_extract_multiline_docstring(self, agent):
        """Extract multiline docstring."""
        signatures = agent.extract_function_signatures(SAMPLE_CODE_MULTILINE_DOCSTRING)

        assert len(signatures) == 1
        assert signatures[0].docstring is not None
        assert "complex data" in signatures[0].docstring.lower()

    def test_extract_parameters_with_types(self, agent):
        """Extract parameters with type hints."""
        signatures = agent.extract_function_signatures(SAMPLE_SIMPLE_CODE)

        params = signatures[0].parameters
        assert len(params) == 2
        assert params[0]["name"] == "a"
        assert params[0]["type"] == "int"

    def test_extract_parameters_with_defaults(self, agent):
        """Extract parameters with default values."""
        signatures = agent.extract_function_signatures(SAMPLE_CODE_VARIOUS_TYPES)

        calculate_sig = next((s for s in signatures if s.name == "calculate"), None)
        assert calculate_sig is not None

        ratio_param = next((p for p in calculate_sig.parameters if p["name"] == "ratio"), None)
        assert ratio_param is not None
        assert ratio_param["default"] == "0.5"

    def test_extract_untyped_parameters(self, agent):
        """Extract parameters without type hints."""
        signatures = agent.extract_function_signatures(SAMPLE_CODE_NO_TYPES)

        assert len(signatures) == 1
        params = signatures[0].parameters
        assert len(params) == 2
        # Untyped parameters should have type as None
        assert params[0]["type"] is None

    def test_extract_empty_code(self, agent):
        """Handle empty code string."""
        signatures = agent.extract_function_signatures("")

        assert signatures == []

    def test_extract_code_without_functions(self, agent):
        """Handle code with no functions."""
        code = "x = 1\ny = 2\n"
        signatures = agent.extract_function_signatures(code)

        assert signatures == []


# =============================================================================
# Parameter Parsing Tests
# =============================================================================


class TestParameterParsing:
    """Tests for _parse_parameters method."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_parse_empty_params(self, agent):
        """Parse empty parameter string."""
        params = agent._parse_parameters("")

        assert params == []

    def test_parse_whitespace_only(self, agent):
        """Parse whitespace-only parameter string."""
        params = agent._parse_parameters("   ")

        assert params == []

    def test_parse_self_param(self, agent):
        """Filter out self parameter."""
        params = agent._parse_parameters("self, x: int")

        assert len(params) == 1
        assert params[0]["name"] == "x"

    def test_parse_cls_param(self, agent):
        """Filter out cls parameter."""
        params = agent._parse_parameters("cls, value: str")

        assert len(params) == 1
        assert params[0]["name"] == "value"

    def test_parse_typed_param(self, agent):
        """Parse parameter with type annotation."""
        params = agent._parse_parameters("count: int")

        assert len(params) == 1
        assert params[0]["name"] == "count"
        assert params[0]["type"] == "int"
        assert params[0]["default"] is None

    def test_parse_param_with_default(self, agent):
        """Parse parameter with default value."""
        params = agent._parse_parameters("limit: int = 10")

        assert len(params) == 1
        assert params[0]["name"] == "limit"
        assert params[0]["type"] == "int"
        assert params[0]["default"] == "10"

    def test_parse_untyped_param_with_default(self, agent):
        """Parse untyped parameter with default."""
        params = agent._parse_parameters("flag=True")

        assert len(params) == 1
        assert params[0]["name"] == "flag"
        assert params[0]["type"] is None
        assert params[0]["default"] == "True"

    def test_parse_multiple_params(self, agent):
        """Parse multiple parameters."""
        params = agent._parse_parameters("a: int, b: str, c: float = 1.0")

        assert len(params) == 3
        assert params[0]["name"] == "a"
        assert params[1]["name"] == "b"
        assert params[2]["name"] == "c"


# =============================================================================
# Docstring Extraction Tests
# =============================================================================


class TestDocstringExtraction:
    """Tests for _extract_docstring method."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_extract_single_line_docstring(self, agent):
        """Extract single-line docstring."""
        lines = [
            "def func():",
            '    """Single line docstring."""',
            "    pass",
        ]
        docstring = agent._extract_docstring(lines, 0)

        assert docstring == "Single line docstring."

    def test_extract_multiline_docstring(self, agent):
        """Extract multi-line docstring."""
        lines = [
            "def func():",
            '    """',
            "    Multi-line docstring.",
            "    More content.",
            '    """',
            "    pass",
        ]
        docstring = agent._extract_docstring(lines, 0)

        assert docstring is not None
        assert "Multi-line" in docstring

    def test_no_docstring(self, agent):
        """Handle function without docstring."""
        lines = [
            "def func():",
            "    pass",
        ]
        docstring = agent._extract_docstring(lines, 0)

        assert docstring is None

    def test_docstring_out_of_range(self, agent):
        """Handle function at end of file."""
        lines = ["def func():"]
        docstring = agent._extract_docstring(lines, 0)

        assert docstring is None


# =============================================================================
# Decorator Extraction Tests
# =============================================================================


class TestDecoratorExtraction:
    """Tests for _extract_decorators method."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_extract_single_decorator(self, agent):
        """Extract single decorator."""
        lines = [
            "@property",
            "def value(self):",
            "    pass",
        ]
        decorators = agent._extract_decorators(lines, 1)

        assert decorators == ["@property"]

    def test_extract_multiple_decorators(self, agent):
        """Extract multiple decorators."""
        lines = [
            "@classmethod",
            "@cache",
            "def method(cls):",
            "    pass",
        ]
        decorators = agent._extract_decorators(lines, 2)

        assert len(decorators) == 2
        assert "@classmethod" in decorators
        assert "@cache" in decorators

    def test_no_decorators(self, agent):
        """Handle function without decorators."""
        lines = [
            "def func():",
            "    pass",
        ]
        decorators = agent._extract_decorators(lines, 0)

        assert decorators == []


# =============================================================================
# Test Suggestion Generation Tests
# =============================================================================


class TestSuggestTests:
    """Tests for suggest_tests method."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_happy_path_test_always_generated(self, agent):
        """Happy path test is always generated."""
        sig = FunctionSignature(
            name="add",
            parameters=[{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
            return_type="int",
            docstring=None,
        )

        suggestions = agent.suggest_tests(sig)

        happy_path = next((s for s in suggestions if "happy_path" in s.test_name), None)
        assert happy_path is not None
        assert happy_path.test_type == TestType.UNIT
        assert happy_path.priority == 1

    def test_invalid_input_test_always_generated(self, agent):
        """Invalid input test is always generated."""
        sig = FunctionSignature(
            name="process",
            parameters=[],
            return_type=None,
            docstring=None,
        )

        suggestions = agent.suggest_tests(sig)

        invalid_test = next((s for s in suggestions if "invalid" in s.test_name), None)
        assert invalid_test is not None
        assert invalid_test.priority == 2

    def test_string_edge_cases_generated(self, agent):
        """String parameters generate edge case tests."""
        sig = FunctionSignature(
            name="process",
            parameters=[{"name": "name", "type": "str"}],
            return_type="str",
            docstring=None,
        )

        suggestions = agent.suggest_tests(sig)

        test_names = [s.test_name for s in suggestions]
        assert any("empty_string" in name for name in test_names)
        assert any("whitespace" in name for name in test_names)

    def test_numeric_edge_cases_generated(self, agent):
        """Numeric parameters generate edge case tests."""
        sig = FunctionSignature(
            name="calculate",
            parameters=[{"name": "value", "type": "int"}],
            return_type="float",
            docstring=None,
        )

        suggestions = agent.suggest_tests(sig)

        test_names = [s.test_name for s in suggestions]
        assert any("zero" in name for name in test_names)
        assert any("negative" in name for name in test_names)

    def test_list_edge_cases_generated(self, agent):
        """List parameters generate edge case tests."""
        sig = FunctionSignature(
            name="process_items",
            parameters=[{"name": "items", "type": "list[str]"}],
            return_type="list",
            docstring=None,
        )

        suggestions = agent.suggest_tests(sig)

        test_names = [s.test_name for s in suggestions]
        assert any("empty_list" in name for name in test_names)

    def test_optional_edge_cases_generated(self, agent):
        """Optional parameters generate None edge case tests."""
        sig = FunctionSignature(
            name="get_user",
            parameters=[{"name": "user_id", "type": "Optional[str]", "default": "None"}],
            return_type="dict",
            docstring=None,
        )

        suggestions = agent.suggest_tests(sig)

        test_names = [s.test_name for s in suggestions]
        assert any("none" in name for name in test_names)

    def test_async_function_concurrent_test(self, agent):
        """Async functions generate concurrency tests."""
        sig = FunctionSignature(
            name="fetch_data",
            parameters=[{"name": "url", "type": "str"}],
            return_type="dict",
            docstring=None,
            is_async=True,
        )

        suggestions = agent.suggest_tests(sig)

        concurrent_test = next((s for s in suggestions if "concurrent" in s.test_name), None)
        assert concurrent_test is not None
        assert concurrent_test.test_type == TestType.INTEGRATION

    def test_edge_cases_marked_as_edge_case(self, agent):
        """Edge case tests have edge_case=True."""
        sig = FunctionSignature(
            name="process",
            parameters=[{"name": "value", "type": "int"}],
            return_type="int",
            docstring=None,
        )

        suggestions = agent.suggest_tests(sig)

        zero_test = next((s for s in suggestions if "zero" in s.test_name), None)
        assert zero_test is not None
        assert zero_test.edge_case is True


# =============================================================================
# Sample Input Generation Tests
# =============================================================================


class TestGenerateSampleInputs:
    """Tests for _generate_sample_inputs method."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_string_type_generates_test_value(self, agent):
        """String type generates test_value."""
        params = [{"name": "text", "type": "str"}]
        inputs = agent._generate_sample_inputs(params)

        assert inputs["text"] == "test_value"

    def test_int_type_generates_one(self, agent):
        """Int type generates 1."""
        params = [{"name": "count", "type": "int"}]
        inputs = agent._generate_sample_inputs(params)

        assert inputs["count"] == 1

    def test_float_type_generates_one_point_zero(self, agent):
        """Float type generates 1.0."""
        params = [{"name": "rate", "type": "float"}]
        inputs = agent._generate_sample_inputs(params)

        assert inputs["rate"] == 1.0

    def test_bool_type_generates_true(self, agent):
        """Bool type generates True."""
        params = [{"name": "flag", "type": "bool"}]
        inputs = agent._generate_sample_inputs(params)

        assert inputs["flag"] is True

    def test_list_type_generates_empty_list(self, agent):
        """List type generates empty list."""
        params = [{"name": "items", "type": "list"}]
        inputs = agent._generate_sample_inputs(params)

        assert inputs["items"] == []

    def test_dict_type_generates_empty_dict(self, agent):
        """Dict type generates empty dict."""
        params = [{"name": "data", "type": "dict"}]
        inputs = agent._generate_sample_inputs(params)

        assert inputs["data"] == {}

    def test_default_value_used(self, agent):
        """Default value is used when not None."""
        params = [{"name": "limit", "type": "int", "default": "100"}]
        inputs = agent._generate_sample_inputs(params)

        assert inputs["limit"] == "100"

    def test_unknown_type_generates_none(self, agent):
        """Unknown type generates None."""
        params = [{"name": "obj", "type": "CustomClass"}]
        inputs = agent._generate_sample_inputs(params)

        assert inputs["obj"] is None

    def test_empty_params(self, agent):
        """Empty params generates empty dict."""
        inputs = agent._generate_sample_inputs([])

        assert inputs == {}


# =============================================================================
# Pytest Code Generation Tests
# =============================================================================


class TestGeneratePytestCode:
    """Tests for generate_pytest_code method."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_generates_import_statements(self, agent):
        """Generated code includes import statements."""
        suggestions = [
            TestSuggestion(
                function_name="add",
                test_name="test_add_happy_path",
                test_type=TestType.UNIT,
                description="Test happy path",
                input_values={"a": 1, "b": 2},
            )
        ]

        code = agent.generate_pytest_code(suggestions, "mymodule")

        assert "import pytest" in code
        assert "from mymodule import *" in code

    def test_generates_test_function(self, agent):
        """Generated code includes test function."""
        suggestions = [
            TestSuggestion(
                function_name="add",
                test_name="test_add_happy_path",
                test_type=TestType.UNIT,
                description="Test happy path",
                input_values={"a": 1, "b": 2},
            )
        ]

        code = agent.generate_pytest_code(suggestions)

        assert "def test_add_happy_path():" in code

    def test_includes_docstring(self, agent):
        """Generated code includes test docstring."""
        suggestions = [
            TestSuggestion(
                function_name="add",
                test_name="test_add",
                test_type=TestType.UNIT,
                description="Test addition function",
                input_values={},
            )
        ]

        code = agent.generate_pytest_code(suggestions)

        assert "Test addition function" in code

    def test_edge_case_noted_in_docstring(self, agent):
        """Edge case tests note in docstring."""
        suggestions = [
            TestSuggestion(
                function_name="process",
                test_name="test_process_empty",
                test_type=TestType.UNIT,
                description="Test empty input",
                input_values={"data": ""},
                edge_case=True,
            )
        ]

        code = agent.generate_pytest_code(suggestions)

        assert "Edge case test" in code

    def test_arrange_section(self, agent):
        """Generated code includes Arrange section."""
        suggestions = [
            TestSuggestion(
                function_name="func",
                test_name="test_func",
                test_type=TestType.UNIT,
                description="Test",
                input_values={"x": 1, "y": "hello"},
            )
        ]

        code = agent.generate_pytest_code(suggestions)

        assert "# Arrange" in code
        assert "x = 1" in code
        assert 'y = "hello"' in code

    def test_act_section(self, agent):
        """Generated code includes Act section."""
        suggestions = [
            TestSuggestion(
                function_name="calculate",
                test_name="test_calculate",
                test_type=TestType.UNIT,
                description="Test",
                input_values={"value": 10},
            )
        ]

        code = agent.generate_pytest_code(suggestions)

        assert "# Act" in code
        assert "result = calculate(" in code

    def test_assert_section_with_expected(self, agent):
        """Generated code includes Assert with expected value."""
        suggestions = [
            TestSuggestion(
                function_name="add",
                test_name="test_add",
                test_type=TestType.UNIT,
                description="Test",
                input_values={"a": 1, "b": 2},
                expected_output=3,
            )
        ]

        code = agent.generate_pytest_code(suggestions)

        assert "# Assert" in code
        assert "assert result == 3" in code

    def test_assert_section_without_expected(self, agent):
        """Generated code includes TODO assertion when no expected value."""
        suggestions = [
            TestSuggestion(
                function_name="process",
                test_name="test_process",
                test_type=TestType.UNIT,
                description="Test",
                input_values={"data": {}},
            )
        ]

        code = agent.generate_pytest_code(suggestions)

        assert "assert result is not None" in code
        assert "# TODO:" in code

    def test_multiple_suggestions(self, agent):
        """Generated code handles multiple suggestions."""
        suggestions = [
            TestSuggestion(
                function_name="func",
                test_name="test_func_1",
                test_type=TestType.UNIT,
                description="Test 1",
                input_values={},
            ),
            TestSuggestion(
                function_name="func",
                test_name="test_func_2",
                test_type=TestType.UNIT,
                description="Test 2",
                input_values={},
            ),
        ]

        code = agent.generate_pytest_code(suggestions)

        assert "def test_func_1():" in code
        assert "def test_func_2():" in code


# =============================================================================
# Coverage Gap Analysis Tests
# =============================================================================


class TestAnalyzeCoverageGaps:
    """Tests for analyze_coverage_gaps method."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_detects_no_tests(self, agent):
        """Detect functions with no tests."""
        code = """
def untested_function():
    pass
"""
        existing_tests = ""

        gaps = agent.analyze_coverage_gaps(code, existing_tests, "/src/module.py")

        assert len(gaps) >= 1
        assert gaps[0].gap_type == "no_tests"
        assert "untested_function" in gaps[0].function_name

    def test_includes_suggested_tests(self, agent):
        """Gap includes suggested test names."""
        code = """
def my_func(x: int) -> int:
    return x
"""
        existing_tests = ""

        gaps = agent.analyze_coverage_gaps(code, existing_tests, "/src/module.py")

        no_tests_gap = next((g for g in gaps if g.gap_type == "no_tests"), None)
        assert no_tests_gap is not None
        assert len(no_tests_gap.suggested_tests) > 0

    def test_function_with_tests_not_flagged(self, agent):
        """Functions with tests are not flagged as no_tests."""
        code = """
def add(a, b):
    return a + b
"""
        existing_tests = """
def test_add():
    assert add(1, 2) == 3
"""

        gaps = agent.analyze_coverage_gaps(code, existing_tests, "/src/module.py")

        no_tests_gaps = [g for g in gaps if g.gap_type == "no_tests"]
        assert len(no_tests_gaps) == 0

    def test_detects_missing_edge_cases(self, agent):
        """Detect functions with tests but missing edge cases."""
        code = """
def process(value):
    return value
"""
        existing_tests = """
def test_process():
    assert process(1) == 1
"""

        gaps = agent.analyze_coverage_gaps(code, existing_tests, "/src/module.py")

        edge_case_gap = next((g for g in gaps if g.gap_type == "missing_edge_cases"), None)
        assert edge_case_gap is not None
        assert edge_case_gap.priority == 2

    def test_file_path_in_gap(self, agent):
        """Gap includes correct file path."""
        code = "def func(): pass"
        existing_tests = ""

        gaps = agent.analyze_coverage_gaps(code, existing_tests, "/custom/path.py")

        assert gaps[0].file_path == "/custom/path.py"

    def test_empty_code(self, agent):
        """Handle empty code."""
        gaps = agent.analyze_coverage_gaps("", "", "/test.py")

        assert gaps == []


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestGenerateTestsForCode:
    """Tests for generate_tests_for_code convenience function."""

    def test_generates_tests_for_simple_code(self):
        """Generate tests for simple code."""
        code = """
def multiply(a: int, b: int) -> int:
    return a * b
"""
        result = generate_tests_for_code(code, "math_module")

        assert "import pytest" in result
        assert "from math_module import *" in result
        assert "test_multiply" in result

    def test_handles_empty_code(self):
        """Handle empty code string."""
        result = generate_tests_for_code("", "empty_module")

        # Should still have imports but no test functions
        assert "import pytest" in result
        assert "def test_" not in result

    def test_uses_default_module_name(self):
        """Use default module name when not provided."""
        result = generate_tests_for_code("def func(): pass")

        assert "from module import *" in result


# =============================================================================
# Agent Configuration Tests
# =============================================================================


class TestAgentConfigs:
    """Tests for AGENT_CONFIGS constant."""

    def test_test_generator_config_exists(self):
        """test_generator config exists."""
        assert "test_generator" in AGENT_CONFIGS

    def test_config_has_class(self):
        """Config has class field."""
        assert AGENT_CONFIGS["test_generator"]["class"] == "TestGeneratorAgent"

    def test_config_has_description(self):
        """Config has description."""
        assert "test cases" in AGENT_CONFIGS["test_generator"]["description"].lower()

    def test_config_has_capabilities(self):
        """Config has capabilities list."""
        capabilities = AGENT_CONFIGS["test_generator"]["capabilities"]
        assert "unit_test_generation" in capabilities
        assert "edge_case_identification" in capabilities
        assert "coverage_gap_analysis" in capabilities


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Edge case tests for TestGeneratorAgent."""

    @pytest.fixture
    def agent(self):
        """Create a TestGeneratorAgent for testing."""
        return TestGeneratorAgent()

    def test_function_with_no_params(self, agent):
        """Handle function with no parameters."""
        code = """
def get_timestamp() -> float:
    pass
"""
        signatures = agent.extract_function_signatures(code)

        assert len(signatures) == 1
        assert signatures[0].parameters == []

        suggestions = agent.suggest_tests(signatures[0])
        assert len(suggestions) >= 2  # happy path + invalid input

    def test_function_with_complex_return_type(self, agent):
        """Handle function with complex return type."""
        code = """
def get_items() -> list[dict[str, Any]]:
    pass
"""
        signatures = agent.extract_function_signatures(code)

        assert len(signatures) == 1
        assert "list[dict" in signatures[0].return_type

    def test_nested_class_methods(self, agent):
        """Handle nested class methods."""
        code = """
class Outer:
    class Inner:
        def method(self):
            pass
"""
        signatures = agent.extract_function_signatures(code)

        # Should find the method
        assert any(s.name == "method" for s in signatures)

    def test_lambda_not_extracted(self, agent):
        """Lambda expressions are not extracted as functions."""
        code = """
process = lambda x: x * 2
"""
        signatures = agent.extract_function_signatures(code)

        assert len(signatures) == 0

    def test_special_methods_filtered(self, agent):
        """Private dunder methods are filtered."""
        code = """
class MyClass:
    def __str__(self):
        return "MyClass"

    def __repr__(self):
        return "MyClass()"
"""
        signatures = agent.extract_function_signatures(code)

        names = [s.name for s in signatures]
        assert "__str__" not in names
        assert "__repr__" not in names

    def test_init_and_call_included(self, agent):
        """__init__ and __call__ are included."""
        code = """
class Callable:
    def __init__(self):
        pass

    def __call__(self):
        pass
"""
        signatures = agent.extract_function_signatures(code)

        names = [s.name for s in signatures]
        assert "__init__" in names
        assert "__call__" in names

    def test_generate_pytest_code_empty_suggestions(self, agent):
        """Handle empty suggestions list."""
        code = agent.generate_pytest_code([])

        assert "import pytest" in code
        assert "def test_" not in code

    def test_parameter_with_complex_type_annotation(self, agent):
        """Handle parameters with complex type annotations."""
        code = """
def process(data: dict[str, list[tuple[int, str]]]) -> None:
    pass
"""
        signatures = agent.extract_function_signatures(code)

        assert len(signatures) == 1
        assert len(signatures[0].parameters) == 1

    def test_function_signature_full_name_edge_case(self):
        """full_name handles edge cases."""
        # Empty class name
        sig = FunctionSignature(
            name="func",
            parameters=[],
            return_type=None,
            docstring=None,
            class_name="",
        )
        # Empty string is falsy, so full_name should return just the function name
        assert sig.full_name == "func"
