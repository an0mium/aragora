"""
Smart Test Generator Module.

Generates comprehensive test suites using multi-agent debate for:
- Unit tests
- Integration tests
- Edge case tests
- Property-based tests

Supports multiple test frameworks and languages.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TestFramework(str, Enum):
    """Supported test frameworks."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    VITEST = "vitest"
    RSPEC = "rspec"
    GO_TEST = "go_test"
    RUST_TEST = "rust_test"


class TestType(str, Enum):
    """Types of tests."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PROPERTY = "property"
    EDGE_CASE = "edge_case"
    ERROR = "error"
    BOUNDARY = "boundary"


@dataclass
class TestCase:
    """A single test case."""

    name: str
    description: str
    test_type: TestType
    function_under_test: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    expected_error: Optional[str] = None
    assertions: List[str] = field(default_factory=list)
    setup: Optional[str] = None
    teardown: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    priority: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "function_under_test": self.function_under_test,
            "inputs": self.inputs,
            "expected_output": self.expected_output,
            "expected_error": self.expected_error,
            "assertions": self.assertions,
            "setup": self.setup,
            "teardown": self.teardown,
            "tags": self.tags,
            "priority": self.priority,
        }


@dataclass
class TestSuite:
    """A collection of test cases."""

    name: str
    file_path: str
    framework: TestFramework
    tests: List[TestCase] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    coverage_target: float = 80.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "framework": self.framework.value,
            "tests": [t.to_dict() for t in self.tests],
            "imports": self.imports,
            "fixtures": self.fixtures,
            "generated_at": self.generated_at.isoformat(),
            "coverage_target": self.coverage_target,
            "test_count": len(self.tests),
        }

    def to_code(self) -> str:
        """Generate test code from the suite."""
        if self.framework == TestFramework.PYTEST:
            return self._to_pytest()
        elif self.framework == TestFramework.JEST:
            return self._to_jest()
        elif self.framework == TestFramework.UNITTEST:
            return self._to_unittest()
        else:
            return self._to_pytest()  # Default

    def _to_pytest(self) -> str:
        """Generate pytest code."""
        lines = [
            '"""',
            f"Auto-generated tests for {self.name}",
            f"Generated at: {self.generated_at.isoformat()}",
            '"""',
            "",
            "import pytest",
        ]

        # Add imports
        for imp in self.imports:
            lines.append(imp)

        lines.append("")

        # Add fixtures
        for fixture in self.fixtures:
            lines.append(fixture)
            lines.append("")

        # Add test functions
        for test in self.tests:
            # Docstring
            lines.append(f"def test_{_to_snake_case(test.name)}():")
            lines.append('    """')
            lines.append(f"    {test.description}")
            lines.append(f"    Test type: {test.test_type.value}")
            lines.append('    """')

            # Setup
            if test.setup:
                for setup_line in test.setup.split("\n"):
                    lines.append(f"    {setup_line}")

            # Test body
            if test.inputs:
                for param, value in test.inputs.items():
                    lines.append(f"    {param} = {repr(value)}")

            # Call function
            if test.expected_error:
                lines.append(f"    with pytest.raises({test.expected_error}):")
                lines.append(f"        {test.function_under_test}()")
            else:
                lines.append(f"    result = {test.function_under_test}()")

            # Assertions
            if test.expected_output is not None:
                lines.append(f"    assert result == {repr(test.expected_output)}")

            for assertion in test.assertions:
                lines.append(f"    assert {assertion}")

            # Teardown
            if test.teardown:
                for teardown_line in test.teardown.split("\n"):
                    lines.append(f"    {teardown_line}")

            lines.append("")

        return "\n".join(lines)

    def _to_jest(self) -> str:
        """Generate Jest code."""
        lines = [
            "/**",
            f" * Auto-generated tests for {self.name}",
            f" * Generated at: {self.generated_at.isoformat()}",
            " */",
            "",
        ]

        # Add imports
        for imp in self.imports:
            lines.append(imp)

        lines.append("")
        lines.append(f"describe('{self.name}', () => {{")

        # Add test functions
        for test in self.tests:
            lines.append(f"  test('{test.description}', () => {{")

            # Setup
            if test.setup:
                for setup_line in test.setup.split("\n"):
                    lines.append(f"    {setup_line}")

            # Test body
            if test.inputs:
                for param, value in test.inputs.items():
                    lines.append(f"    const {param} = {_js_repr(value)};")

            # Call function
            if test.expected_error:
                lines.append(f"    expect(() => {test.function_under_test}()).toThrow();")
            else:
                lines.append(f"    const result = {test.function_under_test}();")

            # Assertions
            if test.expected_output is not None:
                lines.append(f"    expect(result).toEqual({_js_repr(test.expected_output)});")

            for assertion in test.assertions:
                lines.append(f"    expect({assertion}).toBeTruthy();")

            lines.append("  });")
            lines.append("")

        lines.append("});")

        return "\n".join(lines)

    def _to_unittest(self) -> str:
        """Generate unittest code."""
        class_name = _to_pascal_case(self.name)

        lines = [
            '"""',
            f"Auto-generated tests for {self.name}",
            f"Generated at: {self.generated_at.isoformat()}",
            '"""',
            "",
            "import unittest",
        ]

        # Add imports
        for imp in self.imports:
            lines.append(imp)

        lines.append("")
        lines.append(f"class Test{class_name}(unittest.TestCase):")

        # Add test methods
        for test in self.tests:
            lines.append(f"    def test_{_to_snake_case(test.name)}(self):")
            lines.append('        """')
            lines.append(f"        {test.description}")
            lines.append('        """')

            # Setup
            if test.setup:
                for setup_line in test.setup.split("\n"):
                    lines.append(f"        {setup_line}")

            # Test body
            if test.inputs:
                for param, value in test.inputs.items():
                    lines.append(f"        {param} = {repr(value)}")

            # Call function
            if test.expected_error:
                lines.append(f"        with self.assertRaises({test.expected_error}):")
                lines.append(f"            {test.function_under_test}()")
            else:
                lines.append(f"        result = {test.function_under_test}()")

            # Assertions
            if test.expected_output is not None:
                lines.append(f"        self.assertEqual(result, {repr(test.expected_output)})")

            for assertion in test.assertions:
                lines.append(f"        self.assertTrue({assertion})")

            lines.append("")

        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    unittest.main()")

        return "\n".join(lines)


class TestGenerator:
    """
    Multi-agent test generator.

    Uses analysis and optionally debate to generate comprehensive tests.
    """

    def __init__(
        self,
        framework: TestFramework = TestFramework.PYTEST,
        coverage_target: float = 80.0,
    ) -> None:
        self.framework = framework
        self.coverage_target = coverage_target

    def analyze_function(self, code: str, function_name: str) -> Dict[str, Any]:
        """
        Analyze a function to understand its behavior for test generation.

        Returns analysis including:
        - Parameters and types
        - Return type
        - Branches and conditions
        - Edge cases
        - Error conditions
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"error": "Failed to parse code"}

        analysis: Dict[str, Any] = {
            "function_name": function_name,
            "parameters": [],
            "return_type": None,
            "branches": 0,
            "loops": 0,
            "raises": [],
            "calls": [],
            "complexity": 1,
        }

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    # Parameters
                    for arg in node.args.args:
                        param = {"name": arg.arg}
                        if arg.annotation:
                            param["type"] = ast.unparse(arg.annotation)
                        analysis["parameters"].append(param)

                    # Return annotation
                    if node.returns:
                        analysis["return_type"] = ast.unparse(node.returns)

                    # Analyze body
                    for child in ast.walk(node):
                        if isinstance(child, ast.If):
                            analysis["branches"] += 1
                            analysis["complexity"] += 1
                        elif isinstance(child, (ast.For, ast.While)):
                            analysis["loops"] += 1
                            analysis["complexity"] += 1
                        elif isinstance(child, ast.Raise):
                            if child.exc:
                                analysis["raises"].append(ast.unparse(child.exc))
                        elif isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                analysis["calls"].append(child.func.id)
                            elif isinstance(child.func, ast.Attribute):
                                analysis["calls"].append(child.func.attr)

        return analysis

    def generate_test_cases(
        self,
        function_name: str,
        analysis: Dict[str, Any],
        context: Optional[str] = None,
    ) -> List[TestCase]:
        """
        Generate test cases from function analysis.

        Generates:
        - Happy path tests
        - Edge case tests
        - Error handling tests
        - Boundary tests
        """
        tests = []

        # Happy path test
        tests.append(
            TestCase(
                name=f"{function_name}_happy_path",
                description=f"Test {function_name} with valid inputs",
                test_type=TestType.UNIT,
                function_under_test=function_name,
                tags=["happy_path"],
                priority=1,
            )
        )

        # Parameter-based tests
        for param in analysis.get("parameters", []):
            param_name = param.get("name", "")
            param_type = param.get("type", "")

            # None/null test
            tests.append(
                TestCase(
                    name=f"{function_name}_{param_name}_none",
                    description=f"Test {function_name} with {param_name}=None",
                    test_type=TestType.EDGE_CASE,
                    function_under_test=function_name,
                    inputs={param_name: None},
                    tags=["edge_case", "null"],
                    priority=2,
                )
            )

            # Type-specific tests
            if "str" in param_type.lower():
                tests.extend(
                    [
                        TestCase(
                            name=f"{function_name}_{param_name}_empty_string",
                            description=f"Test with empty {param_name}",
                            test_type=TestType.BOUNDARY,
                            function_under_test=function_name,
                            inputs={param_name: ""},
                            tags=["boundary", "empty"],
                        ),
                        TestCase(
                            name=f"{function_name}_{param_name}_long_string",
                            description=f"Test with very long {param_name}",
                            test_type=TestType.BOUNDARY,
                            function_under_test=function_name,
                            inputs={param_name: "x" * 10000},
                            tags=["boundary", "large_input"],
                        ),
                    ]
                )
            elif "int" in param_type.lower() or "float" in param_type.lower():
                tests.extend(
                    [
                        TestCase(
                            name=f"{function_name}_{param_name}_zero",
                            description=f"Test with {param_name}=0",
                            test_type=TestType.BOUNDARY,
                            function_under_test=function_name,
                            inputs={param_name: 0},
                            tags=["boundary", "zero"],
                        ),
                        TestCase(
                            name=f"{function_name}_{param_name}_negative",
                            description=f"Test with negative {param_name}",
                            test_type=TestType.BOUNDARY,
                            function_under_test=function_name,
                            inputs={param_name: -1},
                            tags=["boundary", "negative"],
                        ),
                    ]
                )
            elif "list" in param_type.lower() or "List" in param_type:
                tests.extend(
                    [
                        TestCase(
                            name=f"{function_name}_{param_name}_empty_list",
                            description=f"Test with empty {param_name}",
                            test_type=TestType.BOUNDARY,
                            function_under_test=function_name,
                            inputs={param_name: []},
                            tags=["boundary", "empty"],
                        ),
                        TestCase(
                            name=f"{function_name}_{param_name}_single_element",
                            description=f"Test with single element {param_name}",
                            test_type=TestType.BOUNDARY,
                            function_under_test=function_name,
                            inputs={param_name: ["item"]},
                            tags=["boundary"],
                        ),
                    ]
                )

        # Error handling tests
        for exc in analysis.get("raises", []):
            exc_type = exc.split("(")[0] if "(" in exc else exc
            tests.append(
                TestCase(
                    name=f"{function_name}_raises_{_to_snake_case(exc_type)}",
                    description=f"Test {function_name} raises {exc_type}",
                    test_type=TestType.ERROR,
                    function_under_test=function_name,
                    expected_error=exc_type,
                    tags=["error_handling"],
                    priority=2,
                )
            )

        return tests

    def generate_test_suite(
        self,
        code: str,
        module_name: str,
        functions: Optional[List[str]] = None,
    ) -> TestSuite:
        """
        Generate a complete test suite for a module.

        Args:
            code: Source code to test
            module_name: Name of the module
            functions: Specific functions to test (or all if None)

        Returns:
            TestSuite with generated tests
        """
        suite = TestSuite(
            name=module_name,
            file_path=f"test_{module_name}.py",
            framework=self.framework,
            imports=[f"from {module_name} import *"],
            coverage_target=self.coverage_target,
        )

        # Find all functions if not specified
        if functions is None:
            functions = self._extract_function_names(code)

        # Generate tests for each function
        for func_name in functions:
            analysis = self.analyze_function(code, func_name)
            if "error" not in analysis:
                test_cases = self.generate_test_cases(func_name, analysis)
                suite.tests.extend(test_cases)

        return suite

    def _extract_function_names(self, code: str) -> List[str]:
        """Extract all function names from code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):  # Skip private functions
                    names.append(node.name)
        return names


# Utility functions


def _to_snake_case(name: str) -> str:
    """Convert string to snake_case."""
    # Handle spaces
    name = name.replace(" ", "_")
    # Handle camelCase
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def _to_pascal_case(name: str) -> str:
    """Convert string to PascalCase."""
    words = re.split(r"[_\s-]", name)
    return "".join(word.capitalize() for word in words)


def _js_repr(value: Any) -> str:
    """Convert Python value to JavaScript representation."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (list, tuple)):
        items = ", ".join(_js_repr(v) for v in value)
        return f"[{items}]"
    elif isinstance(value, dict):
        items = ", ".join(f"{k}: {_js_repr(v)}" for k, v in value.items())
        return f"{{{items}}}"
    else:
        return str(value)


# Convenience functions


def generate_tests_for_function(
    code: str,
    function_name: str,
    framework: TestFramework = TestFramework.PYTEST,
) -> Tuple[List[TestCase], str]:
    """
    Generate tests for a single function.

    Returns (test_cases, test_code)
    """
    generator = TestGenerator(framework=framework)
    analysis = generator.analyze_function(code, function_name)
    test_cases = generator.generate_test_cases(function_name, analysis)

    suite = TestSuite(
        name=function_name,
        file_path=f"test_{function_name}.py",
        framework=framework,
        tests=test_cases,
    )

    return test_cases, suite.to_code()


def generate_tests_for_file(
    file_path: str,
    framework: TestFramework = TestFramework.PYTEST,
    functions: Optional[List[str]] = None,
) -> TestSuite:
    """
    Generate tests for a file.

    Args:
        file_path: Path to the source file
        framework: Test framework to use
        functions: Specific functions to test (or all if None)

    Returns:
        TestSuite with generated tests
    """
    import os

    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    module_name = os.path.splitext(os.path.basename(file_path))[0]

    generator = TestGenerator(framework=framework)
    return generator.generate_test_suite(code, module_name, functions)
