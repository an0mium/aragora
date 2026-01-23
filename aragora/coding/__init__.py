"""
Coding Assistance Module.

Provides tools for:
- Test generation
- Code generation
- Refactoring assistance
"""

from .test_generator import (
    TestGenerator,
    TestCase,
    TestSuite,
    TestFramework,
    generate_tests_for_function,
    generate_tests_for_file,
)

__all__ = [
    "TestGenerator",
    "TestCase",
    "TestSuite",
    "TestFramework",
    "generate_tests_for_function",
    "generate_tests_for_file",
]
