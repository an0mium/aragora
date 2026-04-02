"""Tests for ExecutionMode enum."""

from aragora.pipeline.execution_mode import ExecutionMode


def test_autonomous_value():
    assert ExecutionMode.AUTONOMOUS == "autonomous"


def test_interactive_value():
    assert ExecutionMode.INTERACTIVE == "interactive"


def test_enum_is_str_subclass():
    assert isinstance(ExecutionMode.AUTONOMOUS, str)


def test_default_comparison():
    mode = "autonomous"
    assert mode == ExecutionMode.AUTONOMOUS
