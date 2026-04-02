"""Tests for ExecutionMode enum."""

from types import SimpleNamespace

from aragora.pipeline.execution_mode import ExecutionMode, resolve_safety_mode


def test_autonomous_value():
    assert ExecutionMode.AUTONOMOUS == "autonomous"


def test_interactive_value():
    assert ExecutionMode.INTERACTIVE == "interactive"


def test_enum_is_str_subclass():
    assert isinstance(ExecutionMode.AUTONOMOUS, str)


def test_default_comparison():
    mode = "autonomous"
    assert mode == ExecutionMode.AUTONOMOUS


def test_resolve_safety_mode_prefers_explicit_value():
    assert resolve_safety_mode(ExecutionMode.INTERACTIVE) == ExecutionMode.INTERACTIVE


def test_resolve_safety_mode_infers_interactive_from_auth_context():
    auth_context = SimpleNamespace(user_id="user-1")
    assert resolve_safety_mode(None, auth_context=auth_context) == ExecutionMode.INTERACTIVE


def test_resolve_safety_mode_defaults_to_autonomous_without_context():
    assert resolve_safety_mode(None) == ExecutionMode.AUTONOMOUS
