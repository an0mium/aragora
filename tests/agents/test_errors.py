"""
Unit tests for aragora/agents/errors.py (B0-cohort issue #5186).

Covers the backward-compatibility shim surface:
- Every name in ``__all__`` is importable from ``aragora.agents.errors``
- The literal ``aragora/agents/errors.py`` shim file loads and stays in sync
- Exception hierarchy relationships (isinstance chains)
- Exception attributes, recoverability semantics, and __str__ formatting
- Error classification re-exports (ErrorClassifier, classify_cli_error)
- Handler utilities (handle_agent_operation, AgentErrorHandler,
  make_fallback_message, _build_error_action)
- Retry delay calculation with jitter

No external services are used; all operations are local and synchronous or
driven by asyncio's test event loop (asyncio_mode = "auto").
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

import aragora.agents.errors as errors_pkg
from aragora.agents.errors import (
    AgentAPIError,
    AgentCircuitOpenError,
    AgentConnectionError,
    AgentError,
    AgentErrorHandler,
    AgentRateLimitError,
    AgentResponseError,
    AgentStreamError,
    AgentTimeoutError,
    CLIAgentError,
    CLINotFoundError,
    CLIParseError,
    CLISubprocessError,
    CLITimeoutError,
    ClassifiedError,
    ErrorCategory,
    ErrorClassifier,
    ErrorSeverity,
    RecoveryAction,
    calculate_retry_delay_with_jitter,
    classify_cli_error,
    handle_agent_operation,
    make_fallback_message,
    sanitize_error,
)
from aragora.agents.errors.decorators import _calculate_retry_delay_with_jitter
from aragora.agents.errors.handlers import _build_error_action
from aragora.exceptions import AragoraError

SHIM_PATH = Path(__file__).resolve().parents[2] / "aragora" / "agents" / "errors.py"


@pytest.fixture
def base_error() -> AgentError:
    """An AgentError with all optional attributes populated."""
    return AgentError(
        "generation failed",
        agent_name="claude",
        cause=ValueError("bad value"),
        recoverable=False,
    )


@pytest.fixture
def shim_module():
    """Load the literal errors.py shim file (normally shadowed by the package)."""
    spec = importlib.util.spec_from_file_location("aragora_agents_errors_shim", SHIM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Shim / re-export surface
# =============================================================================


class TestShimSurface:
    def test_package_all_names_resolve(self):
        """Every name in the package __all__ is importable."""
        for name in errors_pkg.__all__:
            assert getattr(errors_pkg, name, None) is not None, f"missing export: {name}"

    def test_shim_file_loads_and_all_names_resolve(self, shim_module):
        """The literal errors.py shim executes and exposes its full __all__."""
        for name in shim_module.__all__:
            assert hasattr(shim_module, name), f"shim missing export: {name}"

    def test_shim_exports_are_same_objects_as_package(self, shim_module):
        """Shim re-exports must be the same objects as the package exports."""
        for name in ("AgentError", "ErrorClassifier", "classify_cli_error", "sanitize_error"):
            assert getattr(shim_module, name) is getattr(errors_pkg, name)

    def test_shim_all_is_subset_of_package_surface(self, shim_module):
        """Every shim export name must also resolve on the canonical package."""
        missing = [n for n in shim_module.__all__ if not hasattr(errors_pkg, n)]
        assert missing == []

    def test_pattern_constants_are_nonempty_string_tuples(self):
        for name in (
            "RATE_LIMIT_PATTERNS",
            "NETWORK_ERROR_PATTERNS",
            "AUTH_ERROR_PATTERNS",
            "VALIDATION_ERROR_PATTERNS",
            "MODEL_ERROR_PATTERNS",
            "CONTENT_POLICY_PATTERNS",
            "CLI_ERROR_PATTERNS",
            "ALL_FALLBACK_PATTERNS",
        ):
            patterns = getattr(errors_pkg, name)
            assert isinstance(patterns, tuple) and patterns, name
            assert all(isinstance(p, str) and p for p in patterns), name

    def test_sanitize_error_is_callable_reexport(self):
        result = sanitize_error("connection failed for key sk-abc")
        assert isinstance(result, str)


# =============================================================================
# Exception hierarchy
# =============================================================================


class TestExceptionHierarchy:
    API_ERRORS = [
        AgentConnectionError,
        AgentTimeoutError,
        AgentRateLimitError,
        AgentAPIError,
        AgentResponseError,
        AgentStreamError,
        AgentCircuitOpenError,
    ]
    CLI_ERRORS = [CLIParseError, CLITimeoutError, CLISubprocessError, CLINotFoundError]

    def test_agent_error_inherits_aragora_error(self):
        err = AgentError("boom")
        assert isinstance(err, AragoraError)
        assert isinstance(err, Exception)

    @pytest.mark.parametrize("exc_cls", API_ERRORS)
    def test_api_errors_are_agent_errors(self, exc_cls):
        assert issubclass(exc_cls, AgentError)

    @pytest.mark.parametrize("exc_cls", CLI_ERRORS)
    def test_cli_errors_chain_through_cli_agent_error(self, exc_cls):
        assert issubclass(exc_cls, CLIAgentError)
        assert issubclass(exc_cls, AgentError)
        assert issubclass(exc_cls, AragoraError)

    def test_cli_agent_error_is_not_api_error_sibling(self):
        """CLI errors must not accidentally inherit from API-specific types."""
        assert not issubclass(CLIAgentError, AgentAPIError)
        assert not issubclass(AgentAPIError, CLIAgentError)

    def test_catching_agent_error_catches_all_subtypes(self):
        with pytest.raises(AgentError):
            raise CLINotFoundError("missing tool")
        with pytest.raises(AgentError):
            raise AgentRateLimitError("slow down")


# =============================================================================
# Exception attributes and formatting
# =============================================================================


class TestAgentErrorAttributes:
    def test_defaults(self):
        err = AgentError("plain failure")
        assert err.agent_name is None
        assert err.cause is None
        assert err.recoverable is True

    def test_populated_attributes_and_details(self, base_error):
        assert base_error.agent_name == "claude"
        assert isinstance(base_error.cause, ValueError)
        assert base_error.recoverable is False
        assert base_error.details["agent_name"] == "claude"
        assert base_error.details["cause_type"] == "ValueError"
        assert base_error.details["recoverable"] is False

    def test_str_plain_message(self):
        assert str(AgentError("plain failure")) == "plain failure"

    def test_str_with_agent_name_prefix(self):
        err = AgentError("failed", agent_name="gpt")
        assert str(err) == "[gpt] failed"

    def test_str_with_cause_suffix(self, base_error):
        text = str(base_error)
        assert text.startswith("[claude] generation failed")
        assert "(caused by: ValueError: bad value)" in text


class TestSpecificExceptionAttributes:
    def test_connection_error_status_code(self):
        err = AgentConnectionError("conn refused", agent_name="a", status_code=503)
        assert err.status_code == 503
        assert err.recoverable is True

    def test_timeout_error_fields(self):
        err = AgentTimeoutError("timed out", timeout_seconds=30.0, partial_content="partial")
        assert err.timeout_seconds == 30.0
        assert err.partial_content == "partial"
        assert err.recoverable is True

    def test_rate_limit_error_retry_after(self):
        err = AgentRateLimitError("429", retry_after=12.5)
        assert err.retry_after == 12.5
        assert err.recoverable is True

    @pytest.mark.parametrize(
        "status_code,expected_recoverable",
        [(None, True), (400, False), (401, False), (404, False), (500, True), (503, True)],
    )
    def test_api_error_recoverability_by_status(self, status_code, expected_recoverable):
        err = AgentAPIError("api failed", status_code=status_code)
        assert err.recoverable is expected_recoverable
        assert err.status_code == status_code

    def test_api_error_error_type(self):
        err = AgentAPIError("bad auth", status_code=401, error_type="authentication_error")
        assert err.error_type == "authentication_error"

    def test_response_error_not_recoverable(self):
        err = AgentResponseError("bad json", response_data={"raw": "x"})
        assert err.recoverable is False
        assert err.response_data == {"raw": "x"}

    def test_stream_error_partial_content(self):
        err = AgentStreamError("stream cut", partial_content="half an answer")
        assert err.partial_content == "half an answer"
        assert err.recoverable is True

    def test_circuit_open_error_cooldown(self):
        err = AgentCircuitOpenError("circuit open", cooldown_seconds=60.0)
        assert err.cooldown_seconds == 60.0
        assert err.recoverable is True


class TestCLIExceptionAttributes:
    def test_cli_agent_error_fields(self):
        err = CLIAgentError("cli failed", returncode=2, stderr="boom", recoverable=False)
        assert err.returncode == 2
        assert err.stderr == "boom"
        assert err.recoverable is False

    def test_parse_error_not_recoverable_with_raw_output(self):
        err = CLIParseError("bad output", raw_output="not-json")
        assert err.recoverable is False
        assert err.raw_output == "not-json"

    def test_timeout_error_returncode_sigkill(self):
        err = CLITimeoutError("timed out", timeout_seconds=15.0)
        assert err.returncode == -9
        assert err.timeout_seconds == 15.0
        assert err.recoverable is True

    def test_not_found_error_returncode_127_not_recoverable(self):
        err = CLINotFoundError("no such cli", cli_name="codex")
        assert err.returncode == 127
        assert err.cli_name == "codex"
        assert err.recoverable is False

    def test_subprocess_error_recoverable(self):
        err = CLISubprocessError("exit 1", returncode=1, stderr="err")
        assert err.recoverable is True
        assert err.returncode == 1


# =============================================================================
# Classification re-exports
# =============================================================================


class TestClassificationReexports:
    @pytest.mark.parametrize(
        "error,expected",
        [
            (TimeoutError("slow"), (True, "timeout")),
            (RuntimeError("429 too many requests"), (True, "rate_limit")),
            (ConnectionError("conn reset"), (True, "network")),
            (RuntimeError("invalid api key provided"), (True, "auth")),
            (ValueError("totally novel failure xyzzy"), (False, "unknown")),
        ],
    )
    def test_classify_error_via_shim_surface(self, error, expected):
        assert ErrorClassifier.classify_error(error) == expected

    def test_classify_full_rate_limit_defaults_retry_after(self):
        result = ErrorClassifier.classify_full(RuntimeError("rate limit exceeded"))
        assert isinstance(result, ClassifiedError)
        assert result.category is ErrorCategory.RATE_LIMIT
        assert result.severity is ErrorSeverity.INFO
        assert result.action is RecoveryAction.WAIT
        assert result.should_fallback is True
        assert result.retry_after == 60.0

    def test_classified_error_is_recoverable_property(self):
        recoverable = ClassifiedError(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=True,
        )
        fatal = ClassifiedError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            action=RecoveryAction.ABORT,
            should_fallback=False,
        )
        assert recoverable.is_recoverable is True
        assert fatal.is_recoverable is False
        assert fatal.category_str == "validation"

    def test_classify_cli_error_timeout(self):
        err = classify_cli_error(returncode=-9, stderr="", stdout="x", timeout_seconds=20.0)
        assert isinstance(err, CLITimeoutError)
        assert err.timeout_seconds == 20.0
        assert "20.0" in str(err)

    def test_classify_cli_error_not_found(self):
        err = classify_cli_error(returncode=127, stderr="zsh: command not found: foo", stdout="x")
        assert isinstance(err, CLINotFoundError)
        assert err.recoverable is False

    def test_classify_cli_error_rate_limit(self):
        err = classify_cli_error(returncode=1, stderr="429 too many requests", stdout="x")
        assert isinstance(err, CLIAgentError)
        assert not isinstance(err, (CLITimeoutError, CLINotFoundError, CLIParseError))
        assert err.recoverable is True

    def test_classify_cli_error_empty_stdout_is_parse_error(self):
        err = classify_cli_error(returncode=0, stderr="", stdout="   ")
        assert isinstance(err, CLIParseError)
        assert err.recoverable is False

    def test_classify_cli_error_invalid_json_is_parse_error(self):
        err = classify_cli_error(returncode=0, stderr="", stdout="{not valid json")
        assert isinstance(err, CLIParseError)
        assert err.raw_output == "{not valid json"

    def test_classify_cli_error_json_error_payload(self):
        err = classify_cli_error(returncode=0, stderr="", stdout='{"error": "model overloaded"}')
        assert isinstance(err, CLIAgentError)
        assert "model overloaded" in str(err)
        assert err.recoverable is True

    def test_classify_cli_error_generic_subprocess(self):
        err = classify_cli_error(returncode=3, stderr="segfault", stdout="output")
        assert isinstance(err, CLISubprocessError)
        assert err.returncode == 3
        assert "segfault" in str(err)


# =============================================================================
# Handlers
# =============================================================================


class TestHandleAgentOperation:
    async def test_returns_operation_result_on_success(self):
        async def operation():
            return "answer"

        result = await handle_agent_operation(operation, "claude", "generate")
        assert result == "answer"

    async def test_timeout_returns_fallback_message(self):
        async def operation():
            raise asyncio.TimeoutError()

        result = await handle_agent_operation(
            operation, "claude", "generate", fallback_message="[System: skipped]"
        )
        assert result == "[System: skipped]"

    async def test_connection_error_returns_fallback_value(self):
        async def operation():
            raise ConnectionError("refused")

        result = await handle_agent_operation(operation, "claude", "critique", fallback_value=None)
        assert result is None

    async def test_unexpected_error_returns_fallback_value(self):
        async def operation():
            raise KeyError("missing")

        result = await handle_agent_operation(operation, "claude", "vote", fallback_value="default")
        assert result == "default"

    async def test_fallback_message_takes_precedence_over_value(self):
        async def operation():
            raise OSError("io down")

        result = await handle_agent_operation(
            operation,
            "claude",
            "generate",
            fallback_value="value",
            fallback_message="message",
        )
        assert result == "message"


class TestAgentErrorHandler:
    async def test_success_path_keeps_result(self):
        async with AgentErrorHandler("claude", "generate") as handler:
            handler.set_result("ok")
        assert handler.result == "ok"
        assert handler.error is None

    async def test_timeout_suppressed_and_fallback_used(self):
        async with AgentErrorHandler("claude", "generate", fallback_value="fb") as handler:
            raise asyncio.TimeoutError()
        assert handler.result == "fb"
        assert isinstance(handler.error, asyncio.TimeoutError)

    async def test_connection_error_suppressed(self):
        async with AgentErrorHandler("claude", "generate") as handler:
            raise ConnectionError("down")
        assert handler.result is None
        assert isinstance(handler.error, ConnectionError)

    async def test_generic_exception_suppressed_with_error_recorded(self):
        async with AgentErrorHandler("claude", "vote", fallback_value=0) as handler:
            raise ValueError("bad vote")
        assert handler.result == 0
        assert isinstance(handler.error, ValueError)


class TestFallbackMessageAndErrorAction:
    def test_make_fallback_message_default_operation(self):
        msg = make_fallback_message("claude")
        assert msg == "[System: Agent claude encountered an error - skipping this turn]"

    def test_make_fallback_message_custom_operation(self):
        msg = make_fallback_message("gpt", operation="critique")
        assert "gpt" in msg and msg.endswith("skipping this critique]")

    def test_build_error_action_known_category(self):
        category, message, exc_info = _build_error_action(TimeoutError("slow"), context="memory")
        assert category == "timeout"
        assert message.startswith("[memory] TimeoutError (timeout):")
        assert exc_info is False

    def test_build_error_action_unknown_category_uses_exc_info(self):
        category, message, exc_info = _build_error_action(ValueError("novel xyzzy"))
        assert category == "unknown"
        assert "ValueError (unknown)" in message
        assert exc_info is True


# =============================================================================
# Retry delay calculation
# =============================================================================


class TestCalculateRetryDelayWithJitter:
    def test_delay_within_jitter_bounds(self):
        base, factor = 2.0, 0.3
        for attempt in range(4):
            expected = min(base * (2**attempt), 60.0)
            delay = calculate_retry_delay_with_jitter(attempt, base, 60.0, jitter_factor=factor)
            assert expected * (1 - factor) - 1e-9 <= delay <= expected * (1 + factor) + 1e-9

    def test_delay_capped_at_max(self):
        delay = calculate_retry_delay_with_jitter(20, 1.0, max_delay=5.0, jitter_factor=0.0)
        assert delay == 5.0

    def test_zero_jitter_is_deterministic_exponential(self):
        assert calculate_retry_delay_with_jitter(0, 1.0, 60.0, jitter_factor=0.0) == 1.0
        assert calculate_retry_delay_with_jitter(3, 1.0, 60.0, jitter_factor=0.0) == 8.0

    def test_minimum_delay_floor(self):
        delay = calculate_retry_delay_with_jitter(0, 0.0, 60.0, jitter_factor=0.0)
        assert delay == 0.1

    def test_private_alias_is_same_function(self):
        assert _calculate_retry_delay_with_jitter is calculate_retry_delay_with_jitter
