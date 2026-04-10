"""Tests for queue retry policy helpers."""

from __future__ import annotations

import pytest

from aragora.queue import RetryPolicy as ExportedRetryPolicy
from aragora.queue import is_retryable_error as exported_is_retryable_error
from aragora.queue.config import QueueConfig, reset_queue_config, set_queue_config
from aragora.queue.retry import NON_RETRYABLE_EXCEPTIONS, RetryPolicy, is_retryable_error


@pytest.fixture(autouse=True)
def reset_queue_config_singleton():
    reset_queue_config()
    yield
    reset_queue_config()


def test_default_retry_policy_values():
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.base_delay_seconds == 1.0
    assert policy.max_delay_seconds == 300.0
    assert policy.exponential_base == 2.0
    assert policy.jitter is True
    assert policy.retryable_exceptions == (Exception,)


def test_from_config_uses_queue_retry_settings():
    set_queue_config(
        QueueConfig(
            retry_max_attempts=5,
            retry_base_delay=2.5,
            retry_max_delay=30.0,
        )
    )

    policy = RetryPolicy.from_config()

    assert policy.max_attempts == 5
    assert policy.base_delay_seconds == 2.5
    assert policy.max_delay_seconds == 30.0
    assert policy.exponential_base == 2.0
    assert policy.jitter is True


def test_get_delay_uses_exponential_backoff_without_jitter():
    policy = RetryPolicy(
        base_delay_seconds=1.5,
        max_delay_seconds=100.0,
        exponential_base=3.0,
        jitter=False,
    )

    assert policy.get_delay(0) == 1.5
    assert policy.get_delay(1) == 4.5
    assert policy.get_delay(2) == 13.5


def test_get_delay_caps_before_jitter(monkeypatch):
    monkeypatch.setattr("aragora.queue.retry.random.uniform", lambda _low, _high: 1.2)
    policy = RetryPolicy(
        base_delay_seconds=10.0,
        max_delay_seconds=12.0,
        exponential_base=3.0,
        jitter=True,
    )

    assert policy.get_delay(2) == pytest.approx(14.4)


def test_get_delay_with_jitter_uses_expected_bounds(monkeypatch):
    calls: list[tuple[float, float]] = []

    def fake_uniform(low: float, high: float) -> float:
        calls.append((low, high))
        return 0.8

    monkeypatch.setattr("aragora.queue.retry.random.uniform", fake_uniform)
    policy = RetryPolicy(base_delay_seconds=5.0, exponential_base=2.0, jitter=True)

    assert policy.get_delay(1) == 8.0
    assert calls == [(0.8, 1.2)]


def test_should_retry_respects_attempt_limit():
    policy = RetryPolicy(max_attempts=3)

    assert policy.should_retry(0) is True
    assert policy.should_retry(2) is True
    assert policy.should_retry(3) is False
    assert policy.should_retry(4) is False


@pytest.mark.parametrize("error", [ValueError("invalid"), TypeError("bad"), KeyError("missing")])
def test_should_retry_rejects_validation_errors(error):
    policy = RetryPolicy(max_attempts=3)

    assert policy.should_retry(1, error) is False


def test_should_retry_respects_custom_retryable_exception_types():
    policy = RetryPolicy(max_attempts=3, retryable_exceptions=(ConnectionError,))

    assert policy.should_retry(1, ConnectionError("transient")) is True
    assert policy.should_retry(1, RuntimeError("not configured as retryable")) is False


def test_get_remaining_attempts_never_goes_negative():
    policy = RetryPolicy(max_attempts=3)

    assert policy.get_remaining_attempts(0) == 3
    assert policy.get_remaining_attempts(1) == 2
    assert policy.get_remaining_attempts(3) == 0
    assert policy.get_remaining_attempts(5) == 0


@pytest.mark.parametrize("error_type", NON_RETRYABLE_EXCEPTIONS)
def test_is_retryable_error_rejects_non_retryable_exception_types(error_type):
    assert is_retryable_error(error_type("broken")) is False


@pytest.mark.parametrize(
    "message",
    [
        "invalid request",
        "resource not found",
        "unauthorized token",
        "forbidden workspace",
        "bad request payload",
        "validation failed",
    ],
)
def test_is_retryable_error_rejects_non_retryable_messages(message):
    assert is_retryable_error(RuntimeError(message)) is False


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("temporary upstream outage"),
        ConnectionError("connection reset"),
        TimeoutError("timed out"),
        Exception("429 rate limit"),
    ],
)
def test_is_retryable_error_accepts_transient_errors(error):
    assert is_retryable_error(error) is True


def test_queue_package_exports_retry_helpers():
    assert ExportedRetryPolicy is RetryPolicy
    assert exported_is_retryable_error is is_retryable_error
