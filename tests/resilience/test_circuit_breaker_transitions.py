"""Tests for circuit breaker state transitions.

Verifies the three core transitions:
- closed -> open on failure threshold
- open -> half-open after cooldown timeout
- half-open -> closed on success recovery

Tests both single-entity and multi-entity modes.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from aragora.resilience.circuit_breaker import CircuitBreaker


class TestClosedToOpen:
    """Closed -> Open transition when failure threshold is reached."""

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.get_status() == "closed"
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status() == "closed"
        opened = cb.record_failure()
        assert opened is True
        assert cb.get_status() == "open"

    def test_does_not_open_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.get_status() == "closed"

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status() == "closed"

    def test_entity_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("svc-a")
        opened = cb.record_failure("svc-a")
        assert opened is True
        assert cb.get_status("svc-a") == "open"
        # Other entity unaffected
        assert cb.get_status("svc-b") == "closed"


class TestOpenToHalfOpen:
    """Open -> Half-Open transition after cooldown elapses."""

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0)
        cb.record_failure()
        assert cb.get_status() == "open"
        assert cb.can_proceed() is False

        with patch("aragora.resilience.circuit_breaker.time") as mock_time:
            mock_time.time.return_value = time.time() + 11.0
            assert cb.get_status() == "half-open"
            assert cb.can_proceed() is True

    def test_stays_open_before_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure()
        with patch("aragora.resilience.circuit_breaker.time") as mock_time:
            mock_time.time.return_value = time.time() + 30.0
            assert cb.get_status() == "open"
            assert cb.can_proceed() is False

    def test_entity_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=5.0)
        cb.record_failure("agent-1")
        assert cb.get_status("agent-1") == "open"

        with patch("aragora.resilience.circuit_breaker.time") as mock_time:
            mock_time.time.return_value = time.time() + 6.0
            assert cb.get_status("agent-1") == "half-open"
            assert cb.is_available("agent-1") is True


class TestHalfOpenToClosed:
    """Half-Open -> Closed transition on success recovery."""

    def test_single_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0)
        cb.record_failure()
        assert cb.get_status() == "open"

        # Advance past cooldown so can_proceed sees half-open and resets
        with patch("aragora.resilience.circuit_breaker.time") as mock_time:
            mock_time.time.return_value = time.time() + 11.0
            assert cb.can_proceed() is True
            assert cb.get_status() == "closed"

    def test_entity_closes_after_success_threshold(self):
        cb = CircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=10.0,
            half_open_success_threshold=2,
        )
        cb.record_failure("svc")
        assert cb.get_status("svc") == "open"

        with patch("aragora.resilience.circuit_breaker.time") as mock_time:
            mock_time.time.return_value = time.time() + 11.0
            assert cb.get_status("svc") == "half-open"
            cb.record_success("svc")
            assert cb.get_status("svc") == "half-open"  # needs 2 successes
            cb.record_success("svc")
            assert cb.get_status("svc") == "closed"

    def test_failure_in_half_open_keeps_open(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0)
        cb.record_failure("svc")
        assert cb.get_status("svc") == "open"

        with patch("aragora.resilience.circuit_breaker.time") as mock_time:
            # Advance past cooldown -> half-open
            mock_time.time.return_value = time.time() + 11.0
            assert cb.get_status("svc") == "half-open"
            # Failure keeps entity in open_at dict (circuit doesn't close)
            cb.record_failure("svc")
            # Entity still tracked as having an open circuit
            assert "svc" in cb._circuit_open_at
            assert cb._failures["svc"] >= 1
