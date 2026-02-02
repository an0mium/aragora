"""Tests for aragora.resilience.http_client module.

Covers:
- TRANSIENT_HTTP_STATUSES constant
- HttpErrorInfo dataclass
- classify_http_error function
- CircuitBreakerMixin class
- ResilientRequestConfig defaults
- make_resilient_request async function
- _is_transient_exception helper
- _extract_status_code helper
- _extract_retry_after helper
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.resilience.http_client import (
    TRANSIENT_HTTP_STATUSES,
    CircuitBreakerMixin,
    HttpErrorInfo,
    ResilientRequestConfig,
    _extract_retry_after,
    _extract_status_code,
    _is_transient_exception,
    classify_http_error,
    make_resilient_request,
)


# ---------------------------------------------------------------------------
# 1. TRANSIENT_HTTP_STATUSES
# ---------------------------------------------------------------------------


class TestTransientHttpStatuses:
    """Tests for the TRANSIENT_HTTP_STATUSES frozenset."""

    def test_contains_429(self):
        assert 429 in TRANSIENT_HTTP_STATUSES

    def test_contains_500(self):
        assert 500 in TRANSIENT_HTTP_STATUSES

    def test_contains_502(self):
        assert 502 in TRANSIENT_HTTP_STATUSES

    def test_contains_503(self):
        assert 503 in TRANSIENT_HTTP_STATUSES

    def test_contains_504(self):
        assert 504 in TRANSIENT_HTTP_STATUSES

    def test_exactly_five_codes(self):
        assert len(TRANSIENT_HTTP_STATUSES) == 5

    def test_does_not_contain_400(self):
        assert 400 not in TRANSIENT_HTTP_STATUSES

    def test_does_not_contain_401(self):
        assert 401 not in TRANSIENT_HTTP_STATUSES

    def test_does_not_contain_404(self):
        assert 404 not in TRANSIENT_HTTP_STATUSES

    def test_does_not_contain_200(self):
        assert 200 not in TRANSIENT_HTTP_STATUSES

    def test_is_frozenset(self):
        assert isinstance(TRANSIENT_HTTP_STATUSES, frozenset)


# ---------------------------------------------------------------------------
# 2. HttpErrorInfo
# ---------------------------------------------------------------------------


class TestHttpErrorInfo:
    """Tests for HttpErrorInfo dataclass creation and field access."""

    def test_basic_creation(self):
        info = HttpErrorInfo(
            is_transient=True,
            is_rate_limit=True,
            is_auth_error=False,
            retry_after=30.0,
            status_code=429,
            message="Rate limited",
        )
        assert info.is_transient is True
        assert info.is_rate_limit is True
        assert info.is_auth_error is False
        assert info.retry_after == 30.0
        assert info.status_code == 429
        assert info.message == "Rate limited"

    def test_retry_after_none(self):
        info = HttpErrorInfo(
            is_transient=False,
            is_rate_limit=False,
            is_auth_error=True,
            retry_after=None,
            status_code=401,
            message="Unauthorized",
        )
        assert info.retry_after is None

    def test_non_transient_error(self):
        info = HttpErrorInfo(
            is_transient=False,
            is_rate_limit=False,
            is_auth_error=False,
            retry_after=None,
            status_code=404,
            message="Not found",
        )
        assert info.is_transient is False
        assert info.status_code == 404


# ---------------------------------------------------------------------------
# 3. classify_http_error
# ---------------------------------------------------------------------------


class TestClassifyHttpError:
    """Tests for classify_http_error function."""

    # -- Transient status codes --

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_status_codes(self, status):
        info = classify_http_error(status)
        assert info.is_transient is True
        assert info.status_code == status

    # -- Non-transient status codes --

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 409, 422])
    def test_non_transient_status_codes(self, status):
        info = classify_http_error(status)
        assert info.is_transient is False
        assert info.status_code == status

    # -- Rate limit detection --

    def test_rate_limit_429(self):
        info = classify_http_error(429)
        assert info.is_rate_limit is True

    def test_non_rate_limit_500(self):
        info = classify_http_error(500)
        assert info.is_rate_limit is False

    # -- Auth error detection --

    def test_auth_error_401(self):
        info = classify_http_error(401)
        assert info.is_auth_error is True

    def test_auth_error_403(self):
        info = classify_http_error(403)
        assert info.is_auth_error is True

    def test_non_auth_error_404(self):
        info = classify_http_error(404)
        assert info.is_auth_error is False

    def test_non_auth_error_429(self):
        info = classify_http_error(429)
        assert info.is_auth_error is False

    # -- Retry-After header extraction --

    def test_retry_after_title_case(self):
        info = classify_http_error(429, headers={"Retry-After": "120"})
        assert info.retry_after == 120.0

    def test_retry_after_lower_case(self):
        info = classify_http_error(429, headers={"retry-after": "45"})
        assert info.retry_after == 45.0

    def test_retry_after_float_value(self):
        info = classify_http_error(429, headers={"Retry-After": "1.5"})
        assert info.retry_after == 1.5

    def test_retry_after_invalid_value(self):
        info = classify_http_error(429, headers={"Retry-After": "not-a-number"})
        assert info.retry_after is None

    def test_retry_after_absent(self):
        info = classify_http_error(429, headers={"Content-Type": "application/json"})
        assert info.retry_after is None

    def test_retry_after_no_headers(self):
        info = classify_http_error(429)
        assert info.retry_after is None

    # -- Body as dict --

    def test_body_dict_with_error_key(self):
        info = classify_http_error(500, body={"error": "Internal server error"})
        assert info.message == "Internal server error"

    def test_body_dict_with_message_key(self):
        info = classify_http_error(500, body={"message": "Something broke"})
        assert info.message == "Something broke"

    def test_body_dict_with_message_capital_key(self):
        info = classify_http_error(500, body={"Message": "Capital M"})
        assert info.message == "Capital M"

    def test_body_dict_fallback_to_str(self):
        info = classify_http_error(500, body={"code": 5000, "detail": "oops"})
        assert "code" in info.message
        assert "5000" in info.message

    def test_body_dict_error_takes_priority(self):
        info = classify_http_error(500, body={"error": "first", "message": "second"})
        assert info.message == "first"

    # -- Body as string --

    def test_body_string(self):
        info = classify_http_error(500, body="Server encountered an error")
        assert info.message == "Server encountered an error"

    def test_body_string_truncated_to_200(self):
        long_body = "x" * 500
        info = classify_http_error(500, body=long_body)
        assert len(info.message) == 200

    def test_body_empty_string(self):
        info = classify_http_error(500, body="")
        assert info.message == "HTTP 500"

    # -- Body None --

    def test_body_none(self):
        info = classify_http_error(404)
        assert info.message == "HTTP 404"

    # -- Custom transient statuses --

    def test_custom_transient_statuses(self):
        custom = frozenset({418, 429})
        info_418 = classify_http_error(418, transient_statuses=custom)
        assert info_418.is_transient is True

        info_500 = classify_http_error(500, transient_statuses=custom)
        assert info_500.is_transient is False


# ---------------------------------------------------------------------------
# 4. CircuitBreakerMixin
# ---------------------------------------------------------------------------


class _TestConnector(CircuitBreakerMixin):
    """Concrete class for testing the mixin."""

    pass


class TestCircuitBreakerMixin:
    """Tests for the CircuitBreakerMixin."""

    # -- init_circuit_breaker --

    def test_init_with_provided_breaker(self):
        breaker = MagicMock()
        connector = _TestConnector()
        connector.init_circuit_breaker(name="test", circuit_breaker=breaker)
        assert connector._circuit_breaker is breaker

    @patch("aragora.resilience.http_client.CircuitBreaker")
    def test_init_with_enable_true_creates_new(self, mock_cb_cls):
        mock_instance = MagicMock()
        mock_cb_cls.return_value = mock_instance

        connector = _TestConnector()
        connector.init_circuit_breaker(
            name="test-conn",
            enable=True,
            failure_threshold=5,
            cooldown_seconds=30.0,
        )

        mock_cb_cls.assert_called_once_with(
            name="test-conn",
            failure_threshold=5,
            cooldown_seconds=30.0,
        )
        assert connector._circuit_breaker is mock_instance

    def test_init_with_enable_false_no_breaker(self):
        connector = _TestConnector()
        connector.init_circuit_breaker(name="test", enable=False)
        assert connector._circuit_breaker is None

    def test_init_provided_breaker_takes_precedence_over_enable_false(self):
        breaker = MagicMock()
        connector = _TestConnector()
        connector.init_circuit_breaker(name="test", circuit_breaker=breaker, enable=False)
        assert connector._circuit_breaker is breaker

    # -- check_circuit_breaker --

    def test_check_returns_true_when_no_breaker(self):
        connector = _TestConnector()
        connector._circuit_breaker = None
        assert connector.check_circuit_breaker() is True

    def test_check_delegates_to_can_proceed(self):
        breaker = MagicMock()
        breaker.can_proceed.return_value = True
        connector = _TestConnector()
        connector._circuit_breaker = breaker
        assert connector.check_circuit_breaker() is True
        breaker.can_proceed.assert_called_once()

    def test_check_returns_false_when_open(self):
        breaker = MagicMock()
        breaker.can_proceed.return_value = False
        connector = _TestConnector()
        connector._circuit_breaker = breaker
        assert connector.check_circuit_breaker() is False

    # -- record_circuit_success --

    def test_record_success_with_breaker(self):
        breaker = MagicMock()
        connector = _TestConnector()
        connector._circuit_breaker = breaker
        connector.record_circuit_success()
        breaker.record_success.assert_called_once()

    def test_record_success_without_breaker(self):
        connector = _TestConnector()
        connector._circuit_breaker = None
        # Should not raise
        connector.record_circuit_success()

    # -- record_circuit_failure --

    def test_record_failure_with_breaker(self):
        breaker = MagicMock()
        connector = _TestConnector()
        connector._circuit_breaker = breaker
        connector.record_circuit_failure()
        breaker.record_failure.assert_called_once()

    def test_record_failure_without_breaker(self):
        connector = _TestConnector()
        connector._circuit_breaker = None
        # Should not raise
        connector.record_circuit_failure()

    # -- get_circuit_cooldown --

    def test_get_cooldown_no_breaker(self):
        connector = _TestConnector()
        connector._circuit_breaker = None
        assert connector.get_circuit_cooldown() == 0.0

    def test_get_cooldown_delegates_to_breaker(self):
        breaker = MagicMock()
        breaker.cooldown_remaining.return_value = 42.5
        connector = _TestConnector()
        connector._circuit_breaker = breaker
        assert connector.get_circuit_cooldown() == 42.5
        breaker.cooldown_remaining.assert_called_once()


# ---------------------------------------------------------------------------
# 5. ResilientRequestConfig
# ---------------------------------------------------------------------------


class TestResilientRequestConfig:
    """Tests for ResilientRequestConfig defaults."""

    def test_default_max_retries(self):
        config = ResilientRequestConfig()
        assert config.max_retries == 3

    def test_default_base_delay(self):
        config = ResilientRequestConfig()
        assert config.base_delay == 0.5

    def test_default_transient_statuses(self):
        config = ResilientRequestConfig()
        assert config.transient_statuses is TRANSIENT_HTTP_STATUSES

    def test_default_record_client_errors(self):
        config = ResilientRequestConfig()
        assert config.record_client_errors is False

    def test_custom_values(self):
        custom_statuses = frozenset({429, 503})
        config = ResilientRequestConfig(
            max_retries=5,
            base_delay=1.0,
            transient_statuses=custom_statuses,
            record_client_errors=True,
        )
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.transient_statuses == custom_statuses
        assert config.record_client_errors is True


# ---------------------------------------------------------------------------
# 6. make_resilient_request
# ---------------------------------------------------------------------------


class TestMakeResilientRequest:
    """Tests for the make_resilient_request async function."""

    @pytest.mark.asyncio
    async def test_success_returns_result(self):
        request_func = AsyncMock(return_value={"data": 42})
        result = await make_resilient_request(request_func=request_func)
        assert result == {"data": 42}
        request_func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_records_to_circuit_breaker(self):
        breaker = MagicMock()
        breaker.can_proceed.return_value = True
        request_func = AsyncMock(return_value="ok")

        await make_resilient_request(
            request_func=request_func,
            circuit_breaker=breaker,
        )

        breaker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_raises(self):
        breaker = MagicMock()
        breaker.can_proceed.return_value = False
        breaker.cooldown_remaining.return_value = 30.0

        from aragora.resilience.circuit_breaker import CircuitOpenError

        with pytest.raises(CircuitOpenError) as exc_info:
            await make_resilient_request(
                request_func=AsyncMock(),
                circuit_breaker=breaker,
                connector_name="test-api",
            )
        assert exc_info.value.circuit_name == "test-api"
        assert exc_info.value.cooldown_remaining == 30.0

    @pytest.mark.asyncio
    @patch("aragora.resilience.http_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_retries_with_backoff(self, mock_sleep):
        """Transient error retries with exponential backoff."""
        exc = ConnectionError("connection lost")
        request_func = AsyncMock(side_effect=[exc, exc, "success"])

        config = ResilientRequestConfig(max_retries=3, base_delay=1.0)
        result = await make_resilient_request(
            request_func=request_func,
            config=config,
        )

        assert result == "success"
        assert request_func.await_count == 3
        # First retry: 1.0 * 2^0 = 1.0, second retry: 1.0 * 2^1 = 2.0
        assert mock_sleep.await_count == 2
        mock_sleep.assert_any_await(1.0)
        mock_sleep.assert_any_await(2.0)

    @pytest.mark.asyncio
    @patch("aragora.resilience.http_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_transient_error_records_circuit_failure(self, mock_sleep):
        breaker = MagicMock()
        breaker.can_proceed.return_value = True

        exc = TimeoutError("timed out")
        request_func = AsyncMock(side_effect=[exc, "ok"])

        config = ResilientRequestConfig(max_retries=2, base_delay=0.1)
        await make_resilient_request(
            request_func=request_func,
            circuit_breaker=breaker,
            config=config,
        )

        breaker.record_failure.assert_called_once()
        breaker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_transient_error_does_not_retry(self):
        """Non-transient errors are raised immediately."""

        class ClientError(Exception):
            status_code = 400

        request_func = AsyncMock(side_effect=ClientError("Bad request"))
        config = ResilientRequestConfig(max_retries=3)

        with pytest.raises(ClientError):
            await make_resilient_request(
                request_func=request_func,
                config=config,
            )

        request_func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_transient_error_records_failure_if_configured(self):
        """record_client_errors=True records non-transient failures."""
        breaker = MagicMock()
        breaker.can_proceed.return_value = True

        class ClientError(Exception):
            status_code = 400

        request_func = AsyncMock(side_effect=ClientError("Bad request"))
        config = ResilientRequestConfig(record_client_errors=True)

        with pytest.raises(ClientError):
            await make_resilient_request(
                request_func=request_func,
                circuit_breaker=breaker,
                config=config,
            )

        breaker.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_transient_error_no_failure_record_by_default(self):
        """record_client_errors=False (default) does not record non-transient failures."""
        breaker = MagicMock()
        breaker.can_proceed.return_value = True

        class ClientError(Exception):
            status_code = 400

        request_func = AsyncMock(side_effect=ClientError("Bad request"))
        config = ResilientRequestConfig(record_client_errors=False)

        with pytest.raises(ClientError):
            await make_resilient_request(
                request_func=request_func,
                circuit_breaker=breaker,
                config=config,
            )

        breaker.record_failure.assert_not_called()

    @pytest.mark.asyncio
    @patch("aragora.resilience.http_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_exhausted_raises(self, mock_sleep):
        """After max retries, the last exception is raised."""
        exc = ConnectionError("connection lost")
        request_func = AsyncMock(side_effect=exc)

        config = ResilientRequestConfig(max_retries=2, base_delay=0.1)

        with pytest.raises(ConnectionError, match="connection lost"):
            await make_resilient_request(
                request_func=request_func,
                config=config,
            )

        # 1 initial attempt + 2 retries = 3 total
        assert request_func.await_count == 3

    @pytest.mark.asyncio
    @patch("aragora.resilience.http_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_on_transient_error_callback(self, mock_sleep):
        """on_transient_error callback is invoked on each transient retry."""
        callback = AsyncMock()

        class TransientError(Exception):
            status_code = 503

        exc = TransientError("Service unavailable")
        request_func = AsyncMock(side_effect=[exc, "success"])

        config = ResilientRequestConfig(max_retries=3, base_delay=0.5)
        await make_resilient_request(
            request_func=request_func,
            config=config,
            on_transient_error=callback,
        )

        callback.assert_awaited_once()
        args = callback.await_args[0]
        assert args[0] == 503  # status code
        assert args[1] == 0.5  # delay = base_delay * 2^0

    @pytest.mark.asyncio
    @patch("aragora.resilience.http_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_after_respected(self, mock_sleep):
        """Retry-After value from exception overrides exponential backoff."""

        class RateLimitError(Exception):
            status_code = 429
            retry_after = 10.0

        exc = RateLimitError("Rate limited")
        request_func = AsyncMock(side_effect=[exc, "ok"])

        config = ResilientRequestConfig(max_retries=2, base_delay=0.5)
        await make_resilient_request(
            request_func=request_func,
            config=config,
        )

        mock_sleep.assert_awaited_once_with(10.0)

    @pytest.mark.asyncio
    @patch("aragora.resilience.http_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_after_from_response_headers(self, mock_sleep):
        """Retry-After extracted from exception.response.headers."""
        resp = SimpleNamespace(
            status_code=429,
            headers={"Retry-After": "25"},
        )
        exc = Exception("Rate limited")
        exc.response = resp

        # We also need status_code for _is_transient_exception to detect it
        exc.status_code = 429

        request_func = AsyncMock(side_effect=[exc, "ok"])

        config = ResilientRequestConfig(max_retries=2, base_delay=0.5)
        await make_resilient_request(
            request_func=request_func,
            config=config,
        )

        mock_sleep.assert_awaited_once_with(25.0)

    @pytest.mark.asyncio
    async def test_no_circuit_breaker_still_works(self):
        """Request succeeds without a circuit breaker."""
        request_func = AsyncMock(return_value="result")
        result = await make_resilient_request(
            request_func=request_func,
            circuit_breaker=None,
        )
        assert result == "result"

    @pytest.mark.asyncio
    @patch("aragora.resilience.http_client.asyncio.sleep", new_callable=AsyncMock)
    async def test_zero_retries_no_retry(self, mock_sleep):
        """max_retries=0 means no retries at all."""
        exc = ConnectionError("fail")
        request_func = AsyncMock(side_effect=exc)

        config = ResilientRequestConfig(max_retries=0)

        with pytest.raises(ConnectionError):
            await make_resilient_request(
                request_func=request_func,
                config=config,
            )

        request_func.assert_awaited_once()
        mock_sleep.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. _is_transient_exception
# ---------------------------------------------------------------------------


class TestIsTransientException:
    """Tests for _is_transient_exception helper."""

    def test_timeout_error(self):
        assert _is_transient_exception(TimeoutError(), TRANSIENT_HTTP_STATUSES) is True

    def test_asyncio_timeout_error(self):
        assert _is_transient_exception(asyncio.TimeoutError(), TRANSIENT_HTTP_STATUSES) is True

    def test_connection_error(self):
        assert _is_transient_exception(ConnectionError(), TRANSIENT_HTTP_STATUSES) is True

    def test_os_error(self):
        assert _is_transient_exception(OSError("network"), TRANSIENT_HTTP_STATUSES) is True

    def test_connection_reset_error(self):
        # ConnectionResetError is a subclass of ConnectionError
        assert _is_transient_exception(ConnectionResetError(), TRANSIENT_HTTP_STATUSES) is True

    def test_status_code_in_transient(self):
        exc = Exception("server error")
        exc.status_code = 503
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is True

    def test_status_code_not_in_transient(self):
        exc = Exception("bad request")
        exc.status_code = 400
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is False

    def test_status_code_429(self):
        exc = Exception("rate limited")
        exc.status_code = 429
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is True

    def test_message_pattern_timeout(self):
        exc = Exception("request timeout exceeded")
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is True

    def test_message_pattern_connection(self):
        exc = Exception("connection refused by server")
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is True

    def test_message_pattern_rate_limit(self):
        exc = Exception("rate limit exceeded")
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is True

    def test_message_pattern_503(self):
        exc = Exception("HTTP 503 Service Unavailable")
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is True

    def test_message_pattern_502(self):
        exc = Exception("502 bad gateway")
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is True

    def test_non_transient_exception(self):
        exc = Exception("invalid parameter format")
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is False

    def test_value_error_non_transient(self):
        assert _is_transient_exception(ValueError("bad value"), TRANSIENT_HTTP_STATUSES) is False

    def test_custom_transient_statuses(self):
        custom = frozenset({418})
        exc = Exception("teapot")
        exc.status_code = 418
        assert _is_transient_exception(exc, custom) is True

    def test_status_via_response_attribute(self):
        """Status code extracted from exc.response.status_code."""
        resp = SimpleNamespace(status_code=502)
        exc = Exception("gateway error")
        exc.response = resp
        assert _is_transient_exception(exc, TRANSIENT_HTTP_STATUSES) is True


# ---------------------------------------------------------------------------
# 8. _extract_status_code
# ---------------------------------------------------------------------------


class TestExtractStatusCode:
    """Tests for _extract_status_code helper."""

    def test_status_code_attribute(self):
        exc = Exception("error")
        exc.status_code = 500
        assert _extract_status_code(exc) == 500

    def test_status_attribute(self):
        exc = Exception("error")
        exc.status = 503
        assert _extract_status_code(exc) == 503

    def test_response_status_code(self):
        resp = SimpleNamespace(status_code=429)
        exc = Exception("error")
        exc.response = resp
        assert _extract_status_code(exc) == 429

    def test_response_status(self):
        resp = SimpleNamespace(status=502)
        exc = Exception("error")
        exc.response = resp
        assert _extract_status_code(exc) == 502

    def test_no_status_code_returns_none(self):
        exc = Exception("plain error")
        assert _extract_status_code(exc) is None

    def test_status_code_takes_priority_over_response(self):
        """Direct status_code attribute is checked first."""
        resp = SimpleNamespace(status_code=502)
        exc = Exception("error")
        exc.status_code = 429
        exc.response = resp
        assert _extract_status_code(exc) == 429

    def test_status_takes_priority_over_response(self):
        """Direct status attribute is checked before response."""
        resp = SimpleNamespace(status_code=502)
        exc = Exception("error")
        exc.status = 503
        exc.response = resp
        assert _extract_status_code(exc) == 503

    def test_response_without_status_attributes(self):
        resp = SimpleNamespace(body="data")
        exc = Exception("error")
        exc.response = resp
        assert _extract_status_code(exc) is None


# ---------------------------------------------------------------------------
# 9. _extract_retry_after
# ---------------------------------------------------------------------------


class TestExtractRetryAfter:
    """Tests for _extract_retry_after helper."""

    def test_retry_after_attribute_float(self):
        exc = Exception("rate limited")
        exc.retry_after = 30.0
        assert _extract_retry_after(exc) == 30.0

    def test_retry_after_attribute_int(self):
        exc = Exception("rate limited")
        exc.retry_after = 60
        assert _extract_retry_after(exc) == 60.0

    def test_retry_after_attribute_string(self):
        exc = Exception("rate limited")
        exc.retry_after = "45"
        assert _extract_retry_after(exc) == 45.0

    def test_retry_after_attribute_none(self):
        exc = Exception("rate limited")
        exc.retry_after = None
        assert _extract_retry_after(exc) is None

    def test_retry_after_attribute_invalid(self):
        exc = Exception("rate limited")
        exc.retry_after = "not-a-number"
        assert _extract_retry_after(exc) is None

    def test_retry_after_from_response_headers_title_case(self):
        resp = SimpleNamespace(headers={"Retry-After": "120"})
        exc = Exception("rate limited")
        exc.response = resp
        assert _extract_retry_after(exc) == 120.0

    def test_retry_after_from_response_headers_lower_case(self):
        resp = SimpleNamespace(headers={"retry-after": "90"})
        exc = Exception("rate limited")
        exc.response = resp
        assert _extract_retry_after(exc) == 90.0

    def test_retry_after_from_response_headers_invalid(self):
        resp = SimpleNamespace(headers={"Retry-After": "bad-value"})
        exc = Exception("rate limited")
        exc.response = resp
        assert _extract_retry_after(exc) is None

    def test_no_retry_after_returns_none(self):
        exc = Exception("plain error")
        assert _extract_retry_after(exc) is None

    def test_response_without_headers(self):
        resp = SimpleNamespace(body="data")
        exc = Exception("error")
        exc.response = resp
        assert _extract_retry_after(exc) is None

    def test_response_headers_not_dict(self):
        """Non-dict headers are ignored gracefully."""
        resp = SimpleNamespace(headers="not-a-dict")
        exc = Exception("error")
        exc.response = resp
        assert _extract_retry_after(exc) is None

    def test_retry_after_attribute_takes_priority_over_headers(self):
        """Direct retry_after attribute is checked before response headers."""
        resp = SimpleNamespace(headers={"Retry-After": "200"})
        exc = Exception("rate limited")
        exc.retry_after = 50.0
        exc.response = resp
        assert _extract_retry_after(exc) == 50.0

    def test_response_headers_empty_retry_after(self):
        resp = SimpleNamespace(headers={"Retry-After": ""})
        exc = Exception("error")
        exc.response = resp
        assert _extract_retry_after(exc) is None
