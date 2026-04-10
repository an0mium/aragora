import logging

from aragora.server.request_logging import RequestLoggingMixin


class DummyRequestLogger(RequestLoggingMixin):
    def __init__(
        self,
        *,
        client_address: tuple[str, int] | None = ("127.0.0.1", 12345),
        headers: dict[str, str] | None = None,
        enabled: bool = True,
        slow_threshold_ms: int = 1000,
    ) -> None:
        if client_address is not None:
            self.client_address = client_address
        self.headers = headers or {}
        self._request_log_enabled = enabled
        self._slow_request_threshold_ms = slow_threshold_ms


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_log_request_captures_method_path_status_duration_and_ip(caplog) -> None:
    request_logger = DummyRequestLogger(client_address=("198.51.100.10", 12345))

    with caplog.at_level(logging.INFO, logger="aragora.server.request_logging"):
        request_logger._log_request("GET", "/api/v1/status", 200, 12.34)

    assert _messages(caplog) == [
        "[request] GET /api/v1/status status=200 duration=12.3ms ip=198.51.100.10"
    ]
    assert caplog.records[0].levelno == logging.INFO


def test_log_request_does_not_log_when_disabled(caplog) -> None:
    request_logger = DummyRequestLogger(enabled=False)

    with caplog.at_level(logging.INFO, logger="aragora.server.request_logging"):
        request_logger._log_request("GET", "/api/v1/status", 200, 12.0)

    assert caplog.records == []


def test_client_error_logs_warning(caplog) -> None:
    request_logger = DummyRequestLogger()

    with caplog.at_level(logging.WARNING, logger="aragora.server.request_logging"):
        request_logger._log_request("POST", "/api/v1/debates", 404, 8.0)

    assert caplog.records[0].levelno == logging.WARNING
    assert "status=404" in caplog.records[0].getMessage()


def test_server_error_logs_error(caplog) -> None:
    request_logger = DummyRequestLogger()

    with caplog.at_level(logging.ERROR, logger="aragora.server.request_logging"):
        request_logger._log_request("POST", "/api/v1/debates", 502, 8.0)

    assert caplog.records[0].levelno == logging.ERROR
    assert "status=502" in caplog.records[0].getMessage()


def test_slow_request_logs_warning_and_marks_slow(caplog) -> None:
    request_logger = DummyRequestLogger(slow_threshold_ms=50)

    with caplog.at_level(logging.WARNING, logger="aragora.server.request_logging"):
        request_logger._log_request("GET", "/api/v1/debates", 200, 50.1)

    message = caplog.records[0].getMessage()
    assert caplog.records[0].levelno == logging.WARNING
    assert "duration=50.1ms" in message
    assert message.endswith("SLOW")


def test_sensitive_extra_fields_are_redacted(caplog) -> None:
    request_logger = DummyRequestLogger()

    with caplog.at_level(logging.INFO, logger="aragora.server.request_logging"):
        request_logger._log_request(
            "GET",
            "/api/v1/status",
            200,
            12.0,
            extra={
                "Authorization": "Bearer secret-token",
                "Cookie": "session=abc",
                "X-API-Key": "key-123",
                "route": "status",
            },
        )

    message = caplog.records[0].getMessage()
    assert "Authorization=[REDACTED]" in message
    assert "Cookie=[REDACTED]" in message
    assert "X-API-Key=[REDACTED]" in message
    assert "route=status" in message
    assert "secret-token" not in message
    assert "session=abc" not in message
    assert "key-123" not in message


def test_long_extra_values_are_truncated(caplog) -> None:
    request_logger = DummyRequestLogger()
    body = "x" * 600

    with caplog.at_level(logging.INFO, logger="aragora.server.request_logging"):
        request_logger._log_request(
            "POST",
            "/api/v1/debates",
            200,
            12.0,
            extra={"request_body": body},
        )

    message = caplog.records[0].getMessage()
    assert f"request_body={'x' * 500}...[truncated]" in message
    assert "x" * 550 not in message


def test_normalize_endpoint_replaces_dynamic_segments() -> None:
    request_logger = DummyRequestLogger()

    assert (
        request_logger._normalize_endpoint(
            "/api/users/550e8400-e29b-41d4-a716-446655440000/items/12345"
        )
        == "/api/users/{id}/items/{id}"
    )
    assert request_logger._normalize_endpoint("/api/debates/abc123def456/messages") == (
        "/api/debates/{id}/messages"
    )


def test_get_client_ip_uses_forwarded_for_from_trusted_proxy() -> None:
    request_logger = DummyRequestLogger(
        client_address=("127.0.0.1", 443),
        headers={"X-Forwarded-For": "203.0.113.44, 127.0.0.1"},
    )

    assert request_logger._get_client_ip() == "203.0.113.44"


def test_get_client_ip_ignores_forwarded_for_from_untrusted_client() -> None:
    request_logger = DummyRequestLogger(
        client_address=("198.51.100.99", 443),
        headers={"X-Forwarded-For": "203.0.113.44"},
    )

    assert request_logger._get_client_ip() == "198.51.100.99"


def test_get_client_ip_returns_unknown_without_client_address() -> None:
    request_logger = DummyRequestLogger(client_address=None)

    assert request_logger._get_client_ip() == "unknown"
