import json

from aragora.server.handlers.base import handle_errors


def test_handle_errors_adds_validation_error_code_without_changing_error_shape():
    @handle_errors("test operation")
    def handler():
        raise ValueError("bad input")

    result = handler()
    body = json.loads(result.body)

    assert result.status_code == 400
    assert body["error"] == "Invalid data format"
    assert body["error_code"] == "VALIDATION_ERROR"


def test_handle_errors_adds_not_found_error_code():
    @handle_errors("test operation")
    def handler():
        raise FileNotFoundError("missing")

    result = handler()
    body = json.loads(result.body)

    assert result.status_code == 404
    assert body["error"] == "Resource not found"
    assert body["error_code"] == "NOT_FOUND"


def test_handle_errors_adds_internal_error_code_for_generic_exception():
    @handle_errors("test operation")
    def handler():
        raise RuntimeError("boom")

    result = handler()
    body = json.loads(result.body)

    assert result.status_code == 500
    assert body["error"] == "An error occurred"
    assert body["error_code"] == "INTERNAL_ERROR"
