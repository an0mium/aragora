"""Tests for aiohttp response helpers."""

from __future__ import annotations

import json

from aragora.server.handlers.utils.aiohttp_responses import web_error_response


def test_web_error_response_returns_simple_error_payload() -> None:
    response = web_error_response("Invalid input")

    assert response.status == 400
    assert response.content_type == "application/json"
    assert json.loads(response.text) == {"error": "Invalid input"}


def test_web_error_response_returns_structured_error_with_code() -> None:
    response = web_error_response("Not found", 404, code="NOT_FOUND")

    assert response.status == 404
    assert json.loads(response.text) == {"error": {"code": "NOT_FOUND", "message": "Not found"}}


def test_web_error_response_returns_structured_error_with_details() -> None:
    details = {"field": "email", "reason": "missing"}

    response = web_error_response("Validation failed", 422, details=details)

    assert response.status == 422
    assert json.loads(response.text) == {
        "error": {"message": "Validation failed", "details": details}
    }
