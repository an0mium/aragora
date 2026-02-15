"""
Tests for Email Triage Rules Management API.

Covers:
- GET  /api/v1/email/triage/rules (list rules)
- PUT  /api/v1/email/triage/rules (update rules)
- POST /api/v1/email/triage/test  (test message against rules)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.analysis.email_triage import TriageConfig, TriageRule, TriageRuleEngine
from aragora.server.handlers.email_triage import (
    EmailTriageHandler,
    _set_engine,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_mock_handler(method="GET", body=None):
    mock = MagicMock()
    mock.command = method
    if body is not None:
        body_bytes = json.dumps(body).encode()
    else:
        body_bytes = b"{}"
    mock.rfile = MagicMock()
    mock.rfile.read = MagicMock(return_value=body_bytes)
    mock.headers = {"Content-Length": str(len(body_bytes))}
    return mock


def _parse(result) -> dict[str, Any]:
    if result is None:
        return {}
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body) if body else {}


@pytest.fixture
def handler():
    return EmailTriageHandler(ctx={})


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset engine to default before each test."""
    config = TriageConfig(
        rules=[
            TriageRule(label="urgent_order", keywords=["urgent", "rush"], priority="high"),
            TriageRule(label="refund", keywords=["refund", "return"], priority="medium"),
            TriageRule(label="newsletter", keywords=["unsubscribe", "newsletter"], priority="low"),
        ],
        escalation_keywords=["legal", "lawsuit"],
        auto_handle_threshold=0.85,
    )
    _set_engine(TriageRuleEngine(config))
    yield
    _set_engine(None)


# ============================================================================
# Route Tests
# ============================================================================


class TestRouting:
    def test_can_handle_rules_path(self, handler):
        assert handler.can_handle("/api/v1/email/triage/rules") is True

    def test_can_handle_test_path(self, handler):
        assert handler.can_handle("/api/v1/email/triage/test") is True

    def test_cannot_handle_other_paths(self, handler):
        assert handler.can_handle("/api/v1/email/other") is False


# ============================================================================
# GET - List Rules
# ============================================================================


class TestGetRules:
    def test_returns_current_rules(self, handler):
        http = _make_mock_handler()
        result = handler.handle("/api/v1/email/triage/rules", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert len(body["rules"]) == 3
        assert body["escalation_keywords"] == ["legal", "lawsuit"]
        assert body["auto_handle_threshold"] == 0.85

    def test_rules_have_expected_fields(self, handler):
        http = _make_mock_handler()
        result = handler.handle("/api/v1/email/triage/rules", {}, http)
        body = _parse(result)
        rule = body["rules"][0]
        assert "label" in rule
        assert "keywords" in rule
        assert "priority" in rule

    def test_returns_none_for_unhandled_path(self, handler):
        http = _make_mock_handler()
        result = handler.handle("/api/v1/other", {}, http)
        assert result is None


# ============================================================================
# PUT - Update Rules
# ============================================================================


class TestUpdateRules:
    def test_update_rules(self, handler):
        http = _make_mock_handler(
            method="PUT",
            body={
                "rules": [
                    {"label": "vip", "keywords": ["vip", "priority"], "priority": "high"},
                ],
            },
        )
        result = handler.handle_put("/api/v1/email/triage/rules", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["rules_count"] == 1

        # Verify rules were actually updated
        http2 = _make_mock_handler()
        result2 = handler.handle("/api/v1/email/triage/rules", {}, http2)
        body2 = _parse(result2)
        assert len(body2["rules"]) == 1
        assert body2["rules"][0]["label"] == "vip"

    def test_update_with_escalation_keywords(self, handler):
        http = _make_mock_handler(
            method="PUT",
            body={
                "rules": [
                    {"label": "support", "keywords": ["help"], "priority": "medium"},
                ],
                "escalation_keywords": ["emergency", "outage"],
            },
        )
        result = handler.handle_put("/api/v1/email/triage/rules", {}, http)
        assert result.status_code == 200

    def test_update_invalid_priority(self, handler):
        http = _make_mock_handler(
            method="PUT",
            body={
                "rules": [
                    {"label": "bad", "keywords": ["test"], "priority": "critical"},
                ],
            },
        )
        result = handler.handle_put("/api/v1/email/triage/rules", {}, http)
        assert result.status_code == 400

    def test_update_invalid_json(self, handler):
        mock = MagicMock()
        mock.command = "PUT"
        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=b"not json")
        mock.headers = {"Content-Length": "8"}
        result = handler.handle_put("/api/v1/email/triage/rules", {}, mock)
        assert result.status_code == 400

    def test_update_returns_none_for_unhandled_path(self, handler):
        http = _make_mock_handler(method="PUT", body={})
        result = handler.handle_put("/api/v1/other", {}, http)
        assert result is None


# ============================================================================
# POST - Test Message
# ============================================================================


class TestTestMessage:
    def test_high_priority_match(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "subject": "URGENT: Rush order needed",
                "from_address": "customer@example.com",
                "snippet": "Please process this rush order immediately",
            },
        )
        result = handler.handle_post("/api/v1/email/triage/test", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["priority"] == "high"
        assert body["matched_rule"] == "urgent_order"
        assert body["score_boost"] > 0

    def test_medium_priority_match(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "subject": "Refund request",
                "snippet": "I would like a refund for my order",
            },
        )
        result = handler.handle_post("/api/v1/email/triage/test", {}, http)
        body = _parse(result)
        assert body["priority"] == "medium"

    def test_low_priority_match(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "subject": "Weekly Newsletter",
                "snippet": "Click to unsubscribe",
            },
        )
        result = handler.handle_post("/api/v1/email/triage/test", {}, http)
        body = _parse(result)
        assert body["priority"] == "low"

    def test_no_match(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "subject": "Hello there",
                "snippet": "Just saying hi",
            },
        )
        result = handler.handle_post("/api/v1/email/triage/test", {}, http)
        body = _parse(result)
        assert body["priority"] == "none"

    def test_escalation_flag(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "subject": "Legal notice from lawsuit",
                "snippet": "Urgent legal matter",
            },
        )
        result = handler.handle_post("/api/v1/email/triage/test", {}, http)
        body = _parse(result)
        assert body["should_escalate"] is True

    def test_missing_subject_and_snippet(self, handler):
        http = _make_mock_handler(
            method="POST",
            body={
                "from_address": "test@example.com",
            },
        )
        result = handler.handle_post("/api/v1/email/triage/test", {}, http)
        assert result.status_code == 400

    def test_returns_none_for_unhandled_path(self, handler):
        http = _make_mock_handler(method="POST", body={})
        result = handler.handle_post("/api/v1/other", {}, http)
        assert result is None
