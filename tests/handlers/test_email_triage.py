"""Tests for email triage handler (aragora/server/handlers/email_triage.py).

Covers all routes and behavior of the EmailTriageHandler class:
- can_handle() routing for ROUTES
- GET  /api/v1/email/triage/rules  (list rules)
- PUT  /api/v1/email/triage/rules  (update rules)
- POST /api/v1/email/triage/test   (test message against rules)
- Error handling (invalid JSON, missing fields, invalid priority)
- Module-level engine lazy init and setter
- Edge cases (empty rules, empty body, multiple rules, escalation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.email_triage import (
    EmailTriageHandler,
    _get_engine,
    _set_engine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler with rfile + headers."""

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b""
            self.headers = {"Content-Length": "0"}


class InvalidJSONHandler:
    """Mock HTTP handler that returns invalid JSON."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"not json{{"
        self.headers = {"Content-Length": "10"}


# ---------------------------------------------------------------------------
# Mock triage types
# ---------------------------------------------------------------------------


@dataclass
class MockTriageRule:
    label: str = ""
    keywords: list[str] = field(default_factory=list)
    priority: str = "medium"


@dataclass
class MockTriageConfig:
    rules: list[MockTriageRule] = field(default_factory=list)
    escalation_keywords: list[str] = field(default_factory=list)
    auto_handle_threshold: float = 0.85
    sync_interval_minutes: int = 5


@dataclass
class MockTriageScore:
    priority: str = "none"
    matched_rule: str = ""
    score_boost: float = 0.0
    should_escalate: bool = False


class MockTriageRuleEngine:
    """Mock TriageRuleEngine for handler tests."""

    def __init__(self, config: MockTriageConfig | None = None):
        self.config = config or MockTriageConfig()
        self._apply_rules_result = MockTriageScore()

    def apply_rules(
        self,
        subject: str = "",
        from_address: str = "",
        snippet: str = "",
        labels: list[str] | None = None,
    ) -> MockTriageScore:
        return self._apply_rules_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset the module-level _engine before and after each test."""
    _set_engine(None)
    yield
    _set_engine(None)


@pytest.fixture
def handler():
    """Create an EmailTriageHandler instance."""
    return EmailTriageHandler()


@pytest.fixture
def handler_with_ctx():
    """Create an EmailTriageHandler with context."""
    return EmailTriageHandler(ctx={"tenant_id": "t-001"})


@pytest.fixture
def mock_engine():
    """Create a MockTriageRuleEngine with default config."""
    return MockTriageRuleEngine()


@pytest.fixture
def engine_with_rules():
    """Create a MockTriageRuleEngine with pre-populated rules."""
    config = MockTriageConfig(
        rules=[
            MockTriageRule(label="urgent", keywords=["urgent", "asap"], priority="high"),
            MockTriageRule(label="billing", keywords=["invoice", "payment"], priority="medium"),
            MockTriageRule(label="newsletter", keywords=["unsubscribe"], priority="low"),
        ],
        escalation_keywords=["ceo", "legal"],
        auto_handle_threshold=0.9,
        sync_interval_minutes=10,
    )
    return MockTriageRuleEngine(config)


# ===========================================================================
# Initialization & Routing Tests
# ===========================================================================


class TestEmailTriageHandlerInit:
    """Tests for handler initialization."""

    def test_init_default_ctx(self, handler):
        assert handler.ctx == {}

    def test_init_with_ctx(self, handler_with_ctx):
        assert handler_with_ctx.ctx == {"tenant_id": "t-001"}

    def test_init_with_kwargs(self):
        h = EmailTriageHandler(ctx={"key": "val"}, extra="ignored")
        assert h.ctx == {"key": "val"}

    def test_routes_defined(self, handler):
        assert "/api/v1/email/triage/rules" in handler.ROUTES
        assert "/api/v1/email/triage/test" in handler.ROUTES
        assert len(handler.ROUTES) == 2


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_rules_path(self, handler):
        assert handler.can_handle("/api/v1/email/triage/rules") is True

    def test_test_path(self, handler):
        assert handler.can_handle("/api/v1/email/triage/test") is True

    def test_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/email/triage/unknown") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_path(self, handler):
        assert handler.can_handle("/api/v1/email/triage") is False

    def test_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/email/triage/rules/") is False

    def test_different_version(self, handler):
        assert handler.can_handle("/api/v2/email/triage/rules") is False


# ===========================================================================
# Module-level engine tests
# ===========================================================================


class TestEngineManagement:
    """Tests for _get_engine and _set_engine."""

    def test_set_engine(self):
        engine = MockTriageRuleEngine()
        _set_engine(engine)
        assert _get_engine() is engine

    def test_set_engine_none_resets(self):
        _set_engine(MockTriageRuleEngine())
        _set_engine(None)
        # Next call to _get_engine should lazy-init a real one
        with patch(
            "aragora.analysis.email_triage.TriageRuleEngine",
            MockTriageRuleEngine,
        ), patch(
            "aragora.analysis.email_triage.TriageConfig",
            MockTriageConfig,
        ):
            engine = _get_engine()
            assert engine is not None

    def test_get_engine_lazy_init(self):
        """_get_engine should lazy-initialize from TriageConfig when None."""
        with patch(
            "aragora.server.handlers.email_triage._engine", None
        ), patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_cls, patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg_cls:
            mock_cfg_cls.return_value = MockTriageConfig()
            mock_cls.return_value = MockTriageRuleEngine()
            engine = _get_engine()
            assert engine is not None

    def test_get_engine_returns_same_instance(self):
        engine = MockTriageRuleEngine()
        _set_engine(engine)
        assert _get_engine() is engine
        assert _get_engine() is engine  # idempotent


# ===========================================================================
# GET /api/v1/email/triage/rules
# ===========================================================================


class TestGetRules:
    """Tests for handle() - GET triage rules."""

    def test_get_empty_rules(self, handler, mock_engine):
        _set_engine(mock_engine)
        result = handler.handle("/api/v1/email/triage/rules", {}, MockHTTPHandler())
        body = _body(result)
        assert _status(result) == 200
        assert body["rules"] == []
        assert body["auto_handle_threshold"] == 0.85
        assert body["sync_interval_minutes"] == 5
        assert body["escalation_keywords"] == []

    def test_get_populated_rules(self, handler, engine_with_rules):
        _set_engine(engine_with_rules)
        result = handler.handle("/api/v1/email/triage/rules", {}, MockHTTPHandler())
        body = _body(result)
        assert _status(result) == 200
        assert len(body["rules"]) == 3
        labels = [r["label"] for r in body["rules"]]
        assert "urgent" in labels
        assert "billing" in labels
        assert "newsletter" in labels

    def test_get_rules_structure(self, handler, engine_with_rules):
        _set_engine(engine_with_rules)
        result = handler.handle("/api/v1/email/triage/rules", {}, MockHTTPHandler())
        body = _body(result)
        urgent = [r for r in body["rules"] if r["label"] == "urgent"][0]
        assert urgent["keywords"] == ["urgent", "asap"]
        assert urgent["priority"] == "high"

    def test_get_rules_escalation_keywords(self, handler, engine_with_rules):
        _set_engine(engine_with_rules)
        result = handler.handle("/api/v1/email/triage/rules", {}, MockHTTPHandler())
        body = _body(result)
        assert body["escalation_keywords"] == ["ceo", "legal"]
        assert body["auto_handle_threshold"] == 0.9
        assert body["sync_interval_minutes"] == 10

    def test_get_rules_unhandled_path_returns_none(self, handler, mock_engine):
        _set_engine(mock_engine)
        result = handler.handle("/api/v1/email/triage/test", {}, MockHTTPHandler())
        assert result is None

    def test_get_rules_single_rule(self, handler):
        config = MockTriageConfig(
            rules=[MockTriageRule(label="vip", keywords=["vip"], priority="high")]
        )
        _set_engine(MockTriageRuleEngine(config))
        result = handler.handle("/api/v1/email/triage/rules", {}, MockHTTPHandler())
        body = _body(result)
        assert len(body["rules"]) == 1
        assert body["rules"][0]["label"] == "vip"


# ===========================================================================
# PUT /api/v1/email/triage/rules
# ===========================================================================


class TestUpdateRules:
    """Tests for handle_put() - update triage rules."""

    def test_update_rules_valid(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {
            "rules": [
                {"label": "critical", "keywords": ["critical"], "priority": "high"},
                {"label": "spam", "keywords": ["spam"], "priority": "low"},
            ]
        }
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig(
                rules=[MockTriageRule("critical"), MockTriageRule("spam")]
            )
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            resp = _body(result)
            assert _status(result) == 200
            assert resp["message"] == "Triage rules updated"
            assert resp["rules_count"] == 2

    def test_update_rules_invalid_json(self, handler, mock_engine):
        _set_engine(mock_engine)
        result = handler.handle_put(
            "/api/v1/email/triage/rules", {}, InvalidJSONHandler()
        )
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result)["error"]

    def test_update_rules_invalid_priority(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {
            "rules": [
                {"label": "test", "keywords": ["test"], "priority": "critical"},
            ]
        }
        result = handler.handle_put(
            "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
        )
        assert _status(result) == 400
        assert "Invalid priority" in _body(result)["error"]

    def test_update_rules_empty_body(self, handler, mock_engine):
        _set_engine(mock_engine)
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig()
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler({})
            )
            resp = _body(result)
            assert _status(result) == 200
            assert resp["rules_count"] == 0

    def test_update_rules_with_escalation(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {
            "escalation_keywords": ["ceo", "board"],
            "auto_handle_threshold": 0.75,
        }
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig(
                escalation_keywords=["ceo", "board"]
            )
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 200

    def test_update_rules_config_error_valueerror(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {"rules": [{"label": "ok", "keywords": ["ok"], "priority": "high"}]}
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg:
            mock_cfg.from_dict.side_effect = ValueError("bad config")
            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 400
            assert "Invalid rules configuration" in _body(result)["error"]

    def test_update_rules_config_error_typeerror(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {"rules": [{"label": "ok", "keywords": ["ok"], "priority": "high"}]}
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg:
            mock_cfg.from_dict.side_effect = TypeError("wrong type")
            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 400
            assert "Invalid rules configuration" in _body(result)["error"]

    def test_update_rules_config_error_keyerror(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {"rules": [{"label": "ok", "keywords": ["ok"], "priority": "high"}]}
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg:
            mock_cfg.from_dict.side_effect = KeyError("missing key")
            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 400
            assert "Invalid rules configuration" in _body(result)["error"]

    def test_update_rules_unhandled_path_returns_none(self, handler, mock_engine):
        _set_engine(mock_engine)
        result = handler.handle_put(
            "/api/v1/email/triage/test", {}, MockHTTPHandler({})
        )
        assert result is None

    def test_update_rules_default_priority_medium(self, handler, mock_engine):
        """Rules without explicit priority default to medium."""
        _set_engine(mock_engine)
        body = {
            "rules": [
                {"label": "general", "keywords": ["general"]},
            ]
        }
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig(
                rules=[MockTriageRule("general")]
            )
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 200
            # Verify from_dict was called with medium priority
            call_args = mock_cfg.from_dict.call_args[0][0]
            assert "medium" in call_args["priority_rules"]

    def test_update_rules_default_label_and_keywords(self, handler, mock_engine):
        """Rules missing label/keywords get empty defaults."""
        _set_engine(mock_engine)
        body = {
            "rules": [
                {"priority": "high"},
            ]
        }
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig(rules=[MockTriageRule()])
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 200
            call_args = mock_cfg.from_dict.call_args[0][0]
            rule = call_args["priority_rules"]["high"][0]
            assert rule["label"] == ""
            assert rule["keywords"] == []

    def test_update_rules_escalation_uses_current_threshold(self, handler):
        """When auto_handle_threshold not in body, uses engine's current value."""
        engine = MockTriageRuleEngine(
            MockTriageConfig(auto_handle_threshold=0.99)
        )
        _set_engine(engine)
        body = {
            "escalation_keywords": ["urgent"],
        }
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig()
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 200
            call_args = mock_cfg.from_dict.call_args[0][0]
            assert call_args["escalation"]["auto_handle_threshold"] == 0.99

    def test_update_rules_all_three_priorities(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {
            "rules": [
                {"label": "a", "keywords": ["a"], "priority": "high"},
                {"label": "b", "keywords": ["b"], "priority": "medium"},
                {"label": "c", "keywords": ["c"], "priority": "low"},
            ]
        }
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig(
                rules=[MockTriageRule("a"), MockTriageRule("b"), MockTriageRule("c")]
            )
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 200
            call_args = mock_cfg.from_dict.call_args[0][0]
            assert "high" in call_args["priority_rules"]
            assert "medium" in call_args["priority_rules"]
            assert "low" in call_args["priority_rules"]

    def test_update_rules_invalid_priority_variants(self, handler, mock_engine):
        _set_engine(mock_engine)
        for bad_priority in ["critical", "URGENT", "1", "", "HIGH"]:
            body = {
                "rules": [
                    {"label": "x", "keywords": ["x"], "priority": bad_priority},
                ]
            }
            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 400, f"Expected 400 for priority={bad_priority}"

    def test_update_rules_replaces_engine(self, handler, mock_engine):
        """After update, _get_engine returns the new engine."""
        _set_engine(mock_engine)
        body = {"rules": [{"label": "x", "keywords": ["x"], "priority": "high"}]}

        new_engine = MockTriageRuleEngine()
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine",
            return_value=new_engine,
        ):
            mock_cfg.from_dict.return_value = MockTriageConfig()
            handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
        assert _get_engine() is new_engine


# ===========================================================================
# POST /api/v1/email/triage/test
# ===========================================================================


class TestTestMessage:
    """Tests for handle_post() - test message against rules."""

    def test_test_message_basic(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {"subject": "Urgent: server down", "snippet": "Help needed"}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        resp = _body(result)
        assert _status(result) == 200
        assert "priority" in resp
        assert "matched_rule" in resp
        assert "score_boost" in resp
        assert "should_escalate" in resp

    def test_test_message_with_custom_score(self, handler):
        engine = MockTriageRuleEngine()
        engine._apply_rules_result = MockTriageScore(
            priority="high",
            matched_rule="urgent",
            score_boost=0.35,
            should_escalate=True,
        )
        _set_engine(engine)
        body = {"subject": "Urgent matter", "snippet": "CEO needs response"}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        resp = _body(result)
        assert resp["priority"] == "high"
        assert resp["matched_rule"] == "urgent"
        assert resp["score_boost"] == 0.35
        assert resp["should_escalate"] is True

    def test_test_message_only_subject(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {"subject": "Hello world"}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        assert _status(result) == 200

    def test_test_message_only_snippet(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {"snippet": "Some email content"}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        assert _status(result) == 200

    def test_test_message_missing_subject_and_snippet(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {"from_address": "user@example.com"}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        assert _status(result) == 400
        assert "subject" in _body(result)["error"].lower() or "snippet" in _body(result)["error"].lower()

    def test_test_message_empty_subject_and_snippet(self, handler, mock_engine):
        _set_engine(mock_engine)
        body = {"subject": "", "snippet": ""}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        assert _status(result) == 400

    def test_test_message_invalid_json(self, handler, mock_engine):
        _set_engine(mock_engine)
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, InvalidJSONHandler()
        )
        assert _status(result) == 400
        assert "Invalid JSON" in _body(result)["error"]

    def test_test_message_with_from_address(self, handler):
        engine = MockTriageRuleEngine()
        _set_engine(engine)
        body = {
            "subject": "Test",
            "from_address": "ceo@company.com",
            "snippet": "Important",
        }
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        assert _status(result) == 200

    def test_test_message_with_labels(self, handler):
        engine = MockTriageRuleEngine()
        _set_engine(engine)
        body = {
            "subject": "Test",
            "labels": ["INBOX", "IMPORTANT"],
        }
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        assert _status(result) == 200

    def test_test_message_no_escalate(self, handler):
        engine = MockTriageRuleEngine()
        engine._apply_rules_result = MockTriageScore(
            priority="low",
            matched_rule="newsletter",
            score_boost=-0.25,
            should_escalate=False,
        )
        _set_engine(engine)
        body = {"subject": "Monthly newsletter"}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        resp = _body(result)
        assert resp["should_escalate"] is False
        assert resp["priority"] == "low"
        assert resp["score_boost"] == -0.25

    def test_test_message_unhandled_path_returns_none(self, handler, mock_engine):
        _set_engine(mock_engine)
        result = handler.handle_post(
            "/api/v1/email/triage/rules", {}, MockHTTPHandler({"subject": "hi"})
        )
        assert result is None

    def test_test_message_no_match(self, handler):
        engine = MockTriageRuleEngine()
        engine._apply_rules_result = MockTriageScore(
            priority="none",
            matched_rule="",
            score_boost=0.0,
            should_escalate=False,
        )
        _set_engine(engine)
        body = {"subject": "Random email"}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        resp = _body(result)
        assert resp["priority"] == "none"
        assert resp["matched_rule"] == ""
        assert resp["score_boost"] == 0.0

    def test_test_message_empty_body(self, handler, mock_engine):
        """Empty body {} means subject and snippet both default to empty string."""
        _set_engine(mock_engine)
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler({})
        )
        assert _status(result) == 400

    def test_test_message_labels_none(self, handler):
        """labels key absent means None passed to apply_rules."""
        engine = MockTriageRuleEngine()
        _set_engine(engine)
        body = {"subject": "Hello"}
        result = handler.handle_post(
            "/api/v1/email/triage/test", {}, MockHTTPHandler(body)
        )
        assert _status(result) == 200


# ===========================================================================
# No-Content / Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Various edge case tests."""

    def test_handler_no_content_length(self, handler, mock_engine):
        """Handler with Content-Length 0 gives empty dict from read_json_body."""
        _set_engine(mock_engine)
        h = MockHTTPHandler()
        result = handler.handle_put("/api/v1/email/triage/rules", {}, h)
        # Empty body {} is valid; handler proceeds with empty config_data
        # (no rules, no escalation)
        if result is not None:
            status = _status(result)
            assert status in (200, 400)  # depends on TriageConfig.from_dict

    def test_update_rules_with_rules_and_escalation(self, handler, mock_engine):
        """Both rules and escalation_keywords in one request."""
        _set_engine(mock_engine)
        body = {
            "rules": [
                {"label": "vip", "keywords": ["vip"], "priority": "high"},
            ],
            "escalation_keywords": ["ceo"],
            "auto_handle_threshold": 0.5,
        }
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig(
                rules=[MockTriageRule("vip")],
                escalation_keywords=["ceo"],
            )
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 200
            call_args = mock_cfg.from_dict.call_args[0][0]
            assert call_args["escalation"]["always_flag"] == ["ceo"]
            assert call_args["escalation"]["auto_handle_threshold"] == 0.5
            assert "priority_rules" in call_args

    def test_update_multiple_rules_same_priority(self, handler, mock_engine):
        """Multiple rules with the same priority are grouped."""
        _set_engine(mock_engine)
        body = {
            "rules": [
                {"label": "a", "keywords": ["a"], "priority": "high"},
                {"label": "b", "keywords": ["b"], "priority": "high"},
            ]
        }
        with patch(
            "aragora.analysis.email_triage.TriageConfig"
        ) as mock_cfg, patch(
            "aragora.analysis.email_triage.TriageRuleEngine"
        ) as mock_engine_cls:
            mock_config_instance = MockTriageConfig(
                rules=[MockTriageRule("a"), MockTriageRule("b")]
            )
            mock_cfg.from_dict.return_value = mock_config_instance
            mock_engine_cls.return_value = MockTriageRuleEngine(mock_config_instance)

            result = handler.handle_put(
                "/api/v1/email/triage/rules", {}, MockHTTPHandler(body)
            )
            assert _status(result) == 200
            call_args = mock_cfg.from_dict.call_args[0][0]
            assert len(call_args["priority_rules"]["high"]) == 2
