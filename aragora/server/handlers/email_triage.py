"""
Email Triage Rules Management Handler.

Provides API endpoints for managing email triage rules:
- GET  /api/v1/email/triage/rules  (list rules)
- PUT  /api/v1/email/triage/rules  (update rules)
- POST /api/v1/email/triage/test   (test a message against rules)
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.server.handlers.utils.decorators import require_permission

logger = logging.getLogger(__name__)

# Module-level engine instance (lazy initialized)
_engine = None


def _get_engine():
    """Get or create the triage rule engine."""
    global _engine
    if _engine is None:
        from aragora.analysis.email_triage import TriageConfig, TriageRuleEngine

        _engine = TriageRuleEngine(TriageConfig())
    return _engine


def _set_engine(engine):
    """Set the engine instance (used for testing)."""
    global _engine
    _engine = engine


class EmailTriageHandler(BaseHandler):
    """Handler for email triage rules management."""

    ROUTES = [
        "/api/v1/email/triage/rules",
        "/api/v1/email/triage/test",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None, **kwargs: Any):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        return path in self.ROUTES

    @handle_errors("get triage rules")
    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        if path == "/api/v1/email/triage/rules":
            return self._handle_get_rules()
        return None

    @require_permission("email:manage_rules")
    @handle_errors("update triage rules")
    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        if path == "/api/v1/email/triage/rules":
            return self._handle_update_rules(handler)
        return None

    @handle_errors("test triage rules")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        if path == "/api/v1/email/triage/test":
            return self._handle_test_message(handler)
        return None

    def _handle_get_rules(self) -> HandlerResult:
        """Return current triage rules."""
        engine = _get_engine()
        config = engine.config

        rules = []
        for rule in config.rules:
            rules.append({
                "label": rule.label,
                "keywords": rule.keywords,
                "priority": rule.priority,
            })

        return json_response({
            "rules": rules,
            "escalation_keywords": config.escalation_keywords,
            "auto_handle_threshold": config.auto_handle_threshold,
            "sync_interval_minutes": config.sync_interval_minutes,
        })

    def _handle_update_rules(self, handler: Any) -> HandlerResult:
        """Update triage rules from request body."""
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        from aragora.analysis.email_triage import TriageConfig, TriageRuleEngine

        try:
            # Build config from body
            config_data = {}
            if "rules" in body:
                # Convert flat rules list to priority_rules format
                priority_rules: dict[str, list[dict]] = {}
                for rule in body["rules"]:
                    priority = rule.get("priority", "medium")
                    if priority not in ("high", "medium", "low"):
                        return error_response(
                            f"Invalid priority '{priority}'. Must be high, medium, or low",
                            400,
                        )
                    priority_rules.setdefault(priority, []).append({
                        "label": rule.get("label", ""),
                        "keywords": rule.get("keywords", []),
                    })
                config_data["priority_rules"] = priority_rules

            if "escalation_keywords" in body:
                config_data["escalation"] = {
                    "always_flag": body["escalation_keywords"],
                    "auto_handle_threshold": body.get(
                        "auto_handle_threshold",
                        _get_engine().config.auto_handle_threshold,
                    ),
                }

            config = TriageConfig.from_dict(config_data)
            new_engine = TriageRuleEngine(config)
            _set_engine(new_engine)

            logger.info("Triage rules updated: %d rules", len(config.rules))

            return json_response({
                "message": "Triage rules updated",
                "rules_count": len(config.rules),
            })

        except (ValueError, TypeError, KeyError) as e:
            return error_response(f"Invalid rules configuration: {e}", 400)

    def _handle_test_message(self, handler: Any) -> HandlerResult:
        """Test a message against current triage rules."""
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        subject = body.get("subject", "")
        from_address = body.get("from_address", "")
        snippet = body.get("snippet", "")
        labels = body.get("labels")

        if not subject and not snippet:
            return error_response("At least 'subject' or 'snippet' is required", 400)

        engine = _get_engine()
        score = engine.apply_rules(
            subject=subject,
            from_address=from_address,
            snippet=snippet,
            labels=labels,
        )

        return json_response({
            "priority": score.priority,
            "matched_rule": score.matched_rule,
            "score_boost": score.score_boost,
            "should_escalate": score.should_escalate,
        })


__all__ = ["EmailTriageHandler"]
