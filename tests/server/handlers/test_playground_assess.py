"""Tests for the /api/v1/playground/assess endpoint in PlaygroundHandler.

Covers:
- Clear question returns { type: "ready", option: { ... } }
- Ambiguous question returns { type: "confirm", preflight: { options: [...] } }
- Timeout returns { type: "ready" } (never blocks the caller)
- Empty question returns { type: "ready" }
- Malformed JSON from model returns { type: "ready" }
- Model response wrapped in markdown code blocks is parsed correctly
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.playground import PlaygroundHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def handler() -> PlaygroundHandler:
    """Create a PlaygroundHandler instance for testing."""
    return PlaygroundHandler()


def _make_http_handler(body: dict) -> MagicMock:
    """Build a fake HTTP handler with a JSON body."""
    h = MagicMock()
    h.body = json.dumps(body).encode()
    h.client_address = ("127.0.0.1", 12345)
    return h


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandleAssess:
    """Tests for PlaygroundHandler._handle_assess."""

    def test_clear_question_returns_ready(self, handler: PlaygroundHandler) -> None:
        """When the model says the question is clear, return type=ready."""
        model_response = json.dumps({"clear": True, "topic": "Should we use Kubernetes?"})
        http_handler = _make_http_handler({"question": "Should we use Kubernetes?"})

        with patch.object(handler, "_call_frontier_model", return_value=model_response):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "ready"
        assert body["option"]["debatePrompt"] == "Should we use Kubernetes?"
        assert body["option"]["id"] == "original"

    def test_ambiguous_question_returns_confirm(self, handler: PlaygroundHandler) -> None:
        """When the model finds ambiguity, return type=confirm with interpretation options."""
        model_response = json.dumps(
            {
                "clear": False,
                "interpretations": [
                    "Should we use Kubernetes for our 5-person startup?",
                    "Is Kubernetes worth the complexity for enterprise-scale deployments?",
                    "Should we migrate from Docker Compose to Kubernetes?",
                ],
            }
        )
        http_handler = _make_http_handler({"question": "Should we use Kubernetes?"})

        with patch.object(handler, "_call_frontier_model", return_value=model_response):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "confirm"
        preflight = body["preflight"]
        assert preflight["title"] == "This question could mean a few things"
        # 3 interpretations + 1 "use original wording"
        assert len(preflight["options"]) == 4
        assert preflight["options"][0]["recommended"] is True
        assert preflight["options"][-1]["id"] == "original"
        assert preflight["options"][-1]["label"] == "Use original wording"
        # Each interpretation option carries the original question
        for opt in preflight["options"]:
            assert opt["originalQuestion"] == "Should we use Kubernetes?"

    def test_timeout_returns_ready(self, handler: PlaygroundHandler) -> None:
        """When the model call times out, gracefully return type=ready."""
        http_handler = _make_http_handler({"question": "Complex question here"})

        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError("timed out")):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "ready"
        assert body["option"]["debatePrompt"] == "Complex question here"

    def test_empty_question_returns_ready(self, handler: PlaygroundHandler) -> None:
        """An empty question should return type=ready immediately without calling the model."""
        http_handler = _make_http_handler({"question": ""})

        with patch.object(handler, "_call_frontier_model") as mock_call:
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "ready"
        # Should NOT have called the model at all
        mock_call.assert_not_called()

    def test_missing_question_field_returns_ready(self, handler: PlaygroundHandler) -> None:
        """When the body has no question field, return type=ready."""
        http_handler = _make_http_handler({})

        with patch.object(handler, "_call_frontier_model") as mock_call:
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "ready"
        mock_call.assert_not_called()

    def test_malformed_model_json_returns_ready(self, handler: PlaygroundHandler) -> None:
        """When the model returns unparseable garbage, gracefully return type=ready."""
        http_handler = _make_http_handler({"question": "Some question"})

        with patch.object(handler, "_call_frontier_model", return_value="not json at all"):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "ready"
        assert body["option"]["debatePrompt"] == "Some question"

    def test_model_response_in_markdown_code_block(self, handler: PlaygroundHandler) -> None:
        """When the model wraps JSON in ```json ... ```, we still parse it."""
        model_response = '```json\n{"clear": false, "interpretations": ["A", "B"]}\n```'
        http_handler = _make_http_handler({"question": "Test?"})

        with patch.object(handler, "_call_frontier_model", return_value=model_response):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "confirm"
        # 2 interpretations + 1 original
        assert len(body["preflight"]["options"]) == 3

    def test_connection_error_returns_ready(self, handler: PlaygroundHandler) -> None:
        """Network errors should not block — return type=ready."""
        http_handler = _make_http_handler({"question": "Something"})

        with patch.object(handler, "_call_frontier_model", side_effect=ConnectionError("refused")):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "ready"

    def test_runtime_error_returns_ready(self, handler: PlaygroundHandler) -> None:
        """RuntimeError (no agent available) should not block."""
        http_handler = _make_http_handler({"question": "Something"})

        with patch.object(
            handler,
            "_call_frontier_model",
            side_effect=RuntimeError("No frontier model available"),
        ):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        assert body["type"] == "ready"

    def test_interpretations_capped_at_four(self, handler: PlaygroundHandler) -> None:
        """At most 4 interpretation options (plus the original) are returned."""
        model_response = json.dumps(
            {
                "clear": False,
                "interpretations": [f"Interpretation {i}" for i in range(10)],
            }
        )
        http_handler = _make_http_handler({"question": "Broad question"})

        with patch.object(handler, "_call_frontier_model", return_value=model_response):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        # 4 interpretations + 1 original = 5
        assert len(body["preflight"]["options"]) == 5

    def test_option_fields_are_complete(self, handler: PlaygroundHandler) -> None:
        """Every option returned has all required fields."""
        model_response = json.dumps(
            {
                "clear": False,
                "interpretations": ["Version A", "Version B"],
            }
        )
        http_handler = _make_http_handler({"question": "My question"})

        with patch.object(handler, "_call_frontier_model", return_value=model_response):
            result = handler._handle_assess(http_handler)

        body = json.loads(result.body)
        for opt in body["preflight"]["options"]:
            assert "id" in opt
            assert "label" in opt
            assert "description" in opt
            assert "originalQuestion" in opt
            assert "interpretedQuestion" in opt
            assert "debatePrompt" in opt
            assert "agents" in opt
            assert "rounds" in opt


class TestBuildReadyOption:
    """Tests for PlaygroundHandler._build_ready_option."""

    def test_structure(self, handler: PlaygroundHandler) -> None:
        option = handler._build_ready_option("Test question?")
        assert option["id"] == "original"
        assert option["label"] == "Use original wording"
        assert option["description"] == "Test question?"
        assert option["originalQuestion"] == "Test question?"
        assert option["interpretedQuestion"] == "Test question?"
        assert option["debatePrompt"] == "Test question?"
        assert option["agents"] == 3
        assert option["rounds"] == 2

    def test_empty_question(self, handler: PlaygroundHandler) -> None:
        option = handler._build_ready_option("")
        assert option["debatePrompt"] == ""
