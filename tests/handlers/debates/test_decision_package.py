"""Comprehensive tests for the DecisionPackageHandler.

Routes tested:
    GET /api/v1/debates/{id}/package          - JSON decision package
    GET /api/v1/debates/{id}/package/markdown  - Markdown export

Covers:
- DecisionPackageHandler instantiation, ROUTES, ctx defaults
- can_handle routing (accept/reject paths)
- _extract_debate_id for various path formats
- handle() dispatching to JSON vs markdown
- handle() missing debate ID
- _assemble_package success paths (receipt, no receipt, various verdicts)
- _assemble_package error paths (no storage, missing debate, incomplete status)
- Receipt store graceful degradation on import/runtime errors
- Argument map graceful degradation on import/runtime errors
- _handle_json: full JSON response structure validation
- _handle_markdown: markdown rendering, content type, encoding
- _generate_next_steps: all verdict/confidence/consensus combos
- _build_markdown: all optional sections, missing data
- get_storage helper
- Edge cases: empty debate data, None result, special characters
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.debates.decision_package import (
    DecisionPackageHandler,
    _build_markdown,
    _generate_next_steps,
)
from aragora.server.handlers.utils.responses import HandlerResult


@pytest.fixture(autouse=True)
def _isolate_stores(monkeypatch):
    """Prevent lazy store imports from hitting real PostgreSQL."""
    monkeypatch.delenv("ARAGORA_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("SUPABASE_POSTGRES_DSN", raising=False)
    # Reset global singletons that may have been cached from prior tests
    import aragora.storage.webhook_config_store as _wcs

    monkeypatch.setattr(_wcs, "_webhook_config_store", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract JSON body from a HandlerResult."""
    if result is None:
        return {}
    raw = result.body
    if isinstance(raw, bytes):
        return json.loads(raw.decode())
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _status(result: HandlerResult) -> int:
    """Extract status code from a HandlerResult."""
    if result is None:
        return 0
    return result.status_code


def _text(result: HandlerResult) -> str:
    """Extract text body from a HandlerResult."""
    if result is None:
        return ""
    raw = result.body
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _make_http_handler(body: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock HTTP handler object."""
    h = MagicMock()
    h.command = "GET"
    h.client_address = ("10.0.0.1", 12345)
    if body:
        body_bytes = json.dumps(body).encode()
        h.rfile.read.return_value = body_bytes
        h.headers = {"Content-Length": str(len(body_bytes))}
    else:
        h.rfile.read.return_value = b"{}"
        h.headers = {"Content-Length": "2"}
    return h


def _make_storage(debates: dict[str, dict] | None = None) -> MagicMock:
    """Create a mock storage that returns debates by ID."""
    storage = MagicMock()
    store = debates or {}

    def get_debate(debate_id: str):
        return store.get(debate_id)

    storage.get_debate = MagicMock(side_effect=get_debate)
    return storage


def _completed_debate(
    debate_id: str = "test-debate-001",
    question: str = "Should we use microservices?",
    confidence: float = 0.85,
    consensus_reached: bool = True,
    status: str = "completed",
    final_answer: str = "Yes, microservices are recommended.",
    explanation_summary: str = "All agents agreed on microservices.",
    participants: list[str] | None = None,
    messages: list[dict] | None = None,
    agents: list[str] | None = None,
    total_cost_usd: float = 0.0042,
    per_agent_cost: dict | None = None,
) -> dict[str, Any]:
    """Build a completed debate dict for testing."""
    return {
        "debate_id": debate_id,
        "question": question,
        "status": status,
        "agents": agents or ["claude", "gpt-4", "gemini"],
        "messages": messages
        or [
            {"agent": "claude", "role": "proposal", "content": "I propose X.", "round": 1},
            {"agent": "gpt-4", "role": "critique", "content": "I critique X.", "round": 1},
        ],
        "result": {
            "confidence": confidence,
            "consensus_reached": consensus_reached,
            "final_answer": final_answer,
            "explanation_summary": explanation_summary,
            "participants": participants or ["claude", "gpt-4", "gemini"],
            "total_cost_usd": total_cost_usd,
            "per_agent_cost": per_agent_cost or {"claude": 0.0020, "gpt-4": 0.0022},
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a DecisionPackageHandler with no storage."""
    return DecisionPackageHandler()


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler."""
    return _make_http_handler()


@pytest.fixture
def storage_with_debate():
    """Storage with a single completed debate."""
    debate = _completed_debate()
    return _make_storage({"test-debate-001": debate})


@pytest.fixture
def handler_with_storage(storage_with_debate):
    """Handler with storage containing a completed debate."""
    return DecisionPackageHandler(ctx={"storage": storage_with_debate})


# ===========================================================================
# Instantiation and configuration
# ===========================================================================


class TestInstantiation:
    """Test handler creation and configuration."""

    def test_default_ctx(self):
        h = DecisionPackageHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"storage": MagicMock(), "extra": "data"}
        h = DecisionPackageHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_defaults_to_empty_dict(self):
        h = DecisionPackageHandler(ctx=None)
        assert h.ctx == {}

    def test_routes_defined(self):
        assert "/api/v1/debates/*/package" in DecisionPackageHandler.ROUTES
        assert "/api/v1/debates/*/package/markdown" in DecisionPackageHandler.ROUTES

    def test_routes_count(self):
        assert len(DecisionPackageHandler.ROUTES) == 2


# ===========================================================================
# can_handle routing
# ===========================================================================


class TestCanHandle:
    """Test path routing via can_handle."""

    def test_json_package_path(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-123/package") is True

    def test_markdown_package_path(self, handler):
        assert handler.can_handle("/api/v1/debates/abc-123/package/markdown") is True

    def test_wrong_prefix(self, handler):
        assert handler.can_handle("/api/v2/debates/abc/package") is False

    def test_missing_package_segment(self, handler):
        assert handler.can_handle("/api/v1/debates/abc") is False

    def test_extra_trailing_segment(self, handler):
        assert handler.can_handle("/api/v1/debates/abc/package/markdown/extra") is False

    def test_wrong_last_segment(self, handler):
        assert handler.can_handle("/api/v1/debates/abc/package/html") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/agents/list") is False

    def test_debates_list_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_package_without_debate_id(self, handler):
        # /api/v1/debates/package -> parts[4]='package', parts[5] doesn't exist
        assert handler.can_handle("/api/v1/debates/package") is False

    def test_numeric_debate_id(self, handler):
        assert handler.can_handle("/api/v1/debates/12345/package") is True

    def test_uuid_debate_id(self, handler):
        assert (
            handler.can_handle("/api/v1/debates/550e8400-e29b-41d4-a716-446655440000/package")
            is True
        )

    def test_markdown_with_uuid(self, handler):
        assert handler.can_handle("/api/v1/debates/550e8400/package/markdown") is True

    def test_no_api_prefix(self, handler):
        assert handler.can_handle("/debates/abc/package") is False

    def test_wrong_api_version_segment(self, handler):
        assert handler.can_handle("/api/v3/debates/abc/package") is False


# ===========================================================================
# _extract_debate_id
# ===========================================================================


class TestExtractDebateId:
    """Test debate ID extraction from path."""

    def test_json_path(self, handler):
        assert handler._extract_debate_id("/api/v1/debates/my-id/package") == "my-id"

    def test_markdown_path(self, handler):
        assert handler._extract_debate_id("/api/v1/debates/my-id/package/markdown") == "my-id"

    def test_uuid_id(self, handler):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert handler._extract_debate_id(f"/api/v1/debates/{uid}/package") == uid

    def test_too_short_path(self, handler):
        assert handler._extract_debate_id("/api/v1/debates") is None

    def test_short_path_four_parts(self, handler):
        # /api/v1/debates -> 4 parts after split, so parts[4] doesn't exist
        assert handler._extract_debate_id("/api/v1") is None

    def test_empty_path(self, handler):
        assert handler._extract_debate_id("") is None

    def test_numeric_id(self, handler):
        assert handler._extract_debate_id("/api/v1/debates/42/package") == "42"


# ===========================================================================
# get_storage
# ===========================================================================


class TestGetStorage:
    """Test storage retrieval from context."""

    def test_returns_storage_from_ctx(self):
        storage = MagicMock()
        h = DecisionPackageHandler(ctx={"storage": storage})
        assert h.get_storage() is storage

    def test_returns_none_when_no_storage(self):
        h = DecisionPackageHandler(ctx={})
        assert h.get_storage() is None

    def test_returns_none_for_empty_ctx(self):
        h = DecisionPackageHandler()
        assert h.get_storage() is None


# ===========================================================================
# _generate_next_steps (module-level function)
# ===========================================================================


class TestGenerateNextSteps:
    """Test the next steps generation logic."""

    def test_approved_high_confidence(self):
        steps = _generate_next_steps("APPROVED", 0.9, True, "test question")
        actions = [s["action"] for s in steps]
        assert "Proceed with implementation" in actions
        assert "Document decision rationale for audit trail" in actions

    def test_approved_low_confidence(self):
        steps = _generate_next_steps("APPROVED", 0.4, True, "test question")
        actions = [s["action"] for s in steps]
        # Low confidence still "APPROVED" but confidence < 0.8 falls through to else
        # Actually verdict == "APPROVED" but confidence < 0.8 -> falls to else branch
        assert "Review debate results and determine next action" in actions
        # Also low confidence appended
        assert any("Low confidence" in a for a in actions)

    def test_approved_exactly_0_8(self):
        steps = _generate_next_steps("APPROVED", 0.8, True, "q")
        actions = [s["action"] for s in steps]
        assert "Proceed with implementation" in actions

    def test_approved_with_conditions(self):
        steps = _generate_next_steps("APPROVED_WITH_CONDITIONS", 0.7, True, "q")
        actions = [s["action"] for s in steps]
        assert "Address conditions before proceeding" in actions
        assert "Schedule follow-up review after conditions met" in actions
        assert "Document conditions and acceptance criteria" in actions

    def test_needs_review_with_consensus(self):
        steps = _generate_next_steps("NEEDS_REVIEW", 0.6, True, "q")
        actions = [s["action"] for s in steps]
        assert "Escalate to human decision-maker" in actions
        assert "Gather additional evidence or expert input" in actions
        # Consensus reached, so no follow-up debate suggestion
        assert not any("follow-up debate" in a for a in actions)

    def test_needs_review_no_consensus(self):
        steps = _generate_next_steps("NEEDS_REVIEW", 0.6, False, "q")
        actions = [s["action"] for s in steps]
        assert any("follow-up debate" in a for a in actions)

    def test_unknown_verdict(self):
        steps = _generate_next_steps("UNKNOWN", 0.7, False, "q")
        actions = [s["action"] for s in steps]
        assert "Review debate results and determine next action" in actions

    def test_low_confidence_appended_for_any_verdict(self):
        for verdict in ["APPROVED", "APPROVED_WITH_CONDITIONS", "NEEDS_REVIEW", "UNKNOWN"]:
            steps = _generate_next_steps(verdict, 0.3, True, "q")
            actions = [s["action"] for s in steps]
            assert any("Low confidence" in a for a in actions), (
                f"No low-confidence step for {verdict}"
            )

    def test_high_confidence_no_low_confidence_step(self):
        steps = _generate_next_steps("APPROVED", 0.9, True, "q")
        actions = [s["action"] for s in steps]
        assert not any("Low confidence" in a for a in actions)

    def test_steps_have_priority_field(self):
        steps = _generate_next_steps("APPROVED", 0.9, True, "q")
        for step in steps:
            assert "priority" in step
            assert step["priority"] in ("high", "medium", "low")

    def test_steps_have_action_field(self):
        steps = _generate_next_steps("NEEDS_REVIEW", 0.3, False, "q")
        for step in steps:
            assert "action" in step
            assert isinstance(step["action"], str)

    def test_boundary_confidence_0_5(self):
        """Confidence == 0.5 should NOT trigger low confidence step."""
        steps = _generate_next_steps("APPROVED", 0.5, True, "q")
        actions = [s["action"] for s in steps]
        assert not any("Low confidence" in a for a in actions)

    def test_boundary_confidence_0_49(self):
        """Confidence < 0.5 should trigger low confidence step."""
        steps = _generate_next_steps("APPROVED", 0.49, True, "q")
        actions = [s["action"] for s in steps]
        assert any("Low confidence" in a for a in actions)

    def test_approved_with_conditions_low_confidence(self):
        steps = _generate_next_steps("APPROVED_WITH_CONDITIONS", 0.2, True, "q")
        actions = [s["action"] for s in steps]
        # Should have conditions steps AND low confidence step
        assert "Address conditions before proceeding" in actions
        assert any("Low confidence" in a for a in actions)

    def test_needs_review_low_confidence_no_consensus(self):
        steps = _generate_next_steps("NEEDS_REVIEW", 0.1, False, "q")
        actions = [s["action"] for s in steps]
        # Should have all: escalate, gather evidence, follow-up, low confidence
        assert "Escalate to human decision-maker" in actions
        assert any("follow-up debate" in a for a in actions)
        assert any("Low confidence" in a for a in actions)


# ===========================================================================
# _build_markdown (module-level function)
# ===========================================================================


class TestBuildMarkdown:
    """Test markdown rendering."""

    def test_title_uses_debate_id(self):
        md = _build_markdown({"debate_id": "d-001"})
        assert "# Decision Package: d-001" in md

    def test_title_unknown_when_no_id(self):
        md = _build_markdown({})
        assert "# Decision Package: Unknown" in md

    def test_summary_section_present(self):
        md = _build_markdown(
            {
                "question": "To be or not to be?",
                "verdict": "APPROVED",
                "confidence": 0.92,
                "consensus_reached": True,
                "status": "completed",
            }
        )
        assert "## Summary" in md
        assert "To be or not to be?" in md
        assert "APPROVED" in md
        assert "92%" in md
        assert "Yes" in md
        assert "completed" in md

    def test_consensus_no(self):
        md = _build_markdown({"consensus_reached": False})
        assert "**Consensus:** No" in md

    def test_final_answer_section(self):
        md = _build_markdown({"final_answer": "The answer is 42."})
        assert "## Final Answer" in md
        assert "The answer is 42." in md

    def test_no_final_answer_when_empty(self):
        md = _build_markdown({"final_answer": ""})
        assert "## Final Answer" not in md

    def test_explanation_section(self):
        md = _build_markdown({"explanation_summary": "All agents agreed."})
        assert "## Explanation" in md
        assert "All agents agreed." in md

    def test_no_explanation_when_empty(self):
        md = _build_markdown({"explanation_summary": ""})
        assert "## Explanation" not in md

    def test_cost_breakdown(self):
        md = _build_markdown(
            {
                "cost": {
                    "total_cost_usd": 0.0042,
                    "per_agent_cost": {"claude": 0.002, "gpt-4": 0.0022},
                }
            }
        )
        assert "## Cost Breakdown" in md
        assert "$0.0042" in md
        assert "claude" in md
        assert "gpt-4" in md

    def test_no_cost_when_empty(self):
        md = _build_markdown({"cost": {}})
        assert "## Cost Breakdown" not in md

    def test_receipt_section(self):
        md = _build_markdown(
            {
                "receipt": {
                    "receipt_id": "r-123",
                    "risk_level": "LOW",
                    "checksum": "abc123def",
                }
            }
        )
        assert "## Receipt" in md
        assert "r-123" in md
        assert "LOW" in md
        assert "abc123def" in md

    def test_no_receipt_when_none(self):
        md = _build_markdown({"receipt": None})
        assert "## Receipt" not in md

    def test_next_steps_section(self):
        md = _build_markdown(
            {
                "next_steps": [
                    {"action": "Do something", "priority": "high"},
                    {"action": "Do another", "priority": "medium"},
                ]
            }
        )
        assert "## Next Steps" in md
        assert "[HIGH] Do something" in md
        assert "[MEDIUM] Do another" in md

    def test_no_next_steps_when_empty(self):
        md = _build_markdown({"next_steps": []})
        assert "## Next Steps" not in md

    def test_participants_section(self):
        md = _build_markdown({"participants": ["claude", "gpt-4"]})
        assert "## Participants" in md
        assert "- claude" in md
        assert "- gpt-4" in md

    def test_no_participants_when_empty(self):
        md = _build_markdown({"participants": []})
        assert "## Participants" not in md

    def test_argument_map_section(self):
        md = _build_markdown(
            {
                "argument_map": {
                    "nodes": [{"id": 1}, {"id": 2}, {"id": 3}],
                    "edges": [{"from": 1, "to": 2}],
                }
            }
        )
        assert "## Argument Map" in md
        assert "**Nodes:** 3" in md
        assert "**Edges:** 1" in md

    def test_no_argument_map_when_none(self):
        md = _build_markdown({"argument_map": None})
        assert "## Argument Map" not in md

    def test_no_argument_map_when_no_nodes(self):
        md = _build_markdown({"argument_map": {"nodes": [], "edges": []}})
        assert "## Argument Map" not in md

    def test_export_formats_section(self):
        md = _build_markdown({"export_formats": ["json", "markdown", "csv"]})
        assert "## Export Formats" in md
        assert "- json" in md
        assert "- markdown" in md
        assert "- csv" in md

    def test_assembled_at_in_footer(self):
        md = _build_markdown({"assembled_at": "2026-01-01T00:00:00Z"})
        assert "2026-01-01T00:00:00Z" in md
        assert "Generated at" in md

    def test_full_package_renders_all_sections(self):
        package = {
            "debate_id": "d-full",
            "question": "Full test?",
            "verdict": "APPROVED",
            "confidence": 0.95,
            "consensus_reached": True,
            "status": "completed",
            "final_answer": "Yes.",
            "explanation_summary": "All agree.",
            "cost": {"total_cost_usd": 0.005, "per_agent_cost": {"a": 0.005}},
            "receipt": {"receipt_id": "r-1", "risk_level": "LOW", "checksum": "aaa"},
            "next_steps": [{"action": "Go", "priority": "high"}],
            "participants": ["agent-1"],
            "argument_map": {"nodes": [{"id": 1}], "edges": []},
            "export_formats": ["json"],
            "assembled_at": "2026-02-23T12:00:00Z",
        }
        md = _build_markdown(package)
        for section in [
            "## Summary",
            "## Final Answer",
            "## Explanation",
            "## Cost Breakdown",
            "## Receipt",
            "## Next Steps",
            "## Participants",
            "## Argument Map",
            "## Export Formats",
        ]:
            assert section in md


# ===========================================================================
# handle() dispatch and error handling
# ===========================================================================


class TestHandleDispatch:
    """Test the handle() method dispatching."""

    def test_json_package_returns_200(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        assert _status(result) == 200

    def test_markdown_package_returns_200(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package/markdown",
            {},
            http_handler,
        )
        assert _status(result) == 200

    def test_json_content_type(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        assert "json" in result.content_type

    def test_markdown_content_type(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package/markdown",
            {},
            http_handler,
        )
        assert "text/markdown" in result.content_type

    @patch(
        "aragora.server.handlers.debates.decision_package.DecisionPackageHandler._extract_debate_id",
        return_value=None,
    )
    def test_missing_debate_id_returns_400(self, mock_extract, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates//package",
            {},
            http_handler,
        )
        assert _status(result) == 400
        assert "Missing debate ID" in _body(result).get("error", "")


# ===========================================================================
# _assemble_package: error paths
# ===========================================================================


class TestAssemblePackageErrors:
    """Test error paths in _assemble_package."""

    def test_no_storage_returns_503(self, http_handler):
        h = DecisionPackageHandler(ctx={})
        result = h.handle("/api/v1/debates/abc/package", {}, http_handler)
        assert _status(result) == 503
        body = _body(result)
        assert "Storage not available" in body.get("error", "")

    def test_debate_not_found_returns_404(self, http_handler):
        storage = _make_storage({})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/missing/package", {}, http_handler)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    def test_debate_in_progress_returns_409(self, http_handler):
        debate = _completed_debate(status="in_progress")
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 409
        body = _body(result)
        assert "not completed" in body.get("error", "").lower()

    def test_debate_pending_returns_409(self, http_handler):
        debate = _completed_debate(status="pending")
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 409

    def test_debate_cancelled_returns_409(self, http_handler):
        debate = _completed_debate(status="cancelled")
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 409

    def test_debate_unknown_status_returns_409(self, http_handler):
        debate = _completed_debate(status="unknown")
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 409


# ===========================================================================
# _assemble_package: success with 'completed' status
# ===========================================================================


class TestAssemblePackageCompleted:
    """Test successful package assembly for completed debates."""

    def test_basic_fields_present(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert body["debate_id"] == "test-debate-001"
        assert body["question"] == "Should we use microservices?"
        assert body["status"] == "completed"
        assert body["confidence"] == 0.85
        assert body["consensus_reached"] is True

    def test_final_answer_in_package(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert body["final_answer"] == "Yes, microservices are recommended."

    def test_explanation_summary_in_package(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert body["explanation_summary"] == "All agents agreed on microservices."

    def test_participants_in_package(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert "claude" in body["participants"]
        assert "gpt-4" in body["participants"]

    def test_cost_in_package(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert body["cost"]["total_cost_usd"] == 0.0042
        assert "claude" in body["cost"]["per_agent_cost"]

    def test_export_formats_listed(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert set(body["export_formats"]) == {"json", "markdown", "csv", "html", "txt"}

    def test_assembled_at_present(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert "assembled_at" in body
        assert "T" in body["assembled_at"]

    def test_next_steps_present(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert len(body["next_steps"]) > 0


# ===========================================================================
# _assemble_package: timeout status is allowed
# ===========================================================================


class TestAssemblePackageTimeout:
    """Test package assembly for debates with 'timeout' status."""

    def test_timeout_is_valid_status(self, http_handler):
        debate = _completed_debate(status="timeout")
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "timeout"


# ===========================================================================
# Verdict computation without receipt
# ===========================================================================


class TestVerdictComputation:
    """Test verdict/confidence derivation when no receipt is available."""

    def _make_handler_and_get(self, confidence, consensus_reached, http_handler):
        debate = _completed_debate(
            confidence=confidence,
            consensus_reached=consensus_reached,
        )
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        return _body(result)

    @patch("aragora.storage.receipt_store.get_receipt_store", side_effect=ImportError)
    def test_approved_verdict(self, _mock_store, http_handler):
        body = self._make_handler_and_get(0.85, True, http_handler)
        assert body["verdict"] == "APPROVED"

    @patch("aragora.storage.receipt_store.get_receipt_store", side_effect=ImportError)
    def test_approved_with_conditions_verdict(self, _mock_store, http_handler):
        body = self._make_handler_and_get(0.5, True, http_handler)
        assert body["verdict"] == "APPROVED_WITH_CONDITIONS"

    @patch("aragora.storage.receipt_store.get_receipt_store", side_effect=ImportError)
    def test_needs_review_verdict(self, _mock_store, http_handler):
        body = self._make_handler_and_get(0.9, False, http_handler)
        assert body["verdict"] == "NEEDS_REVIEW"

    @patch("aragora.storage.receipt_store.get_receipt_store", side_effect=ImportError)
    def test_confidence_boundary_0_7(self, _mock_store, http_handler):
        """Consensus + confidence exactly 0.7 -> APPROVED."""
        body = self._make_handler_and_get(0.7, True, http_handler)
        assert body["verdict"] == "APPROVED"

    @patch("aragora.storage.receipt_store.get_receipt_store", side_effect=ImportError)
    def test_confidence_boundary_0_69(self, _mock_store, http_handler):
        """Consensus + confidence 0.69 -> APPROVED_WITH_CONDITIONS."""
        body = self._make_handler_and_get(0.69, True, http_handler)
        assert body["verdict"] == "APPROVED_WITH_CONDITIONS"


# ===========================================================================
# Receipt integration
# ===========================================================================


class TestReceiptIntegration:
    """Test receipt store integration and graceful degradation."""

    def test_receipt_import_error_graceful(self, http_handler):
        """ImportError from receipt store should not break package assembly."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            side_effect=ImportError("no module"),
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["receipt"] is None

    def test_receipt_runtime_error_graceful(self, http_handler):
        """RuntimeError from receipt store should not break package assembly."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            side_effect=RuntimeError("connection failed"),
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["receipt"] is None

    def test_receipt_present_in_package(self, http_handler):
        """When receipt is found, it should be included in the package."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "rcpt-001"
        mock_receipt.verdict = "APPROVED"
        mock_receipt.confidence = 0.92
        mock_receipt.risk_level = "LOW"
        mock_receipt.risk_score = 0.1
        mock_receipt.checksum = "sha256abc"
        mock_receipt.created_at = "2026-01-01T00:00:00Z"

        mock_store = MagicMock()
        mock_store.get_by_gauntlet.return_value = mock_receipt

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_store,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        body = _body(result)
        assert body["receipt"]["receipt_id"] == "rcpt-001"
        assert body["receipt"]["verdict"] == "APPROVED"
        assert body["receipt"]["confidence"] == 0.92
        assert body["receipt"]["risk_level"] == "LOW"
        assert body["receipt"]["risk_score"] == 0.1
        assert body["receipt"]["checksum"] == "sha256abc"

    def test_receipt_verdict_overrides_computed(self, http_handler):
        """Receipt verdict should take priority over computed verdict."""
        debate = _completed_debate(confidence=0.3, consensus_reached=False)
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "rcpt-002"
        mock_receipt.verdict = "APPROVED"
        mock_receipt.confidence = 0.95
        mock_receipt.risk_level = "LOW"
        mock_receipt.risk_score = None
        mock_receipt.checksum = "xyz"
        mock_receipt.created_at = None

        mock_store = MagicMock()
        mock_store.get_by_gauntlet.return_value = mock_receipt

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_store,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        body = _body(result)
        assert body["verdict"] == "APPROVED"
        assert body["confidence"] == 0.95

    def test_receipt_not_found(self, http_handler):
        """When receipt store returns None, receipt should be None."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_store = MagicMock()
        mock_store.get_by_gauntlet.return_value = None

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_store,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        body = _body(result)
        assert body["receipt"] is None

    def test_receipt_store_called_with_debate_prefix(self, http_handler):
        """Receipt store should be queried with 'debate-{id}' key."""
        debate = _completed_debate()
        storage = _make_storage({"my-debate": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_store = MagicMock()
        mock_store.get_by_gauntlet.return_value = None

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_store,
        ):
            h.handle("/api/v1/debates/my-debate/package", {}, http_handler)

        mock_store.get_by_gauntlet.assert_called_once_with("debate-my-debate")


# ===========================================================================
# Argument map integration
# ===========================================================================


class TestArgumentMapIntegration:
    """Test argument map visualization integration and graceful degradation."""

    def test_argument_map_import_error_graceful(self, http_handler):
        """ImportError from ArgumentCartographer should not break assembly."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        with patch(
            "aragora.visualization.mapper.ArgumentCartographer",
            side_effect=ImportError("no viz module"),
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["argument_map"] is None

    def test_argument_map_generated_when_messages_present(self, http_handler):
        """Argument map should be built from debate messages."""
        debate = _completed_debate(
            messages=[
                {"agent": "claude", "role": "proposal", "content": "X is good.", "round": 1},
            ]
        )
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_cart = MagicMock()
        mock_cart.export_json.return_value = json.dumps(
            {
                "nodes": [{"id": "n1"}],
                "edges": [],
            }
        )

        with patch(
            "aragora.visualization.mapper.ArgumentCartographer",
            return_value=mock_cart,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        body = _body(result)
        assert body["argument_map"] is not None
        assert len(body["argument_map"]["nodes"]) == 1
        mock_cart.set_debate_context.assert_called_once()
        mock_cart.update_from_message.assert_called_once()

    def test_no_argument_map_when_no_messages_key(self, http_handler):
        """No messages key -> no argument map attempt."""
        debate = _completed_debate()
        # Remove messages entirely so debate.get("messages", []) returns []
        debate.pop("messages", None)
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        body = _body(result)
        assert body["argument_map"] is None

    def test_argument_map_json_decode_error_graceful(self, http_handler):
        """JSON decode error from cartographer should degrade gracefully."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_cart = MagicMock()
        mock_cart.export_json.return_value = "not valid json{{"

        with patch(
            "aragora.visualization.mapper.ArgumentCartographer",
            return_value=mock_cart,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        body = _body(result)
        assert body["argument_map"] is None


# ===========================================================================
# Markdown endpoint
# ===========================================================================


class TestMarkdownEndpoint:
    """Test the markdown export endpoint."""

    def test_markdown_response_is_text(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package/markdown",
            {},
            http_handler,
        )
        text = _text(result)
        assert "# Decision Package:" in text

    def test_markdown_contains_question(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package/markdown",
            {},
            http_handler,
        )
        text = _text(result)
        assert "Should we use microservices?" in text

    def test_markdown_content_type_charset(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package/markdown",
            {},
            http_handler,
        )
        assert "charset=utf-8" in result.content_type

    def test_markdown_body_is_bytes(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package/markdown",
            {},
            http_handler,
        )
        assert isinstance(result.body, bytes)

    def test_markdown_error_propagates(self, http_handler):
        """Errors from _assemble_package should propagate in markdown too."""
        h = DecisionPackageHandler(ctx={})
        result = h.handle("/api/v1/debates/abc/package/markdown", {}, http_handler)
        assert _status(result) == 503


# ===========================================================================
# Edge cases in _assemble_package
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_debate_with_none_result(self, http_handler):
        """Debate with result=None should still assemble."""
        debate = _completed_debate()
        debate["result"] = None
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["confidence"] == 0.0
        assert body["verdict"] == "UNKNOWN"

    def test_debate_with_empty_result(self, http_handler):
        """Debate with result={} should use defaults."""
        debate = _completed_debate()
        debate["result"] = {}
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["confidence"] == 0.0
        assert body["final_answer"] == ""

    def test_participants_fallback_to_agents(self, http_handler):
        """When result has no participants, use debate.agents."""
        debate = _completed_debate(agents=["agent-a", "agent-b"])
        debate["result"] = {"confidence": 0.5}  # No participants key
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        body = _body(result)
        assert body["participants"] == ["agent-a", "agent-b"]

    def test_special_characters_in_debate_id(self, http_handler):
        """Debate ID with special chars should work."""
        debate = _completed_debate()
        storage = _make_storage({"debate-with-dashes": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle(
            "/api/v1/debates/debate-with-dashes/package",
            {},
            http_handler,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-with-dashes"

    def test_debate_with_no_question(self, http_handler):
        """Debate without a question field defaults to empty string."""
        debate = _completed_debate()
        del debate["question"]
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        body = _body(result)
        assert body["question"] == ""

    def test_debate_with_no_messages(self, http_handler):
        """Debate without messages should not crash."""
        debate = _completed_debate(messages=[])
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        assert _status(result) == 200

    def test_cost_defaults_to_zero(self, http_handler):
        """Missing cost data defaults to 0."""
        debate = _completed_debate()
        debate["result"] = {}
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})
        result = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        body = _body(result)
        assert body["cost"]["total_cost_usd"] == 0.0
        assert body["cost"]["per_agent_cost"] == {}

    def test_multiple_debates_in_storage(self, http_handler):
        """Handler retrieves the correct debate by ID."""
        d1 = _completed_debate(question="Question 1")
        d2 = _completed_debate(question="Question 2")
        storage = _make_storage({"d1": d1, "d2": d2})
        h = DecisionPackageHandler(ctx={"storage": storage})

        result1 = h.handle("/api/v1/debates/d1/package", {}, http_handler)
        result2 = h.handle("/api/v1/debates/d2/package", {}, http_handler)

        assert _body(result1)["question"] == "Question 1"
        assert _body(result2)["question"] == "Question 2"

    def test_receipt_created_at_none(self, http_handler):
        """Receipt with created_at=None should not crash."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_receipt = MagicMock()
        mock_receipt.receipt_id = "r-1"
        mock_receipt.verdict = "APPROVED"
        mock_receipt.confidence = 0.9
        mock_receipt.risk_level = "LOW"
        mock_receipt.risk_score = None
        mock_receipt.checksum = "abc"
        mock_receipt.created_at = None

        mock_store = MagicMock()
        mock_store.get_by_gauntlet.return_value = mock_receipt

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_store,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        body = _body(result)
        assert body["receipt"]["created_at"] is None

    def test_debate_status_is_included(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert body["status"] == "completed"

    def test_argument_map_with_messages_missing_agent(self, http_handler):
        """Messages missing agent field should still work (fallback to 'role')."""
        debate = _completed_debate(
            messages=[
                {"role": "proposal", "content": "Something", "round": 1},
            ]
        )
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_cart = MagicMock()
        mock_cart.export_json.return_value = json.dumps({"nodes": [], "edges": []})

        with patch(
            "aragora.visualization.mapper.ArgumentCartographer",
            return_value=mock_cart,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        assert _status(result) == 200
        # Verify it used the role as fallback for agent
        call_kwargs = mock_cart.update_from_message.call_args
        assert call_kwargs is not None

    def test_receipt_type_error_graceful(self, http_handler):
        """TypeError from receipt store should degrade gracefully."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            side_effect=TypeError("bad type"),
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["receipt"] is None

    def test_receipt_value_error_graceful(self, http_handler):
        """ValueError from receipt store should degrade gracefully."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            side_effect=ValueError("bad value"),
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["receipt"] is None

    def test_receipt_os_error_graceful(self, http_handler):
        """OSError from receipt store should degrade gracefully."""
        debate = _completed_debate()
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            side_effect=OSError("disk failure"),
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        assert _status(result) == 200


# ===========================================================================
# JSON structure completeness
# ===========================================================================


class TestJSONStructure:
    """Verify all expected keys in the JSON response."""

    def test_all_top_level_keys(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        expected_keys = {
            "debate_id",
            "question",
            "status",
            "verdict",
            "confidence",
            "consensus_reached",
            "final_answer",
            "explanation_summary",
            "participants",
            "receipt",
            "cost",
            "argument_map",
            "next_steps",
            "export_formats",
            "assembled_at",
        }
        assert expected_keys.issubset(set(body.keys()))

    def test_cost_structure(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        assert "total_cost_usd" in body["cost"]
        assert "per_agent_cost" in body["cost"]

    def test_next_steps_structure(self, handler_with_storage, http_handler):
        result = handler_with_storage.handle(
            "/api/v1/debates/test-debate-001/package",
            {},
            http_handler,
        )
        body = _body(result)
        for step in body["next_steps"]:
            assert "action" in step
            assert "priority" in step


# ===========================================================================
# Verdict with receipt vs without
# ===========================================================================


class TestVerdictPrecedence:
    """Test that receipt verdict takes precedence over computed verdict."""

    def test_without_receipt_uses_result_data(self, http_handler):
        """No receipt -> verdict computed from consensus + confidence."""
        debate = _completed_debate(confidence=0.5, consensus_reached=True)
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_store = MagicMock()
        mock_store.get_by_gauntlet.return_value = None

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_store,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        body = _body(result)
        assert body["verdict"] == "APPROVED_WITH_CONDITIONS"
        assert body["confidence"] == 0.5

    def test_no_result_data_gives_unknown(self, http_handler):
        """No receipt and no result data -> UNKNOWN verdict."""
        debate = _completed_debate()
        debate["result"] = None
        storage = _make_storage({"d1": debate})
        h = DecisionPackageHandler(ctx={"storage": storage})

        mock_store = MagicMock()
        mock_store.get_by_gauntlet.return_value = None

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_store,
        ):
            result = h.handle("/api/v1/debates/d1/package", {}, http_handler)

        body = _body(result)
        assert body["verdict"] == "UNKNOWN"
        assert body["confidence"] == 0.0
