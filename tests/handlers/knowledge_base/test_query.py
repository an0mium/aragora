"""Tests for QueryOperationsMixin (aragora/server/handlers/knowledge_base/query.py).

Covers the POST /api/v1/knowledge/query endpoint:
- Valid queries with default and custom options
- Missing/empty question validation
- Invalid JSON body handling
- Zero Content-Length (empty body)
- QueryOptions construction from request data
- Query engine call with correct arguments
- Engine errors (KeyError, ValueError, OSError, TypeError, RuntimeError)
- Successful response with to_dict() serialization
- Workspace ID handling (default vs custom)
- Partial options (only some fields provided)
- Edge cases: huge question, unicode, special characters
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.query_engine import QueryOptions
from aragora.knowledge.types import (
    Fact,
    QueryResult,
    ValidationStatus,
)
from aragora.server.handlers.knowledge_base.handler import KnowledgeHandler


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
    """Mock HTTP request handler for query tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "POST",
        raw_bytes: bytes | None = None,
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}

        if raw_bytes is not None:
            self.rfile = io.BytesIO(raw_bytes)
            self.headers["Content-Length"] = str(len(raw_bytes))
        elif body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile = io.BytesIO(body_bytes)
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile = io.BytesIO(b"")
            self.headers["Content-Length"] = "0"


def _make_fact(
    fact_id: str = "fact-001",
    statement: str = "The sky is blue",
    confidence: float = 0.9,
    workspace_id: str = "default",
) -> Fact:
    """Create a Fact instance for testing."""
    return Fact(
        id=fact_id,
        statement=statement,
        confidence=confidence,
        evidence_ids=[],
        source_documents=[],
        workspace_id=workspace_id,
        validation_status=ValidationStatus.UNVERIFIED,
        topics=["science"],
        metadata={},
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        updated_at=datetime(2025, 1, 1, 12, 0, 0),
        superseded_by=None,
    )


def _make_query_result(
    answer: str = "The sky is blue due to Rayleigh scattering.",
    confidence: float = 0.85,
    question: str = "Why is the sky blue?",
    workspace_id: str = "default",
) -> QueryResult:
    """Create a QueryResult instance for testing."""
    return QueryResult(
        answer=answer,
        facts=[_make_fact()],
        evidence_ids=["ev-001"],
        confidence=confidence,
        query=question,
        workspace_id=workspace_id,
        processing_time_ms=150,
        agent_contributions={"claude": "Rayleigh scattering"},
        metadata={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_query_engine():
    """Create a mock query engine."""
    engine = MagicMock()
    engine.query = MagicMock(return_value=_make_query_result())
    return engine


@pytest.fixture
def handler(mock_query_engine):
    """Create a KnowledgeHandler with injected mock query engine."""
    h = KnowledgeHandler(server_context={})
    h._query_engine = mock_query_engine
    # Also set a mock fact store to avoid initialization
    h._fact_store = MagicMock()
    return h


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge/query - Valid requests
# ---------------------------------------------------------------------------


class TestHandleQueryBasic:
    """Basic valid query scenarios."""

    def test_query_with_question_returns_200(self, handler, mock_query_engine):
        """A valid question produces a 200 response."""
        body = {"question": "Why is the sky blue?"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 200

    def test_query_returns_result_to_dict(self, handler, mock_query_engine):
        """Response body contains the to_dict() output of QueryResult."""
        qr = _make_query_result(answer="42 is the answer")
        body = {"question": "What is the meaning of life?"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=qr,
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert data["answer"] == "42 is the answer"
        assert data["confidence"] == 0.85

    def test_query_passes_question_to_engine(self, handler, mock_query_engine):
        """The question text is forwarded to the query engine."""
        body = {"question": "How does gravity work?"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ) as mock_run:
            handler._handle_query({}, http)

        call_args = mock_run.call_args
        # _run_async receives engine.query(question, workspace_id, options)
        # We check the coroutine was created with correct args
        mock_query_engine.query.assert_called_once()
        args = mock_query_engine.query.call_args
        assert args[0][0] == "How does gravity work?"

    def test_query_default_workspace(self, handler, mock_query_engine):
        """Without workspace_id, default workspace is used."""
        body = {"question": "Test question"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        args = mock_query_engine.query.call_args
        assert args[0][1] == "default"

    def test_query_custom_workspace(self, handler, mock_query_engine):
        """Custom workspace_id is forwarded to the engine."""
        body = {"question": "Test question", "workspace_id": "my-workspace"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        args = mock_query_engine.query.call_args
        assert args[0][1] == "my-workspace"

    def test_query_result_contains_facts(self, handler):
        """Query result serialization includes facts array."""
        body = {"question": "What are facts?"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert "facts" in data
        assert len(data["facts"]) == 1

    def test_query_result_contains_evidence_ids(self, handler):
        """Query result serialization includes evidence_ids."""
        body = {"question": "Show evidence"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert data["evidence_ids"] == ["ev-001"]

    def test_query_result_contains_agent_contributions(self, handler):
        """Query result serialization includes agent_contributions."""
        body = {"question": "Which agents contributed?"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert data["agent_contributions"] == {"claude": "Rayleigh scattering"}


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge/query - QueryOptions construction
# ---------------------------------------------------------------------------


class TestQueryOptions:
    """Test QueryOptions are correctly built from request payload."""

    def test_default_options(self, handler, mock_query_engine):
        """When no options are provided, defaults are used."""
        body = {"question": "Test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert isinstance(opts, QueryOptions)
        assert opts.max_chunks == 10
        assert opts.search_alpha == 0.5
        assert opts.use_agents is False
        assert opts.extract_facts is True
        assert opts.include_citations is True

    def test_custom_max_chunks(self, handler, mock_query_engine):
        """Custom max_chunks is forwarded."""
        body = {"question": "Test", "options": {"max_chunks": 25}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.max_chunks == 25

    def test_custom_search_alpha(self, handler, mock_query_engine):
        """Custom search_alpha is forwarded."""
        body = {"question": "Test", "options": {"search_alpha": 0.8}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.search_alpha == 0.8

    def test_use_agents_true(self, handler, mock_query_engine):
        """use_agents=True is forwarded."""
        body = {"question": "Test", "options": {"use_agents": True}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.use_agents is True

    def test_extract_facts_false(self, handler, mock_query_engine):
        """extract_facts=False is forwarded."""
        body = {"question": "Test", "options": {"extract_facts": False}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.extract_facts is False

    def test_include_citations_false(self, handler, mock_query_engine):
        """include_citations=False is forwarded."""
        body = {"question": "Test", "options": {"include_citations": False}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.include_citations is False

    def test_all_options_custom(self, handler, mock_query_engine):
        """All options provided together."""
        body = {
            "question": "Full test",
            "options": {
                "max_chunks": 5,
                "search_alpha": 0.1,
                "use_agents": True,
                "extract_facts": False,
                "include_citations": False,
            },
        }
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.max_chunks == 5
        assert opts.search_alpha == 0.1
        assert opts.use_agents is True
        assert opts.extract_facts is False
        assert opts.include_citations is False

    def test_empty_options_object(self, handler, mock_query_engine):
        """Empty options object uses defaults."""
        body = {"question": "Test", "options": {}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.max_chunks == 10
        assert opts.search_alpha == 0.5

    def test_extra_options_ignored(self, handler, mock_query_engine):
        """Unknown options fields are silently ignored."""
        body = {
            "question": "Test",
            "options": {"max_chunks": 3, "unknown_field": "ignored"},
        }
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 200
        opts = mock_query_engine.query.call_args[0][2]
        assert opts.max_chunks == 3

    def test_options_partial_override(self, handler, mock_query_engine):
        """Providing only some options keeps defaults for the rest."""
        body = {"question": "Test", "options": {"use_agents": True}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.use_agents is True
        assert opts.max_chunks == 10  # default
        assert opts.search_alpha == 0.5  # default
        assert opts.extract_facts is True  # default
        assert opts.include_citations is True  # default


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge/query - Error: missing/empty question
# ---------------------------------------------------------------------------


class TestHandleQueryMissingQuestion:
    """Test validation when question is missing or empty."""

    def test_empty_body_returns_400(self, handler):
        """Empty JSON body (no question) returns 400."""
        http = MockHTTPHandler(body={})

        result = handler._handle_query({}, http)

        assert _status(result) == 400
        assert "required" in _body(result).get("error", "").lower()

    def test_missing_question_key_returns_400(self, handler):
        """Body without 'question' key returns 400."""
        body = {"workspace_id": "test"}
        http = MockHTTPHandler(body=body)

        result = handler._handle_query({}, http)

        assert _status(result) == 400

    def test_empty_string_question_returns_400(self, handler):
        """Empty string question returns 400."""
        body = {"question": ""}
        http = MockHTTPHandler(body=body)

        result = handler._handle_query({}, http)

        assert _status(result) == 400

    def test_null_question_returns_400(self, handler):
        """null/None question returns 400."""
        body = {"question": None}
        http = MockHTTPHandler(body=body)

        result = handler._handle_query({}, http)

        assert _status(result) == 400

    def test_whitespace_only_question_succeeds(self, handler, mock_query_engine):
        """Whitespace-only question passes validation (handler does not strip).

        The handler checks `if not question:` which is True for "" but
        whitespace " " is truthy, so it passes.
        """
        body = {"question": "   "}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 200


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge/query - Error: invalid body
# ---------------------------------------------------------------------------


class TestHandleQueryInvalidBody:
    """Test JSON parsing and body errors."""

    def test_invalid_json_returns_400(self, handler):
        """Malformed JSON body returns 400."""
        http = MockHTTPHandler(raw_bytes=b"not valid json{{{")

        result = handler._handle_query({}, http)

        assert _status(result) == 400
        assert "invalid" in _body(result).get("error", "").lower()

    def test_zero_content_length_no_question_returns_400(self, handler):
        """Zero Content-Length means empty body, which has no question."""
        http = MockHTTPHandler()  # default: no body, Content-Length=0

        result = handler._handle_query({}, http)

        assert _status(result) == 400

    def test_non_utf8_body_returns_400(self, handler):
        """Non-UTF-8 body triggers decode error and returns 400."""
        # This will be valid Content-Length but invalid UTF-8
        raw = b"\x80\x81\x82\x83"
        http = MockHTTPHandler(raw_bytes=raw)

        result = handler._handle_query({}, http)

        assert _status(result) == 400

    def test_array_body_returns_500(self, handler):
        """JSON array causes AttributeError on .get() -> caught by @handle_errors."""
        raw = json.dumps([1, 2, 3]).encode()
        http = MockHTTPHandler(raw_bytes=raw)

        result = handler._handle_query({}, http)

        # json.loads succeeds (valid JSON), but data is a list, so data.get()
        # raises AttributeError which is caught by @handle_errors -> 500
        assert _status(result) == 500

    def test_string_body_returns_500(self, handler):
        """JSON string body causes AttributeError on .get() -> caught by @handle_errors."""
        raw = json.dumps("just a string").encode()
        http = MockHTTPHandler(raw_bytes=raw)

        result = handler._handle_query({}, http)

        # json.loads succeeds (valid JSON), but data is a str, so data.get()
        # raises AttributeError which is caught by @handle_errors -> 500
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge/query - Engine errors
# ---------------------------------------------------------------------------


class TestHandleQueryEngineErrors:
    """Test error handling when the query engine raises exceptions."""

    @pytest.mark.parametrize(
        "exc_class",
        [KeyError, ValueError, OSError, TypeError, RuntimeError],
        ids=["KeyError", "ValueError", "OSError", "TypeError", "RuntimeError"],
    )
    def test_engine_error_returns_500(self, handler, exc_class):
        """Known engine exceptions are caught and return 500."""
        body = {"question": "Will this fail?"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            side_effect=exc_class("engine failure"),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 500
        assert "failed" in _body(result).get("error", "").lower()

    def test_key_error_message_is_generic(self, handler):
        """KeyError returns a generic message, not the raw exception."""
        body = {"question": "KeyError test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            side_effect=KeyError("secret_key"),
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert "secret_key" not in data.get("error", "")
        assert _status(result) == 500

    def test_unhandled_exception_caught_by_handle_errors(self, handler):
        """Exceptions not in the explicit list are caught by @handle_errors."""
        body = {"question": "Unexpected error test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            side_effect=MemoryError("out of memory"),
        ):
            result = handler._handle_query({}, http)

        # @handle_errors catches all exceptions and returns 500
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge/query - Edge cases
# ---------------------------------------------------------------------------


class TestHandleQueryEdgeCases:
    """Edge cases and boundary conditions."""

    def test_unicode_question(self, handler, mock_query_engine):
        """Unicode characters in question are handled correctly."""
        body = {"question": "Pourquoi le ciel est-il bleu? \u2603 \U0001f600"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 200

    def test_long_question(self, handler, mock_query_engine):
        """Very long question strings are handled."""
        body = {"question": "x" * 10000}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 200

    def test_special_characters_in_question(self, handler, mock_query_engine):
        """Question with special chars (HTML, SQL-like) is handled."""
        body = {"question": "<script>alert('xss')</script> OR 1=1; DROP TABLE facts;"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 200

    def test_numeric_question_still_passed(self, handler, mock_query_engine):
        """Numeric-valued question is treated as truthy string."""
        body = {"question": "123"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 200

    def test_workspace_id_empty_string_uses_empty(self, handler, mock_query_engine):
        """Empty workspace_id is passed as-is (handler does not default empty to 'default')."""
        body = {"question": "Test", "workspace_id": ""}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        args = mock_query_engine.query.call_args
        assert args[0][1] == ""

    def test_extra_body_fields_ignored(self, handler, mock_query_engine):
        """Extra top-level fields in the body are silently ignored."""
        body = {"question": "Test", "extra_field": "should be ignored", "another": 42}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert _status(result) == 200

    def test_options_as_non_dict_uses_defaults(self, handler, mock_query_engine):
        """If options is not a dict, .get() calls fail gracefully.

        In practice the handler calls data.get("options", {}) then
        options_data.get(...) which works because the result is dict-like.
        If options is a list or string, AttributeError is caught by @handle_errors.
        """
        body = {"question": "Test", "options": "not_a_dict"}
        http = MockHTTPHandler(body=body)

        # .get() on a string will raise AttributeError → caught by @handle_errors
        result = handler._handle_query({}, http)

        # @handle_errors catches the AttributeError from calling .get() on a string
        assert _status(result) == 500

    def test_search_alpha_zero(self, handler, mock_query_engine):
        """search_alpha=0 (pure vector search) is accepted."""
        body = {"question": "Test", "options": {"search_alpha": 0.0}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.search_alpha == 0.0

    def test_search_alpha_one(self, handler, mock_query_engine):
        """search_alpha=1 (pure keyword search) is accepted."""
        body = {"question": "Test", "options": {"search_alpha": 1.0}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.search_alpha == 1.0

    def test_max_chunks_zero(self, handler, mock_query_engine):
        """max_chunks=0 is passed through (engine decides behavior)."""
        body = {"question": "Test", "options": {"max_chunks": 0}}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            handler._handle_query({}, http)

        opts = mock_query_engine.query.call_args[0][2]
        assert opts.max_chunks == 0

    def test_query_result_processing_time(self, handler):
        """Processing time from QueryResult is included in response."""
        qr = _make_query_result()
        body = {"question": "Timing test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=qr,
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert data["processing_time_ms"] == 150


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge/query - Content-Type / Content-Length edge cases
# ---------------------------------------------------------------------------


class TestContentLengthEdgeCases:
    """Content-Length header edge cases."""

    def test_content_length_non_numeric_returns_400(self, handler):
        """Non-numeric Content-Length triggers ValueError and returns 400."""
        http = MockHTTPHandler()
        http.headers["Content-Length"] = "not-a-number"
        http.rfile = io.BytesIO(b'{"question": "test"}')

        result = handler._handle_query({}, http)

        assert _status(result) == 400

    def test_content_length_negative_treats_as_zero(self, handler):
        """Negative Content-Length is parsed as int but <= 0, so data={}."""
        http = MockHTTPHandler()
        http.headers["Content-Length"] = "-1"
        http.rfile = io.BytesIO(b'{"question": "test"}')

        result = handler._handle_query({}, http)

        # data={} since content_length <= 0, so no question -> 400
        assert _status(result) == 400

    def test_content_length_missing_header_defaults_zero(self, handler):
        """Missing Content-Length header defaults to 0, empty data."""
        http = MockHTTPHandler()
        del http.headers["Content-Length"]
        http.rfile = io.BytesIO(b'{"question": "test"}')

        result = handler._handle_query({}, http)

        # headers.get("Content-Length", 0) -> 0 -> data = {}
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/knowledge/query - Response structure
# ---------------------------------------------------------------------------


class TestQueryResponseStructure:
    """Verify the structure of successful query responses."""

    def test_response_has_answer_field(self, handler):
        body = {"question": "Test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert "answer" in _body(result)

    def test_response_has_confidence_field(self, handler):
        body = {"question": "Test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert "confidence" in _body(result)

    def test_response_has_query_field(self, handler):
        body = {"question": "Test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert "query" in data
        assert data["query"] == "Why is the sky blue?"

    def test_response_has_workspace_id(self, handler):
        body = {"question": "Test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert data["workspace_id"] == "default"

    def test_response_has_metadata(self, handler):
        body = {"question": "Test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        data = _body(result)
        assert "metadata" in data

    def test_response_content_type_json(self, handler):
        """Response has JSON content type."""
        body = {"question": "Test"}
        http = MockHTTPHandler(body=body)

        with patch(
            "aragora.server.handlers.knowledge_base.query._run_async",
            return_value=_make_query_result(),
        ):
            result = handler._handle_query({}, http)

        assert hasattr(result, "content_type")
        assert "json" in result.content_type
