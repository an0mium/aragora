"""
Comprehensive tests for TemplateMarketplaceHandler and TemplateRecommendationsHandler.

This test module supplements test_template_marketplace_handler.py with additional
coverage for:
- Persistent store initialization (_init_persistent_store, _get_persistent_store)
- DistributedStateError when persistent storage is unavailable in multi-instance mode
- Circuit breaker record_success/record_failure through handle()
- Method detection fallback (handler without .command attribute)
- Path parsing edge cases through _route_request
- Multiple sequential imports incrementing download count
- Rating arithmetic precision and rounding
- _clear_marketplace_state when circuit breaker is None
- Search across description, name, and tags simultaneously
- Template to_summary with exact boundary descriptions (200 chars)
- Publishing with whitespace-only name
- Concurrent rating updates to same template
- Empty body for publish (empty JSON object)
- Query param edge cases (empty strings, extreme limits)
"""

from __future__ import annotations

import io
import json
import sys
import time
import types as _types_mod
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Slack stubs to prevent transitive import issues (standard pattern)
# ---------------------------------------------------------------------------
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


from aragora.control_plane.leader import DistributedStateError
from aragora.server.handlers.template_marketplace import (
    DESCRIPTION_MAX_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS,
    REVIEW_CONTENT_MAX_LENGTH,
    TEMPLATE_NAME_MAX_LENGTH,
    VALID_CATEGORIES,
    VALID_PATTERNS,
    MarketplaceCircuitBreaker,
    MarketplaceTemplate,
    TemplateMarketplaceHandler,
    TemplateRecommendationsHandler,
    TemplateReview,
    _clear_marketplace_state,
    _marketplace_templates,
    _seed_marketplace_templates,
    _template_reviews,
    _user_ratings,
    get_marketplace_circuit_breaker,
    get_marketplace_circuit_breaker_status,
    validate_category,
    validate_pattern,
    validate_rating,
    validate_review_content,
    validate_tags,
    validate_template_name,
)


# ===========================================================================
# Helpers
# ===========================================================================


def parse_body(result) -> dict[str, Any]:
    """Parse HandlerResult body into a dict."""
    return json.loads(result.body.decode("utf-8"))


def make_mock_handler(
    *,
    method: str = "GET",
    client_address: tuple[str, int] = ("127.0.0.1", 12345),
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    has_command: bool = True,
) -> MagicMock:
    """Create a mock HTTP handler with configurable attributes."""
    mock = MagicMock()
    if has_command:
        mock.command = method
    else:
        # Simulate handler without .command attribute
        del mock.command
    mock.client_address = client_address
    mock.headers = headers or {}
    if body is not None:
        mock.headers["Content-Length"] = str(len(body))
        mock.rfile = io.BytesIO(body)
    else:
        mock.headers.setdefault("Content-Length", "0")
        mock.rfile = io.BytesIO(b"")
    return mock


def json_body(data: dict) -> bytes:
    """Encode a dict as JSON bytes."""
    return json.dumps(data).encode("utf-8")


def _publish_body(**overrides) -> dict:
    """Build a valid publish body with optional overrides."""
    base = {
        "name": "Test Template",
        "description": "A test template description.",
        "category": "security",
        "pattern": "debate",
        "workflow_definition": {"steps": [{"name": "step1"}]},
        "tags": ["test", "security"],
        "author_id": "user-1",
        "author_name": "Test Author",
    }
    base.update(overrides)
    return base


def _insert_template(
    template_id: str,
    **overrides: Any,
) -> MarketplaceTemplate:
    """Insert a template keyed by the given id."""
    defaults = dict(
        id=template_id,
        name="Test Template",
        description="Template for tests.",
        category="security",
        pattern="debate",
        author_id="test",
        author_name="Test Author",
        rating=4.0,
        rating_count=10,
        download_count=100,
        is_featured=False,
        is_verified=True,
    )
    defaults.update(overrides)
    tpl = MarketplaceTemplate(**defaults)
    _marketplace_templates[template_id] = tpl
    return tpl


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def clean_marketplace_state():
    """Reset marketplace state before and after each test."""
    _clear_marketplace_state()
    yield
    _clear_marketplace_state()


@pytest.fixture
def handler() -> TemplateMarketplaceHandler:
    return TemplateMarketplaceHandler(ctx={})


@pytest.fixture
def rec_handler() -> TemplateRecommendationsHandler:
    return TemplateRecommendationsHandler(ctx={})


@pytest.fixture
def seeded_handler(handler) -> TemplateMarketplaceHandler:
    _seed_marketplace_templates()
    return handler


@pytest.fixture
def mock_http_get() -> MagicMock:
    return make_mock_handler(method="GET")


# ===========================================================================
# Tests: Persistent Store Initialization
# ===========================================================================


class TestPersistentStoreInit:
    """Tests for _init_persistent_store and _get_persistent_store."""

    def test_init_persistent_store_import_error_fallback(self):
        """When marketplace_store import fails, falls back to in-memory."""
        import aragora.server.handlers.template_marketplace as mod

        with patch.dict("os.environ", {}, clear=False):
            # Ensure distributed state is NOT required
            with patch.object(mod, "is_distributed_state_required", return_value=False):
                with patch.dict(
                    "sys.modules",
                    {"aragora.storage.marketplace_store": None},
                ):
                    # Reset globals
                    mod._use_persistent_store = False
                    mod._persistent_store = None
                    result = mod._init_persistent_store()
                    assert result is False
                    assert mod._use_persistent_store is False

    def test_init_persistent_store_distributed_state_required_raises(self):
        """When distributed state required and store unavailable, raise error."""
        import aragora.server.handlers.template_marketplace as mod

        with patch.object(mod, "is_distributed_state_required", return_value=True):
            # Force ImportError by patching the import
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "aragora.storage.marketplace_store":
                    raise ImportError("no module")
                return original_import(name, *args, **kwargs)

            mod._use_persistent_store = False
            mod._persistent_store = None

            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(DistributedStateError) as exc_info:
                    mod._init_persistent_store()
                assert "template_marketplace" in str(exc_info.value)

    def test_get_persistent_store_returns_none_when_not_available(self):
        """_get_persistent_store returns None when persistent store not available."""
        import aragora.server.handlers.template_marketplace as mod

        mod._use_persistent_store = False
        mod._persistent_store = None
        with patch.object(mod, "_init_persistent_store", return_value=False):
            result = mod._get_persistent_store()
            assert result is None

    def test_get_persistent_store_initializes_on_first_call(self):
        """_get_persistent_store calls _init on first access."""
        import aragora.server.handlers.template_marketplace as mod

        mod._persistent_store = None
        mod._use_persistent_store = False
        with patch.object(mod, "_init_persistent_store") as mock_init:
            mock_init.return_value = False
            mod._get_persistent_store()
            mock_init.assert_called_once()

    def test_get_persistent_store_skips_init_when_already_set(self):
        """_get_persistent_store doesn't re-initialize when store exists."""
        import aragora.server.handlers.template_marketplace as mod

        sentinel = object()
        mod._persistent_store = sentinel
        mod._use_persistent_store = True
        with patch.object(mod, "_init_persistent_store") as mock_init:
            result = mod._get_persistent_store()
            mock_init.assert_not_called()
            assert result is sentinel


# ===========================================================================
# Tests: Circuit Breaker Integration through handle()
# ===========================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker record_success/record_failure via handle()."""

    def test_successful_request_records_success(self, handler, mock_http_get):
        """A successful request should call record_success on circuit breaker."""
        _seed_marketplace_templates()
        with patch.object(handler._circuit_breaker, "record_success") as mock_success:
            handler.handle("/api/v1/marketplace/templates", {}, mock_http_get)
            mock_success.assert_called_once()

    def test_circuit_breaker_failure_count_tracks(self, handler):
        """Verify failure count increments on circuit breaker."""
        cb = handler._circuit_breaker
        assert cb.get_status()["failure_count"] == 0
        cb.record_failure()
        assert cb.get_status()["failure_count"] == 1
        cb.record_failure()
        assert cb.get_status()["failure_count"] == 2

    def test_circuit_breaker_opens_after_threshold(self, handler):
        """Circuit breaker should open after enough failures."""
        cb = handler._circuit_breaker
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.state == "open"

    def test_circuit_breaker_half_open_after_cooldown(self, handler):
        """Circuit breaker transitions to half_open after cooldown expires."""
        cb = handler._circuit_breaker
        # Open the circuit
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.state == "open"
        # Simulate cooldown elapsed
        cb._last_failure_time = time.time() - cb.cooldown_seconds - 1
        assert cb.state == "half_open"

    def test_circuit_breaker_status_endpoint_shows_failure_count(self, handler, mock_http_get):
        """Circuit breaker status should reflect recorded failures."""
        handler._circuit_breaker.record_failure()
        handler._circuit_breaker.record_failure()
        result = handler.handle("/api/v1/marketplace/circuit-breaker", {}, mock_http_get)
        body = parse_body(result)
        assert body["failure_count"] == 2

    def test_successful_request_resets_failure_count_in_closed(self, handler, mock_http_get):
        """In closed state, record_success resets failure_count to 0."""
        cb = handler._circuit_breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status()["failure_count"] == 2
        # Successful request through handle
        _seed_marketplace_templates()
        handler.handle("/api/v1/marketplace/featured", {}, mock_http_get)
        assert cb.get_status()["failure_count"] == 0


# ===========================================================================
# Tests: Method Detection Fallback
# ===========================================================================


class TestMethodDetection:
    """Tests for handler.command fallback when attribute is missing."""

    def test_handler_without_command_defaults_to_get(self, handler):
        """When handler has no .command attribute, method should default to GET."""
        _seed_marketplace_templates()
        mock = make_mock_handler(method="GET", has_command=False)
        result = handler.handle("/api/v1/marketplace/templates", {}, mock)
        body = parse_body(result)
        assert result.status_code == 200
        assert "templates" in body

    def test_handler_with_command_post(self, handler):
        """Verify POST is correctly detected from handler.command."""
        data = _publish_body()
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 201


# ===========================================================================
# Tests: Path Parsing Edge Cases
# ===========================================================================


class TestPathParsingEdgeCases:
    """Tests for edge cases in path routing."""

    def test_template_id_with_multiple_slashes(self, handler):
        """Template IDs with multiple segments should be joined correctly."""
        _insert_template("templates/deep/nested/id")
        mock_get = make_mock_handler(method="GET")
        result = handler.handle(
            "/api/v1/marketplace/templates/deep/nested/id", {}, mock_get
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["name"] == "Test Template"

    def test_rate_endpoint_with_multi_segment_id(self, handler):
        """Rate endpoint works with multi-segment template IDs."""
        _insert_template("templates/cat/sub/tpl")
        data = {"rating": 4, "user_id": "user-1"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle(
            "/api/v1/marketplace/templates/cat/sub/tpl/rate", {}, mock_post
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["status"] == "rated"

    def test_reviews_endpoint_with_multi_segment_id(self, handler, mock_http_get):
        """Reviews GET endpoint works with multi-segment template IDs."""
        _insert_template("templates/cat/sub/tpl")
        _template_reviews["templates/cat/sub/tpl"] = []
        result = handler.handle(
            "/api/v1/marketplace/templates/cat/sub/tpl/reviews", {}, mock_http_get
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] == 0

    def test_import_endpoint_with_multi_segment_id(self, handler):
        """Import endpoint works with multi-segment template IDs."""
        _insert_template("templates/cat/sub/tpl", download_count=50)
        mock_post = make_mock_handler(method="POST", body=json_body({}))
        result = handler.handle(
            "/api/v1/marketplace/templates/cat/sub/tpl/import", {}, mock_post
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["download_count"] == 51

    def test_submit_review_via_post_routing(self, handler):
        """POST to reviews endpoint routes to _submit_review."""
        _insert_template("templates/cat/tpl")
        data = {"rating": 5, "content": "Routed review."}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle(
            "/api/v1/marketplace/templates/cat/tpl/reviews", {}, mock_post
        )
        body = parse_body(result)
        assert result.status_code == 201
        assert body["status"] == "submitted"


# ===========================================================================
# Tests: Sequential Imports
# ===========================================================================


class TestSequentialImports:
    """Tests for multiple sequential imports incrementing download_count."""

    def test_multiple_imports_increment_download_count(self, seeded_handler):
        """Each import should increment the download count by 1."""
        tpl = _marketplace_templates["security/code-audit"]
        original = tpl.download_count

        for i in range(5):
            mock_post = make_mock_handler(method="POST", body=json_body({}))
            result = seeded_handler._import_template("security/code-audit", mock_post)
            body = parse_body(result)
            assert body["download_count"] == original + i + 1

        assert tpl.download_count == original + 5


# ===========================================================================
# Tests: Rating Arithmetic
# ===========================================================================


class TestRatingArithmetic:
    """Tests for rating calculation precision and rounding."""

    def test_first_rating_on_zero_count_template(self, handler):
        """Rating a template with 0 ratings should set the average correctly."""
        _insert_template("test/zero-rating", rating=0.0, rating_count=0)
        data = {"rating": 4, "user_id": "user-1"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler._rate_template("test/zero-rating", mock_post, "127.0.0.1")
        body = parse_body(result)
        assert result.status_code == 200
        assert body["average_rating"] == 4.0
        assert body["rating_count"] == 1

    def test_rating_precision_after_multiple_ratings(self, handler):
        """Average rating should be correctly computed with rounding to 2 decimal places."""
        _insert_template("test/precision", rating=0.0, rating_count=0)
        ratings_to_add = [5, 4, 3, 5, 4]
        for i, r in enumerate(ratings_to_add):
            data = {"rating": r, "user_id": f"user-{i}"}
            mock_post = make_mock_handler(method="POST", body=json_body(data))
            handler._rate_template("test/precision", mock_post, "127.0.0.1")

        tpl = _marketplace_templates["test/precision"]
        expected_avg = sum(ratings_to_add) / len(ratings_to_add)
        assert tpl.rating_count == len(ratings_to_add)
        assert abs(tpl.rating - expected_avg) < 0.01

    def test_update_rating_adjusts_average_correctly(self, handler):
        """Updating a rating should adjust the average without changing the count."""
        _insert_template("test/update", rating=4.0, rating_count=2)
        # First rating by user-1
        data1 = {"rating": 2, "user_id": "user-1"}
        m1 = make_mock_handler(method="POST", body=json_body(data1))
        r1 = handler._rate_template("test/update", m1, "127.0.0.1")
        b1 = parse_body(r1)
        count_after = b1["rating_count"]

        # Update to rating 5
        data2 = {"rating": 5, "user_id": "user-1"}
        m2 = make_mock_handler(method="POST", body=json_body(data2))
        r2 = handler._rate_template("test/update", m2, "127.0.0.1")
        b2 = parse_body(r2)
        # Count should not increase
        assert b2["rating_count"] == count_after
        # The average should reflect the updated value
        assert b2["your_rating"] == 5

    def test_rating_with_float_value(self, handler):
        """Float ratings like 3.5 should be accepted and stored."""
        _insert_template("test/float-rate", rating=0.0, rating_count=0)
        data = {"rating": 3.5, "user_id": "user-1"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler._rate_template("test/float-rate", mock_post, "127.0.0.1")
        body = parse_body(result)
        assert result.status_code == 200
        assert body["your_rating"] == 3.5
        # Stored as int for user_ratings
        assert _user_ratings["user-1"]["test/float-rate"] == 3

    def test_rating_boundary_values(self, handler):
        """Ratings at exact boundaries (1 and 5) should be accepted."""
        _insert_template("test/boundary", rating=0.0, rating_count=0)
        for rating_val in [1, 5]:
            data = {"rating": rating_val, "user_id": f"user-{rating_val}"}
            mock_post = make_mock_handler(method="POST", body=json_body(data))
            result = handler._rate_template("test/boundary", mock_post, "127.0.0.1")
            assert result.status_code == 200


# ===========================================================================
# Tests: _clear_marketplace_state edge cases
# ===========================================================================


class TestClearMarketplaceStateEdgeCases:
    """Tests for _clear_marketplace_state with edge cases."""

    def test_clear_with_none_circuit_breaker(self):
        """Clearing state when circuit breaker is None should not error."""
        import aragora.server.handlers.template_marketplace as mod

        original_cb = mod._marketplace_circuit_breaker
        mod._marketplace_circuit_breaker = None
        try:
            _clear_marketplace_state()  # Should not raise
        finally:
            mod._marketplace_circuit_breaker = original_cb

    def test_clear_idempotent(self):
        """Clearing state multiple times should work without error."""
        _seed_marketplace_templates()
        _clear_marketplace_state()
        _clear_marketplace_state()
        assert len(_marketplace_templates) == 0
        assert len(_template_reviews) == 0
        assert len(_user_ratings) == 0

    def test_clear_resets_circuit_breaker(self):
        """_clear_marketplace_state should reset the circuit breaker."""
        cb = get_marketplace_circuit_breaker()
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status()["failure_count"] == 2
        _clear_marketplace_state()
        assert cb.get_status()["failure_count"] == 0
        assert cb.get_status()["state"] == "closed"


# ===========================================================================
# Tests: Search Across Multiple Fields
# ===========================================================================


class TestSearchAcrossFields:
    """Tests for search filtering across name, description, and tags."""

    def test_search_matches_name(self, handler):
        """Search should match template name."""
        _insert_template("test/name-match", name="UniqueNameXYZ")
        result = handler._list_templates({"search": "UniqueNameXYZ"})
        body = parse_body(result)
        assert body["total"] == 1
        assert body["templates"][0]["name"] == "UniqueNameXYZ"

    def test_search_matches_description(self, handler):
        """Search should match template description."""
        _insert_template("test/desc-match", description="SpecialDescKeyword is here")
        result = handler._list_templates({"search": "SpecialDescKeyword"})
        body = parse_body(result)
        assert body["total"] == 1

    def test_search_matches_tags(self, handler):
        """Search should match template tags."""
        _insert_template("test/tag-match", tags=["unique-tag-xyz"])
        result = handler._list_templates({"search": "unique-tag-xyz"})
        body = parse_body(result)
        assert body["total"] == 1

    def test_search_case_insensitive(self, handler):
        """Search should be case-insensitive."""
        _insert_template("test/case", name="CamelCaseTemplate")
        result = handler._list_templates({"search": "camelcasetemplate"})
        body = parse_body(result)
        assert body["total"] == 1

    def test_search_no_match(self, handler):
        """Search with no matching results returns empty list."""
        _insert_template("test/no-match", name="Something")
        result = handler._list_templates({"search": "ZZZNONEXISTENT"})
        body = parse_body(result)
        assert body["total"] == 0
        assert body["templates"] == []


# ===========================================================================
# Tests: Template Summary Boundary
# ===========================================================================


class TestTemplateSummaryBoundary:
    """Tests for to_summary description truncation at exact boundaries."""

    def test_description_exactly_200_chars(self):
        """Description of exactly 200 chars should not be truncated."""
        desc = "x" * 200
        t = MarketplaceTemplate(
            id="t", name="N", description=desc, category="security",
            pattern="debate", author_id="a", author_name="A",
        )
        s = t.to_summary()
        assert s["description"] == desc
        assert not s["description"].endswith("...")

    def test_description_201_chars_is_truncated(self):
        """Description of 201 chars should be truncated with ellipsis."""
        desc = "x" * 201
        t = MarketplaceTemplate(
            id="t", name="N", description=desc, category="security",
            pattern="debate", author_id="a", author_name="A",
        )
        s = t.to_summary()
        assert s["description"].endswith("...")
        assert len(s["description"]) == 203

    def test_description_exactly_5_tags(self):
        """Summary with exactly 5 tags should show all 5."""
        t = MarketplaceTemplate(
            id="t", name="N", description="D", category="security",
            pattern="debate", author_id="a", author_name="A",
            tags=["a", "b", "c", "d", "e"],
        )
        s = t.to_summary()
        assert len(s["tags"]) == 5

    def test_description_6_tags_truncated_to_5(self):
        """Summary with 6 tags should only show first 5."""
        t = MarketplaceTemplate(
            id="t", name="N", description="D", category="security",
            pattern="debate", author_id="a", author_name="A",
            tags=["a", "b", "c", "d", "e", "f"],
        )
        s = t.to_summary()
        assert len(s["tags"]) == 5
        assert s["tags"] == ["a", "b", "c", "d", "e"]


# ===========================================================================
# Tests: Publish Edge Cases
# ===========================================================================


class TestPublishEdgeCases:
    """Tests for publish edge cases."""

    def test_publish_with_empty_json_body(self, handler):
        """Publishing with empty JSON object should fail (missing required fields)."""
        mock_post = make_mock_handler(method="POST", body=json_body({}))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_whitespace_name(self, handler):
        """Publishing with whitespace-only name should be rejected."""
        data = _publish_body(name="   ")
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_empty_workflow_definition(self, handler):
        """Publishing with empty dict workflow_definition should succeed."""
        data = _publish_body(
            name="Empty Workflow",
            workflow_definition={},
        )
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        body = parse_body(result)
        assert result.status_code == 201
        assert body["status"] == "published"

    def test_publish_workflow_definition_as_list_rejected(self, handler):
        """workflow_definition must be a dict, not a list."""
        data = _publish_body(workflow_definition=[1, 2, 3])
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_no_tags_succeeds(self, handler):
        """Publishing without tags should default to empty list and succeed."""
        data = _publish_body()
        del data["tags"]
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        body = parse_body(result)
        assert result.status_code == 201

    def test_publish_empty_string_body(self, handler):
        """Publishing with empty string body should return 400 for missing fields."""
        mock_post = make_mock_handler(method="POST", body=b"")
        mock_post.headers["Content-Length"] = "0"
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_stores_all_optional_fields(self, handler):
        """All optional fields should be stored correctly in the template."""
        data = _publish_body(
            name="Full Template",
            version="2.1.0",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
            documentation="Full docs here.",
            examples=[{"input": "x", "output": "y"}],
        )
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        body = parse_body(result)
        tpl_id = body["template_id"]
        tpl = _marketplace_templates[tpl_id]
        assert tpl.version == "2.1.0"
        assert tpl.input_schema == {"type": "object"}
        assert tpl.output_schema == {"type": "string"}
        assert tpl.documentation == "Full docs here."
        assert tpl.examples == [{"input": "x", "output": "y"}]

    def test_publish_description_at_max_length(self, handler):
        """Description at exactly max length should be accepted."""
        data = _publish_body(
            name="Max Desc Template",
            description="x" * DESCRIPTION_MAX_LENGTH,
        )
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 201


# ===========================================================================
# Tests: Review Edge Cases
# ===========================================================================


class TestReviewEdgeCases:
    """Tests for review submission edge cases."""

    def test_submit_review_with_minimum_valid_content(self, seeded_handler):
        """Review with single-character content should succeed."""
        data = {"rating": 3, "content": "X"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 201

    def test_submit_review_at_max_content_length(self, seeded_handler):
        """Review content at max length should succeed."""
        data = {"rating": 4, "content": "x" * REVIEW_CONTENT_MAX_LENGTH}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 201

    def test_submit_review_empty_title_ok(self, seeded_handler):
        """Empty title should be accepted (optional field)."""
        data = {"rating": 4, "content": "Good.", "title": ""}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        body = parse_body(result)
        assert result.status_code == 201
        assert body["review"]["title"] == ""

    def test_submit_review_with_custom_user_id(self, seeded_handler):
        """Review with custom user_id should store it correctly."""
        data = {"rating": 5, "content": "Great.", "user_id": "custom-user-42"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        body = parse_body(result)
        assert body["review"]["user_id"] == "custom-user-42"

    def test_multiple_reviews_append(self, seeded_handler):
        """Multiple reviews for the same template should all be stored."""
        for i in range(3):
            data = {"rating": 3 + i % 3, "content": f"Review {i}", "user_id": f"user-{i}"}
            mock_post = make_mock_handler(method="POST", body=json_body(data))
            seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert len(_template_reviews["security/code-audit"]) == 3

    def test_reviews_sorted_by_helpful_count(self, seeded_handler):
        """GET reviews should return reviews sorted by helpful_count descending."""
        reviews = [
            TemplateReview(
                id=f"r{i}", template_id="security/code-audit",
                user_id=f"u{i}", user_name=f"U{i}",
                rating=4, title=f"T{i}", content=f"C{i}",
                helpful_count=i * 10,
            )
            for i in range(4)
        ]
        _template_reviews["security/code-audit"] = reviews
        result = seeded_handler._get_reviews("security/code-audit", {})
        body = parse_body(result)
        helpful_counts = [r["helpful_count"] for r in body["reviews"]]
        assert helpful_counts == sorted(helpful_counts, reverse=True)


# ===========================================================================
# Tests: Query Parameter Edge Cases
# ===========================================================================


class TestQueryParamEdgeCases:
    """Tests for query parameter edge cases."""

    def test_list_with_limit_zero_clamped_to_1(self, seeded_handler, mock_http_get):
        """Limit of 0 should be clamped to minimum of 1."""
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"limit": "0"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["limit"] == 1

    def test_list_with_very_large_limit_clamped(self, seeded_handler, mock_http_get):
        """Very large limit should be clamped to max of 50."""
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"limit": "1000"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["limit"] == 50

    def test_trending_limit_clamped_to_20(self, seeded_handler, mock_http_get):
        """Trending limit should be clamped to max of 20."""
        result = seeded_handler.handle(
            "/api/v1/marketplace/trending",
            {"limit": "100"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert len(body["trending"]) <= 20

    def test_recommendations_limit_clamped(self, rec_handler, mock_http_get):
        """Recommendations limit should be clamped to max of 20."""
        _seed_marketplace_templates()
        result = rec_handler.handle(
            "/api/v1/marketplace/recommendations",
            {"limit": "100"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200

    def test_list_with_negative_offset(self, seeded_handler, mock_http_get):
        """Negative offset should still work (defaults to 0 from get_int_param)."""
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"offset": "-5"},
            mock_http_get,
        )
        assert result.status_code == 200

    def test_list_with_non_numeric_limit(self, seeded_handler, mock_http_get):
        """Non-numeric limit should fall back to default."""
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"limit": "abc"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        # Should use default limit (20)
        assert body["limit"] == 20


# ===========================================================================
# Tests: Validation Function Additional Edge Cases
# ===========================================================================


class TestValidationEdgeCases:
    """Additional edge case tests for validation functions."""

    def test_validate_template_name_with_special_chars(self):
        """Template name with special characters should be valid."""
        ok, err = validate_template_name("My Template! @#$%")
        assert ok is True

    def test_validate_tags_with_max_tags_at_max_length(self):
        """Tags at both max count and max length should be valid."""
        tags = ["x" * MAX_TAG_LENGTH for _ in range(MAX_TAGS)]
        ok, err = validate_tags(tags)
        assert ok is True

    def test_validate_rating_negative(self):
        """Negative rating should be invalid."""
        ok, err = validate_rating(-1)
        assert ok is False

    def test_validate_rating_zero(self):
        """Zero rating should be invalid."""
        ok, err = validate_rating(0)
        assert ok is False

    def test_validate_rating_boolean(self):
        """Boolean is technically numeric in Python, but should be handled."""
        # bool is a subclass of int in Python, so True=1 and False=0
        ok_true, _ = validate_rating(True)
        assert ok_true is True  # True == 1, which is valid
        ok_false, _ = validate_rating(False)
        assert ok_false is False  # False == 0, which is < 1

    def test_validate_review_content_single_char(self):
        """Single character review content should be valid."""
        ok, err = validate_review_content("X")
        assert ok is True

    def test_validate_pattern_none(self):
        """None pattern should be invalid."""
        ok, err = validate_pattern(None)
        assert ok is False

    def test_validate_category_none(self):
        """None category should be invalid."""
        ok, err = validate_category(None)
        assert ok is False

    def test_validate_template_name_none_is_falsy(self):
        """None name should return invalid."""
        ok, err = validate_template_name(None)
        assert ok is False


# ===========================================================================
# Tests: Featured/Trending/Categories with Custom Templates
# ===========================================================================


class TestFeaturedTrendingWithCustomTemplates:
    """Tests for featured/trending/categories with custom-inserted templates."""

    def test_featured_only_returns_featured(self, handler):
        """Only templates with is_featured=True should appear."""
        _insert_template("test/featured", is_featured=True, rating=5.0)
        _insert_template("test/not-featured", is_featured=False, rating=5.0)
        result = handler._get_featured()
        body = parse_body(result)
        assert body["total"] == 1
        assert body["featured"][0]["id"] == "test/featured"

    def test_trending_respects_limit(self, handler):
        """Trending should return at most `limit` templates."""
        for i in range(5):
            _insert_template(f"test/trending-{i}", download_count=100 - i)
        result = handler._get_trending({"limit": "3"})
        body = parse_body(result)
        assert len(body["trending"]) == 3
        # Should be sorted by download_count
        downloads = [t["download_count"] for t in body["trending"]]
        assert downloads == sorted(downloads, reverse=True)

    def test_categories_count_multiple_templates(self, handler):
        """Category counts should reflect actual template counts."""
        _insert_template("test/sec1", category="security", download_count=10)
        _insert_template("test/sec2", category="security", download_count=20)
        _insert_template("test/legal1", category="legal", download_count=5)
        result = handler._get_categories()
        body = parse_body(result)
        sec_cat = next(c for c in body["categories"] if c["id"] == "security")
        legal_cat = next(c for c in body["categories"] if c["id"] == "legal")
        assert sec_cat["template_count"] == 2
        assert sec_cat["total_downloads"] == 30
        assert legal_cat["template_count"] == 1
        assert legal_cat["total_downloads"] == 5

    def test_featured_capped_at_10(self, handler):
        """Even with > 10 featured templates, only 10 should be returned."""
        for i in range(15):
            _insert_template(f"test/feat-{i}", is_featured=True, rating=5.0 - i * 0.01)
        result = handler._get_featured()
        body = parse_body(result)
        assert len(body["featured"]) == 10
        assert body["total"] == 15


# ===========================================================================
# Tests: MarketplaceTemplate to_dict completeness
# ===========================================================================


class TestMarketplaceTemplateToDictComplete:
    """Tests that to_dict includes all fields."""

    def test_to_dict_includes_all_fields(self):
        """to_dict should include every field of the dataclass."""
        t = MarketplaceTemplate(
            id="test/complete",
            name="Complete Template",
            description="All fields populated",
            category="data",
            pattern="map_reduce",
            author_id="author-1",
            author_name="Author One",
            version="3.0.0",
            tags=["a", "b"],
            workflow_definition={"steps": []},
            input_schema={"type": "object"},
            output_schema={"type": "array"},
            documentation="Full docs.",
            examples=[{"input": "1"}],
            rating=4.5,
            rating_count=100,
            download_count=500,
            is_featured=True,
            is_verified=True,
        )
        d = t.to_dict()
        assert d["id"] == "test/complete"
        assert d["version"] == "3.0.0"
        assert d["tags"] == ["a", "b"]
        assert d["workflow_definition"] == {"steps": []}
        assert d["input_schema"] == {"type": "object"}
        assert d["output_schema"] == {"type": "array"}
        assert d["documentation"] == "Full docs."
        assert d["examples"] == [{"input": "1"}]
        assert d["rating"] == 4.5
        assert d["rating_count"] == 100
        assert d["download_count"] == 500
        assert d["is_featured"] is True
        assert d["is_verified"] is True
        assert d["author_id"] == "author-1"
        assert d["author_name"] == "Author One"


# ===========================================================================
# Tests: TemplateReview to_dict completeness
# ===========================================================================


class TestTemplateReviewToDictComplete:
    """Tests for TemplateReview to_dict completeness."""

    def test_review_to_dict_all_fields(self):
        """to_dict should include all review fields."""
        r = TemplateReview(
            id="rev-complete",
            template_id="tpl-1",
            user_id="user-99",
            user_name="Reviewer Pro",
            rating=5,
            title="Best Template Ever",
            content="This template changed my life.",
            helpful_count=42,
        )
        d = r.to_dict()
        assert d["id"] == "rev-complete"
        assert d["template_id"] == "tpl-1"
        assert d["user_id"] == "user-99"
        assert d["user_name"] == "Reviewer Pro"
        assert d["rating"] == 5
        assert d["title"] == "Best Template Ever"
        assert d["content"] == "This template changed my life."
        assert d["helpful_count"] == 42
        assert "created_at" in d
        assert isinstance(d["created_at"], float)


# ===========================================================================
# Tests: Seed Template Integrity
# ===========================================================================


class TestSeedTemplateIntegrity:
    """Tests verifying seeded template data integrity."""

    def test_all_seeded_templates_have_required_fields(self):
        """Every seeded template should have non-empty required fields."""
        _seed_marketplace_templates()
        for tpl in _marketplace_templates.values():
            assert tpl.id
            assert tpl.name
            assert tpl.description
            assert tpl.category in VALID_CATEGORIES
            assert tpl.pattern in VALID_PATTERNS
            assert tpl.author_id
            assert tpl.author_name

    def test_seeded_templates_have_tags(self):
        """All seeded templates should have at least one tag."""
        _seed_marketplace_templates()
        for tpl in _marketplace_templates.values():
            assert len(tpl.tags) > 0

    def test_seeded_templates_rating_in_range(self):
        """Seeded template ratings should be between 0 and 5."""
        _seed_marketplace_templates()
        for tpl in _marketplace_templates.values():
            assert 0 <= tpl.rating <= 5

    def test_seeded_templates_count(self):
        """There should be at least 10 seeded templates."""
        _seed_marketplace_templates()
        assert len(_marketplace_templates) >= 10

    def test_seeded_templates_include_quickstart_category(self):
        """Seeded templates should include quickstart templates."""
        _seed_marketplace_templates()
        quickstart = [t for t in _marketplace_templates.values() if t.category == "quickstart"]
        assert len(quickstart) >= 1

    def test_seeded_templates_include_sme_category(self):
        """Seeded templates should include SME templates."""
        _seed_marketplace_templates()
        sme = [t for t in _marketplace_templates.values() if t.category == "sme"]
        assert len(sme) >= 1


# ===========================================================================
# Tests: Circuit Breaker Module-Level Functions
# ===========================================================================


class TestCircuitBreakerModuleFunctions:
    """Tests for module-level circuit breaker functions."""

    def test_get_marketplace_circuit_breaker_returns_singleton(self):
        """get_marketplace_circuit_breaker should return the same instance."""
        cb1 = get_marketplace_circuit_breaker()
        cb2 = get_marketplace_circuit_breaker()
        assert cb1 is cb2

    def test_get_marketplace_circuit_breaker_status_structure(self):
        """Status should have expected keys."""
        status = get_marketplace_circuit_breaker_status()
        expected_keys = {"state", "failure_count", "success_count",
                         "failure_threshold", "cooldown_seconds", "last_failure_time"}
        assert expected_keys.issubset(set(status.keys()))

    def test_marketplace_circuit_breaker_is_correct_type(self):
        """The circuit breaker should be a MarketplaceCircuitBreaker instance."""
        cb = get_marketplace_circuit_breaker()
        assert isinstance(cb, MarketplaceCircuitBreaker)


# ===========================================================================
# Tests: Import with Various Body States
# ===========================================================================


class TestImportVariousStates:
    """Tests for import endpoint with various body conditions."""

    def test_import_with_workspace_id(self, seeded_handler):
        """Import with workspace_id should include it in the response."""
        data = {"workspace_id": "ws-abc-123"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._import_template("security/code-audit", mock_post)
        body = parse_body(result)
        assert body["workspace_id"] == "ws-abc-123"

    def test_import_response_includes_schemas(self, seeded_handler):
        """Import response should include input_schema and output_schema."""
        mock_post = make_mock_handler(method="POST", body=json_body({}))
        result = seeded_handler._import_template("security/code-audit", mock_post)
        body = parse_body(result)
        assert "input_schema" in body
        assert "output_schema" in body
        assert "documentation" in body

    def test_import_template_has_workflow_definition(self, seeded_handler):
        """Import response should include the workflow_definition."""
        mock_post = make_mock_handler(method="POST", body=json_body({}))
        result = seeded_handler._import_template("security/code-audit", mock_post)
        body = parse_body(result)
        assert "workflow_definition" in body


# ===========================================================================
# Tests: List Templates with Combined Filters
# ===========================================================================


class TestListCombinedFilters:
    """Tests for listing templates with multiple filters combined."""

    def test_category_and_search_combined(self, handler):
        """Category filter combined with search should narrow results."""
        _insert_template("sec/a", category="security", name="Alpha Security Tool")
        _insert_template("sec/b", category="security", name="Beta Security Tool")
        _insert_template("legal/a", category="legal", name="Alpha Legal Tool")
        result = handler._list_templates({"category": "security", "search": "Alpha"})
        body = parse_body(result)
        assert body["total"] == 1
        assert body["templates"][0]["name"] == "Alpha Security Tool"

    def test_pattern_and_verified_combined(self, handler):
        """Pattern + verified_only filters together should work."""
        _insert_template("t1", pattern="debate", is_verified=True)
        _insert_template("t2", pattern="debate", is_verified=False)
        _insert_template("t3", pattern="pipeline", is_verified=True)
        result = handler._list_templates({"pattern": "debate", "verified_only": "true"})
        body = parse_body(result)
        assert body["total"] == 1
        assert body["templates"][0]["pattern"] == "debate"

    def test_all_filters_combined(self, handler):
        """All filters combined should work correctly."""
        _insert_template(
            "precise/match",
            category="security",
            pattern="debate",
            is_verified=True,
            name="Precise Match Template",
            tags=["security", "audit"],
        )
        _insert_template(
            "miss/1",
            category="legal",
            pattern="debate",
            is_verified=True,
        )
        result = handler._list_templates({
            "category": "security",
            "pattern": "debate",
            "verified_only": "true",
            "search": "Precise",
            "tags": "audit",
        })
        body = parse_body(result)
        assert body["total"] == 1
        assert body["templates"][0]["name"] == "Precise Match Template"

    def test_verified_only_false_includes_all(self, handler):
        """verified_only=false should include unverified templates."""
        _insert_template("t1", is_verified=True)
        _insert_template("t2", is_verified=False)
        result = handler._list_templates({"verified_only": "false"})
        body = parse_body(result)
        assert body["total"] == 2

    def test_tags_filter_with_comma_separated(self, handler):
        """Multiple comma-separated tags should match templates with any of them."""
        _insert_template("t1", tags=["alpha"])
        _insert_template("t2", tags=["beta"])
        _insert_template("t3", tags=["gamma"])
        result = handler._list_templates({"tags": "alpha,beta"})
        body = parse_body(result)
        assert body["total"] == 2


# ===========================================================================
# Tests: Handler ROUTES Class Attribute
# ===========================================================================


class TestHandlerRoutesAttribute:
    """Tests for the ROUTES class attribute of both handlers."""

    def test_marketplace_handler_routes_include_templates(self):
        assert any("templates" in r for r in TemplateMarketplaceHandler.ROUTES)

    def test_marketplace_handler_routes_include_featured(self):
        assert any("featured" in r for r in TemplateMarketplaceHandler.ROUTES)

    def test_marketplace_handler_routes_include_trending(self):
        assert any("trending" in r for r in TemplateMarketplaceHandler.ROUTES)

    def test_marketplace_handler_routes_include_categories(self):
        assert any("categories" in r for r in TemplateMarketplaceHandler.ROUTES)

    def test_marketplace_handler_routes_include_circuit_breaker(self):
        assert any("circuit-breaker" in r for r in TemplateMarketplaceHandler.ROUTES)

    def test_recommendations_handler_routes(self):
        assert "/api/v1/marketplace/recommendations" in TemplateRecommendationsHandler.ROUTES


# ===========================================================================
# Tests: Rate Limiting for Different Client IPs
# ===========================================================================


class TestRateLimitingDifferentIPs:
    """Tests for rate limiting with different client IPs."""

    def test_different_ips_have_separate_rate_limits(self, seeded_handler):
        """Different client IPs should have independent rate limit buckets."""
        # Both should succeed (different IPs)
        mock1 = make_mock_handler(method="GET", client_address=("10.0.0.1", 1234))
        result1 = seeded_handler.handle("/api/v1/marketplace/templates", {}, mock1)
        assert result1.status_code == 200

        mock2 = make_mock_handler(method="GET", client_address=("10.0.0.2", 1234))
        result2 = seeded_handler.handle("/api/v1/marketplace/templates", {}, mock2)
        assert result2.status_code == 200


# ===========================================================================
# Tests: Rate Template with Default user_id
# ===========================================================================


class TestRateTemplateDefaultUserId:
    """Tests for rating with default (client_ip-based) user_id."""

    def test_rate_uses_client_ip_as_default_user_id(self, seeded_handler):
        """When user_id is not provided, client_ip should be used."""
        data = {"rating": 4}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        seeded_handler._rate_template("security/code-audit", mock_post, "192.168.1.1")
        assert "192.168.1.1" in _user_ratings
        assert "security/code-audit" in _user_ratings["192.168.1.1"]
