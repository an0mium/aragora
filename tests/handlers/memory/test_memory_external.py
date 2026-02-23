"""Tests for MemoryExternalMixin (aragora/server/handlers/memory/memory_external.py).

Covers all methods and behavior of the MemoryExternalMixin class:
- _get_supermemory_adapter: adapter client creation and caching
- _search_supermemory: searching supermemory with results, errors, empty states
- _search_claude_mem: searching claude-mem with results, errors, empty states
- _estimate_tokens: static token estimation
- MEMORY_READ_PERMISSION constant
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Test harness: concrete class that mixes in MemoryExternalMixin
# ---------------------------------------------------------------------------


class _TestExternalHandler:
    """Minimal concrete class that uses MemoryExternalMixin for testing."""

    pass


def _make_handler_class():
    """Build a test class that inherits from MemoryExternalMixin."""
    from aragora.server.handlers.memory.memory_external import MemoryExternalMixin

    class ConcreteHandler(MemoryExternalMixin):
        pass

    return ConcreteHandler


def _make_handler():
    """Instantiate a ConcreteHandler."""
    cls = _make_handler_class()
    return cls()


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_supermemory_result_item(
    *,
    memory_id: str = "sm-001",
    content: str = "Supermemory result content",
    similarity: float = 0.95,
    metadata: dict | None = None,
    container_tag: str | None = None,
):
    """Create a mock supermemory search result item."""
    item = MagicMock()
    item.memory_id = memory_id
    item.content = content
    item.similarity = similarity
    item.metadata = metadata or {}
    item.container_tag = container_tag
    return item


def _make_claude_mem_item(
    *,
    id: str = "cm-001",
    content: str = "Claude-mem result content",
    metadata: dict | None = None,
    created_at: str = "2026-01-15T10:00:00Z",
):
    """Create a mock claude-mem evidence item."""
    item = MagicMock()
    item.id = id
    item.content = content
    item.metadata = metadata or {}
    item.created_at = created_at
    return item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a MemoryExternalMixin handler instance."""
    return _make_handler()


# ===========================================================================
# Tests: MEMORY_READ_PERMISSION constant
# ===========================================================================


class TestMemoryReadPermission:
    """Tests for the module-level MEMORY_READ_PERMISSION constant."""

    def test_permission_constant_value(self):
        from aragora.server.handlers.memory.memory_external import MEMORY_READ_PERMISSION

        assert MEMORY_READ_PERMISSION == "memory:read"

    def test_permission_constant_is_string(self):
        from aragora.server.handlers.memory.memory_external import MEMORY_READ_PERMISSION

        assert isinstance(MEMORY_READ_PERMISSION, str)


# ===========================================================================
# Tests: _estimate_tokens
# ===========================================================================


class TestEstimateTokens:
    """Tests for the static _estimate_tokens method."""

    def test_empty_string_returns_zero(self, handler):
        assert handler._estimate_tokens("") == 0

    def test_none_like_empty(self, handler):
        # Empty string is falsy
        assert handler._estimate_tokens("") == 0

    def test_single_char(self, handler):
        # 1 char / 4 = 0.25, ceil = 1, max(1, 1) = 1
        assert handler._estimate_tokens("a") == 1

    def test_four_chars(self, handler):
        # 4 / 4 = 1.0, ceil = 1
        assert handler._estimate_tokens("abcd") == 1

    def test_five_chars(self, handler):
        # 5 / 4 = 1.25, ceil = 2
        assert handler._estimate_tokens("abcde") == 2

    def test_hundred_chars(self, handler):
        text = "x" * 100
        # 100 / 4 = 25
        assert handler._estimate_tokens(text) == 25

    def test_large_text(self, handler):
        text = "word " * 1000  # 5000 chars
        expected = math.ceil(len(text) / 4)
        assert handler._estimate_tokens(text) == expected

    def test_always_returns_at_least_one_for_nonempty(self, handler):
        assert handler._estimate_tokens("x") >= 1

    def test_returns_int(self, handler):
        result = handler._estimate_tokens("some text here")
        assert isinstance(result, int)

    def test_static_method_callable_on_class(self):
        cls = _make_handler_class()
        assert cls._estimate_tokens("abcd") == 1

    def test_unicode_text(self, handler):
        # Unicode chars still count by len()
        text = "\u00e9\u00e8\u00ea\u00eb"  # 4 accented chars
        assert handler._estimate_tokens(text) == 1


# ===========================================================================
# Tests: _get_supermemory_adapter
# ===========================================================================


class TestGetSupermemoryAdapter:
    """Tests for the _get_supermemory_adapter method."""

    def test_returns_cached_client(self, handler):
        """If _supermemory_client already set, return it directly."""
        fake_client = MagicMock()
        handler._supermemory_client = fake_client
        result = handler._get_supermemory_adapter()
        assert result is fake_client

    @patch(
        "aragora.server.handlers.memory.memory_external.MemoryExternalMixin._get_supermemory_adapter"
    )
    def test_returns_none_on_import_error(self, mock_method, handler):
        """If supermemory connector is not installed, return None."""
        # Test the real behavior by not using the mock
        pass

    def test_import_error_returns_none(self, handler):
        """When aragora.connectors.supermemory is not importable, return None."""
        with patch.dict("sys.modules", {"aragora.connectors.supermemory": None}):
            # Force a fresh lookup (no cached client)
            assert not hasattr(handler, "_supermemory_client")
            result = handler._get_supermemory_adapter()
            assert result is None

    def test_config_from_env_returns_none(self, handler):
        """When SupermemoryConfig.from_env() returns None, return None."""
        mock_module = MagicMock()
        mock_module.SupermemoryConfig.from_env.return_value = None
        with patch.dict(
            "sys.modules", {"aragora.connectors.supermemory": mock_module}
        ):
            result = handler._get_supermemory_adapter()
            assert result is None

    def test_successful_client_creation(self, handler):
        """When config and client are valid, return client and cache it."""
        mock_config = MagicMock()
        mock_config.container_tag = "test-container"
        mock_client = MagicMock()

        mock_module = MagicMock()
        mock_module.SupermemoryConfig.from_env.return_value = mock_config
        mock_module.get_client.return_value = mock_client

        with patch.dict(
            "sys.modules", {"aragora.connectors.supermemory": mock_module}
        ):
            result = handler._get_supermemory_adapter()
            assert result is mock_client
            # Verify it was cached
            assert handler._supermemory_client is mock_client
            assert handler._supermemory_config is mock_config

    def test_client_creation_caches_for_subsequent_calls(self, handler):
        """Second call returns cached client without re-creating."""
        mock_config = MagicMock()
        mock_client = MagicMock()

        mock_module = MagicMock()
        mock_module.SupermemoryConfig.from_env.return_value = mock_config
        mock_module.get_client.return_value = mock_client

        with patch.dict(
            "sys.modules", {"aragora.connectors.supermemory": mock_module}
        ):
            first = handler._get_supermemory_adapter()
            # Second call should use cached
            second = handler._get_supermemory_adapter()
            assert first is second
            assert first is mock_client

    def test_get_client_connection_error_returns_none(self, handler):
        """When get_client raises ConnectionError, return None."""
        mock_config = MagicMock()
        mock_module = MagicMock()
        mock_module.SupermemoryConfig.from_env.return_value = mock_config
        mock_module.get_client.side_effect = ConnectionError("refused")

        with patch.dict(
            "sys.modules", {"aragora.connectors.supermemory": mock_module}
        ):
            result = handler._get_supermemory_adapter()
            assert result is None

    def test_get_client_timeout_error_returns_none(self, handler):
        """When get_client raises TimeoutError, return None."""
        mock_config = MagicMock()
        mock_module = MagicMock()
        mock_module.SupermemoryConfig.from_env.return_value = mock_config
        mock_module.get_client.side_effect = TimeoutError("timed out")

        with patch.dict(
            "sys.modules", {"aragora.connectors.supermemory": mock_module}
        ):
            result = handler._get_supermemory_adapter()
            assert result is None

    def test_get_client_value_error_returns_none(self, handler):
        """When get_client raises ValueError, return None."""
        mock_config = MagicMock()
        mock_module = MagicMock()
        mock_module.SupermemoryConfig.from_env.return_value = mock_config
        mock_module.get_client.side_effect = ValueError("bad config")

        with patch.dict(
            "sys.modules", {"aragora.connectors.supermemory": mock_module}
        ):
            result = handler._get_supermemory_adapter()
            assert result is None

    def test_get_client_runtime_error_returns_none(self, handler):
        """When get_client raises RuntimeError, return None."""
        mock_config = MagicMock()
        mock_module = MagicMock()
        mock_module.SupermemoryConfig.from_env.return_value = mock_config
        mock_module.get_client.side_effect = RuntimeError("init failed")

        with patch.dict(
            "sys.modules", {"aragora.connectors.supermemory": mock_module}
        ):
            result = handler._get_supermemory_adapter()
            assert result is None

    def test_get_client_os_error_returns_none(self, handler):
        """When get_client raises OSError, return None."""
        mock_config = MagicMock()
        mock_module = MagicMock()
        mock_module.SupermemoryConfig.from_env.return_value = mock_config
        mock_module.get_client.side_effect = OSError("network error")

        with patch.dict(
            "sys.modules", {"aragora.connectors.supermemory": mock_module}
        ):
            result = handler._get_supermemory_adapter()
            assert result is None


# ===========================================================================
# Tests: _search_supermemory
# ===========================================================================


class TestSearchSupermemory:
    """Tests for the _search_supermemory method."""

    def test_no_client_returns_empty_list(self, handler):
        """When adapter is not available, return empty list."""
        with patch.object(handler, "_get_supermemory_adapter", return_value=None):
            result = handler._search_supermemory("test query")
            assert result == []

    def test_basic_search_returns_formatted_results(self, handler):
        """Successful search returns properly formatted result dicts."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        item = _make_supermemory_result_item(
            memory_id="sm-100",
            content="Found memory content",
            similarity=0.88,
            metadata={"key": "val"},
            container_tag="tag-1",
        )
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = "test-tag"
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("test query", limit=5)

        assert len(results) == 1
        r = results[0]
        assert r["id"] == "sm-100"
        assert r["source"] == "supermemory"
        assert r["preview"] == "Found memory content"
        assert r["score"] == 0.88
        assert r["metadata"] == {"key": "val"}
        assert r["container_tag"] == "tag-1"

    def test_search_with_default_limit(self, handler):
        """Default limit is 10."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = []

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ) as mock_run:
                handler._search_supermemory("query")
                call_args = mock_run.call_args
                # The coroutine is called with limit=10
                mock_client.search.assert_called_once_with(
                    query="query", limit=10, container_tag=None
                )

    def test_search_custom_limit(self, handler):
        """Custom limit is passed through."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = []

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = "my-tag"
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                handler._search_supermemory("query", limit=25)
                mock_client.search.assert_called_once_with(
                    query="query", limit=25, container_tag="my-tag"
                )

    def test_preview_truncation_at_220(self, handler):
        """Content longer than 220 chars gets truncated with ellipsis."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        long_content = "A" * 300
        item = _make_supermemory_result_item(content=long_content)
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        preview = results[0]["preview"]
        assert preview.endswith("...")
        # 220 chars + "..."
        assert len(preview) == 223

    def test_preview_no_truncation_under_220(self, handler):
        """Content at or under 220 chars is not truncated."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        short_content = "B" * 220
        item = _make_supermemory_result_item(content=short_content)
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert "..." not in results[0]["preview"]
        assert results[0]["preview"] == short_content

    def test_preview_exactly_221_chars_gets_truncated(self, handler):
        """Content of exactly 221 chars is truncated."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        content = "C" * 221
        item = _make_supermemory_result_item(content=content)
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results[0]["preview"].endswith("...")

    def test_fallback_id_when_memory_id_is_none(self, handler):
        """When item has no memory_id, use super_{idx} as fallback."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        item = _make_supermemory_result_item(memory_id=None)
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results[0]["id"] == "super_0"

    def test_fallback_id_uses_index(self, handler):
        """Fallback id uses the enumeration index."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        items = [
            _make_supermemory_result_item(memory_id=None, content=f"item {i}")
            for i in range(3)
        ]
        mock_response.results = items

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results[0]["id"] == "super_0"
        assert results[1]["id"] == "super_1"
        assert results[2]["id"] == "super_2"

    def test_score_rounded_to_four_decimals(self, handler):
        """Similarity score is rounded to 4 decimal places."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        item = _make_supermemory_result_item(similarity=0.123456789)
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results[0]["score"] == 0.1235

    def test_token_estimate_included(self, handler):
        """token_estimate is computed via _estimate_tokens."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        content = "x" * 100  # 100/4 = 25 tokens
        item = _make_supermemory_result_item(content=content)
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results[0]["token_estimate"] == 25

    def test_empty_content_handling(self, handler):
        """Empty content results in preview='' and token_estimate=0."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        item = _make_supermemory_result_item(content="")
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results[0]["preview"] == ""
        assert results[0]["token_estimate"] == 0

    def test_none_content_treated_as_empty(self, handler):
        """None content is treated as empty string."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        item = MagicMock()
        item.memory_id = "sm-x"
        item.content = None
        item.similarity = 0.5
        item.metadata = {}
        item.container_tag = None
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results[0]["preview"] == ""
        assert results[0]["token_estimate"] == 0

    def test_none_metadata_treated_as_empty_dict(self, handler):
        """None metadata becomes empty dict."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        item = MagicMock()
        item.memory_id = "sm-m"
        item.content = "test"
        item.similarity = 0.5
        item.metadata = None
        item.container_tag = None
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results[0]["metadata"] == {}

    def test_response_with_no_results_attribute(self, handler):
        """When response has no results attribute, return empty list."""
        mock_client = MagicMock()
        mock_response = MagicMock(spec=[])  # no attributes
        # getattr(response, "results", []) will return []

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results == []

    def test_response_with_none_results(self, handler):
        """When response.results is None, return empty list."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = None

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results == []

    def test_no_config_attribute_uses_none_container_tag(self, handler):
        """When _supermemory_config not set, container_tag defaults gracefully."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = []

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            # Don't set _supermemory_config at all
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert results == []
        # container_tag should be None since config is None
        mock_client.search.assert_called_once_with(
            query="query", limit=10, container_tag=None
        )

    def test_multiple_results(self, handler):
        """Multiple search results are all returned."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        items = [
            _make_supermemory_result_item(
                memory_id=f"sm-{i}", content=f"Content {i}", similarity=0.9 - i * 0.1
            )
            for i in range(5)
        ]
        mock_response.results = items

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        assert len(results) == 5
        assert results[0]["id"] == "sm-0"
        assert results[4]["id"] == "sm-4"

    def test_search_connection_error_returns_empty(self, handler):
        """ConnectionError during search returns empty list."""
        mock_client = MagicMock()

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=ConnectionError("connection lost"),
            ):
                results = handler._search_supermemory("query")

        assert results == []

    def test_search_timeout_error_returns_empty(self, handler):
        """TimeoutError during search returns empty list."""
        mock_client = MagicMock()

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=TimeoutError("timed out"),
            ):
                results = handler._search_supermemory("query")

        assert results == []

    def test_search_os_error_returns_empty(self, handler):
        """OSError during search returns empty list."""
        mock_client = MagicMock()

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=OSError("os error"),
            ):
                results = handler._search_supermemory("query")

        assert results == []

    def test_search_value_error_returns_empty(self, handler):
        """ValueError during search returns empty list."""
        mock_client = MagicMock()

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=ValueError("invalid"),
            ):
                results = handler._search_supermemory("query")

        assert results == []

    def test_search_runtime_error_returns_empty(self, handler):
        """RuntimeError during search returns empty list."""
        mock_client = MagicMock()

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=RuntimeError("runtime fail"),
            ):
                results = handler._search_supermemory("query")

        assert results == []


# ===========================================================================
# Tests: _search_claude_mem
# ===========================================================================


class TestSearchClaudeMem:
    """Tests for the _search_claude_mem method."""

    def test_import_error_returns_empty_list(self, handler):
        """When ClaudeMemConnector is not importable, return empty list."""
        with patch.dict("sys.modules", {"aragora.connectors": MagicMock(spec=[])}):
            # spec=[] means no attributes, so importing ClaudeMemConnector will fail
            # But we need to simulate ImportError from the from-import
            with patch(
                "builtins.__import__", side_effect=ImportError("no module")
            ):
                result = handler._search_claude_mem("query")
                assert result == []

    def test_basic_search_returns_formatted_results(self, handler):
        """Successful search returns properly formatted result dicts."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        evidence = [
            _make_claude_mem_item(
                id="cm-100",
                content="Found memory",
                metadata={"tag": "test"},
                created_at="2026-02-01T12:00:00Z",
            )
        ]

        with patch(
            "aragora.server.handlers.memory.memory_external.run_async",
            return_value=evidence,
        ):
            with patch.dict("sys.modules", {}):
                # Patch the import within the method
                mock_module = MagicMock()
                mock_module.ClaudeMemConnector.return_value = mock_connector
                mock_module.ClaudeMemConfig.from_env.return_value = mock_config

                import importlib
                import sys

                with patch.object(
                    importlib, "import_module", return_value=mock_module
                ):
                    # Use a more direct approach - patch at the point of use
                    pass

        # Better approach: patch the connectors module directly
        mock_conn_cls = MagicMock()
        mock_conf_cls = MagicMock()
        mock_conn_cls.return_value = mock_connector
        mock_conf_cls.from_env.return_value = mock_config

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector = mock_conn_cls
        mock_connectors.ClaudeMemConfig = mock_conf_cls

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=evidence,
            ):
                results = handler._search_claude_mem("test query")

        assert len(results) == 1
        r = results[0]
        assert r["id"] == "cm-100"
        assert r["source"] == "claude-mem"
        assert r["preview"] == "Found memory"
        assert r["metadata"] == {"tag": "test"}
        assert r["created_at"] == "2026-02-01T12:00:00Z"

    def test_search_with_project_parameter(self, handler):
        """Project parameter is passed to connector.search."""
        mock_connector = MagicMock()
        mock_config = MagicMock()

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=[],
            ):
                handler._search_claude_mem("query", limit=15, project="my-project")
                mock_connector.search.assert_called_once_with(
                    "query", limit=15, project="my-project"
                )

    def test_search_default_params(self, handler):
        """Default limit=10 and project=None."""
        mock_connector = MagicMock()
        mock_config = MagicMock()

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=[],
            ):
                handler._search_claude_mem("query")
                mock_connector.search.assert_called_once_with(
                    "query", limit=10, project=None
                )

    def test_preview_truncation_at_220(self, handler):
        """Content longer than 220 chars gets truncated with ellipsis."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        long_content = "Z" * 250
        evidence = [_make_claude_mem_item(content=long_content)]

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=evidence,
            ):
                results = handler._search_claude_mem("query")

        assert results[0]["preview"].endswith("...")
        assert len(results[0]["preview"]) == 223

    def test_preview_no_truncation_under_220(self, handler):
        """Content at 220 chars is not truncated."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        content = "Y" * 200
        evidence = [_make_claude_mem_item(content=content)]

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=evidence,
            ):
                results = handler._search_claude_mem("query")

        assert "..." not in results[0]["preview"]

    def test_empty_content_handling(self, handler):
        """Empty content string handled correctly."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        evidence = [_make_claude_mem_item(content="")]

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=evidence,
            ):
                results = handler._search_claude_mem("query")

        assert results[0]["preview"] == ""
        assert results[0]["token_estimate"] == 0

    def test_none_content_treated_as_empty(self, handler):
        """None content is treated as empty string."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        item = MagicMock()
        item.id = "cm-n"
        item.content = None
        item.metadata = {}
        item.created_at = None

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=[item],
            ):
                results = handler._search_claude_mem("query")

        assert results[0]["preview"] == ""
        assert results[0]["token_estimate"] == 0

    def test_none_metadata_treated_as_empty_dict(self, handler):
        """None metadata becomes empty dict."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        item = MagicMock()
        item.id = "cm-m"
        item.content = "test"
        item.metadata = None
        item.created_at = "2026-01-01"

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=[item],
            ):
                results = handler._search_claude_mem("query")

        assert results[0]["metadata"] == {}

    def test_multiple_results(self, handler):
        """Multiple evidence items are all returned."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        evidence = [
            _make_claude_mem_item(id=f"cm-{i}", content=f"Memory {i}") for i in range(4)
        ]

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=evidence,
            ):
                results = handler._search_claude_mem("query")

        assert len(results) == 4
        for i in range(4):
            assert results[i]["id"] == f"cm-{i}"
            assert results[i]["source"] == "claude-mem"

    def test_search_connection_error_returns_empty(self, handler):
        """ConnectionError during search returns empty list."""
        mock_connector = MagicMock()
        mock_config = MagicMock()

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=ConnectionError("conn error"),
            ):
                results = handler._search_claude_mem("query")

        assert results == []

    def test_search_timeout_error_returns_empty(self, handler):
        """TimeoutError during search returns empty list."""
        mock_connector = MagicMock()
        mock_config = MagicMock()

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=TimeoutError("timeout"),
            ):
                results = handler._search_claude_mem("query")

        assert results == []

    def test_search_runtime_error_returns_empty(self, handler):
        """RuntimeError during search returns empty list."""
        mock_connector = MagicMock()
        mock_config = MagicMock()

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=RuntimeError("runtime"),
            ):
                results = handler._search_claude_mem("query")

        assert results == []

    def test_search_value_error_returns_empty(self, handler):
        """ValueError during search returns empty list."""
        mock_connector = MagicMock()
        mock_config = MagicMock()

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=ValueError("invalid"),
            ):
                results = handler._search_claude_mem("query")

        assert results == []

    def test_search_os_error_returns_empty(self, handler):
        """OSError during search returns empty list."""
        mock_connector = MagicMock()
        mock_config = MagicMock()

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                side_effect=OSError("os error"),
            ):
                results = handler._search_claude_mem("query")

        assert results == []

    def test_token_estimate_included(self, handler):
        """token_estimate is computed via _estimate_tokens."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        content = "a" * 80  # 80/4 = 20 tokens
        evidence = [_make_claude_mem_item(content=content)]

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=evidence,
            ):
                results = handler._search_claude_mem("query")

        assert results[0]["token_estimate"] == 20

    def test_created_at_included(self, handler):
        """created_at is extracted from evidence items."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        evidence = [_make_claude_mem_item(created_at="2026-03-01T08:00:00Z")]

        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=evidence,
            ):
                results = handler._search_claude_mem("query")

        assert results[0]["created_at"] == "2026-03-01T08:00:00Z"

    def test_none_id_preserved(self, handler):
        """When item has no id attribute, None is included."""
        mock_connector = MagicMock()
        mock_config = MagicMock()
        item = MagicMock(spec=[])  # no attributes
        # getattr with defaults will return None/empty
        mock_connectors = MagicMock()
        mock_connectors.ClaudeMemConnector.return_value = mock_connector
        mock_connectors.ClaudeMemConfig.from_env.return_value = mock_config

        with patch.dict("sys.modules", {"aragora.connectors": mock_connectors}):
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=[item],
            ):
                results = handler._search_claude_mem("query")

        assert results[0]["id"] is None


# ===========================================================================
# Tests: Integration / cross-cutting
# ===========================================================================


class TestMixinIntegration:
    """Cross-cutting tests for the mixin."""

    def test_mixin_is_a_class(self):
        """MemoryExternalMixin is importable as a class."""
        from aragora.server.handlers.memory.memory_external import MemoryExternalMixin

        assert isinstance(MemoryExternalMixin, type)

    def test_mixin_methods_exist(self, handler):
        """All expected methods are present."""
        assert hasattr(handler, "_get_supermemory_adapter")
        assert hasattr(handler, "_search_supermemory")
        assert hasattr(handler, "_search_claude_mem")
        assert hasattr(handler, "_estimate_tokens")

    def test_estimate_tokens_is_static(self):
        """_estimate_tokens is a static method."""
        from aragora.server.handlers.memory.memory_external import MemoryExternalMixin

        # Can call without instance
        assert MemoryExternalMixin._estimate_tokens("test") == 1

    def test_search_supermemory_uses_estimate_tokens(self, handler):
        """_search_supermemory calls _estimate_tokens for each result."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        item = _make_supermemory_result_item(content="twelve char")
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                with patch.object(
                    handler, "_estimate_tokens", return_value=42
                ) as mock_est:
                    results = handler._search_supermemory("query")
                    mock_est.assert_called_once_with("twelve char")
                    assert results[0]["token_estimate"] == 42

    def test_preview_trailing_space_stripped(self, handler):
        """Preview text has trailing whitespace stripped before truncation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Content with trailing spaces that truncation should strip
        content = "A" * 218 + "  " + "B" * 100  # total > 220
        item = _make_supermemory_result_item(content=content)
        mock_response.results = [item]

        with patch.object(
            handler, "_get_supermemory_adapter", return_value=mock_client
        ):
            handler._supermemory_config = MagicMock()
            handler._supermemory_config.container_tag = None
            with patch(
                "aragora.server.handlers.memory.memory_external.run_async",
                return_value=mock_response,
            ):
                results = handler._search_supermemory("query")

        preview = results[0]["preview"]
        # First 220 chars = "A"*218 + "  ", rstripped = "A"*218, then "..."
        assert preview == "A" * 218 + "..."
