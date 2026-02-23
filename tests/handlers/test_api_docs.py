"""Tests for the ApiDocsHandler REST endpoints.

Covers all 3 endpoints:
- GET /api/v1/docs/openapi.json  -- Full OpenAPI 3.1 spec (cached)
- GET /api/v1/docs/routes        -- Lightweight route summary from handler registry
- GET /api/v1/docs/stats         -- API statistics (endpoint counts by tag/method)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.api_docs import ApiDocsHandler


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult or raw dict."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


@pytest.fixture
def handler():
    return ApiDocsHandler({})


# ---------------------------------------------------------------------------
# ROUTES class attribute
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_routes_defined(self):
        assert len(ApiDocsHandler.ROUTES) == 3

    def test_routes_contains_openapi(self):
        assert "/api/v1/docs/openapi.json" in ApiDocsHandler.ROUTES

    def test_routes_contains_routes(self):
        assert "/api/v1/docs/routes" in ApiDocsHandler.ROUTES

    def test_routes_contains_stats(self):
        assert "/api/v1/docs/stats" in ApiDocsHandler.ROUTES


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_openapi_json_path(self, handler):
        assert handler.can_handle("/api/v1/docs/openapi.json", "GET")

    def test_routes_path(self, handler):
        assert handler.can_handle("/api/v1/docs/routes", "GET")

    def test_stats_path(self, handler):
        assert handler.can_handle("/api/v1/docs/stats", "GET")

    def test_default_method_is_get(self, handler):
        assert handler.can_handle("/api/v1/docs/openapi.json")

    def test_post_not_handled(self, handler):
        assert not handler.can_handle("/api/v1/docs/openapi.json", "POST")

    def test_put_not_handled(self, handler):
        assert not handler.can_handle("/api/v1/docs/routes", "PUT")

    def test_delete_not_handled(self, handler):
        assert not handler.can_handle("/api/v1/docs/stats", "DELETE")

    def test_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_partial_path_no_match(self, handler):
        assert not handler.can_handle("/api/v1/docs")

    def test_extra_suffix_no_match(self, handler):
        assert not handler.can_handle("/api/v1/docs/openapi.json/extra")


# ---------------------------------------------------------------------------
# handle() dispatch
# ---------------------------------------------------------------------------


class TestHandleDispatch:
    def test_dispatches_openapi(self, handler):
        with patch.object(handler, "_get_openapi_spec") as mock:
            mock.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/v1/docs/openapi.json", {}, None)
            mock.assert_called_once()
            assert result is not None

    def test_dispatches_routes(self, handler):
        with patch.object(handler, "_get_routes") as mock:
            mock.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/v1/docs/routes", {"tag": "foo"}, None)
            mock.assert_called_once_with({"tag": "foo"})
            assert result is not None

    def test_dispatches_stats(self, handler):
        with patch.object(handler, "_get_stats") as mock:
            mock.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/v1/docs/stats", {}, None)
            mock.assert_called_once()
            assert result is not None

    def test_returns_none_for_unknown_path(self, handler):
        result = handler.handle("/api/v1/unknown", {}, None)
        assert result is None


# ---------------------------------------------------------------------------
# GET /api/v1/docs/openapi.json
# ---------------------------------------------------------------------------


class TestGetOpenApiSpec:
    def test_success(self, handler):
        mock_schema = {
            "openapi": "3.1.0",
            "info": {"title": "Aragora", "version": "1.0.0"},
            "paths": {},
        }
        with patch(
            "aragora.server.handlers.api_docs.generate_openapi_schema",
            return_value=mock_schema,
            create=True,
        ):
            # Patch at the import location used inside the method
            with patch(
                "aragora.server.openapi_impl.generate_openapi_schema",
                return_value=mock_schema,
            ):
                result = handler._get_openapi_spec()

        assert result.status_code == 200
        assert result.content_type == "application/json"
        body = json.loads(result.body)
        assert body["openapi"] == "3.1.0"

    def test_success_has_cache_control_header(self, handler):
        mock_schema = {"openapi": "3.1.0", "paths": {}}
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_openapi_spec()

        assert result.headers is not None
        assert "Cache-Control" in result.headers
        assert "max-age=600" in result.headers["Cache-Control"]

    def test_import_error_returns_503(self, handler):
        with patch.dict("sys.modules", {"aragora.server.openapi_impl": None}):
            result = handler._get_openapi_spec()

        assert result.status_code == 503
        body = _body(result)
        assert "error" in body

    def test_value_error_returns_500(self, handler):
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            side_effect=ValueError("bad schema"),
        ):
            result = handler._get_openapi_spec()

        assert result.status_code == 500
        body = _body(result)
        assert "error" in body

    def test_type_error_returns_500(self, handler):
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            side_effect=TypeError("unexpected type"),
        ):
            result = handler._get_openapi_spec()

        assert result.status_code == 500
        body = _body(result)
        assert "error" in body

    def test_key_error_returns_500(self, handler):
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            side_effect=KeyError("missing_key"),
        ):
            result = handler._get_openapi_spec()

        assert result.status_code == 500

    def test_runtime_error_returns_500(self, handler):
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            side_effect=RuntimeError("schema gen failed"),
        ):
            result = handler._get_openapi_spec()

        assert result.status_code == 500

    def test_os_error_returns_500(self, handler):
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            side_effect=OSError("disk error"),
        ):
            result = handler._get_openapi_spec()

        assert result.status_code == 500

    def test_body_is_compact_json(self, handler):
        """Verify the JSON is serialized with compact separators (no spaces)."""
        mock_schema = {"openapi": "3.1.0", "info": {"title": "Test"}}
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_openapi_spec()

        raw = result.body.decode("utf-8")
        # Compact JSON uses no spaces after separators
        assert ": " not in raw
        assert ", " not in raw


# ---------------------------------------------------------------------------
# GET /api/v1/docs/routes
# ---------------------------------------------------------------------------


class TestGetRoutes:
    def _make_mock_handler_cls(self, name, routes, doc="", methods=None):
        """Create a mock handler class for introspection."""
        cls = type(name, (), {"ROUTES": routes, "__doc__": doc})
        if methods:
            for m in methods:
                setattr(cls, m, lambda self: None)
        return cls

    def test_success_returns_routes(self, handler):
        mock_cls = self._make_mock_handler_cls(
            "TestHandler",
            ["/api/v1/test"],
            doc="Test handler for testing.",
        )
        mock_rules = [("test", "Testing")]
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [mock_cls],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                mock_rules,
            ):
                result = handler._get_routes({})

        body = _body(result)
        assert result.status_code == 200
        assert body["total"] >= 1
        assert isinstance(body["routes"], list)
        route = body["routes"][0]
        assert route["path"] == "/api/v1/test"
        assert route["handler"] == "TestHandler"
        assert route["description"] == "Test handler for testing."
        assert route["tag"] == "Testing"

    def test_filter_by_tag(self, handler):
        cls_a = self._make_mock_handler_cls(
            "HandlerA", ["/api/v1/alpha"], doc="Alpha handler."
        )
        cls_b = self._make_mock_handler_cls(
            "HandlerB", ["/api/v1/beta"], doc="Beta handler."
        )
        mock_rules = [("alpha", "Alpha"), ("beta", "Beta")]
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [cls_a, cls_b],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                mock_rules,
            ):
                result = handler._get_routes({"tag": "alpha"})

        body = _body(result)
        assert body["total"] == 1
        assert body["routes"][0]["tag"] == "Alpha"

    def test_filter_by_tag_case_insensitive(self, handler):
        cls = self._make_mock_handler_cls(
            "HandlerA", ["/api/v1/alpha"], doc="Alpha handler."
        )
        mock_rules = [("alpha", "Alpha")]
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [cls],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                mock_rules,
            ):
                result = handler._get_routes({"tag": "ALPHA"})

        body = _body(result)
        assert body["total"] == 1

    def test_filter_by_method(self, handler):
        cls_get = self._make_mock_handler_cls(
            "GetHandler", ["/api/v1/get-only"], doc="GET only."
        )
        cls_post = self._make_mock_handler_cls(
            "PostHandler",
            ["/api/v1/post-only"],
            doc="POST only.",
            methods=["handle_post"],
        )
        mock_rules = []
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [cls_get, cls_post],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                mock_rules,
            ):
                result = handler._get_routes({"method": "POST"})

        body = _body(result)
        assert body["total"] == 1
        assert body["routes"][0]["handler"] == "PostHandler"

    def test_filter_by_method_case_insensitive(self, handler):
        cls = self._make_mock_handler_cls(
            "GetHandler", ["/api/v1/get-only"], doc="GET only."
        )
        mock_rules = []
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [cls],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                mock_rules,
            ):
                result = handler._get_routes({"method": "get"})

        body = _body(result)
        assert body["total"] == 1

    def test_combined_tag_and_method_filter(self, handler):
        cls = self._make_mock_handler_cls(
            "PostHandler",
            ["/api/v1/alpha"],
            doc="Alpha POST.",
            methods=["handle_post"],
        )
        mock_rules = [("alpha", "Alpha")]
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [cls],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                mock_rules,
            ):
                # Filter by tag=Alpha and method=GET should exclude this handler
                result = handler._get_routes({"tag": "Alpha", "method": "GET"})

        body = _body(result)
        assert body["total"] == 0

    def test_empty_routes(self, handler):
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                [],
            ):
                result = handler._get_routes({})

        body = _body(result)
        assert body["total"] == 0
        assert body["routes"] == []

    def test_duplicate_paths_deduplicated(self, handler):
        cls_a = self._make_mock_handler_cls(
            "HandlerA", ["/api/v1/dup"], doc="First."
        )
        cls_b = self._make_mock_handler_cls(
            "HandlerB", ["/api/v1/dup"], doc="Second."
        )
        mock_rules = []
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [cls_a, cls_b],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                mock_rules,
            ):
                result = handler._get_routes({})

        body = _body(result)
        paths = [r["path"] for r in body["routes"]]
        assert paths.count("/api/v1/dup") == 1

    def test_routes_sorted_by_path(self, handler):
        cls_b = self._make_mock_handler_cls(
            "HandlerB", ["/api/v1/zeta"], doc=""
        )
        cls_a = self._make_mock_handler_cls(
            "HandlerA", ["/api/v1/alpha"], doc=""
        )
        mock_rules = []
        with patch(
            "aragora.server.handlers._registry.ALL_HANDLERS",
            [cls_b, cls_a],
        ):
            with patch(
                "aragora.server.openapi_impl.TAG_INFERENCE_RULES",
                mock_rules,
            ):
                result = handler._get_routes({})

        body = _body(result)
        paths = [r["path"] for r in body["routes"]]
        assert paths == sorted(paths)

    def test_import_error_returns_500(self, handler):
        with patch.dict("sys.modules", {"aragora.server.handlers._registry": None}):
            result = handler._get_routes({})

        assert result.status_code == 500
        body = _body(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# GET /api/v1/docs/stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_success(self, handler):
        mock_schema = {
            "paths": {
                "/api/v1/debates": {
                    "get": {"tags": ["Debates"], "summary": "List debates"},
                    "post": {"tags": ["Debates"], "summary": "Create debate"},
                },
                "/api/v1/agents": {
                    "get": {"tags": ["Agents"], "summary": "List agents"},
                },
            }
        }
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_stats()

        assert result.status_code == 200
        body = _body(result)
        assert body["total_endpoints"] == 3
        assert body["total_paths"] == 2
        assert body["by_method"]["GET"] == 2
        assert body["by_method"]["POST"] == 1
        assert isinstance(body["by_tag"], list)

    def test_tags_sorted_descending(self, handler):
        mock_schema = {
            "paths": {
                "/a": {"get": {"tags": ["Alpha"]}},
                "/b": {"get": {"tags": ["Beta"]}, "post": {"tags": ["Beta"]}},
                "/c": {"get": {"tags": ["Gamma"]}, "post": {"tags": ["Gamma"]}, "delete": {"tags": ["Gamma"]}},
            }
        }
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_stats()

        body = _body(result)
        tag_names = [t["tag"] for t in body["by_tag"]]
        tag_counts = [t["count"] for t in body["by_tag"]]
        # Should be sorted by count descending
        assert tag_counts == sorted(tag_counts, reverse=True)
        assert tag_names[0] == "Gamma"

    def test_untagged_endpoints(self, handler):
        mock_schema = {
            "paths": {
                "/api/v1/health": {
                    "get": {"summary": "Health check"},
                },
            }
        }
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_stats()

        body = _body(result)
        tag_names = [t["tag"] for t in body["by_tag"]]
        assert "Untagged" in tag_names

    def test_skips_x_extension_keys(self, handler):
        mock_schema = {
            "paths": {
                "/api/v1/test": {
                    "get": {"tags": ["Test"]},
                    "x-custom-extension": {"foo": "bar"},
                },
            }
        }
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_stats()

        body = _body(result)
        assert body["total_endpoints"] == 1

    def test_skips_parameters_key(self, handler):
        mock_schema = {
            "paths": {
                "/api/v1/test": {
                    "get": {"tags": ["Test"]},
                    "parameters": [{"name": "id", "in": "path"}],
                },
            }
        }
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_stats()

        body = _body(result)
        assert body["total_endpoints"] == 1

    def test_empty_paths(self, handler):
        mock_schema = {"paths": {}}
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_stats()

        body = _body(result)
        assert body["total_endpoints"] == 0
        assert body["total_paths"] == 0
        assert body["by_method"] == {}
        assert body["by_tag"] == []

    def test_import_error_returns_500(self, handler):
        with patch.dict("sys.modules", {"aragora.server.openapi_impl": None}):
            result = handler._get_stats()

        assert result.status_code == 500
        body = _body(result)
        assert "error" in body

    def test_runtime_error_returns_500(self, handler):
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            side_effect=RuntimeError("failed"),
        ):
            result = handler._get_stats()

        assert result.status_code == 500

    def test_multiple_tags_per_endpoint(self, handler):
        mock_schema = {
            "paths": {
                "/api/v1/multi": {
                    "get": {"tags": ["TagA", "TagB"]},
                },
            }
        }
        with patch(
            "aragora.server.openapi_impl.generate_openapi_schema",
            return_value=mock_schema,
        ):
            result = handler._get_stats()

        body = _body(result)
        assert body["total_endpoints"] == 1
        tag_names = [t["tag"] for t in body["by_tag"]]
        assert "TagA" in tag_names
        assert "TagB" in tag_names


# ---------------------------------------------------------------------------
# _infer_methods
# ---------------------------------------------------------------------------


class TestInferMethods:
    def test_no_handle_methods_defaults_to_get(self, handler):
        cls = type("EmptyHandler", (), {})
        methods = handler._infer_methods(cls)
        assert methods == ["GET"]

    def test_handle_get(self, handler):
        cls = type("GetHandler", (), {"handle_get": lambda self: None})
        methods = handler._infer_methods(cls)
        assert "GET" in methods

    def test_handle_post(self, handler):
        cls = type("PostHandler", (), {"handle_post": lambda self: None})
        methods = handler._infer_methods(cls)
        assert methods == ["POST"]

    def test_handle_put(self, handler):
        cls = type("PutHandler", (), {"handle_put": lambda self: None})
        methods = handler._infer_methods(cls)
        assert methods == ["PUT"]

    def test_handle_delete(self, handler):
        cls = type("DeleteHandler", (), {"handle_delete": lambda self: None})
        methods = handler._infer_methods(cls)
        assert methods == ["DELETE"]

    def test_handle_patch(self, handler):
        cls = type("PatchHandler", (), {"handle_patch": lambda self: None})
        methods = handler._infer_methods(cls)
        assert methods == ["PATCH"]

    def test_multiple_methods(self, handler):
        cls = type(
            "MultiHandler",
            (),
            {
                "handle_get": lambda self: None,
                "handle_post": lambda self: None,
                "handle_delete": lambda self: None,
            },
        )
        methods = handler._infer_methods(cls)
        assert set(methods) == {"GET", "POST", "DELETE"}


# ---------------------------------------------------------------------------
# _infer_tag
# ---------------------------------------------------------------------------


class TestInferTag:
    def test_matching_rule(self, handler):
        rules = [("debates", "Debates"), ("agents", "Agents")]
        tag = handler._infer_tag("/api/v1/debates/list", rules)
        assert tag == "Debates"

    def test_no_matching_rule_returns_other(self, handler):
        rules = [("debates", "Debates")]
        tag = handler._infer_tag("/api/v1/agents/list", rules)
        assert tag == "Other"

    def test_strips_v1_prefix(self, handler):
        rules = [("test", "TestTag")]
        tag = handler._infer_tag("/api/v1/test/foo", rules)
        assert tag == "TestTag"

    def test_strips_v2_prefix(self, handler):
        rules = [("test", "TestTag")]
        tag = handler._infer_tag("/api/v2/test/foo", rules)
        assert tag == "TestTag"

    def test_strips_api_prefix(self, handler):
        rules = [("test", "TestTag")]
        tag = handler._infer_tag("/api/test/foo", rules)
        assert tag == "TestTag"

    def test_empty_rules(self, handler):
        tag = handler._infer_tag("/api/v1/anything", [])
        assert tag == "Other"

    def test_exact_prefix_match(self, handler):
        rules = [("docs", "Documentation")]
        tag = handler._infer_tag("/api/v1/docs", rules)
        assert tag == "Documentation"


# ---------------------------------------------------------------------------
# Constructor / context
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_context(self):
        h = ApiDocsHandler({})
        assert h.ctx == {}

    def test_custom_context(self):
        h = ApiDocsHandler({"key": "val"})
        assert h.ctx["key"] == "val"

    def test_routes_count(self):
        assert len(ApiDocsHandler.ROUTES) == 3
