"""Tests for the shared SDK path normalization module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from sdk_path_normalize import normalize_sdk_path


class TestVersionPrefixStripping:
    def test_strips_v1_prefix(self):
        assert normalize_sdk_path("/api/v1/foo") == "/api/foo"

    def test_strips_v2_prefix(self):
        assert normalize_sdk_path("/api/v2/bar") == "/api/bar"

    def test_strips_v10_prefix(self):
        assert normalize_sdk_path("/api/v10/baz") == "/api/baz"

    def test_preserves_non_versioned_api_path(self):
        assert normalize_sdk_path("/api/foo") == "/api/foo"

    def test_preserves_non_api_path(self):
        assert normalize_sdk_path("/health") == "/health"


class TestParamNormalization:
    def test_express_colon_param(self):
        assert normalize_sdk_path("/api/users/:user_id") == "/api/users/{param}"

    def test_named_brace_param(self):
        assert normalize_sdk_path("/api/users/{user_id}") == "/api/users/{param}"

    def test_template_literal_param(self):
        assert normalize_sdk_path("/api/users/${userId}") == "/api/users/{param}"

    def test_wildcard_param(self):
        assert normalize_sdk_path("/api/users/*") == "/api/users/{param}"

    def test_multiple_params(self):
        result = normalize_sdk_path("/api/v1/debates/:debate_id/rounds/:round_id")
        assert result == "/api/debates/{param}/rounds/{param}"

    def test_mixed_param_styles(self):
        result = normalize_sdk_path("/api/v1/items/:id/sub/{sub_id}")
        assert result == "/api/items/{param}/sub/{param}"


class TestTrailingSlash:
    def test_strips_trailing_slash(self):
        assert normalize_sdk_path("/api/v1/foo/") == "/api/foo"

    def test_preserves_root(self):
        assert normalize_sdk_path("/") == "/"

    def test_no_trailing_slash_unchanged(self):
        assert normalize_sdk_path("/api/foo") == "/api/foo"


class TestQueryString:
    def test_strips_query_string(self):
        assert normalize_sdk_path("/api/v1/foo?bar=baz") == "/api/foo"


class TestLowercase:
    def test_lowercases_path(self):
        assert normalize_sdk_path("/API/V1/Foo") == "/api/foo"


class TestEdgeCases:
    def test_empty_string(self):
        assert normalize_sdk_path("") == ""

    def test_bare_api(self):
        assert normalize_sdk_path("/api") == "/api"

    def test_bare_api_v1(self):
        assert normalize_sdk_path("/api/v1/") == "/api"

    def test_double_param(self):
        result = normalize_sdk_path("/api/v1/a/{x}/b/{y}")
        assert result == "/api/a/{param}/b/{param}"

    def test_idempotent(self):
        path = "/api/v1/debates/:id/rounds/:round_id/"
        once = normalize_sdk_path(path)
        twice = normalize_sdk_path(once)
        assert once == twice
