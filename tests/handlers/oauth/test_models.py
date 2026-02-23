"""Tests for aragora/server/handlers/oauth/models.py.

Covers:
- OAuthUserInfo dataclass construction (all fields, defaults, optional)
- OAuthUserInfo equality, repr, field types
- _get_param with scalar values
- _get_param with list values (single-element, multi-element, empty)
- _get_param with missing keys
- _get_param default handling (explicit default, None default)
- _get_param edge cases (empty string, None values, non-string types)
"""

from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from aragora.server.handlers.oauth.models import OAuthUserInfo, _get_param


# ---------------------------------------------------------------------------
# OAuthUserInfo dataclass tests
# ---------------------------------------------------------------------------


class TestOAuthUserInfoConstruction:
    """Tests for OAuthUserInfo instantiation and defaults."""

    def test_create_with_required_fields(self):
        """OAuthUserInfo can be created with only required fields."""
        info = OAuthUserInfo(
            provider="google",
            provider_user_id="12345",
            email="user@example.com",
            name="Test User",
        )
        assert info.provider == "google"
        assert info.provider_user_id == "12345"
        assert info.email == "user@example.com"
        assert info.name == "Test User"

    def test_default_picture_is_none(self):
        """picture defaults to None when not provided."""
        info = OAuthUserInfo(
            provider="github",
            provider_user_id="abc",
            email="u@test.com",
            name="User",
        )
        assert info.picture is None

    def test_default_email_verified_is_false(self):
        """email_verified defaults to False when not provided."""
        info = OAuthUserInfo(
            provider="github",
            provider_user_id="abc",
            email="u@test.com",
            name="User",
        )
        assert info.email_verified is False

    def test_create_with_all_fields(self):
        """OAuthUserInfo accepts all fields including optional ones."""
        info = OAuthUserInfo(
            provider="microsoft",
            provider_user_id="ms-99",
            email="admin@corp.com",
            name="Admin Corp",
            picture="https://example.com/avatar.png",
            email_verified=True,
        )
        assert info.provider == "microsoft"
        assert info.provider_user_id == "ms-99"
        assert info.email == "admin@corp.com"
        assert info.name == "Admin Corp"
        assert info.picture == "https://example.com/avatar.png"
        assert info.email_verified is True

    def test_picture_set_explicitly(self):
        """picture can be set to a URL string."""
        info = OAuthUserInfo(
            provider="google",
            provider_user_id="1",
            email="a@b.com",
            name="A",
            picture="https://img.example.com/photo.jpg",
        )
        assert info.picture == "https://img.example.com/photo.jpg"

    def test_email_verified_set_to_true(self):
        """email_verified can be explicitly set to True."""
        info = OAuthUserInfo(
            provider="google",
            provider_user_id="1",
            email="a@b.com",
            name="A",
            email_verified=True,
        )
        assert info.email_verified is True

    def test_email_verified_set_to_false_explicitly(self):
        """email_verified can be explicitly set to False."""
        info = OAuthUserInfo(
            provider="google",
            provider_user_id="1",
            email="a@b.com",
            name="A",
            email_verified=False,
        )
        assert info.email_verified is False


class TestOAuthUserInfoEquality:
    """Tests for OAuthUserInfo equality and identity."""

    def test_equal_instances(self):
        """Two OAuthUserInfo with identical fields are equal."""
        a = OAuthUserInfo(
            provider="google",
            provider_user_id="x",
            email="e@e.com",
            name="E",
        )
        b = OAuthUserInfo(
            provider="google",
            provider_user_id="x",
            email="e@e.com",
            name="E",
        )
        assert a == b

    def test_unequal_provider(self):
        """Different providers make instances unequal."""
        a = OAuthUserInfo(provider="google", provider_user_id="x", email="e@e.com", name="E")
        b = OAuthUserInfo(provider="github", provider_user_id="x", email="e@e.com", name="E")
        assert a != b

    def test_unequal_email_verified(self):
        """Different email_verified makes instances unequal."""
        a = OAuthUserInfo(
            provider="google",
            provider_user_id="x",
            email="e@e.com",
            name="E",
            email_verified=True,
        )
        b = OAuthUserInfo(
            provider="google",
            provider_user_id="x",
            email="e@e.com",
            name="E",
            email_verified=False,
        )
        assert a != b


class TestOAuthUserInfoDataclass:
    """Tests for dataclass features of OAuthUserInfo."""

    def test_asdict(self):
        """asdict produces the expected dictionary."""
        info = OAuthUserInfo(
            provider="apple",
            provider_user_id="ap-1",
            email="user@icloud.com",
            name="Apple User",
            picture="https://apple.com/pic.png",
            email_verified=True,
        )
        d = asdict(info)
        assert d == {
            "provider": "apple",
            "provider_user_id": "ap-1",
            "email": "user@icloud.com",
            "name": "Apple User",
            "picture": "https://apple.com/pic.png",
            "email_verified": True,
        }

    def test_field_names(self):
        """OAuthUserInfo has exactly 6 fields with correct names."""
        field_names = [f.name for f in fields(OAuthUserInfo)]
        assert field_names == [
            "provider",
            "provider_user_id",
            "email",
            "name",
            "picture",
            "email_verified",
        ]

    def test_repr(self):
        """repr contains class name and field values."""
        info = OAuthUserInfo(
            provider="google",
            provider_user_id="id1",
            email="t@t.com",
            name="T",
        )
        r = repr(info)
        assert "OAuthUserInfo" in r
        assert "google" in r
        assert "id1" in r
        assert "t@t.com" in r

    def test_is_mutable(self):
        """OAuthUserInfo fields can be mutated (not frozen)."""
        info = OAuthUserInfo(
            provider="google",
            provider_user_id="id1",
            email="old@test.com",
            name="Old",
        )
        info.email = "new@test.com"
        info.name = "New"
        info.picture = "https://new.com/pic.png"
        info.email_verified = True
        assert info.email == "new@test.com"
        assert info.name == "New"
        assert info.picture == "https://new.com/pic.png"
        assert info.email_verified is True


# ---------------------------------------------------------------------------
# _get_param tests
# ---------------------------------------------------------------------------


class TestGetParamScalar:
    """Tests for _get_param when the value is a scalar (string)."""

    def test_scalar_string_value(self):
        """Returns the string value directly."""
        result = _get_param({"code": "abc123"}, "code")
        assert result == "abc123"

    def test_scalar_empty_string(self):
        """Returns empty string when parameter value is empty string."""
        result = _get_param({"code": ""}, "code")
        assert result == ""

    def test_scalar_numeric_string(self):
        """Returns numeric string as-is."""
        result = _get_param({"page": "42"}, "page")
        assert result == "42"


class TestGetParamList:
    """Tests for _get_param when the value is a list."""

    def test_single_element_list(self):
        """Returns first element of a single-element list."""
        result = _get_param({"code": ["abc123"]}, "code")
        assert result == "abc123"

    def test_multi_element_list(self):
        """Returns first element of a multi-element list."""
        result = _get_param({"tag": ["first", "second", "third"]}, "tag")
        assert result == "first"

    def test_empty_list_returns_default(self):
        """Returns default when the list is empty."""
        result = _get_param({"code": []}, "code")
        assert result is None

    def test_empty_list_with_explicit_default(self):
        """Returns explicit default when the list is empty."""
        result = _get_param({"code": []}, "code", default="fallback")
        assert result == "fallback"


class TestGetParamMissing:
    """Tests for _get_param when the key is missing."""

    def test_missing_key_returns_none(self):
        """Returns None when key is missing and no default provided."""
        result = _get_param({}, "code")
        assert result is None

    def test_missing_key_returns_explicit_default(self):
        """Returns explicit default when key is missing."""
        result = _get_param({}, "code", default="my_default")
        assert result == "my_default"

    def test_missing_key_with_other_keys(self):
        """Returns None for missing key even when other keys exist."""
        result = _get_param({"other": "value"}, "code")
        assert result is None


class TestGetParamEdgeCases:
    """Edge case tests for _get_param."""

    def test_none_value_in_dict(self):
        """Returns None when the dict value is explicitly None."""
        result = _get_param({"code": None}, "code")
        assert result is None

    def test_none_value_with_default(self):
        """Returns None (the actual value) even when default is provided.

        The dict contains the key with None as value, so .get() returns None,
        not the default.
        """
        result = _get_param({"code": None}, "code", default="fallback")
        assert result is None

    def test_integer_value_passthrough(self):
        """Non-string, non-list values pass through unchanged."""
        result = _get_param({"count": 42}, "count")
        assert result == 42

    def test_empty_dict(self):
        """Returns None for empty dict with no default."""
        result = _get_param({}, "anything")
        assert result is None

    def test_default_parameter_none_explicitly(self):
        """Passing default=None is equivalent to not passing it."""
        result = _get_param({}, "code", default=None)
        assert result is None

    def test_list_with_none_first_element(self):
        """Returns None when list's first element is None."""
        result = _get_param({"code": [None, "other"]}, "code")
        assert result is None

    def test_list_with_empty_string_first(self):
        """Returns empty string when first list element is empty string."""
        result = _get_param({"code": ["", "nonempty"]}, "code")
        assert result == ""

    def test_bool_value_passthrough(self):
        """Boolean values pass through unchanged (not a list)."""
        result = _get_param({"flag": True}, "flag")
        assert result is True

    def test_multiple_params_extracts_correct_one(self):
        """Extracts the correct parameter from a dict with multiple keys."""
        params = {"code": "auth_code", "state": "state_val", "scope": "openid"}
        assert _get_param(params, "code") == "auth_code"
        assert _get_param(params, "state") == "state_val"
        assert _get_param(params, "scope") == "openid"
