"""
Tests for XSS protection middleware.

Tests cover:
- HTML escaping functions
- SafeHTMLBuilder
- Cookie security enforcement
- CSP nonce generation
- Integration with response utils
"""

from __future__ import annotations

import pytest


class TestHTMLEscaping:
    """Tests for HTML escaping functions."""

    def test_escape_html_script_tag(self):
        """Should escape script tags."""
        from aragora.server.middleware.xss_protection import escape_html

        result = escape_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escape_html_ampersand(self):
        """Should escape ampersands."""
        from aragora.server.middleware.xss_protection import escape_html

        assert escape_html("Tom & Jerry") == "Tom &amp; Jerry"

    def test_escape_html_less_than_greater_than(self):
        """Should escape angle brackets."""
        from aragora.server.middleware.xss_protection import escape_html

        result = escape_html("<div>content</div>")
        assert "<div>" not in result
        assert "&lt;div&gt;" in result
        assert "&lt;/div&gt;" in result

    def test_escape_html_quotes(self):
        """Should escape quotes."""
        from aragora.server.middleware.xss_protection import escape_html

        result = escape_html('He said "hello"')
        # markupsafe escapes double quotes
        assert '"' not in result or "&quot;" in result or "&#34;" in result

    def test_escape_html_single_quotes(self):
        """Should escape single quotes."""
        from aragora.server.middleware.xss_protection import escape_html

        result = escape_html("It's a test")
        assert "'" not in result or "&#39;" in result or "&#x27;" in result

    def test_escape_html_none(self):
        """Should handle None input."""
        from aragora.server.middleware.xss_protection import escape_html

        assert escape_html(None) == ""

    def test_escape_html_number(self):
        """Should convert numbers to string."""
        from aragora.server.middleware.xss_protection import escape_html

        assert escape_html(123) == "123"
        assert escape_html(3.14) == "3.14"

    def test_escape_html_boolean(self):
        """Should convert booleans to string."""
        from aragora.server.middleware.xss_protection import escape_html

        assert escape_html(True) == "True"
        assert escape_html(False) == "False"

    def test_escape_html_unicode(self):
        """Should handle unicode strings."""
        from aragora.server.middleware.xss_protection import escape_html

        result = escape_html("Hello 世界 🌍")
        assert "Hello" in result
        assert "世界" in result

    def test_escape_html_attribute_backtick(self):
        """Should escape backticks in attribute context."""
        from aragora.server.middleware.xss_protection import escape_html_attribute

        result = escape_html_attribute("value`with`backticks")
        assert "`" not in result
        assert "&#x60;" in result

    def test_escape_html_attribute_single_quote(self):
        """Should escape single quotes in attribute context."""
        from aragora.server.middleware.xss_protection import escape_html_attribute

        result = escape_html_attribute("it's a value")
        assert "'" not in result
        # markupsafe escapes to &#39; which we then convert to &#x27;
        assert "&#x27;" in result or "&#39;" in result

    def test_mark_safe(self):
        """Should mark content as safe without escaping."""
        from aragora.server.middleware.xss_protection import escape_html, mark_safe

        safe_html = mark_safe("<b>bold</b>")
        # When passed through escape again, should not double-escape
        result = escape_html(safe_html)
        assert "<b>bold</b>" in result


class TestSafeHTMLBuilder:
    """Tests for SafeHTMLBuilder class."""

    def test_add_text_escapes(self):
        """Should escape text content."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_text("<script>evil</script>")
        result = builder.build()

        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_add_raw_does_not_escape(self):
        """Should not escape raw HTML."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_raw("<div class='trusted'>")
        result = builder.build()

        assert "<div class='trusted'>" in result

    def test_add_element_with_content(self):
        """Should create elements with escaped content."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_element("div", "Hello <world>", class_="test")
        result = builder.build()

        assert 'class="test"' in result
        assert "&lt;world&gt;" in result
        assert "<div" in result
        assert "</div>" in result

    def test_add_element_escapes_attributes(self):
        """Should escape attribute values."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_element("a", "link", href='javascript:alert("xss")')
        result = builder.build()

        # The javascript URL should be escaped
        assert 'javascript:alert("xss")' not in result

    def test_add_element_self_closing(self):
        """Should create self-closing tags when content is None."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_element("br")
        builder.add_element("img", src="image.png", alt="test")
        result = builder.build()

        assert "<br />" in result
        assert "<img " in result
        assert "/>" in result

    def test_add_element_underscore_to_hyphen(self):
        """Should convert underscores to hyphens in attribute names."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_element("div", "content", data_id="123", aria_label="test")
        result = builder.build()

        assert 'data-id="123"' in result
        assert 'aria-label="test"' in result

    def test_add_element_skips_none_attributes(self):
        """Should skip attributes with None values."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_element("div", "content", class_="test", id=None)
        result = builder.build()

        assert 'class="test"' in result
        assert "id=" not in result

    def test_rejects_invalid_tag_name(self):
        """Should reject invalid tag names."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        with pytest.raises(ValueError, match="Invalid tag name"):
            builder.add_element("<script>", "evil")

    def test_rejects_invalid_tag_name_with_space(self):
        """Should reject tag names with spaces."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        with pytest.raises(ValueError, match="Invalid tag name"):
            builder.add_element("div onclick", "content")

    def test_rejects_invalid_attribute_name(self):
        """Should reject invalid attribute names."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        # Attribute name with special characters
        with pytest.raises(ValueError, match="Invalid attribute name"):
            builder.add_element("div", "content", **{"on<click": "evil()"})

    def test_chaining(self):
        """Should support method chaining."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        result = (
            SafeHTMLBuilder()
            .add_raw("<!DOCTYPE html>")
            .add_raw("<html>")
            .add_element("head")
            .add_raw("<body>")
            .add_element("h1", "Title")
            .add_text("Some text")
            .add_raw("</body></html>")
            .build()
        )

        assert "<!DOCTYPE html>" in result
        assert "<h1>Title</h1>" in result
        assert "Some text" in result

    def test_complex_page(self):
        """Should build complex HTML pages correctly."""
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_raw("<!DOCTYPE html><html><head>")
        builder.add_element("title", "Test Page")
        builder.add_raw("</head><body>")
        builder.add_element("h1", "Welcome, <User>!", class_="header")
        builder.add_element("p", "This is a test & demo.", id="intro")
        builder.add_raw("</body></html>")
        result = builder.build()

        assert "<title>Test Page</title>" in result
        assert "&lt;User&gt;" in result
        assert "test &amp; demo" in result


class TestCookieSecurity:
    """Tests for cookie security functions."""

    def test_build_secure_cookie_defaults(self):
        """Should include security flags by default."""
        from aragora.server.middleware.xss_protection import build_secure_cookie

        cookie = build_secure_cookie("session", "abc123")

        assert "session=abc123" in cookie
        assert "HttpOnly" in cookie
        assert "Secure" in cookie
        assert "SameSite=" in cookie
        assert "Path=/" in cookie

    def test_build_secure_cookie_with_max_age(self):
        """Should include max-age when specified."""
        from aragora.server.middleware.xss_protection import build_secure_cookie

        cookie = build_secure_cookie("session", "abc123", max_age=3600)

        assert "Max-Age=3600" in cookie

    def test_build_secure_cookie_with_domain(self):
        """Should include domain when specified."""
        from aragora.server.middleware.xss_protection import build_secure_cookie

        cookie = build_secure_cookie("session", "abc123", domain=".example.com")

        assert "Domain=.example.com" in cookie

    def test_build_secure_cookie_custom_path(self):
        """Should support custom path."""
        from aragora.server.middleware.xss_protection import build_secure_cookie

        cookie = build_secure_cookie("session", "abc123", path="/api")

        assert "Path=/api" in cookie

    def test_build_secure_cookie_samesite_strict(self):
        """Should support SameSite=Strict."""
        from aragora.server.middleware.xss_protection import (
            XSSProtectionConfig,
            build_secure_cookie,
        )

        config = XSSProtectionConfig()
        config.cookie_samesite = "Strict"
        cookie = build_secure_cookie("session", "abc123", config=config)

        assert "SameSite=Strict" in cookie

    def test_build_secure_cookie_samesite_none(self):
        """Should support SameSite=None."""
        from aragora.server.middleware.xss_protection import (
            XSSProtectionConfig,
            build_secure_cookie,
        )

        config = XSSProtectionConfig()
        config.cookie_samesite = "None"
        cookie = build_secure_cookie("session", "abc123", config=config)

        assert "SameSite=None" in cookie

    def test_build_secure_cookie_no_httponly(self):
        """Should allow disabling HttpOnly."""
        from aragora.server.middleware.xss_protection import (
            XSSProtectionConfig,
            build_secure_cookie,
        )

        config = XSSProtectionConfig()
        config.cookie_httponly = False
        cookie = build_secure_cookie("session", "abc123", config=config)

        assert "HttpOnly" not in cookie

    def test_build_secure_cookie_no_secure(self):
        """Should allow disabling Secure flag."""
        from aragora.server.middleware.xss_protection import (
            XSSProtectionConfig,
            build_secure_cookie,
        )

        config = XSSProtectionConfig()
        config.cookie_secure = False
        cookie = build_secure_cookie("session", "abc123", config=config)

        # Check Secure is not in the cookie parts
        parts = cookie.split("; ")
        assert "Secure" not in parts


class TestCSPNonce:
    """Tests for CSP nonce generation."""

    def test_nonce_is_generated(self):
        """Should generate a nonce."""
        from aragora.server.middleware.xss_protection import CSPNonceContext

        ctx = CSPNonceContext()
        nonce = ctx.nonce

        assert nonce is not None
        assert len(nonce) >= 20  # Base64 of 16 bytes

    def test_nonce_is_cached_per_context(self):
        """Should return same nonce within same context."""
        from aragora.server.middleware.xss_protection import CSPNonceContext

        ctx = CSPNonceContext()
        nonce1 = ctx.nonce
        nonce2 = ctx.nonce

        assert nonce1 == nonce2

    def test_script_tag_includes_nonce(self):
        """Should generate script tag with nonce."""
        from aragora.server.middleware.xss_protection import CSPNonceContext

        ctx = CSPNonceContext()
        script = ctx.script_tag("console.log('hello')")

        assert f'nonce="{ctx.nonce}"' in script
        assert "<script" in script
        assert "</script>" in script

    def test_script_tag_escapes_content(self):
        """Should escape script content."""
        from aragora.server.middleware.xss_protection import CSPNonceContext

        ctx = CSPNonceContext()
        script = ctx.script_tag("</script><script>evil()")

        assert "</script><script>" not in script

    def test_different_contexts_have_different_nonces(self):
        """Different contexts should have different nonces."""
        from aragora.server.middleware.xss_protection import CSPNonceContext

        ctx1 = CSPNonceContext()
        ctx2 = CSPNonceContext()

        assert ctx1.nonce != ctx2.nonce

    def test_inline_script_attr(self):
        """Should generate nonce attribute string."""
        from aragora.server.middleware.xss_protection import CSPNonceContext

        ctx = CSPNonceContext()
        attr = ctx.inline_script_attr()

        assert attr == f'nonce="{ctx.nonce}"'

    def test_get_request_nonce(self):
        """Should get consistent nonce for current request."""
        from aragora.server.middleware.xss_protection import (
            clear_request_nonce,
            get_request_nonce,
        )

        try:
            nonce1 = get_request_nonce()
            nonce2 = get_request_nonce()

            assert nonce1 == nonce2
        finally:
            clear_request_nonce()

    def test_clear_request_nonce(self):
        """Should clear nonce and generate new one after clear."""
        from aragora.server.middleware.xss_protection import (
            clear_request_nonce,
            get_request_nonce,
        )

        nonce1 = get_request_nonce()
        clear_request_nonce()
        nonce2 = get_request_nonce()
        clear_request_nonce()

        assert nonce1 != nonce2

    @pytest.mark.asyncio
    async def test_request_nonce_context(self):
        """Should auto-clear nonce after context exits."""
        from aragora.server.middleware.xss_protection import (
            get_request_nonce,
            request_nonce_context,
        )

        async with request_nonce_context() as nonce1:
            # Should get same nonce within context
            assert get_request_nonce() == nonce1

        # After context, new nonce should be generated
        async with request_nonce_context() as nonce2:
            assert nonce1 != nonce2


class TestXSSProtectionConfig:
    """Tests for XSSProtectionConfig."""

    def test_default_values(self):
        """Should have secure defaults."""
        from aragora.server.middleware.xss_protection import XSSProtectionConfig

        config = XSSProtectionConfig()

        assert config.auto_escape_html is True
        assert config.enforce_cookie_security is True
        assert config.cookie_httponly is True
        assert config.cookie_secure is True
        assert config.enable_csp_by_default is True
        assert config.cookie_samesite == "Lax"

    def test_config_from_env_samesite(self, monkeypatch):
        """Should read SameSite from environment."""
        monkeypatch.setenv("ARAGORA_COOKIE_SAMESITE", "Strict")

        from aragora.server.middleware.xss_protection import XSSProtectionConfig

        config = XSSProtectionConfig()
        assert config.cookie_samesite == "Strict"

    def test_config_from_env_disable_secure(self, monkeypatch):
        """Should read Secure flag from environment."""
        monkeypatch.setenv("ARAGORA_COOKIE_SECURE", "false")

        from aragora.server.middleware.xss_protection import XSSProtectionConfig

        config = XSSProtectionConfig()
        assert config.cookie_secure is False

    def test_config_from_env_csp_report_uri(self, monkeypatch):
        """Should read CSP report URI from environment."""
        monkeypatch.setenv("ARAGORA_CSP_REPORT_URI", "/custom/csp-report")

        from aragora.server.middleware.xss_protection import XSSProtectionConfig

        config = XSSProtectionConfig()
        assert config.csp_report_uri == "/custom/csp-report"
