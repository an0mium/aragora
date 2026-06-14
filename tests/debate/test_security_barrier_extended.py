"""
Extended security tests for SecurityBarrier.

Tests cover:
1. Bypass attempts and evasion patterns
2. Edge cases and boundary conditions
3. Performance with large content
4. Pattern caching behavior
5. Google API key detection
"""

import pytest
import re
from unittest.mock import Mock

from aragora.debate.security_barrier import SecurityBarrier, TelemetryVerifier


class TestSecurityBarrierBypassAttempts:
    """Test resistance to redaction bypass attempts."""

    def test_case_variations(self):
        """Should detect secrets regardless of case."""
        barrier = SecurityBarrier()

        test_cases = [
            "API_KEY = secret123",
            "api_key = secret123",
            "Api_Key = secret123",
            "API_key = secret123",
            "ApI_KeY = secret123",
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed to redact: {content}"

    def test_whitespace_variations(self):
        """Should detect secrets with various whitespace patterns."""
        barrier = SecurityBarrier()

        test_cases = [
            "api_key=secret123",
            "api_key = secret123",
            "api_key  =  secret123",
            "api_key\t=\tsecret123",
            "api_key:secret123",
            "api_key: secret123",
            "api_key : secret123",
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed to redact: {content}"

    def test_multiple_secrets_same_line(self):
        """Should detect multiple secrets on the same line."""
        barrier = SecurityBarrier()

        content = "api_key=secret1 token=secret2 password=secret3"
        result = barrier.redact(content)

        # Count redactions
        assert result.count("[REDACTED]") >= 3

    def test_secrets_in_json_format(self):
        """Should detect secrets in JSON-like format."""
        barrier = SecurityBarrier()

        test_cases = [
            '{"api_key": "secret123"}',
            "{'token': 'secret123'}",
            '"password": "secret123"',
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed to redact: {content}"

    def test_openai_key_variations(self):
        """Should detect various OpenAI key formats."""
        barrier = SecurityBarrier()

        test_cases = [
            "sk-TESTKEY123456789012345678901234",  # Old format
            "sk-proj-abc123def456ghi789jkl012mno345",  # Project key
            "sk-ant-TESTKEY12345678901234567890",  # Anthropic-style
            "sk-svc-abc123def456ghi789jkl012mno345",  # Service key
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed to redact: {content}"

    def test_google_api_key_detection(self):
        """Should detect Google API key format."""
        barrier = SecurityBarrier()

        # Google API keys start with AIza followed by 35 chars
        test_keys = [
            "AIzaTESTKEYEXAMPLE1234567890123456789AB",  # Example format
            "AIzaTESTKEYEXAMPLE1234567890123456789ABwx",
        ]

        for key in test_keys:
            result = barrier.redact(f"Using key: {key}")
            assert "[REDACTED]" in result, f"Failed to redact Google API key: {key}"

    def test_bearer_token_variations(self):
        """Should detect various bearer token formats."""
        barrier = SecurityBarrier()

        test_cases = [
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "BEARER eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "Bearer abc123-def456_ghi789",
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed to redact: {content}"

    def test_url_credential_variations(self):
        """Should detect URL credentials in http/https formats."""
        barrier = SecurityBarrier()

        # Only http/https URLs are covered by default patterns
        test_cases = [
            "https://user:pass@example.com",
            "http://admin:secret123@localhost:8080",
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed to redact: {content}"

        # Non-http URLs require custom patterns
        barrier.add_pattern(r"\w+://[^:]+:[^@]+@")
        assert "[REDACTED]" in barrier.redact("postgres://dbuser:dbpass@host:5432/db")
        assert "[REDACTED]" in barrier.redact("redis://default:mypassword@redis.example.com:6379")


class TestSecurityBarrierEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self):
        """Should handle empty string."""
        barrier = SecurityBarrier()
        assert barrier.redact("") == ""

    def test_none_value(self):
        """Should handle None gracefully."""
        barrier = SecurityBarrier()
        assert barrier.redact(None) is None

    def test_whitespace_only(self):
        """Should preserve whitespace-only content."""
        barrier = SecurityBarrier()
        assert barrier.redact("   \n\t  ") == "   \n\t  "

    def test_unicode_content(self):
        """Should handle unicode characters."""
        barrier = SecurityBarrier()

        content = "api_key = secret123 with unicode: \u00e9\u00e8\u00ea"
        result = barrier.redact(content)
        assert "[REDACTED]" in result
        # Unicode should be preserved
        assert "\u00e9" in result or "[REDACTED]" in result

    def test_very_long_secret(self):
        """Should handle very long secrets."""
        barrier = SecurityBarrier()

        long_secret = "sk-" + "a" * 1000
        result = barrier.redact(f"key: {long_secret}")
        assert "[REDACTED]" in result

    def test_secret_at_boundaries(self):
        """Should detect secrets at string boundaries."""
        barrier = SecurityBarrier()

        # At start
        result1 = barrier.redact("sk-abcdefghijklmnop is the key")
        assert "[REDACTED]" in result1

        # At end
        result2 = barrier.redact("The key is sk-abcdefghijklmnop")
        assert "[REDACTED]" in result2

        # Entire string
        result3 = barrier.redact("sk-abcdefghijklmnop")
        assert "[REDACTED]" in result3

    def test_nested_dict_with_none_values(self):
        """Should handle dicts with None values."""
        barrier = SecurityBarrier()

        data = {
            "key": None,
            "nested": {"value": None},
            "list": [None, "api_key=secret"],
        }

        result = barrier.redact_dict(data)
        assert result["key"] is None
        assert result["nested"]["value"] is None
        assert result["list"][0] is None
        assert "[REDACTED]" in result["list"][1]

    def test_deeply_nested_dict(self):
        """Should handle deeply nested dictionaries."""
        barrier = SecurityBarrier()

        data = {"level1": {"level2": {"level3": {"level4": {"secret": "api_key=deep_secret"}}}}}

        result = barrier.redact_dict(data)
        assert "[REDACTED]" in result["level1"]["level2"]["level3"]["level4"]["secret"]

    def test_list_of_dicts(self):
        """Should handle list of dictionaries."""
        barrier = SecurityBarrier()

        data = {
            "items": [
                {"secret": "api_key=secret1"},
                {"secret": "token=secret2"},
                {"safe": "no secrets here"},
            ]
        }

        result = barrier.redact_dict(data)
        assert "[REDACTED]" in result["items"][0]["secret"]
        assert "[REDACTED]" in result["items"][1]["secret"]
        assert result["items"][2]["safe"] == "no secrets here"

    def test_mixed_types_in_list(self):
        """Should handle lists with mixed types."""
        barrier = SecurityBarrier()

        data = {
            "mixed": [
                "api_key=secret",
                42,
                3.14,
                True,
                None,
                {"nested": "token=nested_secret"},
            ]
        }

        result = barrier.redact_dict(data)
        assert "[REDACTED]" in result["mixed"][0]
        assert result["mixed"][1] == 42
        assert result["mixed"][2] == 3.14
        assert result["mixed"][3] is True
        assert result["mixed"][4] is None
        assert "[REDACTED]" in result["mixed"][5]["nested"]


class TestSecurityBarrierPatternCaching:
    """Test pattern caching behavior."""

    def test_custom_pattern_invalidates_cache(self):
        """Adding custom pattern should invalidate cache."""
        barrier = SecurityBarrier()

        # First call builds cache
        barrier.redact("test")
        assert barrier._all_patterns_cache is not None

        # Adding pattern should invalidate
        barrier.add_pattern(r"CUSTOM_\d+")
        assert barrier._all_patterns_cache is None

        # Next call rebuilds cache
        barrier.redact("test")
        assert barrier._all_patterns_cache is not None

    def test_refresh_patterns_invalidates_cache(self):
        """refresh_patterns should invalidate cache."""
        barrier = SecurityBarrier()

        barrier.redact("test")
        assert barrier._all_patterns_cache is not None

        barrier.refresh_patterns()
        assert barrier._all_patterns_cache is None

    def test_cache_includes_custom_patterns(self):
        """Cache should include custom patterns."""
        barrier = SecurityBarrier()
        barrier.add_pattern(r"MY_SECRET_\d+")

        # Build cache
        barrier.redact("test")

        # Custom pattern should work after cache is built
        result = barrier.redact("Using MY_SECRET_12345 here")
        assert "[REDACTED]" in result


class TestSecurityBarrierPerformance:
    """Test performance with large content."""

    def test_large_content_without_secrets(self):
        """Should handle large content efficiently."""
        barrier = SecurityBarrier()

        # 1MB of safe content
        large_content = "This is safe content without any secrets. " * 25000

        result = barrier.redact(large_content)
        assert result == large_content  # No redaction needed

    def test_large_content_with_secrets(self):
        """Should find secrets in large content."""
        barrier = SecurityBarrier()

        # Build large content with a secret in the middle
        prefix = "Safe content. " * 10000
        secret = "api_key = super_secret_value"
        suffix = " More safe content." * 10000

        result = barrier.redact(prefix + secret + suffix)
        assert "[REDACTED]" in result

    def test_many_secrets(self):
        """Should handle content with many secrets."""
        barrier = SecurityBarrier()

        # Create content with 100 secrets in formats that match default patterns
        secrets = [f"api_key = sk-secret{i}abcdefghij" for i in range(100)]
        content = "\n".join(secrets)

        result = barrier.redact(content)
        # Should redact all of them (the sk- pattern matches)
        assert result.count("[REDACTED]") >= 100


class TestSecurityBarrierContainsSensitive:
    """Test contains_sensitive method."""

    def test_returns_false_for_safe_content(self):
        """Should return False for safe content."""
        barrier = SecurityBarrier()

        assert not barrier.contains_sensitive("This is normal text")
        assert not barrier.contains_sensitive("API design patterns")
        assert not barrier.contains_sensitive("The password policy requires...")

    def test_returns_true_for_sensitive_content(self):
        """Should return True for sensitive content."""
        barrier = SecurityBarrier()

        assert barrier.contains_sensitive("api_key = secret")
        assert barrier.contains_sensitive("sk-abc123def456ghi789")
        assert barrier.contains_sensitive("Bearer token123")

    def test_includes_custom_patterns(self):
        """Should check custom patterns."""
        barrier = SecurityBarrier()
        barrier.add_pattern(r"CUSTOM_SECRET_\w+")

        assert not barrier.contains_sensitive("CUSTOM_SECRET")  # No match
        assert barrier.contains_sensitive("CUSTOM_SECRET_abc123")  # Match

    def test_empty_and_none(self):
        """Should handle empty and None values."""
        barrier = SecurityBarrier()

        assert not barrier.contains_sensitive("")
        assert not barrier.contains_sensitive(None)


class TestTelemetryVerifierEdgeCases:
    """Test edge cases for TelemetryVerifier."""

    def test_unknown_telemetry_level(self):
        """Should handle unknown telemetry level."""
        verifier = TelemetryVerifier()

        agent = Mock()
        agent.name = "test-agent"
        agent.generate = Mock()

        # Unknown level should return empty requirements
        result = verifier.verify_telemetry_level("unknown_level", agent)
        assert result is True  # No requirements to fail

    def test_empty_required_capabilities(self):
        """Should pass with empty required capabilities."""
        verifier = TelemetryVerifier()

        agent = Mock()
        agent.name = "minimal-agent"

        passed, missing = verifier.verify_agent(agent, [])
        assert passed is True
        assert missing == []

    def test_agent_with_none_attribute(self):
        """Should detect None attribute values as missing."""
        verifier = TelemetryVerifier()

        agent = Mock()
        agent.name = "test-agent"
        agent.generate = None  # Attribute exists but is None

        passed, missing = verifier.verify_agent(agent, ["name", "generate"])
        assert not passed
        assert "generate" in missing

    def test_verification_for_all_levels(self):
        """Should verify all defined telemetry levels."""
        verifier = TelemetryVerifier()

        # Full capability agent
        agent = Mock()
        agent.name = "full-agent"
        agent.generate = Mock()
        agent.model = "test-model"

        for level in TelemetryVerifier.CAPABILITY_REQUIREMENTS.keys():
            result = verifier.verify_telemetry_level(level, agent)
            assert result is True, f"Failed for level: {level}"

    def test_verification_report_empty(self):
        """Should handle empty verification history."""
        verifier = TelemetryVerifier()

        report = verifier.get_verification_report()
        assert report["total"] == 0
        assert report["passed"] == 0
        assert report["failed"] == 0
        assert report["agents"] == []

    def test_capability_cache_content(self):
        """Should properly cache agent capabilities."""
        verifier = TelemetryVerifier()

        agent = Mock()
        agent.name = "cached-agent"
        agent.generate = Mock()
        agent.model = "test"

        verifier.verify_agent(agent, ["generate", "model"])

        # Check cache content
        assert "cached-agent" in verifier._capability_cache
        assert "generate" in verifier._capability_cache["cached-agent"]
        assert "model" in verifier._capability_cache["cached-agent"]

    def test_agent_without_name_attribute(self):
        """Should handle agent without name attribute."""
        verifier = TelemetryVerifier()

        # Object without name attribute
        agent = object()

        passed, missing = verifier.verify_agent(agent, ["name"])
        assert not passed
        assert "name" in missing


class TestPrivateKeyDetection:
    """Test detection of private key patterns."""

    def test_rsa_private_key(self):
        """Should detect RSA private key header."""
        barrier = SecurityBarrier()

        content = """-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEA0Z3...
-----END RSA PRIVATE KEY-----"""

        result = barrier.redact(content)
        assert "[REDACTED]" in result

    def test_generic_private_key(self):
        """Should detect generic private key header."""
        barrier = SecurityBarrier()

        content = """-----BEGIN PRIVATE KEY-----
<REDACTED_PRIVATE_KEY>
-----END PRIVATE KEY-----"""

        result = barrier.redact(content)
        assert "[REDACTED]" in result

    def test_ec_private_key(self):
        """Should detect EC private key header."""
        barrier = SecurityBarrier()
        barrier.add_pattern(r"-----BEGIN\s+EC\s+PRIVATE\s+KEY-----")

        content = """-----BEGIN EC PRIVATE KEY-----
MHQCAQEEIBk...
-----END EC PRIVATE KEY-----"""

        result = barrier.redact(content)
        assert "[REDACTED]" in result


class TestEnvironmentVariablePatterns:
    """Test detection of environment variable patterns."""

    def test_anthropic_api_key(self):
        """Should detect ANTHROPIC_API_KEY."""
        barrier = SecurityBarrier()

        test_cases = [
            "ANTHROPIC_API_KEY = sk-ant-abc123",
            "ANTHROPIC_KEY=test123",
            "ANTHROPIC API_KEY: abc",
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed: {content}"

    def test_openai_api_key(self):
        """Should detect OPENAI_API_KEY."""
        barrier = SecurityBarrier()

        test_cases = [
            "OPENAI_API_KEY = sk-abc123",
            "OPENAI_KEY=test123",
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed: {content}"

    def test_gemini_api_key(self):
        """Should detect GEMINI_API_KEY."""
        barrier = SecurityBarrier()

        result = barrier.redact("GEMINI_API_KEY = AIzaSy123456")
        assert "[REDACTED]" in result

    def test_xai_and_grok_keys(self):
        """Should detect XAI and GROK keys."""
        barrier = SecurityBarrier()

        test_cases = [
            "XAI_API_KEY = xai-abc123",
            "GROK_API_KEY = grok-abc123",
        ]

        for content in test_cases:
            result = barrier.redact(content)
            assert "[REDACTED]" in result, f"Failed: {content}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
