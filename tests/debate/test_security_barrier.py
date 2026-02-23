"""Tests for aragora.debate.security_barrier — SecurityBarrier and TelemetryVerifier."""

from __future__ import annotations

import pytest

from aragora.debate.security_barrier import SecurityBarrier, TelemetryVerifier


# ---------------------------------------------------------------------------
# SecurityBarrier — init
# ---------------------------------------------------------------------------


class TestSecurityBarrierInit:
    def test_default_patterns(self):
        sb = SecurityBarrier()
        assert len(sb._patterns) == len(SecurityBarrier.DEFAULT_PATTERNS)

    def test_custom_patterns(self):
        sb = SecurityBarrier(patterns=[r"secret_\d+"])
        assert len(sb._patterns) == 1

    def test_custom_redaction_marker(self):
        sb = SecurityBarrier(redaction_marker="***")
        assert sb._redaction_marker == "***"


# ---------------------------------------------------------------------------
# SecurityBarrier — redact
# ---------------------------------------------------------------------------


class TestSecurityBarrierRedact:
    def test_empty_content(self):
        sb = SecurityBarrier()
        assert sb.redact("") == ""
        assert sb.redact(None) is None

    def test_no_sensitive_content(self):
        sb = SecurityBarrier()
        assert sb.redact("Hello world, this is safe text") == "Hello world, this is safe text"

    def test_redacts_api_key(self):
        sb = SecurityBarrier()
        result = sb.redact("Use api_key = my-secret-key-12345")
        assert "[REDACTED]" in result
        assert "my-secret-key" not in result

    def test_redacts_bearer_token(self):
        sb = SecurityBarrier()
        result = sb.redact("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        assert "[REDACTED]" in result

    def test_redacts_openai_key(self):
        sb = SecurityBarrier()
        result = sb.redact("key is sk-proj-abc123def456ghi789jkl0123456789")
        assert "[REDACTED]" in result
        assert "sk-proj" not in result

    def test_redacts_google_api_key(self):
        sb = SecurityBarrier()
        result = sb.redact("google key: AIzaSyC1234567890abcdefghijklmnopqrstuv")
        assert "[REDACTED]" in result

    def test_redacts_env_variable(self):
        sb = SecurityBarrier()
        result = sb.redact("ANTHROPIC_API_KEY = sk-ant-abc123")
        assert "[REDACTED]" in result

    def test_redacts_url_with_credentials(self):
        sb = SecurityBarrier()
        result = sb.redact("connect to https://user:password@example.com/db")
        assert "[REDACTED]" in result

    def test_redacts_private_key(self):
        sb = SecurityBarrier()
        result = sb.redact("-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBg...")
        assert "[REDACTED]" in result

    def test_redacts_rsa_private_key(self):
        sb = SecurityBarrier()
        result = sb.redact("-----BEGIN RSA PRIVATE KEY-----\nMIIBog...")
        assert "[REDACTED]" in result

    def test_custom_marker(self):
        sb = SecurityBarrier(redaction_marker="<CENSORED>")
        result = sb.redact("token = abc123def456")
        assert "<CENSORED>" in result

    def test_multiple_sensitive_items(self):
        sb = SecurityBarrier()
        text = "key: sk-proj-abc123def456ghi789jkl and OPENAI_API_KEY = xyz"
        result = sb.redact(text)
        assert result.count("[REDACTED]") >= 2


# ---------------------------------------------------------------------------
# SecurityBarrier — add_pattern / refresh
# ---------------------------------------------------------------------------


class TestSecurityBarrierCustomPatterns:
    def test_add_pattern(self):
        sb = SecurityBarrier()
        sb.add_pattern(r"CUSTOM_SECRET_\w+")
        result = sb.redact("Found CUSTOM_SECRET_abc123 in logs")
        assert "[REDACTED]" in result

    def test_add_pattern_invalidates_cache(self):
        sb = SecurityBarrier()
        # Access cache to populate it
        sb._get_all_patterns()
        assert sb._all_patterns_cache is not None
        sb.add_pattern(r"new_pattern")
        assert sb._all_patterns_cache is None

    def test_refresh_patterns_invalidates_cache(self):
        sb = SecurityBarrier()
        sb._get_all_patterns()
        assert sb._all_patterns_cache is not None
        sb.refresh_patterns()
        assert sb._all_patterns_cache is None

    def test_get_all_patterns_caches(self):
        sb = SecurityBarrier()
        p1 = sb._get_all_patterns()
        p2 = sb._get_all_patterns()
        assert p1 is p2  # Same object from cache


# ---------------------------------------------------------------------------
# SecurityBarrier — redact_dict
# ---------------------------------------------------------------------------


class TestSecurityBarrierRedactDict:
    def test_empty_dict(self):
        sb = SecurityBarrier()
        assert sb.redact_dict({}) == {}

    def test_none_dict(self):
        sb = SecurityBarrier()
        assert sb.redact_dict(None) is None

    def test_string_values(self):
        sb = SecurityBarrier()
        result = sb.redact_dict({"key": "token = mySecret123"})
        assert "[REDACTED]" in result["key"]

    def test_nested_dict(self):
        sb = SecurityBarrier()
        result = sb.redact_dict({"outer": {"inner": "password = abc123"}})
        assert "[REDACTED]" in result["outer"]["inner"]

    def test_list_values(self):
        sb = SecurityBarrier()
        result = sb.redact_dict({"items": ["safe text", "api_key = secret123"]})
        assert result["items"][0] == "safe text"
        assert "[REDACTED]" in result["items"][1]

    def test_list_with_dicts(self):
        sb = SecurityBarrier()
        result = sb.redact_dict({"items": [{"key": "token = abc"}]})
        assert "[REDACTED]" in result["items"][0]["key"]

    def test_non_string_values_preserved(self):
        sb = SecurityBarrier()
        result = sb.redact_dict({"count": 42, "active": True, "ratio": 3.14})
        assert result == {"count": 42, "active": True, "ratio": 3.14}

    def test_mixed_types(self):
        sb = SecurityBarrier()
        result = sb.redact_dict(
            {
                "name": "safe",
                "secret": "api_key = xyz",
                "count": 5,
                "nested": {"password": "secret = abc"},
                "items": ["ok", "token = bad"],
            }
        )
        assert result["name"] == "safe"
        assert "[REDACTED]" in result["secret"]
        assert result["count"] == 5


# ---------------------------------------------------------------------------
# SecurityBarrier — contains_sensitive
# ---------------------------------------------------------------------------


class TestContainsSensitive:
    def test_empty(self):
        sb = SecurityBarrier()
        assert sb.contains_sensitive("") is False
        assert sb.contains_sensitive(None) is False

    def test_safe_content(self):
        sb = SecurityBarrier()
        assert sb.contains_sensitive("Hello world") is False

    def test_sensitive_content(self):
        sb = SecurityBarrier()
        assert sb.contains_sensitive("api_key = mySecret") is True

    def test_uses_custom_patterns(self):
        sb = SecurityBarrier()
        sb.add_pattern(r"CUSTOM_\d+")
        assert sb.contains_sensitive("Found CUSTOM_123") is True


# ---------------------------------------------------------------------------
# TelemetryVerifier — init
# ---------------------------------------------------------------------------


class TestTelemetryVerifierInit:
    def test_empty_cache(self):
        tv = TelemetryVerifier()
        assert tv._capability_cache == {}
        assert tv._verification_results == []

    def test_capability_requirements(self):
        assert "thought_streaming" in TelemetryVerifier.CAPABILITY_REQUIREMENTS
        assert "capability_probe" in TelemetryVerifier.CAPABILITY_REQUIREMENTS
        assert "diagnostic" in TelemetryVerifier.CAPABILITY_REQUIREMENTS


# ---------------------------------------------------------------------------
# TelemetryVerifier — verify_agent
# ---------------------------------------------------------------------------


class TestVerifyAgent:
    def test_all_capabilities_present(self):
        class FakeAgent:
            name = "claude"
            generate = lambda: None
            model = "opus"

        tv = TelemetryVerifier()
        passed, missing = tv.verify_agent(FakeAgent(), ["name", "generate", "model"])
        assert passed is True
        assert missing == []

    def test_missing_capability(self):
        class FakeAgent:
            name = "claude"

        tv = TelemetryVerifier()
        passed, missing = tv.verify_agent(FakeAgent(), ["name", "generate"])
        assert passed is False
        assert "generate" in missing

    def test_none_capability_is_missing(self):
        class FakeAgent:
            name = "claude"
            generate = None

        tv = TelemetryVerifier()
        passed, missing = tv.verify_agent(FakeAgent(), ["name", "generate"])
        assert passed is False
        assert "generate" in missing

    def test_default_capabilities(self):
        class FakeAgent:
            name = "claude"
            generate = lambda: None

        tv = TelemetryVerifier()
        passed, missing = tv.verify_agent(FakeAgent())
        # Default is thought_streaming: ["generate", "name"]
        assert passed is True

    def test_caches_result(self):
        class FakeAgent:
            name = "claude"
            generate = lambda: None

        tv = TelemetryVerifier()
        tv.verify_agent(FakeAgent(), ["name", "generate"])
        assert "claude" in tv._capability_cache
        assert tv._capability_cache["claude"] == {"name", "generate"}

    def test_records_verification(self):
        class FakeAgent:
            name = "claude"

        tv = TelemetryVerifier()
        tv.verify_agent(FakeAgent(), ["name"])
        assert len(tv._verification_results) == 1
        assert tv._verification_results[0]["agent"] == "claude"
        assert tv._verification_results[0]["passed"] is True

    def test_agent_without_name_uses_str(self):
        tv = TelemetryVerifier()
        passed, missing = tv.verify_agent("string_agent", ["name"])
        assert passed is False
        assert "name" in missing


# ---------------------------------------------------------------------------
# TelemetryVerifier — verify_telemetry_level
# ---------------------------------------------------------------------------


class TestVerifyTelemetryLevel:
    def test_thought_streaming_pass(self):
        class FakeAgent:
            name = "claude"
            generate = lambda: None

        tv = TelemetryVerifier()
        assert tv.verify_telemetry_level("thought_streaming", FakeAgent()) is True

    def test_capability_probe_fail(self):
        class FakeAgent:
            name = "claude"
            generate = lambda: None
            # Missing: model

        tv = TelemetryVerifier()
        assert tv.verify_telemetry_level("capability_probe", FakeAgent()) is False

    def test_diagnostic_pass(self):
        class FakeAgent:
            name = "claude"

        tv = TelemetryVerifier()
        assert tv.verify_telemetry_level("diagnostic", FakeAgent()) is True

    def test_unknown_level(self):
        class FakeAgent:
            name = "claude"

        tv = TelemetryVerifier()
        # Unknown level has empty requirements
        assert tv.verify_telemetry_level("unknown_level", FakeAgent()) is True


# ---------------------------------------------------------------------------
# TelemetryVerifier — get_verification_report
# ---------------------------------------------------------------------------


class TestGetVerificationReport:
    def test_empty_report(self):
        tv = TelemetryVerifier()
        report = tv.get_verification_report()
        assert report == {"total": 0, "passed": 0, "failed": 0, "agents": []}

    def test_report_with_results(self):
        class PassAgent:
            name = "passer"
            generate = lambda: None

        class FailAgent:
            name = "failer"

        tv = TelemetryVerifier()
        tv.verify_agent(PassAgent(), ["name", "generate"])
        tv.verify_agent(FailAgent(), ["name", "generate"])
        report = tv.get_verification_report()
        assert report["total"] == 2
        assert report["passed"] == 1
        assert report["failed"] == 1
        assert len(report["agents"]) == 2


# ---------------------------------------------------------------------------
# TelemetryVerifier — clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clears_all(self):
        class FakeAgent:
            name = "claude"

        tv = TelemetryVerifier()
        tv.verify_agent(FakeAgent(), ["name"])
        assert len(tv._capability_cache) > 0
        assert len(tv._verification_results) > 0
        tv.clear_cache()
        assert tv._capability_cache == {}
        assert tv._verification_results == []
