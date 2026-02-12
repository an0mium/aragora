"""Tests for aragora.gateway.enterprise.audit_interceptor module.

Covers:
- RedactionType and AuditEventType enums
- PIIRedactionRule creation, matching, and all redaction types
  (MASK, HASH, REMOVE, TRUNCATE, TOKENIZE, custom handler)
- AuditConfig defaults, custom values, auto-generated PII rules, default rules
- AuditRecord creation, serialization (to_dict, to_signed_dict, from_dict),
  hash computation, signature computation/verification
- Hash chain integrity across multiple records
- HMAC signing key management (get, set, env vars, production guard)
- InMemoryAuditStorage operations (store, get, query, delete, count, eviction, chain)
- AuditInterceptor intercept (request/response logging, header redaction,
  body hashing, PII redaction, response hashing, body truncation)
- AuditInterceptor verify_chain (valid chain, broken chain, tampered records)
- AuditInterceptor apply_retention
- AuditInterceptor export_soc2_evidence
- Event handler management and emission
- Metrics collection
- Webhook integration
- PostgresAuditStorage initialization
- Edge cases and error handling
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from aragora.gateway.enterprise.audit_interceptor import (
    AuditConfig,
    AuditEventType,
    AuditInterceptor,
    AuditRecord,
    AuditStorage,
    InMemoryAuditStorage,
    PIIRedactionRule,
    PostgresAuditStorage,
    RedactionType,
    get_interceptor_signing_key,
    set_interceptor_signing_key,
)

import aragora.gateway.enterprise.audit_interceptor as _mod


# =============================================================================
# Signing Key Management
# =============================================================================


class TestSigningKeyManagement:
    """Test HMAC signing key get/set."""

    @pytest.fixture(autouse=True)
    def reset_signing_key(self, monkeypatch):
        """Reset global signing key before each test."""
        _mod._INTERCEPTOR_SIGNING_KEY = None
        monkeypatch.delenv("ARAGORA_AUDIT_INTERCEPTOR_KEY", raising=False)
        monkeypatch.delenv("ARAGORA_AUDIT_SIGNING_KEY", raising=False)
        monkeypatch.delenv("ARAGORA_ENV", raising=False)

    def test_generates_ephemeral_key_in_development(self):
        key = get_interceptor_signing_key()
        assert isinstance(key, bytes)
        assert len(key) >= 32

    def test_loads_key_from_env(self, monkeypatch):
        hex_key = secrets.token_hex(32)
        monkeypatch.setenv("ARAGORA_AUDIT_INTERCEPTOR_KEY", hex_key)
        key = get_interceptor_signing_key()
        assert key == bytes.fromhex(hex_key)

    def test_loads_key_from_fallback_env(self, monkeypatch):
        hex_key = secrets.token_hex(32)
        monkeypatch.setenv("ARAGORA_AUDIT_SIGNING_KEY", hex_key)
        key = get_interceptor_signing_key()
        assert key == bytes.fromhex(hex_key)

    def test_primary_env_takes_precedence_over_fallback(self, monkeypatch):
        primary = secrets.token_hex(32)
        fallback = secrets.token_hex(32)
        monkeypatch.setenv("ARAGORA_AUDIT_INTERCEPTOR_KEY", primary)
        monkeypatch.setenv("ARAGORA_AUDIT_SIGNING_KEY", fallback)
        key = get_interceptor_signing_key()
        assert key == bytes.fromhex(primary)

    def test_raises_in_production_without_key(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with pytest.raises(RuntimeError, match="required in production"):
            get_interceptor_signing_key()

    def test_raises_in_staging_without_key(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "staging")
        with pytest.raises(RuntimeError, match="required in staging"):
            get_interceptor_signing_key()

    def test_raises_in_prod_short_name(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "prod")
        with pytest.raises(RuntimeError):
            get_interceptor_signing_key()

    def test_raises_on_short_key(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_AUDIT_INTERCEPTOR_KEY", "abcd1234")
        with pytest.raises(RuntimeError, match="32 bytes"):
            get_interceptor_signing_key()

    def test_raises_on_invalid_hex(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_AUDIT_INTERCEPTOR_KEY", "not-hex-string")
        with pytest.raises(RuntimeError, match="Invalid signing key"):
            get_interceptor_signing_key()

    def test_set_signing_key(self):
        key = secrets.token_bytes(32)
        set_interceptor_signing_key(key)
        assert get_interceptor_signing_key() == key

    def test_set_short_key_raises(self):
        with pytest.raises(ValueError, match="32 bytes"):
            set_interceptor_signing_key(b"short")

    def test_set_key_exact_32_bytes(self):
        key = b"\x00" * 32
        set_interceptor_signing_key(key)
        assert get_interceptor_signing_key() == key

    def test_key_is_cached_after_first_call(self):
        k1 = get_interceptor_signing_key()
        k2 = get_interceptor_signing_key()
        assert k1 is k2


# =============================================================================
# RedactionType Enum Tests
# =============================================================================


class TestRedactionType:
    """Tests for RedactionType enum."""

    def test_all_values_present(self):
        assert RedactionType.MASK == "mask"
        assert RedactionType.HASH == "hash"
        assert RedactionType.REMOVE == "remove"
        assert RedactionType.TRUNCATE == "truncate"
        assert RedactionType.TOKENIZE == "tokenize"

    def test_is_string_enum(self):
        assert isinstance(RedactionType.MASK, str)
        assert RedactionType.MASK == "mask"

    def test_member_count(self):
        assert len(RedactionType) == 5


# =============================================================================
# AuditEventType Enum Tests
# =============================================================================


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_all_values_present(self):
        assert AuditEventType.REQUEST_RECEIVED == "request_received"
        assert AuditEventType.RESPONSE_SENT == "response_sent"
        assert AuditEventType.REQUEST_FAILED == "request_failed"
        assert AuditEventType.PII_REDACTED == "pii_redacted"
        assert AuditEventType.CHAIN_VERIFIED == "chain_verified"
        assert AuditEventType.CHAIN_BROKEN == "chain_broken"
        assert AuditEventType.RETENTION_APPLIED == "retention_applied"
        assert AuditEventType.EXPORT_GENERATED == "export_generated"

    def test_is_string_enum(self):
        assert isinstance(AuditEventType.RESPONSE_SENT, str)

    def test_member_count(self):
        assert len(AuditEventType) == 8


# =============================================================================
# PIIRedactionRule Tests
# =============================================================================


class TestPIIRedactionRule:
    """Test PII redaction strategies."""

    def test_matches_field(self):
        rule = PIIRedactionRule(field_pattern=r".*email.*", redaction_type=RedactionType.MASK)
        assert rule.matches("user_email") is True
        assert rule.matches("email_address") is True
        assert rule.matches("name") is False

    def test_matches_case_insensitive(self):
        rule = PIIRedactionRule(field_pattern=r".*email.*", redaction_type=RedactionType.MASK)
        assert rule.matches("User_Email") is True
        assert rule.matches("EMAIL") is True

    def test_matches_complex_pattern(self):
        rule = PIIRedactionRule(
            field_pattern=r"^(ssn|social_security)$", redaction_type=RedactionType.MASK
        )
        assert rule.matches("ssn") is True
        assert rule.matches("social_security") is True
        assert not rule.matches("my_ssn_field")

    def test_redact_none(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.MASK)
        assert rule.redact(None) is None

    # -- MASK redaction --

    def test_redact_mask(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.MASK)
        result = rule.redact("test@example.com")
        assert result[0] == "t"
        assert result[-1] == "m"
        assert "*" in result
        assert len(result) == len("test@example.com")

    def test_redact_mask_short_string_2_chars(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.MASK)
        result = rule.redact("ab")
        assert result == "**"

    def test_redact_mask_single_char(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.MASK)
        result = rule.redact("x")
        assert result == "*"

    def test_redact_mask_three_chars(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.MASK)
        result = rule.redact("abc")
        assert result == "a*c"

    def test_redact_mask_custom_char(self):
        rule = PIIRedactionRule(
            field_pattern=r".*", redaction_type=RedactionType.MASK, mask_char="#"
        )
        result = rule.redact("hello")
        assert result == "h###o"

    # -- HASH redaction --

    def test_redact_hash(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.HASH)
        result = rule.redact("test@example.com")
        assert result.startswith("[HASH:")
        assert result.endswith("]")
        hex_part = result[6:-1]
        assert len(hex_part) == 16

    def test_redact_hash_deterministic(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.HASH)
        r1 = rule.redact("same-value")
        r2 = rule.redact("same-value")
        assert r1 == r2

    def test_redact_hash_different_values_differ(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.HASH)
        assert rule.redact("value1") != rule.redact("value2")

    def test_redact_hash_matches_sha256_prefix(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.HASH)
        result = rule.redact("test")
        expected = hashlib.sha256(b"test").hexdigest()[:16]
        assert result == f"[HASH:{expected}]"

    # -- REMOVE redaction --

    def test_redact_remove(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.REMOVE)
        assert rule.redact("sensitive-data") == "[REDACTED]"

    def test_redact_remove_any_type(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.REMOVE)
        assert rule.redact(12345) == "[REDACTED]"
        assert rule.redact(True) == "[REDACTED]"

    # -- TRUNCATE redaction --

    def test_redact_truncate(self):
        rule = PIIRedactionRule(
            field_pattern=r".*", redaction_type=RedactionType.TRUNCATE, visible_chars=2
        )
        result = rule.redact("test@example.com")
        assert result.startswith("te")
        assert result.endswith("om")
        assert "..." in result

    def test_redact_truncate_short_string(self):
        rule = PIIRedactionRule(
            field_pattern=r".*", redaction_type=RedactionType.TRUNCATE, visible_chars=3
        )
        result = rule.redact("ab")
        assert result == "**"

    def test_redact_truncate_exact_boundary(self):
        rule = PIIRedactionRule(
            field_pattern=r".*", redaction_type=RedactionType.TRUNCATE, visible_chars=2
        )
        # 4 chars == 2 * visible_chars, so falls into mask branch
        result = rule.redact("abcd")
        assert result == "****"

    def test_redact_truncate_custom_visible_chars(self):
        rule = PIIRedactionRule(
            field_pattern=r".*", redaction_type=RedactionType.TRUNCATE, visible_chars=3
        )
        result = rule.redact("alice@example.com")
        assert result == "ali...com"

    # -- TOKENIZE redaction --

    def test_redact_tokenize(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.TOKENIZE)
        result = rule.redact("test-value")
        assert result.startswith("[TOKEN:")
        assert result.endswith("]")
        token_part = result[7:-1]
        assert len(token_part) == 12

    def test_redact_tokenize_deterministic(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.TOKENIZE)
        assert rule.redact("val") == rule.redact("val")

    def test_redact_tokenize_matches_sha256_prefix(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.TOKENIZE)
        result = rule.redact("mydata")
        expected = hashlib.sha256(b"mydata").hexdigest()[:12]
        assert result == f"[TOKEN:{expected}]"

    # -- Custom handler --

    def test_custom_handler(self):
        def custom(pattern, value):
            return f"CUSTOM:{value}"

        rule = PIIRedactionRule(
            field_pattern=r".*",
            redaction_type=RedactionType.MASK,
            custom_handler=custom,
        )
        assert rule.redact("hello") == "CUSTOM:hello"

    def test_custom_handler_overrides_type(self):
        handler = MagicMock(return_value="handled")
        rule = PIIRedactionRule(
            field_pattern=r".*", redaction_type=RedactionType.REMOVE, custom_handler=handler
        )
        result = rule.redact("value")
        assert result == "handled"
        handler.assert_called_once()

    # -- Non-string values --

    def test_redact_integer_mask(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.MASK)
        result = rule.redact(12345)
        assert result == "1***5"

    def test_redact_boolean_remove(self):
        rule = PIIRedactionRule(field_pattern=r".*", redaction_type=RedactionType.REMOVE)
        assert rule.redact(False) == "[REDACTED]"


# =============================================================================
# AuditConfig Tests
# =============================================================================


class TestAuditConfig:
    """Test AuditConfig initialization."""

    def test_defaults(self):
        config = AuditConfig()
        assert config.retention_days == 365
        assert config.emit_events is True
        assert config.storage_backend == "memory"
        assert config.hash_responses is False
        assert config.max_body_size == 1024 * 1024
        assert config.webhook_url is None
        assert config.webhook_headers == {}
        assert config.webhook_timeout == 5.0
        assert config.enable_metrics is True
        assert config.metrics_prefix == "aragora_audit"
        assert config.chain_verification_interval == 3600

    def test_default_pii_rules_always_present(self):
        config = AuditConfig()
        assert len(config.pii_rules) >= 8

    def test_default_sensitive_headers(self):
        config = AuditConfig()
        assert "authorization" in config.sensitive_headers
        assert "x-api-key" in config.sensitive_headers
        assert "cookie" in config.sensitive_headers
        assert "x-csrf-token" in config.sensitive_headers
        assert "x-auth-token" in config.sensitive_headers

    def test_pii_fields_create_rules(self):
        config = AuditConfig(pii_fields=["email", "phone"])
        matching = [r for r in config.pii_rules if r.matches("email")]
        assert len(matching) > 0
        assert matching[0].redaction_type == RedactionType.MASK

    def test_pii_fields_create_rules_for_phone(self):
        config = AuditConfig(pii_fields=["phone"])
        phone_rules = [r for r in config.pii_rules if r.matches("user_phone")]
        assert len(phone_rules) > 0

    def test_default_password_rule(self):
        config = AuditConfig()
        password_rules = [r for r in config.pii_rules if r.matches("password")]
        assert len(password_rules) > 0
        assert password_rules[0].redaction_type == RedactionType.REMOVE

    def test_default_secret_rule(self):
        config = AuditConfig()
        secret_rules = [r for r in config.pii_rules if r.matches("client_secret")]
        assert len(secret_rules) > 0
        assert secret_rules[0].redaction_type == RedactionType.REMOVE

    def test_default_token_rule(self):
        config = AuditConfig()
        token_rules = [r for r in config.pii_rules if r.matches("access_token")]
        assert len(token_rules) > 0
        assert token_rules[0].redaction_type == RedactionType.HASH

    def test_default_api_key_rule(self):
        config = AuditConfig()
        rules = [r for r in config.pii_rules if r.matches("api_key")]
        assert len(rules) > 0
        assert rules[0].redaction_type == RedactionType.HASH

    def test_default_ssn_rule(self):
        config = AuditConfig()
        rules = [r for r in config.pii_rules if r.matches("ssn")]
        assert len(rules) > 0
        assert rules[0].redaction_type == RedactionType.MASK

    def test_default_credit_card_rule(self):
        config = AuditConfig()
        rules = [r for r in config.pii_rules if r.matches("credit_card")]
        assert len(rules) > 0
        assert rules[0].redaction_type == RedactionType.MASK

    def test_user_rules_take_precedence(self):
        user_rule = PIIRedactionRule(
            field_pattern=r".*password.*", redaction_type=RedactionType.HASH
        )
        config = AuditConfig(pii_rules=[user_rule])
        password_rules = [r for r in config.pii_rules if r.matches("password")]
        assert password_rules[0].redaction_type == RedactionType.HASH

    def test_custom_values(self):
        config = AuditConfig(
            retention_days=30,
            emit_events=False,
            webhook_url="https://siem.example.com",
            hash_responses=True,
            max_body_size=512,
            storage_backend="postgres",
            metrics_prefix="custom_audit",
        )
        assert config.retention_days == 30
        assert config.emit_events is False
        assert config.webhook_url == "https://siem.example.com"
        assert config.hash_responses is True
        assert config.max_body_size == 512
        assert config.storage_backend == "postgres"
        assert config.metrics_prefix == "custom_audit"

    def test_pii_fields_not_used_when_rules_provided(self):
        """When pii_rules is provided, pii_fields should not auto-generate rules."""
        user_rule = PIIRedactionRule(
            field_pattern=r".*custom.*", redaction_type=RedactionType.TOKENIZE
        )
        config = AuditConfig(pii_fields=["email"], pii_rules=[user_rule])
        # The first rule should be the user-provided one, not auto-generated from pii_fields
        assert config.pii_rules[0].redaction_type == RedactionType.TOKENIZE


# =============================================================================
# AuditRecord Tests
# =============================================================================


class TestAuditRecord:
    """Test AuditRecord data structure."""

    @pytest.fixture(autouse=True)
    def setup_signing_key(self):
        """Ensure a signing key is available."""
        _mod._INTERCEPTOR_SIGNING_KEY = None
        os.environ.pop("ARAGORA_ENV", None)
        os.environ.pop("ARAGORA_AUDIT_INTERCEPTOR_KEY", None)

    @pytest.fixture
    def record(self):
        return AuditRecord(
            id="test-record-1",
            correlation_id="corr-123",
            request_method="POST",
            request_path="/api/users",
            request_body={"name": "Alice"},
            request_body_hash="abc123",
            response_status=200,
            response_body={"id": 1},
            response_body_hash="def456",
            duration_ms=42.5,
            user_id="user-1",
            org_id="org-1",
        )

    def test_creation_defaults(self):
        record = AuditRecord()
        assert record.id  # UUID generated
        assert record.correlation_id == ""
        assert isinstance(record.timestamp, datetime)
        assert record.request_method == ""
        assert record.request_path == ""
        assert record.request_headers == {}
        assert record.request_body is None
        assert record.response_status == 0
        assert record.duration_ms == 0.0
        assert record.user_id is None
        assert record.org_id is None
        assert record.record_hash == ""
        assert record.previous_hash == ""
        assert record.signature == ""
        assert record.metadata == {}
        assert record.pii_fields_redacted == []

    def test_unique_ids(self):
        r1 = AuditRecord()
        r2 = AuditRecord()
        assert r1.id != r2.id

    def test_compute_hash_deterministic(self, record):
        h1 = record.compute_hash()
        h2 = record.compute_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_hash_changes_with_data(self, record):
        h1 = record.compute_hash()
        record.request_method = "GET"
        h2 = record.compute_hash()
        assert h1 != h2

    def test_compute_hash_sensitive_to_id(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        r1 = AuditRecord(id="rec-1", timestamp=ts)
        r2 = AuditRecord(id="rec-2", timestamp=ts)
        assert r1.compute_hash() != r2.compute_hash()

    def test_compute_hash_sensitive_to_previous_hash(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        r1 = AuditRecord(id="rec-1", timestamp=ts, previous_hash="")
        r2 = AuditRecord(id="rec-1", timestamp=ts, previous_hash="different")
        assert r1.compute_hash() != r2.compute_hash()

    def test_compute_hash_uses_sha256(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        record = AuditRecord(
            id="rec-1",
            correlation_id="corr-1",
            timestamp=ts,
            request_method="GET",
            request_path="/api",
            request_body_hash="bh",
            response_status=200,
            response_body_hash="rbh",
            duration_ms=42.0,
            user_id="user-1",
            previous_hash="prev",
        )
        expected_data = (
            f"{record.id}|{record.correlation_id}|{ts.isoformat()}|"
            f"{record.request_method}|{record.request_path}|{record.request_body_hash}|"
            f"{record.response_status}|{record.response_body_hash}|{record.duration_ms}|"
            f"{record.user_id}|{record.previous_hash}"
        )
        expected_hash = hashlib.sha256(expected_data.encode()).hexdigest()
        assert record.compute_hash() == expected_hash

    def test_compute_signature(self, record):
        record.record_hash = record.compute_hash()
        sig = record.compute_signature()
        assert len(sig) == 64

    def test_compute_signature_deterministic(self, record):
        record.record_hash = record.compute_hash()
        assert record.compute_signature() == record.compute_signature()

    def test_verify_signature_valid(self, record):
        record.record_hash = record.compute_hash()
        record.signature = record.compute_signature()
        assert record.verify_signature() is True

    def test_verify_signature_invalid(self, record):
        record.record_hash = record.compute_hash()
        record.signature = "invalid" * 8
        assert record.verify_signature() is False

    def test_verify_signature_empty(self, record):
        assert record.verify_signature() is False

    def test_signature_changes_with_different_keys(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        record = AuditRecord(id="rec-1", timestamp=ts)
        record.record_hash = record.compute_hash()

        set_interceptor_signing_key(b"\x01" * 32)
        sig1 = record.compute_signature()

        _mod._INTERCEPTOR_SIGNING_KEY = None
        set_interceptor_signing_key(b"\x02" * 32)
        sig2 = record.compute_signature()

        assert sig1 != sig2

    def test_to_dict(self, record):
        d = record.to_dict()
        assert d["id"] == "test-record-1"
        assert d["correlation_id"] == "corr-123"
        assert d["request_method"] == "POST"
        assert d["request_path"] == "/api/users"
        assert d["response_status"] == 200
        assert d["duration_ms"] == 42.5
        assert d["user_id"] == "user-1"
        assert d["org_id"] == "org-1"

    def test_to_dict_all_fields_present(self):
        d = AuditRecord().to_dict()
        expected_keys = {
            "id",
            "correlation_id",
            "timestamp",
            "request_method",
            "request_path",
            "request_headers",
            "request_body",
            "request_body_hash",
            "response_status",
            "response_headers",
            "response_body",
            "response_body_hash",
            "duration_ms",
            "user_id",
            "org_id",
            "ip_address",
            "user_agent",
            "record_hash",
            "previous_hash",
            "signature",
            "metadata",
            "pii_fields_redacted",
        }
        assert set(d.keys()) == expected_keys

    def test_to_signed_dict(self, record):
        d = record.to_signed_dict()
        assert d["record_hash"] != ""
        assert d["signature"] != ""
        assert record.record_hash == d["record_hash"]
        assert record.signature == d["signature"]

    def test_from_dict_roundtrip(self, record):
        d = record.to_dict()
        restored = AuditRecord.from_dict(d)
        assert restored.id == record.id
        assert restored.correlation_id == record.correlation_id
        assert restored.request_method == record.request_method
        assert restored.response_status == record.response_status
        assert restored.user_id == record.user_id

    def test_from_dict_with_iso_timestamp(self):
        d = {
            "id": "r1",
            "timestamp": "2026-01-30T12:00:00+00:00",
            "request_method": "GET",
        }
        record = AuditRecord.from_dict(d)
        assert record.id == "r1"
        assert record.timestamp.year == 2026

    def test_from_dict_with_z_suffix_timestamp(self):
        d = {"id": "r1", "timestamp": "2024-06-15T12:00:00Z"}
        record = AuditRecord.from_dict(d)
        assert record.timestamp.tzinfo is not None

    def test_from_dict_missing_timestamp(self):
        d = {"id": "r1"}
        record = AuditRecord.from_dict(d)
        assert isinstance(record.timestamp, datetime)

    def test_from_dict_defaults(self):
        record = AuditRecord.from_dict({})
        assert record.request_method == ""
        assert record.response_status == 0
        assert record.metadata == {}
        assert record.request_headers == {}
        assert record.pii_fields_redacted == []

    def test_from_dict_preserves_metadata(self):
        d = {"metadata": {"tenant": "acme", "version": 2}}
        record = AuditRecord.from_dict(d)
        assert record.metadata == {"tenant": "acme", "version": 2}

    def test_from_dict_preserves_pii_fields(self):
        d = {"pii_fields_redacted": ["email", "ssn"]}
        record = AuditRecord.from_dict(d)
        assert record.pii_fields_redacted == ["email", "ssn"]


# =============================================================================
# InMemoryAuditStorage Tests
# =============================================================================


class TestInMemoryAuditStorage:
    """Test in-memory audit storage operations."""

    @pytest.fixture
    def storage(self):
        return InMemoryAuditStorage(max_records=100)

    @pytest.fixture
    def sample_record(self):
        return AuditRecord(
            id="r1",
            correlation_id="c1",
            request_method="POST",
            request_path="/api/users",
            response_status=200,
            duration_ms=50.0,
            user_id="user-1",
            org_id="org-1",
        )

    @pytest.mark.asyncio
    async def test_store_and_get(self, storage, sample_record):
        await storage.store(sample_record)
        result = await storage.get("r1")
        assert result is not None
        assert result.id == "r1"

    @pytest.mark.asyncio
    async def test_get_missing(self, storage):
        result = await storage.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_count_empty(self, storage):
        assert await storage.count() == 0

    @pytest.mark.asyncio
    async def test_count_after_store(self, storage, sample_record):
        await storage.store(sample_record)
        assert await storage.count() == 1

    @pytest.mark.asyncio
    async def test_count_multiple(self, storage):
        await storage.store(AuditRecord(id="r1"))
        await storage.store(AuditRecord(id="r2"))
        assert await storage.count() == 2

    @pytest.mark.asyncio
    async def test_query_no_filters(self, storage):
        await storage.store(AuditRecord(id="r1"))
        await storage.store(AuditRecord(id="r2"))
        results = await storage.query()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_user(self, storage):
        r1 = AuditRecord(id="r1", user_id="alice", request_method="GET")
        r2 = AuditRecord(id="r2", user_id="bob", request_method="GET")
        await storage.store(r1)
        await storage.store(r2)

        results = await storage.query(user_id="alice")
        assert len(results) == 1
        assert results[0].id == "r1"

    @pytest.mark.asyncio
    async def test_query_by_org(self, storage):
        r1 = AuditRecord(id="r1", org_id="org-a", request_method="GET")
        r2 = AuditRecord(id="r2", org_id="org-b", request_method="GET")
        await storage.store(r1)
        await storage.store(r2)

        results = await storage.query(org_id="org-a")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_by_correlation_id(self, storage):
        r1 = AuditRecord(id="r1", correlation_id="c1", request_method="GET")
        r2 = AuditRecord(id="r2", correlation_id="c2", request_method="GET")
        await storage.store(r1)
        await storage.store(r2)

        results = await storage.query(correlation_id="c1")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_by_path_prefix(self, storage):
        r1 = AuditRecord(id="r1", request_path="/api/users/1", request_method="GET")
        r2 = AuditRecord(id="r2", request_path="/api/admin/settings", request_method="GET")
        await storage.store(r1)
        await storage.store(r2)

        results = await storage.query(request_path="/api/users")
        assert len(results) == 1
        assert results[0].request_path.startswith("/api/users")

    @pytest.mark.asyncio
    async def test_query_pagination(self, storage):
        for i in range(10):
            await storage.store(AuditRecord(id=f"r{i}", request_method="GET"))

        page1 = await storage.query(limit=3, offset=0)
        page2 = await storage.query(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id

    @pytest.mark.asyncio
    async def test_query_returns_most_recent_first(self, storage):
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        await storage.store(AuditRecord(id="older", timestamp=ts1))
        await storage.store(AuditRecord(id="newer", timestamp=ts2))
        results = await storage.query()
        assert results[0].id == "newer"
        assert results[1].id == "older"

    @pytest.mark.asyncio
    async def test_query_by_date_range(self, storage):
        now = datetime.now(timezone.utc)
        old = AuditRecord(
            id="r-old",
            timestamp=now - timedelta(days=30),
            request_method="GET",
        )
        recent = AuditRecord(
            id="r-recent",
            timestamp=now - timedelta(hours=1),
            request_method="GET",
        )
        await storage.store(old)
        await storage.store(recent)

        results = await storage.query(start_date=now - timedelta(days=1))
        assert len(results) == 1
        assert results[0].id == "r-recent"

    @pytest.mark.asyncio
    async def test_query_by_end_date(self, storage):
        now = datetime.now(timezone.utc)
        r1 = AuditRecord(id="r1", timestamp=now - timedelta(days=10))
        r2 = AuditRecord(id="r2", timestamp=now)
        await storage.store(r1)
        await storage.store(r2)

        results = await storage.query(end_date=now - timedelta(days=5))
        assert len(results) == 1
        assert results[0].id == "r1"

    @pytest.mark.asyncio
    async def test_get_last_hash_empty(self, storage):
        assert await storage.get_last_hash() == ""

    @pytest.mark.asyncio
    async def test_get_last_hash_updates(self, storage):
        r = AuditRecord(id="r1", record_hash="hash123", request_method="GET")
        await storage.store(r)
        assert await storage.get_last_hash() == "hash123"

        r2 = AuditRecord(id="r2", record_hash="hash456")
        await storage.store(r2)
        assert await storage.get_last_hash() == "hash456"

    @pytest.mark.asyncio
    async def test_get_chain_order(self, storage):
        now = datetime.now(timezone.utc)
        r1 = AuditRecord(id="r1", timestamp=now - timedelta(hours=2), request_method="GET")
        r2 = AuditRecord(id="r2", timestamp=now - timedelta(hours=1), request_method="GET")
        await storage.store(r1)
        await storage.store(r2)

        chain = await storage.get_chain()
        assert len(chain) == 2
        assert chain[0].id == "r1"  # oldest first

    @pytest.mark.asyncio
    async def test_get_chain_with_date_range(self, storage):
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        ts3 = datetime(2024, 12, 1, tzinfo=timezone.utc)
        await storage.store(AuditRecord(id="r1", timestamp=ts1))
        await storage.store(AuditRecord(id="r2", timestamp=ts2))
        await storage.store(AuditRecord(id="r3", timestamp=ts3))
        chain = await storage.get_chain(
            start_date=datetime(2024, 3, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 9, 1, tzinfo=timezone.utc),
        )
        assert len(chain) == 1
        assert chain[0].id == "r2"

    @pytest.mark.asyncio
    async def test_delete_before(self, storage):
        now = datetime.now(timezone.utc)
        old = AuditRecord(id="r-old", timestamp=now - timedelta(days=400), request_method="GET")
        recent = AuditRecord(id="r-recent", timestamp=now, request_method="GET")
        await storage.store(old)
        await storage.store(recent)

        deleted = await storage.delete_before(now - timedelta(days=365))
        assert deleted == 1
        assert await storage.count() == 1
        assert await storage.get("r-old") is None
        assert await storage.get("r-recent") is not None

    @pytest.mark.asyncio
    async def test_delete_before_no_match(self, storage):
        now = datetime.now(timezone.utc)
        await storage.store(AuditRecord(id="r1", timestamp=now))
        deleted = await storage.delete_before(now - timedelta(days=365))
        assert deleted == 0
        assert await storage.count() == 1

    @pytest.mark.asyncio
    async def test_max_records_eviction(self):
        storage = InMemoryAuditStorage(max_records=3)
        for i in range(5):
            await storage.store(AuditRecord(id=f"r{i}", request_method="GET"))

        assert await storage.count() == 3
        assert await storage.get("r0") is None
        assert await storage.get("r1") is None
        assert await storage.get("r4") is not None

    @pytest.mark.asyncio
    async def test_count_with_date_range(self, storage):
        now = datetime.now(timezone.utc)
        for i in range(5):
            await storage.store(
                AuditRecord(
                    id=f"r{i}",
                    timestamp=now - timedelta(hours=i),
                    request_method="GET",
                )
            )

        count = await storage.count(start_date=now - timedelta(hours=2))
        assert count == 3  # r0, r1, r2

    @pytest.mark.asyncio
    async def test_count_with_end_date(self, storage):
        now = datetime.now(timezone.utc)
        await storage.store(AuditRecord(id="r1", timestamp=now - timedelta(days=10)))
        await storage.store(AuditRecord(id="r2", timestamp=now))
        count = await storage.count(end_date=now - timedelta(days=5))
        assert count == 1


# =============================================================================
# PostgresAuditStorage Initialization Tests
# =============================================================================


class TestPostgresAuditStorage:
    """Test PostgresAuditStorage initialization (no real DB)."""

    def test_requires_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("ARAGORA_POSTGRES_DSN", raising=False)
        with pytest.raises(ValueError, match="PostgreSQL URL required"):
            PostgresAuditStorage()

    def test_explicit_url(self):
        storage = PostgresAuditStorage(database_url="postgresql://localhost/test")
        assert storage._database_url == "postgresql://localhost/test"

    def test_url_from_env_database_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://env/test")
        storage = PostgresAuditStorage()
        assert storage._database_url == "postgresql://env/test"

    def test_url_from_env_aragora_dsn(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("ARAGORA_POSTGRES_DSN", "postgresql://dsn/test")
        storage = PostgresAuditStorage()
        assert storage._database_url == "postgresql://dsn/test"

    def test_schema_constant(self):
        assert "gateway_audit_records" in PostgresAuditStorage.SCHEMA
        assert "CREATE TABLE" in PostgresAuditStorage.SCHEMA
        assert "record_hash" in PostgresAuditStorage.SCHEMA
        assert "previous_hash" in PostgresAuditStorage.SCHEMA

    def test_initial_last_hash_empty(self):
        storage = PostgresAuditStorage(database_url="postgresql://localhost/test")
        assert storage._last_hash == ""

    def test_initial_pool_none(self):
        storage = PostgresAuditStorage(database_url="postgresql://localhost/test")
        assert storage._pool is None


# =============================================================================
# AuditInterceptor Tests
# =============================================================================


class TestAuditInterceptor:
    """Test AuditInterceptor core functionality."""

    @pytest.fixture(autouse=True)
    def reset_signing_key(self):
        """Ensure ephemeral signing key for tests."""
        _mod._INTERCEPTOR_SIGNING_KEY = None
        os.environ.pop("ARAGORA_ENV", None)
        os.environ.pop("ARAGORA_AUDIT_INTERCEPTOR_KEY", None)

    @pytest.fixture
    def storage(self):
        return InMemoryAuditStorage()

    @pytest.fixture
    def interceptor(self, storage):
        return AuditInterceptor(
            config=AuditConfig(
                retention_days=30,
                emit_events=False,
                pii_fields=["email", "ssn"],
            ),
            storage=storage,
        )

    # ---------------------------------------------------------------
    # Initialization tests
    # ---------------------------------------------------------------

    def test_default_config_and_storage(self):
        interceptor = AuditInterceptor()
        assert interceptor._config is not None
        assert isinstance(interceptor._storage, InMemoryAuditStorage)

    def test_custom_config(self):
        config = AuditConfig(retention_days=30)
        interceptor = AuditInterceptor(config=config)
        assert interceptor._config.retention_days == 30

    def test_custom_storage(self):
        storage = InMemoryAuditStorage(max_records=50)
        interceptor = AuditInterceptor(storage=storage)
        assert interceptor._storage is storage

    def test_postgres_storage_backend_requires_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("ARAGORA_POSTGRES_DSN", raising=False)
        config = AuditConfig(storage_backend="postgres")
        with pytest.raises(ValueError, match="PostgreSQL URL required"):
            AuditInterceptor(config=config)

    # ---------------------------------------------------------------
    # Intercept tests
    # ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_basic_intercept(self, interceptor, storage):
        record = await interceptor.intercept(
            request={"method": "POST", "path": "/api/users", "body": {"name": "Alice"}},
            response={"status": 200, "body": {"id": 1}},
            correlation_id="corr-1",
            user_id="user-1",
        )

        assert record.request_method == "POST"
        assert record.request_path == "/api/users"
        assert record.response_status == 200
        assert record.correlation_id == "corr-1"
        assert record.user_id == "user-1"
        assert record.record_hash != ""
        assert record.signature != ""

        stored = await storage.get(record.id)
        assert stored is not None

    @pytest.mark.asyncio
    async def test_intercept_generates_correlation_id(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        assert record.correlation_id != ""

    @pytest.mark.asyncio
    async def test_intercept_computes_body_hashes(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "POST", "body": {"data": "value"}},
            response={"status": 200, "body": {"result": "ok"}},
        )
        assert record.request_body_hash != ""
        assert record.response_body_hash != ""
        assert len(record.request_body_hash) == 64

    @pytest.mark.asyncio
    async def test_intercept_body_hash_none(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 204},
        )
        assert record.request_body_hash == ""
        assert record.response_body_hash == ""

    @pytest.mark.asyncio
    async def test_intercept_with_duration(self, interceptor):
        start = time.time() - 0.1  # 100ms ago
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/test"},
            response={"status": 200},
            start_time=start,
        )
        assert record.duration_ms > 0

    @pytest.mark.asyncio
    async def test_intercept_zero_duration_without_start_time(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        assert record.duration_ms == 0.0

    @pytest.mark.asyncio
    async def test_intercept_redacts_sensitive_headers(self, interceptor):
        record = await interceptor.intercept(
            request={
                "method": "GET",
                "path": "/test",
                "headers": {
                    "Authorization": "Bearer sk-xxx",
                    "Content-Type": "application/json",
                },
            },
            response={"status": 200},
        )
        assert record.request_headers.get("Authorization") == "[REDACTED]"
        assert record.request_headers.get("Content-Type") == "application/json"

    @pytest.mark.asyncio
    async def test_intercept_redacts_headers_case_insensitive(self, interceptor):
        record = await interceptor.intercept(
            request={
                "method": "GET",
                "path": "/api",
                "headers": {"AUTHORIZATION": "Bearer xyz", "X-Api-Key": "key"},
            },
            response={"status": 200},
        )
        assert record.request_headers["AUTHORIZATION"] == "[REDACTED]"
        assert record.request_headers["X-Api-Key"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_intercept_redacts_pii_in_body(self, interceptor):
        record = await interceptor.intercept(
            request={
                "method": "POST",
                "body": {"email": "alice@example.com", "name": "Alice"},
            },
            response={"status": 200},
        )
        assert record.request_body["email"] != "alice@example.com"
        assert record.request_body["name"] == "Alice"
        assert len(record.pii_fields_redacted) > 0

    @pytest.mark.asyncio
    async def test_intercept_redacts_password(self, interceptor):
        record = await interceptor.intercept(
            request={
                "method": "POST",
                "path": "/api",
                "body": {"password": "s3cret", "name": "Alice"},
            },
            response={"status": 200},
        )
        assert record.request_body["password"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_intercept_redacts_nested_pii(self, interceptor):
        record = await interceptor.intercept(
            request={
                "method": "POST",
                "path": "/api",
                "body": {"user": {"email": "alice@example.com", "name": "Alice"}},
            },
            response={"status": 200},
        )
        assert record.request_body["user"]["email"] != "alice@example.com"
        assert record.request_body["user"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_intercept_redacts_list_in_body(self, interceptor):
        record = await interceptor.intercept(
            request={
                "method": "POST",
                "path": "/api",
                "body": [
                    {"password": "pw1", "name": "Alice"},
                    {"password": "pw2", "name": "Bob"},
                ],
            },
            response={"status": 200},
        )
        assert record.request_body[0]["password"] == "[REDACTED]"
        assert record.request_body[1]["password"] == "[REDACTED]"
        assert record.request_body[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_intercept_redacts_response_body_pii(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={
                "status": 200,
                "body": {"token": "secret-token-value"},
            },
        )
        body = record.response_body
        assert body["token"].startswith("[HASH:")

    @pytest.mark.asyncio
    async def test_intercept_chain_continuity(self, interceptor, storage):
        r1 = await interceptor.intercept(
            request={"method": "GET"},
            response={"status": 200},
        )
        r2 = await interceptor.intercept(
            request={"method": "POST"},
            response={"status": 201},
        )
        assert r1.previous_hash == ""
        assert r2.previous_hash == r1.record_hash

    @pytest.mark.asyncio
    async def test_intercept_stores_metadata(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
            metadata={"tenant": "acme"},
        )
        assert record.metadata == {"tenant": "acme"}

    @pytest.mark.asyncio
    async def test_intercept_ip_and_user_agent(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )
        assert record.ip_address == "192.168.1.1"
        assert record.user_agent == "Mozilla/5.0"

    @pytest.mark.asyncio
    async def test_intercept_org_id(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
            org_id="org-42",
        )
        assert record.org_id == "org-42"

    @pytest.mark.asyncio
    async def test_intercept_empty_request(self, interceptor):
        record = await interceptor.intercept(request={}, response={})
        assert record.request_method == ""
        assert record.request_path == ""
        assert record.response_status == 0

    @pytest.mark.asyncio
    async def test_intercept_missing_headers(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        assert record.request_headers == {}

    @pytest.mark.asyncio
    async def test_intercept_empty_headers(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api", "headers": {}},
            response={"status": 200, "headers": {}},
        )
        assert record.request_headers == {}

    @pytest.mark.asyncio
    async def test_intercept_none_body(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api", "body": None},
            response={"status": 200, "body": None},
        )
        assert record.request_body is None
        assert record.response_body is None

    @pytest.mark.asyncio
    async def test_intercept_with_hash_responses(self, storage):
        config = AuditConfig(emit_events=False, hash_responses=True)
        interceptor = AuditInterceptor(config=config, storage=storage)
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200, "body": {"data": "secret"}},
        )
        assert record.response_body["_hashed"] is True
        assert "hash" in record.response_body

    @pytest.mark.asyncio
    async def test_intercept_body_truncation(self, storage):
        config = AuditConfig(emit_events=False, max_body_size=50)
        interceptor = AuditInterceptor(config=config, storage=storage)
        large_body = {"data": "x" * 200}
        record = await interceptor.intercept(
            request={"method": "POST", "path": "/api", "body": large_body},
            response={"status": 200},
        )
        assert record.request_body.get("_truncated") is True
        assert "_original_size" in record.request_body
        assert "_preview" in record.request_body

    @pytest.mark.asyncio
    async def test_intercept_body_no_truncation_when_unlimited(self, storage):
        config = AuditConfig(emit_events=False, max_body_size=0)
        interceptor = AuditInterceptor(config=config, storage=storage)
        large_body = {"data": "x" * 10000}
        record = await interceptor.intercept(
            request={"method": "POST", "path": "/api", "body": large_body},
            response={"status": 200},
        )
        assert record.request_body == large_body

    @pytest.mark.asyncio
    async def test_intercept_string_body_hash(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "POST", "path": "/api", "body": "plain-string"},
            response={"status": 200},
        )
        assert record.request_body_hash != ""

    @pytest.mark.asyncio
    async def test_intercept_list_body_hash(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "POST", "path": "/api", "body": [1, 2, 3]},
            response={"status": 200},
        )
        assert record.request_body_hash != ""

    @pytest.mark.asyncio
    async def test_intercept_updates_metrics(self, interceptor):
        await interceptor.intercept(
            request={"method": "GET"},
            response={"status": 200},
        )
        await interceptor.intercept(
            request={"method": "POST"},
            response={"status": 500},
        )

        metrics = interceptor.get_metrics()
        total_key = [k for k in metrics if k.endswith("_requests_total")][0]
        assert metrics[total_key] == 2

    @pytest.mark.asyncio
    async def test_intercept_with_custom_pii_rules(self, storage):
        config = AuditConfig(
            pii_rules=[
                PIIRedactionRule(
                    field_pattern=r"^phone$",
                    redaction_type=RedactionType.TRUNCATE,
                    visible_chars=3,
                ),
            ],
            emit_events=False,
        )
        interceptor = AuditInterceptor(config=config, storage=storage)
        record = await interceptor.intercept(
            request={
                "method": "POST",
                "path": "/api",
                "body": {"phone": "+1-555-123-4567"},
            },
            response={"status": 200},
        )
        phone_val = record.request_body["phone"]
        assert phone_val.startswith("+1-")
        assert "..." in phone_val

    @pytest.mark.asyncio
    async def test_intercept_sequential_chain_integrity(self, interceptor):
        """Multiple sequential intercepts should maintain chain integrity."""
        records = []
        for i in range(10):
            r = await interceptor.intercept(
                request={"method": "GET", "path": f"/{i}"},
                response={"status": 200},
            )
            records.append(r)

        for i in range(1, len(records)):
            assert records[i].previous_hash == records[i - 1].record_hash

    # ---------------------------------------------------------------
    # Verify chain tests
    # ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_verify_empty_chain(self, interceptor):
        is_valid, errors = await interceptor.verify_chain()
        assert is_valid is True
        assert errors == []

    @pytest.mark.asyncio
    async def test_verify_valid_chain(self, interceptor):
        for i in range(5):
            await interceptor.intercept(
                request={"method": "GET", "path": f"/test/{i}"},
                response={"status": 200},
            )

        is_valid, errors = await interceptor.verify_chain()
        assert is_valid is True
        assert errors == []

    @pytest.mark.asyncio
    async def test_verify_single_record(self, interceptor):
        await interceptor.intercept(
            request={"method": "GET", "path": "/test"},
            response={"status": 200},
        )
        is_valid, errors = await interceptor.verify_chain()
        assert is_valid is True
        assert errors == []

    @pytest.mark.asyncio
    async def test_verify_broken_chain_tampered_previous_hash(self, interceptor, storage):
        await interceptor.intercept(
            request={"method": "GET"},
            response={"status": 200},
        )
        r2 = await interceptor.intercept(
            request={"method": "POST"},
            response={"status": 201},
        )

        r2.previous_hash = "tampered"
        storage._records[r2.id] = r2

        is_valid, errors = await interceptor.verify_chain()
        assert is_valid is False
        assert len(errors) > 0
        assert "Chain broken" in errors[0]

    @pytest.mark.asyncio
    async def test_verify_tampered_hash(self, interceptor, storage):
        r = await interceptor.intercept(
            request={"method": "GET"},
            response={"status": 200},
        )

        r.record_hash = "tampered_hash"
        storage._records[r.id] = r

        is_valid, errors = await interceptor.verify_chain()
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_verify_chain_with_date_range(self, interceptor):
        for i in range(3):
            await interceptor.intercept(
                request={"method": "GET", "path": f"/{i}"},
                response={"status": 200},
            )
        is_valid, errors = await interceptor.verify_chain(
            since=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_chain_updates_metrics(self, interceptor):
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        await interceptor.verify_chain()
        metrics = interceptor.get_metrics()
        assert metrics[[k for k in metrics if "chain_verifications" in k][0]] == 1
        assert metrics[[k for k in metrics if "chain_errors" in k][0]] == 0

    @pytest.mark.asyncio
    async def test_verify_chain_error_increments_metric(self, interceptor, storage):
        r = await interceptor.intercept(
            request={"method": "GET"},
            response={"status": 200},
        )
        r.record_hash = "tampered"
        storage._records[r.id] = r

        await interceptor.verify_chain()
        metrics = interceptor.get_metrics()
        assert metrics[[k for k in metrics if "chain_errors" in k][0]] == 1

    # ---------------------------------------------------------------
    # Retention tests
    # ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_apply_retention(self, storage):
        interceptor = AuditInterceptor(
            config=AuditConfig(retention_days=7, emit_events=False),
            storage=storage,
        )

        now = datetime.now(timezone.utc)
        old = AuditRecord(
            id="old",
            timestamp=now - timedelta(days=30),
            request_method="GET",
        )
        recent = AuditRecord(
            id="recent",
            timestamp=now,
            request_method="GET",
        )
        await storage.store(old)
        await storage.store(recent)

        deleted = await interceptor.apply_retention()
        assert deleted == 1
        assert await storage.get("old") is None
        assert await storage.get("recent") is not None

    @pytest.mark.asyncio
    async def test_apply_retention_no_old_records(self, interceptor, storage):
        recent = AuditRecord(
            id="r1",
            timestamp=datetime.now(timezone.utc),
            record_hash="h1",
        )
        await storage.store(recent)
        deleted = await interceptor.apply_retention()
        assert deleted == 0

    # ---------------------------------------------------------------
    # SOC 2 export tests
    # ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_export_basic_structure(self, interceptor):
        now = datetime.now(timezone.utc)
        for i in range(3):
            await interceptor.intercept(
                request={"method": "GET", "path": f"/api/resource/{i}"},
                response={"status": 200},
                user_id=f"user-{i % 2}",
                ip_address=f"10.0.0.{i}",
            )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )

        assert report["report_type"] == "SOC 2 Type II Gateway Audit Evidence"
        assert "generated_at" in report
        assert "audit_period" in report
        assert "integrity" in report
        assert "summary" in report
        assert "breakdown" in report
        assert "control_evidence" in report
        assert "sample_records" in report

    @pytest.mark.asyncio
    async def test_export_summary_stats(self, interceptor):
        now = datetime.now(timezone.utc)
        await interceptor.intercept(
            request={"method": "POST", "path": "/api/users"},
            response={"status": 201},
            user_id="alice",
            ip_address="10.0.0.1",
        )
        await interceptor.intercept(
            request={"method": "GET", "path": "/api/users"},
            response={"status": 404},
            user_id="bob",
            ip_address="10.0.0.2",
        )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )

        summary = report["summary"]
        assert summary["total_requests"] == 2
        assert summary["unique_users"] == 2
        assert summary["unique_ips"] == 2
        assert summary["failed_requests"] == 1  # 404

    @pytest.mark.asyncio
    async def test_export_method_breakdown(self, interceptor):
        now = datetime.now(timezone.utc)
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        await interceptor.intercept(
            request={"method": "POST", "path": "/api"},
            response={"status": 201},
        )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )

        assert report["breakdown"]["by_method"]["GET"] == 1
        assert report["breakdown"]["by_method"]["POST"] == 1

    @pytest.mark.asyncio
    async def test_export_chain_integrity(self, interceptor):
        now = datetime.now(timezone.utc)
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )

        assert report["integrity"]["chain_verified"] is True
        assert report["integrity"]["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_export_control_evidence_access_control(self, interceptor):
        now = datetime.now(timezone.utc)
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
            user_id="alice",
        )
        await interceptor.intercept(
            request={"method": "GET", "path": "/public"},
            response={"status": 200},
        )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )

        cc61 = report["control_evidence"]["CC6.1_access_control"]
        assert cc61["requests_with_user_id"] == 1
        assert cc61["requests_without_user_id"] == 1

    @pytest.mark.asyncio
    async def test_export_control_evidence_audit_logging(self, interceptor):
        now = datetime.now(timezone.utc)
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )

        cc66 = report["control_evidence"]["CC6.6_audit_logging"]
        assert cc66["total_logged"] == 1
        assert cc66["chain_integrity"] is True

    @pytest.mark.asyncio
    async def test_export_control_evidence_monitoring(self, interceptor):
        now = datetime.now(timezone.utc)
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 500},
        )
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )

        cc72 = report["control_evidence"]["CC7.2_monitoring"]
        assert cc72["failed_requests"] == 1
        assert cc72["error_rate_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_export_with_org_filter(self, interceptor, storage):
        now = datetime.now(timezone.utc)
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
            org_id="org-1",
        )
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
            org_id="org-2",
        )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
            org_id="org-1",
        )

        assert report["organization"] == "org-1"
        assert report["summary"]["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_export_empty_period(self, interceptor):
        report = await interceptor.export_soc2_evidence(
            start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2020, 1, 2, tzinfo=timezone.utc),
        )
        assert report["summary"]["total_requests"] == 0
        assert report["summary"]["avg_duration_ms"] == 0

    @pytest.mark.asyncio
    async def test_export_sample_records_limited_to_10(self, interceptor):
        now = datetime.now(timezone.utc)
        for i in range(15):
            await interceptor.intercept(
                request={"method": "GET", "path": f"/{i}"},
                response={"status": 200},
            )

        report = await interceptor.export_soc2_evidence(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
        )
        assert len(report["sample_records"]) <= 10

    @pytest.mark.asyncio
    async def test_export_audit_period_days(self, interceptor):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 4, 1, tzinfo=timezone.utc)
        report = await interceptor.export_soc2_evidence(start_date=start, end_date=end)
        assert report["audit_period"]["days"] == 91

    # ---------------------------------------------------------------
    # Get record and query records tests
    # ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_record(self, interceptor):
        record = await interceptor.intercept(
            request={"method": "GET"},
            response={"status": 200},
        )

        result = await interceptor.get_record(record.id)
        assert result is not None
        assert result.id == record.id

    @pytest.mark.asyncio
    async def test_get_record_missing(self, interceptor):
        result = await interceptor.get_record("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_records(self, interceptor):
        await interceptor.intercept(
            request={"method": "GET"},
            response={"status": 200},
            user_id="alice",
        )
        await interceptor.intercept(
            request={"method": "POST"},
            response={"status": 201},
            user_id="bob",
        )

        results = await interceptor.query_records(user_id="alice")
        assert len(results) == 1
        assert results[0].user_id == "alice"

    @pytest.mark.asyncio
    async def test_query_records_with_all_filters(self, interceptor):
        now = datetime.now(timezone.utc)
        await interceptor.intercept(
            request={"method": "GET", "path": "/api/test"},
            response={"status": 200},
            correlation_id="corr-1",
            user_id="alice",
            org_id="org-1",
        )

        results = await interceptor.query_records(
            start_date=now - timedelta(hours=1),
            end_date=now + timedelta(hours=1),
            user_id="alice",
            org_id="org-1",
            correlation_id="corr-1",
            request_path="/api",
            limit=10,
            offset=0,
        )
        assert len(results) == 1

    # ---------------------------------------------------------------
    # Event handler tests
    # ---------------------------------------------------------------

    def test_add_event_handler(self):
        interceptor = AuditInterceptor(config=AuditConfig(emit_events=False))

        events = []

        def handler(event_type, data):
            events.append((event_type, data))

        interceptor.add_event_handler(handler)
        assert len(interceptor._event_handlers) == 1

    def test_remove_event_handler(self):
        interceptor = AuditInterceptor(config=AuditConfig(emit_events=False))

        def handler(event_type, data):
            pass

        interceptor.add_event_handler(handler)
        interceptor.remove_event_handler(handler)
        assert len(interceptor._event_handlers) == 0

    def test_remove_nonexistent_handler(self):
        interceptor = AuditInterceptor(config=AuditConfig(emit_events=False))

        def handler(event_type, data):
            pass

        interceptor.remove_event_handler(handler)  # Should not raise

    @pytest.mark.asyncio
    async def test_event_emitted_on_intercept(self, storage):
        config = AuditConfig(emit_events=True)
        interceptor = AuditInterceptor(config=config, storage=storage)
        events = []
        interceptor.add_event_handler(lambda t, d: events.append((t, d)))

        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        assert len(events) == 1
        assert events[0][0] == AuditEventType.RESPONSE_SENT

    @pytest.mark.asyncio
    async def test_event_data_contains_record_info(self, storage):
        config = AuditConfig(emit_events=True)
        interceptor = AuditInterceptor(config=config, storage=storage)
        event_data = []
        interceptor.add_event_handler(lambda t, d: event_data.append(d))

        await interceptor.intercept(
            request={"method": "POST", "path": "/api/users"},
            response={"status": 201},
            user_id="alice",
        )

        assert len(event_data) == 1
        data = event_data[0]
        assert "record_id" in data
        assert data["method"] == "POST"
        assert data["path"] == "/api/users"
        assert data["status"] == 201
        assert data["user_id"] == "alice"

    @pytest.mark.asyncio
    async def test_events_disabled_no_emission(self, interceptor):
        events = []
        interceptor.add_event_handler(lambda t, d: events.append(t))
        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_multiple_event_handlers(self, storage):
        config = AuditConfig(emit_events=True)
        interceptor = AuditInterceptor(config=config, storage=storage)
        events1, events2 = [], []

        interceptor.add_event_handler(lambda t, d: events1.append(t))
        interceptor.add_event_handler(lambda t, d: events2.append(t))

        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        assert len(events1) == 1
        assert len(events2) == 1

    @pytest.mark.asyncio
    async def test_event_handler_exception_does_not_propagate(self, storage):
        config = AuditConfig(emit_events=True)
        interceptor = AuditInterceptor(config=config, storage=storage)

        def bad_handler(event_type, data):
            raise RuntimeError("handler error")

        interceptor.add_event_handler(bad_handler)
        record = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        assert record is not None

    @pytest.mark.asyncio
    async def test_chain_verified_event(self, storage):
        config = AuditConfig(emit_events=True)
        interceptor = AuditInterceptor(config=config, storage=storage)
        events = []
        interceptor.add_event_handler(lambda t, d: events.append(t))

        await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        await interceptor.verify_chain()
        assert AuditEventType.CHAIN_VERIFIED in events

    @pytest.mark.asyncio
    async def test_chain_broken_event(self, storage):
        config = AuditConfig(emit_events=True)
        interceptor = AuditInterceptor(config=config, storage=storage)
        events = []
        interceptor.add_event_handler(lambda t, d: events.append(t))

        r = await interceptor.intercept(
            request={"method": "GET", "path": "/api"},
            response={"status": 200},
        )
        r.record_hash = "tampered"

        await interceptor.verify_chain()
        assert AuditEventType.CHAIN_BROKEN in events

    @pytest.mark.asyncio
    async def test_retention_applied_event(self, storage):
        config = AuditConfig(retention_days=1, emit_events=True)
        interceptor = AuditInterceptor(config=config, storage=storage)
        events = []
        interceptor.add_event_handler(lambda t, d: events.append(t))

        old = AuditRecord(
            id="old",
            timestamp=datetime.now(timezone.utc) - timedelta(days=10),
            record_hash="h1",
        )
        await storage.store(old)
        await interceptor.apply_retention()
        assert AuditEventType.RETENTION_APPLIED in events

    @pytest.mark.asyncio
    async def test_export_generated_event(self, storage):
        config = AuditConfig(emit_events=True)
        interceptor = AuditInterceptor(config=config, storage=storage)
        events = []
        interceptor.add_event_handler(lambda t, d: events.append(t))

        await interceptor.export_soc2_evidence(
            start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2030, 12, 31, tzinfo=timezone.utc),
        )
        assert AuditEventType.EXPORT_GENERATED in events

    # ---------------------------------------------------------------
    # Metrics tests
    # ---------------------------------------------------------------

    def test_initial_metrics(self):
        interceptor = AuditInterceptor(config=AuditConfig(emit_events=False))
        metrics = interceptor.get_metrics()
        assert any(k.endswith("_requests_total") for k in metrics)
        assert any(k.endswith("_pii_redactions_total") for k in metrics)
        assert any(k.endswith("_chain_verifications_total") for k in metrics)
        assert any(k.endswith("_chain_errors_total") for k in metrics)

    @pytest.mark.asyncio
    async def test_metrics_requests_by_status(self, interceptor):
        await interceptor.intercept(
            request={"method": "GET"},
            response={"status": 200},
        )
        await interceptor.intercept(
            request={"method": "POST"},
            response={"status": 400},
        )

        metrics = interceptor.get_metrics()
        by_status = metrics[[k for k in metrics if "by_status" in k][0]]
        assert by_status == {200: 1, 400: 1}

    @pytest.mark.asyncio
    async def test_metrics_pii_redactions(self, interceptor):
        await interceptor.intercept(
            request={
                "method": "POST",
                "path": "/api",
                "body": {"password": "secret", "email": "a@b.com"},
            },
            response={"status": 200},
        )
        metrics = interceptor.get_metrics()
        pii_key = [k for k in metrics if "pii_redactions" in k][0]
        assert metrics[pii_key] >= 2

    @pytest.mark.asyncio
    async def test_metrics_custom_prefix(self, storage):
        config = AuditConfig(metrics_prefix="custom_prefix", emit_events=False)
        interceptor = AuditInterceptor(config=config, storage=storage)
        metrics = interceptor.get_metrics()
        assert "custom_prefix_requests_total" in metrics

    # ---------------------------------------------------------------
    # Webhook tests
    # ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_webhook_task_created_on_intercept(self, storage):
        config = AuditConfig(
            webhook_url="https://siem.example.com/webhook",
            emit_events=False,
        )
        interceptor = AuditInterceptor(config=config, storage=storage)

        with patch("asyncio.create_task") as mock_task:
            await interceptor.intercept(
                request={"method": "GET", "path": "/api"},
                response={"status": 200},
            )
            mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_webhook_when_not_configured(self, interceptor):
        with patch("asyncio.create_task") as mock_task:
            await interceptor.intercept(
                request={"method": "GET", "path": "/api"},
                response={"status": 200},
            )
            mock_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_webhook_no_url_returns_early(self, interceptor):
        """_send_webhook should return early if webhook_url is None."""
        record = AuditRecord(id="r1")
        # Should not raise
        await interceptor._send_webhook(record)

    @pytest.mark.asyncio
    async def test_send_webhook_with_aiohttp_import_error(self, storage):
        config = AuditConfig(
            webhook_url="https://siem.example.com/webhook",
            emit_events=False,
        )
        interceptor = AuditInterceptor(config=config, storage=storage)
        record = AuditRecord(id="r1")

        with patch.dict("sys.modules", {"aiohttp": None}):
            # Should not raise - logs a warning instead
            await interceptor._send_webhook(record)
