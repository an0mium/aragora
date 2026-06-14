"""Unit tests for aragora/server/handlers/bots/slack/signature.py (B0-cohort #5184).

Covers the full public API surface:

- ``compute_slack_signature``
  - Known-answer HMAC-SHA256 vector using Slack's documented v0 basestring
    format (``v0:{timestamp}:{body}``)
  - Independently computed HMAC cross-check
  - TypeError on non-bytes body
  - ValueError on empty timestamp / empty signing secret
  - Unicode body handling

- ``verify_slack_signature``
  - Valid/invalid signature round-trips (deterministic via the ``now=``
    injection parameter -- no wall clock reads)
  - Replay window enforcement at exact boundaries of
    ``TIMESTAMP_TOLERANCE_SECONDS`` (past and future)
  - Negative ``now`` rejection (fail-closed)
  - Malformed/missing timestamp headers
  - Missing signature / missing signing secret (fail-closed)
  - Non-UTF-8 body rejection (UnicodeDecodeError path)
  - Timing-safe comparison via ``hmac.compare_digest``

All tests are deterministic: time is either injected through ``now=`` or
frozen via monkeypatch where the module would read ``time.time()``.
No external services or network calls are involved.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

from aragora.server.handlers.bots.slack import signature as signature_module
from aragora.server.handlers.bots.slack.signature import (
    TIMESTAMP_TOLERANCE_SECONDS,
    compute_slack_signature,
    verify_slack_signature,
)

# ---------------------------------------------------------------------------
# Fixed, deterministic inputs
# ---------------------------------------------------------------------------

SECRET = "test-signing-secret-0123456789"
NOW = 1700000000  # frozen reference clock
TS = str(NOW)
BODY = b'{"type":"event_callback","event":{"type":"app_mention"}}'

# Known-answer vector using Slack's documented signing-secret example.
# Independently computed with hashlib/hmac over the v0 basestring
# "v0:{timestamp}:{body}".
SLACK_DOC_SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
SLACK_DOC_TIMESTAMP = "1531420618"
SLACK_DOC_BODY = (
    b"token=xyzz0WbapA4vBCDEFasx0q6G&team_id=T1DC2JH3J&team_domain=testteamnow"
    b"&channel_id=G8PSS9T3V&channel_name=foobar&user_id=U2CERLKJA"
    b"&user_name=roadrunner&command=%2Fwebhook-collect&text="
    b"&response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands%2FT1DC2JH3J"
    b"%2F397700885554%2F96rGlfmibIGlgcZRskXaIFfN"
    b"&trigger_id=398738663015.47445629121.803a0bc887a14d10d2c447fce8b6703c"
)
SLACK_DOC_EXPECTED = "v0=a2114d57b48eac39b9ad189dd8316235a7b4a8d21a10bd27519666489c69b503"


def _reference_signature(body: bytes, timestamp: str, secret: str) -> str:
    """Independent HMAC-SHA256 implementation of the v0 signing scheme."""
    basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    digest = hmac.new(secret.encode(), basestring.encode(), hashlib.sha256).hexdigest()
    return f"v0={digest}"


@pytest.fixture
def valid_signature() -> str:
    """A valid signature for the canonical fixed inputs."""
    return compute_slack_signature(BODY, TS, SECRET)


# ---------------------------------------------------------------------------
# compute_slack_signature
# ---------------------------------------------------------------------------


class TestComputeSlackSignature:
    """Tests for the signature computation primitive."""

    def test_known_answer_vector(self):
        """Computation must reproduce the precomputed HMAC-SHA256 vector."""
        result = compute_slack_signature(SLACK_DOC_BODY, SLACK_DOC_TIMESTAMP, SLACK_DOC_SECRET)
        assert result == SLACK_DOC_EXPECTED

    def test_matches_independent_hmac_implementation(self):
        """Computation must match an HMAC built independently in the test."""
        assert compute_slack_signature(BODY, TS, SECRET) == _reference_signature(BODY, TS, SECRET)

    def test_v0_prefix_and_hex_digest_shape(self):
        """Result must be 'v0=' + 64 lowercase hex chars (SHA-256)."""
        sig = compute_slack_signature(BODY, TS, SECRET)
        assert sig.startswith("v0=")
        digest = sig[3:]
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_uses_v0_basestring_format(self):
        """Changing any basestring component must change the signature."""
        base = compute_slack_signature(BODY, TS, SECRET)
        assert compute_slack_signature(b"other-body", TS, SECRET) != base
        assert compute_slack_signature(BODY, str(NOW + 1), SECRET) != base
        assert compute_slack_signature(BODY, TS, SECRET + "x") != base

    def test_non_bytes_body_raises_type_error(self):
        """A str body must raise TypeError (API requires raw bytes)."""
        with pytest.raises(TypeError, match="body must be bytes"):
            compute_slack_signature("not-bytes", TS, SECRET)  # type: ignore[arg-type]

    def test_empty_timestamp_raises_value_error(self):
        with pytest.raises(ValueError, match="timestamp must not be empty"):
            compute_slack_signature(BODY, "", SECRET)

    def test_empty_signing_secret_raises_value_error(self):
        with pytest.raises(ValueError, match="signing_secret must not be empty"):
            compute_slack_signature(BODY, TS, "")

    def test_empty_body_is_allowed(self):
        """An empty (but bytes) body is valid and signs the 'v0:{ts}:' string."""
        assert compute_slack_signature(b"", TS, SECRET) == _reference_signature(b"", TS, SECRET)

    def test_unicode_body(self):
        """UTF-8 multibyte bodies must sign their decoded text form."""
        body = "{'text': 'café ünïcode ✓'}".encode("utf-8")
        assert compute_slack_signature(body, TS, SECRET) == _reference_signature(body, TS, SECRET)


# ---------------------------------------------------------------------------
# verify_slack_signature: happy path and signature mismatch (frozen clock)
# ---------------------------------------------------------------------------


class TestVerifySignature:
    """Deterministic verification tests using the now= injection parameter."""

    def test_valid_signature_accepted(self, valid_signature):
        assert verify_slack_signature(BODY, TS, valid_signature, SECRET, now=NOW) is True

    def test_wrong_signature_rejected(self):
        bogus = "v0=" + "0" * 64
        assert verify_slack_signature(BODY, TS, bogus, SECRET, now=NOW) is False

    def test_signature_from_wrong_secret_rejected(self):
        sig = compute_slack_signature(BODY, TS, "attacker-secret")
        assert verify_slack_signature(BODY, TS, sig, SECRET, now=NOW) is False

    def test_tampered_body_rejected(self, valid_signature):
        assert verify_slack_signature(b"tampered", TS, valid_signature, SECRET, now=NOW) is False

    def test_tampered_timestamp_rejected(self, valid_signature):
        """A replayed signature with a shifted timestamp must fail HMAC."""
        shifted_ts = str(NOW + 10)
        assert verify_slack_signature(BODY, shifted_ts, valid_signature, SECRET, now=NOW) is False

    def test_truncated_signature_rejected(self, valid_signature):
        assert verify_slack_signature(BODY, TS, valid_signature[:-4], SECRET, now=NOW) is False

    def test_signature_without_v0_prefix_rejected(self, valid_signature):
        assert verify_slack_signature(BODY, TS, valid_signature[3:], SECRET, now=NOW) is False

    def test_wall_clock_default_used_when_now_omitted(self, monkeypatch, valid_signature):
        """When now is omitted, the module reads time.time(); freeze it."""
        monkeypatch.setattr(signature_module.time, "time", lambda: float(NOW))
        assert verify_slack_signature(BODY, TS, valid_signature, SECRET) is True


# ---------------------------------------------------------------------------
# Replay window / timestamp staleness
# ---------------------------------------------------------------------------


class TestReplayWindow:
    """Replay-attack prevention via the +/- TIMESTAMP_TOLERANCE_SECONDS window."""

    def test_tolerance_constant_is_five_minutes(self):
        assert TIMESTAMP_TOLERANCE_SECONDS == 300

    @pytest.mark.parametrize(
        "offset",
        [0, -1, 1, -TIMESTAMP_TOLERANCE_SECONDS, TIMESTAMP_TOLERANCE_SECONDS],
        ids=["now", "1s-old", "1s-future", "exact-past-boundary", "exact-future-boundary"],
    )
    def test_timestamps_within_window_accepted(self, offset):
        ts = str(NOW + offset)
        sig = compute_slack_signature(BODY, ts, SECRET)
        assert verify_slack_signature(BODY, ts, sig, SECRET, now=NOW) is True

    @pytest.mark.parametrize(
        "offset",
        [
            -(TIMESTAMP_TOLERANCE_SECONDS + 1),
            TIMESTAMP_TOLERANCE_SECONDS + 1,
            -86400,
            86400,
        ],
        ids=["1s-past-boundary", "1s-future-boundary", "1-day-old", "1-day-future"],
    )
    def test_timestamps_outside_window_rejected(self, offset):
        """Stale or far-future timestamps must be rejected even with valid HMAC."""
        ts = str(NOW + offset)
        sig = compute_slack_signature(BODY, ts, SECRET)
        assert verify_slack_signature(BODY, ts, sig, SECRET, now=NOW) is False

    def test_replay_outside_window_logs_warning(self, caplog):
        import logging

        ts = str(NOW - TIMESTAMP_TOLERANCE_SECONDS - 1)
        sig = compute_slack_signature(BODY, ts, SECRET)
        with caplog.at_level(logging.WARNING):
            verify_slack_signature(BODY, ts, sig, SECRET, now=NOW)
        assert any("timestamp too old" in msg for msg in caplog.messages)

    def test_negative_now_rejected(self, valid_signature):
        """A negative injected clock must fail closed."""
        assert verify_slack_signature(BODY, TS, valid_signature, SECRET, now=-1) is False

    def test_negative_request_timestamp_rejected(self):
        """A negative request timestamp parses but falls outside the window."""
        sig = compute_slack_signature(BODY, "-100", SECRET)
        assert verify_slack_signature(BODY, "-100", sig, SECRET, now=NOW) is False


# ---------------------------------------------------------------------------
# Malformed / missing inputs (fail-closed behavior)
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    """All malformed or missing inputs must return False, never raise."""

    @pytest.mark.parametrize(
        "bad_ts",
        ["", "abc", "12.5", "  ", "0x10", "1e9", None],
        ids=["empty", "alpha", "float", "whitespace", "hex", "scientific", "none"],
    )
    def test_unparseable_timestamp_rejected(self, bad_ts):
        assert verify_slack_signature(BODY, bad_ts, "v0=" + "a" * 64, SECRET, now=NOW) is False

    def test_missing_signature_rejected(self):
        assert verify_slack_signature(BODY, TS, "", SECRET, now=NOW) is False

    def test_none_signature_rejected(self):
        assert verify_slack_signature(BODY, TS, None, SECRET, now=NOW) is False

    def test_missing_signing_secret_rejected(self, valid_signature):
        """Without a configured secret, verification must fail closed."""
        assert verify_slack_signature(BODY, TS, valid_signature, "", now=NOW) is False

    def test_none_signing_secret_rejected(self, valid_signature):
        assert verify_slack_signature(BODY, TS, valid_signature, None, now=NOW) is False

    def test_missing_inputs_log_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            verify_slack_signature(BODY, TS, "", SECRET, now=NOW)
        assert any("Missing Slack signature" in msg for msg in caplog.messages)

    def test_non_utf8_body_rejected(self):
        """Bytes that cannot decode as UTF-8 must be rejected, not raise."""
        invalid_utf8 = b"\xff\xfe\x80\x81"
        assert verify_slack_signature(invalid_utf8, TS, "v0=" + "a" * 64, SECRET, now=NOW) is False

    def test_non_utf8_body_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            verify_slack_signature(b"\xff\xfe", TS, "v0=" + "a" * 64, SECRET, now=NOW)
        assert any("body encoding" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Timing-safe comparison
# ---------------------------------------------------------------------------


class TestTimingSafeComparison:
    """The final digest comparison must go through hmac.compare_digest."""

    def test_compare_digest_invoked_with_expected_and_provided(self, monkeypatch, valid_signature):
        calls: list[tuple[str, str]] = []
        real_compare = hmac.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real_compare(a, b)

        monkeypatch.setattr(signature_module.hmac, "compare_digest", spy)
        assert verify_slack_signature(BODY, TS, valid_signature, SECRET, now=NOW) is True
        assert calls == [(valid_signature, valid_signature)]

    def test_compare_digest_result_is_returned(self, monkeypatch, valid_signature):
        """verify must return exactly what the timing-safe compare decides."""
        monkeypatch.setattr(signature_module.hmac, "compare_digest", lambda a, b: False)
        assert verify_slack_signature(BODY, TS, valid_signature, SECRET, now=NOW) is False

    def test_no_early_exit_before_compare_for_well_formed_input(self, monkeypatch):
        """A wrong-but-well-formed signature still reaches the timing-safe compare."""
        reached = []

        def spy(a, b):
            reached.append(True)
            return False

        monkeypatch.setattr(signature_module.hmac, "compare_digest", spy)
        bogus = "v0=" + "f" * 64
        assert verify_slack_signature(BODY, TS, bogus, SECRET, now=NOW) is False
        assert reached == [True]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_all_exports(self):
        assert signature_module.__all__ == [
            "TIMESTAMP_TOLERANCE_SECONDS",
            "compute_slack_signature",
            "verify_slack_signature",
        ]

    def test_exports_resolve(self):
        for name in signature_module.__all__:
            assert getattr(signature_module, name) is not None
