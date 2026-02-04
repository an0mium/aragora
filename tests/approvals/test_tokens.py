import time

from aragora.approvals import tokens as tokens_module
from aragora.approvals.tokens import decode_approval_action, encode_approval_action
from aragora.approvals.chat import parse_chat_targets


def _reset_tokens_cache() -> None:
    tokens_module._SECRET = None
    tokens_module._SECRET_INSECURE = False


def test_encode_decode_roundtrip(monkeypatch):
    _reset_tokens_cache()
    monkeypatch.setenv("ARAGORA_APPROVAL_ACTION_SECRET", "test-secret")

    token = encode_approval_action(kind="workflow", target_id="apr_123", action="approve")
    assert token

    decoded = decode_approval_action(token)
    assert decoded is not None
    assert decoded.kind == "workflow"
    assert decoded.target_id == "apr_123"
    assert decoded.action == "approve"
    assert not decoded.is_expired


def test_decode_expired_token(monkeypatch):
    _reset_tokens_cache()
    monkeypatch.setenv("ARAGORA_APPROVAL_ACTION_SECRET", "test-secret")

    token = encode_approval_action(
        kind="workflow",
        target_id="apr_999",
        action="approve",
        issued_at=1,
        expires_at=2,
    )
    assert token

    decoded = decode_approval_action(token)
    assert decoded is None

    decoded_allow = decode_approval_action(token, allow_expired=True)
    assert decoded_allow is not None
    assert decoded_allow.is_expired


def test_decode_tampered_token(monkeypatch):
    _reset_tokens_cache()
    monkeypatch.setenv("ARAGORA_APPROVAL_ACTION_SECRET", "test-secret")

    token = encode_approval_action(kind="workflow", target_id="apr_123", action="approve")
    assert token

    tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
    decoded = decode_approval_action(tampered)
    assert decoded is None


def test_parse_chat_targets():
    mapping = parse_chat_targets(
        [
            "slack:#alerts",
            "teams:19:abc",
            "@owner",
            "user@example.com",
            "discord:12345",
        ]
    )

    assert mapping["slack"] == ["#alerts", "@owner"]
    assert mapping["teams"] == ["19:abc"]
    assert mapping["discord"] == ["12345"]
    assert "email" not in mapping
