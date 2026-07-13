"""Tests for the foundation client-IP extraction utility."""

from types import SimpleNamespace

from aragora.utils.request_ip import extract_client_ip


def _handler(ip=None, forwarded=""):
    client_address = (ip, 443) if ip is not None else None
    return SimpleNamespace(
        client_address=client_address,
        headers={"X-Forwarded-For": forwarded} if forwarded else {},
    )


def test_extract_client_ip_prefers_direct_connection(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_CIDRS", raising=False)
    assert extract_client_ip(_handler("203.0.113.10", "198.51.100.5")) == "203.0.113.10"


def test_extract_client_ip_trusts_forwarded_header_from_configured_proxy(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_CIDRS", "10.0.0.0/8")
    assert extract_client_ip(_handler("10.1.2.3", "198.51.100.5, 10.1.2.2")) == "198.51.100.5"


def test_extract_client_ip_uses_nearest_proxy_when_direct_ip_missing(monkeypatch):
    monkeypatch.delenv("TRUSTED_PROXY_CIDRS", raising=False)
    assert extract_client_ip(_handler(None, "198.51.100.5, 10.1.2.2")) == "10.1.2.2"


def test_extract_client_ip_handles_none():
    assert extract_client_ip(None) is None
