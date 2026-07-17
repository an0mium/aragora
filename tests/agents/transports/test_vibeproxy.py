from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from aragora.agents.transports import vibeproxy


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.fixture(autouse=True)
def _clear_catalog_cache():
    vibeproxy._CATALOG_CACHE.clear()
    yield
    vibeproxy._CATALOG_CACHE.clear()


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8318", "http://192.168.1.9:8318", "http://proxy.example:8318"],
)
def test_plaintext_requires_literal_loopback(url: str) -> None:
    with pytest.raises(vibeproxy.VibeProxyConfigurationError, match="literal loopback"):
        vibeproxy.VibeProxyClient(url)


def test_remote_https_requires_explicit_key() -> None:
    with pytest.raises(vibeproxy.VibeProxyConfigurationError, match="explicit API key"):
        vibeproxy.VibeProxyClient("https://proxy.example")


@pytest.mark.parametrize("url", ["http://127.0.0.1:8317", "http://127.0.0.1:bad"])
def test_rejects_prohibited_or_invalid_ports(url: str) -> None:
    with pytest.raises(vibeproxy.VibeProxyConfigurationError, match="port"):
        vibeproxy.VibeProxyClient(url)


def test_invalid_catalog_ttl_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_MODEL_TRANSPORT", "vibeproxy-prefer")
    monkeypatch.setenv("ARAGORA_VIBEPROXY_CATALOG_TTL_SECONDS", "nan")

    with pytest.raises(vibeproxy.VibeProxyConfigurationError, match="must be finite"):
        vibeproxy.ModelTransportPolicy.from_env()


def test_catalog_is_sanitized_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        assert request.full_url == "http://127.0.0.1:8318/v1/models"
        assert timeout == 1.5
        return _Response(json.dumps({"data": [{"id": "claude-fable-5"}]}).encode())

    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(client._opener, "open", fake_urlopen)

    first = client.sanitized_status()
    second = client.sanitized_status()

    assert first["models"] == ["claude-fable-5"]
    assert first["loopback"] is True
    assert "api_key" not in json.dumps(first)
    assert second == first
    assert calls == 1


def test_client_disables_environment_proxies_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: tuple[object, ...] = ()

    def fake_build_opener(*configured: object):
        nonlocal handlers
        handlers = configured
        return SimpleNamespace()

    monkeypatch.setattr(vibeproxy.urllib.request, "build_opener", fake_build_opener)
    vibeproxy.VibeProxyClient()

    proxy = next(
        handler
        for handler in handlers
        if isinstance(handler, vibeproxy.urllib.request.ProxyHandler)
    )
    assert proxy.proxies == {}
    assert any(isinstance(handler, vibeproxy._NoRedirectHandler) for handler in handlers)


def test_client_classifies_url_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(
        client._opener,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            vibeproxy.urllib.error.URLError(TimeoutError())
        ),
    )

    with pytest.raises(vibeproxy.VibeProxyTimeoutError):
        client.catalog(force=True)


class _FakeClient:
    base_url = "http://127.0.0.1:8318/v1"

    def __init__(self, models: set[str]):
        self.models = models

    def catalog(self):
        return SimpleNamespace(models=frozenset(self.models))


def test_prefer_resolves_exact_model() -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.PREFER,
        client=_FakeClient({"claude-fable-5"}),  # type: ignore[arg-type]
    )

    route = policy.resolve("anthropic", "claude-fable-5")

    assert route.transport == "vibeproxy"
    assert route.requested_model == route.resolved_model == "claude-fable-5"


def test_alias_is_explicit_and_observable() -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.PREFER,
        client=_FakeClient({"proxy-model"}),  # type: ignore[arg-type]
        model_map={"anthropic:logical-model": "proxy-model"},
    )

    route = policy.resolve("anthropic", "logical-model")

    assert route.requested_model == "logical-model"
    assert route.resolved_model == "proxy-model"


def test_prefer_falls_back_when_model_missing() -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.PREFER,
        client=_FakeClient(set()),  # type: ignore[arg-type]
    )

    route = policy.resolve("anthropic", "claude-fable-5")

    assert route.transport == "direct"
    assert route.fallback_reason == "model not in VibeProxy catalog: claude-fable-5"


def test_required_fails_closed_when_model_missing() -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.REQUIRED,
        client=_FakeClient(set()),  # type: ignore[arg-type]
    )

    with pytest.raises(vibeproxy.VibeProxyUnavailableError, match="model not"):
        policy.resolve("anthropic", "claude-fable-5")


def test_unsupported_web_search_uses_direct_in_prefer_mode() -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.PREFER,
        client=_FakeClient({"gpt-5.5"}),  # type: ignore[arg-type]
    )

    route = policy.resolve("openai", "gpt-5.5", capabilities=("chat", "web_search"))

    assert route.transport == "direct"
    assert route.fallback_reason == "unsupported capabilities: web_search"


def test_anthropic_message_extracts_only_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "answer"},
            ]
        },
    )

    assert client.anthropic_message(model="claude-fable-5", prompt="q", timeout=1) == "answer"


def test_anthropic_message_rejects_truncated_output(monkeypatch: pytest.MonkeyPatch) -> None:
    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "partial"}],
        },
    )

    with pytest.raises(vibeproxy.VibeProxyUnavailableError, match="truncated"):
        client.anthropic_message(model="claude-fable-5", prompt="q", timeout=1)
