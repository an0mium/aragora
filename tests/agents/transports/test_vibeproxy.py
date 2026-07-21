from __future__ import annotations

import http.client
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


def test_empty_transport_mode_uses_direct_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_MODEL_TRANSPORT", "")

    policy = vibeproxy.ModelTransportPolicy.from_env()

    assert policy.mode is vibeproxy.TransportMode.DIRECT
    assert policy.client is None


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


def test_catalog_cache_isolated_by_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"first": 0, "second": 0}
    first = vibeproxy.VibeProxyClient("https://proxy.example", "first-key")
    second = vibeproxy.VibeProxyClient("https://proxy.example", "second-key")

    def first_open(*_args, **_kwargs):
        calls["first"] += 1
        return _Response(json.dumps({"data": [{"id": "first-model"}]}).encode())

    def second_open(*_args, **_kwargs):
        calls["second"] += 1
        return _Response(json.dumps({"data": [{"id": "second-model"}]}).encode())

    monkeypatch.setattr(first._opener, "open", first_open)
    monkeypatch.setattr(second._opener, "open", second_open)

    assert first.catalog().models == frozenset({"first-model"})
    assert second.catalog().models == frozenset({"second-model"})
    assert calls == {"first": 1, "second": 1}


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
    assert getattr(proxy, "proxies") == {}
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


@pytest.mark.parametrize(
    "error",
    [
        http.client.BadStatusLine("garbage status"),
        http.client.IncompleteRead(b"partial", 32),
    ],
)
def test_client_classifies_http_protocol_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: http.client.HTTPException,
) -> None:
    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(
        client._opener,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(vibeproxy.VibeProxyUnavailableError, match=type(error).__name__):
        client.catalog(force=True)


def test_client_times_out_slow_streaming_response(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}
    read_amounts: list[int] = []

    class SlowResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read1(self, amount: int) -> bytes:
            read_amounts.append(amount)
            clock["now"] += 0.6
            return b" "

    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(vibeproxy.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(client._opener, "open", lambda *_args, **_kwargs: SlowResponse())

    with pytest.raises(vibeproxy.VibeProxyTimeoutError):
        client._request("/models", timeout=1.0)

    assert read_amounts == [vibeproxy.RESPONSE_READ_CHUNK_BYTES] * 2


def test_client_accepts_response_with_bounded_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_amounts: list[int] = []

    class ReadOnlyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, amount: int) -> bytes:
            read_amounts.append(amount)
            return b'{"data": []}' if len(read_amounts) == 1 else b""

    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(client._opener, "open", lambda *_args, **_kwargs: ReadOnlyResponse())

    assert client._request("/models", timeout=1.0) == {"data": []}
    assert read_amounts == [vibeproxy.RESPONSE_READ_CHUNK_BYTES] * 2


def test_client_rejects_response_without_bounded_read_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(client._opener, "open", lambda *_args, **_kwargs: UnsupportedResponse())

    with pytest.raises(vibeproxy.VibeProxyUnavailableError, match="bounded reads"):
        client._request("/models", timeout=1.0)


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


@pytest.mark.parametrize(
    ("protocol", "path"),
    [
        (vibeproxy.OpenAIProtocol.CHAT, "/chat/completions"),
        (vibeproxy.OpenAIProtocol.RESPONSES, "/responses"),
    ],
)
def test_openai_protocol_request_uses_exact_model_and_path(
    monkeypatch: pytest.MonkeyPatch,
    protocol: vibeproxy.OpenAIProtocol,
    path: str,
) -> None:
    client = vibeproxy.VibeProxyClient()
    seen: dict[str, object] = {}

    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["authorization"] = request.headers["Authorization"]
        seen["anthropic_version"] = request.headers.get("Anthropic-version")
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return _Response(json.dumps({"model": "gpt-5.5", "ok": True}).encode())

    monkeypatch.setattr(client._opener, "open", fake_open)

    body = client.openai_request(
        protocol=protocol,
        model="gpt-5.5",
        payload={"model": "gpt-5.5", "input": "hello"},
        timeout=2.5,
    )

    assert body == {"model": "gpt-5.5", "ok": True}
    assert seen == {
        "url": f"http://127.0.0.1:8318/v1{path}",
        "authorization": f"Bearer {vibeproxy.LOCAL_API_KEY}",
        "anthropic_version": None,
        "payload": {"model": "gpt-5.5", "input": "hello"},
        "timeout": 2.5,
    }


def test_openai_protocol_request_rejects_payload_model_mismatch() -> None:
    client = vibeproxy.VibeProxyClient()

    with pytest.raises(vibeproxy.VibeProxyConfigurationError, match="payload model"):
        client.openai_request(
            protocol=vibeproxy.OpenAIProtocol.CHAT,
            model="gpt-5.5",
            payload={"model": "gpt-5.4"},
            timeout=1.0,
        )


def test_openai_protocol_request_rejects_response_model_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {"model": "gpt-5.4"},
    )

    with pytest.raises(vibeproxy.VibeProxyUnavailableError, match="requested model"):
        client.openai_request(
            protocol=vibeproxy.OpenAIProtocol.RESPONSES,
            model="gpt-5.5",
            payload={"model": "gpt-5.5", "input": "hello"},
            timeout=1.0,
        )


def test_anthropic_message_keeps_protocol_version_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = vibeproxy.VibeProxyClient()
    seen: dict[str, str | None] = {}

    def fake_open(request, timeout):
        seen["anthropic_version"] = request.headers.get("Anthropic-version")
        return _Response(
            json.dumps(
                {
                    "model": "claude-fable-5",
                    "content": [{"type": "text", "text": "answer"}],
                }
            ).encode()
        )

    monkeypatch.setattr(client._opener, "open", fake_open)

    assert client.anthropic_message(model="claude-fable-5", prompt="q", timeout=1.0) == "answer"
    assert seen["anthropic_version"] == "2023-06-01"


@pytest.mark.parametrize(
    "protocol", [vibeproxy.OpenAIProtocol.CHAT, vibeproxy.OpenAIProtocol.RESPONSES]
)
def test_openai_exact_protocols_are_policy_eligible(
    protocol: vibeproxy.OpenAIProtocol,
) -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.PREFER,
        client=_FakeClient({"gpt-5.5"}),  # type: ignore[arg-type]
    )

    route = policy.resolve("openai", "gpt-5.5", capabilities=(protocol.value,))

    assert route.transport == "vibeproxy"
    assert route.requested_model == route.resolved_model == "gpt-5.5"


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


@pytest.mark.parametrize("mode", [vibeproxy.TransportMode.PREFER, vibeproxy.TransportMode.REQUIRED])
def test_catalog_timeout_preserves_timeout_type_in_required_mode(
    mode: vibeproxy.TransportMode,
) -> None:
    class TimeoutClient:
        base_url = "http://127.0.0.1:8318/v1"

        def catalog(self):
            raise vibeproxy.VibeProxyTimeoutError("catalog timed out")

    policy = vibeproxy.ModelTransportPolicy(
        mode,
        client=TimeoutClient(),  # type: ignore[arg-type]
    )

    if mode is vibeproxy.TransportMode.REQUIRED:
        with pytest.raises(vibeproxy.VibeProxyTimeoutError, match="catalog timed out"):
            policy.resolve("anthropic", "claude-fable-5")
    else:
        route = policy.resolve("anthropic", "claude-fable-5")
        assert route.transport == "direct"
        assert route.fallback_reason == "catalog timed out"


def test_unsupported_web_search_uses_direct_in_prefer_mode() -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.PREFER,
        client=_FakeClient({"gpt-5.5"}),  # type: ignore[arg-type]
    )

    route = policy.resolve("openai", "gpt-5.5", capabilities=("chat", "web_search"))

    assert route.transport == "direct"
    assert route.fallback_reason == "unsupported capabilities: web_search"


@pytest.mark.parametrize("provider", ["grok", "gemini", "kimi"])
def test_unimplemented_provider_is_not_advertised(provider: str) -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.PREFER,
        client=_FakeClient({"model"}),  # type: ignore[arg-type]
    )

    route = policy.resolve(provider, "model")

    assert route.transport == "direct"
    assert route.fallback_reason == "unsupported capabilities: chat"


def test_anthropic_streaming_is_not_advertised() -> None:
    policy = vibeproxy.ModelTransportPolicy(
        vibeproxy.TransportMode.PREFER,
        client=_FakeClient({"claude-fable-5"}),  # type: ignore[arg-type]
    )

    route = policy.resolve("anthropic", "claude-fable-5", capabilities=("chat", "stream"))

    assert route.transport == "direct"
    assert route.fallback_reason == "unsupported capabilities: stream"


def test_anthropic_message_extracts_only_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "model": "claude-fable-5",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "answer"},
            ],
        },
    )

    assert client.anthropic_message(model="claude-fable-5", prompt="q", timeout=1) == "answer"


def test_anthropic_message_rejects_truncated_output(monkeypatch: pytest.MonkeyPatch) -> None:
    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "model": "claude-fable-5",
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "partial"}],
        },
    )

    with pytest.raises(vibeproxy.VibeProxyUnavailableError, match="truncated"):
        client.anthropic_message(model="claude-fable-5", prompt="q", timeout=1)


@pytest.mark.parametrize("response_model", [None, "claude-opus-4-8"])
def test_anthropic_message_rejects_response_model_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    response_model: str | None,
) -> None:
    client = vibeproxy.VibeProxyClient()
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {
            "model": response_model,
            "content": [{"type": "text", "text": "answer"}],
        },
    )

    with pytest.raises(vibeproxy.VibeProxyUnavailableError, match="requested model"):
        client.anthropic_message(model="claude-fable-5", prompt="q", timeout=1)
