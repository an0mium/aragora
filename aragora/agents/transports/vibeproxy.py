"""Secure, capability-aware routing for the local VibeProxy transport.

VibeProxy is a transport, not a provider family. Logical provider and model
identity remain attached to every resolved route.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import ipaddress
import json
import math
import os
import threading
import time
from typing import Any, Iterable
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8318"
DEFAULT_CATALOG_TTL_SECONDS = 60.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 1.5
LOCAL_API_KEY = "vibeproxy-local"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
PROHIBITED_PORTS = {8317}


class VibeProxyConfigurationError(ValueError):
    """Raised for an unsafe or malformed VibeProxy configuration."""


class VibeProxyUnavailableError(RuntimeError):
    """Raised when required VibeProxy routing cannot be satisfied."""


class TransportMode(str, Enum):
    DIRECT = "direct"
    PREFER = "vibeproxy-prefer"
    REQUIRED = "vibeproxy-required"


@dataclass(frozen=True)
class VibeProxyCatalog:
    models: frozenset[str]
    fetched_at: float


@dataclass(frozen=True)
class ResolvedModelRoute:
    provider: str
    requested_model: str
    resolved_model: str
    transport: str
    base_url: str | None
    capabilities: frozenset[str]
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "transport": self.transport,
            "base_url": self.base_url,
            "capabilities": sorted(self.capabilities),
            "fallback_reason": self.fallback_reason,
        }


_CATALOG_CACHE: dict[str, VibeProxyCatalog] = {}
_CATALOG_LOCK = threading.Lock()


def _bounded_float(value: str | int | float, *, name: str, minimum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VibeProxyConfigurationError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < minimum:
        raise VibeProxyConfigurationError(f"{name} must be finite and >= {minimum}")
    return result


def _normalize_base_url(raw: str, api_key: str | None) -> tuple[str, bool]:
    value = raw.strip().rstrip("/")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise VibeProxyConfigurationError("VibeProxy URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise VibeProxyConfigurationError(
            "VibeProxy URL must not contain credentials or query data"
        )
    if parsed.path not in {"", "/", "/v1"}:
        raise VibeProxyConfigurationError("VibeProxy URL path must be empty or /v1")
    try:
        port = parsed.port
    except ValueError as exc:
        raise VibeProxyConfigurationError("VibeProxy URL contains an invalid port") from exc
    if port in PROHIBITED_PORTS:
        raise VibeProxyConfigurationError(f"VibeProxy port {port} is not permitted")

    try:
        host = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        host = None
    is_loopback = bool(host and host.is_loopback)
    if parsed.scheme == "http" and not is_loopback:
        raise VibeProxyConfigurationError(
            "plaintext VibeProxy is allowed only on a literal loopback IP"
        )
    if not is_loopback and not api_key:
        raise VibeProxyConfigurationError("remote VibeProxy requires an explicit API key")

    root = value[:-3] if value.endswith("/v1") else value
    return root + "/v1", is_loopback


def _parse_model_map(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VibeProxyConfigurationError("ARAGORA_VIBEPROXY_MODEL_MAP must be valid JSON") from exc
    if not isinstance(value, dict):
        raise VibeProxyConfigurationError("ARAGORA_VIBEPROXY_MODEL_MAP must be a JSON object")
    result: dict[str, str] = {}
    for key, mapped in value.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(mapped, str)
            or not mapped.strip()
        ):
            raise VibeProxyConfigurationError("VibeProxy model-map keys and values must be strings")
        result[key.strip()] = mapped.strip()
    return result


class VibeProxyClient:
    """Small stdlib client that never exposes credentials in diagnostics."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        *,
        catalog_ttl_seconds: float = DEFAULT_CATALOG_TTL_SECONDS,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url, self.is_loopback = _normalize_base_url(base_url, api_key)
        self.api_key = api_key or (LOCAL_API_KEY if self.is_loopback else None)
        self.catalog_ttl_seconds = _bounded_float(
            catalog_ttl_seconds, name="catalog TTL", minimum=0.0
        )
        self.connect_timeout_seconds = _bounded_float(
            connect_timeout_seconds, name="connect timeout", minimum=0.1
        )

    def _request(
        self, path: str, *, timeout: float | None = None, payload: dict | None = None
    ) -> dict:
        headers = {"authorization": f"Bearer {self.api_key}"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers.update({"content-type": "application/json", "anthropic-version": "2023-06-01"})
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(
                request, timeout=timeout or self.connect_timeout_seconds
            ) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise VibeProxyUnavailableError(f"VibeProxy HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise VibeProxyUnavailableError(
                f"VibeProxy request failed: {type(exc).__name__}"
            ) from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise VibeProxyUnavailableError("VibeProxy response exceeded size limit")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise VibeProxyUnavailableError("VibeProxy returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise VibeProxyUnavailableError("VibeProxy returned a non-object response")
        return body

    def catalog(self, *, force: bool = False) -> VibeProxyCatalog:
        now = time.monotonic()
        with _CATALOG_LOCK:
            cached = _CATALOG_CACHE.get(self.base_url)
            if cached and not force and now - cached.fetched_at < self.catalog_ttl_seconds:
                return cached
        body = self._request("/models")
        entries = body.get("data")
        if not isinstance(entries, list):
            raise VibeProxyUnavailableError("VibeProxy model catalog is malformed")
        models = frozenset(
            item["id"].strip()
            for item in entries
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
        )
        if not models:
            raise VibeProxyUnavailableError("VibeProxy model catalog is empty")
        catalog = VibeProxyCatalog(models=models, fetched_at=now)
        with _CATALOG_LOCK:
            _CATALOG_CACHE[self.base_url] = catalog
        return catalog

    def anthropic_message(
        self,
        *,
        model: str,
        prompt: str,
        timeout: float,
        system: str | None = None,
        max_tokens: int = 8192,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        body = self._request("/messages", timeout=timeout, payload=payload)
        content = body.get("content")
        if not isinstance(content, list):
            raise VibeProxyUnavailableError("VibeProxy returned no Claude content")
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not text:
            raise VibeProxyUnavailableError("VibeProxy returned no Claude text")
        return text

    def sanitized_status(self, *, force: bool = False) -> dict[str, Any]:
        try:
            catalog = self.catalog(force=force)
        except VibeProxyUnavailableError as exc:
            return {
                "ok": False,
                "base_url": self.base_url,
                "loopback": self.is_loopback,
                "error": str(exc),
            }
        return {
            "ok": True,
            "base_url": self.base_url,
            "loopback": self.is_loopback,
            "model_count": len(catalog.models),
            "models": sorted(catalog.models),
        }


class ModelTransportPolicy:
    """Resolve an exact logical model onto VibeProxy or the direct path."""

    CAPABILITIES: dict[str, frozenset[str]] = {
        "anthropic": frozenset({"chat", "stream"}),
        "openai": frozenset({"chat", "stream", "responses"}),
        "grok": frozenset({"chat", "stream"}),
        "gemini": frozenset({"chat", "stream"}),
        "kimi": frozenset({"chat", "stream"}),
    }

    def __init__(
        self,
        mode: TransportMode = TransportMode.DIRECT,
        *,
        client: VibeProxyClient | None = None,
        model_map: dict[str, str] | None = None,
    ) -> None:
        self.mode = mode
        self.client = client
        self.model_map = dict(model_map or {})

    @classmethod
    def from_env(
        cls, *, default_mode: TransportMode = TransportMode.DIRECT
    ) -> "ModelTransportPolicy":
        raw_mode = os.environ.get("ARAGORA_MODEL_TRANSPORT", default_mode.value).strip()
        try:
            mode = TransportMode(raw_mode)
        except ValueError as exc:
            raise VibeProxyConfigurationError(
                f"invalid ARAGORA_MODEL_TRANSPORT: {raw_mode}"
            ) from exc
        if mode is TransportMode.DIRECT:
            return cls(mode)
        ttl = _bounded_float(
            os.environ.get("ARAGORA_VIBEPROXY_CATALOG_TTL_SECONDS", DEFAULT_CATALOG_TTL_SECONDS),
            name="ARAGORA_VIBEPROXY_CATALOG_TTL_SECONDS",
            minimum=0.0,
        )
        client = VibeProxyClient(
            os.environ.get("ARAGORA_VIBEPROXY_BASE_URL", DEFAULT_BASE_URL),
            os.environ.get("ARAGORA_VIBEPROXY_API_KEY") or None,
            catalog_ttl_seconds=ttl,
        )
        return cls(
            mode,
            client=client,
            model_map=_parse_model_map(os.environ.get("ARAGORA_VIBEPROXY_MODEL_MAP")),
        )

    def resolve(
        self,
        provider: str,
        model: str,
        capabilities: Iterable[str] = ("chat",),
    ) -> ResolvedModelRoute:
        requested = frozenset(capabilities)
        direct = ResolvedModelRoute(provider, model, model, "direct", None, requested)
        if self.mode is TransportMode.DIRECT:
            return direct
        supported = self.CAPABILITIES.get(provider, frozenset())
        unsupported = requested - supported
        if unsupported:
            return self._unavailable(
                direct, "unsupported capabilities: " + ", ".join(sorted(unsupported))
            )
        if self.client is None:
            return self._unavailable(direct, "VibeProxy client is not configured")
        mapped = self.model_map.get(f"{provider}:{model}", self.model_map.get(model, model))
        try:
            catalog = self.client.catalog()
        except VibeProxyUnavailableError as exc:
            return self._unavailable(direct, str(exc))
        if mapped not in catalog.models:
            return self._unavailable(direct, f"model not in VibeProxy catalog: {mapped}")
        return ResolvedModelRoute(
            provider, model, mapped, "vibeproxy", self.client.base_url, requested
        )

    def _unavailable(self, direct: ResolvedModelRoute, reason: str) -> ResolvedModelRoute:
        if self.mode is TransportMode.REQUIRED:
            raise VibeProxyUnavailableError(reason)
        return ResolvedModelRoute(
            direct.provider,
            direct.requested_model,
            direct.resolved_model,
            "direct",
            None,
            direct.capabilities,
            fallback_reason=reason,
        )


__all__ = [
    "DEFAULT_BASE_URL",
    "ModelTransportPolicy",
    "ResolvedModelRoute",
    "TransportMode",
    "VibeProxyClient",
    "VibeProxyConfigurationError",
    "VibeProxyUnavailableError",
]
