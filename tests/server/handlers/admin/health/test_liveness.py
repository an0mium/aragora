"""Unit tests for the admin health liveness handler."""

from __future__ import annotations

import pytest

from aragora.server.handlers.base import HandlerResult, json_response
from aragora.server.handlers.admin.health import liveness
from aragora.server.handlers.admin.health.liveness import LivenessHandler


def test_init_without_context_creates_empty_context() -> None:
    handler = LivenessHandler()

    assert handler.ctx == {}


def test_init_without_context_uses_distinct_dicts() -> None:
    first = LivenessHandler()
    second = LivenessHandler()

    first.ctx["marker"] = "first"

    assert second.ctx == {}


def test_init_preserves_provided_context() -> None:
    ctx = {"storage": object()}

    handler = LivenessHandler(ctx)

    assert handler.ctx is ctx


def test_init_preserves_empty_context_identity() -> None:
    ctx: dict = {}

    handler = LivenessHandler(ctx)

    assert handler.ctx is ctx


def test_route_metadata_marks_healthz_public() -> None:
    assert LivenessHandler.ROUTES == ["/healthz"]
    assert LivenessHandler.PUBLIC_ROUTES == {"/healthz"}
    assert liveness.__all__ == ["LivenessHandler"]


def test_can_handle_matches_healthz_exactly() -> None:
    handler = LivenessHandler()

    assert handler.can_handle("/healthz")
    assert not handler.can_handle("/readyz")
    assert not handler.can_handle("/healthz/")


@pytest.mark.asyncio
async def test_handle_healthz_returns_liveness_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = json_response({"status": "ok", "source": "test"})
    calls: list[LivenessHandler] = []

    def fake_liveness_probe(handler: LivenessHandler) -> HandlerResult:
        calls.append(handler)
        return expected

    monkeypatch.setattr(liveness, "liveness_probe", fake_liveness_probe)
    handler = LivenessHandler()

    result = await handler.handle("/healthz", {"ignored": "value"}, object())

    assert result is expected
    assert calls == [handler]


@pytest.mark.asyncio
async def test_handle_unknown_path_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(handler: LivenessHandler) -> HandlerResult:
        raise AssertionError("liveness probe should not run")

    monkeypatch.setattr(liveness, "liveness_probe", fail_if_called)
    handler = LivenessHandler()

    assert await handler.handle("/readyz", {}, object()) is None


def test_liveness_probe_delegates_with_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = json_response({"status": "ok"})
    calls: list[LivenessHandler] = []

    def fake_liveness_probe(handler: LivenessHandler) -> HandlerResult:
        calls.append(handler)
        return expected

    monkeypatch.setattr(liveness, "liveness_probe", fake_liveness_probe)
    handler = LivenessHandler()

    assert handler._liveness_probe() is expected
    assert calls == [handler]


@pytest.mark.asyncio
async def test_handle_returns_handler_result_shape() -> None:
    result = await LivenessHandler().handle("/healthz", {}, object())

    assert result is not None
    assert result.status_code == 200
    assert result.content_type == "application/json"
    assert result.to_dict()["body"]["status"] == "ok"
