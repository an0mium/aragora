from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from aragora.agents.transports.claude_vibeproxy import (
    VIBEPROXY_HARNESS,
    VIBEPROXY_TIMEOUT_ENV,
    run_claude_vibeproxy,
)
from aragora.agents.transports.vibeproxy import (
    ModelTransportPolicy,
    TransportMode,
    VibeProxyCatalog,
)


@dataclass
class FakeClient:
    models: frozenset[str] = frozenset({"claude-opus-4-8"})
    base_url: str = "http://127.0.0.1:8318/v1"
    calls: list[dict[str, object]] = field(default_factory=list)

    def catalog(self) -> VibeProxyCatalog:
        return VibeProxyCatalog(models=self.models, fetched_at=0.0)

    def anthropic_message(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return "Verdict: PASS"


def test_direct_mode_does_not_touch_vibeproxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VIBEPROXY_TIMEOUT_ENV, "invalid-but-unused")
    client = FakeClient()
    policy = ModelTransportPolicy(TransportMode.DIRECT, client=client)

    result = run_claude_vibeproxy("prompt", reviewer_timeout=600, policy=policy)

    assert result.attempted is False
    assert result.ok is False
    assert client.calls == []


def test_prefer_mode_runs_exact_model_and_discloses_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VIBEPROXY_TIMEOUT_ENV, "30")
    client = FakeClient()
    policy = ModelTransportPolicy(TransportMode.PREFER, client=client)

    result = run_claude_vibeproxy("review prompt", reviewer_timeout=600, policy=policy)

    assert result.ok is True
    assert result.required is False
    assert result.text == "Verdict: PASS"
    assert result.harness == f"{VIBEPROXY_HARNESS} (model: claude-opus-4-8)"
    assert result.timeout_seconds == 30.0
    assert client.calls == [
        {
            "model": "claude-opus-4-8",
            "prompt": "review prompt",
            "timeout": 30.0,
        }
    ]


def test_prefer_mode_reserves_half_the_reviewer_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VIBEPROXY_TIMEOUT_ENV, "500")
    client = FakeClient()
    policy = ModelTransportPolicy(TransportMode.PREFER, client=client)

    result = run_claude_vibeproxy("prompt", reviewer_timeout=80, policy=policy)

    assert result.ok is True
    assert result.timeout_seconds == 40.0
    assert client.calls[0]["timeout"] == 40.0


def test_prefer_mode_allows_direct_fallback_when_model_is_unavailable() -> None:
    policy = ModelTransportPolicy(
        TransportMode.PREFER,
        client=FakeClient(models=frozenset()),
    )

    result = run_claude_vibeproxy("prompt", reviewer_timeout=600, policy=policy)

    assert result.attempted is True
    assert result.required is False
    assert result.ok is False
    assert result.error == "model not in VibeProxy catalog: claude-opus-4-8"


def test_required_mode_fails_closed_when_model_is_unavailable() -> None:
    policy = ModelTransportPolicy(
        TransportMode.REQUIRED,
        client=FakeClient(models=frozenset()),
    )

    result = run_claude_vibeproxy("prompt", reviewer_timeout=600, policy=policy)

    assert result.attempted is True
    assert result.required is True
    assert result.ok is False
    assert result.error == "model not in VibeProxy catalog: claude-opus-4-8"


def test_invalid_timeout_degrades_to_default_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    # A malformed timeout env is a tuning typo, not a transport-intent signal:
    # it must NOT escalate prefer -> required or suppress fallback. It degrades
    # to the default budget instead (min(120 default, 600/2 half) == 120).
    monkeypatch.setenv(VIBEPROXY_TIMEOUT_ENV, "not-a-number")
    policy = ModelTransportPolicy(TransportMode.PREFER, client=FakeClient())

    result = run_claude_vibeproxy("prompt", reviewer_timeout=600, policy=policy)

    assert result.required is False
    assert result.ok is True
    assert result.timeout_seconds == 120.0


def test_invalid_timeout_in_required_mode_still_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VIBEPROXY_TIMEOUT_ENV, "120s")  # unit suffix -> unparseable
    policy = ModelTransportPolicy(TransportMode.REQUIRED, client=FakeClient())

    result = run_claude_vibeproxy("prompt", reviewer_timeout=600, policy=policy)

    assert result.required is True
    assert result.ok is True
    assert result.timeout_seconds == 120.0


def test_bad_transport_env_degrades_to_direct_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # from_env() fails on a typo'd ARAGORA_MODEL_TRANSPORT; with no explicit
    # required intent the Claude leg degrades to direct (attempted=False) rather
    # than failing closed with all fallbacks suppressed.
    monkeypatch.setenv("ARAGORA_MODEL_TRANSPORT", "vibeproxy-prefre")  # typo

    result = run_claude_vibeproxy("prompt", reviewer_timeout=600, policy=None)

    assert result.attempted is False
    assert result.required is False


def test_bad_transport_env_fails_closed_only_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_MODEL_TRANSPORT", "vibeproxy-required")
    monkeypatch.setenv("ARAGORA_VIBEPROXY_MODEL_MAP", "{bad")  # from_env raises

    result = run_claude_vibeproxy("prompt", reviewer_timeout=600, policy=None)

    assert result.attempted is True
    assert result.required is True
    assert result.ok is False
