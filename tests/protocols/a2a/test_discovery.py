"""Tests for aragora.protocols.a2a.discovery (AGT-02 capability discovery)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.protocols.a2a.discovery import (
    AgentCapabilitySummary,
    DiscoveryError,
    PlatformCapability,
    agent_catalog,
    discovery_enabled,
    platform_catalog,
)
from aragora.protocols.a2a.types import AgentCapability


def _server(agent_id: str | None = None, caps: list[str] | None = None):
    s = MagicMock()
    if agent_id is None:
        s.get_agent.return_value = None
    else:
        card = MagicMock()
        card.capabilities = caps or []
        s.get_agent.return_value = card
    return s


# -- feature gate ------------------------------------------------------------


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("ARAGORA_A2A_DISCOVERY_ENABLED", raising=False)
    assert discovery_enabled() is False


@pytest.mark.parametrize("v", ["1", "true", "yes", "on"])
def test_gate_truthy(monkeypatch, v):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", v)
    assert discovery_enabled() is True


@pytest.mark.parametrize("v", ["0", "false", ""])
def test_gate_falsy(monkeypatch, v):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", v)
    assert discovery_enabled() is False


def test_platform_catalog_raises_when_off(monkeypatch):
    monkeypatch.delenv("ARAGORA_A2A_DISCOVERY_ENABLED", raising=False)
    with pytest.raises(DiscoveryError):
        platform_catalog()


def test_agent_catalog_raises_when_off(monkeypatch):
    monkeypatch.delenv("ARAGORA_A2A_DISCOVERY_ENABLED", raising=False)
    with pytest.raises(DiscoveryError):
        agent_catalog("x")


# -- platform_catalog --------------------------------------------------------


def test_catalog_contains_core_ids(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    ids = {c.capability_id for c in platform_catalog()}
    assert {"aragora.debate.run", "aragora.receipt.fetch", "aragora.agent.register"} <= ids


def test_catalog_sorted_by_id(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    ids = [c.capability_id for c in platform_catalog()]
    assert ids == sorted(ids)


def test_catalog_category_filter(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    caps = platform_catalog(category="registry")
    assert caps and all(c.category == "registry" for c in caps)


def test_catalog_unknown_category_empty(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    assert platform_catalog(category="nonexistent") == []


def test_catalog_bypasses_gate(monkeypatch):
    monkeypatch.delenv("ARAGORA_A2A_DISCOVERY_ENABLED", raising=False)
    assert len(platform_catalog(require_enabled=False)) >= 5


def test_capability_to_dict_and_frozen(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    cap = platform_catalog()[0]
    d = cap.to_dict()
    assert set(d) >= {
        "capability_id",
        "name",
        "description",
        "schema_version",
        "category",
        "flag_required",
    }
    with pytest.raises((AttributeError, TypeError)):
        cap.name = "bad"  # type: ignore[misc]


# -- agent_catalog -----------------------------------------------------------


def test_agent_catalog_no_server(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    s = agent_catalog("ghost", server=None)
    assert s.agent_id == "ghost" and s.declared_capabilities == []


def test_agent_catalog_bypasses_gate(monkeypatch):
    monkeypatch.delenv("ARAGORA_A2A_DISCOVERY_ENABLED", raising=False)
    assert agent_catalog("x", server=None, require_enabled=False).agent_id == "x"


def test_agent_catalog_unknown_and_known(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    assert agent_catalog("none", server=_server()).declared_capabilities == []
    s = agent_catalog("codex", server=_server("codex", ["debate", "audit"]))
    assert {"debate", "audit"} <= set(s.declared_capabilities)


def test_agent_catalog_serializes_agent_capability_values(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    s = agent_catalog(
        "codex",
        server=_server("codex", [AgentCapability.DEBATE, AgentCapability.CRITIQUE]),
    )
    assert s.declared_capabilities == ["debate", "critique"]


def test_agent_summary_defaults_and_to_dict(monkeypatch):
    monkeypatch.setenv("ARAGORA_A2A_DISCOVERY_ENABLED", "1")
    s = AgentCapabilitySummary(agent_id="z")
    assert s.declared_capabilities == [] and s.reputation_stub == {}
    d = agent_catalog("a", server=_server("a", ["x"])).to_dict()
    assert set(d) == {"agent_id", "declared_capabilities", "reputation_stub"}
