"""Tests for governance storage metric helpers."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from aragora.storage.governance import metrics


@pytest.fixture
def fake_observability_metrics(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []
    package = types.ModuleType("aragora.observability")
    module = types.ModuleType("aragora.observability.metrics")

    def _make_recorder(name: str):
        def _record(first_value: str, second_value: str) -> None:
            calls.append((name, first_value, second_value))

        return _record

    module.record_governance_verification = _make_recorder("verification")  # type: ignore[attr-defined]
    module.record_governance_decision = _make_recorder("decision")  # type: ignore[attr-defined]
    module.record_governance_approval = _make_recorder("approval")  # type: ignore[attr-defined]
    package.metrics = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aragora.observability", package)
    monkeypatch.setitem(sys.modules, "aragora.observability.metrics", module)
    return calls


def test_record_governance_verification_delegates_to_observability(
    fake_observability_metrics: list[tuple[str, str, str]],
) -> None:
    metrics.record_governance_verification("contract", "passed")

    assert fake_observability_metrics == [("verification", "contract", "passed")]


def test_record_governance_decision_delegates_to_observability(
    fake_observability_metrics: list[tuple[str, str, str]],
) -> None:
    metrics.record_governance_decision("policy", "approved")

    assert fake_observability_metrics == [("decision", "policy", "approved")]


def test_record_governance_approval_delegates_to_observability(
    fake_observability_metrics: list[tuple[str, str, str]],
) -> None:
    metrics.record_governance_approval("manual", "pending")

    assert fake_observability_metrics == [("approval", "manual", "pending")]


def test_helpers_forward_empty_strings(
    fake_observability_metrics: list[tuple[str, str, str]],
) -> None:
    metrics.record_governance_decision("", "")

    assert fake_observability_metrics == [("decision", "", "")]


def test_helpers_forward_unusual_string_values(
    fake_observability_metrics: list[tuple[str, str, str]],
) -> None:
    metrics.record_governance_approval("type:with:colon", "status with spaces")

    assert fake_observability_metrics == [("approval", "type:with:colon", "status with spaces")]


def test_missing_observability_module_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _raise_for_observability(name: str, *args, **kwargs):
        if name == "aragora.observability":
            raise ImportError("observability unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_for_observability)

    metrics.record_governance_verification("contract", "passed")
    metrics.record_governance_decision("policy", "approved")
    metrics.record_governance_approval("manual", "pending")


def test_missing_metric_function_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("aragora.observability")
    module = types.ModuleType("aragora.observability.metrics")
    package.metrics = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aragora.observability", package)
    monkeypatch.setitem(sys.modules, "aragora.observability.metrics", module)

    metrics.record_governance_verification("contract", "passed")


def test_non_callable_metric_attribute_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("aragora.observability")
    module = types.ModuleType("aragora.observability.metrics")
    module.record_governance_decision = "not-callable"  # type: ignore[attr-defined]
    package.metrics = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aragora.observability", package)
    monkeypatch.setitem(sys.modules, "aragora.observability.metrics", module)

    metrics.record_governance_decision("policy", "approved")


def test_public_exports_are_limited_to_record_helpers() -> None:
    assert metrics.__all__ == [
        "record_governance_verification",
        "record_governance_decision",
        "record_governance_approval",
    ]
