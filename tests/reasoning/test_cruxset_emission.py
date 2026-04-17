"""Tests for the AGT-01 flag-gated CruxSet emission seam."""

from __future__ import annotations

import pytest

from aragora.reasoning import cruxset_emission as mod
from aragora.reasoning.cruxset import CruxSet


@pytest.fixture(autouse=True)
def _reset_flag(monkeypatch):
    monkeypatch.delenv(mod.CRUXSET_EMISSION_ENV_VAR, raising=False)
    yield
    monkeypatch.delenv(mod.CRUXSET_EMISSION_ENV_VAR, raising=False)


def _payload_with_two_cruxes() -> dict:
    return {
        "cruxes": [
            {
                "claim_id": "c1",
                "statement": "X holds",
                "author": "alice",
                "crux_score": 0.8,
                "contesting_agents": ["bob"],
                "resolution_impact": 0.5,
            },
            {
                "claim_id": "c2",
                "statement": "Y depends on X",
                "author": "carol",
                "crux_score": 0.6,
                "contesting_agents": [],
                "resolution_impact": 0.3,
            },
        ],
        "average_uncertainty": 0.4,
        "convergence_barrier": 0.2,
        "recommended_focus": ["c1", "c2"],
    }


class TestFeatureFlag:
    def test_disabled_by_default(self) -> None:
        assert mod.cruxset_emission_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "ON"])
    def test_truthy_values_enable(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, value)
        assert mod.cruxset_emission_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_falsy_values_keep_disabled(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, value)
        assert mod.cruxset_emission_enabled() is False

    def test_enable_helper_flips_flag(self) -> None:
        assert mod.cruxset_emission_enabled() is False
        mod.enable_cruxset_emission()
        assert mod.cruxset_emission_enabled() is True


class TestMaybeEmitCruxset:
    def test_returns_none_when_disabled(self) -> None:
        # Even with a perfect payload, emission stays None when the flag is off
        result = mod.maybe_emit_cruxset(
            question="should we ship?",
            analysis_payload=_payload_with_two_cruxes(),
        )
        assert result is None

    def test_returns_cruxset_when_enabled_and_payload_present(self, monkeypatch) -> None:
        monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
        result = mod.maybe_emit_cruxset(
            question="should we ship?",
            analysis_payload=_payload_with_two_cruxes(),
            decision="ship",
            receipt_id="rcpt_a",
            provenance={"debate_id": "d1"},
        )
        assert isinstance(result, CruxSet)
        assert result.question == "should we ship?"
        assert result.decision == "ship"
        assert result.receipt_id == "rcpt_a"
        # Sorted by load_bearing_score desc
        assert [c.crux_id for c in result.cruxes] == ["c1", "c2"]
        assert result.verify_checksum() is True

    def test_returns_none_when_payload_has_no_cruxes(self, monkeypatch) -> None:
        monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
        result = mod.maybe_emit_cruxset(
            question="q",
            analysis_payload={"cruxes": []},
        )
        assert result is None

    def test_returns_none_when_neither_input_supplied(self, monkeypatch) -> None:
        monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
        result = mod.maybe_emit_cruxset(question="q")
        assert result is None

    def test_swallows_detector_errors_and_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")

        class _ExplodingNetwork:
            nodes: dict = {}

        # Patch CruxDetector at import-time to raise
        from aragora.reasoning import crux_detector as cd

        class _BoomDetector:
            def __init__(self, *args, **kwargs):
                pass

            def detect_cruxes(self, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr(cd, "CruxDetector", _BoomDetector)
        result = mod.maybe_emit_cruxset(question="q", network=_ExplodingNetwork())
        assert result is None

    def test_uses_network_when_no_payload_provided(self, monkeypatch) -> None:
        monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")

        class _StubResult:
            def to_dict(self) -> dict:
                return _payload_with_two_cruxes()

        class _StubDetector:
            def __init__(self, *args, **kwargs):
                pass

            def detect_cruxes(self, **kwargs):
                return _StubResult()

        from aragora.reasoning import crux_detector as cd

        monkeypatch.setattr(cd, "CruxDetector", _StubDetector)

        result = mod.maybe_emit_cruxset(question="q", network=object())
        assert isinstance(result, CruxSet)
        assert len(result.cruxes) == 2

    def test_max_cruxes_passes_through_to_builder(self, monkeypatch) -> None:
        monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
        big_payload = {
            "cruxes": [
                {
                    "claim_id": f"c{i}",
                    "statement": f"S{i}",
                    "author": "alice",
                    "crux_score": 1.0 - 0.05 * i,
                    "contesting_agents": [],
                    "resolution_impact": 0.1,
                }
                for i in range(8)
            ],
            "average_uncertainty": 0.5,
            "convergence_barrier": 0.3,
            "recommended_focus": [],
        }
        result = mod.maybe_emit_cruxset(
            question="q",
            analysis_payload=big_payload,
            top_k=3,
        )
        assert isinstance(result, CruxSet)
        assert len(result.cruxes) == 3
