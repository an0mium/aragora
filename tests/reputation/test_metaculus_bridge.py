"""Tests for aragora.reputation.metaculus_bridge (AGT-05 / #6066)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.reputation.metaculus_bridge import (
    bridge_from_metaculus_question,
    reputation_flow_enabled,
)
from aragora.reputation.types import DOMAIN_PREDICTION_MARKET


@dataclass(frozen=True)
class _FQ:
    """Structural stand-in for MetaculusQuestion — no network import needed."""

    question_id: int
    title: str
    question_type: str
    created_time: str | None
    close_time: str | None
    resolve_time: str | None
    active_state: str
    resolution: float | None
    community_q2: float | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.active_state == "resolved"


def _q(
    question_id: int = 123,
    title: str = "Will Aragora ship?",
    resolution: float | None = 1.0,
    resolve_time: str | None = "2026-05-01T00:00:00Z",
    active_state: str = "resolved",
) -> _FQ:
    return _FQ(
        question_id=question_id,
        title=title,
        question_type="binary",
        created_time="2026-04-01T00:00:00Z",
        close_time="2026-04-30T00:00:00Z",
        resolve_time=resolve_time,
        active_state=active_state,
        resolution=resolution,
        community_q2=0.7,
    )


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------


class TestFeatureGate:
    def test_off_by_default_and_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
        assert reputation_flow_enabled() is False
        with pytest.raises(RuntimeError, match="ARAGORA_REPUTATION_FLOW_ENABLED"):
            bridge_from_metaculus_question(_q(), "a", 0.8)

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", val)
        assert reputation_flow_enabled() is True
        claim, _ = bridge_from_metaculus_question(_q(), "a", 0.8)
        assert claim.agent_id == "a"

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_falsy_values_block(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", val)
        with pytest.raises(RuntimeError):
            bridge_from_metaculus_question(_q(), "a", 0.8)

    def test_require_enabled_false_bypasses_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
        claim, resolved = bridge_from_metaculus_question(_q(), "a", 0.8, require_enabled=False)
        assert claim.agent_id == "a"
        assert resolved.outcome == "yes"


# ---------------------------------------------------------------------------
# Outcome mapping
# ---------------------------------------------------------------------------


class TestOutcomeMapping:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")

    def test_resolution_1_0_yes(self) -> None:
        assert bridge_from_metaculus_question(_q(resolution=1.0), "a", 0.8)[1].outcome == "yes"

    def test_resolution_0_0_no(self) -> None:
        assert bridge_from_metaculus_question(_q(resolution=0.0), "a", 0.2)[1].outcome == "no"

    @pytest.mark.parametrize("res", [None, 0.5, 0.73])
    def test_other_resolution_inconclusive(self, res: float | None) -> None:
        assert (
            bridge_from_metaculus_question(_q(resolution=res), "a", 0.6)[1].outcome
            == "inconclusive"
        )


# ---------------------------------------------------------------------------
# Claim shape
# ---------------------------------------------------------------------------


class TestClaimShape:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")

    def test_domain_title_and_probability(self) -> None:
        claim, _ = bridge_from_metaculus_question(_q(title="Will crux ship?"), "a", 0.73)
        assert claim.domain == DOMAIN_PREDICTION_MARKET
        assert claim.statement == "Will crux ship?"
        assert claim.predicted_probability == pytest.approx(0.73)

    def test_position_from_probability(self) -> None:
        assert bridge_from_metaculus_question(_q(), "a", 0.5)[0].position == "yes"
        assert bridge_from_metaculus_question(_q(), "a", 0.49)[0].position == "no"

    def test_resolution_source_default_and_override(self) -> None:
        c1, r1 = bridge_from_metaculus_question(_q(), "a", 0.8)
        assert c1.resolution_source == r1.resolution_source == "metaculus"
        c2, r2 = bridge_from_metaculus_question(_q(), "a", 0.8, resolution_source="custom")
        assert c2.resolution_source == r2.resolution_source == "custom"

    def test_resolution_id_and_provenance(self) -> None:
        claim, _ = bridge_from_metaculus_question(_q(question_id=42), "a", 0.8)
        assert claim.resolution_id == "42"
        assert claim.provenance["question_id"] == 42
        for key in ("question_type", "close_time", "resolve_time", "submitted_at"):
            assert key in claim.provenance

    def test_evidence_shape(self) -> None:
        _, resolved = bridge_from_metaculus_question(_q(resolution=1.0), "a", 0.8)
        assert resolved.evidence["question_id"] == 123
        assert resolved.evidence["resolution"] == 1.0
        assert resolved.evidence["active_state"] == "resolved"

    def test_claim_id_content_addressed(self) -> None:
        q = _q()
        c1, _ = bridge_from_metaculus_question(q, "a", 0.8, require_enabled=False)
        c2, _ = bridge_from_metaculus_question(q, "a", 0.8, require_enabled=False)
        assert c1.claim_id == c2.claim_id

    def test_claim_id_stable_and_unique(self) -> None:
        q = _q()
        c1, _ = bridge_from_metaculus_question(q, "a", 0.8, require_enabled=False)
        c2, _ = bridge_from_metaculus_question(q, "a", 0.8, require_enabled=False)
        cb, _ = bridge_from_metaculus_question(q, "b", 0.8, require_enabled=False)
        assert c1.claim_id == c2.claim_id  # deterministic
        assert c1.claim_id != cb.claim_id  # differs by agent

    def test_resolved_at_and_stake_params(self) -> None:
        _, r = bridge_from_metaculus_question(
            _q(resolve_time="2026-03-01T00:00:00Z"), "a", 0.8, require_enabled=False
        )
        assert r.resolved_at == "2026-03-01T00:00:00Z"
        c, r2 = bridge_from_metaculus_question(
            _q(), "a", 0.8, stake_units=3, stake_policy="scaled", require_enabled=False
        )
        assert c.claim_id == r2.claim_id and c.stake_units == 3 and c.stake_policy == "scaled"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")

    def test_unresolved_question_raises(self) -> None:
        with pytest.raises(ValueError, match="not resolved"):
            bridge_from_metaculus_question(
                _q(active_state="open", resolution=None, resolve_time=None), "a", 0.8
            )

    @pytest.mark.parametrize("prob", [1.001, -0.001])
    def test_out_of_range_probability_raises(self, prob: float) -> None:
        with pytest.raises(ValueError, match="predicted_probability"):
            bridge_from_metaculus_question(_q(), "a", prob)

    @pytest.mark.parametrize("units", [0, -1])
    def test_invalid_stake_units_raises(self, units: int) -> None:
        with pytest.raises(ValueError, match="stake_units"):
            bridge_from_metaculus_question(_q(), "a", 0.8, stake_units=units)

    def test_boundary_probabilities_valid(self) -> None:
        c0, _ = bridge_from_metaculus_question(_q(), "a", 0.0, require_enabled=False)
        c1, _ = bridge_from_metaculus_question(_q(), "a", 1.0, require_enabled=False)
        assert c0.predicted_probability == 0.0
        assert c1.predicted_probability == 1.0
