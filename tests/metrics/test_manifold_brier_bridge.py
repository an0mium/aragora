"""Tests for aragora.metrics.manifold_brier_bridge — AGT-03 sub-deliverable 2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from aragora.metrics.manifold_brier import ManifoldBrierScorer
from aragora.metrics.manifold_brier_bridge import (
    PendingPrediction,
    ResolutionEventProtocol,
    batch_record_resolutions,
    record_resolution,
    resolution_to_binary_outcome,
)

_NOW = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
_TS_MS = 1_747_224_000_000  # 2026-05-14T12:00:00Z


@dataclass(frozen=True)
class _Res:
    """Minimal stub satisfying ResolutionEventProtocol."""

    market_id: str
    outcome: str
    resolved_at_ms: int | None


def _res(outcome: str, market_id: str = "mkt_abc") -> _Res:
    return _Res(market_id=market_id, outcome=outcome, resolved_at_ms=_TS_MS)


def _pend(market_id: str = "mkt_abc", agent: str = "ag", p: float = 0.7) -> PendingPrediction:
    return PendingPrediction(
        market_id=market_id, agent_id=agent, predicted_probability=p, predicted_at=_NOW
    )


# ---------------------------------------------------------------------------


class TestResolutionToBinaryOutcome:
    @pytest.mark.parametrize("outcome,expected", [("yes", 1), ("no", 0), ("inconclusive", None)])
    def test_mapping(self, outcome: str, expected: int | None) -> None:
        assert resolution_to_binary_outcome(_res(outcome)) == expected

    def test_stub_satisfies_protocol(self) -> None:
        assert isinstance(_res("yes"), ResolutionEventProtocol)

    def test_pure_no_flag_needed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_MANIFOLD_BRIER_ENABLED", raising=False)
        assert resolution_to_binary_outcome(_res("yes")) == 1


class TestPendingPrediction:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"market_id": ""}, "market_id"),
            ({"agent_id": ""}, "agent_id"),
            ({"predicted_probability": 1.1}, "predicted_probability"),
            ({"predicted_probability": -0.1}, "predicted_probability"),
        ],
    )
    def test_validation(self, kwargs: dict, match: str) -> None:
        base = dict(market_id="m", agent_id="a", predicted_probability=0.5, predicted_at=_NOW)
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            PendingPrediction(**base)  # type: ignore[arg-type]

    def test_boundary_probabilities_ok(self) -> None:
        for prob in (0.0, 1.0):
            p = PendingPrediction(
                market_id="m", agent_id="a", predicted_probability=prob, predicted_at=_NOW
            )
            assert p.predicted_probability == prob


class TestRecordResolution:
    def test_yes_records_outcome_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        scorer = ManifoldBrierScorer()
        result = record_resolution(scorer, _res("yes"), _pend(p=0.8))
        assert result is not None and result.outcome == 1 and result.predicted_probability == 0.8
        assert len(scorer.all_predictions()) == 1

    def test_no_records_outcome_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        scorer = ManifoldBrierScorer()
        result = record_resolution(scorer, _res("no"), _pend(p=0.3))
        assert result is not None and result.outcome == 0

    def test_inconclusive_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        scorer = ManifoldBrierScorer()
        assert record_resolution(scorer, _res("inconclusive"), _pend()) is None
        assert len(scorer.all_predictions()) == 0

    def test_fields_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        scorer = ManifoldBrierScorer()
        result = record_resolution(scorer, _res("yes", "mkt_xyz"), _pend("mkt_xyz", agent="codex"))
        assert result is not None
        assert result.question_id == "mkt_xyz"
        assert result.agent_id == "codex"
        assert result.resolved_at == datetime.fromtimestamp(_TS_MS / 1000.0, tz=UTC)

    def test_none_resolved_at_ms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        scorer = ManifoldBrierScorer()
        r = _Res(market_id="m", outcome="yes", resolved_at_ms=None)
        result = record_resolution(scorer, r, _pend("m"))
        assert result is not None and result.resolved_at is None

    def test_market_id_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        scorer = ManifoldBrierScorer()
        with pytest.raises(ValueError, match="market_id"):
            record_resolution(scorer, _res("yes", "m_other"), _pend("m_pending"))

    def test_disabled_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_MANIFOLD_BRIER_ENABLED", raising=False)
        with pytest.raises(RuntimeError, match="ARAGORA_MANIFOLD_BRIER_ENABLED"):
            record_resolution(ManifoldBrierScorer(), _res("yes"), _pend())


class TestBatchRecordResolutions:
    def test_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        assert batch_record_resolutions(ManifoldBrierScorer(), {}, []) == (0, 0)

    def test_no_resolutions_all_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        r, s = batch_record_resolutions(ManifoldBrierScorer(), {}, [_pend("m1"), _pend("m2")])
        assert r == 0 and s == 2

    def test_mixed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        scorer = ManifoldBrierScorer()
        resolutions = {
            "m_yes": _res("yes", "m_yes"),
            "m_no": _res("no", "m_no"),
            "m_inc": _res("inconclusive", "m_inc"),
        }
        pending = [_pend("m_yes"), _pend("m_no"), _pend("m_inc"), _pend("m_missing")]
        r, s = batch_record_resolutions(scorer, resolutions, pending)
        assert r == 2 and s == 2
        assert len(scorer.all_predictions()) == 2

    def test_disabled_raises_before_iterating(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_MANIFOLD_BRIER_ENABLED", raising=False)
        with pytest.raises(RuntimeError, match="ARAGORA_MANIFOLD_BRIER_ENABLED"):
            batch_record_resolutions(ManifoldBrierScorer(), {"m": _res("yes")}, [_pend()])

    def test_brier_arithmetic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_MANIFOLD_BRIER_ENABLED", "1")
        scorer = ManifoldBrierScorer()
        resolutions = {"m_a": _res("yes", "m_a"), "m_b": _res("no", "m_b")}
        pending = [
            PendingPrediction(
                market_id="m_a", agent_id="ag", predicted_probability=1.0, predicted_at=_NOW
            ),
            PendingPrediction(
                market_id="m_b", agent_id="ag", predicted_probability=0.9, predicted_at=_NOW
            ),
        ]
        batch_record_resolutions(scorer, resolutions, pending)
        summary = scorer.rolling_score_for_agent("ag", window_days=30, reference_time=_NOW)
        assert summary.n_predictions == 2
        assert abs(summary.mean_brier - 0.405) < 1e-9  # (0.0 + 0.81) / 2
