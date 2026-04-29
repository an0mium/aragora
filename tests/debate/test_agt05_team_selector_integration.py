"""AGT-05 TeamSelector ← ReputationCalibrationBridge integration tests.

Verifies that ``enable_agt05_reputation_selection`` in ``TeamSelectionConfig``
wires the AGT-05 reputation bridge as the calibration scorer when a
``ReputationStore`` is supplied to ``TeamSelector.__init__``.

Gating: ``ARAGORA_REPUTATION_FLOW_ENABLED`` (default off).
Advances issue: #6066 (AGT-05 Skin-in-the-game reputation flow).
"""

from __future__ import annotations

import pytest

from aragora.debate.team_selector import TeamSelectionConfig, TeamSelector
from aragora.reputation.selection_bridge import ReputationBridgeConfig, ReputationCalibrationBridge
from aragora.reputation.store import ReputationStore
from aragora.reputation.types import DOMAIN_PREDICTION_MARKET, ReputationDelta


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_delta(agent_id: str, delta: float = 10.0, idx: int = 0) -> ReputationDelta:
    return ReputationDelta(
        delta_id=f"delta_{agent_id}_{idx}",
        agent_id=agent_id,
        domain=DOMAIN_PREDICTION_MARKET,
        claim_id=f"clm_{agent_id}_{idx}",
        resolution_id=f"res_{agent_id}_{idx}",
        delta=delta,
        scoring_rule="binary",
        applied_at="2026-04-28T00:00:00Z",
        decay_half_life_days=None,
        reason={"test": True},
    )


def _store_with_deltas(agent_id: str, count: int = 5, delta: float = 10.0) -> ReputationStore:
    store = ReputationStore()
    for i in range(count):
        store.record_delta(_make_delta(agent_id, delta=delta, idx=i))
    return store


# ---------------------------------------------------------------------------
# Wiring tests (env flag not required — only constructor / config behaviour)
# ---------------------------------------------------------------------------


class TestTeamSelectorAgt05Wiring:
    def test_default_flag_is_false(self):
        assert TeamSelectionConfig().enable_agt05_reputation_selection is False

    def test_default_bridge_config_is_none(self):
        assert TeamSelectionConfig().reputation_bridge_config is None

    def test_flag_off_bridge_not_installed(self):
        store = ReputationStore()
        cfg = TeamSelectionConfig(enable_agt05_reputation_selection=False)
        selector = TeamSelector(config=cfg, reputation_store=store)
        assert not isinstance(selector.calibration_tracker, ReputationCalibrationBridge)

    def test_flag_on_no_store_no_bridge(self):
        cfg = TeamSelectionConfig(enable_agt05_reputation_selection=True)
        selector = TeamSelector(config=cfg, reputation_store=None)
        assert not isinstance(selector.calibration_tracker, ReputationCalibrationBridge)

    def test_flag_on_with_store_installs_bridge(self):
        cfg = TeamSelectionConfig(enable_agt05_reputation_selection=True)
        selector = TeamSelector(config=cfg, reputation_store=ReputationStore())
        assert isinstance(selector.calibration_tracker, ReputationCalibrationBridge)

    def test_explicit_tracker_superseded_by_bridge_when_flag_on(self):
        """Bridge replaces an explicitly supplied calibration_tracker."""

        class _Dummy:
            def get_brier_score(self, agent_name: str, domain: str | None = None) -> float:
                return 0.99

            def get_brier_scores_batch(
                self, agent_names: list[str], domain: str | None = None
            ) -> dict[str, float]:
                return dict.fromkeys(agent_names, 0.99)

        cfg = TeamSelectionConfig(enable_agt05_reputation_selection=True)
        selector = TeamSelector(
            config=cfg,
            reputation_store=ReputationStore(),
            calibration_tracker=_Dummy(),  # type: ignore[arg-type]
        )
        assert isinstance(selector.calibration_tracker, ReputationCalibrationBridge)

    def test_custom_bridge_config_forwarded_to_bridge(self):
        bridge_cfg = ReputationBridgeConfig(min_samples=3, score_scale=50.0)
        cfg = TeamSelectionConfig(
            enable_agt05_reputation_selection=True,
            reputation_bridge_config=bridge_cfg,
        )
        selector = TeamSelector(config=cfg, reputation_store=ReputationStore())
        bridge = selector.calibration_tracker
        assert isinstance(bridge, ReputationCalibrationBridge)
        assert bridge._cfg.min_samples == 3
        assert bridge._cfg.score_scale == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Scoring behaviour tests (require ARAGORA_REPUTATION_FLOW_ENABLED)
# ---------------------------------------------------------------------------


class TestTeamSelectorAgt05Scoring:
    def test_neutral_when_env_flag_unset(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REPUTATION_FLOW_ENABLED", raising=False)
        cfg = TeamSelectionConfig(enable_agt05_reputation_selection=True)
        selector = TeamSelector(config=cfg, reputation_store=_store_with_deltas("agent-x"))
        score = selector.calibration_tracker.get_brier_score("agent-x")  # type: ignore[union-attr]
        assert score == pytest.approx(0.5)

    def test_non_neutral_with_positive_score_and_env_flag(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
        bridge_cfg = ReputationBridgeConfig(min_samples=1, score_scale=10.0)
        cfg = TeamSelectionConfig(
            enable_agt05_reputation_selection=True,
            reputation_bridge_config=bridge_cfg,
        )
        store = _store_with_deltas("agent-y", count=5, delta=10.0)
        selector = TeamSelector(config=cfg, reputation_store=store)
        score = selector.calibration_tracker.get_brier_score("agent-y")  # type: ignore[union-attr]
        assert score < 0.5, f"expected lower-than-neutral Brier for positive rep; got {score}"

    def test_batch_consistent_with_single(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_REPUTATION_FLOW_ENABLED", "1")
        bridge_cfg = ReputationBridgeConfig(min_samples=3, score_scale=50.0)
        cfg = TeamSelectionConfig(
            enable_agt05_reputation_selection=True,
            reputation_bridge_config=bridge_cfg,
        )
        store = _store_with_deltas("agent-z", count=3, delta=5.0)
        selector = TeamSelector(config=cfg, reputation_store=store)
        bridge = selector.calibration_tracker
        single = bridge.get_brier_score("agent-z")  # type: ignore[union-attr]
        batch = bridge.get_brier_scores_batch(["agent-z"])  # type: ignore[union-attr]
        assert single == pytest.approx(batch["agent-z"])
