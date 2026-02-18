"""Tests for the business context scoring layer."""

from __future__ import annotations

import pytest

from aragora.nomic.business_context import (
    BusinessContext,
    BusinessContextConfig,
    GoalScore,
)


@pytest.fixture
def ctx() -> BusinessContext:
    return BusinessContext()


class TestUserFacingScore:
    def test_all_user_facing(self, ctx: BusinessContext):
        score = ctx.score_goal("Fix UI", file_paths=["aragora/live/page.tsx", "sdk/client.ts"])
        assert score.user_facing == 1.0

    def test_no_user_facing(self, ctx: BusinessContext):
        score = ctx.score_goal("Fix internals", file_paths=["aragora/storage/pg.py"])
        assert score.user_facing == 0.0

    def test_partial_user_facing(self, ctx: BusinessContext):
        score = ctx.score_goal("Mixed", file_paths=["aragora/live/x.tsx", "aragora/storage/pg.py"])
        assert score.user_facing == 0.5

    def test_no_files_zero(self, ctx: BusinessContext):
        score = ctx.score_goal("Vague goal")
        assert score.user_facing == 0.0


class TestRevenueRelevance:
    def test_billing_path(self, ctx: BusinessContext):
        score = ctx.score_goal("Fix billing", file_paths=["aragora/billing/cost.py"])
        assert score.revenue == 1.0

    def test_non_revenue_path(self, ctx: BusinessContext):
        score = ctx.score_goal("Fix debate", file_paths=["aragora/debate/orchestrator.py"])
        assert score.revenue == 0.0


class TestUnblockingScore:
    def test_keyword_match(self, ctx: BusinessContext):
        score = ctx.score_goal("Unblock the SDK release")
        assert score.unblocking >= 0.4

    def test_blocks_count_metadata(self, ctx: BusinessContext):
        score = ctx.score_goal("Fix module", metadata={"blocks_count": 5})
        assert score.unblocking >= 0.3

    def test_dependency_count_metadata(self, ctx: BusinessContext):
        score = ctx.score_goal("Fix module", metadata={"dependency_count": 3})
        assert score.unblocking >= 0.3

    def test_no_signals_zero(self, ctx: BusinessContext):
        score = ctx.score_goal("Improve docs")
        assert score.unblocking == 0.0


class TestTechDebtScore:
    def test_refactor_keyword(self, ctx: BusinessContext):
        score = ctx.score_goal("Refactor the orchestrator")
        assert score.tech_debt > 0.0

    def test_multiple_keywords(self, ctx: BusinessContext):
        score = ctx.score_goal("Refactor legacy deprecated code")
        assert score.tech_debt > 0.5

    def test_no_keywords_zero(self, ctx: BusinessContext):
        score = ctx.score_goal("Add new feature")
        assert score.tech_debt == 0.0


class TestCompositeScore:
    def test_total_is_weighted_sum(self, ctx: BusinessContext):
        score = ctx.score_goal(
            "Refactor billing UI",
            file_paths=["aragora/live/page.tsx", "aragora/billing/cost.py"],
        )
        cfg = ctx.config
        expected = (
            cfg.user_facing_weight * score.user_facing
            + cfg.revenue_weight * score.revenue
            + cfg.unblocking_weight * score.unblocking
            + cfg.tech_debt_weight * score.tech_debt
        )
        assert abs(score.total - round(expected, 4)) < 0.001

    def test_breakdown_present(self, ctx: BusinessContext):
        score = ctx.score_goal("Fix billing", file_paths=["aragora/billing/cost.py"])
        assert "revenue_weighted" in score.breakdown
        assert "user_facing_weighted" in score.breakdown


class TestRankGoals:
    def test_ranks_highest_first(self, ctx: BusinessContext):
        goals = [
            {"goal": "Improve docs", "file_paths": ["docs/README.md"]},
            {"goal": "Fix billing UI", "file_paths": ["aragora/live/x.tsx", "aragora/billing/y.py"]},
            {"goal": "Refactor legacy code"},
        ]
        ranked = ctx.rank_goals(goals)
        assert ranked[0]["goal"] == "Fix billing UI"
        # All should have score attached
        for g in ranked:
            assert isinstance(g["score"], GoalScore)

    def test_empty_list(self, ctx: BusinessContext):
        assert ctx.rank_goals([]) == []


class TestCustomConfig:
    def test_custom_weights(self):
        cfg = BusinessContextConfig(
            user_facing_weight=1.0,
            revenue_weight=0.0,
            unblocking_weight=0.0,
            tech_debt_weight=0.0,
        )
        ctx = BusinessContext(cfg)
        score = ctx.score_goal("Fix UI", file_paths=["aragora/live/page.tsx"])
        assert score.total == score.user_facing

    def test_custom_paths(self):
        cfg = BusinessContextConfig(
            user_facing_paths=["custom/"],
        )
        ctx = BusinessContext(cfg)
        score = ctx.score_goal("Fix", file_paths=["custom/x.py"])
        assert score.user_facing == 1.0
