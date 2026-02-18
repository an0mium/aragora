"""Tests for aragora.debate.phases.evidence_refresh — EvidenceRefresher."""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


from aragora.debate.phases.evidence_refresh import (
    DEFAULT_CALLBACK_TIMEOUT,
    EvidenceRefresher,
    _with_callback_timeout,
)


# ---------------------------------------------------------------------------
# _with_callback_timeout
# ---------------------------------------------------------------------------


class TestWithCallbackTimeout:
    @pytest.mark.asyncio
    async def test_returns_result_within_timeout(self):
        async def fast():
            return 42

        result = await _with_callback_timeout(fast(), timeout=5.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_default_on_timeout(self):
        async def slow():
            await asyncio.sleep(100)

        result = await _with_callback_timeout(slow(), timeout=0.01, default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_returns_none_default(self):
        async def slow():
            await asyncio.sleep(100)

        result = await _with_callback_timeout(slow(), timeout=0.01)
        assert result is None

    @pytest.mark.asyncio
    async def test_default_timeout_value(self):
        assert DEFAULT_CALLBACK_TIMEOUT == 30.0


# ---------------------------------------------------------------------------
# EvidenceRefresher — init
# ---------------------------------------------------------------------------


class TestEvidenceRefresherInit:
    def test_defaults(self):
        r = EvidenceRefresher()
        assert r._refresh_evidence is None
        assert r.hooks == {}
        assert r._notify_spectator is None
        assert r._timeout == DEFAULT_CALLBACK_TIMEOUT
        assert r.skill_registry is None
        assert r.enable_skills is False

    def test_custom(self):
        cb = AsyncMock()
        hooks = {"on_evidence_refresh": MagicMock()}
        spectator = MagicMock()
        registry = MagicMock()
        r = EvidenceRefresher(
            refresh_callback=cb,
            hooks=hooks,
            notify_spectator=spectator,
            timeout=10.0,
            skill_registry=registry,
            enable_skills=True,
        )
        assert r._refresh_evidence is cb
        assert r.hooks is hooks
        assert r._notify_spectator is spectator
        assert r._timeout == 10.0
        assert r.skill_registry is registry
        assert r.enable_skills is True


# ---------------------------------------------------------------------------
# refresh_for_round — early returns
# ---------------------------------------------------------------------------


def _make_ctx(**overrides):
    ctx = MagicMock()
    ctx.proposals = overrides.get("proposals", {})
    ctx.evidence_pack = overrides.get("evidence_pack", None)
    return ctx


class TestRefreshForRoundEarlyReturns:
    @pytest.mark.asyncio
    async def test_no_callback_returns_zero(self):
        r = EvidenceRefresher(refresh_callback=None)
        result = await r.refresh_for_round(_make_ctx(), round_num=1, partial_critiques=[])
        assert result == 0

    @pytest.mark.asyncio
    async def test_even_round_returns_zero(self):
        cb = AsyncMock(return_value=5)
        r = EvidenceRefresher(refresh_callback=cb)
        result = await r.refresh_for_round(_make_ctx(), round_num=2, partial_critiques=[])
        assert result == 0
        cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_even_round_zero_returns_zero(self):
        cb = AsyncMock(return_value=5)
        r = EvidenceRefresher(refresh_callback=cb)
        result = await r.refresh_for_round(_make_ctx(), round_num=0, partial_critiques=[])
        assert result == 0

    @pytest.mark.asyncio
    async def test_no_texts_returns_zero(self):
        cb = AsyncMock(return_value=5)
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 0
        cb.assert_not_called()


# ---------------------------------------------------------------------------
# refresh_for_round — callback invocation
# ---------------------------------------------------------------------------


class TestRefreshForRoundCallback:
    @pytest.mark.asyncio
    async def test_calls_callback_with_combined_text(self):
        cb = AsyncMock(return_value=3)
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"agent1": "proposal text here"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 3
        cb.assert_called_once()
        args = cb.call_args[0]
        assert "proposal text here" in args[0]

    @pytest.mark.asyncio
    async def test_includes_critique_text(self):
        cb = AsyncMock(return_value=2)
        r = EvidenceRefresher(refresh_callback=cb)
        critique = MagicMock()
        critique.to_prompt.return_value = "critique content"
        ctx = _make_ctx(proposals={"a": "proposal"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[critique])
        assert result == 2
        text_arg = cb.call_args[0][0]
        assert "critique content" in text_arg

    @pytest.mark.asyncio
    async def test_critique_without_to_prompt_uses_str(self):
        cb = AsyncMock(return_value=1)
        r = EvidenceRefresher(refresh_callback=cb)

        class PlainCritique:
            def __str__(self):
                return "string critique"

        ctx = _make_ctx(proposals={"a": "p"})
        await r.refresh_for_round(ctx, round_num=1, partial_critiques=[PlainCritique()])
        text_arg = cb.call_args[0][0]
        assert "string critique" in text_arg

    @pytest.mark.asyncio
    async def test_limits_critiques_to_last_five(self):
        cb = AsyncMock(return_value=0)
        r = EvidenceRefresher(refresh_callback=cb)
        critiques = []
        for i in range(10):
            c = MagicMock()
            c.to_prompt.return_value = f"critique_{i}"
            critiques.append(c)
        ctx = _make_ctx(proposals={"a": "p"})
        await r.refresh_for_round(ctx, round_num=1, partial_critiques=critiques)
        text_arg = cb.call_args[0][0]
        # Last 5 critiques (indices 5-9)
        assert "critique_5" in text_arg
        assert "critique_9" in text_arg
        # First critique should not be included
        assert "critique_0" not in text_arg

    @pytest.mark.asyncio
    async def test_truncates_proposal_to_2000_chars(self):
        cb = AsyncMock(return_value=1)
        r = EvidenceRefresher(refresh_callback=cb)
        long_proposal = "x" * 5000
        ctx = _make_ctx(proposals={"a": long_proposal})
        await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        text_arg = cb.call_args[0][0]
        assert len(text_arg) <= 2000

    @pytest.mark.asyncio
    async def test_callback_timeout_returns_zero(self):
        async def slow_cb(text, ctx, round_num):
            await asyncio.sleep(100)
            return 99

        r = EvidenceRefresher(refresh_callback=slow_cb, timeout=0.01)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 0

    @pytest.mark.asyncio
    async def test_callback_returns_none_treated_as_zero(self):
        cb = AsyncMock(return_value=None)
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 0


# ---------------------------------------------------------------------------
# refresh_for_round — spectator notification and hooks
# ---------------------------------------------------------------------------


class TestRefreshNotifications:
    @pytest.mark.asyncio
    async def test_notifies_spectator_on_refresh(self):
        cb = AsyncMock(return_value=5)
        spectator = MagicMock()
        r = EvidenceRefresher(refresh_callback=cb, notify_spectator=spectator)
        ctx = _make_ctx(proposals={"a": "p"})
        await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        spectator.assert_called_once_with(
            "evidence",
            details="Refreshed evidence: 5 new sources",
            metric=5,
            agent="system",
        )

    @pytest.mark.asyncio
    async def test_no_notification_on_zero_snippets(self):
        cb = AsyncMock(return_value=0)
        spectator = MagicMock()
        r = EvidenceRefresher(refresh_callback=cb, notify_spectator=spectator)
        ctx = _make_ctx(proposals={"a": "p"})
        await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        spectator.assert_not_called()

    @pytest.mark.asyncio
    async def test_fires_hook_on_refresh(self):
        cb = AsyncMock(return_value=3)
        hook = MagicMock()
        r = EvidenceRefresher(
            refresh_callback=cb,
            hooks={"on_evidence_refresh": hook},
        )
        ctx = _make_ctx(proposals={"a": "p"})
        await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        hook.assert_called_once_with(round_num=1, new_snippets=3)

    @pytest.mark.asyncio
    async def test_no_hook_fire_on_zero(self):
        cb = AsyncMock(return_value=0)
        hook = MagicMock()
        r = EvidenceRefresher(
            refresh_callback=cb,
            hooks={"on_evidence_refresh": hook},
        )
        ctx = _make_ctx(proposals={"a": "p"})
        await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        hook.assert_not_called()


# ---------------------------------------------------------------------------
# refresh_for_round — error handling
# ---------------------------------------------------------------------------


class TestRefreshErrorHandling:
    @pytest.mark.asyncio
    async def test_runtime_error_returns_zero(self):
        cb = AsyncMock(side_effect=RuntimeError("boom"))
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 0

    @pytest.mark.asyncio
    async def test_attribute_error_returns_zero(self):
        cb = AsyncMock(side_effect=AttributeError("no attr"))
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 0

    @pytest.mark.asyncio
    async def test_type_error_returns_zero(self):
        cb = AsyncMock(side_effect=TypeError("bad type"))
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 0


# ---------------------------------------------------------------------------
# _refresh_with_skills
# ---------------------------------------------------------------------------


class TestRefreshWithSkills:
    @pytest.mark.asyncio
    async def test_no_registry_returns_zero(self):
        r = EvidenceRefresher(enable_skills=True, skill_registry=None)
        result = await r._refresh_with_skills("text", _make_ctx())
        assert result == 0

    @pytest.mark.asyncio
    async def test_skills_disabled_returns_zero(self):
        r = EvidenceRefresher(enable_skills=False, skill_registry=MagicMock())
        result = await r._refresh_with_skills("text", _make_ctx())
        assert result == 0

    @pytest.mark.asyncio
    async def test_skills_combined_with_callback(self):
        cb = AsyncMock(return_value=2)
        r = EvidenceRefresher(refresh_callback=cb, enable_skills=True, skill_registry=MagicMock())
        # Mock _refresh_with_skills to return skill snippets
        r._refresh_with_skills = AsyncMock(return_value=3)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 5  # 2 from callback + 3 from skills


# ---------------------------------------------------------------------------
# Odd round numbers
# ---------------------------------------------------------------------------


class TestOddRounds:
    @pytest.mark.asyncio
    async def test_round_1_triggers_refresh(self):
        cb = AsyncMock(return_value=1)
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 1

    @pytest.mark.asyncio
    async def test_round_3_triggers_refresh(self):
        cb = AsyncMock(return_value=2)
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=3, partial_critiques=[])
        assert result == 2

    @pytest.mark.asyncio
    async def test_round_4_skips_refresh(self):
        cb = AsyncMock(return_value=2)
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"a": "p"})
        result = await r.refresh_for_round(ctx, round_num=4, partial_critiques=[])
        assert result == 0

    @pytest.mark.asyncio
    async def test_multiple_proposals_combined(self):
        cb = AsyncMock(return_value=4)
        r = EvidenceRefresher(refresh_callback=cb)
        ctx = _make_ctx(proposals={"a": "prop_a", "b": "prop_b", "c": "prop_c"})
        result = await r.refresh_for_round(ctx, round_num=1, partial_critiques=[])
        assert result == 4
        text_arg = cb.call_args[0][0]
        assert "prop_a" in text_arg
        assert "prop_b" in text_arg
        assert "prop_c" in text_arg
