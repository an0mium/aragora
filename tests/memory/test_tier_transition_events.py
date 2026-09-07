"""Tests for memory tier transition event handling (B4).

Verifies that:
1. Tier promotion/demotion events are registered in CrossSubscriberManager
2. Demotion to slow/glacial triggers re-validation in KM
3. Promotion triggers importance boost in KM
4. Handlers are graceful when KM unavailable
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.events.types import StreamEvent, StreamEventType


def _make_event(event_type: str, data: dict) -> StreamEvent:
    """Create a StreamEvent with the given data."""
    return StreamEvent(
        type=StreamEventType(event_type),
        data=data,
    )


class TestTierDemotionRevalidation:
    """Test _handle_tier_demotion_to_revalidation handler.

    Relocated from BasicHandlersMixin (P4a Batch E2c): it is knowledge-coupled.
    """

    def _get_handler(self):
        from aragora.knowledge.event_subscribers import KnowledgeEventSubscriber

        return KnowledgeEventSubscriber()._handle_tier_demotion_to_revalidation

    def test_skips_empty_memory_id(self):
        handler = self._get_handler()
        event = _make_event(
            "memory_tier_demotion",
            {"memory_id": "", "from_tier": "medium", "to_tier": "slow"},
        )
        # Should not raise
        handler(event)

    def test_skips_demotion_to_fast_tier(self):
        handler = self._get_handler()
        event = _make_event(
            "memory_tier_demotion",
            {"memory_id": "m1", "from_tier": "medium", "to_tier": "fast"},
        )
        # Should return early since to_tier is not slow/glacial
        handler(event)

    def test_skips_demotion_to_medium_tier(self):
        handler = self._get_handler()
        event = _make_event(
            "memory_tier_demotion",
            {"memory_id": "m1", "from_tier": "slow", "to_tier": "medium"},
        )
        handler(event)

    def test_triggers_revalidation_on_demotion_to_slow(self):
        handler = self._get_handler()
        mock_mound = MagicMock()
        mock_mound.mark_for_revalidation = MagicMock()

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
            create=True,
        ):
            event = _make_event(
                "memory_tier_demotion",
                {"memory_id": "m1", "from_tier": "medium", "to_tier": "slow"},
            )
            handler(event)
            mock_mound.mark_for_revalidation.assert_called_once_with(
                source="continuum:m1",
                reason="tier_demotion:medium->slow",
            )

    def test_triggers_revalidation_on_demotion_to_glacial(self):
        handler = self._get_handler()
        mock_mound = MagicMock()
        mock_mound.mark_for_revalidation = MagicMock()

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
            create=True,
        ):
            event = _make_event(
                "memory_tier_demotion",
                {"memory_id": "m2", "from_tier": "slow", "to_tier": "glacial"},
            )
            handler(event)
            mock_mound.mark_for_revalidation.assert_called_once_with(
                source="continuum:m2",
                reason="tier_demotion:slow->glacial",
            )

    def test_graceful_when_mound_unavailable(self):
        handler = self._get_handler()
        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=None,
            create=True,
        ):
            event = _make_event(
                "memory_tier_demotion",
                {"memory_id": "m3", "from_tier": "medium", "to_tier": "slow"},
            )
            # Should not raise
            handler(event)

    def test_graceful_on_import_error(self):
        handler = self._get_handler()
        with patch.dict("sys.modules", {"aragora.knowledge.mound": None}):
            event = _make_event(
                "memory_tier_demotion",
                {"memory_id": "m4", "from_tier": "medium", "to_tier": "slow"},
            )
            # Should handle ImportError gracefully
            handler(event)

    def test_graceful_on_mound_exception(self):
        handler = self._get_handler()
        mock_mound = MagicMock()
        mock_mound.mark_for_revalidation.side_effect = RuntimeError("test error")

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
            create=True,
        ):
            event = _make_event(
                "memory_tier_demotion",
                {"memory_id": "m5", "from_tier": "medium", "to_tier": "slow"},
            )
            # Should handle exception gracefully
            handler(event)


class TestTierPromotionToKnowledge:
    """Test _handle_tier_promotion_to_knowledge handler.

    Relocated from BasicHandlersMixin (P4a Batch E2c): it is knowledge-coupled.
    """

    def _get_handler(self):
        from aragora.knowledge.event_subscribers import KnowledgeEventSubscriber

        return KnowledgeEventSubscriber()._handle_tier_promotion_to_knowledge

    def test_skips_empty_memory_id(self):
        handler = self._get_handler()
        event = _make_event(
            "memory_tier_promotion",
            {"memory_id": "", "to_tier": "fast", "surprise_score": 0.9},
        )
        handler(event)

    def test_boosts_importance_on_promotion(self):
        handler = self._get_handler()
        mock_mound = MagicMock()
        mock_mound.boost_importance = MagicMock()

        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=mock_mound,
            create=True,
        ):
            event = _make_event(
                "memory_tier_promotion",
                {"memory_id": "m1", "to_tier": "fast", "surprise_score": 0.8},
            )
            handler(event)
            mock_mound.boost_importance.assert_called_once_with(
                source="continuum:m1",
                factor=1.8,
            )

    def test_graceful_when_mound_unavailable(self):
        handler = self._get_handler()
        with patch(
            "aragora.knowledge.mound.get_knowledge_mound",
            return_value=None,
            create=True,
        ):
            event = _make_event(
                "memory_tier_promotion",
                {"memory_id": "m2", "to_tier": "fast", "surprise_score": 0.5},
            )
            handler(event)


class TestCrossSubscriberRegistration:
    """Test that tier handlers are registered in CrossSubscriberManager.

    Both handlers live on ``KnowledgeEventSubscriber`` (P4a Batch E2c) and are
    wired into the manager only via ``apply_registered_subscribers`` at
    bootstrap, so these checks use a bootstrapped manager rather than a bare
    ``CrossSubscriberManager()``.
    """

    @staticmethod
    def _bootstrapped_manager():
        from aragora.events.cross_subscribers import reset_cross_subscriber_manager
        from aragora.debate.event_subscribers import bootstrap_debate_event_subscribers

        reset_cross_subscriber_manager()
        return bootstrap_debate_event_subscribers()

    def test_demotion_handler_registered(self):
        manager = self._bootstrapped_manager()
        # Check the subscriber is registered
        subs = manager._subscribers.get(StreamEventType.MEMORY_TIER_DEMOTION, [])
        names = [name for name, _ in subs]
        assert "tier_demotion_to_revalidation" in names

    def test_promotion_handler_registered(self):
        manager = self._bootstrapped_manager()
        subs = manager._subscribers.get(StreamEventType.MEMORY_TIER_PROMOTION, [])
        names = [name for name, _ in subs]
        assert "tier_promotion_to_knowledge" in names

    def test_event_types_exist(self):
        assert hasattr(StreamEventType, "MEMORY_TIER_PROMOTION")
        assert hasattr(StreamEventType, "MEMORY_TIER_DEMOTION")
        assert StreamEventType.MEMORY_TIER_PROMOTION.value == "memory_tier_promotion"
        assert StreamEventType.MEMORY_TIER_DEMOTION.value == "memory_tier_demotion"
