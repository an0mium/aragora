"""Tests for the NomicLoop._maybe_publish_to_marketplace method."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.nomic_loop import NomicLoop


@pytest.fixture
def loop():
    """Create a minimal NomicLoop instance for testing."""
    with patch.multiple(
        "scripts.nomic_loop",
        Arena=MagicMock(),
        Environment=MagicMock(),
        DebateProtocol=MagicMock(),
    ):
        nl = NomicLoop.__new__(NomicLoop)
        nl.cycle_count = 1
        nl.log_file = None
        nl._log_buffer = []
        nl._current_cycle_record = None
        nl._log = MagicMock()
        return nl


class TestMaybePublishToMarketplace:
    def test_disabled_by_default(self, loop: NomicLoop) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = loop._maybe_publish_to_marketplace("improvement", 0.9)
        assert result is None

    def test_enabled_but_low_confidence(self, loop: NomicLoop) -> None:
        with patch.dict(os.environ, {"ARAGORA_AUTO_PUBLISH_MARKETPLACE": "1"}):
            result = loop._maybe_publish_to_marketplace("improvement", 0.5)
        assert result is not None
        assert result["published"] is False
        assert result["reason"] == "below_confidence_threshold"

    def test_custom_min_confidence(self, loop: NomicLoop) -> None:
        env = {
            "ARAGORA_AUTO_PUBLISH_MARKETPLACE": "1",
            "ARAGORA_MARKETPLACE_MIN_CONFIDENCE": "0.5",
        }
        with patch.dict(os.environ, env):
            with patch("scripts.nomic_loop.SkillPublisher", create=True) as mock_pub_cls:
                mock_pub = MagicMock()
                mock_pub.publish = AsyncMock(return_value=(True, MagicMock(id="listing-1"), []))
                mock_pub_cls.return_value = mock_pub

                with patch("scripts.nomic_loop.Skill", create=True) as mock_skill_cls:
                    mock_skill_cls.return_value = MagicMock(name="nomic-test")

                    # Patch the imports inside the method
                    with patch.dict(
                        "sys.modules",
                        {
                            "aragora.skills.publisher": MagicMock(SkillPublisher=mock_pub_cls),
                            "aragora.skills.base": MagicMock(Skill=mock_skill_cls),
                        },
                    ):
                        result = loop._maybe_publish_to_marketplace("improvement", 0.6)
        # With min_confidence=0.5 and confidence=0.6, it should attempt publishing
        # The actual publish may or may not succeed depending on event loop state
        assert result is not None

    def test_import_error_handled(self, loop: NomicLoop) -> None:
        env = {"ARAGORA_AUTO_PUBLISH_MARKETPLACE": "1"}
        with patch.dict(os.environ, env):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = loop._maybe_publish_to_marketplace("improvement", 0.9)
        assert result is not None
        assert result["published"] is False
        assert result["reason"] == "import_error"

    def test_runtime_error_handled(self, loop: NomicLoop) -> None:
        env = {"ARAGORA_AUTO_PUBLISH_MARKETPLACE": "1"}
        with patch.dict(os.environ, env):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.skills.publisher": MagicMock(
                        SkillPublisher=MagicMock(side_effect=RuntimeError("broken"))
                    ),
                    "aragora.skills.base": MagicMock(
                        Skill=MagicMock(),
                    ),
                },
            ):
                result = loop._maybe_publish_to_marketplace("improvement", 0.9)
        assert result is not None
        assert result["published"] is False

    def test_confidence_at_threshold(self, loop: NomicLoop) -> None:
        env = {"ARAGORA_AUTO_PUBLISH_MARKETPLACE": "1"}
        with patch.dict(os.environ, env):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.skills.publisher": MagicMock(
                        SkillPublisher=MagicMock(side_effect=RuntimeError("test"))
                    ),
                    "aragora.skills.base": MagicMock(
                        Skill=MagicMock(),
                    ),
                },
            ):
                # At exactly 0.85, should NOT be skipped (>= comparison)
                result = loop._maybe_publish_to_marketplace("improvement", 0.85)
        assert result is not None
        # It will attempt publish, not be skipped for confidence
        assert result.get("reason") != "below_confidence_threshold"

    def test_confidence_just_below_threshold(self, loop: NomicLoop) -> None:
        env = {"ARAGORA_AUTO_PUBLISH_MARKETPLACE": "1"}
        with patch.dict(os.environ, env):
            result = loop._maybe_publish_to_marketplace("improvement", 0.84)
        assert result is not None
        assert result["published"] is False
        assert result["reason"] == "below_confidence_threshold"

    def test_disabled_with_explicit_zero(self, loop: NomicLoop) -> None:
        with patch.dict(os.environ, {"ARAGORA_AUTO_PUBLISH_MARKETPLACE": "0"}):
            result = loop._maybe_publish_to_marketplace("improvement", 0.9)
        assert result is None

    def test_skill_name_sanitized(self, loop: NomicLoop) -> None:
        env = {"ARAGORA_AUTO_PUBLISH_MARKETPLACE": "1"}
        with patch.dict(os.environ, env):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.skills.publisher": MagicMock(
                        SkillPublisher=MagicMock(side_effect=RuntimeError("test"))
                    ),
                    "aragora.skills.base": MagicMock(
                        Skill=MagicMock(),
                    ),
                },
            ):
                result = loop._maybe_publish_to_marketplace(
                    "Fix the_broken thing\nExtra details", 0.9
                )
        # Should not crash even with special chars in improvement text
        assert result is not None

    def test_empty_improvement_handled(self, loop: NomicLoop) -> None:
        env = {"ARAGORA_AUTO_PUBLISH_MARKETPLACE": "1"}
        with patch.dict(os.environ, env):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.skills.publisher": MagicMock(
                        SkillPublisher=MagicMock(side_effect=RuntimeError("test"))
                    ),
                    "aragora.skills.base": MagicMock(
                        Skill=MagicMock(),
                    ),
                },
            ):
                result = loop._maybe_publish_to_marketplace("", 0.9)
        assert result is not None
