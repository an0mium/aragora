"""Tests for ArgumentStructureVerifier wiring in PostDebateCoordinator.

Validates that:
1. auto_verify_arguments defaults to False
2. step runs when enabled and debate has messages
3. step is skipped when disabled (default)
4. import error handled gracefully
5. verification result stored on PostDebateResult
6. verification failure doesn't cascade to other steps
7. empty messages produce no verification
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.post_debate_coordinator import (
    PostDebateConfig,
    PostDebateCoordinator,
    PostDebateResult,
)


class TestArgumentVerificationConfigDefaults:
    """Tests for argument verification config defaults."""

    def test_auto_verify_arguments_default_false(self):
        config = PostDebateConfig()
        assert config.auto_verify_arguments is False

    def test_verify_arguments_enabled_via_config(self):
        config = PostDebateConfig(auto_verify_arguments=True)
        assert config.auto_verify_arguments is True


class TestArgumentVerificationStep:
    """Tests for _step_argument_verification method."""

    def _make_coordinator(self, **config_kwargs):
        defaults = {
            "auto_verify_arguments": True,
            "auto_explain": False,
            "auto_create_plan": False,
            "auto_notify": False,
            "auto_persist_receipt": False,
            "auto_gauntlet_validate": False,
            "auto_execution_bridge": False,
            "auto_push_calibration": False,
        }
        defaults.update(config_kwargs)
        config = PostDebateConfig(**defaults)
        return PostDebateCoordinator(config=config)

    def _make_debate_result(self):
        result = MagicMock()
        msg1 = MagicMock()
        msg1.agent = "agent_a"
        msg1.content = "We should use microservices for scalability"
        msg1.role = "proposal"
        msg1.round = 0

        msg2 = MagicMock()
        msg2.agent = "agent_b"
        msg2.content = "Microservices add complexity, a monolith is simpler"
        msg2.role = "critique"
        msg2.round = 1

        msg3 = MagicMock()
        msg3.agent = "agent_a"
        msg3.content = "Revised: use modular monolith as a compromise"
        msg3.role = "proposal"
        msg3.round = 2

        result.messages = [msg1, msg2, msg3]
        result.final_answer = "Use modular monolith"
        result.consensus = "Use modular monolith"
        result.confidence = 0.9
        result.participants = ["agent_a", "agent_b"]
        result.task = "Decide on architecture"
        return result

    def test_step_runs_when_enabled(self):
        """When auto_verify_arguments is True, the step executes."""
        coordinator = self._make_coordinator()
        mock_result = self._make_debate_result()

        mock_verification = MagicMock()
        mock_verification.to_dict.return_value = {
            "valid_chains": [],
            "invalid_chains": [],
            "is_sound": True,
            "soundness_score": 1.0,
        }
        mock_verification.is_sound = True
        mock_verification.soundness_score = 1.0

        mock_verifier = MagicMock()
        mock_verifier.verify = MagicMock(return_value=mock_verification)

        with patch(
            "aragora.debate.post_debate_coordinator.PostDebateCoordinator._step_argument_verification",
            return_value={
                "debate_id": "d1",
                "verification": mock_verification.to_dict(),
                "is_sound": True,
                "soundness_score": 1.0,
            },
        ) as mock_step:
            result = coordinator.run("d1", mock_result, confidence=0.9, task="test")

        mock_step.assert_called_once_with("d1", mock_result, "test")
        assert result.argument_verification is not None
        assert result.argument_verification["is_sound"] is True
        assert result.argument_verification["soundness_score"] == 1.0

    def test_step_skipped_when_disabled(self):
        """When auto_verify_arguments is False (default), the step is skipped."""
        config = PostDebateConfig(
            auto_verify_arguments=False,
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_execution_bridge=False,
            auto_push_calibration=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        mock_result = self._make_debate_result()

        with patch.object(coordinator, "_step_argument_verification") as mock_step:
            result = coordinator.run("d1", mock_result, confidence=0.9, task="test")

        mock_step.assert_not_called()
        assert result.argument_verification is None

    def test_import_error_returns_none(self):
        """When ArgumentStructureVerifier can't be imported, step returns None gracefully."""
        coordinator = self._make_coordinator()

        with patch("builtins.__import__", side_effect=ImportError("no verifier")):
            result = coordinator._step_argument_verification("d1", MagicMock(), "test task")

        assert result is None

    def test_graceful_failure_on_verifier_import_error(self):
        """Simulate import failure for argument_verifier specifically."""
        coordinator = self._make_coordinator()

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def selective_import(name, *args, **kwargs):
            if "argument_verifier" in name:
                raise ImportError("argument_verifier not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=selective_import):
            result = coordinator._step_argument_verification(
                "d1", self._make_debate_result(), "test"
            )

        assert result is None

    def test_result_attached_to_pipeline_output(self):
        """Verification result is stored on PostDebateResult.argument_verification."""
        result = PostDebateResult()
        assert result.argument_verification is None

        verification_data = {
            "debate_id": "d1",
            "verification": {"valid_chains": [], "invalid_chains": []},
            "is_sound": True,
            "soundness_score": 0.95,
        }
        result.argument_verification = verification_data
        assert result.argument_verification["soundness_score"] == 0.95
        assert result.argument_verification["is_sound"] is True

    def test_verification_failure_doesnt_cascade(self):
        """When verification returns None, other steps still execute."""
        coordinator = self._make_coordinator(
            auto_explain=True,
            auto_persist_receipt=True,
        )
        mock_result = self._make_debate_result()

        with (
            patch.object(
                coordinator, "_step_explain", return_value={"explanation": "test"}
            ) as mock_explain,
            patch.object(coordinator, "_step_argument_verification", return_value=None),
            patch.object(coordinator, "_step_persist_receipt", return_value=True) as mock_receipt,
        ):
            result = coordinator.run("d1", mock_result, confidence=0.9, task="test")

        mock_explain.assert_called_once()
        mock_receipt.assert_called_once()
        assert result.argument_verification is None
        assert result.explanation == {"explanation": "test"}
        assert result.receipt_persisted is True

    def test_empty_messages_returns_none(self):
        """When debate has no messages, verification returns None."""
        coordinator = self._make_coordinator()
        mock_result = MagicMock()
        mock_result.messages = []

        mock_graph_cls = MagicMock()
        mock_graph_instance = MagicMock()
        mock_graph_instance.nodes = {}
        mock_graph_cls.return_value = mock_graph_instance

        with (
            patch("aragora.verification.argument_verifier.ArgumentStructureVerifier"),
            patch(
                "aragora.visualization.mapper.ArgumentCartographer",
                mock_graph_cls,
            ),
        ):
            result = coordinator._step_argument_verification("d1", mock_result, "test task")

        assert result is None

    def test_runtime_error_returns_none(self):
        """When verifier raises RuntimeError, step returns None."""
        coordinator = self._make_coordinator()
        mock_result = self._make_debate_result()

        mock_graph_cls = MagicMock()
        mock_graph_instance = MagicMock()
        mock_graph_instance.nodes = {"n1": MagicMock()}
        mock_graph_cls.return_value = mock_graph_instance

        mock_verifier_cls = MagicMock()
        mock_verifier = MagicMock()
        mock_verifier.verify.side_effect = RuntimeError("verification engine broken")
        mock_verifier_cls.return_value = mock_verifier

        with (
            patch(
                "aragora.verification.argument_verifier.ArgumentStructureVerifier",
                mock_verifier_cls,
            ),
            patch(
                "aragora.visualization.mapper.ArgumentCartographer",
                mock_graph_cls,
            ),
            patch(
                "asyncio.run",
                side_effect=RuntimeError("verification engine broken"),
            ),
        ):
            result = coordinator._step_argument_verification("d1", mock_result, "test task")

        assert result is None

    def test_step_with_mock_verifier_returns_dict(self):
        """Full integration: mock the verifier to return a result dict."""
        coordinator = self._make_coordinator()
        mock_result = self._make_debate_result()

        mock_verification_result = MagicMock()
        mock_verification_result.to_dict.return_value = {
            "valid_chains": [{"chain_id": "c1", "name": "test_chain"}],
            "invalid_chains": [],
            "is_sound": True,
            "soundness_score": 0.95,
            "total_nodes_analyzed": 3,
            "total_chains_checked": 1,
        }
        mock_verification_result.is_sound = True
        mock_verification_result.soundness_score = 0.95

        mock_graph_cls = MagicMock()
        mock_graph_instance = MagicMock()
        mock_graph_instance.nodes = {"n1": MagicMock(), "n2": MagicMock(), "n3": MagicMock()}
        mock_graph_cls.return_value = mock_graph_instance

        mock_verifier_cls = MagicMock()
        mock_verifier = MagicMock()
        mock_verifier_cls.return_value = mock_verifier

        with (
            patch(
                "aragora.verification.argument_verifier.ArgumentStructureVerifier",
                mock_verifier_cls,
            ),
            patch(
                "aragora.visualization.mapper.ArgumentCartographer",
                mock_graph_cls,
            ),
            patch(
                "asyncio.run",
                return_value=mock_verification_result,
            ),
        ):
            result = coordinator._step_argument_verification(
                "d1", mock_result, "Decide on architecture"
            )

        assert result is not None
        assert result["debate_id"] == "d1"
        assert result["is_sound"] is True
        assert result["soundness_score"] == 0.95
        assert "verification" in result

    def test_step_ordering_after_gauntlet_before_notify(self):
        """Argument verification runs after gauntlet and before notify."""
        call_order = []

        coordinator = self._make_coordinator(
            auto_gauntlet_validate=True,
            auto_verify_arguments=True,
            auto_notify=True,
            auto_explain=False,
            auto_persist_receipt=False,
        )
        mock_result = self._make_debate_result()

        def track_gauntlet(*args, **kwargs):
            call_order.append("gauntlet")
            return None

        def track_verify(*args, **kwargs):
            call_order.append("argument_verification")
            return None

        def track_notify(*args, **kwargs):
            call_order.append("notify")
            return False

        with (
            patch.object(coordinator, "_step_gauntlet_validate", side_effect=track_gauntlet),
            patch.object(coordinator, "_step_argument_verification", side_effect=track_verify),
            patch.object(coordinator, "_step_notify", side_effect=track_notify),
            patch.object(coordinator, "_step_execution_bridge", return_value=[]),
        ):
            coordinator.run("d1", mock_result, confidence=0.95, task="test")

        assert call_order == ["gauntlet", "argument_verification", "notify"]
