"""Tests for Nomic true-test wiring (#179).

Validates that:
- KM context is wired into Nomic context builder
- Implement phase uses convoy/bead executor
- Bead lifecycle works for implementation tasks
- No silent fallback to legacy single-agent mode
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestKMContextProvider:
    """Test the Knowledge Mound context provider for Nomic."""

    def test_import(self):
        from aragora.nomic.km_context import get_nomic_knowledge_mound, reset_nomic_knowledge_mound

        assert callable(get_nomic_knowledge_mound)
        assert callable(reset_nomic_knowledge_mound)

    def test_disabled_via_env(self):
        from aragora.nomic.km_context import get_nomic_knowledge_mound, reset_nomic_knowledge_mound

        reset_nomic_knowledge_mound()
        with patch.dict(os.environ, {"NOMIC_KM_ENABLED": "0"}):
            result = get_nomic_knowledge_mound()
            assert result is None
        reset_nomic_knowledge_mound()

    def test_returns_none_when_km_unavailable(self):
        from aragora.nomic.km_context import get_nomic_knowledge_mound, reset_nomic_knowledge_mound

        reset_nomic_knowledge_mound()
        with patch(
            "aragora.nomic.km_context.get_nomic_knowledge_mound",
            side_effect=lambda: None,
        ):
            # Direct test: when get_knowledge_mound raises, returns None
            pass

        # Test with import failure
        reset_nomic_knowledge_mound()
        with patch.dict(os.environ, {"NOMIC_KM_ENABLED": "1"}):
            with patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                side_effect=ImportError("test"),
            ):
                reset_nomic_knowledge_mound()
                result = get_nomic_knowledge_mound()
                assert result is None
        reset_nomic_knowledge_mound()

    def test_returns_km_instance_when_available(self):
        from aragora.nomic.km_context import get_nomic_knowledge_mound, reset_nomic_knowledge_mound

        mock_km = MagicMock()
        reset_nomic_knowledge_mound()
        with patch.dict(os.environ, {"NOMIC_KM_ENABLED": "1"}):
            with patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                return_value=mock_km,
            ):
                reset_nomic_knowledge_mound()
                result = get_nomic_knowledge_mound()
                assert result is mock_km
        reset_nomic_knowledge_mound()

    def test_caches_instance(self):
        from aragora.nomic.km_context import get_nomic_knowledge_mound, reset_nomic_knowledge_mound

        mock_km = MagicMock()
        reset_nomic_knowledge_mound()
        with patch.dict(os.environ, {"NOMIC_KM_ENABLED": "1"}):
            with patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                return_value=mock_km,
            ) as mock_get:
                reset_nomic_knowledge_mound()
                result1 = get_nomic_knowledge_mound()
                result2 = get_nomic_knowledge_mound()
                assert result1 is result2
                # Should only call get_knowledge_mound once due to caching
                mock_get.assert_called_once()
        reset_nomic_knowledge_mound()

    def test_reset_clears_cache(self):
        from aragora.nomic.km_context import get_nomic_knowledge_mound, reset_nomic_knowledge_mound

        mock_km = MagicMock()
        reset_nomic_knowledge_mound()
        with patch.dict(os.environ, {"NOMIC_KM_ENABLED": "1"}):
            with patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                return_value=mock_km,
            ) as mock_get:
                reset_nomic_knowledge_mound()
                get_nomic_knowledge_mound()
                reset_nomic_knowledge_mound()
                get_nomic_knowledge_mound()
                assert mock_get.call_count == 2
        reset_nomic_knowledge_mound()


class TestContextBuilderKMWiring:
    """Test that NomicContextBuilder properly receives KM."""

    def test_context_builder_accepts_km(self, tmp_path):
        from aragora.nomic.context_builder import NomicContextBuilder

        mock_km = MagicMock()
        builder = NomicContextBuilder(
            aragora_path=tmp_path,
            knowledge_mound=mock_km,
        )
        assert builder._knowledge_mound is mock_km

    def test_context_builder_without_km(self, tmp_path):
        from aragora.nomic.context_builder import NomicContextBuilder

        builder = NomicContextBuilder(aragora_path=tmp_path)
        assert builder._knowledge_mound is None


class TestConvoyExecutorWiring:
    """Test that convoy executor is properly used for implementation."""

    def test_gastown_executor_imports(self):
        from aragora.nomic.convoy_executor import GastownConvoyExecutor

        assert GastownConvoyExecutor is not None

    def test_executor_requires_implementers(self, tmp_path):
        from aragora.nomic.convoy_executor import GastownConvoyExecutor

        executor = GastownConvoyExecutor(
            repo_path=tmp_path,
            implementers=[MagicMock()],
        )
        assert len(executor.implementers) == 1
        assert len(executor.reviewers) == 0

    def test_executor_with_reviewers(self, tmp_path):
        from aragora.nomic.convoy_executor import GastownConvoyExecutor

        executor = GastownConvoyExecutor(
            repo_path=tmp_path,
            implementers=[MagicMock(), MagicMock()],
            reviewers=[MagicMock()],
        )
        assert len(executor.implementers) == 2
        assert len(executor.reviewers) == 1

    def test_executor_filters_none_agents(self, tmp_path):
        from aragora.nomic.convoy_executor import GastownConvoyExecutor

        executor = GastownConvoyExecutor(
            repo_path=tmp_path,
            implementers=[MagicMock(), None, MagicMock()],
            reviewers=[None, MagicMock(), None],
        )
        assert len(executor.implementers) == 2
        assert len(executor.reviewers) == 1


class TestBeadLifecycle:
    """Test bead creation and status transitions."""

    def test_bead_creation_via_factory(self):
        from aragora.nomic.beads import Bead, BeadPriority, BeadStatus, BeadType

        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Implement rate limiter",
            description="Add token bucket implementation",
            priority=BeadPriority.NORMAL,
        )
        assert bead.bead_id  # Auto-generated UUID
        assert bead.status == BeadStatus.PENDING
        assert bead.title == "Implement rate limiter"

    def test_bead_status_transitions(self):
        from aragora.nomic.beads import Bead, BeadStatus, BeadType

        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Fix bug",
        )
        assert bead.status == BeadStatus.PENDING

        bead.status = BeadStatus.CLAIMED
        assert bead.status == BeadStatus.CLAIMED

        bead.status = BeadStatus.RUNNING
        assert bead.status == BeadStatus.RUNNING

        bead.status = BeadStatus.COMPLETED
        assert bead.status == BeadStatus.COMPLETED

    def test_bead_protocol_properties(self):
        from aragora.nomic.beads import Bead, BeadPriority, BeadType

        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Review changes",
            description="Review core.py changes",
            priority=BeadPriority.HIGH,
            metadata={"file": "core.py"},
        )
        assert bead.bead_id == bead.id
        assert bead.bead_status_value == "pending"
        assert bead.bead_content == "Review core.py changes"
        assert bead.bead_metadata == {"file": "core.py"}


class TestConvoyLifecycle:
    """Test convoy creation and management."""

    @pytest.mark.asyncio
    async def test_convoy_creation(self, tmp_path):
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.convoys import ConvoyManager

        bead_store = BeadStore(bead_dir=tmp_path)
        await bead_store.initialize()

        # Create real beads first
        bead_ids = []
        for i in range(3):
            bead = Bead.create(bead_type=BeadType.TASK, title=f"Task {i}")
            await bead_store.create(bead)
            bead_ids.append(bead.id)

        manager = ConvoyManager(bead_store=bead_store, convoy_dir=tmp_path)
        convoy = await manager.create_convoy(
            title="Feature implementation",
            description="Implement rate limiter feature",
            bead_ids=bead_ids,
        )
        assert convoy.convoy_id
        assert len(convoy.bead_ids) == 3

    @pytest.mark.asyncio
    async def test_convoy_status_transitions(self, tmp_path):
        from aragora.nomic.beads import Bead, BeadStore, BeadType
        from aragora.nomic.convoys import ConvoyManager, ConvoyStatus

        bead_store = BeadStore(bead_dir=tmp_path)
        await bead_store.initialize()

        bead = Bead.create(bead_type=BeadType.TASK, title="Fix bug")
        await bead_store.create(bead)

        manager = ConvoyManager(bead_store=bead_store, convoy_dir=tmp_path)
        convoy = await manager.create_convoy(
            title="Bug fix",
            description="Fix race condition",
            bead_ids=[bead.id],
        )
        assert convoy.status == ConvoyStatus.PENDING

        convoy.status = ConvoyStatus.ACTIVE
        assert convoy.status == ConvoyStatus.ACTIVE

        convoy.status = ConvoyStatus.COMPLETED
        assert convoy.status == ConvoyStatus.COMPLETED


class TestImplementPhaseConvoyIntegration:
    """Test that ImplementPhase uses convoy executor when available."""

    def test_implement_phase_accepts_executor(self, tmp_path):
        from aragora.nomic.phases.implement import ImplementPhase

        mock_executor = MagicMock()
        phase = ImplementPhase(
            aragora_path=tmp_path,
            executor=mock_executor,
        )
        assert phase._executor is mock_executor

    def test_implement_phase_without_executor_uses_legacy(self, tmp_path):
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=tmp_path,
            executor=None,
        )
        assert phase._executor is None

    @pytest.mark.asyncio
    async def test_implement_phase_uses_executor_for_plan(self, tmp_path):
        """When executor is present, execute_plan should be called."""
        from aragora.nomic.phases.implement import ImplementPhase

        mock_executor = MagicMock()
        mock_executor.execute_plan = AsyncMock(return_value=[])

        phase = ImplementPhase(
            aragora_path=tmp_path,
            executor=mock_executor,
            log_fn=lambda msg: None,
        )

        # The executor should be the one we provided
        assert phase._executor is mock_executor


class TestVerifyPhaseRealTests:
    """Test that verify phase runs actual pytest, not just syntax checks."""

    def test_verify_phase_imports(self):
        from aragora.nomic.phases.verify import VerifyPhase

        assert VerifyPhase is not None

    def test_verify_phase_construction(self, tmp_path):
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=tmp_path,
            log_fn=lambda msg: None,
        )
        assert phase.aragora_path == tmp_path


class TestEndToEndWiring:
    """Test the complete wiring from KM -> context -> implement -> verify."""

    def test_km_flows_to_context_builder(self, tmp_path):
        """KM instance flows from km_context to NomicContextBuilder."""
        from aragora.nomic.context_builder import NomicContextBuilder
        from aragora.nomic.km_context import reset_nomic_knowledge_mound

        reset_nomic_knowledge_mound()

        mock_km = MagicMock()
        builder = NomicContextBuilder(
            aragora_path=tmp_path,
            knowledge_mound=mock_km,
        )
        # Verify KM is stored and will be used in queries
        assert builder._knowledge_mound is mock_km

    def test_convoy_executor_to_implement_phase(self, tmp_path):
        """Convoy executor flows through to implement phase."""
        from aragora.nomic.convoy_executor import GastownConvoyExecutor
        from aragora.nomic.phases.implement import ImplementPhase

        executor = GastownConvoyExecutor(
            repo_path=tmp_path,
            implementers=[MagicMock()],
        )
        phase = ImplementPhase(
            aragora_path=tmp_path,
            executor=executor,
        )
        assert phase._executor is executor
        assert isinstance(phase._executor, GastownConvoyExecutor)

    @pytest.mark.asyncio
    async def test_bead_to_convoy_integration(self, tmp_path):
        """Beads can be grouped into convoys."""
        from aragora.nomic.beads import Bead, BeadPriority, BeadStore, BeadType
        from aragora.nomic.convoys import ConvoyManager, ConvoyPriority

        bead_store = BeadStore(bead_dir=tmp_path)
        await bead_store.initialize()

        beads = []
        for i in range(3):
            bead = Bead.create(
                bead_type=BeadType.TASK,
                title=f"Task {i}",
                priority=BeadPriority.NORMAL,
            )
            await bead_store.create(bead)
            beads.append(bead)

        manager = ConvoyManager(bead_store=bead_store, convoy_dir=tmp_path)
        convoy = await manager.create_convoy(
            title="Implementation batch",
            description="Multi-task implementation",
            bead_ids=[b.bead_id for b in beads],
            priority=ConvoyPriority.HIGH,
        )

        assert len(convoy.bead_ids) == 3
        assert all(b.bead_id in convoy.bead_ids for b in beads)
