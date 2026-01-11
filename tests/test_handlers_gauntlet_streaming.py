"""Tests for Gauntlet streaming functionality.

Tests the WebSocket streaming integration for gauntlet stress-tests:
- set_gauntlet_broadcast_fn() registration
- GauntletStreamEmitter event emission
- Progress callbacks during gauntlet runs
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch


class TestSetGauntletBroadcastFn:
    """Tests for the set_gauntlet_broadcast_fn() function."""

    def test_set_broadcast_fn_stores_function(self):
        """set_gauntlet_broadcast_fn stores the provided function."""
        from aragora.server.handlers import gauntlet

        mock_fn = Mock()
        gauntlet.set_gauntlet_broadcast_fn(mock_fn)

        assert gauntlet._gauntlet_broadcast_fn is mock_fn

        # Cleanup
        gauntlet._gauntlet_broadcast_fn = None

    def test_set_broadcast_fn_replaces_existing(self):
        """set_gauntlet_broadcast_fn replaces any existing function."""
        from aragora.server.handlers import gauntlet

        mock_fn1 = Mock()
        mock_fn2 = Mock()

        gauntlet.set_gauntlet_broadcast_fn(mock_fn1)
        gauntlet.set_gauntlet_broadcast_fn(mock_fn2)

        assert gauntlet._gauntlet_broadcast_fn is mock_fn2

        # Cleanup
        gauntlet._gauntlet_broadcast_fn = None

    def test_set_broadcast_fn_accepts_none(self):
        """set_gauntlet_broadcast_fn can accept None to disable streaming."""
        from aragora.server.handlers import gauntlet

        gauntlet.set_gauntlet_broadcast_fn(Mock())
        gauntlet.set_gauntlet_broadcast_fn(None)

        assert gauntlet._gauntlet_broadcast_fn is None


class TestGauntletHandlerInit:
    """Tests for GauntletHandler initialization with streaming."""

    def test_handler_sets_broadcast_from_emitter(self):
        """Handler sets broadcast function from stream_emitter in context."""
        from aragora.server.handlers.gauntlet import GauntletHandler
        from aragora.server.handlers import gauntlet

        mock_emitter = Mock()
        mock_emitter.emit = Mock()

        handler = GauntletHandler({"stream_emitter": mock_emitter})

        assert gauntlet._gauntlet_broadcast_fn is mock_emitter.emit

        # Cleanup
        gauntlet._gauntlet_broadcast_fn = None

    def test_handler_ignores_missing_emitter(self):
        """Handler works without stream_emitter in context."""
        from aragora.server.handlers.gauntlet import GauntletHandler
        from aragora.server.handlers import gauntlet

        # Ensure clean state
        gauntlet._gauntlet_broadcast_fn = None

        handler = GauntletHandler({})

        # Should remain None
        assert gauntlet._gauntlet_broadcast_fn is None

    def test_handler_ignores_emitter_without_emit(self):
        """Handler ignores emitter without emit attribute."""
        from aragora.server.handlers.gauntlet import GauntletHandler
        from aragora.server.handlers import gauntlet

        # Ensure clean state
        gauntlet._gauntlet_broadcast_fn = None

        # Object without emit method
        mock_emitter = Mock(spec=[])

        handler = GauntletHandler({"stream_emitter": mock_emitter})

        # Should remain None since emitter has no emit
        assert gauntlet._gauntlet_broadcast_fn is None


class TestGauntletStreamEmitter:
    """Tests for GauntletStreamEmitter class."""

    @pytest.fixture
    def emitter(self):
        """Create a GauntletStreamEmitter with mock broadcast."""
        from aragora.server.stream.gauntlet_emitter import GauntletStreamEmitter

        mock_broadcast = Mock()
        return GauntletStreamEmitter(
            broadcast_fn=mock_broadcast,
            gauntlet_id="test-gauntlet-123",
        ), mock_broadcast

    def test_emitter_init(self, emitter):
        """Emitter initializes with correct state."""
        emitter_obj, _ = emitter

        assert emitter_obj.gauntlet_id == "test-gauntlet-123"
        assert emitter_obj._seq == 0
        assert emitter_obj._finding_count == 0
        assert emitter_obj._attack_count == 0
        assert emitter_obj._probe_count == 0

    def test_emit_start(self, emitter):
        """emit_start broadcasts gauntlet_start event."""
        emitter_obj, mock_broadcast = emitter

        emitter_obj.emit_start(
            gauntlet_id="gauntlet-456",
            input_type="spec",
            input_summary="Test input",
            agents=["claude", "gpt4"],
            config_summary={"profile": "default"},
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_start"
        assert event.data["gauntlet_id"] == "gauntlet-456"
        assert event.data["input_type"] == "spec"
        assert "claude" in event.data["agents"]

    def test_emit_progress(self, emitter):
        """emit_progress broadcasts gauntlet_progress event."""
        emitter_obj, mock_broadcast = emitter

        # Set start time for elapsed calculation
        emitter_obj._start_time = time.time() - 5  # 5 seconds ago

        emitter_obj.emit_progress(
            progress=0.5,
            phase="redteam",
            message="Running red team attacks",
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_progress"
        assert event.data["progress"] == 0.5
        assert event.data["phase"] == "redteam"
        assert event.data["elapsed_seconds"] >= 5

    def test_emit_phase(self, emitter):
        """emit_phase broadcasts gauntlet_phase event."""
        emitter_obj, mock_broadcast = emitter
        emitter_obj._start_time = time.time()

        emitter_obj.emit_phase("deep_audit", "Starting deep audit")

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_phase"
        assert event.data["phase"] == "deep_audit"

    def test_emit_finding(self, emitter):
        """emit_finding broadcasts gauntlet_finding event and increments counter."""
        emitter_obj, mock_broadcast = emitter

        emitter_obj.emit_finding(
            finding_id="finding-001",
            severity="high",
            category="injection",
            title="SQL Injection Vulnerability",
            description="Found potential SQL injection in user input",
            source="claude",
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_finding"
        assert event.data["severity"] == "high"
        assert event.data["finding_number"] == 1
        assert emitter_obj._finding_count == 1

    def test_emit_attack(self, emitter):
        """emit_attack broadcasts gauntlet_attack event and increments counter."""
        emitter_obj, mock_broadcast = emitter

        emitter_obj.emit_attack(
            attack_type="prompt_injection",
            agent="gpt4",
            target_summary="Test target",
            success=True,
            severity=0.8,
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_attack"
        assert event.data["attack_type"] == "prompt_injection"
        assert event.data["success"] is True
        assert emitter_obj._attack_count == 1

    def test_emit_probe(self, emitter):
        """emit_probe broadcasts gauntlet_probe event and increments counter."""
        emitter_obj, mock_broadcast = emitter

        emitter_obj.emit_probe(
            probe_type="capability_check",
            agent="claude",
            vulnerability_found=True,
            severity="medium",
            description="Found capability bypass",
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_probe"
        assert event.data["vulnerability_found"] is True
        assert emitter_obj._probe_count == 1

    def test_emit_verdict(self, emitter):
        """emit_verdict broadcasts gauntlet_verdict event."""
        emitter_obj, mock_broadcast = emitter

        emitter_obj.emit_verdict(
            verdict="HIGH_RISK",
            confidence=0.85,
            risk_score=0.75,
            robustness_score=0.4,
            critical_count=1,
            high_count=3,
            medium_count=5,
            low_count=2,
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_verdict"
        assert event.data["verdict"] == "HIGH_RISK"
        assert event.data["confidence"] == 0.85
        assert event.data["findings"]["critical"] == 1
        assert event.data["findings"]["total"] == 11

    def test_emit_complete(self, emitter):
        """emit_complete broadcasts gauntlet_complete event."""
        emitter_obj, mock_broadcast = emitter
        emitter_obj._attack_count = 10
        emitter_obj._probe_count = 5

        emitter_obj.emit_complete(
            gauntlet_id="gauntlet-789",
            verdict="APPROVED",
            confidence=0.95,
            findings_count=3,
            duration_seconds=120.5,
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_complete"
        assert event.data["verdict"] == "APPROVED"
        assert event.data["attacks_run"] == 10
        assert event.data["probes_run"] == 5
        assert event.data["duration_seconds"] == 120.5

    def test_emit_verification(self, emitter):
        """emit_verification broadcasts gauntlet_verification event."""
        emitter_obj, mock_broadcast = emitter

        emitter_obj.emit_verification(
            claim="All inputs are validated",
            verified=True,
            method="z3",
            proof_hash="abc123",
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_verification"
        assert event.data["verified"] is True
        assert event.data["method"] == "z3"

    def test_emit_risk(self, emitter):
        """emit_risk broadcasts gauntlet_risk event."""
        emitter_obj, mock_broadcast = emitter

        emitter_obj.emit_risk(
            risk_type="data_exposure",
            level="high",
            description="Potential PII leak detected",
            confidence=0.9,
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_risk"
        assert event.data["level"] == "high"

    def test_emit_agent_active(self, emitter):
        """emit_agent_active broadcasts gauntlet_agent_active event."""
        emitter_obj, mock_broadcast = emitter

        emitter_obj.emit_agent_active(
            agent="claude-api",
            role="auditor",
        )

        assert mock_broadcast.called
        event = mock_broadcast.call_args[0][0]
        assert event.type.value == "gauntlet_agent_active"
        assert event.data["agent"] == "claude-api"
        assert event.agent == "claude-api"

    def test_sequence_numbers_increment(self, emitter):
        """Events have incrementing sequence numbers."""
        emitter_obj, mock_broadcast = emitter
        emitter_obj._start_time = time.time()

        emitter_obj.emit_phase("init")
        emitter_obj.emit_progress(0.25)
        emitter_obj.emit_phase("redteam")

        calls = mock_broadcast.call_args_list
        seq_numbers = [call[0][0].seq for call in calls]

        assert seq_numbers == [1, 2, 3]


class TestGauntletStreamEmitterWithoutBroadcast:
    """Tests for GauntletStreamEmitter without broadcast function."""

    def test_emitter_works_without_broadcast_fn(self):
        """Emitter works when broadcast_fn is None."""
        from aragora.server.stream.gauntlet_emitter import GauntletStreamEmitter

        emitter = GauntletStreamEmitter(broadcast_fn=None)
        emitter._start_time = time.time()

        # Should not raise
        emitter.emit_start("test", "spec", "summary", ["agent"], {})
        emitter.emit_progress(0.5)
        emitter.emit_complete("test", "APPROVED", 0.9, 0, 10.0)

    def test_emitter_handles_broadcast_exception(self):
        """Emitter handles exceptions from broadcast_fn gracefully."""
        from aragora.server.stream.gauntlet_emitter import GauntletStreamEmitter

        def failing_broadcast(event):
            raise RuntimeError("Broadcast failed")

        emitter = GauntletStreamEmitter(broadcast_fn=failing_broadcast)
        emitter._start_time = time.time()

        # Should not raise despite broadcast failure
        emitter.emit_progress(0.5)


class TestCreateGauntletEmitter:
    """Tests for create_gauntlet_emitter factory function."""

    def test_create_emitter_returns_instance(self):
        """create_gauntlet_emitter returns a GauntletStreamEmitter."""
        from aragora.server.stream.gauntlet_emitter import (
            create_gauntlet_emitter,
            GauntletStreamEmitter,
        )

        mock_broadcast = Mock()
        emitter = create_gauntlet_emitter(broadcast_fn=mock_broadcast)

        assert isinstance(emitter, GauntletStreamEmitter)
        assert emitter.broadcast_fn is mock_broadcast

    def test_create_emitter_without_broadcast(self):
        """create_gauntlet_emitter works without broadcast_fn."""
        from aragora.server.stream.gauntlet_emitter import (
            create_gauntlet_emitter,
            GauntletStreamEmitter,
        )

        emitter = create_gauntlet_emitter()

        assert isinstance(emitter, GauntletStreamEmitter)
        assert emitter.broadcast_fn is None
