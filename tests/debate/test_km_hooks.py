"""
Tests for Knowledge Mound hook integration.

Tests the KM handlers in HookHandlerRegistry that enable bidirectional
data flow between debates and the Knowledge Mound.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from aragora.debate.hook_handlers import (
    HookHandlerRegistry,
    create_hook_handler_registry,
)


class MockHookManager:
    """Mock HookManager for testing."""

    def __init__(self):
        self.registered_hooks = {}
        self.callbacks = []

    def register(self, hook_type, callback, name=None, priority=None):
        """Register a hook callback."""
        key = (hook_type, name)
        self.registered_hooks[key] = {
            "callback": callback,
            "priority": priority,
        }
        self.callbacks.append((hook_type, name, callback))

        def unregister():
            if key in self.registered_hooks:
                del self.registered_hooks[key]

        return unregister

    def get_registered(self, hook_type):
        """Get registered callbacks for a hook type."""
        return [
            entry for (ht, _), entry in self.registered_hooks.items()
            if ht == hook_type
        ]


class MockKnowledgeMound:
    """Mock Knowledge Mound for testing."""

    def __init__(self):
        self.debate_end_calls = []
        self.consensus_calls = []
        self.outcome_calls = []

    def on_debate_end(self, ctx, result):
        """Handle debate end."""
        self.debate_end_calls.append({"ctx": ctx, "result": result})

    def on_consensus_reached(self, ctx, consensus_text, confidence):
        """Handle consensus reached."""
        self.consensus_calls.append({
            "ctx": ctx,
            "consensus_text": consensus_text,
            "confidence": confidence,
        })

    def on_outcome_tracked(self, ctx, outcome):
        """Handle outcome tracked."""
        self.outcome_calls.append({"ctx": ctx, "outcome": outcome})


class MockKMCoordinator:
    """Mock BidirectionalCoordinator for testing."""

    def __init__(self):
        self.debate_complete_calls = []
        self.consensus_calls = []

    def on_debate_complete(self, ctx, result):
        """Handle debate complete."""
        self.debate_complete_calls.append({"ctx": ctx, "result": result})

    def on_consensus_reached(self, ctx, consensus_text, confidence):
        """Handle consensus reached."""
        self.consensus_calls.append({
            "ctx": ctx,
            "consensus_text": consensus_text,
            "confidence": confidence,
        })


class TestKMHookRegistration:
    """Tests for KM hook registration."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""
        return MockHookManager()

    @pytest.fixture
    def km(self):
        """Create mock knowledge mound."""
        return MockKnowledgeMound()

    @pytest.fixture
    def coordinator(self):
        """Create mock coordinator."""
        return MockKMCoordinator()

    def test_register_km_handlers(self, hook_manager, km):
        """Test registering KM handlers."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"knowledge_mound": km},
        )

        count = registry.register_all()

        # Should register 3 KM handlers
        assert count >= 3

        # Verify handler names
        handler_names = [name for _, name, _ in hook_manager.callbacks]
        assert "km_debate_end" in handler_names
        assert "km_consensus" in handler_names
        assert "km_outcome_tracked" in handler_names

    def test_register_coordinator_handlers(self, hook_manager, coordinator):
        """Test registering coordinator handlers."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"km_coordinator": coordinator},
        )

        count = registry.register_all()

        # Should register 2 coordinator handlers
        assert count >= 2

        # Verify handler names
        handler_names = [name for _, name, _ in hook_manager.callbacks]
        assert "km_coordinator_sync" in handler_names
        assert "km_coordinator_consensus" in handler_names

    def test_register_both_km_and_coordinator(self, hook_manager, km, coordinator):
        """Test registering both KM and coordinator handlers."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={
                "knowledge_mound": km,
                "km_coordinator": coordinator,
            },
        )

        count = registry.register_all()

        # Should register 5 handlers total
        assert count >= 5

    def test_no_handlers_without_subsystems(self, hook_manager):
        """Test no handlers registered without subsystems."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={},
        )

        count = registry.register_all()

        # No KM-specific handlers
        km_handlers = [
            name for _, name, _ in hook_manager.callbacks
            if name and "km" in name.lower()
        ]
        assert len(km_handlers) == 0


class TestKMHookExecution:
    """Tests for KM hook execution."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""
        return MockHookManager()

    @pytest.fixture
    def km(self):
        """Create mock knowledge mound."""
        return MockKnowledgeMound()

    @pytest.fixture
    def coordinator(self):
        """Create mock coordinator."""
        return MockKMCoordinator()

    def test_km_debate_end_called(self, hook_manager, km):
        """Test KM debate end handler is called."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"knowledge_mound": km},
        )
        registry.register_all()

        # Find and call the debate end handler
        for hook_type, name, callback in hook_manager.callbacks:
            if name == "km_debate_end":
                callback(ctx={"debate_id": "d1"}, result={"success": True})
                break

        assert len(km.debate_end_calls) == 1
        assert km.debate_end_calls[0]["ctx"]["debate_id"] == "d1"

    def test_km_consensus_called(self, hook_manager, km):
        """Test KM consensus handler is called."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"knowledge_mound": km},
        )
        registry.register_all()

        # Find and call the consensus handler
        for hook_type, name, callback in hook_manager.callbacks:
            if name == "km_consensus":
                callback(
                    ctx={"debate_id": "d1"},
                    consensus_text="Agreement reached",
                    confidence=0.85,
                )
                break

        assert len(km.consensus_calls) == 1
        assert km.consensus_calls[0]["consensus_text"] == "Agreement reached"
        assert km.consensus_calls[0]["confidence"] == 0.85

    def test_km_outcome_tracked_called(self, hook_manager, km):
        """Test KM outcome tracked handler is called."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"knowledge_mound": km},
        )
        registry.register_all()

        # Find and call the outcome handler
        for hook_type, name, callback in hook_manager.callbacks:
            if name == "km_outcome_tracked":
                callback(
                    ctx={"debate_id": "d1"},
                    outcome={"success": True, "confidence": 0.9},
                )
                break

        assert len(km.outcome_calls) == 1
        assert km.outcome_calls[0]["outcome"]["success"] is True

    def test_coordinator_sync_called(self, hook_manager, coordinator):
        """Test coordinator sync handler is called."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"km_coordinator": coordinator},
        )
        registry.register_all()

        # Find and call the coordinator sync handler
        for hook_type, name, callback in hook_manager.callbacks:
            if name == "km_coordinator_sync":
                callback(ctx={"debate_id": "d1"}, result={"success": True})
                break

        assert len(coordinator.debate_complete_calls) == 1

    def test_coordinator_consensus_called(self, hook_manager, coordinator):
        """Test coordinator consensus handler is called."""
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"km_coordinator": coordinator},
        )
        registry.register_all()

        # Find and call the coordinator consensus handler
        for hook_type, name, callback in hook_manager.callbacks:
            if name == "km_coordinator_consensus":
                callback(
                    ctx={"debate_id": "d1"},
                    consensus_text="Agreement",
                    confidence=0.8,
                )
                break

        assert len(coordinator.consensus_calls) == 1


class TestKMHookErrorHandling:
    """Tests for KM hook error handling."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""
        return MockHookManager()

    def test_km_handler_error_isolation(self, hook_manager):
        """Test that KM handler errors are isolated."""
        class FailingKM:
            def on_debate_end(self, ctx, result):
                raise ValueError("KM failure")

            def on_consensus_reached(self, ctx, text, confidence):
                pass

            def on_outcome_tracked(self, ctx, outcome):
                pass

        km = FailingKM()
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"knowledge_mound": km},
        )
        registry.register_all()

        # Find and call the debate end handler - should not raise
        for hook_type, name, callback in hook_manager.callbacks:
            if name == "km_debate_end":
                # Should not raise, error should be caught
                callback(ctx={}, result={})
                break

    def test_coordinator_handler_error_isolation(self, hook_manager):
        """Test that coordinator handler errors are isolated."""
        class FailingCoordinator:
            def on_debate_complete(self, ctx, result):
                raise ValueError("Coordinator failure")

            def on_consensus_reached(self, ctx, text, confidence):
                pass

        coordinator = FailingCoordinator()
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"km_coordinator": coordinator},
        )
        registry.register_all()

        # Find and call the coordinator sync handler - should not raise
        for hook_type, name, callback in hook_manager.callbacks:
            if name == "km_coordinator_sync":
                # Should not raise, error should be caught
                callback(ctx={}, result={})
                break


class TestCreateHookHandlerRegistry:
    """Tests for create_hook_handler_registry function."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""
        return MockHookManager()

    def test_create_with_km(self, hook_manager):
        """Test creating registry with KM."""
        km = MockKnowledgeMound()

        registry = create_hook_handler_registry(
            hook_manager,
            knowledge_mound=km,
            auto_register=True,
        )

        assert registry.is_registered
        assert "knowledge_mound" in registry.subsystems

    def test_create_with_coordinator(self, hook_manager):
        """Test creating registry with coordinator."""
        coordinator = MockKMCoordinator()

        registry = create_hook_handler_registry(
            hook_manager,
            km_coordinator=coordinator,
            auto_register=True,
        )

        assert registry.is_registered
        assert "km_coordinator" in registry.subsystems

    def test_create_with_both(self, hook_manager):
        """Test creating registry with both KM and coordinator."""
        km = MockKnowledgeMound()
        coordinator = MockKMCoordinator()

        registry = create_hook_handler_registry(
            hook_manager,
            knowledge_mound=km,
            km_coordinator=coordinator,
            auto_register=True,
        )

        assert registry.is_registered
        assert "knowledge_mound" in registry.subsystems
        assert "km_coordinator" in registry.subsystems

    def test_create_without_auto_register(self, hook_manager):
        """Test creating registry without auto-registration."""
        km = MockKnowledgeMound()

        registry = create_hook_handler_registry(
            hook_manager,
            knowledge_mound=km,
            auto_register=False,
        )

        assert not registry.is_registered
        assert registry.registered_count == 0


class TestKMHookPriority:
    """Tests for KM hook priorities."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""
        return MockHookManager()

    def test_km_handlers_have_correct_priority(self, hook_manager):
        """Test KM handlers have appropriate priorities."""
        km = MockKnowledgeMound()
        coordinator = MockKMCoordinator()

        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={
                "knowledge_mound": km,
                "km_coordinator": coordinator,
            },
        )
        registry.register_all()

        # Check priorities
        for (hook_type, name), entry in hook_manager.registered_hooks.items():
            if name and "km" in name.lower():
                # Coordinator handlers should be LOW priority
                if "coordinator" in name:
                    from aragora.debate.hooks import HookPriority
                    assert entry["priority"] == HookPriority.LOW
                # Main KM outcome tracked should be LOW
                elif "outcome_tracked" in name:
                    from aragora.debate.hooks import HookPriority
                    assert entry["priority"] == HookPriority.LOW


class TestKMHookUnregistration:
    """Tests for KM hook unregistration."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""
        return MockHookManager()

    def test_unregister_km_handlers(self, hook_manager):
        """Test unregistering KM handlers."""
        km = MockKnowledgeMound()
        coordinator = MockKMCoordinator()

        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={
                "knowledge_mound": km,
                "km_coordinator": coordinator,
            },
        )
        registry.register_all()

        initial_count = len(hook_manager.registered_hooks)
        assert initial_count > 0

        unregistered = registry.unregister_all()

        assert unregistered > 0
        assert not registry.is_registered


class TestKMHookIntegration:
    """Integration tests for KM hooks."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""
        return MockHookManager()

    def test_full_km_integration(self, hook_manager):
        """Test full KM integration with all handlers."""
        km = MockKnowledgeMound()
        coordinator = MockKMCoordinator()

        registry = create_hook_handler_registry(
            hook_manager,
            knowledge_mound=km,
            km_coordinator=coordinator,
            auto_register=True,
        )

        # Simulate debate lifecycle
        ctx = {"debate_id": "test_debate"}

        # 1. Consensus reached
        for hook_type, name, callback in hook_manager.callbacks:
            if "consensus" in name:
                callback(
                    ctx=ctx,
                    consensus_text="Agreement reached",
                    confidence=0.9,
                )

        # 2. Debate ends
        for hook_type, name, callback in hook_manager.callbacks:
            if "debate_end" in name or "sync" in name:
                callback(ctx=ctx, result={"success": True})

        # 3. Outcome tracked
        for hook_type, name, callback in hook_manager.callbacks:
            if "outcome_tracked" in name:
                callback(ctx=ctx, outcome={"success": True})

        # Verify all handlers were called
        assert len(km.debate_end_calls) >= 1
        assert len(km.consensus_calls) >= 1
        assert len(km.outcome_calls) >= 1
        assert len(coordinator.debate_complete_calls) >= 1
        assert len(coordinator.consensus_calls) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
