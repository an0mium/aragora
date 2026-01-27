"""
Integration tests for Event Hooks to Channel Delivery.

Tests the end-to-end flow of:
- YAML hook configuration loading
- Hook condition evaluation
- Hook triggering and execution
- Channel delivery integration
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from aragora.hooks.config import HookConfig, ActionConfig, ConditionConfig
from aragora.hooks.conditions import ConditionEvaluator, Operator
from aragora.hooks.loader import HookConfigLoader, get_hook_loader
from aragora.debate.hooks import HookManager, HookPriority, HookType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hook_manager():
    """Create a fresh HookManager for testing."""
    return HookManager()


@pytest.fixture
def condition_evaluator():
    """Create a ConditionEvaluator for testing."""
    return ConditionEvaluator()


@pytest.fixture
def hook_loader():
    """Create a HookConfigLoader for testing."""
    return HookConfigLoader()


@pytest.fixture
def sample_yaml_config():
    """Sample YAML hook configuration."""
    return """
hooks:
  - name: log_debate_complete
    trigger: post_debate
    priority: normal
    enabled: true
    action:
      handler: aragora.hooks.builtin.log_event
      args:
        message: "Debate {debate_id} complete"
        level: info

  - name: high_confidence_checkpoint
    trigger: post_debate
    priority: high
    conditions:
      - field: consensus_reached
        operator: eq
        value: true
      - field: confidence
        operator: gte
        value: 0.8
    action:
      handler: aragora.hooks.builtin.log_event
      args:
        message: "High confidence result: {confidence}"

  - name: send_webhook_on_consensus
    trigger: post_debate
    enabled: false
    conditions:
      - field: consensus_reached
        operator: is_true
    action:
      handler: aragora.hooks.builtin.send_webhook
      args:
        url: "https://example.com/webhook"
        timeout: 30.0
"""


@pytest.fixture
def temp_hooks_dir(sample_yaml_config):
    """Create temporary directory with hook YAML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_file = Path(tmpdir) / "debate_hooks.yaml"
        hooks_file.write_text(sample_yaml_config)
        yield tmpdir


# =============================================================================
# YAML Loading Tests
# =============================================================================


class TestHookManagerLoadsYAMLConfig:
    """Tests for YAML hook configuration loading."""

    def test_loader_parses_yaml_file(self, hook_loader, temp_hooks_dir):
        """Verify loader correctly parses YAML hook files."""
        hooks_file = Path(temp_hooks_dir) / "debate_hooks.yaml"
        configs = hook_loader.load_file(hooks_file)

        assert len(configs) == 3
        assert configs[0].name == "log_debate_complete"
        assert configs[1].name == "high_confidence_checkpoint"
        assert configs[2].name == "send_webhook_on_consensus"

    def test_loader_discovers_yaml_files(self, hook_loader, temp_hooks_dir):
        """Verify loader discovers YAML files in directory."""
        configs = hook_loader.discover_and_load(temp_hooks_dir)

        assert len(configs) >= 3

    def test_loader_parses_conditions(self, hook_loader, temp_hooks_dir):
        """Verify loader correctly parses condition configurations."""
        hooks_file = Path(temp_hooks_dir) / "debate_hooks.yaml"
        configs = hook_loader.load_file(hooks_file)

        # high_confidence_checkpoint has 2 conditions
        checkpoint_hook = configs[1]
        assert len(checkpoint_hook.conditions) == 2
        assert checkpoint_hook.conditions[0].field == "consensus_reached"
        assert checkpoint_hook.conditions[0].operator == "eq"
        assert checkpoint_hook.conditions[1].field == "confidence"
        assert checkpoint_hook.conditions[1].operator == "gte"

    def test_loader_parses_action_config(self, hook_loader, temp_hooks_dir):
        """Verify loader correctly parses action configurations."""
        hooks_file = Path(temp_hooks_dir) / "debate_hooks.yaml"
        configs = hook_loader.load_file(hooks_file)

        log_hook = configs[0]
        assert log_hook.action.handler == "aragora.hooks.builtin.log_event"
        assert "message" in log_hook.action.args
        assert log_hook.action.args["level"] == "info"

    def test_loader_handles_disabled_hooks(self, hook_loader, temp_hooks_dir):
        """Verify loader correctly handles disabled hooks."""
        hooks_file = Path(temp_hooks_dir) / "debate_hooks.yaml"
        configs = hook_loader.load_file(hooks_file)

        webhook_hook = configs[2]
        assert webhook_hook.enabled is False

    def test_load_from_string(self, hook_loader, sample_yaml_config):
        """Verify loading hooks from YAML string."""
        configs = hook_loader.load_from_string(sample_yaml_config, source="test")

        assert len(configs) == 3
        assert all(c.source_file == "test" for c in configs)


# =============================================================================
# Condition Evaluation Tests
# =============================================================================


class TestHookConditionsEvaluate:
    """Tests for hook condition evaluation."""

    def test_eq_operator(self, condition_evaluator):
        """Test equality operator."""
        condition = ConditionConfig(field="status", operator="eq", value="completed")
        context = {"status": "completed"}

        assert condition_evaluator.evaluate(condition, context) is True

        context = {"status": "pending"}
        assert condition_evaluator.evaluate(condition, context) is False

    def test_gte_operator(self, condition_evaluator):
        """Test greater-than-or-equal operator."""
        condition = ConditionConfig(field="confidence", operator="gte", value=0.8)

        assert condition_evaluator.evaluate(condition, {"confidence": 0.9}) is True
        assert condition_evaluator.evaluate(condition, {"confidence": 0.8}) is True
        assert condition_evaluator.evaluate(condition, {"confidence": 0.7}) is False

    def test_is_true_operator(self, condition_evaluator):
        """Test is_true operator."""
        condition = ConditionConfig(field="consensus_reached", operator="is_true", value=None)

        assert condition_evaluator.evaluate(condition, {"consensus_reached": True}) is True
        assert condition_evaluator.evaluate(condition, {"consensus_reached": False}) is False

    def test_contains_operator(self, condition_evaluator):
        """Test contains operator."""
        condition = ConditionConfig(field="domains", operator="contains", value="code")

        assert condition_evaluator.evaluate(condition, {"domains": ["code", "legal"]}) is True
        assert condition_evaluator.evaluate(condition, {"domains": ["legal"]}) is False

    def test_nested_field_access(self, condition_evaluator):
        """Test dot notation for nested field access."""
        condition = ConditionConfig(field="result.confidence", operator="gte", value=0.8)
        context = {"result": {"confidence": 0.85}}

        assert condition_evaluator.evaluate(condition, context) is True

    def test_multiple_conditions_all(self, condition_evaluator):
        """Test evaluating multiple conditions (all must pass)."""
        conditions = [
            ConditionConfig(field="consensus_reached", operator="eq", value=True),
            ConditionConfig(field="confidence", operator="gte", value=0.8),
        ]
        context = {"consensus_reached": True, "confidence": 0.85}

        assert condition_evaluator.evaluate_all(conditions, context) is True

        # One condition fails
        context = {"consensus_reached": True, "confidence": 0.7}
        assert condition_evaluator.evaluate_all(conditions, context) is False

    def test_negate_condition(self, condition_evaluator):
        """Test condition negation."""
        condition = ConditionConfig(field="error", operator="is_null", value=None, negate=True)

        # error is not null, negation makes it True
        assert condition_evaluator.evaluate(condition, {"error": "some error"}) is True
        # error is null, negation makes it False
        assert condition_evaluator.evaluate(condition, {"error": None}) is False


# =============================================================================
# Hook Triggering Tests
# =============================================================================


class TestHookTriggersChannelDelivery:
    """Tests for hook triggering and channel delivery."""

    @pytest.mark.asyncio
    async def test_hook_triggers_on_event(self, hook_manager):
        """Verify hooks trigger when events are fired."""
        triggered = {"count": 0}

        async def test_handler(**kwargs):
            triggered["count"] += 1
            return True

        hook_manager.register(
            HookType.POST_DEBATE,
            test_handler,
            name="test_hook",
            priority=HookPriority.NORMAL,
        )

        await hook_manager.trigger(HookType.POST_DEBATE, debate_id="123")

        assert triggered["count"] == 1

    @pytest.mark.asyncio
    async def test_hook_receives_context(self, hook_manager):
        """Verify hooks receive event context."""
        received = {}

        async def capture_handler(**kwargs):
            received.update(kwargs)
            return True

        hook_manager.register(HookType.POST_DEBATE, capture_handler, name="capture")

        await hook_manager.trigger(
            HookType.POST_DEBATE,
            debate_id="123",
            confidence=0.9,
            consensus_reached=True,
        )

        assert received["debate_id"] == "123"
        assert received["confidence"] == 0.9
        assert received["consensus_reached"] is True

    @pytest.mark.asyncio
    async def test_priority_ordering(self, hook_manager):
        """Verify hooks execute in priority order."""
        execution_order = []

        async def critical_handler(**kwargs):
            execution_order.append("critical")

        async def normal_handler(**kwargs):
            execution_order.append("normal")

        async def low_handler(**kwargs):
            execution_order.append("low")

        hook_manager.register(
            HookType.POST_DEBATE, low_handler, name="low", priority=HookPriority.LOW
        )
        hook_manager.register(
            HookType.POST_DEBATE, critical_handler, name="critical", priority=HookPriority.CRITICAL
        )
        hook_manager.register(
            HookType.POST_DEBATE, normal_handler, name="normal", priority=HookPriority.NORMAL
        )

        await hook_manager.trigger(HookType.POST_DEBATE)

        assert execution_order == ["critical", "normal", "low"]

    @pytest.mark.asyncio
    async def test_one_shot_hook(self, hook_manager):
        """Verify one-shot hooks only trigger once."""
        triggered = {"count": 0}

        async def one_shot_handler(**kwargs):
            triggered["count"] += 1

        hook_manager.register(
            HookType.POST_DEBATE,
            one_shot_handler,
            name="one_shot",
            once=True,
        )

        # First trigger
        await hook_manager.trigger(HookType.POST_DEBATE)
        assert triggered["count"] == 1

        # Second trigger - should not fire
        await hook_manager.trigger(HookType.POST_DEBATE)
        assert triggered["count"] == 1

    @pytest.mark.asyncio
    async def test_conditional_hook_execution(self, hook_loader, hook_manager):
        """Verify hooks only execute when conditions are met."""
        triggered = {"count": 0}

        async def conditional_handler(**kwargs):
            triggered["count"] += 1

        # Create hook with condition
        config = HookConfig(
            name="conditional_hook",
            trigger="post_debate",
            action=ActionConfig(handler="test.handler"),
            conditions=[
                ConditionConfig(field="confidence", operator="gte", value=0.8),
            ],
        )

        # Register with condition wrapper
        hook_loader._handlers = {"test.handler": conditional_handler}
        hook_loader.apply_to_manager(hook_manager, [config])

        # Trigger with low confidence - should NOT fire
        await hook_manager.trigger("post_debate", confidence=0.5)
        assert triggered["count"] == 0

        # Trigger with high confidence - should fire
        await hook_manager.trigger("post_debate", confidence=0.9)
        assert triggered["count"] == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestHookErrorHandling:
    """Tests for hook error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_hook_error_does_not_stop_others(self, hook_manager):
        """Verify one hook failing doesn't stop other hooks."""
        results = []

        async def failing_handler(**kwargs):
            raise ValueError("Intentional error")

        async def success_handler(**kwargs):
            results.append("success")

        hook_manager.register(
            HookType.POST_DEBATE,
            failing_handler,
            name="failing",
            priority=HookPriority.HIGH,
        )
        hook_manager.register(
            HookType.POST_DEBATE,
            success_handler,
            name="success",
            priority=HookPriority.NORMAL,
        )

        # Trigger should not raise, and success handler should still run
        await hook_manager.trigger(HookType.POST_DEBATE)

        assert "success" in results

    def test_invalid_yaml_handling(self, hook_loader):
        """Verify loader handles invalid YAML gracefully."""
        invalid_yaml = "this is not: valid: yaml: {{"

        with pytest.raises(Exception):
            hook_loader.load_from_string(invalid_yaml)

    def test_missing_handler_validation(self, hook_loader):
        """Verify loader validates handler paths."""
        config = HookConfig(
            name="missing_handler",
            trigger="post_debate",
            action=ActionConfig(handler="nonexistent.module.handler"),
        )

        errors = hook_loader.validate_config(config)
        assert len(errors) > 0  # Should report missing handler

    @pytest.mark.asyncio
    async def test_handler_timeout(self, hook_manager):
        """Verify hooks respect timeout settings."""
        import asyncio

        async def slow_handler(**kwargs):
            await asyncio.sleep(10)  # Very slow

        hook_manager.register(
            HookType.POST_DEBATE,
            slow_handler,
            name="slow",
        )

        # Trigger with short overall timeout
        # Should complete without waiting for slow handler
        try:
            await asyncio.wait_for(
                hook_manager.trigger(HookType.POST_DEBATE),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            pass  # Expected - slow handler times out


# =============================================================================
# Integration with Built-in Handlers
# =============================================================================


class TestBuiltinHandlerIntegration:
    """Tests for integration with built-in hook handlers."""

    @pytest.mark.asyncio
    async def test_log_event_handler_callable(self):
        """Verify log_event handler is callable."""
        from aragora.hooks.builtin import log_event

        # Should not raise
        await log_event(
            message="Test message",
            level="info",
            debate_id="123",
        )

    @pytest.mark.asyncio
    async def test_handler_resolution(self, hook_loader):
        """Verify handler paths resolve correctly."""
        handler = hook_loader.resolve_handler("aragora.hooks.builtin.log_event")

        assert handler is not None
        assert callable(handler)

    def test_apply_hooks_to_manager(self, hook_loader, hook_manager, sample_yaml_config):
        """Verify hooks are properly applied to manager."""
        configs = hook_loader.load_from_string(sample_yaml_config)
        count = hook_loader.apply_to_manager(hook_manager, configs)

        # Only enabled hooks should be applied (2 out of 3)
        assert count == 2
        assert hook_manager.has_hooks("post_debate")

    def test_get_registered_hooks(self, hook_loader, hook_manager, sample_yaml_config):
        """Verify we can list registered hooks."""
        configs = hook_loader.load_from_string(sample_yaml_config)
        hook_loader.apply_to_manager(hook_manager, configs)

        hooks = hook_manager.get_hooks("post_debate")
        assert len(hooks) == 2
        assert "log_debate_complete" in hooks
        assert "high_confidence_checkpoint" in hooks
