"""
Tests for Declarative Event Hooks System.

Tests YAML configuration loading, condition evaluation,
and hook registration/execution.
"""

from pathlib import Path

import pytest

from aragora.hooks.config import HookConfig, ActionConfig, ConditionConfig
from aragora.hooks.conditions import ConditionEvaluator, Operator
from aragora.hooks.loader import HookConfigLoader, get_hook_loader


# =============================================================================
# ConditionConfig Tests
# =============================================================================


class TestConditionConfig:
    """Tests for ConditionConfig dataclass."""

    def test_from_dict_basic(self):
        """Test creating condition from dict."""
        data = {
            "field": "confidence",
            "operator": "gte",
            "value": 0.8,
        }
        config = ConditionConfig.from_dict(data)

        assert config.field == "confidence"
        assert config.operator == "gte"
        assert config.value == 0.8
        assert config.negate is False

    def test_from_dict_with_negate(self):
        """Test condition with negation."""
        data = {
            "field": "status",
            "operator": "eq",
            "value": "failed",
            "negate": True,
        }
        config = ConditionConfig.from_dict(data)

        assert config.negate is True

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = ConditionConfig(
            field="result.confidence",
            operator="gt",
            value=0.5,
            negate=True,
        )
        data = original.to_dict()
        restored = ConditionConfig.from_dict(data)

        assert restored.field == original.field
        assert restored.operator == original.operator
        assert restored.value == original.value
        assert restored.negate == original.negate


# =============================================================================
# ActionConfig Tests
# =============================================================================


class TestActionConfig:
    """Tests for ActionConfig dataclass."""

    def test_from_dict_minimal(self):
        """Test creating action with minimal config."""
        data = {"handler": "aragora.hooks.builtin.log_event"}
        config = ActionConfig.from_dict(data)

        assert config.handler == "aragora.hooks.builtin.log_event"
        assert config.args == {}
        assert config.async_execution is True

    def test_from_dict_full(self):
        """Test creating action with all options."""
        data = {
            "handler": "aragora.hooks.builtin.send_webhook",
            "args": {"url": "https://example.com", "method": "POST"},
            "async_execution": False,
            "timeout": 30.0,
        }
        config = ActionConfig.from_dict(data)

        assert config.handler == "aragora.hooks.builtin.send_webhook"
        assert config.args["url"] == "https://example.com"
        assert config.async_execution is False
        assert config.timeout == 30.0


# =============================================================================
# HookConfig Tests
# =============================================================================


class TestHookConfig:
    """Tests for HookConfig dataclass."""

    def test_from_dict_minimal(self):
        """Test creating hook with minimal config."""
        data = {
            "name": "test_hook",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
        }
        config = HookConfig.from_dict(data)

        assert config.name == "test_hook"
        assert config.trigger == "post_debate"
        assert config.priority == "normal"
        assert config.enabled is True

    def test_from_dict_with_conditions(self):
        """Test hook with conditions."""
        data = {
            "name": "conditional_hook",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
            "conditions": [
                {"field": "consensus_reached", "operator": "eq", "value": True},
                {"field": "confidence", "operator": "gte", "value": 0.8},
            ],
        }
        config = HookConfig.from_dict(data)

        assert len(config.conditions) == 2
        assert config.conditions[0].field == "consensus_reached"
        assert config.conditions[1].value == 0.8

    def test_from_dict_with_all_options(self):
        """Test hook with all options."""
        data = {
            "name": "full_hook",
            "trigger": "on_finding",
            "action": {
                "handler": "aragora.hooks.builtin.save_checkpoint",
                "args": {"path": "/data"},
            },
            "conditions": [{"field": "severity", "operator": "eq", "value": "critical"}],
            "priority": "high",
            "enabled": True,
            "one_shot": True,
            "description": "Save critical findings",
            "tags": ["audit", "critical"],
        }
        config = HookConfig.from_dict(data, source_file="test.yaml")

        assert config.priority == "high"
        assert config.one_shot is True
        assert config.description == "Save critical findings"
        assert "audit" in config.tags
        assert config.source_file == "test.yaml"

    def test_string_handler_shorthand(self):
        """Test hook with string handler shorthand."""
        data = {
            "name": "simple_hook",
            "trigger": "post_round",
            "action": "aragora.hooks.builtin.log_event",
        }
        config = HookConfig.from_dict(data)

        assert config.action.handler == "aragora.hooks.builtin.log_event"


# =============================================================================
# ConditionEvaluator Tests
# =============================================================================


class TestConditionEvaluator:
    """Tests for condition evaluation."""

    def test_equality_operators(self):
        """Test eq and ne operators."""
        evaluator = ConditionEvaluator()
        context = {"status": "active", "count": 5}

        # Equal
        cond_eq = ConditionConfig(field="status", operator="eq", value="active")
        assert evaluator.evaluate(cond_eq, context) is True

        cond_eq_false = ConditionConfig(field="status", operator="eq", value="inactive")
        assert evaluator.evaluate(cond_eq_false, context) is False

        # Not equal
        cond_ne = ConditionConfig(field="status", operator="ne", value="inactive")
        assert evaluator.evaluate(cond_ne, context) is True

    def test_numeric_operators(self):
        """Test gt, gte, lt, lte operators."""
        evaluator = ConditionEvaluator()
        context = {"confidence": 0.75, "count": 10}

        assert (
            evaluator.evaluate(
                ConditionConfig(field="confidence", operator="gt", value=0.5), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="confidence", operator="gt", value=0.8), context
            )
            is False
        )

        assert (
            evaluator.evaluate(ConditionConfig(field="count", operator="gte", value=10), context)
            is True
        )
        assert (
            evaluator.evaluate(ConditionConfig(field="count", operator="lt", value=20), context)
            is True
        )
        assert (
            evaluator.evaluate(ConditionConfig(field="count", operator="lte", value=10), context)
            is True
        )

    def test_string_operators(self):
        """Test contains, starts_with, ends_with, matches."""
        evaluator = ConditionEvaluator()
        context = {"message": "Hello World", "email": "user@example.com"}

        assert (
            evaluator.evaluate(
                ConditionConfig(field="message", operator="contains", value="World"), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="message", operator="starts_with", value="Hello"), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="email", operator="ends_with", value=".com"), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="email", operator="matches", value=r"\w+@\w+\.\w+"), context
            )
            is True
        )

    def test_null_and_empty_operators(self):
        """Test is_null, is_not_null, is_empty, is_not_empty."""
        evaluator = ConditionEvaluator()
        context = {"present": "value", "missing": None, "empty_list": [], "filled": [1, 2]}

        assert (
            evaluator.evaluate(
                ConditionConfig(field="missing", operator="is_null", value=None), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="present", operator="is_not_null", value=None), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="empty_list", operator="is_empty", value=None), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="filled", operator="is_not_empty", value=None), context
            )
            is True
        )

    def test_collection_operators(self):
        """Test in, not_in, has_key."""
        evaluator = ConditionEvaluator()
        context = {"status": "active", "data": {"key1": "value1"}}

        assert (
            evaluator.evaluate(
                ConditionConfig(field="status", operator="in", value=["active", "pending"]), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="status", operator="not_in", value=["failed", "cancelled"]),
                context,
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="data", operator="has_key", value="key1"), context
            )
            is True
        )

    def test_boolean_operators(self):
        """Test is_true and is_false."""
        evaluator = ConditionEvaluator()
        context = {"consensus": True, "failed": False, "empty": ""}

        assert (
            evaluator.evaluate(
                ConditionConfig(field="consensus", operator="is_true", value=None), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="failed", operator="is_false", value=None), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="empty", operator="is_false", value=None), context
            )
            is True
        )

    def test_nested_field_access(self):
        """Test dot notation field access."""
        evaluator = ConditionEvaluator()
        context = {
            "result": {
                "confidence": 0.85,
                "metadata": {"source": "debate"},
            },
            "items": [{"id": 1}, {"id": 2}],
        }

        assert (
            evaluator.evaluate(
                ConditionConfig(field="result.confidence", operator="gt", value=0.8), context
            )
            is True
        )
        assert (
            evaluator.evaluate(
                ConditionConfig(field="result.metadata.source", operator="eq", value="debate"),
                context,
            )
            is True
        )
        assert (
            evaluator.evaluate(ConditionConfig(field="items.0.id", operator="eq", value=1), context)
            is True
        )

    def test_negation(self):
        """Test condition negation."""
        evaluator = ConditionEvaluator()
        context = {"status": "active"}

        # Without negation
        cond = ConditionConfig(field="status", operator="eq", value="active", negate=False)
        assert evaluator.evaluate(cond, context) is True

        # With negation
        cond_negated = ConditionConfig(field="status", operator="eq", value="active", negate=True)
        assert evaluator.evaluate(cond_negated, context) is False

    def test_evaluate_all_and_logic(self):
        """Test that evaluate_all uses AND logic."""
        evaluator = ConditionEvaluator()
        context = {"consensus": True, "confidence": 0.9}

        conditions = [
            ConditionConfig(field="consensus", operator="is_true", value=None),
            ConditionConfig(field="confidence", operator="gte", value=0.8),
        ]

        # Both conditions pass
        assert evaluator.evaluate_all(conditions, context) is True

        # One condition fails
        context["confidence"] = 0.5
        assert evaluator.evaluate_all(conditions, context) is False

    def test_evaluate_all_empty(self):
        """Test that empty conditions list returns True."""
        evaluator = ConditionEvaluator()
        assert evaluator.evaluate_all([], {}) is True


# =============================================================================
# HookConfigLoader Tests
# =============================================================================


class TestHookConfigLoader:
    """Tests for hook configuration loading."""

    def test_load_from_string(self):
        """Test loading hooks from YAML string."""
        yaml_content = """
hooks:
  - name: test_hook
    trigger: post_debate
    action:
      handler: aragora.hooks.builtin.log_event
      args:
        message: "Test message"
"""
        loader = HookConfigLoader()
        configs = loader.load_from_string(yaml_content)

        assert len(configs) == 1
        assert configs[0].name == "test_hook"
        assert configs[0].trigger == "post_debate"

    def test_load_multiple_hooks(self):
        """Test loading multiple hooks from one file."""
        yaml_content = """
hooks:
  - name: hook_one
    trigger: post_debate
    action: aragora.hooks.builtin.log_event

  - name: hook_two
    trigger: on_finding
    priority: high
    action:
      handler: aragora.hooks.builtin.save_checkpoint
"""
        loader = HookConfigLoader()
        configs = loader.load_from_string(yaml_content)

        assert len(configs) == 2
        assert loader.get_config("hook_one") is not None
        assert loader.get_config("hook_two") is not None

    def test_get_configs_by_trigger(self):
        """Test filtering configs by trigger."""
        yaml_content = """
hooks:
  - name: debate_hook
    trigger: post_debate
    action: aragora.hooks.builtin.log_event

  - name: finding_hook
    trigger: on_finding
    action: aragora.hooks.builtin.log_event

  - name: another_debate_hook
    trigger: post_debate
    action: aragora.hooks.builtin.log_event
"""
        loader = HookConfigLoader()
        loader.load_from_string(yaml_content)

        debate_hooks = loader.get_configs_by_trigger("post_debate")
        assert len(debate_hooks) == 2

        finding_hooks = loader.get_configs_by_trigger("on_finding")
        assert len(finding_hooks) == 1

    def test_get_configs_by_tag(self):
        """Test filtering configs by tag."""
        yaml_content = """
hooks:
  - name: audit_hook
    trigger: on_finding
    tags: [audit, critical]
    action: aragora.hooks.builtin.log_event

  - name: logging_hook
    trigger: post_debate
    tags: [logging]
    action: aragora.hooks.builtin.log_event
"""
        loader = HookConfigLoader()
        loader.load_from_string(yaml_content)

        audit_hooks = loader.get_configs_by_tag("audit")
        assert len(audit_hooks) == 1
        assert audit_hooks[0].name == "audit_hook"

    def test_resolve_handler(self):
        """Test handler resolution."""
        loader = HookConfigLoader()

        # Valid handler
        handler = loader.resolve_handler("aragora.hooks.builtin.log_event")
        assert handler is not None
        assert callable(handler)

        # Invalid handler
        invalid = loader.resolve_handler("nonexistent.module.handler")
        assert invalid is None

    def test_validate_config(self):
        """Test configuration validation."""
        loader = HookConfigLoader()

        # Valid config
        valid = HookConfig(
            name="test",
            trigger="post_debate",
            action=ActionConfig(handler="aragora.hooks.builtin.log_event"),
        )
        errors = loader.validate_config(valid)
        assert len(errors) == 0

        # Missing name
        invalid = HookConfig(
            name="",
            trigger="post_debate",
            action=ActionConfig(handler="aragora.hooks.builtin.log_event"),
        )
        errors = loader.validate_config(invalid)
        assert any("name" in e.lower() for e in errors)

        # Invalid priority
        invalid_priority = HookConfig(
            name="test",
            trigger="post_debate",
            priority="invalid_priority",
            action=ActionConfig(handler="aragora.hooks.builtin.log_event"),
        )
        errors = loader.validate_config(invalid_priority)
        assert any("priority" in e.lower() for e in errors)

    def test_clear(self):
        """Test clearing loaded configs."""
        yaml_content = """
hooks:
  - name: test_hook
    trigger: post_debate
    action: aragora.hooks.builtin.log_event
"""
        loader = HookConfigLoader()
        loader.load_from_string(yaml_content)
        assert len(loader.configs) > 0

        loader.clear()
        assert len(loader.configs) == 0

    def test_disabled_hooks_filtered(self):
        """Test that disabled hooks are filtered from queries."""
        yaml_content = """
hooks:
  - name: enabled_hook
    trigger: post_debate
    enabled: true
    action: aragora.hooks.builtin.log_event

  - name: disabled_hook
    trigger: post_debate
    enabled: false
    action: aragora.hooks.builtin.log_event
"""
        loader = HookConfigLoader()
        loader.load_from_string(yaml_content)

        # Both configs are loaded
        assert len(loader.configs) == 2

        # Only enabled hooks returned by query
        enabled_hooks = loader.get_configs_by_trigger("post_debate")
        assert len(enabled_hooks) == 1
        assert enabled_hooks[0].name == "enabled_hook"


# =============================================================================
# Integration Tests
# =============================================================================


class TestHookIntegration:
    """Integration tests for hooks with HookManager."""

    @pytest.mark.asyncio
    async def test_apply_to_manager(self):
        """Test applying hooks to a HookManager."""
        from aragora.debate.hooks import HookManager

        yaml_content = """
hooks:
  - name: log_on_debate
    trigger: post_debate
    action:
      handler: aragora.hooks.builtin.log_event
      args:
        message: "Debate complete"
        level: info
"""
        loader = HookConfigLoader()
        loader.load_from_string(yaml_content)

        manager = HookManager()
        count = loader.apply_to_manager(manager)

        assert count == 1
        assert manager.has_hooks("post_debate")

    @pytest.mark.asyncio
    async def test_conditional_hook_execution(self):
        """Test that conditions are evaluated before execution."""
        from aragora.debate.hooks import HookManager

        # Track calls
        calls = []

        def track_call(**kwargs):
            calls.append(kwargs)

        # Create hook with condition
        config = HookConfig(
            name="conditional_hook",
            trigger="post_debate",
            action=ActionConfig(handler="test.handler"),
            conditions=[
                ConditionConfig(field="consensus_reached", operator="eq", value=True),
            ],
        )

        loader = HookConfigLoader()
        loader._configs["conditional_hook"] = config
        loader._handlers["test.handler"] = track_call

        manager = HookManager()
        loader.apply_to_manager(manager)

        # Trigger with condition met
        await manager.trigger("post_debate", consensus_reached=True)
        assert len(calls) == 1

        # Trigger with condition not met
        calls.clear()
        await manager.trigger("post_debate", consensus_reached=False)
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_hook_with_args(self):
        """Test that configured args are passed to handler."""
        from aragora.debate.hooks import HookManager

        received_args = {}

        def capture_args(**kwargs):
            received_args.update(kwargs)

        config = HookConfig(
            name="args_hook",
            trigger="post_debate",
            action=ActionConfig(
                handler="test.handler",
                args={"custom_arg": "custom_value", "number": 42},
            ),
        )

        loader = HookConfigLoader()
        loader._configs["args_hook"] = config
        loader._handlers["test.handler"] = capture_args

        manager = HookManager()
        loader.apply_to_manager(manager)

        await manager.trigger("post_debate", debate_id="123")

        assert received_args["custom_arg"] == "custom_value"
        assert received_args["number"] == 42
        assert received_args["debate_id"] == "123"


# =============================================================================
# Builtin Handlers Tests
# =============================================================================


class TestBuiltinHandlers:
    """Tests for built-in hook handlers."""

    @pytest.mark.asyncio
    async def test_log_event(self):
        """Test log_event handler."""
        from aragora.hooks.builtin import log_event

        # Should not raise
        await log_event(
            message="Test {debate_id}",
            level="info",
            debate_id="123",
        )

    @pytest.mark.asyncio
    async def test_log_metric(self):
        """Test log_metric handler."""
        from aragora.hooks.builtin import log_metric

        await log_metric(
            metric_name="test_metric",
            value=42,
            tags={"env": "test"},
            debate_id="123",
        )

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, tmp_path):
        """Test save_checkpoint handler."""
        from aragora.hooks.builtin import save_checkpoint

        result = await save_checkpoint(
            path=str(tmp_path / "checkpoints"),
            filename_template="test_{debate_id}.json",
            debate_id="123",
            data={"key": "value"},
        )

        assert result is not None
        assert Path(result).exists()

    @pytest.mark.asyncio
    async def test_delay_execution(self):
        """Test delay_execution handler."""
        import time
        from aragora.hooks.builtin import delay_execution

        start = time.time()
        await delay_execution(seconds=0.1)
        elapsed = time.time() - start

        assert elapsed >= 0.1


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_hook_loader_singleton(self):
        """Test that get_hook_loader returns same instance."""
        loader1 = get_hook_loader()
        loader2 = get_hook_loader()

        assert loader1 is loader2
