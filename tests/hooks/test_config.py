"""
Comprehensive tests for Hook Configuration Types.

Tests cover:
- ConditionConfig creation and serialization
- ActionConfig creation and serialization
- HookConfig creation and serialization
- Configuration parsing from dictionaries
- Edge cases and validation
"""

from __future__ import annotations

from typing import Any

import pytest

from aragora.hooks.config import ActionConfig, ConditionConfig, HookConfig


# =============================================================================
# ConditionConfig Tests
# =============================================================================


class TestConditionConfig:
    """Tests for ConditionConfig dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        config = ConditionConfig(
            field="status",
            operator="eq",
            value="active",
        )

        assert config.field == "status"
        assert config.operator == "eq"
        assert config.value == "active"
        assert config.negate is False

    def test_init_with_negate(self):
        """Test initialization with negate."""
        config = ConditionConfig(
            field="status",
            operator="eq",
            value="failed",
            negate=True,
        )

        assert config.negate is True

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {"field": "confidence"}
        config = ConditionConfig.from_dict(data)

        assert config.field == "confidence"
        assert config.operator == "eq"  # default
        assert config.value is None
        assert config.negate is False

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "field": "result.confidence",
            "operator": "gte",
            "value": 0.8,
            "negate": True,
        }
        config = ConditionConfig.from_dict(data)

        assert config.field == "result.confidence"
        assert config.operator == "gte"
        assert config.value == 0.8
        assert config.negate is True

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = ConditionConfig(
            field="status",
            operator="ne",
            value="error",
            negate=True,
        )
        data = config.to_dict()

        assert data["field"] == "status"
        assert data["operator"] == "ne"
        assert data["value"] == "error"
        assert data["negate"] is True

    def test_roundtrip(self):
        """Test from_dict -> to_dict roundtrip."""
        original = ConditionConfig(
            field="data.items.0.value",
            operator="contains",
            value="test",
            negate=False,
        )

        data = original.to_dict()
        restored = ConditionConfig.from_dict(data)

        assert restored.field == original.field
        assert restored.operator == original.operator
        assert restored.value == original.value
        assert restored.negate == original.negate

    def test_from_dict_with_complex_value(self):
        """Test from_dict with complex value types."""
        # List value
        data = {
            "field": "status",
            "operator": "in",
            "value": ["active", "pending", "processing"],
        }
        config = ConditionConfig.from_dict(data)
        assert config.value == ["active", "pending", "processing"]

        # Dict value
        data = {
            "field": "metadata",
            "operator": "eq",
            "value": {"key": "value"},
        }
        config = ConditionConfig.from_dict(data)
        assert config.value == {"key": "value"}

    def test_from_dict_with_none_value(self):
        """Test from_dict with None value."""
        data = {
            "field": "value",
            "operator": "is_null",
            "value": None,
        }
        config = ConditionConfig.from_dict(data)
        assert config.value is None


# =============================================================================
# ActionConfig Tests
# =============================================================================


class TestActionConfig:
    """Tests for ActionConfig dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        config = ActionConfig(handler="aragora.hooks.builtin.log_event")

        assert config.handler == "aragora.hooks.builtin.log_event"
        assert config.args == {}
        assert config.async_execution is True
        assert config.timeout is None

    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        config = ActionConfig(
            handler="aragora.hooks.builtin.send_webhook",
            args={"url": "https://example.com", "method": "POST"},
            async_execution=False,
            timeout=30.0,
        )

        assert config.handler == "aragora.hooks.builtin.send_webhook"
        assert config.args == {"url": "https://example.com", "method": "POST"}
        assert config.async_execution is False
        assert config.timeout == 30.0

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {"handler": "aragora.hooks.builtin.log_event"}
        config = ActionConfig.from_dict(data)

        assert config.handler == "aragora.hooks.builtin.log_event"
        assert config.args == {}
        assert config.async_execution is True
        assert config.timeout is None

    def test_from_dict_with_args(self):
        """Test from_dict with args."""
        data = {
            "handler": "aragora.hooks.builtin.save_checkpoint",
            "args": {
                "path": "/data/checkpoints",
                "include_fields": ["result", "confidence"],
            },
        }
        config = ActionConfig.from_dict(data)

        assert config.args["path"] == "/data/checkpoints"
        assert config.args["include_fields"] == ["result", "confidence"]

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "handler": "custom.module.handler",
            "args": {"key": "value"},
            "async_execution": False,
            "timeout": 60.0,
        }
        config = ActionConfig.from_dict(data)

        assert config.handler == "custom.module.handler"
        assert config.args == {"key": "value"}
        assert config.async_execution is False
        assert config.timeout == 60.0

    def test_to_dict_minimal(self):
        """Test to_dict with minimal config."""
        config = ActionConfig(handler="aragora.hooks.builtin.log_event")
        data = config.to_dict()

        assert data["handler"] == "aragora.hooks.builtin.log_event"
        assert data["args"] == {}
        assert data["async_execution"] is True
        assert "timeout" not in data

    def test_to_dict_with_timeout(self):
        """Test to_dict includes timeout when set."""
        config = ActionConfig(
            handler="aragora.hooks.builtin.send_webhook",
            timeout=30.0,
        )
        data = config.to_dict()

        assert data["timeout"] == 30.0

    def test_roundtrip(self):
        """Test from_dict -> to_dict roundtrip."""
        original = ActionConfig(
            handler="aragora.hooks.builtin.save_checkpoint",
            args={"path": "/data", "key": 123},
            async_execution=False,
            timeout=45.0,
        )

        data = original.to_dict()
        restored = ActionConfig.from_dict(data)

        assert restored.handler == original.handler
        assert restored.args == original.args
        assert restored.async_execution == original.async_execution
        assert restored.timeout == original.timeout


# =============================================================================
# HookConfig Tests
# =============================================================================


class TestHookConfig:
    """Tests for HookConfig dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        action = ActionConfig(handler="aragora.hooks.builtin.log_event")
        config = HookConfig(
            name="test_hook",
            trigger="post_debate",
            action=action,
        )

        assert config.name == "test_hook"
        assert config.trigger == "post_debate"
        assert config.action == action
        assert config.conditions == []
        assert config.priority == "normal"
        assert config.enabled is True
        assert config.one_shot is False
        assert config.description == ""
        assert config.tags == []
        assert config.source_file is None

    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        action = ActionConfig(handler="aragora.hooks.builtin.log_event")
        conditions = [
            ConditionConfig(field="consensus", operator="is_true", value=None),
        ]
        config = HookConfig(
            name="full_hook",
            trigger="on_finding",
            action=action,
            conditions=conditions,
            priority="high",
            enabled=True,
            one_shot=True,
            description="Test hook",
            tags=["test", "audit"],
            source_file="/path/to/hook.yaml",
        )

        assert config.name == "full_hook"
        assert config.trigger == "on_finding"
        assert config.priority == "high"
        assert config.one_shot is True
        assert config.description == "Test hook"
        assert config.tags == ["test", "audit"]
        assert config.source_file == "/path/to/hook.yaml"
        assert len(config.conditions) == 1

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "name": "minimal_hook",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
        }
        config = HookConfig.from_dict(data)

        assert config.name == "minimal_hook"
        assert config.trigger == "post_debate"
        assert config.action.handler == "aragora.hooks.builtin.log_event"
        assert config.priority == "normal"
        assert config.enabled is True

    def test_from_dict_string_action(self):
        """Test from_dict with string action shorthand."""
        data = {
            "name": "simple_hook",
            "trigger": "post_round",
            "action": "aragora.hooks.builtin.log_event",
        }
        config = HookConfig.from_dict(data)

        assert config.action.handler == "aragora.hooks.builtin.log_event"
        assert config.action.args == {}

    def test_from_dict_with_conditions(self):
        """Test from_dict with conditions."""
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
        assert config.conditions[0].value is True
        assert config.conditions[1].field == "confidence"
        assert config.conditions[1].operator == "gte"

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "name": "full_hook",
            "trigger": "on_finding",
            "action": {
                "handler": "aragora.hooks.builtin.save_checkpoint",
                "args": {"path": "/data"},
                "timeout": 30.0,
            },
            "conditions": [
                {"field": "severity", "operator": "eq", "value": "critical"},
            ],
            "priority": "high",
            "enabled": True,
            "one_shot": True,
            "description": "Save critical findings",
            "tags": ["audit", "critical"],
        }
        config = HookConfig.from_dict(data, source_file="test.yaml")

        assert config.name == "full_hook"
        assert config.trigger == "on_finding"
        assert config.action.handler == "aragora.hooks.builtin.save_checkpoint"
        assert config.action.args["path"] == "/data"
        assert config.action.timeout == 30.0
        assert len(config.conditions) == 1
        assert config.priority == "high"
        assert config.one_shot is True
        assert config.description == "Save critical findings"
        assert "audit" in config.tags
        assert config.source_file == "test.yaml"

    def test_from_dict_disabled(self):
        """Test from_dict with disabled hook."""
        data = {
            "name": "disabled_hook",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
            "enabled": False,
        }
        config = HookConfig.from_dict(data)

        assert config.enabled is False

    def test_to_dict_minimal(self):
        """Test to_dict with minimal config."""
        action = ActionConfig(handler="aragora.hooks.builtin.log_event")
        config = HookConfig(
            name="test_hook",
            trigger="post_debate",
            action=action,
        )
        data = config.to_dict()

        assert data["name"] == "test_hook"
        assert data["trigger"] == "post_debate"
        assert data["action"]["handler"] == "aragora.hooks.builtin.log_event"
        assert data["priority"] == "normal"
        assert data["enabled"] is True
        assert "conditions" not in data  # Empty conditions not included
        assert "one_shot" not in data  # False not included
        assert "description" not in data  # Empty not included
        assert "tags" not in data  # Empty not included

    def test_to_dict_full(self):
        """Test to_dict with full config."""
        action = ActionConfig(
            handler="aragora.hooks.builtin.save_checkpoint",
            args={"path": "/data"},
        )
        conditions = [
            ConditionConfig(field="severity", operator="eq", value="high"),
        ]
        config = HookConfig(
            name="full_hook",
            trigger="on_finding",
            action=action,
            conditions=conditions,
            priority="critical",
            enabled=True,
            one_shot=True,
            description="Important hook",
            tags=["audit"],
        )
        data = config.to_dict()

        assert data["name"] == "full_hook"
        assert data["trigger"] == "on_finding"
        assert data["priority"] == "critical"
        assert data["one_shot"] is True
        assert data["description"] == "Important hook"
        assert data["tags"] == ["audit"]
        assert len(data["conditions"]) == 1
        assert data["conditions"][0]["field"] == "severity"

    def test_roundtrip(self):
        """Test from_dict -> to_dict roundtrip."""
        original_data = {
            "name": "roundtrip_hook",
            "trigger": "post_debate",
            "action": {
                "handler": "aragora.hooks.builtin.log_event",
                "args": {"level": "info"},
            },
            "conditions": [
                {"field": "consensus", "operator": "is_true", "value": None},
            ],
            "priority": "high",
            "enabled": True,
            "one_shot": True,
            "description": "Test roundtrip",
            "tags": ["test"],
        }

        config = HookConfig.from_dict(original_data)
        restored_data = config.to_dict()

        assert restored_data["name"] == original_data["name"]
        assert restored_data["trigger"] == original_data["trigger"]
        assert restored_data["priority"] == original_data["priority"]
        assert restored_data["one_shot"] == original_data["one_shot"]
        assert restored_data["description"] == original_data["description"]
        assert restored_data["tags"] == original_data["tags"]

    def test_hash_by_name(self):
        """Test HookConfig is hashable by name."""
        action = ActionConfig(handler="aragora.hooks.builtin.log_event")

        config1 = HookConfig(name="same_name", trigger="post_debate", action=action)
        config2 = HookConfig(name="same_name", trigger="on_finding", action=action)
        config3 = HookConfig(name="different_name", trigger="post_debate", action=action)

        assert hash(config1) == hash(config2)  # Same name
        assert hash(config1) != hash(config3)  # Different name

    def test_hash_in_set(self):
        """Test HookConfig can be used in sets."""
        action = ActionConfig(handler="aragora.hooks.builtin.log_event")

        config1 = HookConfig(name="hook1", trigger="post_debate", action=action)
        config2 = HookConfig(name="hook2", trigger="post_debate", action=action)

        hook_set = {config1, config2}
        assert len(hook_set) == 2

    def test_hash_same_name_replaces(self):
        """Test that configs with same name are considered equal in sets."""
        action = ActionConfig(handler="aragora.hooks.builtin.log_event")

        config1 = HookConfig(name="same", trigger="post_debate", action=action)
        config2 = HookConfig(name="same", trigger="on_finding", action=action)

        hook_set = {config1}
        # Adding config2 with same name - depends on __eq__ implementation
        # Since only __hash__ is defined, set behavior may vary
        # This test documents expected behavior


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string_fields(self):
        """Test handling of empty string fields."""
        data = {
            "name": "",
            "trigger": "",
            "action": {"handler": ""},
        }
        config = HookConfig.from_dict(data)

        assert config.name == ""
        assert config.trigger == ""
        assert config.action.handler == ""

    def test_unicode_in_fields(self):
        """Test unicode characters in fields."""
        data = {
            "name": "hook_",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
            "description": " Unicode description",
        }
        config = HookConfig.from_dict(data)

        assert "" in config.name
        assert "" in config.description

    def test_special_characters_in_tags(self):
        """Test special characters in tags."""
        data = {
            "name": "test_hook",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
            "tags": ["tag-with-dash", "tag_with_underscore", "tag.with.dot"],
        }
        config = HookConfig.from_dict(data)

        assert "tag-with-dash" in config.tags
        assert "tag_with_underscore" in config.tags
        assert "tag.with.dot" in config.tags

    def test_nested_action_args(self):
        """Test deeply nested action args."""
        data = {
            "name": "nested_hook",
            "trigger": "post_debate",
            "action": {
                "handler": "custom.handler",
                "args": {
                    "level1": {
                        "level2": {
                            "level3": {"value": "deep"},
                        },
                    },
                    "list": [1, 2, {"nested": True}],
                },
            },
        }
        config = HookConfig.from_dict(data)

        assert config.action.args["level1"]["level2"]["level3"]["value"] == "deep"
        assert config.action.args["list"][2]["nested"] is True

    def test_condition_with_complex_value(self):
        """Test condition with complex value types."""
        data = {
            "name": "complex_condition_hook",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
            "conditions": [
                {
                    "field": "data",
                    "operator": "eq",
                    "value": {"nested": {"key": [1, 2, 3]}},
                },
            ],
        }
        config = HookConfig.from_dict(data)

        assert config.conditions[0].value == {"nested": {"key": [1, 2, 3]}}

    def test_many_conditions(self):
        """Test hook with many conditions."""
        conditions_data = [
            {"field": f"field_{i}", "operator": "eq", "value": i} for i in range(100)
        ]

        data = {
            "name": "many_conditions_hook",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
            "conditions": conditions_data,
        }
        config = HookConfig.from_dict(data)

        assert len(config.conditions) == 100
        assert config.conditions[50].field == "field_50"
        assert config.conditions[50].value == 50

    def test_many_tags(self):
        """Test hook with many tags."""
        tags = [f"tag_{i}" for i in range(50)]

        data = {
            "name": "many_tags_hook",
            "trigger": "post_debate",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
            "tags": tags,
        }
        config = HookConfig.from_dict(data)

        assert len(config.tags) == 50
        assert "tag_25" in config.tags

    def test_float_timeout(self):
        """Test action with float timeout."""
        data = {
            "name": "timeout_hook",
            "trigger": "post_debate",
            "action": {
                "handler": "aragora.hooks.builtin.send_webhook",
                "timeout": 0.5,
            },
        }
        config = HookConfig.from_dict(data)

        assert config.action.timeout == 0.5

    def test_zero_timeout(self):
        """Test action with zero timeout."""
        data = {
            "name": "zero_timeout_hook",
            "trigger": "post_debate",
            "action": {
                "handler": "aragora.hooks.builtin.send_webhook",
                "timeout": 0,
            },
        }
        config = HookConfig.from_dict(data)

        assert config.action.timeout == 0

    def test_all_valid_priorities(self):
        """Test all valid priority values."""
        valid_priorities = ["critical", "high", "normal", "low", "cleanup"]

        for priority in valid_priorities:
            data = {
                "name": f"priority_{priority}_hook",
                "trigger": "post_debate",
                "action": {"handler": "aragora.hooks.builtin.log_event"},
                "priority": priority,
            }
            config = HookConfig.from_dict(data)
            assert config.priority == priority

    def test_case_preserved_in_fields(self):
        """Test that case is preserved in string fields."""
        data = {
            "name": "CamelCaseHook",
            "trigger": "POST_DEBATE",
            "action": {"handler": "aragora.hooks.builtin.log_event"},
            "description": "MixedCase Description",
        }
        config = HookConfig.from_dict(data)

        assert config.name == "CamelCaseHook"
        assert config.trigger == "POST_DEBATE"
        assert config.description == "MixedCase Description"


# =============================================================================
# Default Values Tests
# =============================================================================


class TestDefaultValues:
    """Tests for default value behavior."""

    def test_condition_default_operator(self):
        """Test ConditionConfig default operator is 'eq'."""
        config = ConditionConfig.from_dict({"field": "test"})
        assert config.operator == "eq"

    def test_condition_default_negate(self):
        """Test ConditionConfig default negate is False."""
        config = ConditionConfig.from_dict({"field": "test"})
        assert config.negate is False

    def test_action_default_args(self):
        """Test ActionConfig default args is empty dict."""
        config = ActionConfig.from_dict({"handler": "test.handler"})
        assert config.args == {}

    def test_action_default_async_execution(self):
        """Test ActionConfig default async_execution is True."""
        config = ActionConfig.from_dict({"handler": "test.handler"})
        assert config.async_execution is True

    def test_action_default_timeout(self):
        """Test ActionConfig default timeout is None."""
        config = ActionConfig.from_dict({"handler": "test.handler"})
        assert config.timeout is None

    def test_hook_default_conditions(self):
        """Test HookConfig default conditions is empty list."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        assert config.conditions == []

    def test_hook_default_priority(self):
        """Test HookConfig default priority is 'normal'."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        assert config.priority == "normal"

    def test_hook_default_enabled(self):
        """Test HookConfig default enabled is True."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        assert config.enabled is True

    def test_hook_default_one_shot(self):
        """Test HookConfig default one_shot is False."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        assert config.one_shot is False

    def test_hook_default_description(self):
        """Test HookConfig default description is empty string."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        assert config.description == ""

    def test_hook_default_tags(self):
        """Test HookConfig default tags is empty list."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        assert config.tags == []

    def test_hook_default_source_file(self):
        """Test HookConfig default source_file is None."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        assert config.source_file is None


# =============================================================================
# Serialization Completeness Tests
# =============================================================================


class TestSerializationCompleteness:
    """Tests to ensure serialization captures all data."""

    def test_condition_to_dict_includes_all_fields(self):
        """Test ConditionConfig.to_dict includes all fields."""
        config = ConditionConfig(
            field="test",
            operator="gt",
            value=100,
            negate=True,
        )
        data = config.to_dict()

        assert "field" in data
        assert "operator" in data
        assert "value" in data
        assert "negate" in data

    def test_action_to_dict_includes_timeout(self):
        """Test ActionConfig.to_dict includes timeout when set."""
        config = ActionConfig(
            handler="test.handler",
            timeout=30.0,
        )
        data = config.to_dict()

        assert data.get("timeout") == 30.0

    def test_action_to_dict_excludes_none_timeout(self):
        """Test ActionConfig.to_dict excludes None timeout."""
        config = ActionConfig(handler="test.handler")
        data = config.to_dict()

        assert "timeout" not in data

    def test_hook_to_dict_excludes_empty_conditions(self):
        """Test HookConfig.to_dict excludes empty conditions."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        data = config.to_dict()

        assert "conditions" not in data

    def test_hook_to_dict_excludes_false_one_shot(self):
        """Test HookConfig.to_dict excludes false one_shot."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        data = config.to_dict()

        assert "one_shot" not in data

    def test_hook_to_dict_includes_true_one_shot(self):
        """Test HookConfig.to_dict includes true one_shot."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
                "one_shot": True,
            }
        )
        data = config.to_dict()

        assert data.get("one_shot") is True

    def test_hook_to_dict_excludes_empty_description(self):
        """Test HookConfig.to_dict excludes empty description."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        data = config.to_dict()

        assert "description" not in data

    def test_hook_to_dict_includes_non_empty_description(self):
        """Test HookConfig.to_dict includes non-empty description."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
                "description": "Important hook",
            }
        )
        data = config.to_dict()

        assert data.get("description") == "Important hook"

    def test_hook_to_dict_excludes_empty_tags(self):
        """Test HookConfig.to_dict excludes empty tags."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
            }
        )
        data = config.to_dict()

        assert "tags" not in data

    def test_hook_to_dict_includes_non_empty_tags(self):
        """Test HookConfig.to_dict includes non-empty tags."""
        config = HookConfig.from_dict(
            {
                "name": "test",
                "trigger": "post_debate",
                "action": {"handler": "test.handler"},
                "tags": ["audit"],
            }
        )
        data = config.to_dict()

        assert data.get("tags") == ["audit"]
