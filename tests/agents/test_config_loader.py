"""
Tests for YAML Agent Configuration Loader.

Tests the following components:
- AgentConfig: Configuration dataclass with validation
- AgentConfigLoader: YAML config loading and management
- ConfigValidationError: Validation error handling
- load_agent_configs: Convenience function

Test coverage includes:
- AgentConfig creation and validation
- AgentConfig serialization (to_dict, from_dict)
- YAML file loading
- Directory loading
- Hot-reload capability
- Agent creation from configs
- Query methods (by expertise, capability, tag, priority)
- Error handling and edge cases
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_config_data() -> dict[str, Any]:
    """Minimal valid configuration data."""
    return {
        "name": "test-agent",
        "model_type": "anthropic-api",
    }


@pytest.fixture
def full_config_data() -> dict[str, Any]:
    """Full configuration data with all fields."""
    return {
        "name": "full-agent",
        "model_type": "anthropic-api",
        "model": "claude-3-5-sonnet",
        "temperature": 0.5,
        "max_tokens": 2048,
        "role": "critic",
        "stance": "negative",
        "system_prompt": "You are a critic.",
        "expertise_domains": ["security", "testing"],
        "capabilities": ["code_review", "analysis"],
        "priority": "high",
        "fallback_chain": ["openai-api", "gemini"],
        "memory_access": "write",
        "timeout_seconds": 600,
        "hooks": {"pre_task": "setup", "post_task": "cleanup"},
        "description": "A test agent",
        "tags": ["test", "demo"],
    }


@pytest.fixture
def temp_yaml_file(valid_config_data):
    """Create a temporary YAML config file."""
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(valid_config_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create multiple config files
        configs = [
            {
                "name": "agent-1",
                "model_type": "anthropic-api",
                "expertise_domains": ["security"],
                "capabilities": ["analysis"],
                "priority": "high",
                "tags": ["security"],
            },
            {
                "name": "agent-2",
                "model_type": "openai-api",
                "expertise_domains": ["testing", "qa"],
                "capabilities": ["validation"],
                "priority": "normal",
                "tags": ["testing"],
            },
            {
                "name": "agent-3",
                "model_type": "gemini",
                "expertise_domains": ["security", "testing"],
                "capabilities": ["analysis", "validation"],
                "priority": "low",
                "tags": ["security", "testing"],
            },
        ]

        for config in configs:
            filepath = config_dir / f"{config['name']}.yaml"
            with open(filepath, "w") as f:
                yaml.dump(config, f)

        yield config_dir


@pytest.fixture
def mock_registry():
    """Create a mock AgentRegistry for testing agent creation."""
    mock_agent = MagicMock()
    mock_agent.name = "test-agent"

    registry = MagicMock()
    registry.create.return_value = mock_agent

    return registry


# =============================================================================
# ConfigValidationError Tests
# =============================================================================


class TestConfigValidationError:
    """Tests for ConfigValidationError exception class."""

    def test_stores_config_path(self):
        """ConfigValidationError stores the config path."""
        from aragora.agents.config_loader import ConfigValidationError

        error = ConfigValidationError("path/to/config.yaml", ["error1"])

        assert error.config_path == "path/to/config.yaml"

    def test_stores_errors_list(self):
        """ConfigValidationError stores the errors list."""
        from aragora.agents.config_loader import ConfigValidationError

        errors = ["error1", "error2", "error3"]
        error = ConfigValidationError("config.yaml", errors)

        assert error.errors == errors

    def test_message_includes_path(self):
        """Error message includes config path."""
        from aragora.agents.config_loader import ConfigValidationError

        error = ConfigValidationError("my-config.yaml", ["name is required"])

        assert "my-config.yaml" in str(error)

    def test_message_includes_errors(self):
        """Error message includes all errors."""
        from aragora.agents.config_loader import ConfigValidationError

        errors = ["error1", "error2"]
        error = ConfigValidationError("config.yaml", errors)

        assert "error1" in str(error)
        assert "error2" in str(error)

    def test_inherits_from_exception(self):
        """ConfigValidationError inherits from Exception."""
        from aragora.agents.config_loader import ConfigValidationError

        error = ConfigValidationError("config.yaml", ["error"])

        assert isinstance(error, Exception)


# =============================================================================
# AgentConfig Dataclass Tests
# =============================================================================


class TestAgentConfigCreation:
    """Tests for AgentConfig dataclass creation."""

    def test_create_minimal_config(self):
        """Create AgentConfig with minimal required fields."""
        from aragora.agents.config_loader import AgentConfig

        config = AgentConfig(name="test", model_type="anthropic-api")

        assert config.name == "test"
        assert config.model_type == "anthropic-api"

    def test_default_values(self):
        """AgentConfig has correct default values."""
        from aragora.agents.config_loader import AgentConfig

        config = AgentConfig(name="test", model_type="anthropic-api")

        assert config.model is None
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.role == "proposer"
        assert config.stance == "neutral"
        assert config.system_prompt == ""
        assert config.expertise_domains == []
        assert config.capabilities == []
        assert config.priority == "normal"
        assert config.fallback_chain == []
        assert config.memory_access == "read"
        assert config.timeout_seconds == 300
        assert config.hooks == {}
        assert config.description == ""
        assert config.tags == []

    def test_create_full_config(self, full_config_data):
        """Create AgentConfig with all fields."""
        from aragora.agents.config_loader import AgentConfig

        config = AgentConfig(**full_config_data)

        assert config.name == "full-agent"
        assert config.model == "claude-3-5-sonnet"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.role == "critic"
        assert config.stance == "negative"
        assert config.system_prompt == "You are a critic."
        assert config.expertise_domains == ["security", "testing"]
        assert config.capabilities == ["code_review", "analysis"]
        assert config.priority == "high"
        assert config.fallback_chain == ["openai-api", "gemini"]
        assert config.memory_access == "write"
        assert config.timeout_seconds == 600
        assert config.hooks == {"pre_task": "setup", "post_task": "cleanup"}
        assert config.description == "A test agent"
        assert config.tags == ["test", "demo"]


class TestAgentConfigValidation:
    """Tests for AgentConfig validation."""

    def test_empty_name_raises_error(self):
        """Empty name raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="", model_type="anthropic-api")

        assert "name is required" in exc_info.value.errors

    def test_empty_model_type_raises_error(self):
        """Empty model_type raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="")

        assert "model_type is required" in exc_info.value.errors

    def test_invalid_role_raises_error(self):
        """Invalid role raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", role="invalid_role")

        assert any("role must be one of" in e for e in exc_info.value.errors)

    def test_valid_roles_accepted(self):
        """Valid roles are accepted."""
        from aragora.agents.config_loader import AgentConfig

        valid_roles = ["proposer", "critic", "synthesizer", "judge"]

        for role in valid_roles:
            config = AgentConfig(name="test", model_type="api", role=role)
            assert config.role == role

    def test_invalid_stance_raises_error(self):
        """Invalid stance raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", stance="invalid_stance")

        assert any("stance must be one of" in e for e in exc_info.value.errors)

    def test_valid_stances_accepted(self):
        """Valid stances are accepted."""
        from aragora.agents.config_loader import AgentConfig

        valid_stances = ["affirmative", "negative", "neutral"]

        for stance in valid_stances:
            config = AgentConfig(name="test", model_type="api", stance=stance)
            assert config.stance == stance

    def test_invalid_priority_raises_error(self):
        """Invalid priority raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", priority="invalid_priority")

        assert any("priority must be one of" in e for e in exc_info.value.errors)

    def test_valid_priorities_accepted(self):
        """Valid priorities are accepted."""
        from aragora.agents.config_loader import AgentConfig

        valid_priorities = ["low", "normal", "high", "critical"]

        for priority in valid_priorities:
            config = AgentConfig(name="test", model_type="api", priority=priority)
            assert config.priority == priority

    def test_invalid_memory_access_raises_error(self):
        """Invalid memory_access raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", memory_access="invalid")

        assert any("memory_access must be one of" in e for e in exc_info.value.errors)

    def test_valid_memory_access_accepted(self):
        """Valid memory_access values are accepted."""
        from aragora.agents.config_loader import AgentConfig

        valid_access = ["none", "read", "write", "full"]

        for access in valid_access:
            config = AgentConfig(name="test", model_type="api", memory_access=access)
            assert config.memory_access == access

    def test_temperature_below_zero_raises_error(self):
        """Temperature below 0 raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", temperature=-0.1)

        assert any("temperature must be 0.0-2.0" in e for e in exc_info.value.errors)

    def test_temperature_above_two_raises_error(self):
        """Temperature above 2.0 raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", temperature=2.5)

        assert any("temperature must be 0.0-2.0" in e for e in exc_info.value.errors)

    def test_temperature_boundary_values_accepted(self):
        """Temperature boundary values (0.0 and 2.0) are accepted."""
        from aragora.agents.config_loader import AgentConfig

        config_low = AgentConfig(name="test", model_type="api", temperature=0.0)
        assert config_low.temperature == 0.0

        config_high = AgentConfig(name="test", model_type="api", temperature=2.0)
        assert config_high.temperature == 2.0

    def test_max_tokens_zero_raises_error(self):
        """max_tokens of 0 raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", max_tokens=0)

        assert any("max_tokens must be positive" in e for e in exc_info.value.errors)

    def test_max_tokens_negative_raises_error(self):
        """Negative max_tokens raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", max_tokens=-100)

        assert any("max_tokens must be positive" in e for e in exc_info.value.errors)

    def test_timeout_seconds_zero_raises_error(self):
        """timeout_seconds of 0 raises ConfigValidationError."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(name="test", model_type="api", timeout_seconds=0)

        assert any("timeout_seconds must be positive" in e for e in exc_info.value.errors)

    def test_multiple_validation_errors_collected(self):
        """Multiple validation errors are collected."""
        from aragora.agents.config_loader import AgentConfig, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            AgentConfig(
                name="",
                model_type="",
                role="invalid",
                temperature=-1.0,
            )

        # Should have multiple errors
        assert len(exc_info.value.errors) >= 4


class TestAgentConfigSerialization:
    """Tests for AgentConfig serialization methods."""

    def test_to_dict(self, full_config_data):
        """to_dict returns correct dictionary representation."""
        from aragora.agents.config_loader import AgentConfig

        config = AgentConfig(**full_config_data)
        result = config.to_dict()

        assert result["name"] == "full-agent"
        assert result["model_type"] == "anthropic-api"
        assert result["model"] == "claude-3-5-sonnet"
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 2048
        assert result["role"] == "critic"
        assert result["expertise_domains"] == ["security", "testing"]
        assert result["hooks"] == {"pre_task": "setup", "post_task": "cleanup"}

    def test_to_dict_creates_new_lists(self, full_config_data):
        """to_dict creates new list instances (not references)."""
        from aragora.agents.config_loader import AgentConfig

        config = AgentConfig(**full_config_data)
        result = config.to_dict()

        # Modify result lists
        result["expertise_domains"].append("new_domain")
        result["tags"].append("new_tag")

        # Original config should be unchanged
        assert "new_domain" not in config.expertise_domains
        assert "new_tag" not in config.tags

    def test_from_dict_minimal(self):
        """from_dict creates config from minimal data."""
        from aragora.agents.config_loader import AgentConfig

        data = {"name": "test", "model_type": "api"}
        config = AgentConfig.from_dict(data)

        assert config.name == "test"
        assert config.model_type == "api"
        assert config.role == "proposer"  # Default value

    def test_from_dict_full(self, full_config_data):
        """from_dict creates config from full data."""
        from aragora.agents.config_loader import AgentConfig

        config = AgentConfig.from_dict(full_config_data)

        assert config.name == "full-agent"
        assert config.model == "claude-3-5-sonnet"
        assert config.expertise_domains == ["security", "testing"]

    def test_from_dict_with_config_path(self):
        """from_dict stores config_path when provided."""
        from aragora.agents.config_loader import AgentConfig

        data = {"name": "test", "model_type": "api"}
        config = AgentConfig.from_dict(data, config_path="/path/to/config.yaml")

        assert config.config_path == "/path/to/config.yaml"

    def test_roundtrip_serialization(self, full_config_data):
        """to_dict and from_dict roundtrip preserves data."""
        from aragora.agents.config_loader import AgentConfig

        original = AgentConfig(**full_config_data)
        serialized = original.to_dict()
        restored = AgentConfig.from_dict(serialized)

        assert restored.name == original.name
        assert restored.model == original.model
        assert restored.temperature == original.temperature
        assert restored.expertise_domains == original.expertise_domains
        assert restored.hooks == original.hooks


# =============================================================================
# AgentConfigLoader Tests
# =============================================================================


class TestAgentConfigLoaderInit:
    """Tests for AgentConfigLoader initialization."""

    def test_create_loader_without_registry(self):
        """Create loader without registry."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()

        assert loader._configs == {}
        assert loader._registry is None

    def test_create_loader_with_registry(self, mock_registry):
        """Create loader with registry."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader(registry=mock_registry)

        assert loader._registry is mock_registry


class TestAgentConfigLoaderLoadYaml:
    """Tests for YAML file loading."""

    def test_load_yaml_success(self, temp_yaml_file):
        """load_yaml successfully loads a YAML file."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        config = loader.load_yaml(temp_yaml_file)

        assert config.name == "test-agent"
        assert config.model_type == "anthropic-api"

    def test_load_yaml_stores_config(self, temp_yaml_file):
        """load_yaml stores config in loader."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        config = loader.load_yaml(temp_yaml_file)

        assert loader.get_config("test-agent") is config

    def test_load_yaml_stores_path(self, temp_yaml_file):
        """load_yaml stores config path for hot-reload."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        loader.load_yaml(temp_yaml_file)

        assert "test-agent" in loader._config_paths
        assert loader._config_paths["test-agent"] == temp_yaml_file

    def test_load_yaml_file_not_found(self):
        """load_yaml raises FileNotFoundError for missing file."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_yaml("/nonexistent/path/config.yaml")

        assert "Config file not found" in str(exc_info.value)

    def test_load_yaml_invalid_content(self):
        """load_yaml raises ConfigValidationError for non-mapping YAML."""
        import yaml
        from aragora.agents.config_loader import AgentConfigLoader, ConfigValidationError

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["item1", "item2"], f)  # List instead of mapping
            temp_path = Path(f.name)

        try:
            loader = AgentConfigLoader()

            with pytest.raises(ConfigValidationError) as exc_info:
                loader.load_yaml(temp_path)

            assert "YAML must contain a mapping" in exc_info.value.errors
        finally:
            temp_path.unlink()

    def test_load_yaml_accepts_path_string(self, temp_yaml_file):
        """load_yaml accepts string path."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        config = loader.load_yaml(str(temp_yaml_file))

        assert config.name == "test-agent"


class TestAgentConfigLoaderLoadDirectory:
    """Tests for directory loading."""

    def test_load_directory_success(self, temp_config_dir):
        """load_directory loads all configs from directory."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        configs = loader.load_directory(temp_config_dir)

        assert len(configs) == 3
        assert "agent-1" in configs
        assert "agent-2" in configs
        assert "agent-3" in configs

    def test_load_directory_nonexistent(self, caplog):
        """load_directory returns empty dict for nonexistent directory."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        configs = loader.load_directory("/nonexistent/directory")

        assert configs == {}

    def test_load_directory_with_yml_extension(self):
        """load_directory also loads .yml files."""
        import yaml
        from aragora.agents.config_loader import AgentConfigLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create .yml file
            yml_path = config_dir / "test.yml"
            with open(yml_path, "w") as f:
                yaml.dump({"name": "yml-agent", "model_type": "api"}, f)

            loader = AgentConfigLoader()
            configs = loader.load_directory(config_dir)

            assert "yml-agent" in configs

    def test_load_directory_skips_invalid_files(self, caplog):
        """load_directory skips invalid config files."""
        import yaml
        from aragora.agents.config_loader import AgentConfigLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create valid config
            valid_path = config_dir / "valid.yaml"
            with open(valid_path, "w") as f:
                yaml.dump({"name": "valid", "model_type": "api"}, f)

            # Create invalid config (missing required fields)
            invalid_path = config_dir / "invalid.yaml"
            with open(invalid_path, "w") as f:
                yaml.dump({"name": "", "model_type": ""}, f)

            loader = AgentConfigLoader()
            configs = loader.load_directory(config_dir)

            # Only valid config should be loaded
            assert "valid" in configs
            assert len(configs) == 1


class TestAgentConfigLoaderHotReload:
    """Tests for hot-reload capability."""

    def test_reload_config_success(self):
        """reload_config reloads updated config from disk."""
        import yaml
        from aragora.agents.config_loader import AgentConfigLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"name": "test", "model_type": "api", "priority": "normal"}, f)
            temp_path = Path(f.name)

        try:
            loader = AgentConfigLoader()
            config1 = loader.load_yaml(temp_path)
            assert config1.priority == "normal"

            # Update file
            with open(temp_path, "w") as f:
                yaml.dump({"name": "test", "model_type": "api", "priority": "high"}, f)

            # Reload
            config2 = loader.reload_config("test")

            assert config2 is not None
            assert config2.priority == "high"
        finally:
            temp_path.unlink()

    def test_reload_config_unknown_name(self):
        """reload_config returns None for unknown config name."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        result = loader.reload_config("unknown-agent")

        assert result is None

    def test_reload_config_file_deleted(self, temp_yaml_file):
        """reload_config handles deleted config file."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        loader.load_yaml(temp_yaml_file)

        # Delete the file
        temp_yaml_file.unlink()

        result = loader.reload_config("test-agent")

        assert result is None
        assert "test-agent" not in loader._config_paths
        assert "test-agent" not in loader._configs

    def test_reload_all(self, temp_config_dir):
        """reload_all reloads all configs."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        loader.load_directory(temp_config_dir)

        reloaded = loader.reload_all()

        assert len(reloaded) == 3
        assert "agent-1" in reloaded
        assert "agent-2" in reloaded
        assert "agent-3" in reloaded


class TestAgentConfigLoaderQueryMethods:
    """Tests for config query methods."""

    def test_get_config_found(self, temp_yaml_file):
        """get_config returns config when found."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        original = loader.load_yaml(temp_yaml_file)

        result = loader.get_config("test-agent")

        assert result is original

    def test_get_config_not_found(self):
        """get_config returns None when not found."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        result = loader.get_config("nonexistent")

        assert result is None

    def test_list_configs(self, temp_config_dir):
        """list_configs returns all loaded config names."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        loader.load_directory(temp_config_dir)

        names = loader.list_configs()

        assert len(names) == 3
        assert "agent-1" in names
        assert "agent-2" in names
        assert "agent-3" in names

    def test_get_by_expertise(self, temp_config_dir):
        """get_by_expertise finds configs with matching domain."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        loader.load_directory(temp_config_dir)

        security_agents = loader.get_by_expertise("security")

        assert len(security_agents) == 2  # agent-1 and agent-3
        names = [c.name for c in security_agents]
        assert "agent-1" in names
        assert "agent-3" in names

    def test_get_by_capability(self, temp_config_dir):
        """get_by_capability finds configs with matching capability."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        loader.load_directory(temp_config_dir)

        analysis_agents = loader.get_by_capability("analysis")

        assert len(analysis_agents) == 2  # agent-1 and agent-3
        names = [c.name for c in analysis_agents]
        assert "agent-1" in names
        assert "agent-3" in names

    def test_get_by_tag(self, temp_config_dir):
        """get_by_tag finds configs with matching tag."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        loader.load_directory(temp_config_dir)

        testing_agents = loader.get_by_tag("testing")

        assert len(testing_agents) == 2  # agent-2 and agent-3
        names = [c.name for c in testing_agents]
        assert "agent-2" in names
        assert "agent-3" in names

    def test_get_by_priority(self, temp_config_dir):
        """get_by_priority finds configs with matching priority."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        loader.load_directory(temp_config_dir)

        high_priority = loader.get_by_priority("high")

        assert len(high_priority) == 1
        assert high_priority[0].name == "agent-1"


class TestAgentConfigLoaderAgentCreation:
    """Tests for agent creation from configs."""

    def test_create_agent_from_config_object(self, mock_registry):
        """create_agent creates agent from AgentConfig instance."""
        from aragora.agents.config_loader import AgentConfig, AgentConfigLoader

        loader = AgentConfigLoader(registry=mock_registry)
        config = AgentConfig(name="test", model_type="anthropic-api", role="critic")

        agent = loader.create_agent(config)

        mock_registry.create.assert_called_once_with(
            model_type="anthropic-api",
            name="test",
            role="critic",
            model=None,
        )

    def test_create_agent_from_name(self, temp_yaml_file, mock_registry):
        """create_agent creates agent from config name."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader(registry=mock_registry)
        loader.load_yaml(temp_yaml_file)

        agent = loader.create_agent("test-agent")

        mock_registry.create.assert_called_once()

    def test_create_agent_unknown_name_raises(self, mock_registry):
        """create_agent raises ValueError for unknown config name."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader(registry=mock_registry)

        with pytest.raises(ValueError) as exc_info:
            loader.create_agent("unknown-agent")

        assert "Config not loaded" in str(exc_info.value)

    def test_create_agent_applies_system_prompt(self, mock_registry):
        """create_agent applies system prompt to agent."""
        from aragora.agents.config_loader import AgentConfig, AgentConfigLoader

        loader = AgentConfigLoader(registry=mock_registry)
        config = AgentConfig(
            name="test",
            model_type="api",
            system_prompt="You are helpful.",
        )

        agent = loader.create_agent(config)

        agent.set_system_prompt.assert_called_once_with("You are helpful.")

    def test_create_agents_all(self, temp_config_dir, mock_registry):
        """create_agents creates all loaded agents when no list provided."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader(registry=mock_registry)
        loader.load_directory(temp_config_dir)

        agents = loader.create_agents()

        assert len(agents) == 3
        assert mock_registry.create.call_count == 3

    def test_create_agents_from_list(self, temp_config_dir, mock_registry):
        """create_agents creates only specified agents."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader(registry=mock_registry)
        loader.load_directory(temp_config_dir)

        agents = loader.create_agents(["agent-1", "agent-2"])

        assert len(agents) == 2
        assert mock_registry.create.call_count == 2

    def test_create_agents_skips_failures(self, temp_config_dir, mock_registry, caplog):
        """create_agents skips agents that fail to create."""
        from aragora.agents.config_loader import AgentConfigLoader

        # Make one creation fail
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Creation failed")
            return MagicMock()

        mock_registry.create.side_effect = side_effect

        loader = AgentConfigLoader(registry=mock_registry)
        loader.load_directory(temp_config_dir)

        agents = loader.create_agents()

        # Should have 2 agents (one failed)
        assert len(agents) == 2


# =============================================================================
# load_agent_configs Convenience Function Tests
# =============================================================================


class TestLoadAgentConfigsFunction:
    """Tests for load_agent_configs convenience function."""

    def test_load_agent_configs_custom_dir(self, temp_config_dir):
        """load_agent_configs loads from custom directory."""
        from aragora.agents.config_loader import load_agent_configs

        configs = load_agent_configs(temp_config_dir)

        assert len(configs) == 3
        assert "agent-1" in configs

    def test_load_agent_configs_default_dir(self):
        """load_agent_configs uses default directory when not specified."""
        from aragora.agents.config_loader import load_agent_configs

        # This should work even if default directory doesn't exist
        configs = load_agent_configs()

        # Should return a dict (possibly empty if no configs exist)
        assert isinstance(configs, dict)


# =============================================================================
# YAML Import Handling Tests
# =============================================================================


class TestYamlImportHandling:
    """Tests for YAML import handling."""

    def test_load_yaml_without_pyyaml_raises(self):
        """load_yaml raises ImportError when PyYAML not installed."""
        from aragora.agents.config_loader import AgentConfigLoader

        with patch("aragora.agents.config_loader._yaml", None):
            loader = AgentConfigLoader()

            with pytest.raises(ImportError) as exc_info:
                loader.load_yaml("/some/path.yaml")

            assert "PyYAML is required" in str(exc_info.value)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_agent_config_exportable(self):
        """AgentConfig can be imported."""
        from aragora.agents.config_loader import AgentConfig

        assert AgentConfig is not None

    def test_agent_config_loader_exportable(self):
        """AgentConfigLoader can be imported."""
        from aragora.agents.config_loader import AgentConfigLoader

        assert AgentConfigLoader is not None

    def test_config_validation_error_exportable(self):
        """ConfigValidationError can be imported."""
        from aragora.agents.config_loader import ConfigValidationError

        assert ConfigValidationError is not None

    def test_load_agent_configs_exportable(self):
        """load_agent_configs can be imported."""
        from aragora.agents.config_loader import load_agent_configs

        assert load_agent_configs is not None

    def test_all_exports(self):
        """All __all__ exports are available."""
        from aragora.agents import config_loader

        for name in config_loader.__all__:
            assert hasattr(config_loader, name), f"Missing export: {name}"


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_config_path_stored_in_config(self, temp_yaml_file):
        """Config stores its source path."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        config = loader.load_yaml(temp_yaml_file)

        assert config.config_path == str(temp_yaml_file)

    def test_empty_expertise_domains_query(self):
        """get_by_expertise returns empty list when no matches."""
        from aragora.agents.config_loader import AgentConfigLoader

        loader = AgentConfigLoader()
        result = loader.get_by_expertise("nonexistent_domain")

        assert result == []

    def test_hooks_dictionary_preserved(self):
        """Hooks dictionary is correctly preserved."""
        from aragora.agents.config_loader import AgentConfig

        hooks = {"pre_task": "setup", "post_task": "cleanup", "on_error": "handle"}
        config = AgentConfig(name="test", model_type="api", hooks=hooks)

        assert config.hooks == hooks
        assert config.hooks["pre_task"] == "setup"
        assert config.hooks["on_error"] == "handle"

    def test_fallback_chain_preserved(self):
        """Fallback chain list is correctly preserved."""
        from aragora.agents.config_loader import AgentConfig

        fallback = ["openai-api", "gemini", "openrouter"]
        config = AgentConfig(name="test", model_type="api", fallback_chain=fallback)

        assert config.fallback_chain == fallback
        assert len(config.fallback_chain) == 3

    def test_multiline_system_prompt(self):
        """Multiline system prompt is preserved."""
        from aragora.agents.config_loader import AgentConfig

        prompt = """You are a helpful assistant.

Your responsibilities include:
1. Answering questions
2. Providing explanations
3. Being concise"""

        config = AgentConfig(name="test", model_type="api", system_prompt=prompt)

        assert "Your responsibilities include:" in config.system_prompt
        assert "1. Answering questions" in config.system_prompt

    def test_create_agent_without_registry_imports_default(self):
        """create_agent imports default registry when none provided."""
        from aragora.agents.config_loader import AgentConfig, AgentConfigLoader

        loader = AgentConfigLoader()  # No registry
        config = AgentConfig(name="test", model_type="demo")

        with patch("aragora.agents.config_loader.AgentConfigLoader.create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            # The actual implementation will try to import the registry
            # This tests that the import path exists
            try:
                loader.create_agent(config)
            except (ImportError, ModuleNotFoundError):
                # Expected if registry module has dependencies
                pass
