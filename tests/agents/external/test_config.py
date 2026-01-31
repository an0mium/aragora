"""Tests for external agent configuration classes."""

import pytest

from aragora.agents.external.config import (
    ApprovalMode,
    AutoGPTConfig,
    CircuitBreakerConfig,
    CrewAIConfig,
    ExternalAgentConfig,
    OpenHandsConfig,
    ToolConfig,
    WorkspaceConfig,
    WorkspaceType,
    get_config_for_adapter,
)


class TestWorkspaceType:
    """Tests for WorkspaceType enum."""

    def test_all_types_exist(self) -> None:
        """Verify all expected workspace types exist."""
        assert WorkspaceType.LOCAL == "local"
        assert WorkspaceType.DOCKER == "docker"
        assert WorkspaceType.KUBERNETES == "kubernetes"


class TestApprovalMode:
    """Tests for ApprovalMode enum."""

    def test_all_modes_exist(self) -> None:
        """Verify all approval modes exist."""
        assert ApprovalMode.AUTO == "auto"
        assert ApprovalMode.MANUAL == "manual"
        assert ApprovalMode.DENY == "deny"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self) -> None:
        """Test default circuit breaker values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.cooldown_seconds == 60.0
        assert config.half_open_max_requests == 2

    def test_custom_values(self) -> None:
        """Test custom circuit breaker values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            cooldown_seconds=120.0,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.cooldown_seconds == 120.0

    def test_immutability(self) -> None:
        """Test config is immutable (frozen)."""
        config = CircuitBreakerConfig()
        with pytest.raises(AttributeError):
            config.failure_threshold = 10


class TestToolConfig:
    """Tests for ToolConfig."""

    def test_default_values(self) -> None:
        """Test default tool config values."""
        config = ToolConfig()
        assert config.enabled is True
        assert config.permission_key == ""
        assert config.risk_level == "low"
        assert config.approval_mode == ApprovalMode.AUTO
        assert config.timeout_seconds == 300.0

    def test_custom_values(self) -> None:
        """Test custom tool config values."""
        config = ToolConfig(
            enabled=False,
            permission_key="computer_use.shell",
            risk_level="high",
            approval_mode=ApprovalMode.MANUAL,
            blocked_patterns=[r"rm\s+-rf"],
        )
        assert config.enabled is False
        assert config.permission_key == "computer_use.shell"
        assert config.approval_mode == ApprovalMode.MANUAL
        assert len(config.blocked_patterns) == 1

    def test_allowed_paths_default_empty(self) -> None:
        """Test allowed_paths defaults to empty list."""
        config = ToolConfig()
        assert config.allowed_paths == []


class TestWorkspaceConfig:
    """Tests for WorkspaceConfig."""

    def test_default_values(self) -> None:
        """Test default workspace values."""
        config = WorkspaceConfig()
        assert config.type == WorkspaceType.DOCKER
        assert config.base_image == "python:3.11-slim"
        assert config.memory_limit_mb == 2048
        assert config.cpu_limit == 1.0
        assert config.network_enabled is True
        assert config.mount_secrets is False
        assert config.cleanup_on_exit is True

    def test_custom_values(self) -> None:
        """Test custom workspace values."""
        config = WorkspaceConfig(
            type=WorkspaceType.KUBERNETES,
            memory_limit_mb=4096,
            network_enabled=False,
        )
        assert config.type == WorkspaceType.KUBERNETES
        assert config.memory_limit_mb == 4096
        assert config.network_enabled is False


class TestExternalAgentConfig:
    """Tests for ExternalAgentConfig base class."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = ExternalAgentConfig()
        assert config.adapter_name == "base"
        assert config.base_url is None
        assert config.timeout_seconds == 3600.0
        assert config.max_concurrent_tasks == 5
        assert config.retry_attempts == 3
        assert config.max_tokens_per_task == 100000
        assert config.max_cost_per_task_usd == 10.0
        assert config.redact_secrets is True

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = ExternalAgentConfig(
            adapter_name="custom",
            base_url="http://example.com",
            max_concurrent_tasks=10,
        )
        assert config.adapter_name == "custom"
        assert config.base_url == "http://example.com"
        assert config.max_concurrent_tasks == 10

    def test_circuit_breaker_default(self) -> None:
        """Test circuit breaker config is included."""
        config = ExternalAgentConfig()
        assert config.circuit_breaker is not None
        assert config.circuit_breaker.failure_threshold == 5

    def test_allowed_blocked_tools(self) -> None:
        """Test allowed and blocked tools."""
        config = ExternalAgentConfig(
            allowed_tools={"shell", "file_read"},
            blocked_tools={"network"},
        )
        assert "shell" in config.allowed_tools
        assert "network" in config.blocked_tools

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        config = ExternalAgentConfig(
            adapter_name="test",
            base_url="http://localhost",
            max_tokens_per_task=50000,
        )
        data = config.to_dict()

        assert data["adapter_name"] == "test"
        assert data["base_url"] == "http://localhost"
        assert data["max_tokens_per_task"] == 50000
        assert "circuit_breaker" in data


class TestOpenHandsConfig:
    """Tests for OpenHandsConfig."""

    def test_default_values(self) -> None:
        """Test default OpenHands config values."""
        config = OpenHandsConfig()
        assert config.adapter_name == "openhands"
        assert "localhost:3000" in config.base_url
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.0

    def test_workspace_config(self) -> None:
        """Test workspace config is included."""
        config = OpenHandsConfig()
        assert config.workspace is not None
        assert config.workspace.type == WorkspaceType.DOCKER

    def test_tool_configs(self) -> None:
        """Test tool configs are included."""
        config = OpenHandsConfig()
        assert "TerminalTool" in config.tools
        assert "FileEditorTool" in config.tools
        assert "BrowserTool" in config.tools

    def test_terminal_tool_config(self) -> None:
        """Test TerminalTool has proper config."""
        config = OpenHandsConfig()
        terminal = config.tools["TerminalTool"]
        assert terminal.risk_level == "high"
        assert len(terminal.blocked_patterns) > 0
        assert any("rm" in p for p in terminal.blocked_patterns)

    def test_browser_tool_requires_approval(self) -> None:
        """Test BrowserTool requires manual approval."""
        config = OpenHandsConfig()
        browser = config.tools["BrowserTool"]
        assert browser.approval_mode == ApprovalMode.MANUAL


class TestAutoGPTConfig:
    """Tests for AutoGPTConfig."""

    def test_default_values(self) -> None:
        """Test default AutoGPT config values."""
        config = AutoGPTConfig()
        assert config.adapter_name == "autogpt"
        assert "localhost:8000" in config.base_url
        assert config.agent_id is None


class TestCrewAIConfig:
    """Tests for CrewAIConfig."""

    def test_default_values(self) -> None:
        """Test default CrewAI config values."""
        config = CrewAIConfig()
        assert config.adapter_name == "crewai"
        assert config.crew_config_path is None
        assert config.verbose is False


class TestGetConfigForAdapter:
    """Tests for get_config_for_adapter factory function."""

    def test_get_openhands_config(self) -> None:
        """Test getting OpenHands config."""
        config = get_config_for_adapter("openhands")
        assert isinstance(config, OpenHandsConfig)
        assert config.adapter_name == "openhands"

    def test_get_autogpt_config(self) -> None:
        """Test getting AutoGPT config."""
        config = get_config_for_adapter("autogpt")
        assert isinstance(config, AutoGPTConfig)
        assert config.adapter_name == "autogpt"

    def test_get_crewai_config(self) -> None:
        """Test getting CrewAI config."""
        config = get_config_for_adapter("crewai")
        assert isinstance(config, CrewAIConfig)
        assert config.adapter_name == "crewai"

    def test_case_insensitive(self) -> None:
        """Test adapter name is case insensitive."""
        config = get_config_for_adapter("OpenHands")
        assert isinstance(config, OpenHandsConfig)

    def test_unknown_adapter_raises(self) -> None:
        """Test unknown adapter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_config_for_adapter("unknown")

    def test_error_lists_valid_adapters(self) -> None:
        """Test error message lists valid adapters."""
        with pytest.raises(ValueError) as exc_info:
            get_config_for_adapter("invalid")
        assert "openhands" in str(exc_info.value).lower()
        assert "autogpt" in str(exc_info.value).lower()
        assert "crewai" in str(exc_info.value).lower()
