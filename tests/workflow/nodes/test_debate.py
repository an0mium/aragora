"""
Tests for Workflow Debate Nodes (DebateStep and QuickDebateStep).

Tests cover:
- DebateStep initialization and config defaults
- DebateStep execution with mocked Arena
- DebateStep empty topic handling
- DebateStep agent creation and failure handling
- DebateStep ArenaConfig interpolation and injection
- DebateStep memory system injection from context state
- DebateStep topic interpolation from inputs, step_outputs, and state
- DebateStep import error handling
- DebateStep general exception handling
- QuickDebateStep initialization and config defaults
- QuickDebateStep execution with parallel agent responses
- QuickDebateStep empty question handling
- QuickDebateStep synthesis generation
- QuickDebateStep max_response_length truncation
- QuickDebateStep partial and full agent failures
- QuickDebateStep question interpolation
- Both: _interpolate_text with various placeholder types
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def patch_arena(*args, **kwargs):
    """Patch `aragora.Arena` even when it's lazily exported."""
    kwargs.setdefault("create", True)
    return patch("aragora.Arena", *args, **kwargs)


def patch_environment(*args, **kwargs):
    """Patch `aragora.Environment` even when it's lazily exported."""
    kwargs.setdefault("create", True)
    return patch("aragora.Environment", *args, **kwargs)


def patch_debate_protocol(*args, **kwargs):
    """Patch `aragora.DebateProtocol` even when it's lazily exported."""
    kwargs.setdefault("create", True)
    return patch("aragora.DebateProtocol", *args, **kwargs)


# ============================================================================
# Helpers
# ============================================================================


def _make_context(
    inputs=None,
    state=None,
    step_outputs=None,
    current_step_config=None,
    metadata=None,
):
    from aragora.workflow.step import WorkflowContext

    return WorkflowContext(
        workflow_id="wf_test",
        definition_id="def_test",
        inputs=inputs or {},
        state=state or {},
        step_outputs=step_outputs or {},
        metadata=metadata or {},
        current_step_config=current_step_config or {},
    )


# ============================================================================
# DebateStep Initialization Tests
# ============================================================================


class TestDebateStepInit:
    """Tests for DebateStep initialization."""

    def test_basic_init(self):
        """Test basic DebateStep initialization with name only."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Test Debate")
        assert step.name == "Test Debate"
        assert step.config == {}

    def test_init_with_config(self):
        """Test DebateStep initialization with full config."""
        from aragora.workflow.nodes.debate import DebateStep

        config = {
            "topic": "Evaluate risk",
            "agents": ["claude", "gpt4"],
            "rounds": 3,
            "topology": "adversarial",
            "consensus_mechanism": "unanimous",
            "enable_critique": False,
            "enable_synthesis": False,
            "timeout_seconds": 60,
            "memory_enabled": True,
            "tenant_id": "tenant-1",
        }
        step = DebateStep(name="Full Config Debate", config=config)
        assert step.name == "Full Config Debate"
        assert step.config["topic"] == "Evaluate risk"
        assert step.config["agents"] == ["claude", "gpt4"]
        assert step.config["rounds"] == 3
        assert step.config["topology"] == "adversarial"

    def test_init_none_config_becomes_empty(self):
        """Test that None config becomes empty dict."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="No Config", config=None)
        assert step.config == {}

    def test_init_preserves_arena_config(self):
        """Test that arena_config in config is preserved."""
        from aragora.workflow.nodes.debate import DebateStep

        config = {
            "topic": "Test",
            "arena_config": {
                "enable_knowledge_retrieval": True,
                "org_id": "org-123",
            },
        }
        step = DebateStep(name="Arena Config", config=config)
        assert step.config["arena_config"]["enable_knowledge_retrieval"] is True
        assert step.config["arena_config"]["org_id"] == "org-123"

    @pytest.mark.xfail(
        reason="isinstance check breaks when module reload changes class identity",
        strict=False,
    )
    def test_is_base_step_subclass(self):
        """Test that DebateStep is a BaseStep subclass."""
        from aragora.workflow.nodes.debate import DebateStep
        from aragora.workflow.step import BaseStep

        step = DebateStep(name="Test")
        assert isinstance(step, BaseStep)


# ============================================================================
# DebateStep Execution Tests
# ============================================================================


class TestDebateStepExecution:
    """Tests for DebateStep.execute()."""

    @pytest.mark.asyncio
    async def test_empty_topic_returns_error(self):
        """Test that an empty topic returns an error result."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Empty Topic", config={"topic": ""})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["success"] is False
        assert result["error"] == "Empty topic"

    @pytest.mark.asyncio
    async def test_no_topic_config_returns_error(self):
        """Test that missing topic config returns an error result."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="No Topic", config={})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["success"] is False
        assert result["error"] == "Empty topic"

    @pytest.mark.asyncio
    async def test_successful_debate_execution(self):
        """Test successful debate execution with mocked Arena."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "debate-123"
        mock_result.rounds_completed = 3
        mock_result.consensus_reached = True
        mock_result.consensus = "Agents agree on X"
        mock_result.synthesis = "Synthesis of all views"
        mock_result.duration_ms = 1500
        # No responses attribute
        del mock_result.responses

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.model = "claude-3"

        with patch(
            "aragora.workflow.nodes.debate.DebateStep.execute.__module__",
            create=True,
        ):
            with patch("aragora.agents.create_agent", return_value=mock_agent) as mock_create:
                with patch_arena( return_value=mock_arena):
                    with patch_environment() as mock_env:
                        with patch_debate_protocol() as mock_protocol:
                            step = DebateStep(
                                name="Test Debate",
                                config={
                                    "topic": "Should we use microservices?",
                                    "agents": ["claude", "gpt4"],
                                    "rounds": 3,
                                },
                            )
                            ctx = _make_context()
                            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["debate_id"] == "debate-123"
        assert result["topic"] == "Should we use microservices?"
        assert result["rounds_completed"] == 3
        assert result["consensus_reached"] is True
        assert result["consensus"] == "Agents agree on X"
        assert result["synthesis"] == "Synthesis of all views"
        assert result["execution_time_ms"] == 1500
        assert len(result["agents"]) == 2

    @pytest.mark.asyncio
    async def test_debate_with_responses_attribute(self):
        """Test that agent responses are included when result has responses."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "debate-456"
        mock_result.rounds_completed = 2
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 800
        mock_result.responses = [
            {"agent": "claude", "round": 1, "content": "Response A"},
            {"agent": "gpt4", "round": 1, "content": "Response B"},
        ]

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Debate With Responses",
                            config={
                                "topic": "Test topic",
                                "agents": ["claude"],
                            },
                        )
                        ctx = _make_context()
                        result = await step.execute(ctx)

        assert result["success"] is True
        assert "responses" in result
        assert len(result["responses"]) == 2
        assert result["responses"][0]["agent"] == "claude"
        assert result["responses"][0]["round"] == 1

    @pytest.mark.asyncio
    async def test_response_content_truncation(self):
        """Test that response content is truncated to 500 chars."""
        from aragora.workflow.nodes.debate import DebateStep

        long_content = "x" * 1000
        mock_result = MagicMock()
        mock_result.debate_id = "debate-789"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 100
        mock_result.responses = [
            {"agent": "claude", "round": 1, "content": long_content},
        ]

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Truncation Test",
                            config={"topic": "Test", "agents": ["claude"]},
                        )
                        ctx = _make_context()
                        result = await step.execute(ctx)

        assert len(result["responses"][0]["content"]) == 500

    @pytest.mark.asyncio
    async def test_responses_limited_to_20(self):
        """Test that responses are limited to 20 entries."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "debate-max"
        mock_result.rounds_completed = 10
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 5000
        mock_result.responses = [
            {"agent": f"agent-{i}", "round": i, "content": f"Response {i}"} for i in range(30)
        ]

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Max Responses",
                            config={"topic": "Test", "agents": ["claude"]},
                        )
                        ctx = _make_context()
                        result = await step.execute(ctx)

        assert len(result["responses"]) == 20

    @pytest.mark.asyncio
    async def test_all_agents_fail_returns_error(self):
        """Test that if all agents fail to create, an error is returned."""
        from aragora.workflow.nodes.debate import DebateStep

        with patch(
            "aragora.agents.create_agent",
            side_effect=RuntimeError("Agent unavailable"),
        ):
            with patch_arena():
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="All Agents Fail",
                            config={
                                "topic": "Test topic",
                                "agents": ["bad_agent_1", "bad_agent_2"],
                            },
                        )
                        ctx = _make_context()
                        result = await step.execute(ctx)

        assert result["success"] is False
        assert result["error"] == "No agents available"

    @pytest.mark.asyncio
    async def test_partial_agent_failure(self):
        """Test that debate continues when some agents fail to create."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.model = "claude-3"

        call_count = 0

        def selective_create(agent_type):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Agent unavailable")
            return mock_agent

        mock_result = MagicMock()
        mock_result.debate_id = "debate-partial"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 100
        del mock_result.responses

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        with patch("aragora.agents.create_agent", side_effect=selective_create):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Partial Failure",
                            config={
                                "topic": "Test",
                                "agents": ["failing_agent", "claude"],
                            },
                        )
                        ctx = _make_context()
                        result = await step.execute(ctx)

        assert result["success"] is True
        assert len(result["agents"]) == 1

    @pytest.mark.asyncio
    async def test_import_error_returns_error(self):
        """Test that ImportError is caught and returns error result."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(
            name="Import Error",
            config={"topic": "Test topic"},
        )
        ctx = _make_context()

        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'aragora'"),
        ):
            # This import error will be caught by the try/except ImportError block
            result = await step.execute(ctx)

        # The step has its own try/except that catches ImportError
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_general_exception_returns_error(self):
        """Test that general exceptions during execution are caught."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(side_effect=RuntimeError("Arena crashed"))

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Exception Test",
                            config={"topic": "Test", "agents": ["claude"]},
                        )
                        ctx = _make_context()
                        result = await step.execute(ctx)

        assert result["success"] is False
        assert "Arena crashed" in result["error"]

    @pytest.mark.asyncio
    async def test_config_merged_with_current_step_config(self):
        """Test that step config is merged with current_step_config."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(
            name="Config Merge",
            config={"topic": "Original topic", "agents": ["claude"]},
        )
        # Override topic via current_step_config
        ctx = _make_context(current_step_config={"topic": "Overridden topic"})

        mock_result = MagicMock()
        mock_result.debate_id = "merge-test"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 100
        del mock_result.responses

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol():
                        result = await step.execute(ctx)

        assert result["success"] is True
        assert result["topic"] == "Overridden topic"


# ============================================================================
# DebateStep Protocol and Arena Configuration Tests
# ============================================================================


class TestDebateStepProtocolConfig:
    """Tests for protocol and arena configuration in DebateStep."""

    @pytest.mark.asyncio
    async def test_protocol_built_with_config_values(self):
        """Test that DebateProtocol is built with config values."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "proto-test"
        mock_result.rounds_completed = 5
        mock_result.consensus_reached = True
        mock_result.consensus = "Agreed"
        mock_result.synthesis = "Synthesized"
        mock_result.duration_ms = 200
        del mock_result.responses

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol() as mock_proto_cls:
                        step = DebateStep(
                            name="Protocol Test",
                            config={
                                "topic": "Test topic",
                                "agents": ["claude"],
                                "rounds": 5,
                                "topology": "adversarial",
                                "consensus_mechanism": "unanimous",
                                "enable_critique": False,
                                "enable_synthesis": False,
                            },
                        )
                        ctx = _make_context()
                        await step.execute(ctx)

        mock_proto_cls.assert_called_once_with(
            rounds=5,
            topology="adversarial",
            consensus="unanimous",
            enable_critique=False,
            enable_synthesis=False,
        )

    @pytest.mark.asyncio
    async def test_arena_config_interpolation(self):
        """Test that arena_config string values are interpolated."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "arena-cfg"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 100
        del mock_result.responses

        mock_arena_cls = MagicMock()
        mock_arena_instance = AsyncMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_result)
        mock_arena_cls.return_value = mock_arena_instance

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        mock_arena_config_cls = MagicMock()
        mock_arena_config_instance = MagicMock()
        mock_arena_config_cls.return_value = mock_arena_config_instance

        # Mock dataclasses.fields to return field names
        mock_field_1 = MagicMock()
        mock_field_1.name = "org_id"
        mock_field_2 = MagicMock()
        mock_field_2.name = "loop_id"
        mock_field_3 = MagicMock()
        mock_field_3.name = "enable_knowledge_retrieval"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( mock_arena_cls):
                with patch_environment():
                    with patch_debate_protocol():
                        with patch(
                            "aragora.debate.arena_config.ArenaConfig",
                            mock_arena_config_cls,
                        ):
                            with patch(
                                "aragora.workflow.nodes.debate.dataclasses.fields",
                                return_value=[mock_field_1, mock_field_2, mock_field_3],
                            ):
                                step = DebateStep(
                                    name="Arena Config Interp",
                                    config={
                                        "topic": "Test",
                                        "agents": ["claude"],
                                        "arena_config": {
                                            "org_id": "{org}",
                                            "enable_knowledge_retrieval": True,
                                        },
                                    },
                                )
                                ctx = _make_context(inputs={"org": "my-org-123"})
                                result = await step.execute(ctx)

        assert result["success"] is True
        # Verify ArenaConfig was constructed with interpolated org_id
        mock_arena_config_cls.assert_called_once()
        call_kwargs = mock_arena_config_cls.call_args[1]
        assert call_kwargs["org_id"] == "my-org-123"
        assert call_kwargs["enable_knowledge_retrieval"] is True

    @pytest.mark.asyncio
    async def test_arena_config_failure_continues(self):
        """Test that ArenaConfig failure is handled gracefully."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "no-cfg"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena) as mock_arena_cls:
                with patch_environment():
                    with patch_debate_protocol():
                        with patch(
                            "aragora.debate.arena_config.ArenaConfig",
                            side_effect=Exception("Bad config"),
                        ):
                            with patch(
                                "aragora.workflow.nodes.debate.dataclasses.fields",
                                return_value=[],
                            ):
                                step = DebateStep(
                                    name="Config Failure",
                                    config={
                                        "topic": "Test",
                                        "agents": ["claude"],
                                        "arena_config": {"bad_field": "value"},
                                    },
                                )
                                ctx = _make_context()
                                result = await step.execute(ctx)

        # Debate should still succeed without arena_config
        assert result["success"] is True
        # Arena should not have received config kwarg
        call_kwargs = mock_arena_cls.call_args[1]
        assert "config" not in call_kwargs

    @pytest.mark.asyncio
    async def test_timeout_seconds_passed_to_arena(self):
        """Test that timeout_seconds is passed to Arena."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "timeout-test"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena_cls = MagicMock()
        mock_arena_instance = AsyncMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_result)
        mock_arena_cls.return_value = mock_arena_instance

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( mock_arena_cls):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Timeout Test",
                            config={
                                "topic": "Test",
                                "agents": ["claude"],
                                "timeout_seconds": 300,
                            },
                        )
                        ctx = _make_context()
                        await step.execute(ctx)

        call_kwargs = mock_arena_cls.call_args[1]
        assert call_kwargs["timeout_seconds"] == 300

    @pytest.mark.asyncio
    async def test_default_timeout_is_120(self):
        """Test that default timeout is 120 seconds."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "default-timeout"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena_cls = MagicMock()
        mock_arena_instance = AsyncMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_result)
        mock_arena_cls.return_value = mock_arena_instance

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( mock_arena_cls):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Default Timeout",
                            config={"topic": "Test", "agents": ["claude"]},
                        )
                        ctx = _make_context()
                        await step.execute(ctx)

        call_kwargs = mock_arena_cls.call_args[1]
        assert call_kwargs["timeout_seconds"] == 120


# ============================================================================
# DebateStep Memory Injection Tests
# ============================================================================


class TestDebateStepMemory:
    """Tests for memory system injection from context state."""

    @pytest.mark.asyncio
    async def test_knowledge_mound_injected_from_state(self):
        """Test that knowledge_mound from state is passed to Arena."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "km-test"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena_cls = MagicMock()
        mock_arena_instance = AsyncMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_result)
        mock_arena_cls.return_value = mock_arena_instance

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        mock_km = MagicMock(name="KnowledgeMound")

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( mock_arena_cls):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="KM Injection",
                            config={"topic": "Test", "agents": ["claude"]},
                        )
                        ctx = _make_context(state={"knowledge_mound": mock_km})
                        await step.execute(ctx)

        call_kwargs = mock_arena_cls.call_args[1]
        assert call_kwargs["knowledge_mound"] is mock_km

    @pytest.mark.asyncio
    async def test_continuum_memory_injected_from_state(self):
        """Test that continuum_memory from state is passed to Arena."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "cm-test"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena_cls = MagicMock()
        mock_arena_instance = AsyncMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_result)
        mock_arena_cls.return_value = mock_arena_instance

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        mock_cm = MagicMock(name="ContinuumMemory")

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( mock_arena_cls):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="CM Injection",
                            config={"topic": "Test", "agents": ["claude"]},
                        )
                        ctx = _make_context(state={"continuum_memory": mock_cm})
                        await step.execute(ctx)

        call_kwargs = mock_arena_cls.call_args[1]
        assert call_kwargs["continuum_memory"] is mock_cm

    @pytest.mark.asyncio
    async def test_no_memory_systems_when_not_in_state(self):
        """Test that memory kwargs are not passed when not in state."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "no-mem"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena_cls = MagicMock()
        mock_arena_instance = AsyncMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_result)
        mock_arena_cls.return_value = mock_arena_instance

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( mock_arena_cls):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="No Memory",
                            config={"topic": "Test", "agents": ["claude"]},
                        )
                        ctx = _make_context()
                        await step.execute(ctx)

        call_kwargs = mock_arena_cls.call_args[1]
        assert "knowledge_mound" not in call_kwargs
        assert "continuum_memory" not in call_kwargs

    @pytest.mark.asyncio
    async def test_both_memory_systems_injected(self):
        """Test that both memory systems can be injected simultaneously."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "both-mem"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena_cls = MagicMock()
        mock_arena_instance = AsyncMock()
        mock_arena_instance.run = AsyncMock(return_value=mock_result)
        mock_arena_cls.return_value = mock_arena_instance

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        mock_km = MagicMock(name="KnowledgeMound")
        mock_cm = MagicMock(name="ContinuumMemory")

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( mock_arena_cls):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Both Memory",
                            config={"topic": "Test", "agents": ["claude"]},
                        )
                        ctx = _make_context(
                            state={
                                "knowledge_mound": mock_km,
                                "continuum_memory": mock_cm,
                            }
                        )
                        await step.execute(ctx)

        call_kwargs = mock_arena_cls.call_args[1]
        assert call_kwargs["knowledge_mound"] is mock_km
        assert call_kwargs["continuum_memory"] is mock_cm


# ============================================================================
# DebateStep Interpolation Tests
# ============================================================================


class TestDebateStepInterpolation:
    """Tests for DebateStep._interpolate_text()."""

    def test_interpolate_input_values(self):
        """Test interpolation of {key} with input values."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(inputs={"contract_name": "NDA Agreement"})
        result = step._interpolate_text("Review the terms of {contract_name}", ctx)
        assert result == "Review the terms of NDA Agreement"

    def test_interpolate_multiple_inputs(self):
        """Test interpolation of multiple input placeholders."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(inputs={"company": "Acme", "product": "Widget"})
        result = step._interpolate_text("Should {company} launch {product}?", ctx)
        assert result == "Should Acme launch Widget?"

    def test_interpolate_step_outputs_string(self):
        """Test interpolation of {step.step_id} with string output."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(step_outputs={"analysis": "The market is growing fast"})
        result = step._interpolate_text("Given that: {step.analysis}, what should we do?", ctx)
        assert result == "Given that: The market is growing fast, what should we do?"

    def test_interpolate_step_outputs_dict_response_key(self):
        """Test interpolation of step output dict with 'response' key."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(step_outputs={"prev": {"response": "Previous analysis result"}})
        result = step._interpolate_text("Based on: {step.prev}", ctx)
        assert result == "Based on: Previous analysis result"

    def test_interpolate_step_outputs_dict_content_key(self):
        """Test interpolation of step output dict with 'content' key."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(step_outputs={"prev": {"content": "Content value"}})
        result = step._interpolate_text("Content: {step.prev}", ctx)
        assert result == "Content: Content value"

    def test_interpolate_step_outputs_dict_result_key(self):
        """Test interpolation of step output dict with 'result' key."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(step_outputs={"prev": {"result": "Result value"}})
        result = step._interpolate_text("Result: {step.prev}", ctx)
        assert result == "Result: Result value"

    def test_interpolate_step_outputs_dict_synthesis_key(self):
        """Test interpolation of step output dict with 'synthesis' key."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(step_outputs={"prev": {"synthesis": "Synthesis value"}})
        result = step._interpolate_text("Synthesis: {step.prev}", ctx)
        assert result == "Synthesis: Synthesis value"

    def test_interpolate_step_outputs_dict_priority_order(self):
        """Test that dict keys are checked in priority order: response, content, result, synthesis."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        # When multiple keys exist, 'response' should win
        ctx = _make_context(
            step_outputs={
                "prev": {
                    "response": "response_val",
                    "content": "content_val",
                    "result": "result_val",
                    "synthesis": "synthesis_val",
                }
            }
        )
        result = step._interpolate_text("Value: {step.prev}", ctx)
        assert result == "Value: response_val"

    def test_interpolate_step_outputs_dict_no_matching_key(self):
        """Test that dict without matching keys is not interpolated."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(step_outputs={"prev": {"other_field": "some value"}})
        result = step._interpolate_text("Value: {step.prev}", ctx)
        # Placeholder remains unresolved
        assert result == "Value: {step.prev}"

    def test_interpolate_state_values(self):
        """Test interpolation of {state.key} with state values."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(state={"iteration": "3", "phase": "review"})
        result = step._interpolate_text("Iteration {state.iteration}, phase: {state.phase}", ctx)
        assert result == "Iteration 3, phase: review"

    def test_interpolate_mixed_sources(self):
        """Test interpolation mixing inputs, step outputs, and state."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(
            inputs={"topic": "AI Safety"},
            step_outputs={"research": "Key findings here"},
            state={"round": "2"},
        )
        result = step._interpolate_text(
            "Discuss {topic} (round {state.round}): {step.research}", ctx
        )
        assert result == "Discuss AI Safety (round 2): Key findings here"

    def test_interpolate_no_placeholders(self):
        """Test that text without placeholders is returned unchanged."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context()
        result = step._interpolate_text("Plain text topic", ctx)
        assert result == "Plain text topic"

    def test_interpolate_unresolved_placeholders_preserved(self):
        """Test that unresolved placeholders remain in text."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context()
        result = step._interpolate_text("Missing: {unknown_var} and {state.missing}", ctx)
        assert result == "Missing: {unknown_var} and {state.missing}"

    def test_interpolate_non_string_values_converted(self):
        """Test that non-string input values are converted to strings."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(inputs={"count": 42, "active": True})
        result = step._interpolate_text("Count: {count}, Active: {active}", ctx)
        assert result == "Count: 42, Active: True"

    def test_interpolate_empty_template(self):
        """Test interpolation of empty template returns empty string."""
        from aragora.workflow.nodes.debate import DebateStep

        step = DebateStep(name="Interp")
        ctx = _make_context(inputs={"topic": "Test"})
        result = step._interpolate_text("", ctx)
        assert result == ""


# ============================================================================
# QuickDebateStep Initialization Tests
# ============================================================================


class TestQuickDebateStepInit:
    """Tests for QuickDebateStep initialization."""

    def test_basic_init(self):
        """Test basic QuickDebateStep initialization."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Quick Debate")
        assert step.name == "Quick Debate"
        assert step.config == {}

    def test_init_with_config(self):
        """Test QuickDebateStep initialization with full config."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        config = {
            "question": "What is the best approach?",
            "agents": ["claude", "gpt4", "gemini"],
            "max_response_length": 1000,
            "synthesize": False,
        }
        step = QuickDebateStep(name="Configured Quick Debate", config=config)
        assert step.config["question"] == "What is the best approach?"
        assert step.config["agents"] == ["claude", "gpt4", "gemini"]
        assert step.config["max_response_length"] == 1000
        assert step.config["synthesize"] is False

    def test_init_none_config_becomes_empty(self):
        """Test that None config becomes empty dict."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="No Config", config=None)
        assert step.config == {}

    @pytest.mark.xfail(reason="class identity pollution", strict=False)
    def test_is_base_step_subclass(self):
        """Test that QuickDebateStep is a BaseStep subclass."""
        from aragora.workflow.nodes.debate import QuickDebateStep
        from aragora.workflow.step import BaseStep

        step = QuickDebateStep(name="Test")
        assert isinstance(step, BaseStep)


# ============================================================================
# QuickDebateStep Execution Tests
# ============================================================================


class TestQuickDebateStepExecution:
    """Tests for QuickDebateStep.execute()."""

    @pytest.mark.asyncio
    async def test_empty_question_returns_error(self):
        """Test that an empty question returns an error result."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Empty Q", config={"question": ""})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["success"] is False
        assert result["error"] == "Empty question"

    @pytest.mark.asyncio
    async def test_no_question_config_returns_error(self):
        """Test that missing question config returns an error result."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="No Q", config={})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["success"] is False
        assert result["error"] == "Empty question"

    @pytest.mark.asyncio
    async def test_successful_quick_debate_no_synthesis(self):
        """Test successful quick debate without synthesis."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        mock_agent_1 = AsyncMock()
        mock_agent_1.generate = AsyncMock(return_value="Response from agent 1")

        mock_agent_2 = AsyncMock()
        mock_agent_2.generate = AsyncMock(return_value="Response from agent 2")

        agents = [mock_agent_1, mock_agent_2]
        agent_idx = 0

        def mock_create(agent_type):
            nonlocal agent_idx
            agent = agents[agent_idx % len(agents)]
            agent_idx += 1
            return agent

        with patch("aragora.agents.create_agent", side_effect=mock_create):
            step = QuickDebateStep(
                name="No Synth",
                config={
                    "question": "What is X?",
                    "agents": ["claude", "gpt4"],
                    "synthesize": False,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["question"] == "What is X?"
        assert result["agents_responded"] == 2
        assert result["agents_total"] == 2
        assert result["synthesis"] is None
        assert len(result["responses"]) == 2
        assert result["responses"][0]["success"] is True
        assert result["responses"][0]["response"] == "Response from agent 1"

    @pytest.mark.asyncio
    async def test_successful_quick_debate_with_synthesis(self):
        """Test successful quick debate with synthesis enabled."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(
            side_effect=["Agent 1 view", "Agent 2 view", "Synthesized view"]
        )

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            step = QuickDebateStep(
                name="With Synth",
                config={
                    "question": "What is X?",
                    "agents": ["claude", "gpt4"],
                    "synthesize": True,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["synthesis"] == "Synthesized view"
        assert result["agents_responded"] == 2

    @pytest.mark.asyncio
    async def test_max_response_length_truncation(self):
        """Test that responses longer than max_response_length are truncated."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        long_response = "x" * 1000

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value=long_response)

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            step = QuickDebateStep(
                name="Truncation",
                config={
                    "question": "Test question",
                    "agents": ["claude"],
                    "max_response_length": 200,
                    "synthesize": False,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["success"] is True
        assert len(result["responses"][0]["response"]) == 200

    @pytest.mark.asyncio
    async def test_response_within_limit_not_truncated(self):
        """Test that responses within max_response_length are not truncated."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        short_response = "short answer"

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value=short_response)

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            step = QuickDebateStep(
                name="No Truncation",
                config={
                    "question": "Test question",
                    "agents": ["claude"],
                    "max_response_length": 500,
                    "synthesize": False,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["responses"][0]["response"] == "short answer"

    @pytest.mark.asyncio
    async def test_default_max_response_length_is_500(self):
        """Test that default max_response_length is 500."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        response_600 = "y" * 600

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value=response_600)

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            step = QuickDebateStep(
                name="Default Length",
                config={
                    "question": "Test question",
                    "agents": ["claude"],
                    "synthesize": False,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert len(result["responses"][0]["response"]) == 500

    @pytest.mark.asyncio
    async def test_partial_agent_failure(self):
        """Test that partial agent failures are handled correctly."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        call_count = 0

        def mock_create(agent_type):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Agent creation failed")
            mock = AsyncMock()
            mock.generate = AsyncMock(return_value="Good response")
            return mock

        with patch("aragora.agents.create_agent", side_effect=mock_create):
            step = QuickDebateStep(
                name="Partial Fail",
                config={
                    "question": "Test question",
                    "agents": ["bad_agent", "good_agent"],
                    "synthesize": False,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["agents_responded"] == 1
        assert result["agents_total"] == 2
        # One response should be success, one should be failure
        successes = [r for r in result["responses"] if r["success"]]
        failures = [r for r in result["responses"] if not r["success"]]
        assert len(successes) == 1
        assert len(failures) == 1
        assert "error" in failures[0]

    @pytest.mark.asyncio
    async def test_all_agents_fail(self):
        """Test that when all agents fail, success is False."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        def mock_create(agent_type):
            raise RuntimeError("All agents down")

        with patch("aragora.agents.create_agent", side_effect=mock_create):
            step = QuickDebateStep(
                name="All Fail",
                config={
                    "question": "Test question",
                    "agents": ["bad1", "bad2"],
                    "synthesize": False,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["success"] is False
        assert result["agents_responded"] == 0
        assert result["agents_total"] == 2

    @pytest.mark.asyncio
    async def test_synthesis_with_single_response_skipped(self):
        """Test that synthesis is skipped when fewer than 2 successful responses."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        call_count = 0

        def mock_create(agent_type):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                mock = AsyncMock()
                mock.generate = AsyncMock(return_value="Only response")
                return mock
            raise RuntimeError("Agent failed")

        with patch("aragora.agents.create_agent", side_effect=mock_create):
            step = QuickDebateStep(
                name="Single Response",
                config={
                    "question": "Test question",
                    "agents": ["good", "bad"],
                    "synthesize": True,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["synthesis"] is None
        assert result["agents_responded"] == 1

    @pytest.mark.asyncio
    async def test_synthesis_failure_handled_gracefully(self):
        """Test that synthesis failure returns None synthesis."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        generate_count = 0

        def mock_create(agent_type):
            mock = AsyncMock()

            async def generate_side_effect(prompt):
                nonlocal generate_count
                generate_count += 1
                if generate_count <= 2:
                    return f"Response {generate_count}"
                # Third call is synthesis - raise error
                raise RuntimeError("Synthesis failed")

            mock.generate = AsyncMock(side_effect=generate_side_effect)
            return mock

        with patch("aragora.agents.create_agent", side_effect=mock_create):
            step = QuickDebateStep(
                name="Synth Fail",
                config={
                    "question": "Test question",
                    "agents": ["a1", "a2"],
                    "synthesize": True,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["synthesis"] is None
        assert result["agents_responded"] == 2

    @pytest.mark.asyncio
    async def test_default_agents_are_claude_and_gpt4(self):
        """Test that default agents are claude and gpt4."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="Response")

        created_agents = []

        def track_create(agent_type):
            created_agents.append(agent_type)
            return mock_agent

        with patch("aragora.agents.create_agent", side_effect=track_create):
            step = QuickDebateStep(
                name="Default Agents",
                config={
                    "question": "Test question",
                    "synthesize": False,
                },
            )
            ctx = _make_context()
            await step.execute(ctx)

        assert "claude" in created_agents
        assert "gpt4" in created_agents

    @pytest.mark.asyncio
    async def test_config_merged_with_current_step_config(self):
        """Test that step config is merged with current_step_config."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="Response")

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            step = QuickDebateStep(
                name="Config Merge",
                config={
                    "question": "Original question",
                    "agents": ["claude"],
                    "synthesize": False,
                },
            )
            ctx = _make_context(current_step_config={"question": "Override question"})
            result = await step.execute(ctx)

        assert result["question"] == "Override question"

    @pytest.mark.asyncio
    async def test_general_exception_returns_error(self):
        """Test that general exceptions are caught and returned as error."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(
            name="Exception",
            config={"question": "Test question"},
        )
        ctx = _make_context()

        # Force an exception by making asyncio.gather fail
        with patch(
            "asyncio.gather",
            side_effect=Exception("Unexpected failure"),
        ):
            with patch("aragora.agents.create_agent"):
                result = await step.execute(ctx)

        assert result["success"] is False
        assert "Unexpected failure" in result["error"]

    @pytest.mark.asyncio
    async def test_agent_generate_exception_captured(self):
        """Test that agent.generate exceptions are captured per agent."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(side_effect=TimeoutError("Agent timed out"))

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            step = QuickDebateStep(
                name="Generate Error",
                config={
                    "question": "Test question",
                    "agents": ["claude"],
                    "synthesize": False,
                },
            )
            ctx = _make_context()
            result = await step.execute(ctx)

        assert result["success"] is False
        assert result["agents_responded"] == 0
        assert result["responses"][0]["success"] is False
        assert "Agent timed out" in result["responses"][0]["error"]


# ============================================================================
# QuickDebateStep Interpolation Tests
# ============================================================================


class TestQuickDebateStepInterpolation:
    """Tests for QuickDebateStep._interpolate_text()."""

    def test_interpolate_input_values(self):
        """Test interpolation of {key} with input values."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Interp")
        ctx = _make_context(inputs={"product": "Widget"})
        result = step._interpolate_text("Should we launch {product}?", ctx)
        assert result == "Should we launch Widget?"

    def test_interpolate_multiple_inputs(self):
        """Test interpolation of multiple input placeholders."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Interp")
        ctx = _make_context(inputs={"team": "Engineering", "quarter": "Q3"})
        result = step._interpolate_text("{team} goals for {quarter}", ctx)
        assert result == "Engineering goals for Q3"

    def test_interpolate_step_outputs_string(self):
        """Test interpolation of {step.step_id} with string output."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Interp")
        ctx = _make_context(step_outputs={"analysis": "Market is growing"})
        result = step._interpolate_text("Context: {step.analysis}", ctx)
        assert result == "Context: Market is growing"

    def test_interpolate_step_outputs_dict_not_resolved(self):
        """Test that QuickDebateStep does not resolve dict step outputs."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Interp")
        ctx = _make_context(step_outputs={"prev": {"response": "Dict value"}})
        # QuickDebateStep only handles string step outputs
        result = step._interpolate_text("Value: {step.prev}", ctx)
        assert result == "Value: {step.prev}"

    def test_interpolate_no_state_support(self):
        """Test that QuickDebateStep does not support state interpolation."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Interp")
        ctx = _make_context(state={"count": "5"})
        result = step._interpolate_text("Count: {state.count}", ctx)
        # State placeholders should remain unresolved
        assert result == "Count: {state.count}"

    def test_interpolate_empty_template(self):
        """Test interpolation of empty template returns empty string."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Interp")
        ctx = _make_context()
        result = step._interpolate_text("", ctx)
        assert result == ""

    def test_interpolate_no_placeholders(self):
        """Test that text without placeholders is returned unchanged."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Interp")
        ctx = _make_context()
        result = step._interpolate_text("Plain question?", ctx)
        assert result == "Plain question?"

    def test_interpolate_non_string_input_converted(self):
        """Test that non-string input values are converted to strings."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        step = QuickDebateStep(name="Interp")
        ctx = _make_context(inputs={"count": 42})
        result = step._interpolate_text("Count is {count}", ctx)
        assert result == "Count is 42"


# ============================================================================
# Topic/Question Interpolation During Execution
# ============================================================================


class TestExecutionInterpolation:
    """Tests for topic/question interpolation during execute()."""

    @pytest.mark.asyncio
    async def test_debate_step_topic_interpolated_from_inputs(self):
        """Test that DebateStep interpolates topic from context inputs."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "interp-topic"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment() as mock_env:
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Topic Interp",
                            config={
                                "topic": "Evaluate {product} for {market}",
                                "agents": ["claude"],
                            },
                        )
                        ctx = _make_context(
                            inputs={
                                "product": "AI Assistant",
                                "market": "healthcare",
                            }
                        )
                        result = await step.execute(ctx)

        assert result["topic"] == "Evaluate AI Assistant for healthcare"

    @pytest.mark.asyncio
    async def test_quick_debate_question_interpolated_from_inputs(self):
        """Test that QuickDebateStep interpolates question from context inputs."""
        from aragora.workflow.nodes.debate import QuickDebateStep

        mock_agent = AsyncMock()
        mock_agent.generate = AsyncMock(return_value="Answer")

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            step = QuickDebateStep(
                name="Question Interp",
                config={
                    "question": "Is {feature} ready for {env}?",
                    "agents": ["claude"],
                    "synthesize": False,
                },
            )
            ctx = _make_context(inputs={"feature": "dark mode", "env": "production"})
            result = await step.execute(ctx)

        assert result["question"] == "Is dark mode ready for production?"

    @pytest.mark.asyncio
    async def test_debate_step_topic_interpolated_from_state(self):
        """Test that DebateStep interpolates topic from context state."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "state-interp"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="State Interp",
                            config={
                                "topic": "Phase: {state.phase}",
                                "agents": ["claude"],
                            },
                        )
                        ctx = _make_context(state={"phase": "review"})
                        result = await step.execute(ctx)

        assert result["topic"] == "Phase: review"

    @pytest.mark.asyncio
    async def test_debate_step_topic_interpolated_from_step_outputs(self):
        """Test that DebateStep interpolates topic from step outputs."""
        from aragora.workflow.nodes.debate import DebateStep

        mock_result = MagicMock()
        mock_result.debate_id = "step-output-interp"
        mock_result.rounds_completed = 1
        mock_result.consensus_reached = False
        mock_result.consensus = None
        mock_result.synthesis = None
        mock_result.duration_ms = 50
        del mock_result.responses

        mock_arena = AsyncMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_agent = MagicMock()
        mock_agent.name = "agent"
        mock_agent.model = "model"

        with patch("aragora.agents.create_agent", return_value=mock_agent):
            with patch_arena( return_value=mock_arena):
                with patch_environment():
                    with patch_debate_protocol():
                        step = DebateStep(
                            name="Step Output Interp",
                            config={
                                "topic": "Review: {step.analysis}",
                                "agents": ["claude"],
                            },
                        )
                        ctx = _make_context(
                            step_outputs={"analysis": "The system needs refactoring"}
                        )
                        result = await step.execute(ctx)

        assert result["topic"] == "Review: The system needs refactoring"
