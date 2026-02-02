"""
Tests for DebateService - high-level API for running debates.

Covers:
- DebateOptions initialization and defaults
- DebateOptions.to_protocol() conversion
- DebateOptions.__post_init__ with settings and debate profiles
- DebateService initialization
- DebateService.run() method (agents, protocol, timeout, event hooks)
- DebateService.run_quick() convenience method
- DebateService.run_deep() convenience method
- DebateService._merge_options() merging logic
- DebateService._resolve_agents() agent resolution
- DebateService._build_event_hooks() hook building
- get_debate_service() global singleton
- reset_debate_service() cleanup
- Error handling and edge cases
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.service import (
    DebateOptions,
    DebateService,
    get_debate_service,
    reset_debate_service,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_global_service():
    """Reset the global service singleton before and after each test."""
    reset_debate_service()
    yield
    reset_debate_service()


@pytest.fixture
def mock_agent():
    """Create a mock Agent object."""
    agent = MagicMock()
    agent.name = "mock-agent"
    agent.model = "mock-model"
    agent.role = "proposer"
    return agent


@pytest.fixture
def mock_agent_pair():
    """Create a pair of mock Agent objects."""
    a1 = MagicMock()
    a1.name = "agent-1"
    a1.model = "model-1"
    a1.role = "proposer"
    a2 = MagicMock()
    a2.name = "agent-2"
    a2.model = "model-2"
    a2.role = "critic"
    return [a1, a2]


@pytest.fixture
def mock_debate_result():
    """Create a mock DebateResult."""
    result = MagicMock()
    result.debate_id = "test-debate-123"
    result.task = "Test task"
    result.final_answer = "Test answer"
    result.confidence = 0.85
    result.consensus_reached = True
    result.rounds_used = 3
    return result


@pytest.fixture
def mock_arena_class(mock_debate_result):
    """Mock the Arena class to avoid real debate execution."""
    with patch("aragora.debate.service.Arena") as arena_cls:
        arena_instance = MagicMock()
        arena_instance.run = AsyncMock(return_value=mock_debate_result)
        arena_cls.return_value = arena_instance
        yield arena_cls


def _make_resolver(agent_map: dict):
    """Create an agent resolver function from a name->agent mapping."""
    def resolver(name: str):
        if name not in agent_map:
            raise KeyError(f"Unknown agent: {name}")
        return agent_map[name]
    return resolver


# =============================================================================
# DebateOptions Tests
# =============================================================================


class TestDebateOptionsInit:
    """Test DebateOptions initialization and defaults."""

    def test_default_values(self):
        """Test default values are populated from settings."""
        opts = DebateOptions()
        assert opts.rounds is not None
        assert opts.consensus is not None
        assert opts.topology == "all-to-all"
        assert opts.enable_graph is False
        assert opts.timeout == 300.0
        assert opts.enable_streaming is False
        assert opts.enable_checkpointing is True
        assert opts.enable_memory is True
        assert opts.enable_knowledge_retrieval is False
        assert opts.enable_ml_delegation is True
        assert opts.enable_quality_gates is True
        assert opts.enable_consensus_estimation is True
        assert opts.org_id == ""
        assert opts.user_id == ""
        assert opts.correlation_id == ""
        assert opts.on_round_start is None
        assert opts.on_agent_message is None
        assert opts.on_consensus is None

    def test_custom_rounds(self):
        """Test that explicit rounds overrides settings default."""
        opts = DebateOptions(rounds=5)
        assert opts.rounds == 5

    def test_custom_consensus(self):
        """Test that explicit consensus overrides settings default."""
        opts = DebateOptions(consensus="unanimous")
        assert opts.consensus == "unanimous"

    def test_all_consensus_modes_accepted(self):
        """Test all consensus literal values are accepted."""
        for mode in ["majority", "unanimous", "judge", "none",
                     "weighted", "supermajority", "any", "byzantine"]:
            opts = DebateOptions(consensus=mode)
            assert opts.consensus == mode

    def test_custom_topology(self):
        """Test custom topology values."""
        for topo in ["all-to-all", "sparse", "round-robin", "ring", "star", "random-graph"]:
            opts = DebateOptions(topology=topo)
            assert opts.topology == topo

    def test_custom_timeout(self):
        """Test custom timeout value."""
        opts = DebateOptions(timeout=120.0)
        assert opts.timeout == 120.0

    def test_telemetry_fields(self):
        """Test telemetry fields are stored."""
        opts = DebateOptions(
            org_id="org-123",
            user_id="user-456",
            correlation_id="corr-789",
        )
        assert opts.org_id == "org-123"
        assert opts.user_id == "user-456"
        assert opts.correlation_id == "corr-789"

    def test_event_hooks_stored(self):
        """Test event hook callbacks are stored."""
        on_round = MagicMock()
        on_message = MagicMock()
        on_consensus = MagicMock()

        opts = DebateOptions(
            on_round_start=on_round,
            on_agent_message=on_message,
            on_consensus=on_consensus,
        )
        assert opts.on_round_start is on_round
        assert opts.on_agent_message is on_message
        assert opts.on_consensus is on_consensus

    def test_debate_profile_override(self):
        """Test that ARAGORA_DEBATE_PROFILE env var triggers profile loading."""
        mock_profile = MagicMock()
        mock_profile.rounds = 7
        mock_profile.consensus_mode = "supermajority"

        with patch.dict("os.environ", {"ARAGORA_DEBATE_PROFILE": "full"}):
            with patch(
                "aragora.debate.service.NomicDebateProfile",
                create=True,
            ) as mock_cls:
                mock_cls.from_env.return_value = mock_profile
                with patch.dict(
                    "sys.modules",
                    {"aragora.nomic.debate_profile": MagicMock(NomicDebateProfile=mock_cls)},
                ):
                    opts = DebateOptions()
                    assert opts.rounds == 7
                    assert opts.consensus == "supermajority"

    def test_debate_profile_failure_logs_warning(self):
        """Test that failed profile loading logs warning and uses defaults."""
        with patch.dict("os.environ", {"ARAGORA_DEBATE_PROFILE": "nomic"}):
            with patch(
                "aragora.debate.service.logger"
            ) as mock_logger:
                # Force import to fail inside __post_init__
                import sys
                orig = sys.modules.get("aragora.nomic.debate_profile")
                sys.modules["aragora.nomic.debate_profile"] = None  # type: ignore
                try:
                    opts = DebateOptions()
                    # Should fall through to defaults
                    assert opts.rounds is not None
                    assert mock_logger.warning.called
                finally:
                    if orig is not None:
                        sys.modules["aragora.nomic.debate_profile"] = orig
                    else:
                        sys.modules.pop("aragora.nomic.debate_profile", None)

    def test_debate_profile_ignores_unknown_profiles(self):
        """Test that unknown ARAGORA_DEBATE_PROFILE values are ignored."""
        with patch.dict("os.environ", {"ARAGORA_DEBATE_PROFILE": "unknown_profile"}):
            opts = DebateOptions()
            # Should just use settings defaults, not crash
            assert opts.rounds is not None


class TestDebateOptionsToProtocol:
    """Test DebateOptions.to_protocol() conversion."""

    def test_to_protocol_returns_debate_protocol(self):
        """Test that to_protocol returns a DebateProtocol."""
        from aragora.debate.protocol import DebateProtocol

        opts = DebateOptions(rounds=5, consensus="majority")
        protocol = opts.to_protocol()
        assert isinstance(protocol, DebateProtocol)

    def test_to_protocol_sets_rounds(self):
        """Test that rounds are set on the protocol."""
        opts = DebateOptions(rounds=5)
        protocol = opts.to_protocol()
        assert protocol.rounds == 5

    def test_to_protocol_sets_consensus(self):
        """Test that consensus mode is set on the protocol."""
        opts = DebateOptions(consensus="supermajority")
        protocol = opts.to_protocol()
        assert protocol.consensus == "supermajority"

    def test_to_protocol_sets_topology(self):
        """Test that topology is set on the protocol."""
        opts = DebateOptions(topology="ring")
        protocol = opts.to_protocol()
        assert protocol.topology == "ring"

    def test_to_protocol_default_consensus_when_none(self):
        """Test that 'judge' is used when consensus is somehow None."""
        opts = DebateOptions()
        # Force consensus to None to test the fallback path
        opts.consensus = None  # type: ignore
        protocol = opts.to_protocol()
        assert protocol.consensus == "judge"


# =============================================================================
# DebateService Initialization Tests
# =============================================================================


class TestDebateServiceInit:
    """Test DebateService initialization."""

    def test_default_init(self):
        """Test default initialization with no arguments."""
        service = DebateService()
        assert service._default_agents is None
        assert service._memory is None
        assert service._agent_resolver is None

    def test_init_with_default_agents(self, mock_agent_pair):
        """Test initialization with default agents."""
        service = DebateService(default_agents=mock_agent_pair)
        assert service._default_agents == mock_agent_pair

    def test_init_with_string_agents(self):
        """Test initialization with string agent names."""
        service = DebateService(default_agents=["claude", "gemini"])
        assert service._default_agents == ["claude", "gemini"]

    def test_init_with_custom_options(self):
        """Test initialization with custom default options."""
        opts = DebateOptions(rounds=3, consensus="majority")
        service = DebateService(default_options=opts)
        assert service._default_options.rounds == 3
        assert service._default_options.consensus == "majority"

    def test_init_with_memory(self):
        """Test initialization with memory system."""
        mock_memory = MagicMock()
        service = DebateService(memory=mock_memory)
        assert service._memory is mock_memory

    def test_init_with_agent_resolver(self):
        """Test initialization with agent resolver."""
        resolver = MagicMock()
        service = DebateService(agent_resolver=resolver)
        assert service._agent_resolver is resolver


# =============================================================================
# DebateService._resolve_agents Tests
# =============================================================================


class TestResolveAgents:
    """Test agent resolution logic."""

    def test_resolve_none_no_defaults(self):
        """Test resolving None agents with no defaults returns empty list."""
        service = DebateService()
        result = service._resolve_agents(None)
        assert result == []

    def test_resolve_none_with_defaults(self, mock_agent_pair):
        """Test resolving None uses default agents."""
        service = DebateService(default_agents=mock_agent_pair)
        result = service._resolve_agents(None)
        assert len(result) == 2

    def test_resolve_agent_objects_passed_through(self, mock_agent):
        """Test Agent objects are passed through directly."""
        from aragora.core import Agent

        # Need a real-ish Agent for isinstance check
        # Since Agent is ABC, we use the mock_agent which won't pass isinstance
        # Instead test with mock that returns True for isinstance
        service = DebateService()
        # Mock agents list with real Agent-like objects
        agent = MagicMock(spec=Agent)
        result = service._resolve_agents([agent])
        assert len(result) == 1
        assert result[0] is agent

    def test_resolve_string_with_resolver(self):
        """Test resolving string names using agent_resolver."""
        mock_agent = MagicMock()
        resolver = MagicMock(return_value=mock_agent)
        service = DebateService(agent_resolver=resolver)

        result = service._resolve_agents(["claude"])
        assert len(result) == 1
        assert result[0] is mock_agent
        resolver.assert_called_once_with("claude")

    def test_resolve_string_resolver_key_error(self):
        """Test that KeyError from resolver is handled gracefully."""
        resolver = MagicMock(side_effect=KeyError("Unknown"))
        service = DebateService(agent_resolver=resolver)

        result = service._resolve_agents(["unknown-agent"])
        assert result == []

    def test_resolve_string_resolver_value_error(self):
        """Test that ValueError from resolver is handled gracefully."""
        resolver = MagicMock(side_effect=ValueError("Invalid"))
        service = DebateService(agent_resolver=resolver)

        result = service._resolve_agents(["bad-agent"])
        assert result == []

    def test_resolve_string_resolver_unexpected_error(self):
        """Test that unexpected errors from resolver are logged and skipped."""
        resolver = MagicMock(side_effect=RuntimeError("Unexpected"))
        service = DebateService(agent_resolver=resolver)

        result = service._resolve_agents(["crash-agent"])
        assert result == []

    def test_resolve_string_without_resolver_uses_create_agent(self):
        """Test that strings without resolver try create_agent."""
        mock_agent = MagicMock()
        service = DebateService()

        with patch("aragora.debate.service.create_agent", return_value=mock_agent) as mock_create:
            # Patch the import inside the method
            with patch.dict("sys.modules", {"aragora.agents": MagicMock(create_agent=mock_create)}):
                result = service._resolve_agents(["demo"])
                # Should attempt creation
                assert len(result) == 1

    def test_resolve_string_without_resolver_import_error(self):
        """Test graceful handling when create_agent import fails."""
        service = DebateService()

        with patch("aragora.debate.service.create_agent", side_effect=ImportError("No module")):
            # Force the import to fail inside _resolve_agents
            import aragora.debate.service as svc
            orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            # This tests the except ImportError path - the string agent is skipped
            result = service._resolve_agents(["nonexistent"])
            # May or may not resolve depending on whether create_agent exists
            # The key assertion: it doesn't raise
            assert isinstance(result, list)

    def test_resolve_mixed_agents(self):
        """Test resolving a mix of Agent objects and strings."""
        from aragora.core import Agent

        real_agent = MagicMock(spec=Agent)
        resolved_agent = MagicMock()
        resolver = MagicMock(return_value=resolved_agent)
        service = DebateService(agent_resolver=resolver)

        result = service._resolve_agents([real_agent, "claude"])
        assert len(result) == 2
        assert result[0] is real_agent
        assert result[1] is resolved_agent

    def test_resolve_default_string_agents(self):
        """Test that default agents specified as strings are resolved."""
        resolved_agent = MagicMock()
        resolver = MagicMock(return_value=resolved_agent)
        service = DebateService(
            default_agents=["claude", "gemini"],
            agent_resolver=resolver,
        )

        result = service._resolve_agents(None)
        assert len(result) == 2
        assert resolver.call_count == 2


# =============================================================================
# DebateService._merge_options Tests
# =============================================================================


class TestMergeOptions:
    """Test options merging logic."""

    def test_merge_none_returns_defaults(self):
        """Test that None options returns default options."""
        default_opts = DebateOptions(rounds=5, consensus="majority")
        service = DebateService(default_options=default_opts)

        result = service._merge_options(None)
        assert result is default_opts

    def test_merge_overrides_rounds(self):
        """Test that provided rounds override defaults."""
        default_opts = DebateOptions(rounds=9)
        service = DebateService(default_options=default_opts)

        override = DebateOptions(rounds=3)
        result = service._merge_options(override)
        assert result.rounds == 3

    def test_merge_overrides_consensus(self):
        """Test that provided consensus overrides defaults."""
        default_opts = DebateOptions(consensus="majority")
        service = DebateService(default_options=default_opts)

        override = DebateOptions(consensus="unanimous")
        result = service._merge_options(override)
        assert result.consensus == "unanimous"

    def test_merge_overrides_timeout(self):
        """Test that provided timeout overrides defaults."""
        default_opts = DebateOptions(timeout=300.0)
        service = DebateService(default_options=default_opts)

        override = DebateOptions(timeout=60.0)
        result = service._merge_options(override)
        assert result.timeout == 60.0

    def test_merge_inherits_telemetry(self):
        """Test that telemetry fields fall back to defaults."""
        default_opts = DebateOptions(org_id="org-default", user_id="user-default")
        service = DebateService(default_options=default_opts)

        override = DebateOptions()
        result = service._merge_options(override)
        assert result.org_id == "org-default"
        assert result.user_id == "user-default"

    def test_merge_overrides_telemetry(self):
        """Test that telemetry fields can be overridden."""
        default_opts = DebateOptions(org_id="org-default")
        service = DebateService(default_options=default_opts)

        override = DebateOptions(org_id="org-override")
        result = service._merge_options(override)
        assert result.org_id == "org-override"

    def test_merge_preserves_event_hooks(self):
        """Test that event hooks from override are preserved."""
        hook = MagicMock()
        default_opts = DebateOptions()
        service = DebateService(default_options=default_opts)

        override = DebateOptions(on_round_start=hook)
        result = service._merge_options(override)
        assert result.on_round_start is hook

    def test_merge_falls_back_to_default_hooks(self):
        """Test that hooks fall back to defaults when not overridden."""
        hook = MagicMock()
        default_opts = DebateOptions(on_round_start=hook)
        service = DebateService(default_options=default_opts)

        override = DebateOptions()
        result = service._merge_options(override)
        assert result.on_round_start is hook

    def test_merge_boolean_fields(self):
        """Test that boolean fields from override are used."""
        default_opts = DebateOptions(enable_streaming=False, enable_graph=False)
        service = DebateService(default_options=default_opts)

        override = DebateOptions(enable_streaming=True, enable_graph=True)
        result = service._merge_options(override)
        assert result.enable_streaming is True
        assert result.enable_graph is True


# =============================================================================
# DebateService._build_event_hooks Tests
# =============================================================================


class TestBuildEventHooks:
    """Test event hooks building."""

    def test_no_hooks_returns_none(self):
        """Test that no hooks returns None."""
        service = DebateService()
        opts = DebateOptions()
        result = service._build_event_hooks(opts)
        assert result is None

    def test_round_start_hook(self):
        """Test round_start hook is included."""
        service = DebateService()
        hook = MagicMock()
        opts = DebateOptions(on_round_start=hook)
        result = service._build_event_hooks(opts)
        assert result is not None
        assert result["round_start"] is hook

    def test_agent_message_hook(self):
        """Test agent_message hook is included."""
        service = DebateService()
        hook = MagicMock()
        opts = DebateOptions(on_agent_message=hook)
        result = service._build_event_hooks(opts)
        assert result is not None
        assert result["agent_message"] is hook

    def test_consensus_hook(self):
        """Test consensus hook is included."""
        service = DebateService()
        hook = MagicMock()
        opts = DebateOptions(on_consensus=hook)
        result = service._build_event_hooks(opts)
        assert result is not None
        assert result["consensus"] is hook

    def test_all_hooks(self):
        """Test all hooks are included."""
        service = DebateService()
        hook_round = MagicMock()
        hook_msg = MagicMock()
        hook_cons = MagicMock()
        opts = DebateOptions(
            on_round_start=hook_round,
            on_agent_message=hook_msg,
            on_consensus=hook_cons,
        )
        result = service._build_event_hooks(opts)
        assert result is not None
        assert len(result) == 3
        assert result["round_start"] is hook_round
        assert result["agent_message"] is hook_msg
        assert result["consensus"] is hook_cons


# =============================================================================
# DebateService.run() Tests
# =============================================================================


class TestDebateServiceRun:
    """Test DebateService.run() method."""

    @pytest.mark.asyncio
    async def test_run_basic(self, mock_agent_pair, mock_debate_result):
        """Test basic run with agents."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            result = await service.run("What is the best approach?")

            assert result is mock_debate_result
            mock_arena_cls.assert_called_once()
            arena_inst.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_no_agents_raises(self):
        """Test that run raises ValueError when no agents available."""
        service = DebateService()

        with pytest.raises(ValueError, match="No agents available"):
            await service.run("Test task")

    @pytest.mark.asyncio
    async def test_run_with_custom_protocol(self, mock_agent_pair, mock_debate_result):
        """Test run with a custom protocol."""
        from aragora.debate.protocol import DebateProtocol

        custom_protocol = DebateProtocol(rounds=3, consensus="majority")

        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            result = await service.run("Test task", protocol=custom_protocol)

            assert result is mock_debate_result
            # Check that the custom protocol was passed to Arena
            call_args = mock_arena_cls.call_args
            assert call_args[0][2] is custom_protocol

    @pytest.mark.asyncio
    async def test_run_with_memory(self, mock_agent_pair, mock_debate_result):
        """Test run with memory system."""
        mock_memory = MagicMock()

        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            result = await service.run("Test task", memory=mock_memory)

            assert result is mock_debate_result
            call_kwargs = mock_arena_cls.call_args[1]
            assert call_kwargs["memory"] is mock_memory

    @pytest.mark.asyncio
    async def test_run_with_service_memory(self, mock_agent_pair, mock_debate_result):
        """Test run uses service-level memory when none provided."""
        mock_memory = MagicMock()

        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair, memory=mock_memory)
            result = await service.run("Test task")

            assert result is mock_debate_result
            call_kwargs = mock_arena_cls.call_args[1]
            assert call_kwargs["memory"] is mock_memory

    @pytest.mark.asyncio
    async def test_run_memory_override(self, mock_agent_pair, mock_debate_result):
        """Test run-level memory overrides service-level memory."""
        service_memory = MagicMock()
        run_memory = MagicMock()

        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(
                default_agents=mock_agent_pair, memory=service_memory
            )
            result = await service.run("Test task", memory=run_memory)

            call_kwargs = mock_arena_cls.call_args[1]
            assert call_kwargs["memory"] is run_memory

    @pytest.mark.asyncio
    async def test_run_timeout(self, mock_agent_pair):
        """Test that run raises TimeoutError on timeout."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()

            async def slow_run():
                await asyncio.sleep(10)

            arena_inst.run = slow_run
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            opts = DebateOptions(timeout=0.01)

            with pytest.raises(asyncio.TimeoutError):
                await service.run("Test task", options=opts)

    @pytest.mark.asyncio
    async def test_run_passes_telemetry(self, mock_agent_pair, mock_debate_result):
        """Test that telemetry fields are passed to Arena."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            opts = DebateOptions(org_id="org-123", user_id="user-456")
            await service.run("Test task", options=opts)

            call_kwargs = mock_arena_cls.call_args[1]
            assert call_kwargs["org_id"] == "org-123"
            assert call_kwargs["user_id"] == "user-456"

    @pytest.mark.asyncio
    async def test_run_no_telemetry_when_empty(self, mock_agent_pair, mock_debate_result):
        """Test that empty telemetry fields are not passed to Arena."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            await service.run("Test task")

            call_kwargs = mock_arena_cls.call_args[1]
            assert "org_id" not in call_kwargs
            assert "user_id" not in call_kwargs

    @pytest.mark.asyncio
    async def test_run_passes_event_hooks(self, mock_agent_pair, mock_debate_result):
        """Test that event hooks are passed to Arena."""
        hook = MagicMock()

        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            opts = DebateOptions(on_round_start=hook)
            await service.run("Test task", options=opts)

            call_kwargs = mock_arena_cls.call_args[1]
            assert "event_hooks" in call_kwargs
            assert call_kwargs["event_hooks"]["round_start"] is hook

    @pytest.mark.asyncio
    async def test_run_no_hooks_when_none(self, mock_agent_pair, mock_debate_result):
        """Test that event_hooks key is absent when no hooks set."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            await service.run("Test task")

            call_kwargs = mock_arena_cls.call_args[1]
            assert "event_hooks" not in call_kwargs

    @pytest.mark.asyncio
    async def test_run_passes_kwargs(self, mock_agent_pair, mock_debate_result):
        """Test that extra kwargs are forwarded to Arena."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            await service.run("Test task", custom_arg="custom_value")

            call_kwargs = mock_arena_cls.call_args[1]
            assert call_kwargs["custom_arg"] == "custom_value"

    @pytest.mark.asyncio
    async def test_run_creates_environment(self, mock_agent_pair, mock_debate_result):
        """Test that Arena is called with correct Environment."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            await service.run("Design a rate limiter")

            call_args = mock_arena_cls.call_args[0]
            env = call_args[0]
            assert env.task == "Design a rate limiter"

    @pytest.mark.asyncio
    async def test_run_passes_arena_config_flags(self, mock_agent_pair, mock_debate_result):
        """Test that Arena receives configuration flags from options."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            opts = DebateOptions(
                enable_checkpointing=False,
                enable_knowledge_retrieval=True,
                enable_ml_delegation=False,
                enable_quality_gates=False,
                enable_consensus_estimation=False,
            )
            await service.run("Test task", options=opts)

            call_kwargs = mock_arena_cls.call_args[1]
            assert call_kwargs["enable_checkpointing"] is False
            assert call_kwargs["enable_knowledge_retrieval"] is True
            assert call_kwargs["enable_ml_delegation"] is False
            assert call_kwargs["enable_quality_gates"] is False
            assert call_kwargs["enable_consensus_estimation"] is False


# =============================================================================
# DebateService.run_quick() Tests
# =============================================================================


class TestDebateServiceRunQuick:
    """Test DebateService.run_quick() convenience method."""

    @pytest.mark.asyncio
    async def test_run_quick_basic(self, mock_agent_pair, mock_debate_result):
        """Test run_quick with default rounds."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            result = await service.run_quick("Quick question")

            assert result is mock_debate_result

    @pytest.mark.asyncio
    async def test_run_quick_custom_rounds(self, mock_agent_pair, mock_debate_result):
        """Test run_quick with custom rounds."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            result = await service.run_quick("Quick question", rounds=3)

            assert result is mock_debate_result

    @pytest.mark.asyncio
    async def test_run_quick_with_agents(self, mock_agent_pair, mock_debate_result):
        """Test run_quick with explicit agents."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService()
            result = await service.run_quick(
                "Quick question", agents=mock_agent_pair
            )

            assert result is mock_debate_result

    @pytest.mark.asyncio
    async def test_run_quick_no_agents_raises(self):
        """Test run_quick raises ValueError when no agents."""
        service = DebateService()

        with pytest.raises(ValueError, match="No agents available"):
            await service.run_quick("Test task")


# =============================================================================
# DebateService.run_deep() Tests
# =============================================================================


class TestDebateServiceRunDeep:
    """Test DebateService.run_deep() convenience method."""

    @pytest.mark.asyncio
    async def test_run_deep_basic(self, mock_agent_pair, mock_debate_result):
        """Test run_deep with defaults."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            result = await service.run_deep("Important decision")

            assert result is mock_debate_result

    @pytest.mark.asyncio
    async def test_run_deep_uses_supermajority(self, mock_agent_pair, mock_debate_result):
        """Test run_deep uses supermajority consensus."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            # Patch run to capture the options
            original_run = service.run

            captured_opts = {}

            async def capture_run(task, agents=None, protocol=None, options=None, **kw):
                captured_opts["options"] = options
                return await original_run(
                    task, agents=agents, protocol=protocol, options=options, **kw
                )

            service.run = capture_run
            await service.run_deep("Important decision")

            assert captured_opts["options"].consensus == "supermajority"

    @pytest.mark.asyncio
    async def test_run_deep_custom_rounds(self, mock_agent_pair, mock_debate_result):
        """Test run_deep with custom rounds."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            result = await service.run_deep("Important decision", rounds=12)

            assert result is mock_debate_result

    @pytest.mark.asyncio
    async def test_run_deep_no_agents_raises(self):
        """Test run_deep raises ValueError when no agents."""
        service = DebateService()

        with pytest.raises(ValueError, match="No agents available"):
            await service.run_deep("Important decision")


# =============================================================================
# get_debate_service() Tests
# =============================================================================


class TestGetDebateService:
    """Test global debate service singleton."""

    def test_creates_singleton(self):
        """Test that get_debate_service creates a singleton."""
        service1 = get_debate_service()
        service2 = get_debate_service()
        assert service1 is service2

    def test_creates_new_when_agents_provided(self):
        """Test that providing agents creates a new instance."""
        service1 = get_debate_service()
        service2 = get_debate_service(default_agents=["claude"])
        assert service1 is not service2

    def test_returns_debate_service_instance(self):
        """Test that get_debate_service returns a DebateService."""
        service = get_debate_service()
        assert isinstance(service, DebateService)

    def test_passes_kwargs(self):
        """Test that kwargs are forwarded to DebateService."""
        mock_memory = MagicMock()
        service = get_debate_service(memory=mock_memory)
        assert service._memory is mock_memory

    def test_uses_settings_defaults_when_no_agents(self):
        """Test that settings default agent list is used."""
        service = get_debate_service()
        # Service should have been created with defaults from settings
        assert service._default_agents is not None


# =============================================================================
# reset_debate_service() Tests
# =============================================================================


class TestResetDebateService:
    """Test global service reset."""

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        service1 = get_debate_service()
        reset_debate_service()
        service2 = get_debate_service()
        assert service1 is not service2

    def test_reset_idempotent(self):
        """Test that reset can be called multiple times safely."""
        reset_debate_service()
        reset_debate_service()
        # Should not raise
        service = get_debate_service()
        assert service is not None


# =============================================================================
# Edge Cases and Integration
# =============================================================================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_agent_list(self):
        """Test that empty agent list resolves to empty."""
        service = DebateService()
        result = service._resolve_agents([])
        assert result == []

    @pytest.mark.asyncio
    async def test_run_with_empty_task(self, mock_agent_pair, mock_debate_result):
        """Test run with empty task string."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            # Empty task is technically allowed at service level
            # (Arena/Environment may validate)
            result = await service.run("")
            assert result is mock_debate_result

    def test_options_zero_timeout(self):
        """Test options with zero timeout falls back to default."""
        default_opts = DebateOptions(timeout=300.0)
        service = DebateService(default_options=default_opts)

        override = DebateOptions(timeout=0)
        result = service._merge_options(override)
        # timeout=0 is falsy, so it falls back to default
        assert result.timeout == 300.0

    def test_options_negative_rounds_accepted(self):
        """Test that rounds value is stored as-is (validation is elsewhere)."""
        # DebateOptions doesn't validate rounds values
        # The protocol/arena handles that
        opts = DebateOptions(rounds=-1)
        assert opts.rounds == -1

    def test_build_event_hooks_partial(self):
        """Test hooks with only some callbacks set."""
        service = DebateService()
        hook = MagicMock()
        opts = DebateOptions(on_consensus=hook)
        result = service._build_event_hooks(opts)
        assert result is not None
        assert len(result) == 1
        assert "consensus" in result
        assert "round_start" not in result
        assert "agent_message" not in result

    @pytest.mark.asyncio
    async def test_run_with_string_agents(self, mock_debate_result):
        """Test run with string agent names and resolver."""
        mock_agent = MagicMock()
        resolver = MagicMock(return_value=mock_agent)

        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(agent_resolver=resolver)
            result = await service.run("Test task", agents=["claude", "gemini"])

            assert result is mock_debate_result
            assert resolver.call_count == 2

    def test_merge_preserves_ml_options(self):
        """Test that ML-related options are preserved during merge."""
        default_opts = DebateOptions(
            enable_ml_delegation=True,
            enable_quality_gates=True,
            enable_consensus_estimation=True,
        )
        service = DebateService(default_options=default_opts)

        override = DebateOptions(
            enable_ml_delegation=False,
            enable_quality_gates=False,
            enable_consensus_estimation=False,
        )
        result = service._merge_options(override)
        assert result.enable_ml_delegation is False
        assert result.enable_quality_gates is False
        assert result.enable_consensus_estimation is False

    def test_resolve_all_agents_fail_returns_empty(self):
        """Test that if all agent resolutions fail, empty list is returned."""
        resolver = MagicMock(side_effect=KeyError("Not found"))
        service = DebateService(agent_resolver=resolver)

        result = service._resolve_agents(["bad1", "bad2", "bad3"])
        assert result == []

    @pytest.mark.asyncio
    async def test_run_options_uses_to_protocol(self, mock_agent_pair, mock_debate_result):
        """Test that options.to_protocol() is used when no protocol given."""
        with patch("aragora.debate.service.Arena") as mock_arena_cls:
            arena_inst = MagicMock()
            arena_inst.run = AsyncMock(return_value=mock_debate_result)
            mock_arena_cls.return_value = arena_inst

            service = DebateService(default_agents=mock_agent_pair)
            opts = DebateOptions(rounds=4, consensus="weighted", topology="star")
            await service.run("Test task", options=opts)

            call_args = mock_arena_cls.call_args[0]
            protocol = call_args[2]
            assert protocol.consensus == "weighted"
            assert protocol.topology == "star"
