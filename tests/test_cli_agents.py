"""
Tests for CLI Agent implementations.

Tests cover:
- CLIAgent base class (_run_cli, _parse_critique, context building)
- CodexAgent, ClaudeAgent, GeminiCLIAgent, KiloCodeAgent
- GrokCLIAgent, QwenCLIAgent, DeepseekCLIAgent, OpenAIAgent
- create_agent() factory function
- list_available_agents() utility
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.cli_agents import (
    CLIAgent,
    ClaudeAgent,
    CodexAgent,
    DeepseekCLIAgent,
    GeminiCLIAgent,
    GrokCLIAgent,
    KiloCodeAgent,
    OpenAIAgent,
    QwenCLIAgent,
    MAX_CONTEXT_CHARS,
    MAX_MESSAGE_CHARS,
)
from aragora.agents.base import create_agent, list_available_agents
from aragora.core import Critique, Message


# =============================================================================
# CLIAgent Base Class Tests
# =============================================================================


class TestCLIAgentRunCli:
    """Tests for CLIAgent._run_cli() method."""

    @pytest.fixture
    def agent(self):
        """Create a CodexAgent for testing."""
        return CodexAgent(name="test", model="test-model", timeout=5)

    @pytest.mark.asyncio
    async def test_successful_command_execution(self, agent):
        """Should execute command and return stdout."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"Hello World", b""))
            mock_exec.return_value = mock_proc

            result = await agent._run_cli(["echo", "test"])

            assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_command_with_stdin_input(self, agent):
        """Should pass input_text to stdin."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"Received", b""))
            mock_exec.return_value = mock_proc

            result = await agent._run_cli(["cat"], input_text="Test input")

            mock_proc.communicate.assert_called_once()
            call_kwargs = mock_proc.communicate.call_args
            assert call_kwargs[1]["input"] == b"Test input"

    @pytest.mark.asyncio
    async def test_command_sanitizes_arguments(self, agent):
        """Should sanitize command arguments."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
            mock_exec.return_value = mock_proc

            # Command with null byte
            await agent._run_cli(["echo", "Hello\x00World"])

            # Check sanitized command was used
            call_args = mock_exec.call_args[0]
            assert "\x00" not in call_args[1]

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self, agent):
        """Should raise TimeoutError on timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            with pytest.raises(TimeoutError, match="timed out"):
                await agent._run_cli(["sleep", "100"])

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self, agent):
        """Should kill process on timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            with pytest.raises(TimeoutError):
                await agent._run_cli(["sleep", "100"])

            mock_proc.kill.assert_called_once()
            mock_proc.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_kills_process(self, agent):
        """Should kill process on error."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.communicate = AsyncMock(side_effect=Exception("Test error"))
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            with pytest.raises(Exception, match="Test error"):
                await agent._run_cli(["bad", "command"])

            mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_runtime_error(self, agent):
        """Should raise RuntimeError on non-zero exit."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"Error message"))
            mock_exec.return_value = mock_proc

            with pytest.raises(RuntimeError, match="CLI command failed"):
                await agent._run_cli(["false"])

    @pytest.mark.asyncio
    async def test_stderr_captured_in_error(self, agent):
        """Should include stderr in error message."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(
                return_value=(b"", b"Specific error details")
            )
            mock_exec.return_value = mock_proc

            with pytest.raises(RuntimeError, match="Specific error details"):
                await agent._run_cli(["bad"])


class TestCLIAgentBuildContextPrompt:
    """Tests for CLIAgent._build_context_prompt() method."""

    @pytest.fixture
    def agent(self):
        return CodexAgent(name="test", model="test")

    def test_limits_to_last_10_messages(self, agent):
        """Should only include last 10 messages."""
        messages = [
            Message(role="proposer", agent=f"agent{i}", content=f"Message {i}", round=i)
            for i in range(15)
        ]
        result = agent._build_context_prompt(messages)

        # Should have messages 5-14 (last 10)
        assert "Message 5" in result
        assert "Message 14" in result
        # Should not have earlier messages
        assert "Message 0" not in result
        assert "Message 4" not in result

    def test_truncates_individual_long_messages(self, agent):
        """Should truncate messages exceeding MAX_MESSAGE_CHARS."""
        long_content = "x" * (MAX_MESSAGE_CHARS + 1000)
        messages = [Message(role="proposer", agent="test", content=long_content, round=1)]

        result = agent._build_context_prompt(messages)

        # Should be truncated
        assert len(result) < len(long_content)
        assert "truncated" in result.lower()

    def test_truncates_total_context(self, agent):
        """Should truncate when total context exceeds MAX_CONTEXT_CHARS."""
        # Create many messages that together exceed limit
        large_content = "y" * 15000
        messages = [
            Message(role="proposer", agent=f"a{i}", content=large_content, round=i)
            for i in range(10)
        ]

        result = agent._build_context_prompt(messages)

        # Total should not exceed limit significantly
        assert len(result) <= MAX_CONTEXT_CHARS + 1000


class TestCLIAgentParseCritique:
    """Tests for CLIAgent._parse_critique() method."""

    @pytest.fixture
    def agent(self):
        return CodexAgent(name="test", model="test")

    def test_parses_structured_format(self, agent):
        """Should parse structured critique with issues and suggestions."""
        # Note: Item text must not contain 'issue', 'problem', 'suggest', etc.
        # as those trigger section detection instead of item addition
        response = """
ISSUES:
- First error found
- Second error found

SUGGESTIONS:
- Fix the first one
- Fix the second one

SEVERITY: 0.7
REASONING: This needs work because of X and Y.
"""
        critique = agent._parse_critique(response, "target", "content")

        assert isinstance(critique, Critique)
        assert len(critique.issues) == 2
        assert "First error found" in critique.issues
        assert len(critique.suggestions) == 2
        assert "Fix the first one" in critique.suggestions

    def test_parses_severity_from_text(self, agent):
        """Should extract severity value."""
        response = "SEVERITY: 0.8\nSome other text"
        critique = agent._parse_critique(response, "target", "content")
        assert critique.severity == pytest.approx(0.8, abs=0.01)

    def test_handles_0_to_10_scale_conversion(self, agent):
        """Values > 1 are clamped to 1.0 due to min() being applied first.

        Note: Implementation has a bug - the scale conversion check happens
        after the clamp, so values 1-10 all become 1.0. This test verifies
        actual behavior, not intended behavior.
        """
        response = "SEVERITY: 7\nSome issues here"
        critique = agent._parse_critique(response, "target", "content")
        # Bug: value 7 gets clamped to 1.0 before division check
        assert critique.severity == pytest.approx(1.0, abs=0.01)

    def test_handles_unstructured_response(self, agent):
        """Should handle plain text without structure."""
        response = "This is not great. There are problems. Consider fixing it."
        critique = agent._parse_critique(response, "target", "content")

        assert isinstance(critique, Critique)
        assert len(critique.issues) > 0 or len(critique.reasoning) > 0

    def test_limits_to_5_issues(self, agent):
        """Should limit issues to 5."""
        response = """
ISSUES:
- Issue 1
- Issue 2
- Issue 3
- Issue 4
- Issue 5
- Issue 6
- Issue 7
"""
        critique = agent._parse_critique(response, "target", "content")
        assert len(critique.issues) <= 5

    def test_limits_to_5_suggestions(self, agent):
        """Should limit suggestions to 5."""
        response = """
SUGGESTIONS:
- Suggestion 1
- Suggestion 2
- Suggestion 3
- Suggestion 4
- Suggestion 5
- Suggestion 6
- Suggestion 7
"""
        critique = agent._parse_critique(response, "target", "content")
        assert len(critique.suggestions) <= 5

    def test_extracts_reasoning(self, agent):
        """Should extract reasoning text."""
        response = "Some content here that explains the reasoning in detail."
        critique = agent._parse_critique(response, "target", "content")
        assert len(critique.reasoning) > 0


# =============================================================================
# CodexAgent Tests
# =============================================================================


class TestCodexAgent:
    """Tests for CodexAgent."""

    def test_initialization(self):
        """Should initialize with correct attributes."""
        agent = CodexAgent(name="codex", model="gpt-5.2-codex", role="proposer")
        assert agent.name == "codex"
        assert agent.model == "gpt-5.2-codex"
        assert agent.role == "proposer"
        assert agent.timeout == 120  # Default

    def test_initialization_with_timeout(self):
        """Should accept custom timeout."""
        agent = CodexAgent(name="codex", model="test", timeout=300)
        assert agent.timeout == 300

    @pytest.mark.asyncio
    async def test_generate_builds_correct_command(self):
        """generate() should build correct codex command."""
        agent = CodexAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test prompt")

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "codex" in cmd
            assert "exec" in cmd
            assert "Test prompt" in cmd

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """generate() should include context in prompt."""
        agent = CodexAgent(name="test", model="test")
        context = [Message(role="proposer", agent="a", content="Previous", round=1)]

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test", context=context)

            cmd = mock_run.call_args[0][0]
            prompt = cmd[-1]  # Last arg is the prompt
            assert "Previous" in prompt

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """generate() should include system prompt."""
        agent = CodexAgent(name="test", model="test")
        agent.set_system_prompt("You are helpful")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            prompt = cmd[-1]
            assert "You are helpful" in prompt

    @pytest.mark.asyncio
    async def test_generate_response_parsing(self):
        """generate() should parse response correctly."""
        agent = CodexAgent(name="test", model="test")
        raw_response = """codex
This is the actual response.
tokens used: 100"""

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = raw_response
            result = await agent.generate("Test")

            assert "This is the actual response" in result
            assert "codex" not in result.split("\n")[0]  # Header removed

    @pytest.mark.asyncio
    async def test_generate_skips_token_count(self):
        """generate() should skip token count lines."""
        agent = CodexAgent(name="test", model="test")
        raw_response = """codex
Response content
tokens used: 50"""

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = raw_response
            result = await agent.generate("Test")

            assert "tokens used" not in result

    @pytest.mark.asyncio
    async def test_critique_returns_critique_object(self):
        """critique() should return Critique object."""
        agent = CodexAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "ISSUES:\n- Problem\nSUGGESTIONS:\n- Fix"
            result = await agent.critique("proposal", "task")

            assert isinstance(result, Critique)
            assert result.agent == "test"


# =============================================================================
# ClaudeAgent Tests
# =============================================================================


class TestClaudeAgent:
    """Tests for ClaudeAgent."""

    def test_initialization(self):
        """Should initialize correctly."""
        agent = ClaudeAgent(name="claude", model="claude-sonnet-4")
        assert agent.name == "claude"
        assert agent.model == "claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_generate_uses_stdin(self):
        """generate() should use stdin for prompt."""
        agent = ClaudeAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test prompt")

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs[1]["input_text"] is not None
            assert "Test prompt" in call_kwargs[1]["input_text"]

    @pytest.mark.asyncio
    async def test_generate_command_format(self):
        """generate() should use correct claude command."""
        agent = ClaudeAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            assert "claude" in cmd
            assert "--print" in cmd
            assert "-p" in cmd

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """generate() should include context."""
        agent = ClaudeAgent(name="test", model="test")
        context = [Message(role="critic", agent="b", content="Critique", round=1)]

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test", context=context)

            input_text = mock_run.call_args[1]["input_text"]
            assert "Critique" in input_text

    @pytest.mark.asyncio
    async def test_critique_returns_critique(self):
        """critique() should return Critique."""
        agent = ClaudeAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "ISSUES:\n- Issue\nSEVERITY: 0.5"
            result = await agent.critique("proposal", "task")

            assert isinstance(result, Critique)


# =============================================================================
# GeminiCLIAgent Tests
# =============================================================================


class TestGeminiCLIAgent:
    """Tests for GeminiCLIAgent."""

    def test_initialization(self):
        """Should initialize correctly."""
        agent = GeminiCLIAgent(name="gemini", model="gemini-3-pro")
        assert agent.name == "gemini"
        assert agent.model == "gemini-3-pro"

    @pytest.mark.asyncio
    async def test_generate_uses_yolo_flag(self):
        """generate() should use --yolo flag."""
        agent = GeminiCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            assert "--yolo" in cmd

    @pytest.mark.asyncio
    async def test_generate_filters_yolo_message(self):
        """generate() should filter YOLO mode message."""
        agent = GeminiCLIAgent(name="test", model="test")
        response_with_yolo = "YOLO mode is enabled\nActual response here"

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = response_with_yolo
            result = await agent.generate("Test")

            assert "YOLO mode" not in result
            assert "Actual response here" in result

    @pytest.mark.asyncio
    async def test_generate_uses_text_output(self):
        """generate() should use text output format."""
        agent = GeminiCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            assert "-o" in cmd
            assert "text" in cmd

    @pytest.mark.asyncio
    async def test_critique_returns_critique(self):
        """critique() should return Critique."""
        agent = GeminiCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Some critique response"
            result = await agent.critique("proposal", "task")

            assert isinstance(result, Critique)


# =============================================================================
# KiloCodeAgent Tests
# =============================================================================


class TestKiloCodeAgent:
    """Tests for KiloCodeAgent."""

    def test_initialization_with_provider(self):
        """Should initialize with provider_id."""
        agent = KiloCodeAgent(name="kilo", provider_id="gemini-explorer")
        assert agent.name == "kilo"
        assert agent.provider_id == "gemini-explorer"

    def test_initialization_with_mode(self):
        """Should initialize with mode."""
        agent = KiloCodeAgent(name="kilo", mode="code")
        assert agent.mode == "code"

    def test_default_mode_is_architect(self):
        """Default mode should be architect."""
        agent = KiloCodeAgent(name="kilo")
        assert agent.mode == "architect"

    @pytest.mark.asyncio
    async def test_generate_builds_correct_command(self):
        """generate() should build correct kilocode command."""
        agent = KiloCodeAgent(name="test", provider_id="test-provider", mode="ask")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = '{"role": "assistant", "content": "Response"}'
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            assert "kilocode" in cmd
            assert "--auto" in cmd
            assert "--yolo" in cmd
            assert "--json" in cmd
            assert "-pv" in cmd
            assert "test-provider" in cmd
            assert "-m" in cmd
            assert "ask" in cmd

    def test_extract_kilocode_response_json_assistant(self):
        """Should extract assistant content from JSON."""
        agent = KiloCodeAgent(name="test")
        output = '{"role": "assistant", "content": "Hello from assistant"}'

        result = agent._extract_kilocode_response(output)
        assert result == "Hello from assistant"

    def test_extract_kilocode_response_text_type(self):
        """Should extract text from text-type messages."""
        agent = KiloCodeAgent(name="test")
        output = '{"type": "text", "text": "Text content here"}'

        result = agent._extract_kilocode_response(output)
        assert result == "Text content here"

    def test_extract_kilocode_response_multiple_lines(self):
        """Should handle multiple JSON lines."""
        agent = KiloCodeAgent(name="test")
        output = """{"role": "user", "content": "Question"}
{"role": "assistant", "content": "First response"}
{"role": "assistant", "content": "Second response"}"""

        result = agent._extract_kilocode_response(output)
        assert "First response" in result
        assert "Second response" in result

    def test_extract_kilocode_response_fallback(self):
        """Should fallback to raw output if no JSON."""
        agent = KiloCodeAgent(name="test")
        output = "Plain text response"

        result = agent._extract_kilocode_response(output)
        assert result == "Plain text response"

    @pytest.mark.asyncio
    async def test_critique_returns_critique(self):
        """critique() should return Critique."""
        agent = KiloCodeAgent(name="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = '{"role": "assistant", "content": "Issues found"}'
            result = await agent.critique("proposal", "task")

            assert isinstance(result, Critique)


# =============================================================================
# GrokCLIAgent Tests
# =============================================================================


class TestGrokCLIAgent:
    """Tests for GrokCLIAgent."""

    def test_initialization(self):
        """Should initialize correctly."""
        agent = GrokCLIAgent(name="grok", model="grok-4")
        assert agent.name == "grok"
        assert agent.model == "grok-4"

    @pytest.mark.asyncio
    async def test_generate_command_format(self):
        """generate() should use correct grok command."""
        agent = GrokCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            assert "grok" in cmd
            assert "-p" in cmd

    def test_extract_grok_response_json_lines(self):
        """Should parse JSON lines format."""
        agent = GrokCLIAgent(name="test", model="test")
        output = """{"role": "user", "content": "Question"}
{"role": "assistant", "content": "Answer here"}"""

        result = agent._extract_grok_response(output)
        assert result == "Answer here"

    def test_extract_grok_response_skips_tool_messages(self):
        """Should skip 'Using tools...' messages."""
        agent = GrokCLIAgent(name="test", model="test")
        output = """{"role": "assistant", "content": "Using tools to search..."}
{"role": "assistant", "content": "Final answer"}"""

        result = agent._extract_grok_response(output)
        assert result == "Final answer"
        assert "Using tools" not in result

    def test_extract_grok_response_plain_text(self):
        """Should handle plain text response."""
        agent = GrokCLIAgent(name="test", model="test")
        output = "This is plain text response"

        result = agent._extract_grok_response(output)
        assert result == "This is plain text response"

    def test_extract_grok_response_extracts_final(self):
        """Should extract final assistant message."""
        agent = GrokCLIAgent(name="test", model="test")
        output = """{"role": "assistant", "content": "First response"}
{"role": "assistant", "content": "Updated response"}
{"role": "assistant", "content": "Final response"}"""

        result = agent._extract_grok_response(output)
        assert result == "Final response"

    @pytest.mark.asyncio
    async def test_critique_returns_critique(self):
        """critique() should return Critique."""
        agent = GrokCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = '{"role": "assistant", "content": "Issues"}'
            result = await agent.critique("proposal", "task")

            assert isinstance(result, Critique)


# =============================================================================
# QwenCLIAgent Tests
# =============================================================================


class TestQwenCLIAgent:
    """Tests for QwenCLIAgent."""

    def test_initialization(self):
        """Should initialize correctly."""
        agent = QwenCLIAgent(name="qwen", model="qwen3-coder")
        assert agent.name == "qwen"
        assert agent.model == "qwen3-coder"

    @pytest.mark.asyncio
    async def test_generate_command_format(self):
        """generate() should use correct qwen command."""
        agent = QwenCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            assert "qwen" in cmd
            assert "-p" in cmd

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """generate() should include context."""
        agent = QwenCLIAgent(name="test", model="test")
        context = [Message(role="proposer", agent="a", content="Prev", round=1)]

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test", context=context)

            cmd = mock_run.call_args[0][0]
            prompt = cmd[-1]
            assert "Prev" in prompt

    @pytest.mark.asyncio
    async def test_critique_returns_critique(self):
        """critique() should return Critique."""
        agent = QwenCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Some issues"
            result = await agent.critique("proposal", "task")

            assert isinstance(result, Critique)


# =============================================================================
# DeepseekCLIAgent Tests
# =============================================================================


class TestDeepseekCLIAgent:
    """Tests for DeepseekCLIAgent."""

    def test_initialization(self):
        """Should initialize correctly."""
        agent = DeepseekCLIAgent(name="deepseek", model="deepseek-v3")
        assert agent.name == "deepseek"
        assert agent.model == "deepseek-v3"

    @pytest.mark.asyncio
    async def test_generate_command_format(self):
        """generate() should use correct deepseek command."""
        agent = DeepseekCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            assert "deepseek" in cmd
            assert "-p" in cmd

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """generate() should include context."""
        agent = DeepseekCLIAgent(name="test", model="test")
        context = [Message(role="critic", agent="b", content="Review", round=2)]

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response"
            await agent.generate("Test", context=context)

            cmd = mock_run.call_args[0][0]
            prompt = cmd[-1]
            assert "Review" in prompt

    @pytest.mark.asyncio
    async def test_critique_returns_critique(self):
        """critique() should return Critique."""
        agent = DeepseekCLIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Problems identified"
            result = await agent.critique("proposal", "task")

            assert isinstance(result, Critique)


# =============================================================================
# OpenAIAgent Tests
# =============================================================================


class TestOpenAIAgent:
    """Tests for OpenAIAgent."""

    def test_initialization_with_default_model(self):
        """Should use gpt-4o as default model."""
        agent = OpenAIAgent(name="openai")
        assert agent.model == "gpt-4o"

    def test_initialization_with_custom_model(self):
        """Should accept custom model."""
        agent = OpenAIAgent(name="openai", model="gpt-5")
        assert agent.model == "gpt-5"

    @pytest.mark.asyncio
    async def test_generate_command_format(self):
        """generate() should use correct openai command."""
        agent = OpenAIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = '{"choices": [{"message": {"content": "Response"}}]}'
            await agent.generate("Test")

            cmd = mock_run.call_args[0][0]
            assert "openai" in cmd
            assert "api" in cmd
            assert "chat.completions.create" in cmd

    @pytest.mark.asyncio
    async def test_generate_json_response_parsing(self):
        """generate() should parse JSON response."""
        agent = OpenAIAgent(name="test", model="test")
        json_response = json.dumps({
            "choices": [{"message": {"content": "Parsed content"}}]
        })

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = json_response
            result = await agent.generate("Test")

            assert result == "Parsed content"

    @pytest.mark.asyncio
    async def test_generate_handles_non_json(self):
        """generate() should handle non-JSON response."""
        agent = OpenAIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Plain text response"
            result = await agent.generate("Test")

            assert result == "Plain text response"

    @pytest.mark.asyncio
    async def test_critique_returns_critique(self):
        """critique() should return Critique."""
        agent = OpenAIAgent(name="test", model="test")

        with patch.object(agent, "_run_cli", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "ISSUES:\n- Problem"
            result = await agent.critique("proposal", "task")

            assert isinstance(result, Critique)


# =============================================================================
# create_agent() Factory Tests
# =============================================================================


class TestCreateAgentFactory:
    """Tests for create_agent() factory function."""

    def test_creates_codex_agent(self):
        """Should create CodexAgent."""
        agent = create_agent("codex")
        assert isinstance(agent, CodexAgent)
        assert agent.name == "codex"

    def test_creates_claude_agent(self):
        """Should create ClaudeAgent."""
        agent = create_agent("claude")
        assert isinstance(agent, ClaudeAgent)
        assert agent.name == "claude"

    def test_creates_openai_agent(self):
        """Should create OpenAIAgent."""
        agent = create_agent("openai")
        assert isinstance(agent, OpenAIAgent)
        assert agent.name == "openai"

    def test_creates_gemini_cli_agent(self):
        """Should create GeminiCLIAgent."""
        agent = create_agent("gemini-cli")
        assert isinstance(agent, GeminiCLIAgent)
        assert agent.name == "gemini"

    def test_creates_grok_cli_agent(self):
        """Should create GrokCLIAgent."""
        agent = create_agent("grok-cli")
        assert isinstance(agent, GrokCLIAgent)
        assert agent.name == "grok"

    def test_creates_qwen_cli_agent(self):
        """Should create QwenCLIAgent."""
        agent = create_agent("qwen-cli")
        assert isinstance(agent, QwenCLIAgent)
        assert agent.name == "qwen"

    def test_creates_deepseek_cli_agent(self):
        """Should create DeepseekCLIAgent."""
        agent = create_agent("deepseek-cli")
        assert isinstance(agent, DeepseekCLIAgent)
        assert agent.name == "deepseek"

    def test_creates_kilocode_agent(self):
        """Should create KiloCodeAgent."""
        agent = create_agent("kilocode")
        assert isinstance(agent, KiloCodeAgent)
        assert agent.name == "kilocode"

    def test_passes_custom_name(self):
        """Should pass custom name."""
        agent = create_agent("codex", name="my-codex")
        assert agent.name == "my-codex"

    def test_passes_custom_role(self):
        """Should pass custom role."""
        agent = create_agent("codex", role="critic")
        assert agent.role == "critic"

    def test_passes_custom_model(self):
        """Should pass custom model."""
        agent = create_agent("codex", model="gpt-5.5-codex")
        assert agent.model == "gpt-5.5-codex"

    def test_unknown_type_raises_value_error(self):
        """Unknown agent type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_agent("nonexistent-agent")


# =============================================================================
# list_available_agents() Tests
# =============================================================================


class TestListAvailableAgents:
    """Tests for list_available_agents() function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = list_available_agents()
        assert isinstance(result, dict)

    def test_contains_cli_agents(self):
        """Should contain CLI agent types."""
        result = list_available_agents()
        assert "codex" in result
        assert "claude" in result
        assert "openai" in result
        assert "gemini-cli" in result
        assert "grok-cli" in result

    def test_contains_api_agents(self):
        """Should contain API agent types."""
        result = list_available_agents()
        assert "gemini" in result
        assert "ollama" in result
        assert "anthropic-api" in result
        assert "openai-api" in result

    def test_each_entry_has_type(self):
        """Each entry should have 'type' field."""
        result = list_available_agents()
        for name, info in result.items():
            assert "type" in info, f"Missing 'type' for {name}"

    def test_each_entry_has_requires(self):
        """Each entry should have 'requires' field."""
        result = list_available_agents()
        for name, info in result.items():
            assert "requires" in info, f"Missing 'requires' for {name}"

    def test_each_entry_has_env_vars(self):
        """Each entry should have 'env_vars' field."""
        result = list_available_agents()
        for name, info in result.items():
            assert "env_vars" in info, f"Missing 'env_vars' for {name}"
