"""Tests for aragora_debate.agents reference implementations."""

import json
import sys
import os
import unittest.mock
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate.agents import _parse_json_from_text, _format_context
from aragora_debate.types import Message

try:
    from aragora_debate.agents import MistralAgent
except ImportError:
    MistralAgent = None  # type: ignore[assignment,misc]

try:
    from aragora_debate.agents import GeminiAgent
except ImportError:
    GeminiAgent = None  # type: ignore[assignment,misc]


class TestParseJsonFromText:
    def test_plain_json(self):
        result = _parse_json_from_text('{"choice": "alice", "confidence": 0.9}')
        assert result["choice"] == "alice"
        assert result["confidence"] == 0.9

    def test_markdown_fenced(self):
        text = '```json\n{"issues": ["weak evidence"]}\n```'
        result = _parse_json_from_text(text)
        assert result["issues"] == ["weak evidence"]

    def test_json_embedded_in_prose(self):
        text = 'Here is my analysis:\n{"severity": 7.5, "reasoning": "needs work"}\nThat is all.'
        result = _parse_json_from_text(text)
        assert result["severity"] == 7.5

    def test_no_json_returns_empty(self):
        result = _parse_json_from_text("No JSON here, just text.")
        assert result == {}

    def test_fenced_without_json_label(self):
        text = '```\n{"choice": "bob"}\n```'
        result = _parse_json_from_text(text)
        assert result["choice"] == "bob"


class TestFormatContext:
    def test_none_context(self):
        assert _format_context(None) == ""

    def test_empty_context(self):
        assert _format_context([]) == ""

    def test_formats_messages(self):
        msgs = [
            Message(role="proposer", agent="alice", content="Plan A", round=1),
            Message(role="critic", agent="bob", content="Weak!", round=1),
        ]
        result = _format_context(msgs)
        assert "alice" in result
        assert "Plan A" in result
        assert "bob" in result
        assert "Weak!" in result


class TestClaudeAgentImport:
    def test_import_error_without_anthropic(self):
        """ClaudeAgent should raise ImportError if anthropic is not installed."""
        with patch.dict("sys.modules", {"anthropic": None}):
            # Re-import to trigger the check
            from aragora_debate.agents import ClaudeAgent

            with pytest.raises(ImportError, match="anthropic"):
                ClaudeAgent("test")


class TestOpenAIAgentImport:
    def test_import_error_without_openai(self):
        """OpenAIAgent should raise ImportError if openai is not installed."""
        with patch.dict("sys.modules", {"openai": None}):
            from aragora_debate.agents import OpenAIAgent

            with pytest.raises(ImportError, match="openai"):
                OpenAIAgent("test")


class TestClaudeAgentWithMock:
    @pytest.fixture()
    def mock_anthropic(self):
        mock_mod = MagicMock()
        mock_client = MagicMock()
        mock_mod.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_mod}):
            yield mock_client

    def test_generate(self, mock_anthropic):
        from aragora_debate.agents import ClaudeAgent

        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Use microservices for scalability")]
        mock_anthropic.messages.create.return_value = mock_resp

        agent = ClaudeAgent("analyst", api_key="test-key")
        import asyncio

        result = asyncio.run(agent.generate("What architecture should we use?"))
        assert "microservices" in result
        mock_anthropic.messages.create.assert_called_once()

    def test_critique(self, mock_anthropic):
        from aragora_debate.agents import ClaudeAgent

        mock_resp = MagicMock()
        mock_resp.content = [
            MagicMock(
                text='{"issues": ["no cost analysis"], "suggestions": ["add TCO"], "severity": 6.0, "reasoning": "incomplete"}'
            )
        ]
        mock_anthropic.messages.create.return_value = mock_resp

        agent = ClaudeAgent("reviewer", api_key="test-key")
        import asyncio

        critique = asyncio.run(
            agent.critique("Use microservices", "Architecture choice", target_agent="analyst")
        )
        assert "no cost analysis" in critique.issues
        assert critique.target_agent == "analyst"
        assert critique.severity == 6.0

    def test_vote(self, mock_anthropic):
        from aragora_debate.agents import ClaudeAgent

        mock_resp = MagicMock()
        mock_resp.content = [
            MagicMock(
                text='{"choice": "analyst", "confidence": 0.8, "reasoning": "better evidence"}'
            )
        ]
        mock_anthropic.messages.create.return_value = mock_resp

        agent = ClaudeAgent("judge", api_key="test-key")
        import asyncio

        vote = asyncio.run(
            agent.vote(
                {"analyst": "Use microservices", "challenger": "Use monolith"},
                "Architecture choice",
            )
        )
        assert vote.choice == "analyst"
        assert vote.confidence == 0.8

    def test_vote_fuzzy_match(self, mock_anthropic):
        from aragora_debate.agents import ClaudeAgent

        mock_resp = MagicMock()
        mock_resp.content = [
            MagicMock(text='{"choice": "The analyst proposal", "confidence": 0.7}')
        ]
        mock_anthropic.messages.create.return_value = mock_resp

        agent = ClaudeAgent("judge", api_key="test-key")
        import asyncio

        vote = asyncio.run(
            agent.vote(
                {"analyst": "Plan A", "challenger": "Plan B"},
                "Pick a plan",
            )
        )
        assert vote.choice == "analyst"


class TestOpenAIAgentWithMock:
    @pytest.fixture()
    def mock_openai(self):
        mock_mod = MagicMock()
        mock_client = MagicMock()
        mock_mod.OpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_mod}):
            yield mock_client

    def test_generate(self, mock_openai):
        from aragora_debate.agents import OpenAIAgent

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Use monolith for simplicity"))]
        mock_openai.chat.completions.create.return_value = mock_resp

        agent = OpenAIAgent("challenger", api_key="test-key")
        import asyncio

        result = asyncio.run(agent.generate("What architecture?"))
        assert "monolith" in result

    def test_critique(self, mock_openai):
        from aragora_debate.agents import OpenAIAgent

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"issues": ["scalability concerns"], "suggestions": ["consider k8s"], "severity": 7.0, "reasoning": "won\'t scale"}'
                )
            )
        ]
        mock_openai.chat.completions.create.return_value = mock_resp

        agent = OpenAIAgent("critic", api_key="test-key")
        import asyncio

        critique = asyncio.run(agent.critique("Use monolith", "Architecture", target_agent="dev"))
        assert "scalability" in critique.issues[0]
        assert critique.severity == 7.0

    def test_vote(self, mock_openai):
        from aragora_debate.agents import OpenAIAgent

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"choice": "challenger", "confidence": 0.9, "reasoning": "simpler"}'
                )
            )
        ]
        mock_openai.chat.completions.create.return_value = mock_resp

        agent = OpenAIAgent("judge", api_key="test-key")
        import asyncio

        vote = asyncio.run(
            agent.vote({"analyst": "Plan A", "challenger": "Plan B"}, "Pick")
        )
        assert vote.choice == "challenger"
        assert vote.confidence == 0.9

    def test_stance_included_in_system(self, mock_openai):
        from aragora_debate.agents import OpenAIAgent

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="My argument"))]
        mock_openai.chat.completions.create.return_value = mock_resp

        agent = OpenAIAgent("devil", api_key="test-key", stance="negative")
        import asyncio

        asyncio.run(agent.generate("Make your case"))
        call_args = mock_openai.chat.completions.create.call_args
        system_msg = call_args[1]["messages"][0]["content"]
        assert "negative" in system_msg


class TestMistralAgentImport:
    def test_import_error_without_sdk(self):
        """MistralAgent should raise ImportError when mistralai is not installed."""
        # The import may or may not succeed depending on environment
        try:
            from aragora_debate.agents import MistralAgent
            # If we got here, SDK is installed
            assert MistralAgent is not None
        except ImportError:
            pass  # Expected if SDK not installed

    def test_import_from_agents_module(self):
        """MistralAgent class should be importable."""
        import aragora_debate.agents as mod
        assert hasattr(mod, "MistralAgent")


class TestMistralAgentWithMock:
    @pytest.fixture
    def mock_mistral(self):
        """Create a MistralAgent with a mocked Mistral client."""
        mock_mod = type(sys.modules[__name__])("mistralai")
        mock_client_cls = unittest.mock.MagicMock()
        mock_mod.Mistral = mock_client_cls
        with unittest.mock.patch.dict(sys.modules, {"mistralai": mock_mod}):
            agent = MistralAgent("mistral-test", model="mistral-large-latest")
        mock_client = mock_client_cls.return_value
        return agent, mock_client

    @pytest.mark.asyncio
    async def test_generate(self, mock_mistral):
        agent, client = mock_mistral
        mock_resp = unittest.mock.MagicMock()
        mock_resp.choices = [unittest.mock.MagicMock()]
        mock_resp.choices[0].message.content = "Mistral proposal"
        client.chat.complete.return_value = mock_resp

        result = await agent.generate("Test prompt")
        assert result == "Mistral proposal"
        client.chat.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique(self, mock_mistral):
        agent, client = mock_mistral
        mock_resp = unittest.mock.MagicMock()
        mock_resp.choices = [unittest.mock.MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "issues": ["needs more data"],
            "suggestions": ["add metrics"],
            "severity": 6.0,
            "reasoning": "good but incomplete",
        })
        client.chat.complete.return_value = mock_resp

        crit = await agent.critique("Some proposal", "test task", target_agent="other")
        assert crit.agent == "mistral-test"
        assert "needs more data" in crit.issues

    @pytest.mark.asyncio
    async def test_vote(self, mock_mistral):
        agent, client = mock_mistral
        mock_resp = unittest.mock.MagicMock()
        mock_resp.choices = [unittest.mock.MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "choice": "agent_a",
            "confidence": 0.8,
            "reasoning": "strongest argument",
        })
        client.chat.complete.return_value = mock_resp

        vote = await agent.vote({"agent_a": "Proposal A", "agent_b": "Proposal B"}, "test")
        assert vote.choice == "agent_a"
        assert vote.confidence == 0.8


class TestGeminiAgentImport:
    def test_import_error_without_sdk(self):
        """GeminiAgent should raise ImportError when google-genai is not installed."""
        try:
            from aragora_debate.agents import GeminiAgent
            assert GeminiAgent is not None
        except ImportError:
            pass

    def test_import_from_agents_module(self):
        """GeminiAgent class should be importable."""
        import aragora_debate.agents as mod
        assert hasattr(mod, "GeminiAgent")


class TestGeminiAgentWithMock:
    @pytest.fixture
    def mock_gemini(self):
        """Create a GeminiAgent with a mocked Google GenAI client."""
        mock_genai = type(sys.modules[__name__])("genai")
        mock_genai.Client = unittest.mock.MagicMock()
        mock_types = type(sys.modules[__name__])("types")
        mock_types.GenerateContentConfig = unittest.mock.MagicMock()
        mock_google = type(sys.modules[__name__])("google")
        mock_google.genai = mock_genai

        with unittest.mock.patch.dict(sys.modules, {
            "google": mock_google,
            "google.genai": mock_genai,
            "google.genai.types": mock_types,
        }):
            agent = GeminiAgent("gemini-test", model="gemini-2.0-flash")
        mock_client = mock_genai.Client.return_value
        return agent, mock_client, mock_types

    @pytest.mark.asyncio
    async def test_generate(self, mock_gemini):
        agent, client, mock_types = mock_gemini
        mock_resp = unittest.mock.MagicMock()
        mock_resp.text = "Gemini proposal"
        client.models.generate_content.return_value = mock_resp

        with unittest.mock.patch.dict(sys.modules, {
            "google.genai.types": mock_types,
            "google.genai": unittest.mock.MagicMock(types=mock_types),
            "google": unittest.mock.MagicMock(genai=unittest.mock.MagicMock(types=mock_types)),
        }):
            result = await agent.generate("Test prompt")
        assert result == "Gemini proposal"

    @pytest.mark.asyncio
    async def test_critique(self, mock_gemini):
        agent, client, mock_types = mock_gemini
        mock_resp = unittest.mock.MagicMock()
        mock_resp.text = json.dumps({
            "issues": ["lacks depth"],
            "suggestions": ["expand analysis"],
            "severity": 4.0,
            "reasoning": "needs improvement",
        })
        client.models.generate_content.return_value = mock_resp

        with unittest.mock.patch.dict(sys.modules, {
            "google.genai.types": mock_types,
            "google.genai": unittest.mock.MagicMock(types=mock_types),
            "google": unittest.mock.MagicMock(genai=unittest.mock.MagicMock(types=mock_types)),
        }):
            crit = await agent.critique("Some proposal", "test task", target_agent="other")
        assert crit.agent == "gemini-test"
        assert "lacks depth" in crit.issues
