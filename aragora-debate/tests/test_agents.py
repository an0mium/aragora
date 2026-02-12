"""Tests for aragora_debate.agents reference implementations."""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate.agents import _parse_json_from_text, _format_context
from aragora_debate.types import Message


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
