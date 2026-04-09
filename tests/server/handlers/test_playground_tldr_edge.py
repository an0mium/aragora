"""Edge-case tests for PlaygroundHandler._synthesize_tldr.

Covers boundary conditions not exercised by test_playground_tldr.py:
- Empty proposals dict
- Multiple proposals with truncation
- Fallback text that ends with period at exact boundary
- Fallback text shorter than 200 chars with no sentence boundary
- AgentError (base class) triggers fallback
- ValueError triggers fallback
- Proposals ordering reflected in prompt
- Fallback text with period at position 0
- Fallback text exactly 200 chars with no sentence boundary
- Fallback text longer than 200 chars with no sentence boundary
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.agents.errors import AgentError
from aragora.server.handlers.playground import PlaygroundHandler


@pytest.fixture()
def handler() -> PlaygroundHandler:
    return PlaygroundHandler()


class TestSynthesizeTldrEdgeCases:
    """Edge-case tests for _synthesize_tldr."""

    def test_empty_proposals_dict(self, handler: PlaygroundHandler) -> None:
        """Empty proposals should still build a valid prompt and call the model."""
        with patch.object(handler, "_call_frontier_model", return_value="Answer.") as mock:
            result = handler._synthesize_tldr(
                question="What is 2+2?",
                proposals={},
            )
        assert result == "Answer."
        # Prompt should contain the question but no agent entries
        prompt_arg = mock.call_args[0][0]
        assert "What is 2+2?" in prompt_arg
        assert "One-sentence answer:" in prompt_arg

    def test_multiple_proposals_all_truncated(self, handler: PlaygroundHandler) -> None:
        """Each proposal is independently truncated to 500 chars."""
        captured = {}

        def capture(prompt: str, timeout: float = 5.0) -> str:
            captured["prompt"] = prompt
            return "Summary."

        with patch.object(handler, "_call_frontier_model", side_effect=capture):
            handler._synthesize_tldr(
                question="Q?",
                proposals={
                    "agent_a": "A" * 800,
                    "agent_b": "B" * 800,
                },
            )
        prompt = captured["prompt"]
        # Each agent's text should be truncated to 500
        assert "A" * 501 not in prompt
        assert "A" * 500 in prompt
        assert "B" * 501 not in prompt
        assert "B" * 500 in prompt

    def test_fallback_text_ends_with_period_space_at_boundary(
        self, handler: PlaygroundHandler
    ) -> None:
        """Fallback text like 'Sentence. ' should return 'Sentence.'"""
        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "x"},
                fallback_text="Sentence. ",
            )
        assert result == "Sentence."

    def test_fallback_text_period_at_position_zero(self, handler: PlaygroundHandler) -> None:
        """A period-space at position 0 means find returns 0, which is not > 0."""
        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "x"},
                fallback_text=". Rest of text here",
            )
        # find(". ") returns 0, condition is `> 0` so it falls through to [:200]
        assert result == ". Rest of text here"

    def test_fallback_text_exactly_200_chars_no_sentence(self, handler: PlaygroundHandler) -> None:
        """Fallback of exactly 200 chars with no '. ' returns all 200 chars."""
        text = "x" * 200
        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "y"},
                fallback_text=text,
            )
        assert result == text
        assert len(result) == 200

    def test_fallback_text_over_200_chars_no_sentence_truncates(
        self, handler: PlaygroundHandler
    ) -> None:
        """Fallback over 200 chars with no '. ' is truncated to 200."""
        text = "x" * 300
        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "y"},
                fallback_text=text,
            )
        assert len(result) == 200

    def test_agent_error_triggers_fallback(self, handler: PlaygroundHandler) -> None:
        """AgentError (base class) should be caught and trigger fallback."""
        with patch.object(handler, "_call_frontier_model", side_effect=AgentError("agent failed")):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "x"},
                fallback_text="Fallback answer. More.",
            )
        assert result == "Fallback answer."

    def test_value_error_triggers_fallback(self, handler: PlaygroundHandler) -> None:
        """ValueError should be caught and trigger fallback."""
        with patch.object(handler, "_call_frontier_model", side_effect=ValueError("bad value")):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "x"},
                fallback_text="Value fallback. Rest.",
            )
        assert result == "Value fallback."

    def test_no_fallback_text_provided_returns_empty(self, handler: PlaygroundHandler) -> None:
        """When fallback_text is omitted (default None) and model fails, return ''."""
        with patch.object(handler, "_call_frontier_model", side_effect=OSError):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "x"},
            )
        assert result == ""

    def test_fallback_with_only_trailing_period(self, handler: PlaygroundHandler) -> None:
        """Fallback text ending in '.' but no '. ' should return first 200 chars."""
        text = "No space after the final period."
        with patch.object(handler, "_call_frontier_model", side_effect=TimeoutError):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "x"},
                fallback_text=text,
            )
        # No ". " in text, so falls through to [:200]
        assert result == text

    def test_fallback_multiple_sentences_returns_first(self, handler: PlaygroundHandler) -> None:
        """With multiple sentences, only the first is returned."""
        with patch.object(handler, "_call_frontier_model", side_effect=RuntimeError):
            result = handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "x"},
                fallback_text="First. Second. Third. Fourth.",
            )
        assert result == "First."

    def test_prompt_contains_question(self, handler: PlaygroundHandler) -> None:
        """The synthesized prompt must include the original question."""
        captured = {}

        def capture(prompt: str, timeout: float = 5.0) -> str:
            captured["prompt"] = prompt
            return "Ok."

        with patch.object(handler, "_call_frontier_model", side_effect=capture):
            handler._synthesize_tldr(
                question="Should I use PostgreSQL or MySQL?",
                proposals={"db_expert": "PostgreSQL for complex queries."},
            )
        assert "Should I use PostgreSQL or MySQL?" in captured["prompt"]

    def test_model_called_with_5s_timeout(self, handler: PlaygroundHandler) -> None:
        """_call_frontier_model is invoked with timeout=5.0."""
        with patch.object(handler, "_call_frontier_model", return_value="Ok.") as mock:
            handler._synthesize_tldr(
                question="Q?",
                proposals={"a": "x"},
            )
        mock.assert_called_once()
        _, kwargs = mock.call_args
        assert kwargs.get("timeout") == 5.0
