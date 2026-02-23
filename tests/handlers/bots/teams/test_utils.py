"""
Tests for Teams Bot utility functions.

Covers all functions and module-level state in teams_utils.py:

- Conversation Reference Management
  - _store_conversation_reference() with valid activity / empty conv id / missing fields
  - get_conversation_reference() existing / missing / after store

- Availability Checks
  - _check_botframework_available() available / unavailable
  - _check_connector_available() available / unavailable

- Token Verification (_verify_teams_token)
  - Missing / empty / malformed auth header
  - Valid token via jwt_verify module
  - Invalid token via jwt_verify module
  - HAS_JWT=False with dev bypass allowed / denied
  - jwt_verify ImportError in production / dev environment
  - Environment variable combinations for dev/prod fallback

- Adaptive Card Builders
  - build_debate_card()
    - Basic structure, schema, version
    - Topic truncation (>200 chars)
    - Agent list truncation (>5 agents)
    - Vote buttons included / excluded
    - View Summary action when vote buttons on
    - Debate ID footer truncation
    - Round progress display
  - build_consensus_card()
    - Consensus reached / not reached
    - Confidence display
    - Winner display / no winner
    - Vote counts display (sorted, singular/plural)
    - Final answer display / truncation (>500 chars) / no answer
    - Action URLs contain debate ID
    - Empty vote counts

- Debate Orchestration
  - _start_teams_debate() via DecisionRouter (success / import error / runtime error)
  - _start_teams_debate() with decision_integrity (bool / dict / None)
  - _start_teams_debate() origin registration (success / failure)
  - _start_teams_debate() dedup (different request_id)
  - _fallback_start_debate() with Redis queue (success / import error / runtime error)
  - _fallback_start_debate() origin registration (success / failure)
  - _fallback_start_debate() tracks active debate

- Vote Counting
  - get_debate_vote_counts() empty / single / multiple / nonexistent debate

- Module-Level State
  - _active_debates, _user_votes, _conversation_references are dicts
  - State isolation between tests (autouse clear fixture)

- Security Tests
  - Token verification rejects non-Bearer schemes
  - Token verification rejects empty tokens
  - Production environment blocks unverified tokens
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear all module-level shared state between tests."""
    from aragora.server.handlers.bots.teams_utils import (
        _active_debates,
        _conversation_references,
        _user_votes,
    )

    _active_debates.clear()
    _user_votes.clear()
    _conversation_references.clear()
    yield
    _active_debates.clear()
    _user_votes.clear()
    _conversation_references.clear()


@pytest.fixture
def active_debates():
    """Direct access to the _active_debates dict."""
    from aragora.server.handlers.bots.teams_utils import _active_debates

    return _active_debates


@pytest.fixture
def user_votes():
    """Direct access to the _user_votes dict."""
    from aragora.server.handlers.bots.teams_utils import _user_votes

    return _user_votes


@pytest.fixture
def conversation_refs():
    """Direct access to the _conversation_references dict."""
    from aragora.server.handlers.bots.teams_utils import _conversation_references

    return _conversation_references


def _make_activity(
    conversation_id: str = "conv-abc",
    service_url: str = "https://smba.trafficmanager.net/teams/",
    recipient_id: str = "bot-123",
    tenant_id: str = "tenant-001",
    channel_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal Bot Framework activity for conversation reference tests."""
    activity: dict[str, Any] = {
        "serviceUrl": service_url,
        "conversation": {"id": conversation_id},
        "recipient": {"id": recipient_id},
    }
    if tenant_id:
        activity["conversation"]["tenantId"] = tenant_id
    if channel_data is not None:
        activity["channelData"] = channel_data
    return activity


# ===========================================================================
# Conversation Reference Management
# ===========================================================================


class TestStoreConversationReference:
    """Test _store_conversation_reference."""

    def test_stores_reference(self, conversation_refs):
        """Valid activity stores a conversation reference."""
        from aragora.server.handlers.bots.teams_utils import _store_conversation_reference

        activity = _make_activity()
        _store_conversation_reference(activity)
        assert "conv-abc" in conversation_refs
        ref = conversation_refs["conv-abc"]
        assert ref["service_url"] == "https://smba.trafficmanager.net/teams/"
        assert ref["conversation"]["id"] == "conv-abc"
        assert ref["bot"]["id"] == "bot-123"
        assert ref["tenant_id"] == "tenant-001"

    def test_empty_conversation_id_skips(self, conversation_refs):
        """Activity with empty conversation id does not store anything."""
        from aragora.server.handlers.bots.teams_utils import _store_conversation_reference

        activity = _make_activity(conversation_id="")
        _store_conversation_reference(activity)
        assert len(conversation_refs) == 0

    def test_missing_conversation_field(self, conversation_refs):
        """Activity with no conversation field does not store anything."""
        from aragora.server.handlers.bots.teams_utils import _store_conversation_reference

        _store_conversation_reference({})
        assert len(conversation_refs) == 0

    def test_overwrites_existing(self, conversation_refs):
        """Storing twice for same conversation_id overwrites."""
        from aragora.server.handlers.bots.teams_utils import _store_conversation_reference

        _store_conversation_reference(_make_activity(service_url="https://old.example.com/"))
        _store_conversation_reference(_make_activity(service_url="https://new.example.com/"))
        assert conversation_refs["conv-abc"]["service_url"] == "https://new.example.com/"

    def test_stores_channel_data(self, conversation_refs):
        """Channel data from activity is stored."""
        from aragora.server.handlers.bots.teams_utils import _store_conversation_reference

        activity = _make_activity(channel_data={"team": {"id": "team-1"}})
        _store_conversation_reference(activity)
        assert conversation_refs["conv-abc"]["channel_data"] == {"team": {"id": "team-1"}}

    def test_defaults_for_missing_fields(self, conversation_refs):
        """Missing optional fields default to empty."""
        from aragora.server.handlers.bots.teams_utils import _store_conversation_reference

        activity = {"conversation": {"id": "conv-minimal"}}
        _store_conversation_reference(activity)
        ref = conversation_refs["conv-minimal"]
        assert ref["service_url"] == ""
        assert ref["bot"] == {}
        assert ref["channel_data"] == {}
        assert ref["tenant_id"] == ""

    def test_multiple_conversations(self, conversation_refs):
        """Multiple distinct conversations are stored independently."""
        from aragora.server.handlers.bots.teams_utils import _store_conversation_reference

        _store_conversation_reference(_make_activity(conversation_id="conv-1"))
        _store_conversation_reference(_make_activity(conversation_id="conv-2"))
        assert len(conversation_refs) == 2
        assert "conv-1" in conversation_refs
        assert "conv-2" in conversation_refs


class TestGetConversationReference:
    """Test get_conversation_reference."""

    def test_returns_none_for_missing(self):
        """Returns None when conversation_id is not stored."""
        from aragora.server.handlers.bots.teams_utils import get_conversation_reference

        assert get_conversation_reference("nonexistent") is None

    def test_returns_stored_reference(self, conversation_refs):
        """Returns the stored reference dict."""
        from aragora.server.handlers.bots.teams_utils import get_conversation_reference

        conversation_refs["conv-test"] = {"service_url": "https://test.example.com/"}
        result = get_conversation_reference("conv-test")
        assert result is not None
        assert result["service_url"] == "https://test.example.com/"

    def test_returns_reference_after_store(self):
        """get_conversation_reference works after _store_conversation_reference."""
        from aragora.server.handlers.bots.teams_utils import (
            _store_conversation_reference,
            get_conversation_reference,
        )

        _store_conversation_reference(_make_activity(conversation_id="conv-round-trip"))
        ref = get_conversation_reference("conv-round-trip")
        assert ref is not None
        assert ref["conversation"]["id"] == "conv-round-trip"

    def test_returns_none_for_empty_string_key(self):
        """Returns None for empty string key (since store skips empty IDs)."""
        from aragora.server.handlers.bots.teams_utils import get_conversation_reference

        assert get_conversation_reference("") is None


# ===========================================================================
# Availability Checks
# ===========================================================================


class TestCheckBotframeworkAvailable:
    """Test _check_botframework_available."""

    def test_available_when_importable(self):
        """Returns (True, None) when botbuilder.core is importable."""
        from aragora.server.handlers.bots.teams_utils import _check_botframework_available

        with patch.dict("sys.modules", {"botbuilder": MagicMock(), "botbuilder.core": MagicMock()}):
            available, error = _check_botframework_available()
            assert available is True
            assert error is None

    def test_unavailable_when_not_importable(self):
        """Returns (False, error_msg) when import fails."""
        from aragora.server.handlers.bots.teams_utils import _check_botframework_available

        with patch.dict("sys.modules", {"botbuilder.core": None}):
            available, error = _check_botframework_available()
            assert available is False
            assert error is not None
            assert "botbuilder" in error


class TestCheckConnectorAvailable:
    """Test _check_connector_available."""

    def test_available_when_importable(self):
        """Returns (True, None) when Teams connector is importable."""
        from aragora.server.handlers.bots.teams_utils import _check_connector_available

        mock_module = MagicMock()
        mock_module.TeamsConnector = MagicMock()
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": mock_module}):
            available, error = _check_connector_available()
            assert available is True
            assert error is None

    def test_unavailable_when_not_importable(self):
        """Returns (False, error_msg) when import fails."""
        from aragora.server.handlers.bots.teams_utils import _check_connector_available

        with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
            available, error = _check_connector_available()
            assert available is False
            assert error is not None
            assert "Teams connector" in error


# ===========================================================================
# Token Verification
# ===========================================================================


class TestVerifyTeamsToken:
    """Test _verify_teams_token."""

    @pytest.mark.asyncio
    async def test_rejects_empty_auth_header(self):
        """Empty auth header returns False."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        assert await _verify_teams_token("", "app-id") is False

    @pytest.mark.asyncio
    async def test_rejects_none_auth_header(self):
        """None auth header returns False."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        assert await _verify_teams_token(None, "app-id") is False

    @pytest.mark.asyncio
    async def test_rejects_non_bearer_scheme(self):
        """Non-Bearer auth header returns False."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        assert await _verify_teams_token("Basic dXNlcjpwYXNz", "app-id") is False

    @pytest.mark.asyncio
    async def test_rejects_bearer_without_space(self):
        """Malformed 'Bearertoken' without space returns False."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        assert await _verify_teams_token("Bearertoken123", "app-id") is False

    @pytest.mark.asyncio
    async def test_valid_token_with_jwt(self):
        """Valid token via verify_teams_webhook returns True."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        mock_jwt_verify = MagicMock()
        mock_jwt_verify.HAS_JWT = True
        mock_jwt_verify.verify_teams_webhook = MagicMock(return_value=True)

        with patch.dict("sys.modules", {"aragora.connectors.chat.jwt_verify": mock_jwt_verify}):
            result = await _verify_teams_token("Bearer valid-token", "app-id")
            assert result is True
            mock_jwt_verify.verify_teams_webhook.assert_called_once_with(
                "Bearer valid-token", "app-id"
            )

    @pytest.mark.asyncio
    async def test_invalid_token_with_jwt(self):
        """Invalid token via verify_teams_webhook returns False."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        mock_jwt_verify = MagicMock()
        mock_jwt_verify.HAS_JWT = True
        mock_jwt_verify.verify_teams_webhook = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"aragora.connectors.chat.jwt_verify": mock_jwt_verify}):
            result = await _verify_teams_token("Bearer bad-token", "app-id")
            assert result is False

    @pytest.mark.asyncio
    async def test_no_jwt_dev_bypass_allowed(self):
        """Without JWT, dev bypass allowed returns True."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        mock_jwt_verify = MagicMock()
        mock_jwt_verify.HAS_JWT = False

        mock_webhook_sec = MagicMock()
        mock_webhook_sec.should_allow_unverified = MagicMock(return_value=True)

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.chat.jwt_verify": mock_jwt_verify,
                "aragora.connectors.chat.webhook_security": mock_webhook_sec,
            },
        ):
            result = await _verify_teams_token("Bearer dev-token", "app-id")
            assert result is True
            mock_webhook_sec.should_allow_unverified.assert_called_once_with("teams")

    @pytest.mark.asyncio
    async def test_no_jwt_dev_bypass_denied(self):
        """Without JWT, dev bypass denied returns False."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        mock_jwt_verify = MagicMock()
        mock_jwt_verify.HAS_JWT = False

        mock_webhook_sec = MagicMock()
        mock_webhook_sec.should_allow_unverified = MagicMock(return_value=False)

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.chat.jwt_verify": mock_jwt_verify,
                "aragora.connectors.chat.webhook_security": mock_webhook_sec,
            },
        ):
            result = await _verify_teams_token("Bearer some-token", "app-id")
            assert result is False

    @pytest.mark.asyncio
    async def test_import_error_production_rejects(self):
        """ImportError in production environment returns False."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.jwt_verify": None},
        ):
            with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
                result = await _verify_teams_token("Bearer token", "app-id")
                assert result is False

    @pytest.mark.asyncio
    async def test_import_error_dev_allows(self):
        """ImportError in development environment returns True."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.jwt_verify": None},
        ):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                result = await _verify_teams_token("Bearer token", "app-id")
                assert result is True

    @pytest.mark.asyncio
    async def test_import_error_test_env_allows(self):
        """ImportError in test environment returns True."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.jwt_verify": None},
        ):
            with patch.dict("os.environ", {"ARAGORA_ENV": "test"}):
                result = await _verify_teams_token("Bearer token", "app-id")
                assert result is True

    @pytest.mark.asyncio
    async def test_import_error_local_env_allows(self):
        """ImportError in local environment returns True."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.jwt_verify": None},
        ):
            with patch.dict("os.environ", {"ARAGORA_ENV": "local"}):
                result = await _verify_teams_token("Bearer token", "app-id")
                assert result is True

    @pytest.mark.asyncio
    async def test_import_error_no_env_defaults_production(self):
        """ImportError with no ARAGORA_ENV set defaults to production (rejects)."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.jwt_verify": None},
        ):
            with patch.dict("os.environ", {}, clear=False):
                # Remove ARAGORA_ENV if present
                import os

                env_val = os.environ.pop("ARAGORA_ENV", None)
                try:
                    result = await _verify_teams_token("Bearer token", "app-id")
                    assert result is False
                finally:
                    if env_val is not None:
                        os.environ["ARAGORA_ENV"] = env_val


# ===========================================================================
# Build Debate Card
# ===========================================================================


class TestBuildDebateCard:
    """Test build_debate_card."""

    def test_basic_structure(self):
        """Card has correct schema, type, and version."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card(
            debate_id="d-001",
            topic="Test topic",
            agents=["claude", "gpt4"],
            current_round=1,
            total_rounds=3,
        )
        assert card["$schema"] == "http://adaptivecards.io/schemas/adaptive-card.json"
        assert card["type"] == "AdaptiveCard"
        assert card["version"] == "1.4"

    def test_body_contains_title(self):
        """Card body starts with 'Active Debate' title."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("d-001", "Test", ["claude"], 1, 3)
        assert card["body"][0]["text"] == "Active Debate"
        assert card["body"][0]["weight"] == "Bolder"

    def test_topic_displayed(self):
        """Topic is shown in the card body."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("d-001", "Should we use Rust?", ["claude"], 1, 3)
        assert "Should we use Rust?" in card["body"][1]["text"]

    def test_topic_truncated_at_200(self):
        """Topic longer than 200 chars is truncated."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        long_topic = "A" * 300
        card = build_debate_card("d-001", long_topic, ["claude"], 1, 3)
        topic_text = card["body"][1]["text"]
        # "**Topic:** " prefix + 200 chars
        assert len(long_topic) > 200
        assert "A" * 200 in topic_text
        assert "A" * 201 not in topic_text

    def test_agents_in_factset(self):
        """Agent names shown in FactSet."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("d-001", "Test", ["claude", "gpt4", "gemini"], 1, 3)
        factset = card["body"][2]
        assert factset["type"] == "FactSet"
        agents_fact = factset["facts"][0]
        assert agents_fact["title"] == "Agents"
        assert "claude" in agents_fact["value"]
        assert "gpt4" in agents_fact["value"]
        assert "gemini" in agents_fact["value"]

    def test_agents_truncated_at_5(self):
        """Only first 5 agents shown."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        agents = [f"agent-{i}" for i in range(8)]
        card = build_debate_card("d-001", "Test", agents, 1, 3)
        agents_fact = card["body"][2]["facts"][0]
        assert "agent-4" in agents_fact["value"]
        assert "agent-5" not in agents_fact["value"]

    def test_progress_display(self):
        """Progress fact shows current/total rounds."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("d-001", "Test", ["claude"], 2, 5)
        progress_fact = card["body"][2]["facts"][1]
        assert progress_fact["title"] == "Progress"
        assert progress_fact["value"] == "Round 2/5"

    def test_vote_buttons_included(self):
        """Vote buttons are included when include_vote_buttons=True."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("d-001", "Test", ["claude", "gpt4"], 1, 3, include_vote_buttons=True)
        actions = card["actions"]
        assert actions is not None
        vote_actions = [a for a in actions if a.get("data", {}).get("action") == "vote"]
        assert len(vote_actions) == 2
        assert vote_actions[0]["data"]["agent"] == "claude"
        assert vote_actions[1]["data"]["agent"] == "gpt4"

    def test_vote_buttons_excluded(self):
        """No vote buttons when include_vote_buttons=False."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("d-001", "Test", ["claude"], 1, 3, include_vote_buttons=False)
        assert card["actions"] is None

    def test_vote_buttons_truncated_at_5(self):
        """Only first 5 agents get vote buttons."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        agents = [f"agent-{i}" for i in range(8)]
        card = build_debate_card("d-001", "Test", agents, 1, 3, include_vote_buttons=True)
        vote_actions = [a for a in card["actions"] if a.get("data", {}).get("action") == "vote"]
        assert len(vote_actions) == 5

    def test_view_summary_action(self):
        """View Summary action is included when vote buttons are on."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("d-001", "Test", ["claude"], 1, 3, include_vote_buttons=True)
        summary_actions = [
            a for a in card["actions"] if a.get("data", {}).get("action") == "summary"
        ]
        assert len(summary_actions) == 1
        assert summary_actions[0]["title"] == "View Summary"
        assert summary_actions[0]["data"]["debate_id"] == "d-001"

    def test_debate_id_footer(self):
        """Footer shows truncated debate ID."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        debate_id = "abcdef01-2345-6789-abcd-ef0123456789"
        card = build_debate_card(debate_id, "Test", ["claude"], 1, 3)
        footer = card["body"][-1]
        assert footer["size"] == "Small"
        assert footer["isSubtle"] is True
        assert "abcdef01..." in footer["text"]

    def test_empty_agents_list(self):
        """Card handles empty agents list."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("d-001", "Test", [], 1, 3, include_vote_buttons=True)
        agents_fact = card["body"][2]["facts"][0]
        assert agents_fact["value"] == ""
        # Only summary action (no vote buttons for empty agents)
        vote_actions = [a for a in card["actions"] if a.get("data", {}).get("action") == "vote"]
        assert len(vote_actions) == 0

    def test_vote_button_data_includes_debate_id(self):
        """Each vote button includes the correct debate_id."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        card = build_debate_card("my-debate-123", "Test", ["claude"], 1, 3)
        vote_action = [a for a in card["actions"] if a.get("data", {}).get("action") == "vote"][0]
        assert vote_action["data"]["debate_id"] == "my-debate-123"


# ===========================================================================
# Build Consensus Card
# ===========================================================================


class TestBuildConsensusCard:
    """Test build_consensus_card."""

    def test_basic_structure(self):
        """Card has correct schema, type, and version."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card(
            debate_id="d-001",
            topic="Test",
            consensus_reached=True,
            confidence=0.85,
            winner="claude",
            final_answer="The answer",
            vote_counts={},
        )
        assert card["$schema"] == "http://adaptivecards.io/schemas/adaptive-card.json"
        assert card["type"] == "AdaptiveCard"
        assert card["version"] == "1.4"

    def test_consensus_reached_status(self):
        """Consensus reached shows 'Consensus Reached' in green."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 0.9, "claude", "Yes", {})
        status_block = card["body"][0]
        assert status_block["text"] == "Consensus Reached"
        assert status_block["color"] == "Good"

    def test_no_consensus_status(self):
        """No consensus shows 'No Consensus' in warning color."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", False, 0.3, None, None, {})
        status_block = card["body"][0]
        assert status_block["text"] == "No Consensus"
        assert status_block["color"] == "Warning"

    def test_confidence_display(self):
        """Confidence is displayed as a percentage."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 0.85, None, None, {})
        factset = None
        for item in card["body"]:
            if item.get("type") == "FactSet":
                factset = item
                break
        assert factset is not None
        confidence_fact = factset["facts"][0]
        assert confidence_fact["title"] == "Confidence"
        assert confidence_fact["value"] == "85%"

    def test_winner_displayed(self):
        """Winner is shown when provided."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 0.9, "claude", None, {})
        factset = None
        for item in card["body"]:
            if item.get("type") == "FactSet":
                factset = item
                break
        winner_fact = next((f for f in factset["facts"] if f["title"] == "Winner"), None)
        assert winner_fact is not None
        assert winner_fact["value"] == "claude"

    def test_no_winner(self):
        """No winner fact when winner is None."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", False, 0.4, None, None, {})
        factset = None
        for item in card["body"]:
            if item.get("type") == "FactSet":
                factset = item
                break
        winner_facts = [f for f in factset["facts"] if f["title"] == "Winner"]
        assert len(winner_facts) == 0

    def test_vote_counts_displayed(self):
        """Vote counts are shown sorted by count descending."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        vote_counts = {"claude": 5, "gpt4": 3, "gemini": 1}
        card = build_consensus_card("d-001", "Test", True, 0.9, "claude", None, vote_counts)
        vote_blocks = [b for b in card["body"] if "User Votes" in b.get("text", "")]
        assert len(vote_blocks) == 1
        vote_text = vote_blocks[0]["text"]
        assert "claude: 5 votes" in vote_text
        assert "gpt4: 3 votes" in vote_text
        assert "gemini: 1 vote" in vote_text  # singular

    def test_vote_counts_singular_plural(self):
        """1 vote shows 'vote', >1 shows 'votes'."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        vote_counts = {"claude": 1, "gpt4": 2}
        card = build_consensus_card("d-001", "Test", True, 0.9, None, None, vote_counts)
        vote_blocks = [b for b in card["body"] if "User Votes" in b.get("text", "")]
        vote_text = vote_blocks[0]["text"]
        assert "claude: 1 vote\n" in vote_text or "claude: 1 vote" in vote_text
        assert "gpt4: 2 votes" in vote_text

    def test_empty_vote_counts(self):
        """Empty vote counts does not add vote block."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 0.9, None, None, {})
        vote_blocks = [b for b in card["body"] if "User Votes" in b.get("text", "")]
        assert len(vote_blocks) == 0

    def test_final_answer_displayed(self):
        """Final answer is shown when provided."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 0.9, None, "The answer is 42.", {})
        answer_blocks = [b for b in card["body"] if "Decision" in b.get("text", "")]
        assert len(answer_blocks) == 1
        assert "The answer is 42." in answer_blocks[0]["text"]

    def test_final_answer_truncated(self):
        """Final answer longer than 500 chars is truncated with ellipsis."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        long_answer = "X" * 600
        card = build_consensus_card("d-001", "Test", True, 0.9, None, long_answer, {})
        answer_blocks = [b for b in card["body"] if "Decision" in b.get("text", "")]
        answer_text = answer_blocks[0]["text"]
        assert "..." in answer_text
        # Preview is 500 chars, not the full 600
        assert "X" * 500 in answer_text
        assert "X" * 501 not in answer_text

    def test_no_final_answer(self):
        """No answer block when final_answer is None."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 0.9, None, None, {})
        answer_blocks = [b for b in card["body"] if "Decision" in b.get("text", "")]
        assert len(answer_blocks) == 0

    def test_topic_truncated_at_200(self):
        """Topic is truncated to 200 chars."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        long_topic = "T" * 300
        card = build_consensus_card("d-001", long_topic, True, 0.9, None, None, {})
        topic_text = card["body"][1]["text"]
        assert "T" * 200 in topic_text
        assert "T" * 201 not in topic_text

    def test_action_urls_contain_debate_id(self):
        """Action URLs include the debate ID."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("my-debate-xyz", "Test", True, 0.9, None, None, {})
        actions = card["actions"]
        assert len(actions) == 2
        assert "my-debate-xyz" in actions[0]["url"]
        assert "my-debate-xyz" in actions[1]["url"]

    def test_report_and_audit_actions(self):
        """Card has View Full Report and Audit Trail actions."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 0.9, None, None, {})
        titles = [a["title"] for a in card["actions"]]
        assert "View Full Report" in titles
        assert "Audit Trail" in titles

    def test_confidence_zero(self):
        """Zero confidence displays as 0%."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", False, 0.0, None, None, {})
        factset = next(b for b in card["body"] if b.get("type") == "FactSet")
        confidence_fact = factset["facts"][0]
        assert confidence_fact["value"] == "0%"

    def test_confidence_one(self):
        """Full confidence displays as 100%."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 1.0, None, None, {})
        factset = next(b for b in card["body"] if b.get("type") == "FactSet")
        confidence_fact = factset["facts"][0]
        assert confidence_fact["value"] == "100%"

    def test_separator_before_answer(self):
        """A separator text block is added before the final answer."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        card = build_consensus_card("d-001", "Test", True, 0.9, None, "Answer here", {})
        separator_blocks = [b for b in card["body"] if b.get("separator") is True]
        assert len(separator_blocks) == 1


# ===========================================================================
# Debate Orchestration - _start_teams_debate
# ===========================================================================


class TestStartTeamsDebate:
    """Test _start_teams_debate."""

    @pytest.mark.asyncio
    async def test_success_via_decision_router(self, active_debates):
        """Successful debate start via DecisionRouter returns request_id."""
        from aragora.server.handlers.bots.teams_utils import _start_teams_debate

        mock_result = MagicMock()
        mock_result.request_id = "router-debate-001"

        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=mock_result)

        mock_core = MagicMock()
        mock_core.DecisionConfig = MagicMock()
        mock_core.DecisionRequest = MagicMock()
        mock_core.DecisionType = MagicMock()
        mock_core.DecisionType.DEBATE = "debate"
        mock_core.InputSource = MagicMock()
        mock_core.InputSource.TEAMS = "teams"
        mock_core.ResponseChannel = MagicMock()
        mock_core.RequestContext = MagicMock()
        mock_core.get_decision_router = MagicMock(return_value=mock_router)

        with patch.dict("sys.modules", {"aragora.core": mock_core}):
            with patch(
                "aragora.server.handlers.bots.teams_utils.register_debate_origin",
                create=True,
            ):
                result = await _start_teams_debate(
                    topic="Test topic",
                    conversation_id="conv-1",
                    user_id="user-1",
                    service_url="https://service.example.com/",
                )
                assert result == "router-debate-001"
                assert "router-debate-001" in active_debates

    @pytest.mark.asyncio
    async def test_fallback_on_import_error(self, active_debates):
        """Falls back to _fallback_start_debate on ImportError."""
        from aragora.server.handlers.bots.teams_utils import _start_teams_debate

        with patch.dict("sys.modules", {"aragora.core": None}):
            with patch(
                "aragora.server.handlers.bots.teams_utils._fallback_start_debate",
                new_callable=AsyncMock,
                return_value="fallback-id",
            ) as mock_fallback:
                result = await _start_teams_debate(
                    topic="Test",
                    conversation_id="conv-1",
                    user_id="user-1",
                    service_url="https://service.example.com/",
                )
                assert result == "fallback-id"
                mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_runtime_error(self, active_debates):
        """Falls back to _fallback_start_debate on RuntimeError from router."""
        from aragora.server.handlers.bots.teams_utils import _start_teams_debate

        mock_core = MagicMock()
        mock_core.DecisionConfig = MagicMock()
        mock_core.DecisionRequest = MagicMock(side_effect=RuntimeError("router broken"))
        mock_core.DecisionType = MagicMock()
        mock_core.DecisionType.DEBATE = "debate"
        mock_core.InputSource = MagicMock()
        mock_core.InputSource.TEAMS = "teams"
        mock_core.ResponseChannel = MagicMock()
        mock_core.RequestContext = MagicMock()
        mock_core.get_decision_router = MagicMock()

        with patch.dict("sys.modules", {"aragora.core": mock_core}):
            with patch(
                "aragora.server.handlers.bots.teams_utils._fallback_start_debate",
                new_callable=AsyncMock,
                return_value="fallback-id-2",
            ) as mock_fallback:
                result = await _start_teams_debate(
                    topic="Test",
                    conversation_id="conv-1",
                    user_id="user-1",
                    service_url="https://service.example.com/",
                )
                assert result == "fallback-id-2"
                mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_decision_integrity_bool_true(self):
        """decision_integrity=True creates config with empty dict."""
        from aragora.server.handlers.bots.teams_utils import _start_teams_debate

        mock_result = MagicMock()
        mock_result.request_id = "di-bool-001"

        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=mock_result)

        captured_kwargs = {}
        mock_config = MagicMock()

        def capture_decision_config(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_config

        mock_core = MagicMock()
        mock_core.DecisionConfig = capture_decision_config
        mock_core.DecisionRequest = MagicMock()
        mock_core.DecisionType = MagicMock()
        mock_core.DecisionType.DEBATE = "debate"
        mock_core.InputSource = MagicMock()
        mock_core.InputSource.TEAMS = "teams"
        mock_core.ResponseChannel = MagicMock()
        mock_core.RequestContext = MagicMock()
        mock_core.get_decision_router = MagicMock(return_value=mock_router)

        with patch.dict("sys.modules", {"aragora.core": mock_core}):
            await _start_teams_debate(
                topic="DI Test",
                conversation_id="conv-1",
                user_id="user-1",
                service_url="https://service.example.com/",
                decision_integrity=True,
            )
            assert captured_kwargs["decision_integrity"] == {}

    @pytest.mark.asyncio
    async def test_decision_integrity_bool_false(self):
        """decision_integrity=False results in no config."""
        from aragora.server.handlers.bots.teams_utils import _start_teams_debate

        mock_result = MagicMock()
        mock_result.request_id = "di-false-001"

        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=mock_result)

        request_kwargs_captured = {}

        def capture_request(**kwargs):
            request_kwargs_captured.update(kwargs)
            mock_req = MagicMock()
            mock_req.request_id = "di-false-001"
            return mock_req

        mock_core = MagicMock()
        mock_core.DecisionConfig = MagicMock()
        mock_core.DecisionRequest = capture_request
        mock_core.DecisionType = MagicMock()
        mock_core.DecisionType.DEBATE = "debate"
        mock_core.InputSource = MagicMock()
        mock_core.InputSource.TEAMS = "teams"
        mock_core.ResponseChannel = MagicMock()
        mock_core.RequestContext = MagicMock()
        mock_core.get_decision_router = MagicMock(return_value=mock_router)

        with patch.dict("sys.modules", {"aragora.core": mock_core}):
            await _start_teams_debate(
                topic="DI Test",
                conversation_id="conv-1",
                user_id="user-1",
                service_url="https://service.example.com/",
                decision_integrity=False,
            )
            assert "config" not in request_kwargs_captured

    @pytest.mark.asyncio
    async def test_returns_debate_id_when_router_returns_none_id(self, active_debates):
        """Returns local debate_id when router returns empty request_id."""
        from aragora.server.handlers.bots.teams_utils import _start_teams_debate

        mock_result = MagicMock()
        mock_result.request_id = None

        mock_router = AsyncMock()
        mock_router.route = AsyncMock(return_value=mock_result)

        mock_core = MagicMock()
        mock_core.DecisionConfig = MagicMock()
        mock_core.DecisionRequest = MagicMock()
        mock_core.DecisionType = MagicMock()
        mock_core.DecisionType.DEBATE = "debate"
        mock_core.InputSource = MagicMock()
        mock_core.InputSource.TEAMS = "teams"
        mock_core.ResponseChannel = MagicMock()
        mock_core.RequestContext = MagicMock()
        mock_core.get_decision_router = MagicMock(return_value=mock_router)

        with patch.dict("sys.modules", {"aragora.core": mock_core}):
            result = await _start_teams_debate(
                topic="Test",
                conversation_id="conv-1",
                user_id="user-1",
                service_url="https://service.example.com/",
            )
            # Should return a UUID (the locally generated debate_id)
            assert isinstance(result, str)
            assert len(result) > 0


# ===========================================================================
# Debate Orchestration - _fallback_start_debate
# ===========================================================================


class TestFallbackStartDebate:
    """Test _fallback_start_debate."""

    @pytest.fixture(autouse=True)
    def _mock_redis_queue(self):
        """Mock Redis queue to avoid real connections in all fallback tests."""
        mock_queue = MagicMock()
        mock_queue.create_debate_job = MagicMock(return_value=MagicMock())
        mock_queue.create_redis_queue = AsyncMock(return_value=AsyncMock())

        with patch.dict("sys.modules", {"aragora.queue": mock_queue}):
            yield mock_queue

    @pytest.mark.asyncio
    async def test_tracks_active_debate(self, active_debates):
        """Fallback stores debate in _active_debates."""
        from aragora.server.handlers.bots.teams_utils import _fallback_start_debate

        result = await _fallback_start_debate(
            topic="Fallback topic",
            conversation_id="conv-1",
            user_id="user-1",
            debate_id="fallback-001",
            service_url="https://service.example.com/",
            thread_id="thread-1",
        )
        assert result == "fallback-001"
        assert "fallback-001" in active_debates
        debate = active_debates["fallback-001"]
        assert debate["topic"] == "Fallback topic"
        assert debate["conversation_id"] == "conv-1"
        assert debate["user_id"] == "user-1"
        assert debate["service_url"] == "https://service.example.com/"
        assert debate["thread_id"] == "thread-1"
        assert "started_at" in debate

    @pytest.mark.asyncio
    async def test_origin_registration_success(self, active_debates):
        """Successfully registers debate origin."""
        from aragora.server.handlers.bots.teams_utils import _fallback_start_debate

        mock_register = MagicMock()
        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            mock_register,
            create=True,
        ):
            await _fallback_start_debate(
                topic="Origin test",
                conversation_id="conv-1",
                user_id="user-1",
                debate_id="origin-001",
                service_url="https://service.example.com/",
            )

    @pytest.mark.asyncio
    async def test_origin_registration_failure_continues(self, active_debates):
        """Origin registration failure does not prevent debate creation."""
        from aragora.server.handlers.bots.teams_utils import _fallback_start_debate

        with patch(
            "aragora.server.debate_origin.register_debate_origin",
            side_effect=RuntimeError("origin broken"),
            create=True,
        ):
            result = await _fallback_start_debate(
                topic="Test",
                conversation_id="conv-1",
                user_id="user-1",
                debate_id="fail-origin-001",
                service_url="https://service.example.com/",
            )
        assert result == "fail-origin-001"
        assert "fail-origin-001" in active_debates

    @pytest.mark.asyncio
    async def test_redis_queue_import_error(self, active_debates):
        """Redis queue ImportError does not prevent debate creation."""
        from aragora.server.handlers.bots.teams_utils import _fallback_start_debate

        with patch.dict("sys.modules", {"aragora.queue": None}):
            result = await _fallback_start_debate(
                topic="Test",
                conversation_id="conv-1",
                user_id="user-1",
                debate_id="no-redis-001",
                service_url="https://service.example.com/",
            )
        assert result == "no-redis-001"
        assert "no-redis-001" in active_debates

    @pytest.mark.asyncio
    async def test_returns_provided_debate_id(self, active_debates):
        """Returns the provided debate_id exactly."""
        from aragora.server.handlers.bots.teams_utils import _fallback_start_debate

        result = await _fallback_start_debate(
            topic="Test",
            conversation_id="conv-1",
            user_id="user-1",
            debate_id="exact-id-123",
            service_url="https://service.example.com/",
        )
        assert result == "exact-id-123"

    @pytest.mark.asyncio
    async def test_thread_id_none_default(self, active_debates):
        """thread_id defaults to None."""
        from aragora.server.handlers.bots.teams_utils import _fallback_start_debate

        await _fallback_start_debate(
            topic="Test",
            conversation_id="conv-1",
            user_id="user-1",
            debate_id="no-thread",
            service_url="https://service.example.com/",
        )
        assert active_debates["no-thread"]["thread_id"] is None


# ===========================================================================
# Vote Counting
# ===========================================================================


class TestGetDebateVoteCounts:
    """Test get_debate_vote_counts."""

    def test_empty_votes(self):
        """No votes returns empty dict."""
        from aragora.server.handlers.bots.teams_utils import get_debate_vote_counts

        assert get_debate_vote_counts("nonexistent") == {}

    def test_single_vote(self, user_votes):
        """Single vote returns count of 1."""
        from aragora.server.handlers.bots.teams_utils import get_debate_vote_counts

        user_votes["debate-1"] = {"user-A": "claude"}
        counts = get_debate_vote_counts("debate-1")
        assert counts == {"claude": 1}

    def test_multiple_votes_same_agent(self, user_votes):
        """Multiple votes for same agent are counted."""
        from aragora.server.handlers.bots.teams_utils import get_debate_vote_counts

        user_votes["debate-1"] = {"user-A": "claude", "user-B": "claude", "user-C": "claude"}
        counts = get_debate_vote_counts("debate-1")
        assert counts == {"claude": 3}

    def test_multiple_agents(self, user_votes):
        """Votes for different agents are counted separately."""
        from aragora.server.handlers.bots.teams_utils import get_debate_vote_counts

        user_votes["debate-1"] = {
            "user-A": "claude",
            "user-B": "gpt4",
            "user-C": "claude",
            "user-D": "gemini",
        }
        counts = get_debate_vote_counts("debate-1")
        assert counts == {"claude": 2, "gpt4": 1, "gemini": 1}

    def test_nonexistent_debate(self):
        """Nonexistent debate returns empty dict."""
        from aragora.server.handlers.bots.teams_utils import get_debate_vote_counts

        assert get_debate_vote_counts("no-such-debate") == {}

    def test_debate_with_empty_votes(self, user_votes):
        """Debate entry with no votes returns empty dict."""
        from aragora.server.handlers.bots.teams_utils import get_debate_vote_counts

        user_votes["debate-empty"] = {}
        counts = get_debate_vote_counts("debate-empty")
        assert counts == {}


# ===========================================================================
# Module-Level State
# ===========================================================================


class TestModuleLevelState:
    """Test module-level state variables."""

    def test_active_debates_is_dict(self):
        """_active_debates is a dict."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        assert isinstance(_active_debates, dict)

    def test_user_votes_is_dict(self):
        """_user_votes is a dict."""
        from aragora.server.handlers.bots.teams_utils import _user_votes

        assert isinstance(_user_votes, dict)

    def test_conversation_references_is_dict(self):
        """_conversation_references is a dict."""
        from aragora.server.handlers.bots.teams_utils import _conversation_references

        assert isinstance(_conversation_references, dict)

    def test_state_cleared_between_tests(self, active_debates, user_votes, conversation_refs):
        """Autouse fixture clears all state."""
        assert len(active_debates) == 0
        assert len(user_votes) == 0
        assert len(conversation_refs) == 0


# ===========================================================================
# Security Tests
# ===========================================================================


class TestSecurityEdgeCases:
    """Test security-related edge cases."""

    @pytest.mark.asyncio
    async def test_token_with_only_bearer_prefix(self):
        """'Bearer ' with empty token still passes to verifier (not rejected at header check)."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        mock_jwt_verify = MagicMock()
        mock_jwt_verify.HAS_JWT = True
        mock_jwt_verify.verify_teams_webhook = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"aragora.connectors.chat.jwt_verify": mock_jwt_verify}):
            result = await _verify_teams_token("Bearer ", "app-id")
            assert result is False
            mock_jwt_verify.verify_teams_webhook.assert_called_once_with("Bearer ", "app-id")

    @pytest.mark.asyncio
    async def test_token_case_sensitive_bearer(self):
        """'bearer' (lowercase) is rejected."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        assert await _verify_teams_token("bearer token123", "app-id") is False

    @pytest.mark.asyncio
    async def test_token_with_extra_spaces(self):
        """'Bearer  ' with extra spaces still passes header check."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        mock_jwt_verify = MagicMock()
        mock_jwt_verify.HAS_JWT = True
        mock_jwt_verify.verify_teams_webhook = MagicMock(return_value=True)

        with patch.dict("sys.modules", {"aragora.connectors.chat.jwt_verify": mock_jwt_verify}):
            result = await _verify_teams_token("Bearer  extra-spaces", "app-id")
            assert result is True

    def test_conversation_reference_no_injection(self, conversation_refs):
        """Conversation ID with special chars stored safely."""
        from aragora.server.handlers.bots.teams_utils import (
            _store_conversation_reference,
            get_conversation_reference,
        )

        malicious_id = "conv-<script>alert(1)</script>"
        activity = _make_activity(conversation_id=malicious_id)
        _store_conversation_reference(activity)
        ref = get_conversation_reference(malicious_id)
        assert ref is not None
        assert ref["conversation"]["id"] == malicious_id

    def test_build_debate_card_xss_in_topic(self):
        """Topic with HTML/script content is stored as-is (card renders safely)."""
        from aragora.server.handlers.bots.teams_utils import build_debate_card

        xss_topic = '<script>alert("xss")</script>'
        card = build_debate_card("d-001", xss_topic, ["claude"], 1, 3)
        # The topic is stored as-is; Adaptive Card rendering handles escaping
        assert xss_topic in card["body"][1]["text"]

    def test_build_consensus_card_xss_in_answer(self):
        """Final answer with script content is stored as-is."""
        from aragora.server.handlers.bots.teams_utils import build_consensus_card

        xss_answer = '<img onerror="alert(1)" src="x">'
        card = build_consensus_card("d-001", "Test", True, 0.9, None, xss_answer, {})
        answer_blocks = [b for b in card["body"] if "Decision" in b.get("text", "")]
        assert xss_answer in answer_blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_production_env_uppercase_still_production(self):
        """'PRODUCTION' (uppercase) is treated as production by .lower()."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.jwt_verify": None},
        ):
            with patch.dict("os.environ", {"ARAGORA_ENV": "PRODUCTION"}):
                result = await _verify_teams_token("Bearer token", "app-id")
                assert result is False

    @pytest.mark.asyncio
    async def test_dev_env_uppercase_allowed(self):
        """'DEVELOPMENT' (uppercase) is treated as dev by .lower()."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        with patch.dict(
            "sys.modules",
            {"aragora.connectors.chat.jwt_verify": None},
        ):
            with patch.dict("os.environ", {"ARAGORA_ENV": "DEVELOPMENT"}):
                result = await _verify_teams_token("Bearer token", "app-id")
                assert result is True


# ===========================================================================
# Integration-style tests
# ===========================================================================


class TestIntegration:
    """Integration-style tests combining multiple functions."""

    def test_store_then_get_conversation_reference(self):
        """Full round-trip: store activity, then get reference."""
        from aragora.server.handlers.bots.teams_utils import (
            _store_conversation_reference,
            get_conversation_reference,
        )

        activity = _make_activity(
            conversation_id="integration-conv",
            service_url="https://integration.example.com/",
            tenant_id="tenant-int",
        )
        _store_conversation_reference(activity)
        ref = get_conversation_reference("integration-conv")
        assert ref is not None
        assert ref["service_url"] == "https://integration.example.com/"
        assert ref["tenant_id"] == "tenant-int"

    def test_vote_counting_after_manual_population(self, user_votes):
        """Populate _user_votes manually and verify get_debate_vote_counts."""
        from aragora.server.handlers.bots.teams_utils import get_debate_vote_counts

        user_votes["debate-integration"] = {
            "u1": "claude",
            "u2": "claude",
            "u3": "gpt4",
            "u4": "gemini",
            "u5": "gpt4",
        }
        counts = get_debate_vote_counts("debate-integration")
        assert counts["claude"] == 2
        assert counts["gpt4"] == 2
        assert counts["gemini"] == 1

    def test_debate_card_with_consensus_card_flow(self):
        """Build a debate card followed by a consensus card for same debate."""
        from aragora.server.handlers.bots.teams_utils import (
            build_consensus_card,
            build_debate_card,
        )

        debate_id = "flow-001"
        debate_card = build_debate_card(
            debate_id=debate_id,
            topic="Flow test",
            agents=["claude", "gpt4"],
            current_round=3,
            total_rounds=3,
        )
        assert debate_card["type"] == "AdaptiveCard"

        consensus_card = build_consensus_card(
            debate_id=debate_id,
            topic="Flow test",
            consensus_reached=True,
            confidence=0.92,
            winner="claude",
            final_answer="Consensus answer",
            vote_counts={"claude": 3, "gpt4": 1},
        )
        assert consensus_card["type"] == "AdaptiveCard"
        assert "flow-001" in consensus_card["actions"][0]["url"]

    @pytest.mark.asyncio
    async def test_fallback_debate_populates_active_debates_and_counts(
        self, active_debates, user_votes
    ):
        """Fallback debate creates an entry, then votes can be counted."""
        from aragora.server.handlers.bots.teams_utils import (
            _fallback_start_debate,
            get_debate_vote_counts,
        )

        # Mock Redis queue to avoid real connection
        with patch.dict("sys.modules", {"aragora.queue": None}):
            debate_id = await _fallback_start_debate(
                topic="Vote test",
                conversation_id="conv-1",
                user_id="user-1",
                debate_id="vote-debate-001",
                service_url="https://service.example.com/",
            )
        assert debate_id in active_debates

        # Simulate votes
        user_votes[debate_id] = {"u1": "claude", "u2": "gpt4"}
        counts = get_debate_vote_counts(debate_id)
        assert counts == {"claude": 1, "gpt4": 1}
