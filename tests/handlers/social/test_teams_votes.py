"""Tests for Teams vote recording functionality."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from aragora.server.handlers.social.teams import TeamsIntegrationHandler


@pytest.fixture
def handler():
    """Create a TeamsIntegrationHandler instance."""
    mock_context = MagicMock()
    return TeamsIntegrationHandler(mock_context)


class TestTeamsVoteRecording:
    """Tests for Teams vote recording in _handle_vote."""

    def test_vote_records_in_database(self, handler):
        """Test that votes are recorded in the debates database."""
        value = {"vote": "agree", "debate_id": "debate_123", "action": "vote"}
        conversation = {"id": "conv_456"}
        service_url = "https://smba.trafficmanager.net/teams/"
        from_user = {"id": "user_789", "name": "Test User"}

        with patch("aragora.server.storage.get_debates_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.record_vote = MagicMock()
            mock_get_db.return_value = mock_db

            result = handler._handle_vote(value, conversation, service_url, from_user)

            # Verify vote was recorded
            mock_db.record_vote.assert_called_once_with(
                debate_id="debate_123",
                voter_id="teams:user_789",
                vote="agree",
                source="teams",
            )

            # Verify response
            assert result[0]["status"] == "vote_recorded"
            assert result[0]["vote"] == "agree"
            assert result[0]["debate_id"] == "debate_123"

    def test_vote_records_in_aggregator(self, handler):
        """Test that votes are recorded in the vote aggregator."""
        value = {"vote": "disagree", "debate_id": "debate_456", "action": "vote"}
        conversation = {"id": "conv_789"}
        service_url = "https://smba.trafficmanager.net/teams/"
        from_user = {"id": "user_abc", "name": "Test User"}

        with patch("aragora.server.storage.get_debates_db") as mock_get_db:
            mock_get_db.return_value = None

            with patch("aragora.debate.vote_aggregator.VoteAggregator") as mock_agg_class:
                mock_aggregator = MagicMock()
                mock_agg_class.get_instance.return_value = mock_aggregator

                handler._handle_vote(value, conversation, service_url, from_user)

                # Verify aggregator was called with "against" for disagree vote
                mock_aggregator.record_vote.assert_called_once_with(
                    "debate_456", "teams:user_abc", "against"
                )

    def test_vote_maps_agree_to_for(self, handler):
        """Test that 'agree' vote maps to 'for' position."""
        value = {"vote": "agree", "debate_id": "debate_789", "action": "vote"}
        conversation = {"id": "conv_123"}
        service_url = "https://smba.trafficmanager.net/teams/"
        from_user = {"id": "user_def"}

        with patch("aragora.server.storage.get_debates_db") as mock_get_db:
            mock_get_db.return_value = None

            with patch("aragora.debate.vote_aggregator.VoteAggregator") as mock_agg_class:
                mock_aggregator = MagicMock()
                mock_agg_class.get_instance.return_value = mock_aggregator

                handler._handle_vote(value, conversation, service_url, from_user)

                mock_aggregator.record_vote.assert_called_once_with(
                    "debate_789", "teams:user_def", "for"
                )

    def test_vote_handles_missing_user(self, handler):
        """Test that votes work when user info is missing."""
        value = {"vote": "agree", "debate_id": "debate_abc", "action": "vote"}
        conversation = {"id": "conv_456"}
        service_url = "https://smba.trafficmanager.net/teams/"

        with patch("aragora.server.storage.get_debates_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.record_vote = MagicMock()
            mock_get_db.return_value = mock_db

            # Pass None for from_user
            result = handler._handle_vote(value, conversation, service_url, None)

            # Verify vote was recorded with "unknown" user
            mock_db.record_vote.assert_called_once_with(
                debate_id="debate_abc",
                voter_id="teams:unknown",
                vote="agree",
                source="teams",
            )

            assert result[0]["status"] == "vote_recorded"

    def test_vote_handles_database_error(self, handler):
        """Test that vote recording handles database errors gracefully."""
        value = {"vote": "agree", "debate_id": "debate_err", "action": "vote"}
        conversation = {"id": "conv_789"}
        service_url = "https://smba.trafficmanager.net/teams/"
        from_user = {"id": "user_ghi"}

        with patch("aragora.server.storage.get_debates_db") as mock_get_db:
            mock_db = MagicMock()
            mock_db.record_vote.side_effect = Exception("DB connection failed")
            mock_get_db.return_value = mock_db

            # Should not raise, just log warning
            result = handler._handle_vote(value, conversation, service_url, from_user)

            # Response should still be success
            assert result[0]["status"] == "vote_recorded"

    def test_vote_handles_missing_aggregator(self, handler):
        """Test that vote recording handles missing aggregator gracefully."""
        value = {"vote": "disagree", "debate_id": "debate_xyz", "action": "vote"}
        conversation = {"id": "conv_abc"}
        service_url = "https://smba.trafficmanager.net/teams/"
        from_user = {"id": "user_jkl"}

        with patch("aragora.server.storage.get_debates_db") as mock_get_db:
            mock_get_db.return_value = None

            with patch("aragora.debate.vote_aggregator.VoteAggregator") as mock_agg_class:
                mock_agg_class.get_instance.return_value = None

                # Should not raise
                result = handler._handle_vote(value, conversation, service_url, from_user)

                assert result[0]["status"] == "vote_recorded"


class TestTeamsInteractiveVote:
    """Tests for vote handling via interactive endpoint."""

    def test_interactive_passes_user_to_vote(self, handler):
        """Test that _handle_interactive passes user info to _handle_vote."""
        body = {
            "value": {"action": "vote", "vote": "agree", "debate_id": "debate_int"},
            "conversation": {"id": "conv_int"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "user_interactive", "name": "Interactive User"},
        }

        with patch.object(handler, "_read_json_body", return_value=body):
            with patch.object(handler, "_handle_vote") as mock_vote:
                mock_vote.return_value = ({"status": "ok"}, 200)

                mock_handler = MagicMock()
                handler._handle_interactive(mock_handler)

                # Verify _handle_vote was called with from_user
                mock_vote.assert_called_once()
                call_args = mock_vote.call_args
                assert call_args[0][3] == {"id": "user_interactive", "name": "Interactive User"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
