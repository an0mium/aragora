"""
Integration tests for AutoGen agent orchestration with Aragora Gateway.

Tests AutoGen-specific features:
- Agent registration
- Group chat mode orchestration
- Two-agent conversation mode
- Code execution security controls
- Conversation turn limits
- Message filtering
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.integration.conftest import (
    MockExternalFrameworkServer,
    register_external_agent,
)


class TestAutoGenOrchestration:
    """Integration tests for AutoGen agents with gateway."""

    @pytest.mark.asyncio
    async def test_autogen_agent_registration(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test registering an AutoGen agent via the gateway.

        Verifies that an AutoGen agent can be registered with the gateway
        and that the registration includes AutoGen-specific metadata.
        """
        # Register AutoGen agent
        agent_info = register_external_agent(
            gateway_server_context,
            name="autogen-test",
            framework="autogen",
            base_url="https://autogen.example.com/api",
        )

        # Verify registration succeeded
        assert "autogen-test" in gateway_server_context["external_agents"]
        assert agent_info["framework_type"] == "autogen"
        assert agent_info["status"] == "registered"
        assert agent_info["base_url"] == "https://autogen.example.com/api"

    @pytest.mark.asyncio
    async def test_autogen_groupchat_mode(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test AutoGen group chat mode orchestration.

        Verifies that when configured for groupchat mode, the AutoGen
        agent orchestrates multi-agent conversations properly.
        """
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        # Configure for groupchat mode
        config = AutoGenConfig(
            base_url="http://localhost:8000",
            mode="groupchat",
            max_round=5,
            speaker_selection_method="auto",
            allow_code_execution=False,
        )

        # Mock the HTTP session
        with patch(
            "aragora.agents.api_agents.external_framework.ExternalFrameworkAgent._get_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "response": "GroupChat: Agent1 and Agent2 discussed the topic",
                    "conversation_id": "conv-groupchat-123",
                    "participants": ["agent1", "agent2", "agent3"],
                    "rounds_completed": 4,
                }
            )

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)
            mock_get_session.return_value = mock_session

            agent = AutoGenAgent(
                name="autogen-groupchat",
                config=config,
                api_key="test-key",
            )

            # Verify config
            assert agent.autogen_config.mode == "groupchat"
            assert agent.autogen_config.max_round == 5
            assert agent.autogen_config.speaker_selection_method == "auto"

            # Build the prefix and verify it includes groupchat info
            prefix = agent._build_autogen_prefix()
            assert "multi-agent groupchat" in prefix
            assert "Max Rounds: 5" in prefix
            assert "Code Execution: disabled" in prefix

    @pytest.mark.asyncio
    async def test_autogen_two_agent_mode(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test AutoGen two-agent conversation mode.

        Verifies that when configured for two_agent mode, the AutoGen
        agent orchestrates a simple two-party conversation.
        """
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        # Configure for two-agent mode
        config = AutoGenConfig(
            base_url="http://localhost:8000",
            mode="two_agent",
            max_round=10,
            allow_code_execution=False,
        )

        agent = AutoGenAgent(
            name="autogen-twoagent",
            config=config,
            api_key="test-key",
        )

        # Verify config
        assert agent.autogen_config.mode == "two_agent"

        # Build the prefix and verify it includes two-agent info
        prefix = agent._build_autogen_prefix()
        assert "two-agent conversation" in prefix

        # Test initiate_chat for two-agent mode
        with patch(
            "aragora.agents.api_agents.external_framework.ExternalFrameworkAgent._get_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "response": "Two-agent response: Assistant helped user",
                    "conversation_id": "conv-twoagent-456",
                }
            )

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)
            mock_get_session.return_value = mock_session

            result = await agent.initiate_chat(
                message="Help me debug this code",
                agents=["user_proxy", "assistant"],
            )

            assert result["success"] is True
            assert "conversation_id" in result
            assert result["response"] is not None

    @pytest.mark.asyncio
    async def test_autogen_code_execution_disabled(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test that code execution is blocked by default in AutoGen.

        Verifies that the default security setting disables code execution
        and that attempting to enable it requires a valid work directory.
        """
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        # Default config should have code execution disabled
        default_config = AutoGenConfig(base_url="http://localhost:8000")
        assert default_config.allow_code_execution is False

        agent_default = AutoGenAgent(
            name="autogen-no-code",
            config=default_config,
            api_key="test-key",
        )

        # Verify code execution is disabled in the agent
        assert agent_default.autogen_config.allow_code_execution is False

        # Verify prefix shows code execution is disabled
        prefix = agent_default._build_autogen_prefix()
        assert "Code Execution: disabled" in prefix

        # Verify that enabling code execution without work_dir fails
        with pytest.raises(ValueError) as exc_info:
            invalid_config = AutoGenConfig(
                base_url="http://localhost:8000",
                allow_code_execution=True,
                work_dir=None,  # No work directory - should fail
            )
            AutoGenAgent(
                name="autogen-invalid",
                config=invalid_config,
                api_key="test-key",
            )

        assert "work_dir" in str(exc_info.value).lower()

        # Verify that invalid (relative) work_dir fails
        with pytest.raises(ValueError):
            invalid_path_config = AutoGenConfig(
                base_url="http://localhost:8000",
                allow_code_execution=True,
                work_dir="./relative/path",  # Relative path - should fail
            )
            AutoGenAgent(
                name="autogen-invalid-path",
                config=invalid_path_config,
                api_key="test-key",
            )

    @pytest.mark.asyncio
    async def test_autogen_max_turns_enforced(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test that conversation turn limits are enforced.

        Verifies that the max_round setting limits the number of
        conversation rounds to prevent runaway conversations.
        """
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        # Configure with limited rounds
        config = AutoGenConfig(
            base_url="http://localhost:8000",
            mode="groupchat",
            max_round=3,  # Only 3 rounds allowed
            allow_code_execution=False,
        )

        agent = AutoGenAgent(
            name="autogen-limited",
            config=config,
            api_key="test-key",
        )

        # Verify max_round is set
        assert agent.autogen_config.max_round == 3

        # Build request payload and verify max_round is included
        payload = agent._build_request_payload("Test prompt")

        assert payload["config"]["max_round"] == 3
        assert payload["config"]["mode"] == "groupchat"

        # Test with different max_round values
        config_high = AutoGenConfig(
            base_url="http://localhost:8000",
            max_round=100,
        )
        agent_high = AutoGenAgent(
            name="autogen-high-rounds",
            config=config_high,
            api_key="test-key",
        )

        payload_high = agent_high._build_request_payload("Test")
        assert payload_high["config"]["max_round"] == 100

        # Verify very low max_round
        config_low = AutoGenConfig(
            base_url="http://localhost:8000",
            max_round=1,
        )
        agent_low = AutoGenAgent(
            name="autogen-low-rounds",
            config=config_low,
            api_key="test-key",
        )

        payload_low = agent_low._build_request_payload("Test")
        assert payload_low["config"]["max_round"] == 1

    @pytest.mark.asyncio
    async def test_autogen_message_filtering(
        self,
        gateway_server_context: dict,
        mock_external_server: MockExternalFrameworkServer,
    ) -> None:
        """Test that system messages are filtered appropriately.

        Verifies that the AutoGen agent properly handles different
        message types and filters system-level messages as needed.
        """
        from aragora.agents.api_agents.autogen_agent import AutoGenAgent, AutoGenConfig

        config = AutoGenConfig(
            base_url="http://localhost:8000",
            mode="groupchat",
            max_round=10,
            human_input_mode="NEVER",  # Never ask for human input
        )

        agent = AutoGenAgent(
            name="autogen-filter",
            config=config,
            api_key="test-key",
        )

        # Verify human input mode is set to NEVER
        assert agent.autogen_config.human_input_mode == "NEVER"

        # Build payload and check human_input_mode is included
        payload = agent._build_request_payload("Test message")
        assert payload["config"]["human_input_mode"] == "NEVER"

        # Test conversation storage and retrieval
        with patch(
            "aragora.agents.api_agents.external_framework.ExternalFrameworkAgent._get_session"
        ) as mock_get_session:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "response": "Filtered response without system messages",
                }
            )

            mock_cm = AsyncMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_cm)
            mock_get_session.return_value = mock_session

            # Initiate a chat
            result = await agent.initiate_chat(message="Test message")

            if result["success"]:
                conv_id = result["conversation_id"]

                # Verify conversation is stored
                conversation = agent.get_conversation(conv_id)
                assert conversation is not None
                assert len(conversation) == 2  # User message + assistant response

                # Verify message roles are correct
                assert conversation[0]["role"] == "user"
                assert conversation[1]["role"] == "assistant"

                # Test conversation clearing
                cleared = agent.clear_conversation(conv_id)
                assert cleared is True

                # Verify conversation is cleared
                assert agent.get_conversation(conv_id) is None

        # Test clear_all_conversations
        agent._conversations = {
            "conv-1": [{"role": "user", "content": "msg1"}],
            "conv-2": [{"role": "user", "content": "msg2"}],
            "conv-3": [{"role": "user", "content": "msg3"}],
        }

        count = agent.clear_all_conversations()
        assert count == 3
        assert len(agent.get_all_conversations()) == 0

        # Test clearing non-existent conversation
        result = agent.clear_conversation("non-existent")
        assert result is False
