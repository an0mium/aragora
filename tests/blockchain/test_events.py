"""
Tests for blockchain event listener module.

Tests cover:
- EventType enum values and member count
- BlockchainEvent creation, defaults, event_id property
- EventFilter matching logic (event types, agent IDs, block range)
- EventListener initialization and handler management
- EventListener dispatch (sync, async, error handling)
- EventListener parse methods (identity, feedback, validation)
- EventListener _get_contract_events (success, empty, error)
- EventListener _poll_events (no new blocks, processes blocks, registries)
- Blockchain __init__.py lazy imports and caching
- Module __all__ exports
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.blockchain.events import (
    BlockchainEvent,
    EventFilter,
    EventListener,
    EventType,
)
from aragora.blockchain.models import ValidationResponse


class TestEventType:
    """Tests for EventType enum."""

    def test_registered_value(self):
        assert EventType.REGISTERED.value == "Registered"

    def test_uri_updated_value(self):
        assert EventType.URI_UPDATED.value == "URIUpdated"

    def test_metadata_set_value(self):
        assert EventType.METADATA_SET.value == "MetadataSet"

    def test_new_feedback_value(self):
        assert EventType.NEW_FEEDBACK.value == "NewFeedback"

    def test_feedback_revoked_value(self):
        assert EventType.FEEDBACK_REVOKED.value == "FeedbackRevoked"

    def test_response_appended_value(self):
        assert EventType.RESPONSE_APPENDED.value == "ResponseAppended"

    def test_validation_request_value(self):
        assert EventType.VALIDATION_REQUEST.value == "ValidationRequest"

    def test_validation_response_value(self):
        assert EventType.VALIDATION_RESPONSE.value == "ValidationResponse"

    def test_member_count(self):
        """Enum has exactly 8 members."""
        assert len(EventType) == 8

    def test_identity_events(self):
        """Identity events are the first three."""
        identity = [EventType.REGISTERED, EventType.URI_UPDATED, EventType.METADATA_SET]
        assert len(identity) == 3

    def test_reputation_events(self):
        """Reputation events are three."""
        reputation = [
            EventType.NEW_FEEDBACK,
            EventType.FEEDBACK_REVOKED,
            EventType.RESPONSE_APPENDED,
        ]
        assert len(reputation) == 3

    def test_validation_events(self):
        """Validation events are two."""
        validation = [EventType.VALIDATION_REQUEST, EventType.VALIDATION_RESPONSE]
        assert len(validation) == 2


class TestBlockchainEvent:
    """Tests for BlockchainEvent dataclass."""

    def test_create_event(self):
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=12345,
            tx_hash="0xabc123",
            log_index=0,
        )
        assert event.event_type == EventType.REGISTERED
        assert event.block_number == 12345
        assert event.tx_hash == "0xabc123"
        assert event.log_index == 0

    def test_event_defaults(self):
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        assert event.timestamp is None
        assert event.args == {}
        assert event.raw is None

    def test_event_id(self):
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0xabc123",
            log_index=5,
        )
        assert event.event_id == "0xabc123:5"

    def test_event_id_zero_log_index(self):
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0xdef",
            log_index=0,
        )
        assert event.event_id == "0xdef:0"

    def test_event_with_timestamp(self):
        now = datetime.now(timezone.utc)
        event = BlockchainEvent(
            event_type=EventType.NEW_FEEDBACK,
            block_number=100,
            tx_hash="0xdef",
            log_index=0,
            timestamp=now,
        )
        assert event.timestamp == now

    def test_event_with_args(self):
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
            args={"agentId": 42, "owner": "0x123"},
        )
        assert event.args["agentId"] == 42
        assert event.args["owner"] == "0x123"

    def test_event_with_raw(self):
        raw_data = {"some": "raw", "data": True}
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
            raw=raw_data,
        )
        assert event.raw == raw_data

    def test_args_is_independent(self):
        """Args dicts are independent across instances."""
        e1 = BlockchainEvent(
            event_type=EventType.REGISTERED, block_number=1, tx_hash="0x1", log_index=0
        )
        e2 = BlockchainEvent(
            event_type=EventType.REGISTERED, block_number=1, tx_hash="0x2", log_index=0
        )
        e1.args["key"] = "value"
        assert "key" not in e2.args


class TestEventFilter:
    """Tests for EventFilter."""

    def test_default_filter(self):
        f = EventFilter()
        assert f.event_types is None
        assert f.agent_ids is None
        assert f.from_block == "latest"
        assert f.to_block == "latest"

    def test_filter_matches_all_when_no_criteria(self):
        f = EventFilter()
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        assert f.matches(event) is True

    def test_filter_matches_event_type(self):
        f = EventFilter(event_types=[EventType.REGISTERED, EventType.URI_UPDATED])
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        assert f.matches(event) is True

    def test_filter_rejects_event_type(self):
        f = EventFilter(event_types=[EventType.REGISTERED])
        event = BlockchainEvent(
            event_type=EventType.NEW_FEEDBACK,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        assert f.matches(event) is False

    def test_filter_matches_agent_id(self):
        f = EventFilter(agent_ids=[42, 43])
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
            args={"agentId": 42},
        )
        assert f.matches(event) is True

    def test_filter_rejects_agent_id(self):
        f = EventFilter(agent_ids=[42])
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
            args={"agentId": 99},
        )
        assert f.matches(event) is False

    def test_filter_matches_when_no_agent_id_in_event(self):
        f = EventFilter(agent_ids=[42])
        event = BlockchainEvent(
            event_type=EventType.METADATA_SET,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
            args={},
        )
        assert f.matches(event) is True

    def test_filter_with_block_range(self):
        f = EventFilter(from_block=100, to_block=200)
        assert f.from_block == 100
        assert f.to_block == 200

    def test_filter_empty_event_types_list(self):
        """Empty event_types list rejects all."""
        f = EventFilter(event_types=[])
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        # Empty list is falsy, so filter passes
        assert f.matches(event) is True

    def test_filter_both_criteria(self):
        """Filter with both event_types and agent_ids must match both."""
        f = EventFilter(event_types=[EventType.REGISTERED], agent_ids=[42])
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
            args={"agentId": 42},
        )
        assert f.matches(event) is True

    def test_filter_event_type_match_agent_mismatch(self):
        f = EventFilter(event_types=[EventType.REGISTERED], agent_ids=[42])
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
            args={"agentId": 99},
        )
        assert f.matches(event) is False


class TestEventListener:
    """Tests for EventListener class."""

    def _make_listener(self):
        provider = MagicMock()
        provider.get_config.return_value = MagicMock(
            chain_id=1,
            has_identity_registry=False,
            has_reputation_registry=False,
            has_validation_registry=False,
        )
        return EventListener(provider), provider

    def test_init(self):
        provider = MagicMock()
        listener = EventListener(provider)
        assert listener._provider is provider
        assert listener._chain_id is None
        assert listener._handlers == {}
        assert listener._running is False
        assert listener._poll_interval == 12.0
        assert listener._last_block is None

    def test_init_with_chain_id(self):
        provider = MagicMock()
        listener = EventListener(provider, chain_id=137)
        assert listener._chain_id == 137

    def test_on_decorator(self):
        listener, _ = self._make_listener()

        @listener.on(EventType.REGISTERED)
        def handler(event):
            pass

        assert EventType.REGISTERED in listener._handlers
        assert handler in listener._handlers[EventType.REGISTERED]

    def test_on_decorator_returns_handler(self):
        """@on decorator returns the original handler function."""
        listener, _ = self._make_listener()

        @listener.on(EventType.REGISTERED)
        def handler(event):
            return "result"

        assert callable(handler)

    def test_on_decorator_multiple_handlers(self):
        listener, _ = self._make_listener()

        @listener.on(EventType.REGISTERED)
        def handler1(event):
            pass

        @listener.on(EventType.REGISTERED)
        def handler2(event):
            pass

        assert len(listener._handlers[EventType.REGISTERED]) == 2

    def test_on_decorator_different_types(self):
        listener, _ = self._make_listener()

        @listener.on(EventType.REGISTERED)
        def handler1(event):
            pass

        @listener.on(EventType.NEW_FEEDBACK)
        def handler2(event):
            pass

        assert len(listener._handlers) == 2

    def test_add_handler(self):
        listener, _ = self._make_listener()

        def handler(event):
            pass

        listener.add_handler(EventType.NEW_FEEDBACK, handler)
        assert handler in listener._handlers[EventType.NEW_FEEDBACK]

    def test_add_handler_to_existing(self):
        listener, _ = self._make_listener()

        def handler1(event):
            pass

        def handler2(event):
            pass

        listener.add_handler(EventType.NEW_FEEDBACK, handler1)
        listener.add_handler(EventType.NEW_FEEDBACK, handler2)
        assert len(listener._handlers[EventType.NEW_FEEDBACK]) == 2

    def test_remove_handler(self):
        listener, _ = self._make_listener()

        def handler(event):
            pass

        listener.add_handler(EventType.REGISTERED, handler)
        listener.remove_handler(EventType.REGISTERED, handler)
        assert listener._handlers[EventType.REGISTERED] == []

    def test_remove_handler_nonexistent_type(self):
        listener, _ = self._make_listener()

        def handler(event):
            pass

        # Should not raise
        listener.remove_handler(EventType.REGISTERED, handler)

    def test_remove_handler_preserves_others(self):
        listener, _ = self._make_listener()

        def handler1(event):
            pass

        def handler2(event):
            pass

        listener.add_handler(EventType.REGISTERED, handler1)
        listener.add_handler(EventType.REGISTERED, handler2)
        listener.remove_handler(EventType.REGISTERED, handler1)
        assert len(listener._handlers[EventType.REGISTERED]) == 1
        assert handler2 in listener._handlers[EventType.REGISTERED]

    def test_stop(self):
        listener, _ = self._make_listener()
        listener._running = True
        listener.stop()
        assert listener._running is False


class TestEventListenerDispatch:
    """Tests for EventListener event dispatch."""

    def _make_listener(self):
        provider = MagicMock()
        return EventListener(provider), provider

    @pytest.mark.asyncio
    async def test_dispatch_sync_handler(self):
        listener, _ = self._make_listener()
        results = []

        def handler(event):
            results.append(event.event_type)

        listener.add_handler(EventType.REGISTERED, handler)

        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        await listener._dispatch_event(event)
        assert results == [EventType.REGISTERED]

    @pytest.mark.asyncio
    async def test_dispatch_async_handler(self):
        listener, _ = self._make_listener()
        results = []

        async def handler(event):
            results.append(event.event_type)

        listener.add_handler(EventType.NEW_FEEDBACK, handler)

        event = BlockchainEvent(
            event_type=EventType.NEW_FEEDBACK,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        await listener._dispatch_event(event)
        assert results == [EventType.NEW_FEEDBACK]

    @pytest.mark.asyncio
    async def test_dispatch_no_handlers(self):
        listener, _ = self._make_listener()
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        # Should not raise
        await listener._dispatch_event(event)

    @pytest.mark.asyncio
    async def test_dispatch_handler_error_doesnt_stop_others(self):
        listener, _ = self._make_listener()
        results = []

        def bad_handler(event):
            raise ValueError("oops")

        def good_handler(event):
            results.append("ok")

        listener.add_handler(EventType.REGISTERED, bad_handler)
        listener.add_handler(EventType.REGISTERED, good_handler)

        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        await listener._dispatch_event(event)
        assert results == ["ok"]

    @pytest.mark.asyncio
    async def test_dispatch_multiple_handlers_in_order(self):
        listener, _ = self._make_listener()
        results = []

        def handler1(event):
            results.append("first")

        def handler2(event):
            results.append("second")

        listener.add_handler(EventType.REGISTERED, handler1)
        listener.add_handler(EventType.REGISTERED, handler2)

        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=1,
            tx_hash="0x1",
            log_index=0,
        )
        await listener._dispatch_event(event)
        assert results == ["first", "second"]


class TestEventListenerParseMethods:
    """Tests for EventListener parse methods."""

    def _make_listener(self):
        provider = MagicMock()
        config = MagicMock()
        config.chain_id = 1
        provider.get_config.return_value = config
        return EventListener(provider), provider

    def test_parse_identity_event_registered(self):
        listener, _ = self._make_listener()
        now = datetime.now(timezone.utc)
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=100,
            tx_hash="0xabc123",
            log_index=0,
            timestamp=now,
            args={
                "agentId": 42,
                "owner": "0x1234567890123456789012345678901234567890",
                "agentURI": "ipfs://QmTest",
            },
        )
        identity = listener.parse_identity_event(event)
        assert identity is not None
        assert identity.token_id == 42
        assert identity.owner == "0x1234567890123456789012345678901234567890"
        assert identity.agent_uri == "ipfs://QmTest"
        assert identity.chain_id == 1
        assert identity.registered_at == now
        assert identity.tx_hash == "0xabc123"

    def test_parse_identity_event_non_registered(self):
        listener, _ = self._make_listener()
        event = BlockchainEvent(
            event_type=EventType.URI_UPDATED,
            block_number=100,
            tx_hash="0xabc123",
            log_index=0,
        )
        result = listener.parse_identity_event(event)
        assert result is None

    def test_parse_identity_event_missing_args(self):
        """parse_identity_event uses defaults for missing args."""
        listener, _ = self._make_listener()
        event = BlockchainEvent(
            event_type=EventType.REGISTERED,
            block_number=100,
            tx_hash="0xabc",
            log_index=0,
            args={},
        )
        identity = listener.parse_identity_event(event)
        assert identity is not None
        assert identity.token_id == 0
        assert identity.owner == ""
        assert identity.agent_uri == ""

    def test_parse_feedback_event_new_feedback(self):
        listener, _ = self._make_listener()
        now = datetime.now(timezone.utc)
        event = BlockchainEvent(
            event_type=EventType.NEW_FEEDBACK,
            block_number=100,
            tx_hash="0xdef456",
            log_index=1,
            timestamp=now,
            args={
                "agentId": 10,
                "clientAddress": "0xABCD1234567890ABCD1234567890ABCD12345678",
                "feedbackIndex": 3,
                "value": 100,
                "valueDecimals": 2,
                "tag1": "accuracy",
                "tag2": "reasoning",
                "endpoint": "/api/chat",
                "feedbackURI": "ipfs://QmFeedback",
                "feedbackHash": b"\xab\xcd",
            },
        )
        feedback = listener.parse_feedback_event(event)
        assert feedback is not None
        assert feedback.agent_id == 10
        assert feedback.client_address == "0xABCD1234567890ABCD1234567890ABCD12345678"
        assert feedback.feedback_index == 3
        assert feedback.value == 100
        assert feedback.value_decimals == 2
        assert feedback.tag1 == "accuracy"
        assert feedback.tag2 == "reasoning"
        assert feedback.endpoint == "/api/chat"
        assert feedback.feedback_uri == "ipfs://QmFeedback"
        assert feedback.feedback_hash == "abcd"
        assert feedback.timestamp == now
        assert feedback.tx_hash == "0xdef456"

    def test_parse_feedback_event_string_hash(self):
        listener, _ = self._make_listener()
        event = BlockchainEvent(
            event_type=EventType.NEW_FEEDBACK,
            block_number=100,
            tx_hash="0xdef456",
            log_index=0,
            args={
                "agentId": 10,
                "clientAddress": "0x1234567890123456789012345678901234567890",
                "feedbackHash": "already_hex_string",
            },
        )
        feedback = listener.parse_feedback_event(event)
        assert feedback is not None
        assert feedback.feedback_hash == "already_hex_string"

    def test_parse_feedback_event_non_feedback(self):
        listener, _ = self._make_listener()
        event = BlockchainEvent(
            event_type=EventType.FEEDBACK_REVOKED,
            block_number=100,
            tx_hash="0xdef456",
            log_index=0,
        )
        result = listener.parse_feedback_event(event)
        assert result is None

    def test_parse_validation_event_response(self):
        listener, _ = self._make_listener()
        now = datetime.now(timezone.utc)
        event = BlockchainEvent(
            event_type=EventType.VALIDATION_RESPONSE,
            block_number=200,
            tx_hash="0xval789",
            log_index=2,
            timestamp=now,
            args={
                "requestHash": b"\xde\xad",
                "agentId": 5,
                "validatorAddress": "0xVALIDATOR1234567890VALIDATOR12345678901234",
                "response": 1,
                "responseURI": "ipfs://QmResponse",
                "responseHash": b"\xbe\xef",
                "tag": "safety",
            },
        )
        record = listener.parse_validation_event(event)
        assert record is not None
        assert record.request_hash == "dead"
        assert record.agent_id == 5
        assert record.validator_address == "0xVALIDATOR1234567890VALIDATOR12345678901234"
        assert record.response == ValidationResponse.PASS
        assert record.response_uri == "ipfs://QmResponse"
        assert record.response_hash == "beef"
        assert record.tag == "safety"
        assert record.last_update == now
        assert record.tx_hash == "0xval789"

    def test_parse_validation_event_string_hashes(self):
        listener, _ = self._make_listener()
        event = BlockchainEvent(
            event_type=EventType.VALIDATION_RESPONSE,
            block_number=200,
            tx_hash="0xval789",
            log_index=0,
            args={
                "requestHash": "hex_string",
                "agentId": 5,
                "response": 0,
                "responseHash": "another_hex",
            },
        )
        record = listener.parse_validation_event(event)
        assert record is not None
        assert record.request_hash == "hex_string"
        assert record.response_hash == "another_hex"

    def test_parse_validation_event_non_response(self):
        listener, _ = self._make_listener()
        event = BlockchainEvent(
            event_type=EventType.VALIDATION_REQUEST,
            block_number=200,
            tx_hash="0xval789",
            log_index=0,
        )
        result = listener.parse_validation_event(event)
        assert result is None

    def test_parse_validation_event_fail_response(self):
        listener, _ = self._make_listener()
        event = BlockchainEvent(
            event_type=EventType.VALIDATION_RESPONSE,
            block_number=200,
            tx_hash="0xval",
            log_index=0,
            args={
                "requestHash": b"\x01",
                "agentId": 1,
                "response": 2,  # FAIL
                "responseHash": b"\x02",
            },
        )
        record = listener.parse_validation_event(event)
        assert record.response == ValidationResponse.FAIL


class TestEventListenerGetContractEvents:
    """Tests for _get_contract_events method."""

    def _make_listener(self):
        provider = MagicMock()
        return EventListener(provider), provider

    def test_get_contract_events_success(self):
        listener, provider = self._make_listener()
        mock_w3 = MagicMock()
        provider.get_web3.return_value = mock_w3

        # Mock block data
        mock_w3.eth.get_block.return_value = {"timestamp": 1700000000}

        # Mock event filter
        mock_log = {
            "blockNumber": 100,
            "transactionHash": MagicMock(hex=lambda: "0xabc123"),
            "logIndex": 0,
            "args": {"agentId": 42},
        }
        mock_filter = MagicMock()
        mock_filter.get_all_entries.return_value = [mock_log]

        mock_contract = MagicMock()
        mock_contract.events.Registered.create_filter.return_value = mock_filter

        events = listener._get_contract_events(mock_contract, [EventType.REGISTERED], 90, 100)

        assert len(events) == 1
        assert events[0].event_type == EventType.REGISTERED
        assert events[0].block_number == 100
        assert events[0].tx_hash == "0xabc123"
        assert events[0].log_index == 0
        assert events[0].args == {"agentId": 42}
        assert events[0].timestamp is not None

    def test_get_contract_events_empty(self):
        listener, provider = self._make_listener()
        mock_w3 = MagicMock()
        provider.get_web3.return_value = mock_w3

        mock_filter = MagicMock()
        mock_filter.get_all_entries.return_value = []

        mock_contract = MagicMock()
        mock_contract.events.Registered.create_filter.return_value = mock_filter

        events = listener._get_contract_events(mock_contract, [EventType.REGISTERED], 90, 100)
        assert events == []

    def test_get_contract_events_error_continues(self):
        listener, provider = self._make_listener()
        mock_w3 = MagicMock()
        provider.get_web3.return_value = mock_w3

        mock_contract = MagicMock()
        mock_contract.events.Registered.create_filter.side_effect = ConnectionError("RPC error")

        # Mock a second event type that succeeds
        mock_filter = MagicMock()
        mock_filter.get_all_entries.return_value = []
        mock_contract.events.URIUpdated.create_filter.return_value = mock_filter

        events = listener._get_contract_events(
            mock_contract, [EventType.REGISTERED, EventType.URI_UPDATED], 90, 100
        )
        # Registered fails, URIUpdated succeeds with empty
        assert events == []

    def test_get_contract_events_multiple_logs(self):
        """Multiple logs from the same event type are all captured."""
        listener, provider = self._make_listener()
        mock_w3 = MagicMock()
        provider.get_web3.return_value = mock_w3
        mock_w3.eth.get_block.return_value = {"timestamp": 1700000000}

        mock_log1 = {
            "blockNumber": 100,
            "transactionHash": MagicMock(hex=lambda: "0xaaa"),
            "logIndex": 0,
            "args": {"agentId": 1},
        }
        mock_log2 = {
            "blockNumber": 101,
            "transactionHash": MagicMock(hex=lambda: "0xbbb"),
            "logIndex": 1,
            "args": {"agentId": 2},
        }
        mock_filter = MagicMock()
        mock_filter.get_all_entries.return_value = [mock_log1, mock_log2]

        mock_contract = MagicMock()
        mock_contract.events.Registered.create_filter.return_value = mock_filter

        events = listener._get_contract_events(mock_contract, [EventType.REGISTERED], 90, 110)
        assert len(events) == 2


class TestEventListenerPollEvents:
    """Tests for _poll_events method."""

    @pytest.mark.asyncio
    async def test_poll_events_no_new_blocks(self):
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 100
        provider.get_web3.return_value = mock_w3

        listener = EventListener(provider)
        listener._last_block = 100

        await listener._poll_events()
        # Should return early, no dispatch calls

    @pytest.mark.asyncio
    async def test_poll_events_last_block_none(self):
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 100
        provider.get_web3.return_value = mock_w3

        listener = EventListener(provider)
        listener._last_block = None

        await listener._poll_events()
        # Should return early since last_block is None

    @pytest.mark.asyncio
    async def test_poll_events_processes_and_updates_last_block(self):
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 110
        provider.get_web3.return_value = mock_w3
        provider.get_config.return_value = MagicMock(
            has_identity_registry=False,
            has_reputation_registry=False,
            has_validation_registry=False,
        )

        listener = EventListener(provider)
        listener._last_block = 100

        await listener._poll_events()

        assert listener._last_block == 110
        provider.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_events_with_identity_registry(self):
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 110
        provider.get_web3.return_value = mock_w3

        config = MagicMock(
            has_identity_registry=True,
            identity_registry_address="0x1111111111111111111111111111111111111111",
            has_reputation_registry=False,
            has_validation_registry=False,
        )
        provider.get_config.return_value = config

        # Mock contract creation to avoid actual web3 calls
        mock_filter = MagicMock()
        mock_filter.get_all_entries.return_value = []
        mock_contract = MagicMock()
        mock_contract.events.Registered.create_filter.return_value = mock_filter
        mock_contract.events.URIUpdated.create_filter.return_value = mock_filter
        mock_contract.events.MetadataSet.create_filter.return_value = mock_filter
        mock_w3.eth.contract.return_value = mock_contract

        listener = EventListener(provider)
        listener._last_block = 100

        await listener._poll_events()
        assert listener._last_block == 110
        mock_w3.eth.contract.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_events_current_block_behind_last(self):
        """poll_events returns early if current_block <= last_block."""
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_w3.eth.block_number = 99
        provider.get_web3.return_value = mock_w3

        listener = EventListener(provider)
        listener._last_block = 100

        await listener._poll_events()
        # Should return early, no config calls
        provider.get_config.assert_not_called()


class TestBlockchainInit:
    """Tests for blockchain __init__.py lazy imports."""

    def test_get_web3_provider(self):
        from aragora.blockchain import get_web3_provider

        with patch("aragora.blockchain.provider.Web3Provider") as mock_cls:
            mock_cls.return_value = MagicMock()
            # Reset the cached class
            import aragora.blockchain as bc_mod

            bc_mod._provider_cls = None

            result = get_web3_provider()
            assert result is mock_cls.return_value

    def test_get_wallet_signer(self):
        from aragora.blockchain import get_wallet_signer

        with patch("aragora.blockchain.wallet.WalletSigner") as mock_cls:
            mock_cls.return_value = MagicMock()
            # Reset the cached class
            import aragora.blockchain as bc_mod

            bc_mod._wallet_cls = None

            result = get_wallet_signer()
            assert result is mock_cls.return_value

    def test_get_web3_provider_caches_class(self):
        from aragora.blockchain import get_web3_provider

        import aragora.blockchain as bc_mod

        bc_mod._provider_cls = None

        with patch("aragora.blockchain.provider.Web3Provider") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_web3_provider()
            get_web3_provider()
            # Module should be cached - provider imported once
            assert bc_mod._provider_cls is mock_cls

    def test_get_wallet_signer_caches_class(self):
        from aragora.blockchain import get_wallet_signer

        import aragora.blockchain as bc_mod

        bc_mod._wallet_cls = None

        with patch("aragora.blockchain.wallet.WalletSigner") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_wallet_signer()
            get_wallet_signer()
            assert bc_mod._wallet_cls is mock_cls

    def test_package_exports(self):
        """blockchain __init__.py exports expected symbols."""
        from aragora.blockchain import __all__

        expected = [
            "ChainConfig",
            "OnChainAgentIdentity",
            "ReputationFeedback",
            "ValidationRecord",
            "get_chain_config",
            "get_default_chain_config",
        ]
        for name in expected:
            assert name in __all__, f"{name} missing from __all__"

    def test_can_import_chainconfig(self):
        from aragora.blockchain import ChainConfig

        assert ChainConfig is not None

    def test_can_import_onchainagentidentity(self):
        from aragora.blockchain import OnChainAgentIdentity

        assert OnChainAgentIdentity is not None

    def test_can_import_reputationfeedback(self):
        from aragora.blockchain import ReputationFeedback

        assert ReputationFeedback is not None

    def test_can_import_validationrecord(self):
        from aragora.blockchain import ValidationRecord

        assert ValidationRecord is not None


class TestEventsModuleExports:
    """Tests for events module __all__ exports."""

    def test_all_exports(self):
        from aragora.blockchain.events import __all__

        expected = [
            "BlockchainEvent",
            "EventFilter",
            "EventListener",
            "EventType",
        ]
        for name in expected:
            assert name in __all__, f"{name} missing from __all__"

    def test_all_count(self):
        from aragora.blockchain.events import __all__

        assert len(__all__) == 4
