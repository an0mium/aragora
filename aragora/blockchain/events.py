"""
Event listener for ERC-8004 registry events.

Provides utilities for monitoring on-chain events from the Identity,
Reputation, and Validation registries. Useful for syncing on-chain
state to Aragora's Knowledge Mound.

Events monitored:
- Identity: Registered, URIUpdated, MetadataSet
- Reputation: NewFeedback, FeedbackRevoked, ResponseAppended
- Validation: ValidationRequest, ValidationResponse
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from aragora.blockchain.models import (
    OnChainAgentIdentity,
    ReputationFeedback,
    ValidationRecord,
    ValidationResponse,
)

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of ERC-8004 events."""

    # Identity Registry
    REGISTERED = "Registered"
    URI_UPDATED = "URIUpdated"
    METADATA_SET = "MetadataSet"

    # Reputation Registry
    NEW_FEEDBACK = "NewFeedback"
    FEEDBACK_REVOKED = "FeedbackRevoked"
    RESPONSE_APPENDED = "ResponseAppended"

    # Validation Registry
    VALIDATION_REQUEST = "ValidationRequest"
    VALIDATION_RESPONSE = "ValidationResponse"


@dataclass
class BlockchainEvent:
    """A parsed blockchain event from an ERC-8004 registry."""

    event_type: EventType
    block_number: int
    tx_hash: str
    log_index: int
    timestamp: datetime | None = None
    args: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    @property
    def event_id(self) -> str:
        """Unique identifier for this event."""
        return f"{self.tx_hash}:{self.log_index}"


EventHandler = Callable[[BlockchainEvent], None]
AsyncEventHandler = Callable[[BlockchainEvent], Any]


@dataclass
class EventFilter:
    """Filter criteria for event subscriptions."""

    event_types: list[EventType] | None = None
    agent_ids: list[int] | None = None
    from_block: int | str = "latest"
    to_block: int | str = "latest"

    def matches(self, event: BlockchainEvent) -> bool:
        """Check if an event matches this filter."""
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.agent_ids:
            agent_id = event.args.get("agentId")
            if agent_id is not None and agent_id not in self.agent_ids:
                return False
        return True


class EventListener:
    """Listens for and processes ERC-8004 registry events.

    Usage:
        from aragora.blockchain.provider import Web3Provider
        from aragora.blockchain.events import EventListener, EventType

        provider = Web3Provider.from_env()
        listener = EventListener(provider)

        # Register a handler
        @listener.on(EventType.REGISTERED)
        def handle_registration(event):
            print(f"New agent registered: {event.args['agentId']}")

        # Start listening
        await listener.start()
    """

    def __init__(self, provider: Any, chain_id: int | None = None) -> None:
        """Initialize the event listener.

        Args:
            provider: Web3Provider instance.
            chain_id: Chain to listen on. Defaults to provider's default.
        """
        self._provider = provider
        self._chain_id = chain_id
        self._handlers: dict[EventType, list[AsyncEventHandler]] = {}
        self._running = False
        self._poll_interval = 12.0  # seconds (roughly one block)
        self._last_block: int | None = None

    def on(self, event_type: EventType) -> Callable[[AsyncEventHandler], AsyncEventHandler]:
        """Decorator to register an event handler.

        Args:
            event_type: Type of event to handle.

        Returns:
            Decorator function.
        """

        def decorator(handler: AsyncEventHandler) -> AsyncEventHandler:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            return handler

        return decorator

    def add_handler(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Add an event handler programmatically.

        Args:
            event_type: Type of event to handle.
            handler: Handler function.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def remove_handler(self, event_type: EventType, handler: AsyncEventHandler) -> None:
        """Remove an event handler.

        Args:
            event_type: Type of event.
            handler: Handler to remove.
        """
        if event_type in self._handlers:
            self._handlers[event_type] = [h for h in self._handlers[event_type] if h != handler]

    async def start(self, from_block: int | str = "latest") -> None:
        """Start listening for events.

        Args:
            from_block: Block number to start from.
        """
        self._running = True
        w3 = self._provider.get_web3(self._chain_id)

        if from_block == "latest":
            self._last_block = w3.eth.block_number
        else:
            self._last_block = int(from_block)

        logger.info(f"Event listener started from block {self._last_block}")

        while self._running:
            try:
                await self._poll_events()
            except Exception as e:
                logger.error(f"Error polling events: {e}")
                self._provider.record_failure(self._chain_id)

            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        """Stop listening for events."""
        self._running = False
        logger.info("Event listener stopped")

    async def _poll_events(self) -> None:
        """Poll for new events since last block."""
        w3 = self._provider.get_web3(self._chain_id)
        current_block = w3.eth.block_number

        if self._last_block is None or current_block <= self._last_block:
            return

        config = self._provider.get_config(self._chain_id)
        events = []

        # Poll Identity Registry events
        if config.has_identity_registry:
            from aragora.blockchain.contracts.identity import IDENTITY_REGISTRY_ABI

            contract = w3.eth.contract(
                address=w3.to_checksum_address(config.identity_registry_address),
                abi=IDENTITY_REGISTRY_ABI,
            )
            events.extend(
                self._get_contract_events(
                    contract,
                    [EventType.REGISTERED, EventType.URI_UPDATED, EventType.METADATA_SET],
                    self._last_block + 1,
                    current_block,
                )
            )

        # Poll Reputation Registry events
        if config.has_reputation_registry:
            from aragora.blockchain.contracts.reputation import REPUTATION_REGISTRY_ABI

            contract = w3.eth.contract(
                address=w3.to_checksum_address(config.reputation_registry_address),
                abi=REPUTATION_REGISTRY_ABI,
            )
            events.extend(
                self._get_contract_events(
                    contract,
                    [
                        EventType.NEW_FEEDBACK,
                        EventType.FEEDBACK_REVOKED,
                        EventType.RESPONSE_APPENDED,
                    ],
                    self._last_block + 1,
                    current_block,
                )
            )

        # Poll Validation Registry events
        if config.has_validation_registry:
            from aragora.blockchain.contracts.validation import VALIDATION_REGISTRY_ABI

            contract = w3.eth.contract(
                address=w3.to_checksum_address(config.validation_registry_address),
                abi=VALIDATION_REGISTRY_ABI,
            )
            events.extend(
                self._get_contract_events(
                    contract,
                    [EventType.VALIDATION_REQUEST, EventType.VALIDATION_RESPONSE],
                    self._last_block + 1,
                    current_block,
                )
            )

        # Process events
        for event in sorted(events, key=lambda e: (e.block_number, e.log_index)):
            await self._dispatch_event(event)

        self._last_block = current_block
        self._provider.record_success(self._chain_id)

    def _get_contract_events(
        self,
        contract: Any,
        event_types: list[EventType],
        from_block: int,
        to_block: int,
    ) -> list[BlockchainEvent]:
        """Get events from a contract."""
        events = []
        w3 = self._provider.get_web3(self._chain_id)

        for event_type in event_types:
            try:
                event_filter = getattr(contract.events, event_type.value).create_filter(
                    fromBlock=from_block,
                    toBlock=to_block,
                )
                for log in event_filter.get_all_entries():
                    block = w3.eth.get_block(log["blockNumber"])
                    timestamp = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc)

                    events.append(
                        BlockchainEvent(
                            event_type=event_type,
                            block_number=log["blockNumber"],
                            tx_hash=log["transactionHash"].hex(),
                            log_index=log["logIndex"],
                            timestamp=timestamp,
                            args=dict(log["args"]),
                            raw=log,
                        )
                    )
            except Exception as e:
                logger.debug(f"Could not get {event_type.value} events: {e}")

        return events

    async def _dispatch_event(self, event: BlockchainEvent) -> None:
        """Dispatch an event to registered handlers."""
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")

    def parse_identity_event(self, event: BlockchainEvent) -> OnChainAgentIdentity | None:
        """Parse an identity event into an OnChainAgentIdentity."""
        if event.event_type == EventType.REGISTERED:
            return OnChainAgentIdentity(
                token_id=event.args.get("agentId", 0),
                owner=event.args.get("owner", ""),
                agent_uri=event.args.get("agentURI", ""),
                chain_id=self._provider.get_config(self._chain_id).chain_id,
                registered_at=event.timestamp,
                tx_hash=event.tx_hash,
            )
        return None

    def parse_feedback_event(self, event: BlockchainEvent) -> ReputationFeedback | None:
        """Parse a feedback event into a ReputationFeedback."""
        if event.event_type == EventType.NEW_FEEDBACK:
            return ReputationFeedback(
                agent_id=event.args.get("agentId", 0),
                client_address=event.args.get("clientAddress", ""),
                feedback_index=event.args.get("feedbackIndex", 0),
                value=event.args.get("value", 0),
                value_decimals=event.args.get("valueDecimals", 0),
                tag1=event.args.get("tag1", ""),
                tag2=event.args.get("tag2", ""),
                endpoint=event.args.get("endpoint", ""),
                feedback_uri=event.args.get("feedbackURI", ""),
                feedback_hash=event.args.get("feedbackHash", b"").hex()
                if isinstance(event.args.get("feedbackHash"), bytes)
                else event.args.get("feedbackHash", ""),
                timestamp=event.timestamp,
                tx_hash=event.tx_hash,
            )
        return None

    def parse_validation_event(self, event: BlockchainEvent) -> ValidationRecord | None:
        """Parse a validation event into a ValidationRecord."""
        if event.event_type == EventType.VALIDATION_RESPONSE:
            return ValidationRecord(
                request_hash=event.args.get("requestHash", b"").hex()
                if isinstance(event.args.get("requestHash"), bytes)
                else event.args.get("requestHash", ""),
                agent_id=event.args.get("agentId", 0),
                validator_address=event.args.get("validatorAddress", ""),
                response=ValidationResponse(event.args.get("response", 0)),
                response_uri=event.args.get("responseURI", ""),
                response_hash=event.args.get("responseHash", b"").hex()
                if isinstance(event.args.get("responseHash"), bytes)
                else event.args.get("responseHash", ""),
                tag=event.args.get("tag", ""),
                last_update=event.timestamp,
                tx_hash=event.tx_hash,
            )
        return None


__all__ = [
    "BlockchainEvent",
    "EventFilter",
    "EventListener",
    "EventType",
]
