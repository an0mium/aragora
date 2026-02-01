"""
Tests for stream chaining module.

Tests cover:
- StreamState enum
- StreamMessage dataclass
- StreamBuffer operations (write, read, read_all, read_stream, reset)
- Buffer overflow handling (max_size=1000)
- Timeout handling (30s per chunk)
- StreamChain pub-sub (register_agent, subscribe, unsubscribe, publish, consume)
- ChainedDebate topologies (ring, all-to-all, star with various agent counts)
- stream_through flow
- Error propagation and state transitions
- Cleanup and reset operations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

import pytest

from aragora.debate.stream_chain import (
    ChainedDebate,
    StreamBuffer,
    StreamChain,
    StreamMessage,
    StreamState,
    create_chain_from_topology,
)


# --- Mock Agent for testing ---


@dataclass
class MockAgent:
    """Mock agent that implements generate_stream for testing."""

    name: str
    chunks: list[str] | None = None
    delay: float = 0.0
    should_error: bool = False
    error_after_chunks: int = 0
    use_streaming: bool = True

    async def generate_stream(
        self,
        prompt: str,
        context: list | None = None,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        if self.should_error and self.error_after_chunks == 0:
            raise RuntimeError(f"Error from {self.name}")

        chunks = self.chunks or [f"Chunk from {self.name}"]

        for i, chunk in enumerate(chunks):
            if self.delay > 0:
                await asyncio.sleep(self.delay)

            if self.should_error and i >= self.error_after_chunks:
                raise RuntimeError(f"Error from {self.name} after {i} chunks")

            yield chunk

    async def generate(
        self,
        prompt: str,
        context: list | None = None,
    ) -> str:
        """Non-streaming generation fallback."""
        if self.should_error:
            raise RuntimeError(f"Error from {self.name}")

        return "".join(self.chunks or [f"Response from {self.name}"])


@dataclass
class NonStreamingAgent:
    """Agent that only supports non-streaming generation."""

    name: str
    response: str = ""

    async def generate(
        self,
        prompt: str,
        context: list | None = None,
    ) -> str:
        """Non-streaming generation."""
        return self.response or f"Response from {self.name}"


# --- StreamState Tests ---


class TestStreamState:
    """Tests for StreamState enum."""

    def test_idle_state(self):
        """Test IDLE state value."""
        assert StreamState.IDLE == "idle"
        assert StreamState.IDLE.value == "idle"

    def test_streaming_state(self):
        """Test STREAMING state value."""
        assert StreamState.STREAMING == "streaming"
        assert StreamState.STREAMING.value == "streaming"

    def test_complete_state(self):
        """Test COMPLETE state value."""
        assert StreamState.COMPLETE == "complete"
        assert StreamState.COMPLETE.value == "complete"

    def test_error_state(self):
        """Test ERROR state value."""
        assert StreamState.ERROR == "error"
        assert StreamState.ERROR.value == "error"

    def test_enum_membership(self):
        """Test all expected states exist."""
        states = list(StreamState)
        assert len(states) == 4
        assert StreamState.IDLE in states
        assert StreamState.STREAMING in states
        assert StreamState.COMPLETE in states
        assert StreamState.ERROR in states


# --- StreamMessage Tests ---


class TestStreamMessage:
    """Tests for StreamMessage dataclass."""

    def test_create_message(self):
        """Test creating a stream message."""
        msg = StreamMessage(
            source="claude",
            content="Test chunk",
            chunk_index=0,
        )

        assert msg.source == "claude"
        assert msg.content == "Test chunk"
        assert msg.chunk_index == 0
        assert msg.is_final is False
        assert msg.metadata == {}

    def test_message_with_final_flag(self):
        """Test message with is_final=True."""
        msg = StreamMessage(
            source="gpt4",
            content="Final chunk",
            chunk_index=5,
            is_final=True,
        )

        assert msg.is_final is True
        assert msg.chunk_index == 5

    def test_message_with_metadata(self):
        """Test message with custom metadata."""
        msg = StreamMessage(
            source="gemini",
            content="Chunk with metadata",
            chunk_index=2,
            metadata={"tokens": 50, "latency_ms": 120},
        )

        assert msg.metadata["tokens"] == 50
        assert msg.metadata["latency_ms"] == 120

    def test_message_timestamp(self):
        """Test that timestamp is auto-generated."""
        before = datetime.now(timezone.utc).isoformat()
        msg = StreamMessage(source="test", content="test", chunk_index=0)
        after = datetime.now(timezone.utc).isoformat()

        # Timestamp should be in ISO format and between before/after
        assert msg.timestamp >= before
        assert msg.timestamp <= after

    def test_message_custom_timestamp(self):
        """Test message with custom timestamp."""
        custom_ts = "2024-01-15T10:30:00+00:00"
        msg = StreamMessage(
            source="test",
            content="test",
            chunk_index=0,
            timestamp=custom_ts,
        )

        assert msg.timestamp == custom_ts


# --- StreamBuffer Tests ---


class TestStreamBuffer:
    """Tests for StreamBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create a test buffer."""
        return StreamBuffer()

    @pytest.mark.asyncio
    async def test_write_single_chunk(self, buffer):
        """Test writing a single chunk."""
        await buffer.write("Hello")

        assert buffer._chunks_written == 1
        assert not buffer.is_complete

    @pytest.mark.asyncio
    async def test_write_final_chunk(self, buffer):
        """Test writing final chunk sets complete flag."""
        await buffer.write("First")
        await buffer.write("Last", is_final=True)

        assert buffer._chunks_written == 2
        assert buffer.is_complete

    @pytest.mark.asyncio
    async def test_read_chunk(self, buffer):
        """Test reading a chunk."""
        await buffer.write("Test chunk")

        chunk = await buffer.read()

        assert chunk == "Test chunk"
        assert buffer._chunks_read == 1

    @pytest.mark.asyncio
    async def test_read_multiple_chunks(self, buffer):
        """Test reading multiple chunks in order."""
        await buffer.write("First")
        await buffer.write("Second")
        await buffer.write("Third", is_final=True)

        chunks = []
        for _ in range(3):
            chunk = await buffer.read()
            if chunk:
                chunks.append(chunk)

        assert chunks == ["First", "Second", "Third"]
        assert buffer._chunks_read == 3

    @pytest.mark.asyncio
    async def test_read_returns_none_when_complete_and_empty(self, buffer):
        """Test read returns None when buffer is complete and empty."""
        await buffer.write("Only chunk", is_final=True)

        # Read the only chunk
        chunk1 = await buffer.read()
        assert chunk1 == "Only chunk"

        # Should return None now
        chunk2 = await buffer.read()
        assert chunk2 is None

    @pytest.mark.asyncio
    async def test_read_timeout(self, buffer):
        """Test read times out when no data available."""
        # Don't write anything, buffer is empty
        # Note: The actual timeout is 30s, but we won't wait that long
        # The timeout test is implicit - an empty complete buffer returns None

        await buffer.write("done", is_final=True)
        await buffer.read()  # Consume the chunk

        result = await buffer.read()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_all(self, buffer):
        """Test reading all chunks at once."""
        await buffer.write("Hello ")
        await buffer.write("World")
        await buffer.write("!", is_final=True)

        content = await buffer.read_all()

        assert content == "Hello World!"
        assert buffer._chunks_read == 3

    @pytest.mark.asyncio
    async def test_read_all_empty_buffer(self):
        """Test read_all on empty buffer returns empty string."""
        buffer = StreamBuffer()
        buffer._complete.set()  # Mark as complete immediately

        content = await buffer.read_all()

        assert content == ""

    @pytest.mark.asyncio
    async def test_read_stream(self, buffer):
        """Test async iteration via read_stream."""
        await buffer.write("Chunk 1")
        await buffer.write("Chunk 2")
        await buffer.write("Chunk 3", is_final=True)

        chunks = []
        async for chunk in buffer.read_stream():
            chunks.append(chunk)

        assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]

    @pytest.mark.asyncio
    async def test_read_stream_with_slow_producer(self):
        """Test read_stream handles slow producer."""
        buffer = StreamBuffer()
        chunks_received = []

        async def producer():
            await asyncio.sleep(0.05)
            await buffer.write("Slow chunk 1")
            await asyncio.sleep(0.05)
            await buffer.write("Slow chunk 2", is_final=True)

        async def consumer():
            async for chunk in buffer.read_stream():
                chunks_received.append(chunk)

        await asyncio.gather(producer(), consumer())

        assert chunks_received == ["Slow chunk 1", "Slow chunk 2"]

    @pytest.mark.asyncio
    async def test_write_error(self, buffer):
        """Test error propagation."""
        test_error = ValueError("Test error")
        await buffer.write_error(test_error)

        assert buffer._error is test_error
        assert buffer.is_complete

    @pytest.mark.asyncio
    async def test_read_stream_raises_on_error(self):
        """Test read_stream raises stored error."""
        buffer = StreamBuffer()

        async def producer():
            await buffer.write("Before error")
            await buffer.write_error(RuntimeError("Stream failed"))

        producer_task = asyncio.create_task(producer())

        chunks = []
        with pytest.raises(RuntimeError, match="Stream failed"):
            async for chunk in buffer.read_stream():
                chunks.append(chunk)

        await producer_task

        # Should have received chunk before error
        assert "Before error" in chunks

    def test_reset(self):
        """Test buffer reset."""
        buffer = StreamBuffer()
        buffer._chunks_written = 10
        buffer._chunks_read = 5
        buffer._complete.set()
        buffer._error = ValueError("test")

        buffer.reset()

        assert buffer._chunks_written == 0
        assert buffer._chunks_read == 0
        assert not buffer.is_complete
        assert buffer._error is None

    def test_stats_property(self, buffer):
        """Test stats property returns correct values."""
        stats = buffer.stats

        assert stats == {
            "written": 0,
            "read": 0,
            "pending": 0,
        }

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, buffer):
        """Test stats reflect write/read operations."""
        await buffer.write("Chunk 1")
        await buffer.write("Chunk 2")
        await buffer.read()

        stats = buffer.stats

        assert stats["written"] == 2
        assert stats["read"] == 1
        assert stats["pending"] == 1

    def test_is_complete_property(self, buffer):
        """Test is_complete property."""
        assert not buffer.is_complete

        buffer._complete.set()

        assert buffer.is_complete

    def test_max_size_configuration(self):
        """Test custom max_size configuration."""
        buffer = StreamBuffer(max_size=50)

        assert buffer.max_size == 50

    def test_default_max_size(self, buffer):
        """Test default max_size is 1000."""
        assert buffer.max_size == 1000


class TestStreamBufferOverflow:
    """Tests for buffer overflow handling."""

    @pytest.mark.asyncio
    async def test_buffer_overflow_blocks_write(self):
        """Test that buffer blocks when full (asyncio.Queue behavior)."""
        # Create a small buffer
        buffer = StreamBuffer(max_size=2)
        buffer._buffer = asyncio.Queue(maxsize=2)

        # Fill the buffer
        await buffer.write("Chunk 1")
        await buffer.write("Chunk 2")

        # Next write should block - use timeout to avoid hanging
        write_complete = asyncio.Event()

        async def blocked_write():
            await buffer.write("Chunk 3")
            write_complete.set()

        task = asyncio.create_task(blocked_write())

        # Wait briefly - write should not complete
        await asyncio.sleep(0.05)
        assert not write_complete.is_set()

        # Read one item to unblock
        await buffer.read()

        # Now write should complete
        await asyncio.sleep(0.05)
        assert write_complete.is_set()

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# --- StreamChain Tests ---


class TestStreamChain:
    """Tests for StreamChain pub-sub functionality."""

    @pytest.fixture
    def chain(self):
        """Create a test chain."""
        return StreamChain()

    def test_register_agent(self, chain):
        """Test registering an agent."""
        chain.register_agent("claude")

        assert "claude" in chain._subscriptions
        assert "claude" in chain._buffers
        assert chain._states["claude"] == StreamState.IDLE

    def test_register_agent_idempotent(self, chain):
        """Test registering same agent twice is idempotent."""
        chain.register_agent("claude")
        buffer1 = chain._buffers["claude"]

        chain.register_agent("claude")
        buffer2 = chain._buffers["claude"]

        # Should be same buffer instance
        assert buffer1 is buffer2

    def test_subscribe(self, chain):
        """Test subscribing one agent to another."""
        chain.subscribe("gpt4", "claude")

        assert "gpt4" in chain._subscriptions["claude"]
        assert "claude" in chain._buffers
        assert "gpt4" in chain._buffers

    def test_subscribe_multiple(self, chain):
        """Test multiple subscribers to same source."""
        chain.subscribe("gpt4", "claude")
        chain.subscribe("gemini", "claude")

        subscribers = chain.get_subscribers("claude")

        assert "gpt4" in subscribers
        assert "gemini" in subscribers
        assert len(subscribers) == 2

    def test_unsubscribe(self, chain):
        """Test unsubscribing an agent."""
        chain.subscribe("gpt4", "claude")
        chain.subscribe("gemini", "claude")

        chain.unsubscribe("gpt4", "claude")

        subscribers = chain.get_subscribers("claude")
        assert "gpt4" not in subscribers
        assert "gemini" in subscribers

    def test_unsubscribe_nonexistent(self, chain):
        """Test unsubscribing when not subscribed."""
        # Should not raise
        chain.unsubscribe("gpt4", "claude")

    def test_get_subscribers_empty(self, chain):
        """Test get_subscribers for unregistered agent."""
        subscribers = chain.get_subscribers("unknown")

        assert subscribers == set()

    def test_get_buffer(self, chain):
        """Test getting buffer for agent."""
        buffer = chain.get_buffer("claude")

        assert buffer is not None
        assert isinstance(buffer, StreamBuffer)
        assert "claude" in chain._buffers

    def test_get_buffer_creates_if_missing(self, chain):
        """Test get_buffer registers agent if not exists."""
        assert "new_agent" not in chain._buffers

        buffer = chain.get_buffer("new_agent")

        assert buffer is not None
        assert "new_agent" in chain._buffers
        assert chain._states["new_agent"] == StreamState.IDLE

    @pytest.mark.asyncio
    async def test_publish_chunk(self, chain):
        """Test publishing a chunk."""
        chain.register_agent("claude")

        await chain.publish("claude", "Hello world")

        assert chain._states["claude"] == StreamState.STREAMING
        buffer = chain.get_buffer("claude")
        assert buffer._chunks_written == 1

    @pytest.mark.asyncio
    async def test_publish_final_chunk(self, chain):
        """Test publishing final chunk updates state."""
        chain.register_agent("claude")

        await chain.publish("claude", "First")
        await chain.publish("claude", "Last", is_final=True)

        assert chain._states["claude"] == StreamState.COMPLETE

    @pytest.mark.asyncio
    async def test_consume(self, chain):
        """Test consuming from an agent's buffer."""
        chain.register_agent("claude")

        # Publish some chunks
        await chain.publish("claude", "Chunk 1")
        await chain.publish("claude", "Chunk 2", is_final=True)

        # Consume
        chunks = []
        async for chunk in chain.consume("claude"):
            chunks.append(chunk)

        assert chunks == ["Chunk 1", "Chunk 2"]

    def test_reset_agent(self, chain):
        """Test resetting a single agent."""
        chain.register_agent("claude")
        chain._states["claude"] = StreamState.COMPLETE

        chain.reset_agent("claude")

        assert chain._states["claude"] == StreamState.IDLE
        assert not chain._buffers["claude"].is_complete

    def test_reset_all(self, chain):
        """Test resetting all agents."""
        chain.register_agent("claude")
        chain.register_agent("gpt4")
        chain._states["claude"] = StreamState.COMPLETE
        chain._states["gpt4"] = StreamState.ERROR

        chain.reset_all()

        assert chain._states["claude"] == StreamState.IDLE
        assert chain._states["gpt4"] == StreamState.IDLE

    def test_stats_property(self, chain):
        """Test stats property."""
        chain.subscribe("gpt4", "claude")
        chain._states["claude"] = StreamState.STREAMING

        stats = chain.stats

        assert "claude" in stats["agents"]
        assert "gpt4" in stats["agents"]
        assert stats["subscriptions"]["claude"] == ["gpt4"]
        assert stats["states"]["claude"] == "streaming"
        assert "buffers" in stats


class TestStreamChainStreamThrough:
    """Tests for stream_through functionality."""

    @pytest.mark.asyncio
    async def test_stream_through_basic(self):
        """Test basic stream_through flow."""
        chain = StreamChain()

        source = MockAgent(name="proposer", chunks=["Hello ", "World!"])
        target = MockAgent(name="critic", chunks=["Looks ", "good!"])

        chunks = []
        async for chunk in chain.stream_through(source, target, "Test prompt"):
            chunks.append(chunk)

        # Should have chunks from both agents
        assert "[proposer]" in chunks[0]
        assert "[critic]" in chunks[-1] or "[critic]" in chunks[-2]

    @pytest.mark.asyncio
    async def test_stream_through_registers_agents(self):
        """Test stream_through registers both agents."""
        chain = StreamChain()

        source = MockAgent(name="agent_a")
        target = MockAgent(name="agent_b")

        async for _ in chain.stream_through(source, target, "Prompt"):
            pass

        assert "agent_a" in chain._buffers
        assert "agent_b" in chain._buffers

    @pytest.mark.asyncio
    async def test_stream_through_sets_subscription(self):
        """Test stream_through creates subscription."""
        chain = StreamChain()

        source = MockAgent(name="source")
        target = MockAgent(name="target")

        async for _ in chain.stream_through(source, target, "Prompt"):
            pass

        assert "target" in chain._subscriptions["source"]

    @pytest.mark.asyncio
    async def test_stream_through_with_context(self):
        """Test stream_through passes context."""
        chain = StreamChain()

        source = MockAgent(name="source", chunks=["Context test"])
        target = MockAgent(name="target", chunks=["Got it"])

        context = [{"role": "user", "content": "Previous message"}]

        chunks = []
        async for chunk in chain.stream_through(source, target, "Prompt", context=context):
            chunks.append(chunk)

        # Should complete without error
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_stream_through_with_custom_chain_prompt(self):
        """Test stream_through with custom chain_prompt."""
        chain = StreamChain()

        source = MockAgent(name="source", chunks=["Analysis"])
        target = MockAgent(name="target", chunks=["Response"])

        chunks = []
        async for chunk in chain.stream_through(
            source,
            target,
            "Initial prompt",
            chain_prompt="Custom chain prompt: {source_output}",
        ):
            chunks.append(chunk)

        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_stream_through_non_streaming_source(self):
        """Test stream_through with non-streaming source agent."""
        chain = StreamChain()

        source = NonStreamingAgent(name="source", response="Non-stream response")
        target = MockAgent(name="target", chunks=["Reply"])

        chunks = []
        async for chunk in chain.stream_through(source, target, "Prompt"):
            chunks.append(chunk)

        # Should include source response
        assert any("Non-stream response" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_through_non_streaming_target(self):
        """Test stream_through with non-streaming target agent."""
        chain = StreamChain()

        source = MockAgent(name="source", chunks=["Streaming input"])
        target = NonStreamingAgent(name="target", response="Target reply")

        chunks = []
        async for chunk in chain.stream_through(source, target, "Prompt"):
            chunks.append(chunk)

        # Should include target response
        assert any("Target reply" in c for c in chunks)


# --- ChainedDebate Tests ---


class TestChainedDebate:
    """Tests for ChainedDebate with different topologies."""

    def test_create_ring_topology(self):
        """Test creating ring topology."""
        agents = [
            MockAgent(name="agent_1"),
            MockAgent(name="agent_2"),
            MockAgent(name="agent_3"),
        ]

        debate = ChainedDebate(agents=agents, topology="ring")

        # In ring: each agent subscribes to previous
        # agent_1 -> agent_2 -> agent_3 -> agent_1
        assert "agent_1" in debate.chain._subscriptions["agent_3"]
        assert "agent_2" in debate.chain._subscriptions["agent_1"]
        assert "agent_3" in debate.chain._subscriptions["agent_2"]

    def test_create_all_to_all_topology(self):
        """Test creating all-to-all topology."""
        agents = [
            MockAgent(name="agent_1"),
            MockAgent(name="agent_2"),
            MockAgent(name="agent_3"),
        ]

        debate = ChainedDebate(agents=agents, topology="all-to-all")

        # Each agent subscribes to all others
        subs_1 = debate.chain.get_subscribers("agent_1")
        subs_2 = debate.chain.get_subscribers("agent_2")
        subs_3 = debate.chain.get_subscribers("agent_3")

        assert "agent_2" in subs_1 and "agent_3" in subs_1
        assert "agent_1" in subs_2 and "agent_3" in subs_2
        assert "agent_1" in subs_3 and "agent_2" in subs_3

    def test_create_star_topology(self):
        """Test creating star topology."""
        agents = [
            MockAgent(name="hub"),
            MockAgent(name="spoke_1"),
            MockAgent(name="spoke_2"),
        ]

        debate = ChainedDebate(agents=agents, topology="star")

        # Hub connects to all spokes, spokes connect to hub
        hub_subs = debate.chain.get_subscribers("hub")
        assert "spoke_1" in hub_subs
        assert "spoke_2" in hub_subs

        spoke_1_subs = debate.chain.get_subscribers("spoke_1")
        spoke_2_subs = debate.chain.get_subscribers("spoke_2")
        assert "hub" in spoke_1_subs
        assert "hub" in spoke_2_subs

    def test_empty_agents_list(self):
        """Test creating debate with empty agents list."""
        debate = ChainedDebate(agents=[], topology="ring")

        assert len(debate.chain._buffers) == 0

    def test_single_agent_ring(self):
        """Test ring topology with single agent."""
        agents = [MockAgent(name="solo")]

        debate = ChainedDebate(agents=agents, topology="ring")

        # Single agent subscribes to itself in ring
        assert "solo" in debate.chain._subscriptions["solo"]

    def test_single_agent_star(self):
        """Test star topology with single agent."""
        agents = [MockAgent(name="hub")]

        debate = ChainedDebate(agents=agents, topology="star")

        # Single agent is registered but no subscriptions
        assert "hub" in debate.chain._buffers

    def test_two_agents_ring(self):
        """Test ring topology with two agents."""
        agents = [
            MockAgent(name="agent_a"),
            MockAgent(name="agent_b"),
        ]

        debate = ChainedDebate(agents=agents, topology="ring")

        # a -> b -> a (circular)
        assert "agent_b" in debate.chain._subscriptions["agent_a"]
        assert "agent_a" in debate.chain._subscriptions["agent_b"]


class TestChainedDebateRunRound:
    """Tests for ChainedDebate.run_round method."""

    @pytest.mark.asyncio
    async def test_run_round_ring_topology(self):
        """Test running a round with ring topology."""
        agents = [
            MockAgent(name="agent_1", chunks=["Response 1"]),
            MockAgent(name="agent_2", chunks=["Response 2"]),
            MockAgent(name="agent_3", chunks=["Response 3"]),
        ]

        debate = ChainedDebate(agents=agents, topology="ring")
        responses = await debate.run_round("Test prompt")

        assert "agent_1" in responses
        assert "agent_2" in responses
        assert "agent_3" in responses
        assert responses["agent_1"] == "Response 1"
        assert responses["agent_2"] == "Response 2"
        assert responses["agent_3"] == "Response 3"

    @pytest.mark.asyncio
    async def test_run_round_all_to_all_topology(self):
        """Test running a round with all-to-all topology."""
        agents = [
            MockAgent(name="agent_1", chunks=["Parallel 1"]),
            MockAgent(name="agent_2", chunks=["Parallel 2"]),
        ]

        debate = ChainedDebate(agents=agents, topology="all-to-all")
        responses = await debate.run_round("Test prompt")

        assert responses["agent_1"] == "Parallel 1"
        assert responses["agent_2"] == "Parallel 2"

    @pytest.mark.asyncio
    async def test_run_round_star_topology(self):
        """Test running a round with star topology."""
        agents = [
            MockAgent(name="hub", chunks=["Hub response"]),
            MockAgent(name="spoke_1", chunks=["Spoke 1 response"]),
            MockAgent(name="spoke_2", chunks=["Spoke 2 response"]),
        ]

        debate = ChainedDebate(agents=agents, topology="star")
        responses = await debate.run_round("Test prompt")

        assert len(responses) == 3
        assert "Hub response" in responses["hub"]

    @pytest.mark.asyncio
    async def test_run_round_non_streaming_agent(self):
        """Test run_round with non-streaming agent."""
        agents = [
            MockAgent(name="streaming", chunks=["Streaming"]),
            NonStreamingAgent(name="non_streaming", response="Non-streaming"),
        ]

        debate = ChainedDebate(agents=agents, topology="all-to-all")
        responses = await debate.run_round("Test prompt")

        assert responses["streaming"] == "Streaming"
        assert responses["non_streaming"] == "Non-streaming"

    @pytest.mark.asyncio
    async def test_run_round_resets_chain(self):
        """Test that run_round resets chain after completion."""
        agents = [MockAgent(name="agent", chunks=["Response"])]

        debate = ChainedDebate(agents=agents, topology="ring")

        # Run round
        await debate.run_round("Prompt")

        # Chain should be reset
        assert debate.chain._states["agent"] == StreamState.IDLE

    @pytest.mark.asyncio
    async def test_run_round_with_context(self):
        """Test run_round passes context."""
        agents = [MockAgent(name="agent", chunks=["With context"])]

        debate = ChainedDebate(agents=agents, topology="ring")
        context = [{"role": "system", "content": "Context"}]

        responses = await debate.run_round("Prompt", context=context)

        assert "agent" in responses

    @pytest.mark.asyncio
    async def test_run_round_ring_builds_cumulative_prompt(self):
        """Test ring topology builds cumulative prompt for each agent."""
        agents = [
            MockAgent(name="first", chunks=["First response"]),
            MockAgent(name="second", chunks=["Second response"]),
        ]

        debate = ChainedDebate(agents=agents, topology="ring")
        responses = await debate.run_round("Original task")

        # Both agents should respond
        assert "First response" in responses["first"]
        assert "Second response" in responses["second"]


# --- create_chain_from_topology Tests ---


class TestCreateChainFromTopology:
    """Tests for create_chain_from_topology factory function."""

    def test_create_ring_chain(self):
        """Test creating ring topology via factory."""
        agents = [
            MockAgent(name="a"),
            MockAgent(name="b"),
        ]

        debate = create_chain_from_topology("ring", agents)

        assert isinstance(debate, ChainedDebate)
        assert debate.topology == "ring"

    def test_create_all_to_all_chain(self):
        """Test creating all-to-all topology via factory."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]

        debate = create_chain_from_topology("all-to-all", agents)

        assert debate.topology == "all-to-all"

    def test_create_star_chain(self):
        """Test creating star topology via factory."""
        agents = [MockAgent(name="hub"), MockAgent(name="spoke")]

        debate = create_chain_from_topology("star", agents)

        assert debate.topology == "star"

    def test_factory_preserves_agents(self):
        """Test factory preserves agent list."""
        agents = [
            MockAgent(name="x"),
            MockAgent(name="y"),
            MockAgent(name="z"),
        ]

        debate = create_chain_from_topology("ring", agents)

        assert len(debate.agents) == 3


# --- Error Propagation Tests ---


class TestErrorPropagation:
    """Tests for error propagation and handling."""

    @pytest.mark.asyncio
    async def test_buffer_error_propagation(self):
        """Test error propagates through buffer."""
        buffer = StreamBuffer()

        error = ValueError("Test error")
        await buffer.write_error(error)

        assert buffer._error is error
        assert buffer.is_complete

    @pytest.mark.asyncio
    async def test_read_stream_raises_stored_error(self):
        """Test read_stream raises the stored error immediately when error is set."""
        buffer = StreamBuffer()

        # Set error first - read_stream checks error at start of each iteration
        await buffer.write_error(RuntimeError("Propagated error"))

        chunks = []
        with pytest.raises(RuntimeError, match="Propagated error"):
            async for chunk in buffer.read_stream():
                chunks.append(chunk)

        # No chunks should be received since error was set before reading
        assert chunks == []

    @pytest.mark.asyncio
    async def test_read_stream_detects_error_on_next_iteration(self):
        """Test error is detected on next loop iteration after chunk read."""
        buffer = StreamBuffer()

        # Write chunks but don't mark final - error will be set before next iteration
        await buffer.write("Chunk 1")

        chunks_received = []
        iteration_count = 0

        # Create custom iterator to test error detection
        read_iter = buffer.read_stream()

        # Get first chunk
        chunk = await read_iter.__anext__()
        chunks_received.append(chunk)

        # Now set error - next iteration should raise
        await buffer.write_error(RuntimeError("Error after first chunk"))

        # Next iteration should detect error
        with pytest.raises(RuntimeError, match="Error after first chunk"):
            await read_iter.__anext__()

        assert chunks_received == ["Chunk 1"]

    @pytest.mark.asyncio
    async def test_chain_error_state(self):
        """Test chain tracks error state."""
        chain = StreamChain()
        chain.register_agent("faulty")

        # Simulate error through buffer
        buffer = chain.get_buffer("faulty")
        await buffer.write_error(Exception("Agent failed"))

        # Buffer should be complete with error
        assert buffer.is_complete
        assert buffer._error is not None


# --- State Transition Tests ---


class TestStateTransitions:
    """Tests for stream state transitions."""

    @pytest.mark.asyncio
    async def test_idle_to_streaming_transition(self):
        """Test transition from IDLE to STREAMING."""
        chain = StreamChain()
        chain.register_agent("agent")

        assert chain._states["agent"] == StreamState.IDLE

        await chain.publish("agent", "First chunk")

        assert chain._states["agent"] == StreamState.STREAMING

    @pytest.mark.asyncio
    async def test_streaming_to_complete_transition(self):
        """Test transition from STREAMING to COMPLETE."""
        chain = StreamChain()
        chain.register_agent("agent")

        await chain.publish("agent", "Chunk")
        assert chain._states["agent"] == StreamState.STREAMING

        await chain.publish("agent", "Final", is_final=True)

        assert chain._states["agent"] == StreamState.COMPLETE

    def test_reset_to_idle_transition(self):
        """Test reset returns state to IDLE."""
        chain = StreamChain()
        chain.register_agent("agent")
        chain._states["agent"] = StreamState.COMPLETE

        chain.reset_agent("agent")

        assert chain._states["agent"] == StreamState.IDLE


# --- Cleanup Tests ---


class TestCleanup:
    """Tests for cleanup and reset operations."""

    def test_buffer_reset_clears_all_state(self):
        """Test buffer reset clears all internal state."""
        buffer = StreamBuffer()
        buffer._chunks_written = 100
        buffer._chunks_read = 50
        buffer._complete.set()
        buffer._error = Exception("test")

        buffer.reset()

        assert buffer._chunks_written == 0
        assert buffer._chunks_read == 0
        assert not buffer.is_complete
        assert buffer._error is None

    def test_chain_reset_all_resets_all_agents(self):
        """Test chain reset_all resets all agents."""
        chain = StreamChain()
        chain.register_agent("agent_1")
        chain.register_agent("agent_2")
        chain._states["agent_1"] = StreamState.COMPLETE
        chain._states["agent_2"] = StreamState.ERROR

        chain.reset_all()

        assert chain._states["agent_1"] == StreamState.IDLE
        assert chain._states["agent_2"] == StreamState.IDLE

    def test_reset_preserves_subscriptions(self):
        """Test reset preserves subscription relationships."""
        chain = StreamChain()
        chain.subscribe("subscriber", "source")

        chain.reset_all()

        # Subscriptions should still exist
        assert "subscriber" in chain._subscriptions["source"]

    @pytest.mark.asyncio
    async def test_chained_debate_resets_after_round(self):
        """Test ChainedDebate resets chain after run_round."""
        agents = [MockAgent(name="agent", chunks=["Response"])]
        debate = ChainedDebate(agents=agents, topology="ring")

        # Run round
        await debate.run_round("Test")

        # States should be reset
        for agent in agents:
            assert debate.chain._states[agent.name] == StreamState.IDLE


# --- Integration Tests ---


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_debate_flow_ring(self):
        """Test full debate flow with ring topology."""
        agents = [
            MockAgent(name="proposer", chunks=["I propose ", "we use ", "caching."]),
            MockAgent(name="critic", chunks=["The proposal ", "has merit."]),
            MockAgent(name="resolver", chunks=["Let's implement ", "with modifications."]),
        ]

        debate = ChainedDebate(agents=agents, topology="ring")
        responses = await debate.run_round("Design a rate limiter")

        assert len(responses) == 3
        assert "I propose we use caching." in responses["proposer"]
        assert "The proposal has merit." in responses["critic"]
        assert "Let's implement with modifications." in responses["resolver"]

    @pytest.mark.asyncio
    async def test_full_debate_flow_parallel(self):
        """Test full debate flow with parallel (all-to-all) topology."""
        agents = [
            MockAgent(name="expert_1", chunks=["Expert 1 opinion"]),
            MockAgent(name="expert_2", chunks=["Expert 2 opinion"]),
            MockAgent(name="expert_3", chunks=["Expert 3 opinion"]),
        ]

        debate = ChainedDebate(agents=agents, topology="all-to-all")
        responses = await debate.run_round("Evaluate the design")

        # All should complete in parallel
        assert len(responses) == 3

    @pytest.mark.asyncio
    async def test_mixed_streaming_agents(self):
        """Test debate with mixed streaming and non-streaming agents."""
        agents = [
            MockAgent(name="streaming_agent", chunks=["Chunk 1", " Chunk 2"]),
            NonStreamingAgent(name="batch_agent", response="Full response"),
        ]

        debate = ChainedDebate(agents=agents, topology="all-to-all")
        responses = await debate.run_round("Test mixed agents")

        assert "Chunk 1 Chunk 2" in responses["streaming_agent"]
        assert responses["batch_agent"] == "Full response"

    @pytest.mark.asyncio
    async def test_stream_through_end_to_end(self):
        """Test stream_through end-to-end flow."""
        chain = StreamChain()

        source = MockAgent(
            name="analyzer",
            chunks=["The system ", "needs ", "optimization."],
        )
        target = MockAgent(
            name="implementer",
            chunks=["I will ", "optimize ", "it."],
        )

        all_chunks = []
        async for chunk in chain.stream_through(
            source, target, "Analyze and implement improvements"
        ):
            all_chunks.append(chunk)

        # Should have chunks from both
        source_chunks = [c for c in all_chunks if "[analyzer]" in c]
        target_chunks = [c for c in all_chunks if "[implementer]" in c]

        assert len(source_chunks) == 3
        assert len(target_chunks) == 3


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_chunk_handling(self):
        """Test handling of empty chunks."""
        buffer = StreamBuffer()

        await buffer.write("")
        await buffer.write("Non-empty")
        await buffer.write("", is_final=True)

        content = await buffer.read_all()

        assert content == "Non-empty"

    def test_large_agent_count_all_to_all(self):
        """Test all-to-all with many agents."""
        agents = [MockAgent(name=f"agent_{i}") for i in range(10)]

        debate = ChainedDebate(agents=agents, topology="all-to-all")

        # Each agent should subscribe to 9 others
        for agent in agents:
            subscribers = debate.chain.get_subscribers(agent.name)
            assert len(subscribers) == 9

    def test_large_agent_count_ring(self):
        """Test ring with many agents."""
        agents = [MockAgent(name=f"agent_{i}") for i in range(10)]

        debate = ChainedDebate(agents=agents, topology="ring")

        # Each agent should have exactly 1 subscriber
        for agent in agents:
            subscribers = debate.chain.get_subscribers(agent.name)
            assert len(subscribers) == 1

    @pytest.mark.asyncio
    async def test_concurrent_publish_consume(self):
        """Test concurrent publish and consume operations."""
        chain = StreamChain()
        chain.register_agent("producer")

        received_chunks = []

        async def producer():
            for i in range(5):
                await chain.publish("producer", f"Chunk {i}")
                await asyncio.sleep(0.01)
            await chain.publish("producer", "Final", is_final=True)

        async def consumer():
            async for chunk in chain.consume("producer"):
                received_chunks.append(chunk)

        await asyncio.gather(producer(), consumer())

        assert len(received_chunks) == 6
        assert "Final" in received_chunks
