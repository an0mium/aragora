"""
Tests for Knowledge Mound context injection into PromptBuilder.

Verifies that KM content is wired as a structured prompt section
(parallel to supermemory) rather than appended to env.context,
giving agents a dedicated "Organizational Knowledge" header.

Tests cover:
- PromptBuilder accepts knowledge_context parameter
- set_knowledge_context updates internal state
- build_proposal_prompt includes KM section when set
- build_proposal_prompt omits KM section when empty
- build_revision_prompt includes KM section
- KM section has proper header ("Organizational Knowledge")
- Backward compatibility (env.context still works)
- Item IDs tracking
- ContextInitializer integration with PromptBuilder
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.debate.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "claude", role: str = "proposer", stance: str | None = None):
        self.name = name
        self.role = role
        self.model = "claude-3-opus"
        if stance:
            self.stance = stance


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, task: str = "Test task", context: str | None = None):
        self.task = task
        self.context = context


class MockProtocol:
    """Mock debate protocol for testing."""

    def __init__(self):
        self.rounds = 3
        self.asymmetric_stances = False
        self.agreement_intensity = None
        self.enable_trending_injection = False
        self.trending_injection_max_topics = 3
        self.trending_relevance_filter = True
        self.language = None
        self.require_evidence = False
        self.require_uncertainty = False
        self.consensus = "majority"
        self.enable_privacy_anonymization = False

    def get_round_phase(self, round_number: int):
        return None


class MockCritique:
    """Mock critique for testing."""

    def __init__(self, agent: str = "critic", issues: list | None = None):
        self.agent = agent
        self.issues = issues or ["Issue 1"]

    def to_prompt(self) -> str:
        return f"[{self.agent}]: {', '.join(self.issues)}"


@pytest.fixture
def mock_protocol():
    return MockProtocol()


@pytest.fixture
def mock_env():
    return MockEnvironment(task="Discuss the best API design approach")


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def builder(mock_protocol, mock_env):
    """Create a basic PromptBuilder for testing."""
    return PromptBuilder(protocol=mock_protocol, env=mock_env)


# ---------------------------------------------------------------------------
# 1. PromptBuilder accepts knowledge_context parameter
# ---------------------------------------------------------------------------

class TestPromptBuilderInit:
    """Test PromptBuilder initialization with knowledge_context."""

    def test_init_without_knowledge_context(self, mock_protocol, mock_env):
        """PromptBuilder initializes with empty knowledge context by default."""
        pb = PromptBuilder(protocol=mock_protocol, env=mock_env)
        assert pb._knowledge_context == ""
        assert pb._km_item_ids == []

    def test_init_with_knowledge_context(self, mock_protocol, mock_env):
        """PromptBuilder stores provided knowledge context."""
        km_text = "Rate limiters should use token bucket algorithm."
        pb = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            knowledge_context=km_text,
        )
        assert pb._knowledge_context == km_text
        assert pb._km_item_ids == []

    def test_init_with_none_knowledge_context(self, mock_protocol, mock_env):
        """Passing None normalizes to empty string."""
        pb = PromptBuilder(
            protocol=mock_protocol,
            env=mock_env,
            knowledge_context=None,
        )
        assert pb._knowledge_context == ""


# ---------------------------------------------------------------------------
# 2. set_knowledge_context updates internal state
# ---------------------------------------------------------------------------

class TestSetKnowledgeContext:
    """Test set_knowledge_context method."""

    def test_set_knowledge_context_basic(self, builder):
        """Setting context updates the internal attribute."""
        builder.set_knowledge_context("New organizational knowledge.")
        assert builder._knowledge_context == "New organizational knowledge."

    def test_set_knowledge_context_with_item_ids(self, builder):
        """Item IDs are stored alongside the context."""
        builder.set_knowledge_context(
            "API best practices from past debates.",
            item_ids=["km-001", "km-002", "km-003"],
        )
        assert builder._knowledge_context == "API best practices from past debates."
        assert builder._km_item_ids == ["km-001", "km-002", "km-003"]

    def test_set_knowledge_context_replaces_previous(self, builder):
        """Calling set twice replaces the first value."""
        builder.set_knowledge_context("First context.", item_ids=["id-1"])
        builder.set_knowledge_context("Second context.", item_ids=["id-2"])
        assert builder._knowledge_context == "Second context."
        assert builder._km_item_ids == ["id-2"]

    def test_set_knowledge_context_empty_clears(self, builder):
        """Setting empty string clears the context."""
        builder.set_knowledge_context("Something")
        builder.set_knowledge_context("")
        assert builder._knowledge_context == ""

    def test_set_knowledge_context_none_clears(self, builder):
        """Setting None normalizes to empty string."""
        builder.set_knowledge_context("Something")
        builder.set_knowledge_context(None)
        assert builder._knowledge_context == ""

    def test_set_knowledge_context_preserves_ids_when_none(self, builder):
        """When item_ids is None, existing IDs are preserved."""
        builder.set_knowledge_context("First", item_ids=["km-001"])
        builder.set_knowledge_context("Second")  # item_ids defaults to None
        assert builder._km_item_ids == ["km-001"]

    def test_set_knowledge_context_empty_ids_list(self, builder):
        """Explicit empty list replaces existing IDs."""
        builder.set_knowledge_context("First", item_ids=["km-001"])
        builder.set_knowledge_context("Second", item_ids=[])
        assert builder._km_item_ids == []


# ---------------------------------------------------------------------------
# 3. get_knowledge_mound_context returns stored content
# ---------------------------------------------------------------------------

class TestGetKnowledgeMoundContext:
    """Test get_knowledge_mound_context method."""

    def test_get_empty_by_default(self, builder):
        """Returns empty string when nothing is set."""
        assert builder.get_knowledge_mound_context() == ""

    def test_get_returns_set_context(self, builder):
        """Returns the context that was set."""
        builder.set_knowledge_context("Institutional memory content.")
        assert builder.get_knowledge_mound_context() == "Institutional memory content."


# ---------------------------------------------------------------------------
# 4. build_proposal_prompt includes KM section when set
# ---------------------------------------------------------------------------

class TestProposalPromptKMSection:
    """Test KM section in proposal prompts."""

    def test_proposal_includes_km_section(self, builder, mock_agent):
        """Proposal prompt contains the KM section with header."""
        builder.set_knowledge_context(
            "Past debates concluded: always use pagination for list endpoints."
        )
        prompt = builder.build_proposal_prompt(mock_agent)
        assert "## Organizational Knowledge" in prompt
        assert "always use pagination for list endpoints" in prompt

    def test_proposal_omits_km_section_when_empty(self, builder, mock_agent):
        """Proposal prompt does NOT include KM header when context is empty."""
        prompt = builder.build_proposal_prompt(mock_agent)
        assert "## Organizational Knowledge" not in prompt

    def test_proposal_km_section_positioned_in_context_block(self, builder, mock_agent):
        """KM section appears within the context block (before the Task line)."""
        builder.set_knowledge_context("Use rate limiting on all endpoints.")
        prompt = builder.build_proposal_prompt(mock_agent)

        # The KM section should appear before "Task:" in the prompt
        km_pos = prompt.find("## Organizational Knowledge")
        task_pos = prompt.find("Task:")
        assert km_pos < task_pos, "KM section should appear before Task"

    def test_proposal_km_alongside_env_context(self, builder, mock_agent, mock_env):
        """KM section and env.context both appear in the prompt."""
        mock_env.context = "We are building a REST API"
        builder.set_knowledge_context("Previous decision: use JWT for auth.")
        prompt = builder.build_proposal_prompt(mock_agent)

        assert "## Organizational Knowledge" in prompt
        assert "Previous decision: use JWT for auth." in prompt
        assert "We are building a REST API" in prompt


# ---------------------------------------------------------------------------
# 5. build_revision_prompt includes KM section
# ---------------------------------------------------------------------------

class TestRevisionPromptKMSection:
    """Test KM section in revision prompts."""

    def test_revision_includes_km_section(self, builder, mock_agent):
        """Revision prompt contains the KM section with header."""
        builder.set_knowledge_context("Prefer async patterns for I/O-bound work.")
        critiques = [MockCritique(agent="gpt-4", issues=["Missing error handling"])]
        prompt = builder.build_revision_prompt(
            mock_agent,
            original="My original proposal about API design.",
            critiques=critiques,
        )
        assert "## Organizational Knowledge" in prompt
        assert "Prefer async patterns for I/O-bound work" in prompt

    def test_revision_omits_km_section_when_empty(self, builder, mock_agent):
        """Revision prompt does NOT include KM header when context is empty."""
        critiques = [MockCritique()]
        prompt = builder.build_revision_prompt(
            mock_agent,
            original="Original proposal.",
            critiques=critiques,
        )
        assert "## Organizational Knowledge" not in prompt


# ---------------------------------------------------------------------------
# 6. KM section header format
# ---------------------------------------------------------------------------

class TestKMSectionHeader:
    """Test the Organizational Knowledge header formatting."""

    def test_header_is_markdown_h2(self, builder, mock_agent):
        """The KM section uses a markdown H2 header."""
        builder.set_knowledge_context("Test content")
        prompt = builder.build_proposal_prompt(mock_agent)
        assert "## Organizational Knowledge\nTest content" in prompt

    def test_multiline_km_content_preserved(self, builder, mock_agent):
        """Multi-line KM content is preserved in the prompt."""
        km_content = (
            "1. Always validate input\n"
            "2. Use structured logging\n"
            "3. Prefer composition over inheritance"
        )
        builder.set_knowledge_context(km_content)
        prompt = builder.build_proposal_prompt(mock_agent)
        assert "Always validate input" in prompt
        assert "Use structured logging" in prompt
        assert "Prefer composition over inheritance" in prompt


# ---------------------------------------------------------------------------
# 7. Backward compatibility (env.context still works)
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Ensure env.context injection still works without KM section."""

    def test_env_context_still_injected(self, builder, mock_agent, mock_env):
        """env.context appears in the prompt even without KM context."""
        mock_env.context = "Existing context about the project"
        prompt = builder.build_proposal_prompt(mock_agent)
        assert "Existing context about the project" in prompt
        assert "## Organizational Knowledge" not in prompt

    def test_both_env_context_and_km_context(self, builder, mock_agent, mock_env):
        """Both env.context and KM context coexist in the prompt."""
        mock_env.context = "Project uses PostgreSQL"
        builder.set_knowledge_context("Past decision: use connection pooling.")
        prompt = builder.build_proposal_prompt(mock_agent)
        assert "Project uses PostgreSQL" in prompt
        assert "Past decision: use connection pooling." in prompt
        assert "## Organizational Knowledge" in prompt


# ---------------------------------------------------------------------------
# 8. Item IDs tracking
# ---------------------------------------------------------------------------

class TestItemIDsTracking:
    """Test KM item ID tracking for outcome validation."""

    def test_item_ids_stored_on_set(self, builder):
        """Item IDs are accessible after setting."""
        builder.set_knowledge_context("content", item_ids=["km-abc", "km-def"])
        assert builder._km_item_ids == ["km-abc", "km-def"]

    def test_item_ids_default_empty(self, builder):
        """Item IDs default to empty list."""
        assert builder._km_item_ids == []

    def test_item_ids_copied_not_referenced(self, builder):
        """Item IDs are copied, not referenced."""
        original_ids = ["km-001"]
        builder.set_knowledge_context("content", item_ids=original_ids)
        original_ids.append("km-002")
        assert builder._km_item_ids == ["km-001"]  # Not affected by mutation


# ---------------------------------------------------------------------------
# 9. ContextInitializer integration
# ---------------------------------------------------------------------------

class TestContextInitializerIntegration:
    """Test ContextInitializer routes KM context to PromptBuilder."""

    @pytest.mark.asyncio
    async def test_inject_knowledge_sets_on_prompt_builder(self):
        """ContextInitializer sets KM context on PromptBuilder when available."""
        from aragora.debate.phases.context_init import ContextInitializer, _knowledge_cache
        import hashlib

        # Setup mocks
        mock_mound = MagicMock()
        mock_pb = MagicMock()
        mock_pb.set_knowledge_context = MagicMock()

        fetch_cb = AsyncMock(return_value="Organizational knowledge from mound")
        fetch_cb.__self__ = MagicMock()
        fetch_cb.__self__._last_km_item_ids = ["km-x1", "km-x2"]

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
            fetch_knowledge_context=fetch_cb,
        )

        task = "Builder injection test task unique 0"
        # Create a mock DebateContext with _prompt_builder
        ctx = MagicMock()
        ctx.env = MockEnvironment(task=task)
        ctx._prompt_builder = mock_pb

        try:
            await initializer._inject_knowledge_context(ctx)

            # Verify it was set on prompt builder
            mock_pb.set_knowledge_context.assert_called_once_with(
                "Organizational knowledge from mound",
                ["km-x1", "km-x2"],
            )
        finally:
            qh = hashlib.md5(task.encode(), usedforsecurity=False).hexdigest()
            _knowledge_cache.pop(qh, None)

    @pytest.mark.asyncio
    async def test_inject_knowledge_falls_back_to_env_context(self):
        """ContextInitializer appends to env.context when no PromptBuilder."""
        from aragora.debate.phases.context_init import ContextInitializer, _knowledge_cache

        mock_mound = MagicMock()
        fetch_cb = AsyncMock(return_value="Fallback knowledge content")

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
            fetch_knowledge_context=fetch_cb,
        )

        # Create context object that does NOT have _prompt_builder
        class BareCtx:
            pass

        ctx = BareCtx()
        ctx.env = MockEnvironment(task="Fallback test task unique 1", context="Existing context")

        try:
            await initializer._inject_knowledge_context(ctx)
            assert "Fallback knowledge content" in ctx.env.context
        finally:
            import hashlib
            qh = hashlib.md5(ctx.env.task.encode(), usedforsecurity=False).hexdigest()
            _knowledge_cache.pop(qh, None)

    @pytest.mark.asyncio
    async def test_inject_knowledge_fallback_when_builder_lacks_method(self):
        """Falls back to env.context if prompt_builder lacks set_knowledge_context."""
        from aragora.debate.phases.context_init import ContextInitializer, _knowledge_cache

        mock_mound = MagicMock()
        fetch_cb = AsyncMock(return_value="Knowledge content")

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
            fetch_knowledge_context=fetch_cb,
        )

        # Context with _prompt_builder but WITHOUT set_knowledge_context method
        class CtxWithBarePB:
            pass

        mock_pb = object()  # bare object, no set_knowledge_context
        ctx = CtxWithBarePB()
        ctx.env = MockEnvironment(task="Fallback test task unique 2", context=None)
        ctx._prompt_builder = mock_pb

        try:
            await initializer._inject_knowledge_context(ctx)
            assert ctx.env.context == "Knowledge content"
        finally:
            import hashlib
            qh = hashlib.md5(ctx.env.task.encode(), usedforsecurity=False).hexdigest()
            _knowledge_cache.pop(qh, None)

    @pytest.mark.asyncio
    async def test_inject_knowledge_empty_env_context(self):
        """When env.context is None and no builder, KM sets it directly."""
        from aragora.debate.phases.context_init import ContextInitializer, _knowledge_cache

        mock_mound = MagicMock()
        fetch_cb = AsyncMock(return_value="Fresh knowledge")

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
            fetch_knowledge_context=fetch_cb,
        )

        class BareCtx:
            pass

        ctx = BareCtx()
        ctx.env = MockEnvironment(task="Empty context test task unique 3", context=None)

        try:
            await initializer._inject_knowledge_context(ctx)
            assert ctx.env.context == "Fresh knowledge"
        finally:
            import hashlib
            qh = hashlib.md5(ctx.env.task.encode(), usedforsecurity=False).hexdigest()
            _knowledge_cache.pop(qh, None)

    @pytest.mark.asyncio
    async def test_inject_knowledge_appends_to_existing_env_context(self):
        """When env.context exists and no builder, KM appends to it."""
        from aragora.debate.phases.context_init import ContextInitializer, _knowledge_cache

        mock_mound = MagicMock()
        fetch_cb = AsyncMock(return_value="Extra knowledge")

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
            fetch_knowledge_context=fetch_cb,
        )

        class BareCtx:
            pass

        ctx = BareCtx()
        ctx.env = MockEnvironment(task="Append test task unique 4", context="Previous context")

        try:
            await initializer._inject_knowledge_context(ctx)
            assert "Previous context" in ctx.env.context
            assert "Extra knowledge" in ctx.env.context
        finally:
            import hashlib
            qh = hashlib.md5(ctx.env.task.encode(), usedforsecurity=False).hexdigest()
            _knowledge_cache.pop(qh, None)

    @pytest.mark.asyncio
    async def test_inject_knowledge_cached_uses_builder(self):
        """Cached knowledge context also routes through PromptBuilder."""
        from aragora.debate.phases.context_init import (
            ContextInitializer,
            _knowledge_cache,
        )
        import hashlib
        import time

        mock_mound = MagicMock()
        fetch_cb = AsyncMock()

        initializer = ContextInitializer(
            knowledge_mound=mock_mound,
            enable_knowledge_retrieval=True,
            fetch_knowledge_context=fetch_cb,
        )

        task = "Cached task"
        query_hash = hashlib.md5(task.encode(), usedforsecurity=False).hexdigest()

        # Pre-populate cache
        _knowledge_cache[query_hash] = ("Cached KM content", time.time())

        mock_pb = MagicMock()
        mock_pb.set_knowledge_context = MagicMock()

        ctx = MagicMock()
        ctx.env = MockEnvironment(task=task)
        ctx._prompt_builder = mock_pb

        try:
            await initializer._inject_knowledge_context(ctx)

            # Should use builder path for cached content too
            mock_pb.set_knowledge_context.assert_called_once()
            call_args = mock_pb.set_knowledge_context.call_args
            assert call_args[0][0] == "Cached KM content"
        finally:
            # Clean up cache to avoid test pollution
            _knowledge_cache.pop(query_hash, None)
