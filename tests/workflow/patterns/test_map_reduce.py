"""
Tests for MapReduce Workflow Pattern.

Tests cover:
- MapReducePattern initialization and defaults
- Split strategy configuration (chunks, lines, sections, files)
- Map and reduce agent configuration
- Workflow generation with all steps
- Step configurations and handlers
- Transition rules between steps
- Visual metadata for workflow builder
- Prompt building for reduce step
- Workflow metadata and tags
- Factory classmethod
- Handler registration for map_reduce_split and map_reduce_map
- Parallel processing with semaphore limiting
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# MapReducePattern Initialization Tests
# ============================================================================


class TestMapReducePatternInit:
    """Tests for MapReducePattern initialization."""

    def test_default_init(self):
        """Test default initialization with minimal parameters."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(name="Test MapReduce")
        assert pattern.name == "Test MapReduce"
        assert pattern.agents == ["claude", "gpt4"]
        assert pattern.task == ""
        assert pattern.split_strategy == "chunks"
        assert pattern.chunk_size == 4000
        assert pattern.map_agent == "claude"
        assert pattern.reduce_agent == "gpt4"
        assert pattern.parallel_limit == 5
        assert pattern.file_pattern == "**/*"
        assert pattern.timeout_per_chunk == 60.0

    def test_custom_agents(self):
        """Test initialization with custom agents list."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(
            name="Custom",
            agents=["gemini", "mistral", "grok"],
        )
        assert pattern.agents == ["gemini", "mistral", "grok"]
        # Map agent defaults to first agent
        assert pattern.map_agent == "gemini"
        # Reduce agent defaults to last agent
        assert pattern.reduce_agent == "grok"

    def test_explicit_map_agent(self):
        """Test explicit map agent configuration."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(
            name="Explicit Map",
            agents=["gpt4"],
            map_agent="claude",
        )
        assert pattern.map_agent == "claude"

    def test_explicit_reduce_agent(self):
        """Test explicit reduce agent configuration."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(
            name="Explicit Reduce",
            agents=["claude"],
            reduce_agent="gpt4",
        )
        assert pattern.reduce_agent == "gpt4"

    def test_split_strategy_configuration(self):
        """Test split_strategy parameter."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        for strategy in ["chunks", "lines", "sections", "files"]:
            pattern = MapReducePattern(name="Test", split_strategy=strategy)
            assert pattern.split_strategy == strategy

    def test_chunk_size_configuration(self):
        """Test chunk_size parameter."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(name="Test", chunk_size=2000)
        assert pattern.chunk_size == 2000

    def test_parallel_limit_configuration(self):
        """Test parallel_limit parameter."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(name="Test", parallel_limit=10)
        assert pattern.parallel_limit == 10

    def test_file_pattern_configuration(self):
        """Test file_pattern parameter."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(name="Test", file_pattern="**/*.py")
        assert pattern.file_pattern == "**/*.py"

    def test_custom_prompts(self):
        """Test custom map and reduce prompts."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(
            name="Custom Prompts",
            map_prompt="Analyze: {chunk}",
            reduce_prompt="Summarize: {map_results}",
        )
        assert pattern.map_prompt == "Analyze: {chunk}"
        assert pattern.reduce_prompt == "Summarize: {map_results}"

    def test_map_prompt_defaults_to_task(self):
        """Test map_prompt defaults to task when not provided."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(
            name="Test",
            task="Analyze security issues",
        )
        assert pattern.map_prompt == "Analyze security issues"

    def test_pattern_type(self):
        """Test that pattern_type is MAP_REDUCE."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern
        from aragora.workflow.patterns.base import PatternType

        assert MapReducePattern.pattern_type == PatternType.MAP_REDUCE


# ============================================================================
# Workflow Generation Tests
# ============================================================================


class TestMapReduceWorkflowGeneration:
    """Tests for MapReduce workflow generation."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        defaults = {"name": "Test MapReduce"}
        defaults.update(kwargs)
        pattern = MapReducePattern(**defaults)
        return pattern.create_workflow()

    def test_returns_workflow_definition(self):
        """Test that create_workflow returns a WorkflowDefinition."""
        from aragora.workflow.types import WorkflowDefinition

        wf = self._create_workflow()
        assert isinstance(wf, WorkflowDefinition)

    def test_workflow_id_prefix(self):
        """Test that workflow ID starts with mr_."""
        wf = self._create_workflow()
        assert wf.id.startswith("mr_")

    def test_workflow_name(self):
        """Test workflow name is set correctly."""
        wf = self._create_workflow(name="Security Scan")
        assert wf.name == "Security Scan"

    def test_workflow_has_all_steps(self):
        """Test that all required steps are present."""
        wf = self._create_workflow()
        step_ids = [s.id for s in wf.steps]
        assert "split" in step_ids
        assert "map" in step_ids
        assert "reduce" in step_ids

    def test_step_count(self):
        """Test workflow has exactly 3 steps."""
        wf = self._create_workflow()
        assert len(wf.steps) == 3

    def test_entry_step_is_split(self):
        """Test that entry step is split."""
        wf = self._create_workflow()
        assert wf.entry_step == "split"

    def test_transition_count(self):
        """Test workflow has correct number of transitions."""
        wf = self._create_workflow()
        # split -> map -> reduce = 2 transitions
        assert len(wf.transitions) == 2

    def test_transition_flow(self):
        """Test the transition flow order."""
        wf = self._create_workflow()
        flow = {t.from_step: t.to_step for t in wf.transitions}
        assert flow["split"] == "map"
        assert flow["map"] == "reduce"

    def test_workflow_description(self):
        """Test workflow description includes strategy and parallel info."""
        wf = self._create_workflow(
            split_strategy="lines",
            parallel_limit=8,
        )
        assert "lines" in wf.description
        assert "8" in wf.description

    def test_workflow_tags(self):
        """Test workflow tags."""
        wf = self._create_workflow()
        assert "map_reduce" in wf.tags
        assert "parallel" in wf.tags
        assert "batch" in wf.tags

    def test_workflow_metadata(self):
        """Test workflow metadata includes pattern information."""
        wf = self._create_workflow(
            split_strategy="sections",
            chunk_size=2000,
            parallel_limit=3,
            map_agent="gemini",
            reduce_agent="claude",
        )
        assert wf.metadata["pattern"] == "map_reduce"
        assert wf.metadata["split_strategy"] == "sections"
        assert wf.metadata["chunk_size"] == 2000
        assert wf.metadata["parallel_limit"] == 3
        assert wf.metadata["map_agent"] == "gemini"
        assert wf.metadata["reduce_agent"] == "claude"

    def test_workflow_category(self):
        """Test workflow category defaults to GENERAL."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow()
        assert wf.category == WorkflowCategory.GENERAL

    def test_custom_workflow_category(self):
        """Test custom workflow category."""
        from aragora.workflow.types import WorkflowCategory

        wf = self._create_workflow(category=WorkflowCategory.CODE)
        assert wf.category == WorkflowCategory.CODE

    def test_custom_tags(self):
        """Test custom tags are included."""
        wf = self._create_workflow(tags=["security", "analysis"])
        assert "map_reduce" in wf.tags
        assert "security" in wf.tags
        assert "analysis" in wf.tags


# ============================================================================
# Step Configuration Tests
# ============================================================================


class TestMapReduceStepConfig:
    """Tests for step configuration details."""

    def _create_workflow(self, **kwargs):
        """Helper to create workflow with optional overrides."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        defaults = {"name": "Test MapReduce"}
        defaults.update(kwargs)
        pattern = MapReducePattern(**defaults)
        return pattern.create_workflow()

    def test_split_step_type(self):
        """Test split step is a task step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "split")
        assert step.step_type == "task"

    def test_split_step_handler(self):
        """Test split step uses correct handler."""
        wf = self._create_workflow(
            split_strategy="lines",
            chunk_size=1000,
            file_pattern="*.txt",
        )
        step = next(s for s in wf.steps if s.id == "split")
        assert step.config["handler"] == "map_reduce_split"
        assert step.config["args"]["strategy"] == "lines"
        assert step.config["args"]["chunk_size"] == 1000
        assert step.config["args"]["file_pattern"] == "*.txt"

    def test_map_step_type(self):
        """Test map step is a task step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "map")
        assert step.step_type == "task"

    def test_map_step_handler(self):
        """Test map step uses correct handler."""
        wf = self._create_workflow(
            map_agent="gemini",
            map_prompt="Analyze: {chunk}",
            parallel_limit=3,
            timeout_per_chunk=45.0,
        )
        step = next(s for s in wf.steps if s.id == "map")
        assert step.config["handler"] == "map_reduce_map"
        assert step.config["args"]["agent_type"] == "gemini"
        assert step.config["args"]["prompt_template"] == "Analyze: {chunk}"
        assert step.config["args"]["parallel_limit"] == 3
        assert step.config["args"]["timeout_per_chunk"] == 45.0

    def test_reduce_step_type(self):
        """Test reduce step is an agent step."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "reduce")
        assert step.step_type == "agent"

    def test_reduce_step_uses_reduce_agent(self):
        """Test reduce step uses reduce agent."""
        wf = self._create_workflow(reduce_agent="mistral")
        step = next(s for s in wf.steps if s.id == "reduce")
        assert step.config["agent_type"] == "mistral"

    def test_reduce_step_prompt(self):
        """Test reduce step has prompt template."""
        wf = self._create_workflow(reduce_prompt="Summarize: {map_results}")
        step = next(s for s in wf.steps if s.id == "reduce")
        assert step.config["prompt_template"] == "Summarize: {map_results}"


# ============================================================================
# Visual Metadata Tests
# ============================================================================


class TestMapReduceVisualMetadata:
    """Tests for visual metadata in generated workflow."""

    def _create_workflow(self):
        """Helper to create workflow."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(name="Test MapReduce")
        return pattern.create_workflow()

    def test_all_steps_have_visual(self):
        """Test that all steps have visual metadata."""
        wf = self._create_workflow()
        for step in wf.steps:
            assert step.visual is not None
            assert step.visual.position is not None

    def test_split_step_category(self):
        """Test split step has CONTROL category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "split")
        assert step.visual.category == NodeCategory.CONTROL

    def test_map_step_category(self):
        """Test map step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "map")
        assert step.visual.category == NodeCategory.AGENT

    def test_reduce_step_category(self):
        """Test reduce step has AGENT category."""
        from aragora.workflow.types import NodeCategory

        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "reduce")
        assert step.visual.category == NodeCategory.AGENT

    def test_transitions_have_visual(self):
        """Test that transitions have visual edge data."""
        from aragora.workflow.types import EdgeType

        wf = self._create_workflow()
        for transition in wf.transitions:
            assert transition.visual is not None
            assert transition.visual.edge_type == EdgeType.DATA_FLOW


# ============================================================================
# Prompt Building Tests
# ============================================================================


class TestMapReducePromptBuilding:
    """Tests for prompt building methods."""

    def test_build_default_reduce_prompt(self):
        """Test default reduce prompt format."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(name="Test")
        prompt = pattern._build_default_reduce_prompt()
        assert "{step.map}" in prompt
        assert "aggregate" in prompt.lower() or "summarize" in prompt.lower()

    def test_custom_reduce_prompt_used(self):
        """Test that custom reduce prompt is used when provided."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        custom_prompt = "Custom reduce: {results}"
        pattern = MapReducePattern(name="Test", reduce_prompt=custom_prompt)
        wf = pattern.create_workflow()
        step = next(s for s in wf.steps if s.id == "reduce")
        assert step.config["prompt_template"] == custom_prompt


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestMapReduceFactory:
    """Tests for factory methods."""

    def test_create_classmethod(self):
        """Test MapReducePattern.create factory method."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern
        from aragora.workflow.types import WorkflowDefinition

        wf = MapReducePattern.create(
            name="Factory Test",
            split_strategy="sections",
            map_agent="claude",
            reduce_agent="gpt4",
            parallel_limit=4,
        )
        assert isinstance(wf, WorkflowDefinition)
        assert wf.metadata["split_strategy"] == "sections"
        assert wf.metadata["map_agent"] == "claude"
        assert wf.metadata["reduce_agent"] == "gpt4"
        assert wf.metadata["parallel_limit"] == 4


# ============================================================================
# Handler Tests - Split
# ============================================================================


class TestMapReduceSplitHandler:
    """Tests for map_reduce_split handler."""

    @pytest.mark.asyncio
    async def test_split_chunks_strategy(self):
        """Test splitting by chunks."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_split")
        assert handler is not None

        context = MagicMock()
        # Create a string longer than chunk_size
        context.inputs = {"input": "A" * 100}

        result = await handler(context, strategy="chunks", chunk_size=30)
        assert "chunks" in result
        assert len(result["chunks"]) == 4  # 100/30 = 4 chunks (rounded up)
        assert result["count"] == 4
        # Each chunk should be 30 chars except possibly the last
        assert len(result["chunks"][0]) == 30

    @pytest.mark.asyncio
    async def test_split_lines_strategy(self):
        """Test splitting by lines."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_split")

        context = MagicMock()
        context.inputs = {"input": "Line 1\nLine 2\nLine 3\nLine 4"}

        result = await handler(context, strategy="lines")
        assert len(result["chunks"]) == 4
        assert result["chunks"][0] == "Line 1"
        assert result["chunks"][3] == "Line 4"

    @pytest.mark.asyncio
    async def test_split_sections_strategy(self):
        """Test splitting by sections (double newlines)."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_split")

        context = MagicMock()
        context.inputs = {"input": "Section 1\n\nSection 2\n\nSection 3"}

        result = await handler(context, strategy="sections")
        assert len(result["chunks"]) == 3
        assert result["chunks"][0] == "Section 1"
        assert result["chunks"][2] == "Section 3"

    @pytest.mark.asyncio
    async def test_split_handles_non_string_input(self):
        """Test splitting handles non-string input."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_split")

        context = MagicMock()
        context.inputs = {"input": {"key": "value"}}  # Non-string input

        result = await handler(context, strategy="chunks")
        assert len(result["chunks"]) == 1
        assert result["chunks"][0] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_split_uses_data_fallback(self):
        """Test split uses 'data' key as fallback for input."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_split")

        context = MagicMock()
        context.inputs = {"data": "Fallback data"}

        result = await handler(context, strategy="chunks")
        assert result["chunks"][0] == "Fallback data"

    @pytest.mark.asyncio
    async def test_split_unknown_strategy_returns_single(self):
        """Test unknown strategy returns input as single chunk."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_split")

        context = MagicMock()
        context.inputs = {"input": "Some text data"}

        result = await handler(context, strategy="unknown_strategy")
        assert len(result["chunks"]) == 1
        assert result["chunks"][0] == "Some text data"


# ============================================================================
# Handler Tests - Map
# ============================================================================


class TestMapReduceMapHandler:
    """Tests for map_reduce_map handler."""

    @pytest.mark.asyncio
    async def test_map_returns_empty_on_no_chunks(self):
        """Test map returns error when no chunks."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_map")

        context = MagicMock()
        context.step_outputs = {"split": {"chunks": []}}

        result = await handler(context, agent_type="claude")
        assert result["results"] == []
        assert "error" in result

    @pytest.mark.asyncio
    async def test_map_processes_chunks(self):
        """Test map processes all chunks."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_map")

        context = MagicMock()
        context.step_outputs = {"split": {"chunks": ["chunk1", "chunk2", "chunk3"]}}

        # Mock create_agent - it's imported inside the handler from aragora.agents
        with patch("aragora.agents.create_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.generate.return_value = "Processed result"
            mock_create.return_value = mock_agent

            result = await handler(
                context,
                agent_type="claude",
                prompt_template="Analyze: {chunk}",
                parallel_limit=5,
                timeout_per_chunk=30.0,
            )

            assert result["total"] == 3
            assert result["successful_count"] == 3
            assert result["failed_count"] == 0

    @pytest.mark.asyncio
    async def test_map_replaces_chunk_placeholder(self):
        """Test map replaces {chunk} and {index} in prompt."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_map")

        context = MagicMock()
        context.step_outputs = {"split": {"chunks": ["test_data"]}}

        captured_prompts = []

        with patch("aragora.agents.create_agent") as mock_create:
            mock_agent = AsyncMock()

            async def capture_prompt(prompt):
                captured_prompts.append(prompt)
                return "Result"

            mock_agent.generate.side_effect = capture_prompt
            mock_create.return_value = mock_agent

            await handler(
                context,
                agent_type="claude",
                prompt_template="Chunk {index}: {chunk}",
                parallel_limit=1,
            )

            assert "test_data" in captured_prompts[0]
            assert "0" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_map_respects_parallel_limit(self):
        """Test map respects parallel_limit with semaphore."""
        from aragora.workflow.nodes.task import get_task_handler
        import asyncio

        handler = get_task_handler("map_reduce_map")

        context = MagicMock()
        context.step_outputs = {"split": {"chunks": ["c1", "c2", "c3", "c4", "c5"]}}

        concurrent_count = 0
        max_concurrent = 0

        with patch("aragora.agents.create_agent") as mock_create:
            mock_agent = AsyncMock()

            async def track_concurrency(prompt):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.01)  # Small delay to allow overlap
                concurrent_count -= 1
                return "Result"

            mock_agent.generate.side_effect = track_concurrency
            mock_create.return_value = mock_agent

            await handler(
                context,
                agent_type="claude",
                parallel_limit=2,
            )

            # Max concurrent should not exceed parallel_limit
            assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_map_handles_agent_errors(self):
        """Test map handles agent errors gracefully."""
        from aragora.workflow.nodes.task import get_task_handler

        handler = get_task_handler("map_reduce_map")

        context = MagicMock()
        context.step_outputs = {"split": {"chunks": ["chunk1", "chunk2"]}}

        with patch("aragora.agents.create_agent") as mock_create:
            mock_agent = AsyncMock()
            # First call succeeds, second fails
            mock_agent.generate.side_effect = ["Result", RuntimeError("API Error")]
            mock_create.return_value = mock_agent

            result = await handler(context, agent_type="claude")

            assert result["successful_count"] == 1
            assert result["failed_count"] == 1
            assert len(result["failed"]) == 1
            assert "Chunk processing failed" in result["failed"][0]["error"]


# ============================================================================
# Next Steps Tests
# ============================================================================


class TestMapReduceNextSteps:
    """Tests for next_steps configuration on steps."""

    def _create_workflow(self):
        """Helper to create workflow."""
        from aragora.workflow.patterns.map_reduce import MapReducePattern

        pattern = MapReducePattern(name="Test MapReduce")
        return pattern.create_workflow()

    def test_split_next_steps(self):
        """Test split step next_steps."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "split")
        assert step.next_steps == ["map"]

    def test_map_next_steps(self):
        """Test map step next_steps."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "map")
        assert step.next_steps == ["reduce"]

    def test_reduce_has_no_next_steps(self):
        """Test reduce step has no next_steps (terminal)."""
        wf = self._create_workflow()
        step = next(s for s in wf.steps if s.id == "reduce")
        assert step.next_steps == []
